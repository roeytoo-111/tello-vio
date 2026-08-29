"""ArUco fiducial ground truth: metric 6-DoF pose from a printed marker.

Why this exists
---------------
Every accuracy number in the technical report is *simulated*. To say anything
true about the estimator on real hardware you need an independent, metric
reference, and indoors on a Tello the realistic options are:

* **Motion capture** -- the gold standard, and not something most people have.
* **A printed fiducial** -- one sheet of paper, accurate to roughly 1-2 cm at
  1-3 m range, and measured by the same camera the estimator uses.
* **Tape-measure waypoints** -- fine for a single distance, useless for a
  trajectory.

The middle option is what this module implements. It is not as good as mocap,
and this docstring is the place to be explicit about that: marker pose error
grows with range and with viewing obliquity, so treat it as ground truth to
~2 cm at 1-2 m and degrading beyond, not as truth to the millimetre.

Conventions
-----------
``solvePnP`` returns the transform taking MARKER points into the CAMERA frame
(``R_cm``, ``t_cm``), with the camera in the optical convention (z forward).
The camera pose in the marker frame is its inverse. Both are returned, because
mixing them up is the standard way a fiducial evaluation ends up mirrored.

The marker frame is the OpenCV convention: origin at the marker centre, x
right, y up, z out of the marker face towards the viewer.

The planar pose ambiguity -- read this before trusting a number
---------------------------------------------------------------
Four coplanar points seen nearly head-on barely constrain tilt: two quite
different orientations reproject almost identically, and the solver picks
between them on sub-pixel evidence. Measured here on noise-free synthetic
renders at 2 m:

    viewing angle     translation error     rotation error
    head-on (0 deg)        1.5 cm               7.3 deg
    oblique (15-45)        1.5 cm               1.8 deg

Translation is unaffected. Rotation is not, and it propagates: the camera
position in the marker frame is ``t_mc = -R_cm^T t_cm``, so 7 deg of rotation
error at 2 m range becomes ~25 cm of apparent camera-position error.

Practical consequences for using this as ground truth:

* Prefer comparing **marker-in-camera** translation (``t_cm``) -- it is the
  robust quantity.
* If you need camera-in-marker pose, mount the marker so the drone views it
  from an angle rather than dead-on, and reject samples whose
  :attr:`MarkerDetection.ambiguity` is high.
* For serious work use a multi-marker board (``cv2.aruco.GridBoard``), whose
  non-coplanar-in-image corner spread removes the ambiguity entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Dictionary used throughout. 4x4_50 is deliberate: large cells survive the
#: Tello's soft, compressed 720p stream at range far better than the denser
#: 6x6/7x7 dictionaries, and 50 ids is plenty for a single-marker reference.
DEFAULT_DICT = "DICT_4X4_50"


@dataclass
class MarkerDetection:
    """One detected marker, already converted to a metric pose."""

    marker_id: int
    #: Marker -> camera (OpenCV optical frame: x right, y down, z forward).
    R_cm: np.ndarray
    t_cm: np.ndarray
    #: Camera pose expressed in the marker frame (the inverse of the above).
    R_mc: np.ndarray
    t_mc: np.ndarray
    #: Mean reprojection error of the four corners, in pixels. The honest
    #: quality signal: below ~1 px the pose is trustworthy, above ~3 px the
    #: marker is too small, too oblique or motion-blurred.
    reproj_px: float
    #: Apparent side length in pixels; a proxy for range and hence for how
    #: much to trust this sample.
    side_px: float
    #: Planar-ambiguity indicator in [0, 1]: the ratio of the second-best to
    #: the best IPPE reprojection error. Near 1 means the two candidate
    #: orientations fit equally well and the ROTATION is untrustworthy (the
    #: translation still is not affected). Gate on this before believing any
    #: orientation-derived quantity. NaN when the solver returned one solution.
    ambiguity: float = float("nan")


def make_detector(dict_name: str = DEFAULT_DICT):
    """Build an ArUco detector across the 4.7+ and legacy OpenCV APIs."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corner refinement is the single biggest accuracy win on a
        # soft stream: without it the corner quantisation alone costs ~1 cm of
        # position error at 2 m.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(d, params)
    return ("legacy", d, cv2.aruco.DetectorParameters_create())


def detect_markers(detector, gray: np.ndarray):
    """Return ``(corners, ids)`` for either API generation."""
    if isinstance(detector, tuple):
        _, d, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def marker_object_points(size_m: float) -> np.ndarray:
    """The four marker corners in the marker frame, in detectMarkers order.

    Order matters and is not arbitrary: ``detectMarkers`` returns corners
    top-left, top-right, bottom-right, bottom-left as seen in the image, and
    the object points must be listed to match or the recovered pose is a
    rotated/mirrored version of the truth.
    """
    h = size_m / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float64)


def estimate_marker_pose(corners, marker_size_m: float, K: np.ndarray,
                         D: np.ndarray | None, marker_id: int) -> MarkerDetection | None:
    """Metric pose of one marker via IPPE_SQUARE.

    ``SOLVEPNP_IPPE_SQUARE`` is used rather than the generic iterative solver:
    it is purpose-built for exactly this problem (four coplanar corners of a
    square), is closed-form, and does not need an initial guess. The generic
    solver on four coplanar points is prone to the classic planar pose
    ambiguity, which shows up as the marker pose flipping between two
    orientations frame to frame.
    """
    obj = marker_object_points(marker_size_m)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).reshape(1, -1)

    # solvePnPGeneric returns BOTH IPPE solutions with their errors, which is
    # what makes the ambiguity measurable rather than merely known about.
    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n_sol < 1:
        return None
    rvec, tvec = rvecs[0], tvecs[0]
    ambiguity = float("nan")
    if n_sol >= 2 and errs is not None:
        e = np.asarray(errs, dtype=np.float64).ravel()
        if e.size >= 2 and e[1] > 0:
            # errs are sorted best-first; ratio -> 1 means indistinguishable.
            ambiguity = float(min(e[0], e[1]) / max(e[0], e[1]))

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img, axis=1)))

    R_cm, _ = cv2.Rodrigues(rvec)
    t_cm = tvec.reshape(3)
    R_mc = R_cm.T
    t_mc = -R_mc @ t_cm

    side = float(np.mean([np.linalg.norm(img[i] - img[(i + 1) % 4]) for i in range(4)]))
    return MarkerDetection(marker_id, R_cm, t_cm, R_mc, t_mc, reproj, side,
                           ambiguity)


def generate_marker_png(path: str, marker_id: int = 0, px: int = 1200,
                        dict_name: str = DEFAULT_DICT, border_px: int = 120) -> str:
    """Write a printable marker with a white quiet zone.

    The quiet zone is not decoration: ArUco needs white margin around the
    black border to segment the marker, and a marker printed edge-to-edge on
    the paper frequently fails to detect at all.
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, marker_id, px)
    else:
        img = cv2.aruco.drawMarker(d, marker_id, px)
    canvas = np.full((px + 2 * border_px, px + 2 * border_px), 255, np.uint8)
    canvas[border_px:border_px + px, border_px:border_px + px] = img
    cv2.imwrite(path, canvas)
    return path
