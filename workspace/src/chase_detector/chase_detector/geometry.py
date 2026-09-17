"""Pinhole geometry for the detected box -- pure Python, no ROS.

Turns a bounding box in pixels into quantities a controller can use:

* range   -- d = fx * W_ref / w_px, the similar-triangles estimate. W_ref is
             the real width of what the detector actually boxes: ~0.180 m for
             a Tello prop-tip span, ~0.098 m for the bare fuselage. Getting
             this choice wrong scales every range by ~1.8x, which is exactly
             why it is a parameter and printed at startup, never implicit.
* bearing -- azimuth/elevation of the box centre through the calibrated
             intrinsics, replacing the paper's raw-pixel-offset proxy.

Intrinsics come from the driver's camera_info (which the driver already
rescales when video_scale != 1 -- tello/node.py), so nothing here hardcodes
ost.txt. Without intrinsics every output is NaN: an honest "don't know",
never a guess.

Sign conventions (optical frame, image v grows downward):
  azimuth   > 0  -> target is to the RIGHT of the optical axis
  elevation > 0  -> target is BELOW the optical axis
"""
import math
from typing import Optional, Tuple


class PinholeGeometry:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        if fx <= 0 or fy <= 0:
            raise ValueError(f"non-positive focal length fx={fx} fy={fy}")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    @classmethod
    def from_k(cls, k) -> Optional['PinholeGeometry']:
        """From a row-major 3x3 K (sensor_msgs/CameraInfo.k). None if unset
        (an all-zero K is what an uncalibrated driver publishes)."""
        if len(k) != 9 or k[0] <= 0 or k[4] <= 0:
            return None
        return cls(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def range_from_width(self, box_width_px: float, ref_width_m: float) -> float:
        if box_width_px <= 0 or ref_width_m <= 0:
            return math.nan
        return self.fx * ref_width_m / box_width_px

    def bearings_deg(self, u: float, v: float) -> Tuple[float, float]:
        az = math.degrees(math.atan((u - self.cx) / self.fx))
        el = math.degrees(math.atan((v - self.cy) / self.fy))
        return az, el

    def hfov_deg(self, image_width: int) -> float:
        return 2.0 * math.degrees(math.atan(image_width / (2.0 * self.fx)))

    def vfov_deg(self, image_height: int) -> float:
        return 2.0 * math.degrees(math.atan(image_height / (2.0 * self.fy)))
