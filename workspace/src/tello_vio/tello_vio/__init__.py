"""Visual-inertial odometry for the DJI Tello.

Layered so that everything except the ``nodes`` subpackage is pure
NumPy/OpenCV and therefore testable, and profilable, without ROS installed::

    lie              SO(3)/SE(3) manifold operations; all conventions declared here
    tello_model      what the Tello actually measures: units, frames, surrogate gyro
    preintegration   on-manifold IMU preintegration (Forster et al.)
    eskf             error-state Kalman filter with stochastic cloning  [real-time]
    smoother         fixed-lag factor graph with marginalisation        [accuracy]
    two_view         essential/homography model selection, pose, triangulation
    frontend         KLT tracking, keyframing, relative-pose measurements
    calib            IMU noise identification, hand-eye extrinsic, time offset
    sim3             Umeyama similarity alignment (map <-> odom, monocular scale)
    ros_utils        message conversions -- the only place conventions are translated
    nodes            ROS 2 entry points
"""

__version__ = "1.0.0"
