# Mission Pad detection: the Tello's own metric ground-truth source.
#
# Tello EDU / RoboMaster TT only (SDK 2.0+). A standard Tello running SDK 1.3
# has no mission pad hardware support and reports pad_id = PAD_UNSUPPORTED.
#
# The drone's downward (and optionally forward) camera recognises a printed
# pad and reports its own position relative to that pad's centre. Because the
# pad is fixed in the room, this is an ABSOLUTE, DRIFT-FREE position reference
# -- which is exactly what a VIO estimator lacks and what makes it usable as
# ground truth for measuring VIO error.
#
# Update rate: 20 Hz with one detection direction enabled, 10 Hz with both.

std_msgs/Header header

# ID of the detected pad, 1-8. Sentinels below.
int32 pad_id

int32 PAD_UNSUPPORTED = -3   # this airframe has no mission pad support
int32 PAD_DISABLED    = -2   # detection not enabled (call with mission_pad_enable)
int32 PAD_NONE        = -1   # detection enabled, but no pad currently in view

# True only when pad_id >= 1, i.e. position/orientation below are meaningful.
bool valid

# Drone position relative to the detected pad's centre, in METRES, converted
# to the ROS FLU convention (x forward, y left, z up) from the SDK's raw
# centimetres. See the driver's mission_pad_axis_signs parameter: the SDK
# documents the origin and plane but not the axis signs, so they are
# configurable and must be verified once per airframe.
geometry_msgs/Point position

# Drone attitude in the pad frame, radians, from the SDK's "mpry" field.
# Reported by the SDK as (pitch, yaw, roll) degrees; re-ordered here to the
# conventional (roll, pitch, yaw) and converted to radians.
geometry_msgs/Vector3 rpy

# Raw SDK values exactly as received, in centimetres and degrees, for
# debugging and for verifying the conversions above.
int32 raw_x_cm
int32 raw_y_cm
int32 raw_z_cm
