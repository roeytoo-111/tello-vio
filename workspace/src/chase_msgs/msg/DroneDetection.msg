# One YOLO inference result on ONE camera frame -- published for EVERY
# processed frame, including misses (detected == false), so downstream
# consumers get an explicit visibility signal instead of a silent gap
# (docs/drone_chasing_rl/implementation_plan.md, sections 2 and 3).
#
# header.stamp is t_capture: copied verbatim from the source image header,
# which the tello driver stamps on UDP arrival. frame_id is copied too.
std_msgs/Header header

# When this node pulled the frame off the image topic (node clock, same host
# and clock as the driver). The pair (header.stamp, t_receive) is what lets
# an error later be attributed to transport latency rather than to the
# detector or the policy -- the paper's dominant failure mode is latency.
builtin_interfaces/Time t_receive

# True when at least one box of a wanted class survived the confidence
# threshold in this frame. When false, confidence/box/range are zero or NaN
# and must not be used.
bool detected

# Best surviving box: highest confidence among the wanted classes.
float32 confidence
string class_name

# Axis-aligned box, pixels, in THIS frame's resolution (see image_width/height
# -- the driver may downscale via its video_scale parameter).
float32 xmin
float32 ymin
float32 xmax
float32 ymax

# Resolution the box coordinates live in.
uint32 image_width
uint32 image_height

# Pinhole range estimate fx * ref_width_m / box_width_px, using fx from the
# driver's camera_info (already rescaled for video_scale by the driver).
# NaN when not detected or no camera_info has arrived yet.
float32 range_m

# YOLO wall time for this frame: preprocess + inference + postprocess, ms.
float32 inference_ms
