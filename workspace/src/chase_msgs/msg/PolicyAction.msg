# The policy's OUTPUT: the raw network action, before the mixer assembles a
# Twist and before the safety supervisor clamps it.
#
# Action convention (checkpoint action_map, verified at load):
#   action[0] = a_vertical -> linear.z   (+up; +linear.z moves the box DOWN)
#   action[1] = a_yaw      -> angular.z  (REP-103 +CCW; +yaw moves box RIGHT)
# Both are normalised stick units in [-1, 1] and are published to /cmd_vel
# WITHOUT rescaling -- the driver scales [-1,1] to the SDK's +/-100 itself.

std_msgs/Header header

builtin_interfaces/Time t_capture     # provenance carried from the observation
builtin_interfaces/Time t_receive

float32[] action                      # raw network output, [-1, 1]
float32 a_vertical                    # action[0], unpacked
float32 a_yaw                         # action[1], unpacked

# True when this action was computed from an observation whose newest
# detection is older than the loss timeout -- i.e. the network is acting on
# a zeroed, stale view. The behaviour node, not the policy, decides what to
# do about it.
bool stale

string checkpoint_id                  # basename of the loaded bundle
string backend                        # 'onnx' | 'torch'
string obs_spec_hash                  # must match the TrackingState it answered
float32 inference_ms
