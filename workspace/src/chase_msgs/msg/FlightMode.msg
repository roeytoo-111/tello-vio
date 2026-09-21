# The behaviour state machine's current mode (block_diagram.md section 5).
#
#   SEARCH     no target yet            -> rotate about own axis
#   TRACK      target detected          -> policy + depth rule active
#   REACQUIRE  target just lost         -> hold position, then rotate
#   PATROL     lost for an extended time -> predefined patrol pattern
#
# Transitions: SEARCH->TRACK on detection; TRACK->REACQUIRE on loss;
# REACQUIRE->TRACK on reacquisition; REACQUIRE->PATROL when the loss exceeds
# the timeout; PATROL->TRACK on detection.

uint8 SEARCH=0
uint8 TRACK=1
uint8 REACQUIRE=2
uint8 PATROL=3

std_msgs/Header header

uint8 mode
string mode_name

# Search/patrol command on the yaw axis, in normalised stick units [-1, 1].
# Zero in TRACK (the policy owns yaw) and during the REACQUIRE hold phase.
float32 search_yaw

# Set while REACQUIRE is holding position before it starts rotating: the
# mixer zeroes the forward axis during the hold so the aircraft does not
# coast onward toward a target it can no longer see.
bool hold_position

float32 time_in_mode_s
float32 detection_age_s
