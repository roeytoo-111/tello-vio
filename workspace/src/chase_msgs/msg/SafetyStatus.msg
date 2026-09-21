# What the safety supervisor decided this tick, and why.
#
# The supervisor is the LAST node before the driver: it owns /cmd_vel and
# nothing else publishes there. It is deliberately a separate process from
# the mixer so that a controller bug cannot disable it
# (design_rationale.md: "its placement as a separate node between any
# controller and the driver ... isolation and cannot be disabled by a
# controller bug").

std_msgs/Header header

bool armed          # operator has armed the stack (engage gate)
bool engaged        # armed AND no veto active -> policy commands reach the drone
bool vetoed         # a rule fired this tick
string veto_reason  # empty when not vetoed; the FIRST rule that fired
bool capped         # the speed cap altered at least one axis this tick

# Evidence the decision was made on. NaN / -1 mean "never received".
float32 battery_frac      # 0..1 (the driver publishes a fraction, not a percent)
float32 height_m          # best available height estimate
float32 telemetry_age_s   # age of the newest /status -- the link-loss signal
float32 detection_age_s
float32 command_age_s     # age of the newest /chase/cmd_raw from the mixer
float32 flight_time_s

# The command actually put on the wire this tick, in normalised sticks.
float32 sent_forward
float32 sent_vertical
float32 sent_yaw
