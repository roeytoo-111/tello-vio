# Full decoded Tello SDK state broadcast (UDP :8890, ~10 Hz).
#
# A Header was added so consumers can time-align this with /imu and /image_raw.
# Without a stamp the message is unusable for sensor fusion: the receive time is
# not recoverable after the fact, and ROS 2 message_filters cannot synchronise it.
std_msgs/Header header

# Acceleration in the drone's own FRD body axes (x forward, y right, z down),
# converted from the SDK's milli-g to m/s^2. NOTE this is NOT the ROS FLU
# convention -- /imu carries the REP-103-compliant version. This message stays
# in the drone's native frame so it remains a faithful record of the raw packet.
geometry_msgs/Vector3 acceleration

# Velocity from the drone's own downward optical-flow estimator, FRD axes, m/s.
# The SDK reports vgx/vgy/vgz with no documented unit; see the driver's
# speed_to_mps parameter.
geometry_msgs/Vector3 speed

# Barometric (pressure) altitude in centimetres. Referenced to sea level, NOT to
# the takeoff point, and it drifts with weather -- use it as a relative
# measurement with an estimated bias, never as an absolute height.
int32 barometer

# Downward time-of-flight range in centimetres. Only meaningful over a flat
# surface within roughly 0.1-1.2 m; reads garbage over table edges and stairs.
int32 distance_tof

# Height above the takeoff point in centimetres (decimetre resolution).
int32 height

# Percentage of battery left
uint8 battery

# Time the motors have been active in seconds
int32 fligth_time

# Highest temperature in celsius
int32 highest_temperature
# Lowest temperature in celsius
int32 lowest_temperature
# Temperature in celsius
float32 temperature

# Attitude in degrees, in the drone's NED-referenced convention (yaw increases
# clockwise seen from above). /imu carries the ENU/FLU version.
int32 pitch
int32 roll
int32 yaw

# Wi-Fi signal-to-noise ratio, as reported by the "wifi?" query
string wifi_snr
