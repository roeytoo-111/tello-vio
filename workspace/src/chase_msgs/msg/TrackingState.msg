# The policy's INPUT, published so that an in-flight decision can be
# reconstructed afterwards from a bag without re-running the detector.
#
# The vector in `observation` is assembled by chase_gym's
# ObservationAssembler -- the SAME module the trainer used -- so what flies
# and what trained are structurally identical (implementation_plan.md: "the
# observation assembly in flight is the same module ... train/deploy skew is
# structurally impossible").

std_msgs/Header header

# t_capture: copied from the detection that fed this observation (which the
# detector copies from the image header). Zero when this tick had no
# detection. t_receive: when chase_state assembled the vector.
builtin_interfaces/Time t_capture
builtin_interfaces/Time t_receive

# The observation exactly as handed to the network, and the hash of the
# spec that defines its layout. A consumer that cannot reproduce the hash
# must not interpret the vector.
float32[] observation
string obs_spec_hash

# The first six elements, unpacked for human reading and plotting. These are
# DERIVED from `observation`, never computed separately.
float32 ex              # normalised horizontal error, [-1, 1], +right
float32 ey              # normalised vertical error, [-1, 1], +down
bool visible            # a detection arrived on THIS control tick
float32 staleness       # seconds since last detection / loss_timeout, capped 1

# Metric interpretation of the same box, for the depth rule and for logging.
# range_m is derived with the POLICY's ref_width_m (see chase_flight.contract):
# the detector's own range_m field is deliberately ignored because its
# ref_width_m default differs (0.180 m prop span vs 0.098 m body width).
float32 range_m         # NaN when not visible
float32 w_px            # box width rescaled into the calibration frame

# Age of the newest detection MESSAGE (the detector publishes one per frame
# including misses), so this is a detector-liveness signal, NOT a
# target-loss signal. -1 when none has ever arrived.
float32 detection_age_s

# Seconds since the last POSITIVE detection -- uncapped, unlike the
# `staleness` element of the observation which saturates at 1.0 by the
# spec. This is the target-loss signal the supervisor disarms on, and it
# must be uncapped for a 30 s rule to be expressible at all.
# Starts at -1 and stays there until the target is seen for the first
# time, so "never acquired" is distinguishable from "recently seen".
float32 time_since_detection_s
