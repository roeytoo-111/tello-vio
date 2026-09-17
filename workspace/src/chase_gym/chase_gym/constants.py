"""The verified numbers, in exactly one place.

Every value below is [V] in the doc set (docs/drone_chasing_rl/README.md
verification record) unless marked otherwise. Nothing downstream may restate
these literals: the observation assembler, the reward, the Tier-B oracle and
the Tier-C camera config all import them from here, which is what makes the
one-contract rule (sim_training_architecture.md section 0) enforceable.
"""
import math

# Camera geometry -- our calibration (workspace/src/tello/resource/ost.txt),
# NOT any simulator's default intrinsics. The policy lives in this geometry.
FRAME_W = 960
FRAME_H = 720
CX = 480.0
CY = 360.0
FX = 919.42          # px; ost.txt, rescale-safe in the driver for video_scale
FY = 919.42

# Derived, recomputed for sim_training_architecture.md: H-FOV 55.1 deg,
# V-FOV 42.8 deg, max centre-to-corner distance exactly 600 px.
HFOV_DEG = 2.0 * math.degrees(math.atan(FRAME_W / (2.0 * FX)))
VFOV_DEG = 2.0 * math.degrees(math.atan(FRAME_H / (2.0 * FY)))
MAX_DIST_PX = math.hypot(CX, CY)                      # 600.0 exactly

# Control period -- the deployment constant (rl_training_guide.md 16.1).
# The sim step IS the control period; change one, change both.
DT = 0.1             # s
CONTROL_RATE_HZ = 10.0

# The k-rule (guide 4.2): k = ceil(Delta_max * f_ctrl) at Delta_max = 0.35 s.
K_ACTION_HISTORY = 4

# Tello target extents, metres. Width/height verified against the
# manufacturer spec; prop-span is the recorded assumption. The reference
# width the oracle projects with is a parameter, never implicit
# (chase_detector/geometry.py states why: the choice scales every range 1.8x).
TELLO_BODY_W = 0.098
TELLO_BODY_H = 0.041
TELLO_PROP_SPAN = 0.180

# Measured platform facts (repo README, [V]).
LATENCY_RANGE_S = (0.150, 0.350)   # video link latency, measured
DEADMAN_S = 0.35                   # driver zeroes RC beyond this staleness

# Reward geometry: threshold 100 px is the original CODE's value
# (rl_block_diagram.md section 5) -- the paper's 110 exists only in its text.
REWARD_THRESHOLD_PX = 100.0
