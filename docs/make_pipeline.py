#!/usr/bin/env python3
"""Full end-to-end pipeline diagram: UDP packet to published pose.

Laid out by a top-down cursor rather than hand-placed coordinates, so bands and
rows cannot collide as the content changes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, LINE = "#14181f", "#616b7a", "#c6ced9"
BLUE, ORANGE, GREEN, PURPLE, RED, GOLD = ("#1f6feb", "#c2600f", "#1f8a4c",
                                          "#6b3fa0", "#b3261e", "#8a6a12")
BG_HW, BG_DRV, BG_TOP = "#e9edf3", "#eef3fb", "#f5f7fa"
BG_VIS, BG_TEL, BG_EST = "#fdf3e9", "#eaf1fb", "#eaf6ee"
BG_OUT, BG_CAL, BG_SLAM = "#f3eefa", "#fff8e3", "#f0eaf8"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 230,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

fig, ax = plt.subplots(figsize=(11.2, 15.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

L, R = 0.010, 0.990
GAP = 0.011                 # between bands
LBL_H = 0.019               # label strip inside a band
PAD = 0.007                 # inner padding


def band(y_top, height, label, color, x=L, w=R - L, lc=INK, ha="left"):
    """Draw a band whose TOP edge is at y_top. Returns the y of its content top."""
    y = y_top - height
    ax.add_patch(FancyBboxPatch((x, y), w, height,
                                boxstyle="round,pad=0.002,rounding_size=0.006",
                                linewidth=0, facecolor=color, alpha=0.5, zorder=0))
    lx = x + 0.009 if ha == "left" else x + w - 0.009
    ax.text(lx, y_top - 0.010, label, ha=ha, va="center",
            fontsize=8.6, weight="bold", color=lc, zorder=1)
    return y_top - LBL_H - PAD


def box(x, y_top, w, h, title, sub=None, fc="white", ec=LINE, tfs=7.3,
        tc=INK, weight="bold", lw=1.0):
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.003,rounding_size=0.005",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=tfs, weight=weight, color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=tfs - 1.2, color=MUTED, zorder=4, linespacing=1.32)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, weight=weight, color=tc, zorder=4, linespacing=1.35)


def arr(p0, p1, color=MUTED, lw=1.1, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9.5,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5, zorder=1))


def cols(x0, x1, n, gap=0.008):
    w = ((x1 - x0) - gap * (n - 1)) / n
    return [(x0 + i * (w + gap), w) for i in range(n)]


# =========================================================================== #
ax.text(0.5, 0.990, "Tello VIO — end-to-end pipeline", ha="center",
        fontsize=15.5, weight="bold", color=INK)
ax.text(0.5, 0.9745, "from UDP packet to published pose: the rate, the data and "
        "the algorithm at every stage", ha="center", fontsize=8.6, color=MUTED)

cur = 0.964

# ------------------------------------------------------------- 1 AIRCRAFT --
H = LBL_H + PAD + 0.046 + PAD
top = band(cur, H, "1 · AIRCRAFT", BG_HW)
c = cols(0.020, 0.980, 5)
box(c[0][0], top, c[0][1], 0.046, "DJI Tello",
    "streams; does not compute\nall processing is off-board", fc="white", ec=INK, tfs=8.0)
box(c[1][0], top, c[1][1], 0.046, "UDP :11111",
    "H.264 · 960×720 · ~30 fps\n150–350 ms late, jittery", fc=BG_VIS)
box(c[2][0], top, c[2][1], 0.046, "UDP :8890",
    "state broadcast · ~10 Hz\naperiodic, no timestamps", fc=BG_TEL)
box(c[3][0], top, c[3][1], 0.046, "UDP :8889",
    "commands · blocking\n7 s timeout × 3 retries", fc=BG_EST)
box(c[4][0], top, c[4][1], 0.046, "NOT on the wire",
    "no gyroscope · no GPS\nattitude fused, whole degrees",
    fc="#fdecec", ec=RED, tc=RED, tfs=7.6)
arr((c[0][0] + c[0][1], top - 0.023), (c[1][0], top - 0.023))
cur -= H + GAP

# --------------------------------------------------------------- 2 DRIVER --
H = LBL_H + PAD + 0.058 + PAD
top = band(cur, H, "2 · DRIVER   ·   tello/node.py   ·   MultiThreadedExecutor, "
                   "3 callback groups + 1 worker thread — no SDK call blocks a callback",
           BG_DRV)
c = cols(0.020, 0.980, 4)
box(c[0][0], top, c[0][1], 0.058, "Video callback group",
    "poll frame → reject duplicates by content\n"
    "INTER_AREA downscale (not subsampling)\nRGB→BGR · stamp at detection", fc=BG_VIS)
box(c[1][0], top, c[1][1], 0.058, "Telemetry callback group",
    "poll at 50 Hz, publish only on change\nmilli-g → m/s²,  FRD → FLU,  NED → ENU\n"
    "unmeasured fields flagged, not zeroed", fc=BG_TEL)
box(c[2][0], top, c[2][1], 0.058, "Control group + worker thread",
    "RC at 20 Hz with a 0.35 s dead-man\ntakeoff/land/flip queued off the executor\n"
    "/emergency bypasses the queue", fc=BG_EST)
box(c[3][0], top, c[3][1], 0.058, "Why it is built this way",
    "Stamping at detection bounds the timestamp\nerror to one 20 ms poll, not one 100 ms\n"
    "telemetry period — a varying offset cannot\nbe calibrated out.", fc="white")
for i, x in enumerate([c[1][0] + c[1][1] / 2, c[2][0] + c[2][1] / 2, c[3][0] + c[3][1] / 2]):
    pass
arr((c[1][0] + 0.03, cur + GAP), (c[0][0] + c[0][1] / 2, top))
arr((c[2][0] + 0.03, cur + GAP), (c[1][0] + c[1][1] / 2, top))
arr((c[2][0] + c[2][1] / 2, top), (c[3][0] + 0.03, cur + GAP), style="<|-")
cur -= H + GAP

# --------------------------------------------------------------- 3 TOPICS --
H = LBL_H + PAD + 0.038 + PAD
top = band(cur, H, "3 · ROS 2 TOPICS   ·   sensor-data QoS: best-effort, depth 1", BG_TOP)
c = cols(0.020, 0.980, 6)
tp = [("/image_raw", "30 Hz · bgr8\nduplicates dropped", BG_VIS),
      ("/camera_info", "rescaled with\nvideo_scale", BG_VIS),
      ("/imu", "10 Hz · SI · FLU\ncov[0] = −1: no gyro", BG_TEL),
      ("/status", "10 Hz · metric\nbody velocity", BG_TEL),
      ("/tof", "10 Hz · Range\n+inf = out of range", BG_TEL),
      ("/cmd_vel", "REP-103 in\n(+ legacy /control)", BG_EST)]
for (x, w), (n, s, fc) in zip(c, tp):
    box(x, top, w, 0.038, n, s, fc=fc, tfs=7.2)
cur -= H + GAP

TOPIC_BOT = top - 0.038

# ------------------------------------------------- 4 PER-FRAME PROCESSING --
SUB_H = 0.015
H = LBL_H + SUB_H + PAD + 4 * 0.041 + 3 * 0.006 + PAD
top = band(cur, H, "4 · PER-FRAME PROCESSING", BG_TOP) - SUB_H
VIS_R = 0.680

ax.add_patch(FancyBboxPatch((0.014, cur - H + 0.004), VIS_R - 0.020, H - 0.024,
                            boxstyle="round,pad=0.002,rounding_size=0.005",
                            linewidth=0, facecolor=BG_VIS, alpha=0.7, zorder=0))
ax.add_patch(FancyBboxPatch((VIS_R, cur - H + 0.004), 0.986 - VIS_R, H - 0.024,
                            boxstyle="round,pad=0.002,rounding_size=0.005",
                            linewidth=0, facecolor=BG_TEL, alpha=0.7, zorder=0))
ax.text(0.020, top + 0.008, "VISUAL FRONT-END   ·   frontend.py  +  two_view.py",
        fontsize=7.5, weight="bold", color=ORANGE)
ax.text(VIS_R + 0.008, top + 0.008, "TELEMETRY CONDITIONING   ·   tello_model.py",
        fontsize=7.5, weight="bold", color=BLUE)

RH, RG = 0.041, 0.006
ys = [top - i * (RH + RG) for i in range(4)]
fw = (VIS_R - 0.032) / 2 - 0.004
fx2 = 0.020 + fw + 0.008

box(0.020, ys[0], fw, RH, "Grayscale + downscale to 480 px",
    "every frame · rescale K with it;\nnever scale D (it is dimensionless)")
box(0.020, ys[1], fw, RH, "Pyramidal KLT tracking",
    "track forward, then back — reject if it\ndoes not return.  ~3 ms for 150 points")
box(0.020, ys[2], fw, RH, "Re-detect only when tracks < 90",
    "grid-bucketed Shi–Tomasi spreads features;\nclustered features = degenerate geometry")
box(0.020, ys[3], fw, RH, "Keyframe? rotation-compensated parallax",
    "raw pixel motion fires on TURNING;\nde-rotating makes it fire on MOVING",
    fc="#fbe6d2", ec=ORANGE)
for i in range(3):
    arr((0.020 + fw / 2, ys[i] - RH), (0.020 + fw / 2, ys[i + 1]), lw=0.9)

box(fx2, ys[0], fw, RH, "Undistort the ~150 points, not the image",
    "on keyframes only · undistortPoints applies\nK⁻¹ and removes distortion in one step")
box(fx2, ys[1], fw, RH, "Model selection:   E   vs   H",
    "symmetric transfer scores. Indoor planes make\nthe essential matrix genuinely degenerate")
box(fx2, ys[2], fw, RH, "recoverPose  →  R,  unit  t̂",
    "then t = −Rᵀ t_cv : OpenCV's t points the OTHER\nway, in the OTHER camera's frame")
box(fx2, ys[3], fw, RH, "Per-measurement σ_rot , σ_dir",
    "σ ∝ 1/√N and 1/parallax · constants fitted from\n64 keyframes across 8 scenes, not invented",
    fc="#fbe6d2", ec=ORANGE)
for i in range(3):
    arr((fx2 + fw / 2, ys[i] - RH), (fx2 + fw / 2, ys[i + 1]), lw=0.9)
arr((0.020 + fw, ys[1] - RH / 2), (fx2, ys[1] - RH / 2), lw=0.9)

tw, tx = 0.980 - VIS_R - 0.014, VIS_R + 0.008
box(tx, ys[0], tw, RH, "Surrogate gyro   ω = Log(q⁻¹ ⊗ q′) / dt",
    "on the manifold, with the measured dt — subtracting\nEuler angles breaks at wrap-around")
box(tx, ys[1], tw, RH, "None on dropout, never zero",
    "a zero rate asserts 'the drone did not rotate',\nwhich is a confident lie the filter will believe",
    ec=RED)
box(tx, ys[2], tw, RH, "Stationarity detector  →  ZUPT / ZARU",
    "accel variance + reported speed + rate, all low and\nsustained. While landed, v = 0 is EXACT")
box(tx, ys[3], tw, RH, "3 s ring buffer",
    "replayed on the image clock, so measurements reach\nthe filter in capture order",
    fc="#dce8f7", ec=BLUE)
for i in range(3):
    arr((tx + tw / 2, ys[i] - RH), (tx + tw / 2, ys[i + 1]), lw=0.9)

arr((0.070, TOPIC_BOT), (0.070, top + 0.014), lw=1.0)
arr((0.400, TOPIC_BOT), (tx + 0.06, top + 0.014), lw=1.0)
arr((0.560, TOPIC_BOT), (tx + 0.10, top + 0.014), lw=1.0)
cur -= H + GAP
PROC_BOT = ys[3] - RH

# --------------------------------------------------------------- 5 FUSION --
H = LBL_H + PAD + 0.038 + 0.006 + 0.050 + 0.006 + 0.026 + PAD
FUS_R = 0.660
top = band(cur, H, "5 · FUSION   ·   eskf.py   ·   22 error states", BG_EST,
           x=L, w=FUS_R - L)
band(cur, H, "OPTIONAL BACKEND   ·   loop closure", BG_SLAM,
     x=FUS_R + 0.006, w=R - FUS_R - 0.006, lc=PURPLE, ha="right")

box(0.020, top, FUS_R - 0.034, 0.038,
    "Advance the filter to the image's corrected capture time   t_img − t_d",
    "buffered telemetry is replayed up to it, so measurements are applied in order.\n"
    "The published pose is ~250 ms old but self-consistent.",
    fc="#d9efe2", ec=GREEN, tfs=7.4)
y2 = top - 0.038 - 0.006
fc3 = cols(0.020, FUS_R - 0.014, 3)
box(fc3[0][0], y2, fc3[0][1], 0.050, "PROPAGATE",
    "nominal state on the manifold\nerror-state F and Q\nJoseph-form covariance")
box(fc3[1][0], y2, fc3[1][1], 0.050, "UPDATE   (each χ²-gated)",
    "attitude 3 · body velocity 3\nbarometer 1 · ToF 1 · ZUPT 3 · ZARU 3\n"
    "visual: rotation 3 + bearing 2")
box(fc3[2][0], y2, fc3[2][1], 0.050, "CLONE",
    "P ← J P Jᵀ copies the full\ncross-covariance, not the mean\n(stochastic cloning)")
arr((fc3[0][0] + fc3[0][1], y2 - 0.025), (fc3[1][0], y2 - 0.025), lw=1.0)
arr((fc3[1][0] + fc3[1][1], y2 - 0.025), (fc3[2][0], y2 - 0.025), lw=1.0)
ax.add_patch(FancyArrowPatch((fc3[2][0] + fc3[2][1] / 2, y2 - 0.050),
                             (fc3[0][0] + fc3[0][1] / 2, y2 - 0.050),
                             connectionstyle="arc3,rad=0.30", arrowstyle="-|>",
                             mutation_scale=9.5, lw=1.0, color=MUTED, zorder=1))
y3 = y2 - 0.050 - 0.006
box(0.020, y3, FUS_R - 0.034, 0.026,
    "Metric scale comes from the flow sensor, barometer and ToF — never from the camera.",
    "The bearing residual is 2-DoF on purpose: a 3-vector residual would pull on magnitude.",
    fc="#fdecec", ec=RED, tc=RED, tfs=7.3)
# Enter to the right of the "5 · FUSION ..." label and to the left of the
# right-aligned backend label, so neither arrow crosses text.
arr((0.170, PROC_BOT), (0.380, top), lw=1.3, color=GREEN)
arr((tx + 0.06, PROC_BOT), (0.560, top), lw=1.3, color=GREEN)

sx, sw = FUS_R + 0.014, R - FUS_R - 0.028
box(sx, top, sw, 0.038, "ORB-SLAM2   ·   slam/src/orbslam2",
    "ORB + DBoW2 + g2o · ~15–25 ms/frame, separate process\n"
    "publishes pose, TF, path, map, tracking state", ec=PURPLE)
box(sx, y2, sw, 0.050, "Tcw → Twc, then optical → ROS axes",
    "raw ORB-SLAM2 coordinates in a frame called 'map'\nlay the map on its side.\n"
    "Scale-free: covariance flagged unavailable, not invented.")
box(sx, y3, sw, 0.026, "map_align  ·  Sim(3)  →  TF map→odom",
    "recovers metres-per-unit; loop closure lands here, so odometry never jumps",
    fc="#e3f5ea", ec=GREEN, tfs=7.2)
arr((sx + sw / 2, top - 0.038), (sx + sw / 2, y2), lw=0.9, color=PURPLE)
arr((sx + sw / 2, y2 - 0.050), (sx + sw / 2, y3), lw=0.9, color=PURPLE)
arr((0.128, TOPIC_BOT), (sx + 0.02, top), lw=1.0, color=PURPLE, ls=":")
cur -= H + GAP
FUS_BOT = y3 - 0.026

# -------------------------------------------------------------- 6 OUTPUTS --
H = LBL_H + PAD + 0.042 + 0.006 + 0.024 + PAD
top = band(cur, H, "6 · OUTPUTS", BG_OUT)
c = cols(0.020, 0.980, 4)
box(c[0][0], top, c[0][1], 0.042, "/tello_vio/odom",
    "pose + twist + full covariance\nchild_frame_id = base_link", ec=PURPLE)
box(c[1][0], top, c[1][1], 0.042, "TF   odom → base_link",
    "CONTINUOUS — controllers\ndifferentiate it, so it must not jump", ec=PURPLE)
box(c[2][0], top, c[2][1], 0.042, "~/pose_predicted",
    "extrapolated across the video latency\nfor control · inflated covariance", ec=PURPLE)
box(c[3][0], top, c[3][1], 0.042, "/diagnostics   ·   2 Hz",
    "per-measurement acceptance + NIS,\nbiases, σ, front-end status", ec=PURPLE)
box(0.020, top - 0.042 - 0.006, 0.960, 0.024,
    "TF tree:    map ──(may jump)──▶ odom ──(continuous)──▶ base_link "
    "──(static, calibrated)──▶ camera_optical        ·        exactly one node owns each edge",
    tfs=7.6, weight="normal")
arr((0.340, FUS_BOT), (0.340, top), lw=1.3, color=PURPLE)
arr((sx + 0.02, FUS_BOT), (0.700, top), lw=1.0, color=GREEN, ls="--")
cur -= H + GAP

# ---------------------------------------------------------- 7 CALIBRATION --
H = LBL_H + PAD + 0.044 + 0.006 + 0.022 + PAD
top = band(cur, H, "7 · CALIBRATION   ·   offline, once per airframe — these feed "
                   "PARAMETERS into every stage above", BG_CAL, lc=GOLD)
c = cols(0.020, 0.980, 3)
box(c[0][0], top, c[0][1], 0.044, "Camera intrinsics   ·   camera_calibration",
    "K and D. Tilt the board and fill the corners —\na flat, centred target trades focal length\n"
    "against distance and gives a confident wrong f")
box(c[1][0], top, c[1][1], 0.044, "IMU noise   ·   imu_calib   ·   120 s static",
    "overlapping Allan deviation → N, B, K per axis.\nTwo minutes, not ten seconds: the random-walk\n"
    "tail only appears in a long log")
box(c[2][0], top, c[2][1], 0.044, "Extrinsic + t_d   ·   camera_imu_calib",
    "rotation-only hand-eye (Kabsch, closed form) +\nangular-rate cross-correlation. A 5° extrinsic\n"
    "error looks exactly like scale drift")
box(0.020, top - 0.044 - 0.006, 0.960, 0.022,
    "Each reports an OBSERVABILITY score, not just a residual — a small residual with low "
    "excitation means the fit is confidently wrong",
    fc="#fdecec", ec=RED, tc=RED, tfs=7.4)
ax.text(0.500, cur + 0.004, "parameters:   R_BC ,  t_d ,  σ_a ,  σ_ω ,  K ,  D",
        fontsize=7.0, color=GOLD, ha="center", weight="bold")
cur -= H + GAP

# --------------------------------------------------------------- 8 FOOTER --
H = 0.076
top = band(cur, H, "WHAT THIS PIPELINE GUARANTEES  —  AND WHAT IT DOES NOT", "#eef0f3")
ax.text(0.024, top - 0.002, "Guarantees", fontsize=7.8, weight="bold", color=GREEN)
for i, s in enumerate([
        "Metric output with no GPS: 0.57 m RMS over 40 m flown (1.4 % of path)",
        "Roll and pitch bounded and absolute, 1.8° RMS — gravity-referenced",
        "Honest covariance: NEES 3.2–3.8 against an ideal 3.0",
        "Unmeasured quantities are representable as unmeasured, not as zero"]):
    ax.text(0.030, top - 0.014 - i * 0.0115, "✓   " + s, fontsize=7.1, color=INK)
ax.text(0.512, top - 0.002, "Does not", fontsize=7.8, weight="bold", color=RED)
for i, s in enumerate([
        "Bound yaw — no magnetometer, ~13° RMS drift over 60 s",
        "Survive a dead flow sensor without vision (20× worse)",
        "Relocalise in the fast path — KLT carries no descriptors",
        "Replace validation on real bags: every figure here is simulated"]):
    ax.text(0.518, top - 0.014 - i * 0.0115, "✗   " + s, fontsize=7.1, color=INK)

# Crop the unused axes space: the drawing is in 0-1 data coordinates, so
# shrinking ylim and the figure height by the same factor preserves proportions.
bottom = cur - 0.012
ax.set_ylim(bottom, 1.0)
fig.set_size_inches(11.2, 15.2 * (1.0 - bottom))

png = os.path.join(OUT, "pipeline.png")
pdf = os.path.join(HERE, "Tello_VIO_Pipeline.pdf")
fig.savefig(png)
fig.savefig(pdf)
print("wrote", png, "and", os.path.basename(pdf))
