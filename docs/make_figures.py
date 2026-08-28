#!/usr/bin/env python3
"""Generate every figure used by the technical report.

Diagrams are drawn rather than hand-placed so they stay in sync with the code,
and the result plots are produced by *running the actual estimator*, not by
transcribing numbers into a chart.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "workspace", "src", "tello_vio"))
sys.path.insert(0, os.path.join(HERE, "..", "workspace", "src", "tello_vio", "test"))

INK = "#14181f"
MUTED = "#5b6472"
LINE = "#c3cad4"
ACCENT = "#1f6feb"
WARN = "#d1651b"
GOOD = "#1f8a4c"
BG_A = "#eef3fb"
BG_B = "#fdf3e9"
BG_C = "#edf7f0"
BG_N = "#f4f6f9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#e6eaf0",
    "grid.linewidth": 0.7,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, text, fc=BG_N, ec=LINE, fs=8, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, weight=weight, color=tc, linespacing=1.45)


def arrow(ax, p0, p1, text=None, color=MUTED, style="-|>", ls="-", fs=7, dx=0, dy=0.014):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.1, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=1))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, text,
                ha="center", va="bottom", fontsize=fs, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    return fig, ax


# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = blank((11.0, 4.3))

    # ---- drone -----------------------------------------------------------
    box(ax, 0.008, 0.40, 0.115, 0.24,
        "DJI Tello\n\nno compute\nhere", fc="#e9edf3", weight="bold", fs=7.5)
    ax.text(0.065, 0.355, "WiFi AP, 2.4 GHz", ha="center", fontsize=6.2, color=MUTED)

    # ---- sockets ---------------------------------------------------------
    box(ax, 0.155, 0.70, 0.135, 0.15, "UDP :11111\nH.264 video", fc=BG_B, fs=7)
    box(ax, 0.155, 0.44, 0.135, 0.15, "UDP :8890\nstate broadcast", fc=BG_A, fs=7)
    box(ax, 0.155, 0.18, 0.135, 0.15, "UDP :8889\ncommands", fc=BG_C, fs=7)
    arrow(ax, (0.123, 0.58), (0.155, 0.76), "30 fps", dy=0.006, fs=6.3)
    arrow(ax, (0.123, 0.52), (0.155, 0.515), "~10 Hz", dy=0.008, fs=6.3)
    arrow(ax, (0.155, 0.25), (0.123, 0.44), None, style="<|-")

    # ---- driver ----------------------------------------------------------
    box(ax, 0.325, 0.16, 0.185, 0.72, "", fc="white", ec=ACCENT)
    ax.text(0.4175, 0.815, "tello driver", ha="center", fontsize=8.5,
            weight="bold", color=ACCENT)
    ax.text(0.4175, 0.765, "MultiThreadedExecutor", ha="center", fontsize=6.4, color=MUTED)
    for i, (t, c) in enumerate([("video callback group", BG_B),
                                ("telemetry callback group", BG_A),
                                ("control callback group", BG_C),
                                ("blocking-command worker", "#f2ecf7")]):
        y = 0.635 - i * 0.115
        box(ax, 0.342, y, 0.152, 0.085, t, fc=c, fs=6.3)
    ax.text(0.4175, 0.20, "no SDK call ever blocks a callback",
            ha="center", fontsize=6.0, color=MUTED, style="italic")

    arrow(ax, (0.290, 0.775), (0.325, 0.68))
    arrow(ax, (0.290, 0.515), (0.325, 0.565))
    arrow(ax, (0.325, 0.41), (0.290, 0.255), None, style="<|-")

    # ---- topics ----------------------------------------------------------
    topics = [
        ("/image_raw",   0.775, BG_B, "30 Hz · bgr8 · dedup"),
        ("/camera_info", 0.675, BG_B, "rescaled w/ video_scale"),
        ("/imu",         0.575, BG_A, "10 Hz · SI · FLU · no gyro"),
        ("/status",      0.475, BG_A, "10 Hz · metric velocity"),
        ("/tof",         0.375, BG_A, "10 Hz · sensor_msgs/Range"),
        ("/cmd_vel",     0.235, BG_C, "REP-103 command in"),
    ]
    for name, y, fc, note in topics:
        box(ax, 0.545, y - 0.042, 0.130, 0.084, "", fc=fc)
        ax.text(0.610, y + 0.014, name, ha="center", va="center", fontsize=6.9)
        ax.text(0.610, y - 0.022, note, ha="center", va="center",
                fontsize=5.3, color=MUTED)
    arrow(ax, (0.510, 0.55), (0.545, 0.55))

    # ---- estimator -------------------------------------------------------
    box(ax, 0.735, 0.660, 0.165, 0.190,
        "VoFrontend\nKLT + two-view geometry\nemits R and t̂ only —\nnever a magnitude",
        fc="white", ec=WARN, fs=6.8)
    box(ax, 0.735, 0.395, 0.165, 0.205,
        "ErrorStateKF\n22 error states\n+ stochastic clone", fc="white",
        ec=ACCENT, fs=7.2, weight="bold")
    box(ax, 0.735, 0.175, 0.165, 0.135,
        "map_align\nSim(3) alignment", fc="white", ec=GOOD, fs=7.2)

    arrow(ax, (0.670, 0.775), (0.735, 0.760), None)
    arrow(ax, (0.670, 0.560), (0.735, 0.520), None)
    arrow(ax, (0.670, 0.470), (0.735, 0.490), None)
    arrow(ax, (0.8175, 0.628), (0.8175, 0.602), None)


    # ---- outputs ---------------------------------------------------------
    ax.annotate("", xy=(0.935, 0.497), xytext=(0.900, 0.497),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
    ax.text(0.943, 0.530, "/tello_vio/odom", fontsize=6.6, color=ACCENT,
            weight="bold", va="center", ha="left")
    ax.text(0.943, 0.492, "TF odom→base_link", fontsize=6.4, color=ACCENT,
            va="center", ha="left")
    ax.text(0.943, 0.454, "~/pose_predicted", fontsize=6.4, color=MUTED,
            va="center", ha="left")

    ax.annotate("", xy=(0.935, 0.243), xytext=(0.900, 0.243),
                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.2))
    ax.text(0.943, 0.243, "TF map→odom", fontsize=6.4, color=GOOD,
            va="center", ha="left")

    ax.text(0.5, 0.055,
            "Everything to the right of the drone runs off-board. The Tello streams; it does not compute.",
            ha="center", fontsize=7.8, color=MUTED, style="italic")
    fig.savefig(os.path.join(OUT, "architecture.png"))
    plt.close(fig)


def fig_tf_tree():
    fig, ax = blank((9.0, 2.5))
    xs = [0.03, 0.28, 0.55, 0.80]
    names = ["map", "odom", "base_link", "camera_optical"]
    subs = ["global, loop-closed", "VIO origin", "IMU / body (FLU)", "optical (z fwd)"]
    for x, n, s in zip(xs, names, subs):
        box(ax, x, 0.50, 0.17, 0.20, n, fc="white", ec=INK, weight="bold", fs=9)
        ax.text(x + 0.085, 0.44, s, ha="center", fontsize=6.8, color=MUTED)

    arrow(ax, (0.20, 0.60), (0.28, 0.60), None, color=WARN)
    ax.text(0.24, 0.78, "map_align", ha="center", fontsize=7, color=WARN, weight="bold")
    ax.text(0.24, 0.72, "MAY JUMP", ha="center", fontsize=6.4, color=WARN)

    arrow(ax, (0.45, 0.60), (0.55, 0.60), None, color=ACCENT)
    ax.text(0.50, 0.78, "tello_vio", ha="center", fontsize=7, color=ACCENT, weight="bold")
    ax.text(0.50, 0.72, "CONTINUOUS", ha="center", fontsize=6.4, color=ACCENT)

    arrow(ax, (0.72, 0.60), (0.80, 0.60), None, color=GOOD)
    ax.text(0.76, 0.78, "static", ha="center", fontsize=7, color=GOOD, weight="bold")
    ax.text(0.76, 0.72, "calibrated", ha="center", fontsize=6.4, color=GOOD)

    ax.text(0.5, 0.22,
            "REP-105: controllers differentiate odom→base_link, so it must never jump.\n"
            "A loop closure is published as a correction to map→odom, which nothing differentiates.",
            ha="center", fontsize=7.6, color=INK)
    ax.text(0.5, 0.06, "Exactly one node owns each edge.",
            ha="center", fontsize=7.6, color=MUTED, style="italic")
    fig.savefig(os.path.join(OUT, "tf_tree.png"))
    plt.close(fig)


def fig_timeline():
    fig, ax = plt.subplots(figsize=(9.0, 3.3))
    ax.set_xlim(-0.05, 1.02); ax.set_ylim(0, 3.2); ax.grid(False)
    ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.set_xlabel("wall-clock time (s)")

    tel = np.arange(0.0, 1.0, 0.1)
    ax.hlines(2.55, 0, 1.0, color=LINE, lw=1)
    ax.plot(tel, np.full_like(tel, 2.55), "o", ms=5, color=ACCENT, zorder=3)
    ax.text(-0.04, 2.85, "telemetry (~10 Hz, ~10 ms late)", fontsize=8, color=ACCENT)

    cap = np.arange(0.0, 1.0, 1 / 30)
    ax.hlines(1.75, 0, 1.0, color=LINE, lw=1)
    ax.plot(cap, np.full_like(cap, 1.75), "|", ms=9, color=GOOD, zorder=3)
    ax.text(-0.04, 2.02, "frame CAPTURE (unobservable)", fontsize=8, color=GOOD)

    lat = 0.25
    arr = cap + lat
    arr = arr[arr <= 1.0]
    ax.hlines(0.95, 0, 1.0, color=LINE, lw=1)
    ax.plot(arr, np.full_like(arr, 0.95), "|", ms=9, color=WARN, zorder=3)
    ax.text(-0.04, 1.22, "frame ARRIVAL (what the header stamps)", fontsize=8, color=WARN)

    for k in (6, 12, 18):
        ax.annotate("", xy=(cap[k] + lat, 1.02), xytext=(cap[k], 1.68),
                    arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.0, ls=":"))
    ax.annotate("", xy=(cap[12] + lat, 0.55), xytext=(cap[12], 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.2))
    ax.text((cap[12] + lat / 2), 0.34, "t_d  ≈ 150–350 ms, jittery, no hardware timestamp",
            ha="center", fontsize=7.8, color=INK)

    ax.axvspan(0.30, 0.55, color=BG_A, alpha=0.55, zorder=0)
    ax.text(0.425, 3.02, "telemetry buffered here is replayed when the\n"
                         "matching frame finally arrives",
            ha="center", fontsize=7.2, color=ACCENT)
    fig.savefig(os.path.join(OUT, "timeline.png"))
    plt.close(fig)


def fig_factor_graph():
    fig, ax = blank((9.0, 3.6))
    xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    for i, x in enumerate(xs):
        fc = "#f0f0f2" if i == 0 else "white"
        ec = MUTED if i == 0 else INK
        ax.add_patch(plt.Circle((x, 0.62), 0.040, fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(x, 0.62, f"x{i}", ha="center", va="center", fontsize=8.5,
                zorder=4, weight="bold", color=ec)
        ax.text(x, 0.535, "R,p,v,b", ha="center", fontsize=6.2, color=MUTED)

    for i in range(len(xs) - 1):
        mid = (xs[i] + xs[i + 1]) / 2
        ax.plot([xs[i], xs[i + 1]], [0.62, 0.62], color=LINE, lw=1.0, zorder=1)
        ax.add_patch(plt.Rectangle((mid - 0.017, 0.603), 0.034, 0.034,
                                   fc=BG_A, ec=ACCENT, lw=1.1, zorder=3))
        ax.text(mid, 0.685, "IMU", ha="center", fontsize=6.5, color=ACCENT)
        # visual factor, drawn below
        ax.plot([xs[i], mid, xs[i + 1]], [0.62, 0.40, 0.62], color=LINE, lw=0.9, zorder=1)
        ax.add_patch(plt.Rectangle((mid - 0.017, 0.383), 0.034, 0.034,
                                   fc=BG_B, ec=WARN, lw=1.1, zorder=3))
        ax.text(mid, 0.345, "vis", ha="center", fontsize=6.5, color=WARN)

    for x in xs:
        ax.plot([x, x], [0.62, 0.80], color=LINE, lw=0.9, zorder=1)
        ax.add_patch(plt.Rectangle((x - 0.017, 0.803), 0.034, 0.034,
                                   fc=BG_C, ec=GOOD, lw=1.1, zorder=3))
    ax.text(xs[2], 0.865, "unary: attitude · body velocity · height",
            ha="center", fontsize=7, color=GOOD)

    ax.add_patch(plt.Rectangle((0.033, 0.555), 0.135, 0.135, fc="none",
                               ec=WARN, ls="--", lw=1.2, zorder=2))
    ax.text(0.10, 0.47, "leaving the window", ha="center", fontsize=6.8, color=WARN)

    ax.annotate("", xy=(0.30, 0.20), xytext=(0.10, 0.20),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.3))
    ax.text(0.50, 0.235,
            "marginalise, don't delete:  $H_{prior} = H_{kk} - H_{km}H_{mm}^{-1}H_{mk}$",
            ha="center", fontsize=8.5, color=INK)
    ax.text(0.50, 0.13,
            "The dropped keyframe's measurements constrain the ones that remain.\n"
            "Deleting it throws that information away and leaves the estimate over-confident.",
            ha="center", fontsize=7.2, color=MUTED)
    fig.savefig(os.path.join(OUT, "factor_graph.png"))
    plt.close(fig)


def fig_frontend():
    fig, ax = blank((9.0, 2.9))
    steps = [
        ("frame\n960×720", BG_N),
        ("grayscale\n+ INTER_AREA\n→ 480×360", BG_N),
        ("KLT track\n+ forward-backward\nreject", BG_B),
        ("re-detect?\ngrid-bucketed\nShi-Tomasi", BG_B),
        ("keyframe?\nrotation-compensated\nparallax", BG_A),
        ("E vs H\nmodel selection", BG_A),
        ("R, t̂\n+ per-measurement σ", BG_C),
    ]
    w, gap = 0.118, 0.0225
    x = 0.012
    for i, (t, fc) in enumerate(steps):
        box(ax, x, 0.42, w, 0.30, t, fc=fc, fs=6.8)
        if i < len(steps) - 1:
            arrow(ax, (x + w, 0.57), (x + w + gap, 0.57), None)
        x += w + gap

    ax.text(0.5, 0.30,
            "Detection runs only when the track count falls below threshold — not every frame.\n"
            "The image is never undistorted; only the ~150 points that reach the geometry stage are.",
            ha="center", fontsize=7.4, color=MUTED)
    ax.text(0.5, 0.13,
            "Measured: ~3 ms/frame at 480 px wide, ~150 features.  ORB detect+describe+match is roughly 10× that.",
            ha="center", fontsize=7.4, color=ACCENT)
    fig.savefig(os.path.join(OUT, "frontend.png"))
    plt.close(fig)


def fig_degeneracy():
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for ax in axes:
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.35, 2.3)
        ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)

    def cam(ax, x, y, ang, c):
        d = 0.26
        for s in (-0.42, 0.42):
            ax.plot([x, x + d * np.sin(ang + s) * 2.2], [y, y + d * np.cos(ang + s) * 2.2],
                    color=c, lw=1.0, alpha=0.55)
        ax.plot([x], [y], "s", ms=6, color=c)

    pts = np.array([[-0.9, 1.9], [-0.3, 2.1], [0.35, 1.85], [0.9, 2.05], [0.0, 1.6]])

    ax = axes[0]
    ax.set_title("good parallax", fontsize=8.5, color=GOOD)
    cam(ax, -0.55, 0.0, 0.0, ACCENT); cam(ax, 0.55, 0.0, 0.0, WARN)
    for p in pts:
        ax.plot([-0.55, p[0]], [0, p[1]], color=LINE, lw=0.6)
        ax.plot([0.55, p[0]], [0, p[1]], color=LINE, lw=0.6)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.5, color=INK)
    ax.annotate("", xy=(0.55, -0.2), xytext=(-0.55, -0.2),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0))
    ax.text(0, -0.32, "baseline", ha="center", fontsize=7, color=INK)

    ax = axes[1]
    ax.set_title("pure rotation → t is noise", fontsize=8.5, color=WARN)
    cam(ax, 0.0, 0.0, -0.25, ACCENT); cam(ax, 0.0, 0.0, 0.25, WARN)
    for p in pts:
        ax.plot([0, p[0]], [0, p[1]], color=LINE, lw=0.6)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.5, color=INK)
    ax.text(0, -0.28, "zero baseline\n→ rotation-only update", ha="center", fontsize=7, color=WARN)

    ax = axes[2]
    ax.set_title("planar scene → use homography", fontsize=8.5, color=WARN)
    cam(ax, -0.55, 0.0, 0.0, ACCENT); cam(ax, 0.55, 0.0, 0.0, WARN)
    wall = np.linspace(-1.2, 1.2, 7)
    ax.plot([-1.35, 1.35], [1.95, 1.95], color=INK, lw=2.0)
    for wx in wall:
        ax.plot([-0.55, wx], [0, 1.95], color=LINE, lw=0.6)
        ax.plot([0.55, wx], [0, 1.95], color=LINE, lw=0.6)
    ax.plot(wall, np.full_like(wall, 1.95), "o", ms=3.5, color=INK)
    ax.text(0, -0.28, "E is not determined\n→ score H against F", ha="center", fontsize=7, color=WARN)

    fig.savefig(os.path.join(OUT, "degeneracy.png"))
    plt.close(fig)


def fig_allan():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    tau = np.logspace(-2, 2.6, 300)
    N, B, K = 0.05, 0.010, 0.004
    adev = np.sqrt((N / np.sqrt(tau)) ** 2 + (B * 0.664) ** 2 + (K * np.sqrt(tau / 3)) ** 2)
    ax.loglog(tau, adev, color=ACCENT, lw=1.8)
    ax.loglog(tau, N / np.sqrt(tau), ls="--", color=MUTED, lw=0.9)
    ax.loglog(tau, K * np.sqrt(tau / 3), ls="--", color=MUTED, lw=0.9)
    ax.axhline(B * 0.664, ls=":", color=MUTED, lw=0.9)
    ax.set_xlabel(r"cluster time  $\tau$  (s)")
    ax.set_ylabel(r"Allan deviation  $\sigma(\tau)$")
    ax.text(0.03, N / np.sqrt(0.03) * 1.15, "slope −1/2\nwhite noise  N", fontsize=7.5, color=MUTED)
    ax.text(1.4, B * 0.664 * 0.62, "bias instability  B", fontsize=7.5, color=MUTED)
    ax.text(60, K * np.sqrt(60 / 3) * 1.1, "slope +1/2\nrandom walk  K", fontsize=7.5, color=MUTED)
    ax.plot([1.0], [N], "o", color=WARN, ms=6, zorder=5)
    ax.annotate(r"read $N$ at $\tau=1\,$s", xy=(1.0, N), xytext=(2.0, N * 2.6),
                fontsize=7.5, color=WARN,
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=0.9))
    ax.set_title("Identifying IMU noise from a static log", fontsize=9)
    fig.savefig(os.path.join(OUT, "allan.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Result figures: produced by running the estimator, not by transcription.
# --------------------------------------------------------------------------- #

def fig_results():
    import test_integration as T

    t, ep, ev, eq, tp, tv, tR, nees, kf = T.simulate(duration=60.0, seed=1)
    dist = T.path_length(tp)
    err = np.linalg.norm(ep - tp, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))

    fig = plt.figure(figsize=(9.4, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.34,
                          bottom=0.24, top=0.88)

    ax = fig.add_subplot(gs[0])
    ax.plot(tp[:, 0], tp[:, 1], color=INK, lw=1.8, label="ground truth", zorder=2)
    ax.plot(ep[:, 0], ep[:, 1], color=ACCENT, lw=1.1, ls="--", label="estimate", zorder=3)
    ax.plot(tp[0, 0], tp[0, 1], "o", color=GOOD, ms=6, label="start", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"60 s figure-of-eight, {dist:.0f} m flown", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, handlelength=1.6)

    ax = fig.add_subplot(gs[1])
    ax.plot(t, err, color=ACCENT, lw=1.2)
    ax.fill_between(t, 0, err, color=ACCENT, alpha=0.10)
    ax.axhline(rms, color=WARN, ls="--", lw=1.0)
    ax.text(1.0, rms + 0.035, f"RMS {rms:.2f} m", ha="left", fontsize=7, color=WARN)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title(f"Grows slowly: {100*err[-1]/dist:.1f} % of path", fontsize=8.5)
    ax.set_ylim(0, max(err) * 1.20)

    ax = fig.add_subplot(gs[2])
    ax.plot(t, np.clip(nees, 0, 12), color=MUTED, lw=0.6, alpha=0.55)
    w = 40
    sm = np.convolve(nees, np.ones(w) / w, "same")
    ax.plot(t[w:-w], sm[w:-w], color=ACCENT, lw=1.7)
    ax.axhline(3.0, color=GOOD, ls="--", lw=1.1)
    ax.text(1.0, 3.3, "ideal = 3", ha="left", fontsize=7, color=GOOD)
    ax.set_ylim(0, 12)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position NEES")
    ax.set_title(f"NEES {np.mean(nees[len(nees)//4:]):.1f} vs ideal 3", fontsize=8.5)

    fig.savefig(os.path.join(OUT, "results.png"))
    plt.close(fig)


def fig_vision_benefit():
    import test_integration as T
    seeds = range(5)
    labels, with_v, without_v = [], [], []
    for fn, lab in ((0.08, "healthy\nσ=0.08"), (0.30, "degraded\nσ=0.30"),
                    (0.80, "bad\nσ=0.80")):
        a, b = [], []
        for s in seeds:
            _, p1, _, _, tp, _, _, _, _ = T.simulate(duration=40.0, seed=s,
                                                     flow_noise=fn, with_vision=True)
            _, p2, _, _, tq, _, _, _, _ = T.simulate(duration=40.0, seed=s,
                                                     flow_noise=fn, with_vision=False)
            n = min(len(p1), len(p2))
            a.append(T.ate(p1[:n], tp[:n])); b.append(T.ate(p2[:n], tq[:n]))
        labels.append(lab); with_v.append(np.mean(a)); without_v.append(np.mean(b))

    # Flow-dead case
    a, b = [], []
    for s in seeds:
        _, p1, _, _, tp, _, _, _, _ = T.simulate(duration=40.0, seed=s,
                                                 with_flow=False, with_vision=True)
        _, p2, _, _, tq, _, _, _, _ = T.simulate(duration=40.0, seed=s,
                                                 with_flow=False, with_vision=False)
        n = min(len(p1), len(p2))
        a.append(T.ate(p1[:n], tp[:n])); b.append(T.ate(p2[:n], tq[:n]))
    labels.append("dead\nno flow"); with_v.append(np.mean(a)); without_v.append(np.mean(b))

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, with_v, w, label="with vision", color=ACCENT)
    ax.bar(x + w / 2, without_v, w, label="without vision", color="#b9c2ce")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ATE (m, log scale)")
    ax.set_xlabel("Tello optical-flow velocity quality")
    ax.set_title("What the camera is actually worth (mean of 5 seeds, 40 s)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    for xi, (a_, b_) in enumerate(zip(with_v, without_v)):
        ax.text(xi, max(a_, b_) * 1.35, f"{100*(1-a_/b_):+.0f}%", ha="center",
                fontsize=7.5, color=GOOD if a_ < b_ else MUTED, weight="bold")
    ax.set_ylim(0.1, 60)
    fig.savefig(os.path.join(OUT, "vision_benefit.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); print("architecture")
    fig_tf_tree(); print("tf_tree")
    fig_timeline(); print("timeline")
    fig_factor_graph(); print("factor_graph")
    fig_frontend(); print("frontend")
    fig_degeneracy(); print("degeneracy")
    fig_allan(); print("allan")
    fig_results(); print("results")
    fig_vision_benefit(); print("vision_benefit")
    print("figures ->", OUT)
