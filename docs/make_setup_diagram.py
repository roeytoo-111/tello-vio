#!/usr/bin/env python3
"""How to physically set up the ArUco ground-truth measurement at home."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, LINE = "#14181f", "#616b7a", "#c6ced9"
BLUE, ORANGE, GREEN, RED, GOLD = "#1f6feb", "#c2600f", "#1f8a4c", "#b3261e", "#8a6a12"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.facecolor": "white"})

fig = plt.figure(figsize=(15.5, 10.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.18)

fig.suptitle("Measuring VIO error at home with one printed marker",
             fontsize=16, weight="bold", y=0.975)
fig.text(0.5, 0.938,
         "The marker never moves, so it is an absolute reference. VIO drifts. "
         "The gap between them is the error you are measuring.",
         ha="center", fontsize=10, color=MUTED)

# ------------------------------------------------------------- top-down ----
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Top-down view of the room", fontsize=11.5, weight="bold")
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.45, 3.75); ax.set_aspect("equal")

ax.add_patch(Rectangle((-2.5, 3.0), 5.0, 0.28, fc="#e3e6ea", ec=MUTED, hatch="///"))
ax.text(-1.95, 3.14, "WALL", ha="center", va="center", fontsize=9,
        color=MUTED, weight="bold")
ax.add_patch(Rectangle((-0.25, 2.90), 0.5, 0.10, fc="black"))
ax.annotate("marker, taped flat (150 mm)", xy=(0.25, 2.95), xytext=(0.62, 3.44),
            fontsize=8.5, weight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))

# usable band
ax.add_patch(Wedge((0, 3.0), 2.5, 208, 332, fc=GREEN, alpha=.10, ec="none"))
ax.add_patch(Wedge((0, 3.0), 1.0, 208, 332, fc="white", ec="none"))
ax.add_patch(Wedge((0, 3.0), 2.5, 208, 332, fill=False, ec=GREEN, lw=1.2, ls="--"))
ax.add_patch(Wedge((0, 3.0), 1.0, 208, 332, fill=False, ec=GREEN, lw=1.2, ls="--"))
ax.text(-1.72, 2.15, "USABLE ZONE\n1.0 - 2.5 m", ha="center", fontsize=9,
        weight="bold", color=GREEN)

th = np.linspace(-0.72, 0.72, 60)
ax.plot(1.45 * np.sin(th), 3.0 - 1.55 * np.cos(th), color=BLUE, lw=2.2, zorder=3)
for a in (-0.72, 0.0, 0.72):
    ax.add_patch(Rectangle((1.45 * np.sin(a) - .11, 3.0 - 1.55 * np.cos(a) - .07),
                           .22, .14, fc=BLUE, ec="none", zorder=4))
ax.annotate("", xy=(1.45 * np.sin(0.72), 3.0 - 1.55 * np.cos(0.72)),
            xytext=(1.45 * np.sin(0.45), 3.0 - 1.55 * np.cos(0.45)),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
ax.text(0, 0.55, "fly an ARC, not straight in and out",
        ha="center", fontsize=9.5, color=BLUE, weight="bold")
ax.text(0, 0.20, "keeps the marker in view AND varies the viewing angle",
        ha="center", fontsize=8, color=MUTED)

ax.plot([0, 0], [2.88, 1.35], color=RED, lw=1.3, ls=":")
ax.text(1.62, 2.05, "avoid dead-on:\nrotation is\nambiguous",
        fontsize=8, color=RED, va="center", ha="center")
ax.set_xlabel("metres"); ax.grid(alpha=.15); ax.set_yticks([])

# ----------------------------------------------------------- side view ----
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Side view", fontsize=11.5, weight="bold")
ax.set_xlim(-0.3, 3.4); ax.set_ylim(-0.1, 2.5); ax.set_aspect("equal")

ax.add_patch(Rectangle((-0.25, 0), 0.25, 2.5, fc="#e3e6ea", ec=MUTED, hatch="///"))
ax.add_patch(Rectangle((-0.02, 1.15), 0.06, 0.30, fc="black"))
ax.annotate("marker at the drone's\ncruising height (~1.2 m)",
            xy=(0.05, 1.30), xytext=(0.42, 2.18), fontsize=8.5, weight="bold",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.plot([-0.3, 3.4], [0, 0], color=MUTED, lw=2)
ax.text(3.3, 0.06, "floor", ha="right", fontsize=8, color=MUTED)

ax.add_patch(Rectangle((1.75, 1.22), 0.26, 0.11, fc=BLUE, ec="none"))
ax.text(1.88, 1.45, "drone", ha="center", fontsize=8.5, color=BLUE, weight="bold")
for dy, a in ((0.42, .35), (0.0, .55), (-0.42, .35)):
    ax.add_patch(FancyArrowPatch((1.75, 1.28), (0.06, 1.30 + dy),
                                 arrowstyle="-", color=ORANGE, lw=1.1, alpha=a))
ax.text(1.35, 0.95, "camera sees the marker", fontsize=8.5, color=ORANGE, ha="center")

ax.annotate("", xy=(1.75, 0.62), xytext=(0.04, 0.62),
            arrowprops=dict(arrowstyle="<|-|>", color=GREEN, lw=1.6))
ax.text(0.90, 0.70, "1.0 - 2.5 m", ha="center", fontsize=9,
        color=GREEN, weight="bold")
ax.text(1.72, 0.30,
        "Match the marker HEIGHT to the drone's flight height.\n"
        "The Tello camera looks FORWARD, not down.",
        fontsize=8.3, color=MUTED)
ax.axis("off")

# ------------------------------------------------------- how it measures --
ax = fig.add_subplot(gs[1, 0])
ax.set_title("How one photo becomes an absolute position",
             fontsize=11.5, weight="bold")
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

steps = [
    ("1", "The marker's real size is known", "you measured it: 150 mm", BLUE),
    ("2", "Its four corners are found in the image", "to sub-pixel accuracy", BLUE),
    ("3", "Apparent size gives DISTANCE", "smaller in frame = further away", ORANGE),
    ("4", "Its distortion gives DIRECTION + ANGLE", "a square looks like a trapezoid\nwhen seen off-axis", ORANGE),
    ("5", "So: camera position relative to the marker", "solvePnP. ~1.5 cm accurate", GREEN),
    ("6", "The marker never moves, so this NEVER DRIFTS", "unlike VIO, which integrates motion", GREEN),
]
y = 5.80
for num, title, sub, col in steps:
    ax.add_patch(plt.Circle((0.42, y), 0.235, fc=col, ec="none"))
    ax.text(0.42, y, num, ha="center", va="center", fontsize=9,
            color="white", weight="bold")
    ax.text(0.92, y + 0.12, title, fontsize=9.6, weight="bold", color=INK, va="center")
    ax.text(0.92, y - 0.22, sub, fontsize=8.3, color=MUTED, va="center")
    y -= 0.83

ax.add_patch(FancyBboxPatch((0.05, 0.00), 9.9, 0.56,
                            boxstyle="round,pad=0.04,rounding_size=0.06",
                            fc="#eaf6ee", ec=GREEN, lw=1.2))
ax.text(5.0, 0.28,
        "error  =  | VIO position  −  marker position |,  every frame",
        ha="center", va="center", fontsize=9.5, weight="bold", color=GREEN)

# --------------------------------------------------------------- do/don't --
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Setup checklist", fontsize=11.5, weight="bold")
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

do = [
    "Tape it FLAT on a wall - any bend corrupts the pose",
    "Height = drone's flight height (camera looks forward)",
    "Stay 1.0-2.5 m away (150 mm marker: 138 px at 1 m, 55 px at 2.5 m)",
    "Fly an ARC so the viewing angle changes",
    "Even, indirect light - no glare, no direct lamp on it",
    "Textured room: VIO needs features too, not just the marker",
]
dont = [
    "Don't fly dead-on only - rotation becomes ambiguous",
    "Don't go past ~3 m - the marker gets too small to detect",
    "Don't let it leave the frame - gaps break the comparison",
    "Don't print on glossy paper - reflections kill detection",
    "Don't trust the nominal size - measure the print",
]
ax.text(0.2, 6.02, "DO", fontsize=10.5, weight="bold", color=GREEN)
y = 5.62
for t in do:
    ax.text(0.35, y, "✓  " + t, fontsize=8.5, color=INK, va="center"); y -= 0.40
y -= 0.16
ax.text(0.2, y, "DON'T", fontsize=10.5, weight="bold", color=RED); y -= 0.38
for t in dont:
    ax.text(0.35, y, "✗  " + t, fontsize=8.5, color=INK, va="center"); y -= 0.40

ax.add_patch(FancyBboxPatch((0.05, 0.00), 9.9, 0.46,
                            boxstyle="round,pad=0.04,rounding_size=0.06",
                            fc="#fff8e3", ec=GOLD, lw=1.2))
ax.text(5.0, 0.23,
        "A 2 × 2 m flight volume is plenty — you measure drift RATE, "
        "not distance flown.",
        ha="center", va="center", fontsize=8.8, weight="bold", color=GOLD)

png = os.path.join(OUT, "ground_truth_setup.png")
fig.savefig(png, bbox_inches="tight")
print("wrote", png)
