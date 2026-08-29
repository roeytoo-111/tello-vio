#!/usr/bin/env python3
"""Generate print-ready A4 targets: an ArUco marker and a calibration board.

Both are emitted as PDF at EXACT physical scale, so "print at 100 %" produces
a target of the stated size. PDF rather than PNG precisely because PDF carries
real physical units -- a PNG's size depends on whatever DPI the print dialog
guesses, which is how calibration targets end up a few percent wrong and
silently poison every metric result downstream.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "print")
os.makedirs(OUT, exist_ok=True)

A4_W_IN, A4_H_IN = 8.268, 11.693        # 210 x 297 mm
MM = 1.0 / 25.4


def a4_page(pdf, draw, footer):
    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 210); ax.set_ylim(0, 297)   # millimetres
    ax.axis("off")
    draw(ax)
    ax.text(105, 12, footer, ha="center", va="center", fontsize=8, color="#444")
    ax.text(105, 6, "PRINT AT 100 % SCALE - disable 'fit to page' / 'shrink to fit'",
            ha="center", va="center", fontsize=8, weight="bold", color="#b3261e")
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------- ArUco -----
MARKER_MM = 150.0        # 15 cm: big enough to detect across a small room

def draw_marker(ax):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = (cv2.aruco.generateImageMarker(d, 0, 800)
           if hasattr(cv2.aruco, "generateImageMarker")
           else cv2.aruco.drawMarker(d, 0, 800))
    x0, y0 = (210 - MARKER_MM) / 2, 297 - MARKER_MM - 45
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=[x0, x0 + MARKER_MM, y0, y0 + MARKER_MM],
              interpolation="nearest", zorder=2)
    # Ruler ticks along the bottom edge so the print can be verified in place.
    for k in range(0, int(MARKER_MM) + 1, 10):
        ax.plot([x0 + k, x0 + k], [y0 - 3, y0 - (7 if k % 50 == 0 else 5)],
                color="k", lw=0.8)
        if k % 50 == 0:
            ax.text(x0 + k, y0 - 11, f"{k}", ha="center", fontsize=7)
    ax.plot([x0, x0 + MARKER_MM], [y0 - 3, y0 - 3], color="k", lw=0.8)
    ax.text(105, y0 - 22,
            f"MEASURE the black square. It should be {MARKER_MM:.0f} mm.",
            ha="center", fontsize=10, weight="bold")
    ax.text(105, y0 - 30,
            "Pass what you actually measure, in metres, as marker_size_m.",
            ha="center", fontsize=9, color="#333")
    ax.text(105, 282, "ArUco 4x4_50  id 0   -   VIO ground truth",
            ha="center", fontsize=12, weight="bold")


# -------------------------------------------------------- checkerboard -----
SQ_MM, COLS, ROWS = 25.0, 9, 7          # 9x7 squares -> 8x6 INNER corners

def draw_board(ax):
    w, h = COLS * SQ_MM, ROWS * SQ_MM
    x0, y0 = (210 - w) / 2, 297 - h - 60
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                ax.add_patch(plt.Rectangle((x0 + c * SQ_MM, y0 + r * SQ_MM),
                                           SQ_MM, SQ_MM, color="black", zorder=2))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="#888", lw=0.5))
    for k in range(0, int(SQ_MM * 4) + 1, 10):
        ax.plot([x0 + k, x0 + k], [y0 - 3, y0 - (7 if k % 50 == 0 else 5)],
                color="k", lw=0.8)
        if k % 50 == 0:
            ax.text(x0 + k, y0 - 11, f"{k}", ha="center", fontsize=7)
    ax.plot([x0, x0 + SQ_MM * 4], [y0 - 3, y0 - 3], color="k", lw=0.8)
    ax.text(105, 282, "Camera calibration target", ha="center",
            fontsize=12, weight="bold")
    ax.text(105, y0 - 22, f"MEASURE one square. It should be {SQ_MM:.0f} mm.",
            ha="center", fontsize=10, weight="bold")
    ax.text(105, y0 - 32,
            f"{COLS}x{ROWS} squares  =>  --size {COLS-1}x{ROWS-1}  "
            "(the tool wants INNER CORNERS, not squares)",
            ha="center", fontsize=9, color="#333")
    ax.text(105, y0 - 41,
            "ros2 run camera_calibration cameracalibrator "
            f"--size {COLS-1}x{ROWS-1} --square 0.025 \\\n"
            "    image:=/image_raw camera:=/camera",
            ha="center", fontsize=7.5, family="monospace", color="#1f6feb")


path = os.path.join(OUT, "tello_vio_print_targets.pdf")
with PdfPages(path) as pdf:
    a4_page(pdf, draw_marker,
            "Tape flat to a wall. Mount it so the drone sees it AT AN ANGLE, "
            "not dead-on (head-on views are rotationally ambiguous).")
    a4_page(pdf, draw_board,
            "Hold it flat and rigid - tape it to card. A bent board silently "
            "corrupts the calibration.")
print("wrote", path)
