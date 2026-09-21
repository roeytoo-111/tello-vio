#!/usr/bin/env python3
"""Diagrams for the drone-chase reference PDF (docs/make_chase_reference.py).

Everything is drawn with matplotlib so the figures regenerate from source
and stay consistent with the document's palette.
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_chase')
os.makedirs(OUT, exist_ok=True)

INK = '#14181f'
MUTED = '#5b6472'
ACCENT = '#1f6feb'
WARN = '#b8520f'
GOOD = '#1f8a4c'
LINE = '#d3dae3'
BG_A = '#eef3fb'
BG_B = '#edf7f0'
BG_C = '#fdf3e9'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.4,
    'axes.edgecolor': LINE, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.dpi': 200, 'savefig.dpi': 200,
})


def _box(ax, x, y, w, h, text, fc, ec, fs=8.0, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=tc, zorder=3, fontweight='bold' if bold else 'normal',
            linespacing=1.45)


def _arrow(ax, p, q, text='', color=MUTED, fs=7.0, rad=0.0, ls='-', dy=0.012):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11,
                                 lw=1.0, color=color, zorder=1,
                                 linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha='center',
                va='bottom', fontsize=fs, color=color, zorder=3)


def fig_tiers():
    """The three tiers, what each proves, and what flows between them."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    cols = [(0.02, BG_A, ACCENT, 'TIER A  ·  Gym',
             'pure Python maths\nno renderer, no physics engine',
             'thousands of steps / min', 'proves the behaviour\ncan be LEARNED'),
            (0.35, BG_B, GOOD, 'TIER B  ·  Gazebo',
             'rigid-body physics, rotors,\ndrag, lockstep, headless',
             '~real time, no window', 'proves the maths\nMATCHES PHYSICS'),
            (0.68, BG_C, WARN, 'TIER C  ·  Unreal',
             'rendered 960x720 frames,\nreal YOLO in the loop',
             '~2 decisions / s (render bound)', 'proves it survives\nREAL VISION')]
    for x, bg, ec, title, what, speed, proves in cols:
        _box(ax, x, 0.30, 0.30, 0.56, '', bg, ec)
        ax.text(x + 0.15, 0.805, title, ha='center', va='center', fontsize=9.2,
                fontweight='bold', color=ec)
        ax.text(x + 0.15, 0.66, what, ha='center', va='center', fontsize=7.6, color=INK)
        ax.text(x + 0.15, 0.525, speed, ha='center', va='center', fontsize=7.2,
                color=MUTED, style='italic')
        ax.text(x + 0.15, 0.395, proves, ha='center', va='center', fontsize=7.4,
                color=ec, fontweight='bold', linespacing=1.5)

    _arrow(ax, (0.325, 0.58), (0.345, 0.58), '', MUTED)
    _arrow(ax, (0.655, 0.58), (0.675, 0.58), '', MUTED)
    ax.text(0.335, 0.60, 'checkpoint', ha='center', fontsize=6.6, color=MUTED, rotation=90)
    ax.text(0.665, 0.60, 'checkpoint', ha='center', fontsize=6.6, color=MUTED, rotation=90)

    _box(ax, 0.02, 0.06, 0.96, 0.16,
         'THE ONE CONTRACT   —   every tier imports the same modules\n'
         'chase_gym.observation  ·  chase_gym.env.forward_command  ·  chase_gym.constants  ·  the same checkpoint format',
         '#ffffff', INK, fs=7.8)
    ax.annotate('', xy=(0.17, 0.22), xytext=(0.17, 0.30),
                arrowprops=dict(arrowstyle='-', lw=0.9, color=LINE))
    ax.annotate('', xy=(0.50, 0.22), xytext=(0.50, 0.30),
                arrowprops=dict(arrowstyle='-', lw=0.9, color=LINE))
    ax.annotate('', xy=(0.83, 0.22), xytext=(0.83, 0.30),
                arrowprops=dict(arrowstyle='-', lw=0.9, color=LINE))
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'tiers.png'), bbox_inches='tight')
    plt.close(fig)


def fig_flight_graph():
    """The six-node ROS 2 flight graph."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.0); ax.axis('off')

    _box(ax, 0.01, 0.875, 0.19, 0.095, 'tello driver\n/image_raw 30 Hz', '#ffffff', MUTED, fs=7.2)
    _box(ax, 0.27, 0.875, 0.20, 0.095, 'chase_detector\nYOLO', '#ffffff', MUTED, fs=7.2)
    _box(ax, 0.55, 0.875, 0.23, 0.095, 'chase_state\n10 Hz control clock', BG_A, ACCENT, fs=7.2, bold=True)

    _arrow(ax, (0.20, 0.9225), (0.27, 0.9225), '/image_raw', dy=0.006)
    _arrow(ax, (0.47, 0.9225), (0.55, 0.9225), '/chase/detection', dy=0.006)

    _box(ax, 0.04, 0.615, 0.24, 0.115, 'chase_policy\ntrained actor (ONNX)\ninference only', BG_A, ACCENT, fs=7.2, bold=True)
    _box(ax, 0.36, 0.615, 0.24, 0.115, 'chase_depth_rule\nmetric standoff law\nchase_gym.forward_command', BG_A, ACCENT, fs=6.8, bold=True)
    _box(ax, 0.68, 0.615, 0.24, 0.115, 'chase_behaviour\nSEARCH / TRACK\nREACQUIRE / PATROL', BG_A, ACCENT, fs=6.8, bold=True)

    for x in (0.16, 0.48, 0.80):
        _arrow(ax, (0.665, 0.875), (x, 0.732), '', LINE)
    ax.text(0.40, 0.805, '/chase/observation   14-dim vector + spec hash',
            ha='center', fontsize=7.0, color=ACCENT)

    _box(ax, 0.22, 0.375, 0.56, 0.105,
         'chase_mixer         exclusive axis ownership\n'
         'linear.z + angular.z = policy    ·    linear.x = depth rule    ·    linear.y ≡ 0',
         '#ffffff', INK, fs=7.0)
    for x in (0.16, 0.48, 0.80):
        _arrow(ax, (x, 0.615), (0.50, 0.482), '', LINE)

    _box(ax, 0.16, 0.175, 0.68, 0.105,
         'chase_safety         the ONLY publisher of /cmd_vel\n'
         'engage gate  ·  speed cap  ·  20 Hz hold-last  ·  latching auto-disarm  →  land',
         BG_C, WARN, fs=7.0)
    _arrow(ax, (0.50, 0.375), (0.50, 0.282), '/chase/cmd_raw', ACCENT, dy=0.005)
    _arrow(ax, (0.50, 0.175), (0.50, 0.108), '/cmd_vel   sticks in [-1,1]', WARN, dy=0.005)
    _box(ax, 0.16, 0.035, 0.68, 0.07,
         'tello driver   ·   scales to ±100   ·   0.35 s dead-man zeroes a stale command',
         '#ffffff', MUTED, fs=7.0)

    # feedback edge: what was actually sent, back to the assembler
    for a, b in (((0.84, 0.2275), (0.955, 0.2275)),
                 ((0.955, 0.2275), (0.955, 0.9225))):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-', lw=0.9, color=GOOD))
    ax.add_patch(FancyArrowPatch((0.955, 0.9225), (0.78, 0.9225), arrowstyle='-|>',
                                 mutation_scale=11, lw=0.9, color=GOOD))
    ax.text(0.945, 0.55, '/chase/action_sent', rotation=90, va='center',
            ha='right', fontsize=6.9, color=GOOD)
    ax.text(0.975, 0.55, 'what actually reached the wire', rotation=90,
            va='center', ha='left', fontsize=6.3, color=GOOD, style='italic')
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'flight_graph.png'), bbox_inches='tight')
    plt.close(fig)


def fig_observation():
    """The 14-dim observation vector, element by element."""
    fig, ax = plt.subplots(figsize=(7.4, 1.65))
    ax.set_xlim(0, 14); ax.set_ylim(0, 1); ax.axis('off')
    groups = [(0, 2, 'e_x  e_y\ncurrent error', ACCENT, BG_A),
              (2, 2, "e_x'  e_y'\nprevious error", ACCENT, BG_A),
              (4, 1, 'visible\n0 / 1', GOOD, BG_B),
              (5, 1, 'staleness\n0..1', GOOD, BG_B),
              (6, 8, 'a(t-1) … a(t-4)     four most recent ACTIONS SENT, newest first\n'
                     'this is what makes the policy latency-aware: k = 4 ticks covers the 350 ms worst-case link delay',
               WARN, BG_C)]
    for start, n, label, ec, bg in groups:
        ax.add_patch(FancyBboxPatch((start + 0.06, 0.22), n - 0.12, 0.56,
                                    boxstyle='round,pad=0.01,rounding_size=0.06',
                                    fc=bg, ec=ec, lw=1.1))
        ax.text(start + n / 2, 0.50, label, ha='center', va='center',
                fontsize=7.2 if n < 8 else 6.9, color=INK, linespacing=1.5)
    for i in range(15):
        ax.plot([i, i], [0.14, 0.19], color=LINE, lw=0.7)
    for i in range(14):
        ax.text(i + 0.5, 0.06, str(i), ha='center', fontsize=6.2, color=MUTED)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'observation.png'), bbox_inches='tight')
    plt.close(fig)


def fig_depth_law():
    """The forward-speed law: published ratio rule vs the metric standoff rule."""
    fx, W = 919.42, 0.098
    w = np.linspace(8, 400, 800)
    d = fx * W / w

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))

    ax = axes[0]
    ax.plot(w, d, color=ACCENT, lw=1.6)
    ax.axhline(2.0, color=GOOD, lw=1.0, ls='--')
    ax.text(300, 2.15, 'standoff 2.0 m', color=GOOD, fontsize=7.0)
    # published thresholds, expressed as box width fractions of a 960 px frame
    for frac, lbl in ((0.20, '20 % frame width\n"too far, go forward"'),
                      (0.55, '55 % frame width\n"too close, back off"')):
        wpx = frac * 960
        if wpx <= w.max():
            ax.axvline(wpx, color=WARN, lw=1.0, ls=':')
            ax.text(wpx + 6, 3.6 if frac < 0.3 else 2.9, lbl, color=WARN, fontsize=6.4)
    ax.set_xlabel('box width  w  (px)')
    ax.set_ylabel('range  d = f$_x$·W / w   (m)')
    ax.set_ylim(0, 6); ax.set_xlim(0, 400)
    ax.set_title('Why the published ratio rule is unusable here', fontsize=8.2,
                 color=INK, pad=6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    dd = np.linspace(0.05, 6, 600)
    follow = np.clip(0.8 * (dd - 2.0), -0.8, 0.8)
    follow[np.abs(dd - 2.0) < 0.25] = 0.0
    inter = np.clip(1.0 * (dd - 0.5), 0.15, 1.2)
    inter[dd <= 0.5] = 0.0
    ax.plot(dd, follow, color=ACCENT, lw=1.7, label='FOLLOW  (standoff 2.0 m)')
    ax.plot(dd, inter, color=WARN, lw=1.7, label='INTERCEPT  (capture 0.5 m)')
    ax.axhline(0, color=LINE, lw=0.8)
    ax.axvspan(1.75, 2.25, color=GOOD, alpha=0.12)
    ax.text(2.0, -0.55, 'deadband', ha='center', fontsize=6.6, color=GOOD)
    ax.set_xlabel('estimated range  d  (m)')
    ax.set_ylabel('commanded v$_{fwd}$  (m/s)')
    ax.set_xlim(0, 6); ax.set_ylim(-0.95, 1.35)
    ax.legend(fontsize=6.6, frameon=False, loc='lower right')
    ax.set_title('The metric law actually used', fontsize=8.2, color=INK, pad=6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, 'depth_law.png'), bbox_inches='tight')
    plt.close(fig)


def fig_timeline():
    """One control tick, and where the 150-350 ms of latency lives."""
    fig, ax = plt.subplots(figsize=(7.4, 2.1))
    ax.set_xlim(0, 460); ax.set_ylim(0, 1); ax.axis('off')
    seg = [(0, 33, 'capture\n+ H.264', '#ffffff', MUTED),
           (33, 150, 'Wi-Fi link + decode\n150–350 ms measured', BG_C, WARN),
           (183, 35, 'YOLO\n~15 ms', BG_B, GOOD),
           (218, 12, 'assemble\nobs', BG_A, ACCENT),
           (230, 10, 'actor\n~0.1 ms', BG_A, ACCENT),
           (240, 14, 'mix +\nsafety', BG_A, ACCENT),
           (254, 60, '/cmd_vel → RC', '#ffffff', MUTED)]
    for x0, w, lbl, fc, ec in seg:
        ax.add_patch(FancyBboxPatch((x0, 0.42), w - 2, 0.3,
                                    boxstyle='round,pad=0.004,rounding_size=3',
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x0 + w / 2, 0.57, lbl, ha='center', va='center', fontsize=6.3,
                color=INK, linespacing=1.4)
    ax.annotate('', xy=(0, 0.30), xytext=(314, 0.30),
                arrowprops=dict(arrowstyle='<->', lw=1.0, color=INK))
    ax.text(157, 0.19, 'end-to-end sensorimotor delay  ≈  1.5 – 3.5 control periods\n'
                       'the k = 4 action history is exactly what compensates it',
            ha='center', fontsize=7.0, color=INK, linespacing=1.5)
    ax.plot([314, 314], [0.40, 0.78], color=ACCENT, lw=1.0, ls='--')
    ax.text(318, 0.80, 'next tick (100 ms)', fontsize=6.6, color=ACCENT)
    ax.plot([350, 350], [0.40, 0.78], color=WARN, lw=1.0, ls='--')
    ax.text(354, 0.80, 'driver dead-man (350 ms)', fontsize=6.6, color=WARN)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'timeline.png'), bbox_inches='tight')
    plt.close(fig)


def fig_behaviour_fsm():
    """The behaviour state machine."""
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    nodes = {'SEARCH': (0.09, 0.62), 'TRACK': (0.40, 0.62),
             'REACQUIRE': (0.70, 0.62), 'PATROL': (0.70, 0.16)}
    sub = {'SEARCH': 'no target yet\nrotate in place',
           'TRACK': 'policy + depth rule\nactive',
           'REACQUIRE': 'hold 1.0 s,\nthen rotate',
           'PATROL': 'lost > 8 s\nslow continuous scan'}
    cols = {'SEARCH': MUTED, 'TRACK': GOOD, 'REACQUIRE': WARN, 'PATROL': WARN}
    for n, (x, y) in nodes.items():
        _box(ax, x, y, 0.22, 0.26, f'{n}\n\n{sub[n]}', '#ffffff', cols[n], fs=7.2)
        ax.text(x + 0.11, y + 0.205, n, ha='center', fontsize=8.4,
                fontweight='bold', color=cols[n])
    _arrow(ax, (0.31, 0.75), (0.40, 0.75), 'target detected', GOOD)
    _arrow(ax, (0.62, 0.70), (0.70, 0.70), 'detection lost', WARN)
    _arrow(ax, (0.70, 0.80), (0.62, 0.80), 'reacquired', GOOD, rad=0.0)
    _arrow(ax, (0.81, 0.62), (0.81, 0.42), 'loss > timeout', WARN)
    ax.add_patch(FancyArrowPatch((0.70, 0.29), (0.51, 0.62), arrowstyle='-|>',
                                 mutation_scale=11, lw=1.0, color=GOOD,
                                 connectionstyle='arc3,rad=0.25'))
    ax.text(0.52, 0.36, 'detected', fontsize=7.0, color=GOOD)
    ax.text(0.02, 0.05,
            'Every timeout here is a design choice, not a measurement: the source work '
            'names the four knobs and fixes none of them.',
            fontsize=6.8, color=MUTED, style='italic')
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'behaviour.png'), bbox_inches='tight')
    plt.close(fig)


def fig_td3():
    """TD3's three fixes over DDPG."""
    fig, ax = plt.subplots(figsize=(7.4, 2.45))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    items = [
        (0.01, 'Twin critics\nQ₁, Q₂', 'target uses min(Q₁,Q₂)\n→ kills the systematic\noverestimation of DDPG'),
        (0.34, 'Delayed policy\nupdates', 'actor + targets updated\nevery d = 2 critic steps\n→ critic leads, actor follows'),
        (0.67, 'Target policy\nsmoothing', "a' = clip(μ'(s') + ε)\nε ~ clip(N(0,σ),−c,c)\n→ no sharp-peak exploits"),
    ]
    for x, title, body in items:
        _box(ax, x, 0.10, 0.32, 0.78, '', BG_A, ACCENT)
        ax.text(x + 0.16, 0.76, title, ha='center', va='center', fontsize=9.0,
                fontweight='bold', color=ACCENT, linespacing=1.4)
        ax.text(x + 0.16, 0.40, body, ha='center', va='center', fontsize=7.4,
                color=INK, linespacing=1.6)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, 'td3.png'), bbox_inches='tight')
    plt.close(fig)


def fig_results():
    """The measured results, as a single scoreboard."""
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    labels = ['Tier B\nA↔B gate\n(px, lower better)', 'Tier C follow\ndetection %',
              'Detector v2\nprecision %', 'Detector v2\nrecall %']
    vals = [8.0, 100.0, 99.6, 70.5]
    tol = [40.0, None, None, None]
    colsb = [GOOD, GOOD, GOOD, WARN]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=colsb, width=0.5, zorder=3)
    if tol[0]:
        ax.plot([-0.3, 0.3], [tol[0], tol[0]], color=WARN, lw=1.6, zorder=4)
        ax.text(0, tol[0] + 3, 'tolerance 40', ha='center', fontsize=6.8, color=WARN)
    for x, v in zip(xs, vals):
        ax.text(x, v + 2.5, f'{v:g}', ha='center', fontsize=8.0,
                fontweight='bold', color=INK, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylim(0, 115); ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=LINE, lw=0.5, zorder=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, 'results.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    for fn in (fig_tiers, fig_flight_graph, fig_observation, fig_depth_law,
               fig_timeline, fig_behaviour_fsm, fig_td3, fig_results):
        fn()
        print('wrote', fn.__name__)
    print('figures ->', OUT)


if __name__ == '__main__':
    main()
