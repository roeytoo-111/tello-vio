#!/usr/bin/env python3
"""Build the Tello VIO technical report PDF with ReportLab."""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable,
                                Image, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "Tello_VIO_Technical_Report.pdf")

INK = colors.HexColor("#14181f")
MUTED = colors.HexColor("#5b6472")
LINE = colors.HexColor("#d3dae3")
ACCENT = colors.HexColor("#1f6feb")
WARN = colors.HexColor("#b8520f")
GOOD = colors.HexColor("#1f8a4c")
CODEBG = colors.HexColor("#f6f8fa")
NOTEBG = colors.HexColor("#eef3fb")
WARNBG = colors.HexColor("#fdf3e9")
GOODBG = colors.HexColor("#edf7f0")

PW, PH = A4
MARGIN = 19 * mm

ss = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


BODY = S("body", fontName="Helvetica", fontSize=9.3, leading=13.6, textColor=INK,
         alignment=TA_JUSTIFY, spaceAfter=6)
BODY_T = S("bodyt", parent=BODY, spaceAfter=2)
LEAD = S("lead", parent=BODY, fontSize=10.4, leading=15.4, textColor=INK, spaceAfter=8)
H1 = S("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK,
       spaceBefore=2, spaceAfter=3, alignment=TA_LEFT)
H1SUB = S("h1sub", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED,
          spaceAfter=10)
H2 = S("h2", fontName="Helvetica-Bold", fontSize=11.6, leading=15, textColor=ACCENT,
       spaceBefore=12, spaceAfter=4)
H3 = S("h3", fontName="Helvetica-Bold", fontSize=9.9, leading=13, textColor=INK,
       spaceBefore=9, spaceAfter=3)
CODE = S("code", fontName="Courier", fontSize=7.7, leading=10.4, textColor=INK,
         alignment=TA_LEFT, spaceAfter=0, spaceBefore=0)
EQ = S("eq", fontName="Courier", fontSize=8.4, leading=12.4, textColor=INK,
       alignment=TA_LEFT, leftIndent=10, spaceBefore=3, spaceAfter=3)
CAP = S("cap", fontName="Helvetica-Oblique", fontSize=7.9, leading=11, textColor=MUTED,
        alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
TH = S("th", fontName="Helvetica-Bold", fontSize=7.9, leading=10.4, textColor=colors.white)
TD = S("td", fontName="Helvetica", fontSize=7.9, leading=10.4, textColor=INK)
TDC = S("tdc", parent=TD, fontName="Courier", fontSize=7.2)
TDB = S("tdb", parent=TD, fontName="Helvetica-Bold")
BUL = S("bul", parent=BODY, spaceAfter=3, alignment=TA_LEFT)
COVER_T = S("ct", fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=INK,
            alignment=TA_LEFT)
COVER_S = S("cs", fontName="Helvetica", fontSize=12.6, leading=18, textColor=MUTED,
            alignment=TA_LEFT)
NOTE = S("note", parent=BODY, fontSize=8.9, leading=12.8, spaceAfter=0, spaceBefore=0)
NOTE_H = S("noteh", parent=NOTE, fontName="Helvetica-Bold", spaceAfter=3)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def para(t, style=BODY):
    return Paragraph(t, style)


def bullets(items, style=BUL, bullet="•"):
    return [ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=12, value=bullet) for i in items],
        bulletType="bullet", start=bullet, leftIndent=13, bulletFontSize=7,
        spaceBefore=1, spaceAfter=6,
    )]


def code(text, bg=CODEBG):
    lines = [Paragraph(l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                       .replace(" ", "&nbsp;") or "&nbsp;", CODE)
             for l in text.strip("\n").split("\n")]
    t = Table([[l] for l in lines], colWidths=[PW - 2 * MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 8)]


def eq(text):
    lines = [Paragraph(l.replace(" ", "&nbsp;") or "&nbsp;", EQ)
             for l in text.strip("\n").split("\n")]
    t = Table([[l] for l in lines], colWidths=[PW - 2 * MARGIN])
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4),
        ("LINEBEFORE", (0, 0), (0, -1), 1.6, colors.HexColor("#c9d6ea")),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 7)]


def callout(title, body, kind="note"):
    bg, ec = {"note": (NOTEBG, ACCENT), "warn": (WARNBG, WARN),
              "good": (GOODBG, GOOD)}[kind]
    inner = [Paragraph(title, ParagraphStyle("ch", parent=NOTE_H, textColor=ec))]
    for b in (body if isinstance(body, list) else [body]):
        inner.append(Paragraph(b, NOTE))
    t = Table([[inner]], colWidths=[PW - 2 * MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, ec),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 9)]


def table(rows, widths, header=True, styles=None, align=None, fs=7.9):
    styles = styles or {}
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, cell in enumerate(row):
            if isinstance(cell, Paragraph) or isinstance(cell, Flowable):
                out.append(cell)
            else:
                st = TH if (header and r == 0) else styles.get(c, TD)
                out.append(Paragraph(str(cell), st))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, LINE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), INK),
                 ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white)]
        cmds += [("ROWBACKGROUNDS", (0, 1), (-1, -1),
                  [colors.white, colors.HexColor("#fafbfd")])]
    else:
        cmds += [("ROWBACKGROUNDS", (0, 0), (-1, -1),
                  [colors.white, colors.HexColor("#fafbfd")])]
    for a in (align or []):
        cmds.append(a)
    t.setStyle(TableStyle(cmds))
    return [Spacer(1, 3), t, Spacer(1, 10)]


def figure(name, caption, width=None):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = width or (PW - 2 * MARGIN)
    h = w * ih / iw
    max_h = PH - 2 * MARGIN - 60
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return [Spacer(1, 5), img, Paragraph(caption, CAP)]


def h1(n, title, sub=""):
    out = [Spacer(1, 2), Paragraph(f"{n}&nbsp;&nbsp;{title}", H1)]
    if sub:
        out.append(Paragraph(sub, H1SUB))
    out.append(HRFlowable(width="100%", thickness=1.4, color=INK,
                          spaceBefore=1, spaceAfter=11))
    return out


# --------------------------------------------------------------------------- #
# page furniture
# --------------------------------------------------------------------------- #

def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#0f1620"))
    canvas.rect(0, PH - 78 * mm, PW, 78 * mm, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PH - 78 * mm, PW, 2.2 * mm, stroke=0, fill=1)
    canvas.restoreState()


def body_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PH - MARGIN + 7 * mm, PW - MARGIN, PH - MARGIN + 7 * mm)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PH - MARGIN + 9.5 * mm,
                      "Tello VIO  ·  Visual-Inertial Odometry on a DJI Tello  ·  ROS 2 Humble")
    canvas.line(MARGIN, MARGIN - 5 * mm, PW - MARGIN, MARGIN - 5 * mm)
    canvas.drawRightString(PW - MARGIN, MARGIN - 9 * mm, str(canvas.getPageNumber()))
    canvas.drawString(MARGIN, MARGIN - 9 * mm, "Technical Report")
    canvas.restoreState()


def build_cover():
    st = []
    st.append(Spacer(1, 26 * mm))
    st.append(Paragraph(
        '<font color="#ffffff">Visual-Inertial Odometry<br/>on a DJI Tello</font>',
        ParagraphStyle("t", parent=COVER_T, fontSize=29, leading=35)))
    st.append(Spacer(1, 7 * mm))
    st.append(Paragraph(
        '<font color="#9fb3cd">Algorithms, ROS 2 architecture, calibration and '
        'measured results<br/>for metric pose estimation without GPS</font>',
        ParagraphStyle("s", parent=COVER_S, fontSize=12, leading=17)))
    st.append(Spacer(1, 28 * mm))

    st.append(Paragraph(
        "A stock Tello gives you a soft 720p stream 250&nbsp;ms late, fused attitude "
        "in whole degrees at 10&nbsp;Hz, and <b>no gyroscope at all</b>. Textbook "
        "tightly-coupled VIO assumes none of that is true. This report is about what "
        "you build instead, why, and how well it works.", LEAD))
    st.append(Spacer(1, 6 * mm))

    rows = [
        ["Absolute trajectory error, 60&nbsp;s / 40&nbsp;m simulated flight", "0.57 m RMS"],
        ["Final drift as a fraction of path length", "1.4 %"],
        ["Metric scale accuracy (no GPS, no motion capture)", "1.034"],
        ["Roll / pitch RMS error", "1.8°"],
        ["Visual front-end cost", "~3 ms / frame"],
        ["Tests, all passing without a drone", "116"],
        ["Confirmed defects found and fixed in the existing code", "30"],
    ]
    st += table([[Paragraph(a, TD), Paragraph(f"<b>{b}</b>", TD)] for a, b in rows],
                widths=[118 * mm, 46 * mm], header=False)
    st.append(Spacer(1, 10 * mm))
    st.append(Paragraph(
        '<font color="#5b6472" size="8.5">Repository: <font face="Courier">'
        'tello-vio</font> &nbsp;·&nbsp; ROS 2 Humble / Ubuntu 22.04 &nbsp;·&nbsp; '
        'packages <font face="Courier">tello</font>, '
        '<font face="Courier">tello_msg</font>, '
        '<font face="Courier">tello_control</font>, '
        '<font face="Courier">tello_vio</font>, '
        '<font face="Courier">orbslam2</font></font>', BODY))
    st.append(PageBreak())
    return st


def build_toc():
    st = h1("", "Contents")
    entries = [
        ("1", "Executive summary", "What was found, what was built, what it achieves"),
        ("2", "The platform you actually have", "Why a Tello is not a normal VIO rig"),
        ("3", "Code review findings", "30 confirmed defects, and the ones that mattered"),
        ("4", "Mathematical foundations", "Manifolds, error states, and why the convention is load-bearing"),
        ("5", "IMU preintegration", "Compressing a burst of inertial data into one constraint"),
        ("6", "The visual front-end", "Tracking, model selection, degeneracy, uncertainty"),
        ("7", "Fusion I — the error-state Kalman filter", "The real-time path, with stochastic cloning"),
        ("8", "Fusion II — the fixed-lag smoother", "Factor graphs and why marginalisation is not optional"),
        ("9", "Calibration", "Three quantities you must measure yourself"),
        ("10", "ROS 2 concepts, as used here", "Executors, QoS, TF, and the decisions they force"),
        ("11", "Results", "Measured performance, and what the camera is actually worth"),
        ("12", "Bring-up procedure", "The order to do things in"),
        ("13", "Limitations and next steps", "What this does not do"),
        ("A", "Reference", "Files, parameters, topics"),
    ]
    rows = []
    for n, t, s in entries:
        rows.append([Paragraph(f'<font color="#1f6feb"><b>{n}</b></font>', TD),
                     Paragraph(f"<b>{t}</b><br/><font color='#5b6472' size='7.4'>{s}</font>", TD)])
    st += table(rows, widths=[12 * mm, 152 * mm], header=False)
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 1. Executive summary
# --------------------------------------------------------------------------- #

def sec1():
    st = h1("1", "Executive summary",
            "What was found, what was built, and what it achieves")

    st.append(para(
        "This report covers two pieces of work on the <font face='Courier'>tello-vio</font> "
        "repository: an adversarial review of the existing ROS 2 code, and the design and "
        "implementation of a visual-inertial odometry system on top of it.", LEAD))

    st.append(Paragraph("The review", H2))
    st.append(para(
        "Thirty defects were confirmed against the real code, real ROS 2 semantics and real "
        "DJI SDK behaviour. Several were reachable in flight. The most serious:"))
    st += bullets([
        "<b>Blocking SDK calls ran inside ROS callbacks.</b> <font face='Courier'>land()</font> "
        "retries for up to 21&nbsp;s. On a single-threaded executor that froze "
        "<font face='Courier'>/emergency</font> <i>and</i> the RC dead-man for the entire "
        "window, so a drone commanded forward kept flying forward with no way to stop it.",
        "<b>The IMU topic published a fabricated zero angular velocity with a zero covariance</b> "
        "— which in ROS means “measured, and exactly known”. The Tello has no gyroscope; "
        "any consumer would have believed and integrated that zero.",
        "<b>Accelerations were published in the drone's FRD frame with a 2&nbsp;% scale error</b>, "
        "so a level, stationary drone reported <font face='Courier'>z ≈ −10 m/s²</font> instead "
        "of <font face='Courier'>+9.81</font>.",
        "<b><font face='Courier'>camera_info</font> was not rescaled when the image was "
        "downscaled.</b> At <font face='Courier'>video_scale=0.5</font> it claimed twice the "
        "true focal length — an error SLAM absorbs as scale and never recovers from.",
        "<b>The ORB-SLAM2 wrapper computed a camera pose every frame and threw it away.</b> "
        "No TF, no odometry. A SLAM node that publishes no pose cannot localise anything.",
        "<b><font face='Courier'>ORB_SLAM2_ENABLE=OFF</font> forced the build instead of "
        "skipping it</b>, and <font face='Courier'>scripts/build.sh</font> combined a relative "
        "<font face='Courier'>cd</font> with a recursive delete.",
    ])
    st.append(para(
        "All thirty are fixed. Section&nbsp;3 gives the full list with failure scenarios."))

    st.append(Paragraph("What was built", H2))
    st.append(para(
        "A new <font face='Courier'>tello_vio</font> package, layered so that everything "
        "except the ROS nodes is pure NumPy/OpenCV and therefore testable and profilable "
        "without ROS installed:"))
    st += bullets([
        "<b>Visual front-end</b> — grid-bucketed Shi-Tomasi detection, pyramidal KLT tracking "
        "with forward-backward validation, essential/homography model selection, and a "
        "rotation-compensated keyframe trigger. ~3&nbsp;ms per frame.",
        "<b>Error-state Kalman filter</b> — 22 states including a stochastic clone, fusing "
        "attitude, the drone's own optical-flow velocity, barometer, ToF, zero-velocity "
        "updates and relative visual measurements. This is the real-time path.",
        "<b>Fixed-lag factor-graph smoother</b> — IMU preintegration, visual and unary factors, "
        "with proper Schur-complement marginalisation, verified to reproduce the full-batch "
        "solution.",
        "<b>Calibration</b> — Allan-variance IMU noise identification, rotation-only hand-eye "
        "for the camera-IMU extrinsic, and cross-correlation time-offset estimation, each "
        "reporting an observability score rather than just a residual.",
        "<b>ORB-SLAM2 backend</b> — now publishes pose, TF, path and tracking state, with a "
        "Sim(3) alignment node that turns loop closures into a <font face='Courier'>map→odom</font> "
        "correction so the odometry never jumps.",
    ])

    st += callout(
        "The one design decision everything follows from",
        "A monocular camera cannot observe scale. On this airframe the metric anchor is the "
        "Tello's own downward optical-flow velocity, supported by the barometer and ToF. So the "
        "visual measurement is deliberately restricted to <b>rotation and translation "
        "direction</b> — a 2-degree-of-freedom bearing on the unit sphere — and never a "
        "magnitude. Section&nbsp;7 derives it; Section&nbsp;11 measures what happens when you "
        "take the flow sensor away.", "note")

    st.append(Paragraph("What it achieves", H2))
    st += figure("results.png",
                 "Figure 1 — 60&nbsp;s simulated flight with the real drone's limitations "
                 "modelled: 10&nbsp;Hz jittery telemetry, whole-degree attitude, milli-g "
                 "accelerometer quantisation, integer-decimetre velocity, a drifting barometer "
                 "and a free-running yaw. Left: trajectory. Centre: position error. Right: "
                 "covariance calibration.")
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 2. Platform
# --------------------------------------------------------------------------- #

def sec2():
    st = h1("2", "The platform you actually have",
            "Why a Tello is not a normal visual-inertial rig")

    st.append(para(
        "Every design decision in this repository traces back to a handful of hardware facts. "
        "They are unglamorous and they are not negotiable, so they come first.", LEAD))

    st.append(Paragraph("2.1  What the drone sends", H2))
    st.append(para(
        "The Tello exposes three UDP endpoints on its own WiFi access point. There is no ROS "
        "on the aircraft and no compute available to us on it: <b>everything in this report "
        "runs on the ground station</b>. “Computationally efficient” therefore means <i>fits "
        "inside a 33&nbsp;ms frame budget on one laptop core</i>, not <i>fits on the drone</i>."))
    st += table([
        ["Port", "Contents", "Rate", "What it costs you"],
        ["11111", "H.264 video, 960×720", "~30 fps nominal", "150–350 ms latency, jittery, "
         "no hardware timestamp, visible compression artefacts"],
        ["8890", "State broadcast: attitude, accelerations, velocities, "
         "barometer, ToF, height, battery, temperatures", "~10 Hz, aperiodic",
         "whole-degree attitude; milli-g accelerations; velocity in integer units; "
         "<b>no angular rate</b>"],
        ["8889", "Commands and responses", "on demand",
         "every command is a blocking round trip with a 7&nbsp;s timeout and 3 retries"],
    ], widths=[14 * mm, 56 * mm, 24 * mm, 70 * mm])

    st.append(Paragraph("2.2  There is no gyroscope", H2))
    st.append(para(
        "This is the fact that reshapes the whole project. The state packet carries "
        "<i>fused attitude</i> — <font face='Courier'>pitch</font>, "
        "<font face='Courier'>roll</font>, <font face='Courier'>yaw</font> as whole degrees "
        "— not angular rate. Every textbook tightly-coupled VIO system (VINS-Mono, OpenVINS, "
        "ORB-SLAM3-VI, MSCKF) preintegrates 100–200&nbsp;Hz gyroscope and accelerometer data "
        "between camera frames. <b>That input does not exist on this aircraft.</b>"))
    st.append(para(
        "The only route to an angular rate is differentiating orientation, and doing it "
        "correctly means doing it on the manifold rather than subtracting Euler angles "
        "(which breaks at wrap-around and is simply wrong whenever roll and yaw both change):"))
    st += eq("omega  =  Log( q_{k-1}^-1  (x)  q_k ) / dt")
    st.append(para(
        "The resulting signal is real but weak. A uniform quantiser of width <i>q</i> has "
        "standard deviation <i>q</i>/√12; differentiated over <i>dt</i> that becomes "
        "<i>q</i>/(<i>dt</i>·√6), which at one degree and 10&nbsp;Hz is roughly "
        "<b>0.07&nbsp;rad/s ≈ 4°/s</b> of noise on a signal whose useful range is a few rad/s. "
        "It is good enough to predict where features will move and to gate outliers. It is "
        "not good enough to integrate."))

    st += callout(
        "Consequence: loosely coupled, with an explicit scale source",
        "Tightly-coupled VIO is off the table, so this system is loosely coupled and gets its "
        "metric scale from sensors that are already metric. That is not a compromise forced by "
        "laziness — given this sensor suite it is the <i>correct</i> architecture, and "
        "Section&nbsp;11 measures the cost of pretending otherwise. If you bolt a real IMU to "
        "the airframe, the preintegration in Section&nbsp;5 is rate-agnostic and runs unchanged.",
        "warn")

    st.append(Paragraph("2.3  Units and frames are not ROS units and frames", H2))
    st.append(para(
        "The SDK reports in an aerospace FRD/NED convention: body axes x&nbsp;forward, "
        "y&nbsp;right, z&nbsp;down, with yaw increasing clockwise seen from above. ROS mandates "
        "FLU/ENU (REP-103). The body-frame conversion is a 180° rotation about x, "
        "<font face='Courier'>diag(1,&nbsp;−1,&nbsp;−1)</font>; for the attitude it negates "
        "pitch and yaw and leaves roll alone."))
    st += table([
        ["Field", "Raw meaning", "SI conversion", "Trap"],
        ["agx/agy/agz", "specific force, milli-g, FRD",
         "× 9.80665/1000, then FRD→FLU",
         "dividing by 100 gives 10.0 at rest — <i>looks</i> like 9.81 and is 2 % high"],
        ["vgx/vgy/vgz", "velocity from the onboard optical flow",
         "× 0.1 (decimetres/s), FRD→FLU",
         "the SDK documents no unit; this is a <b>parameter</b>, not a constant"],
        ["baro", "pressure altitude in metres",
         "relative use only, with an estimated bias",
         "referenced to sea level, not to takeoff; it drifts with weather"],
        ["tof", "downward range, centimetres", "÷ 100, gated to 0.1–1.2 m",
         "reads garbage over table edges and stairs"],
        ["pitch/roll/yaw", "degrees, NED-referenced", "roll, −pitch, −yaw",
         "yaw has no absolute reference — no usable magnetometer"],
    ], widths=[22 * mm, 42 * mm, 44 * mm, 56 * mm])
    st.append(para(
        "A level Tello at rest reports <font face='Courier'>agz ≈ −1000</font>. An "
        "accelerometer measures specific force <i>f</i>&nbsp;=&nbsp;<i>a</i>&nbsp;−&nbsp;<i>g</i>; "
        "at rest in an FRD frame that is <font face='Courier'>[0, 0, −9.81]</font>, which is "
        "exactly the observed sign and confirms the frame reading. After conversion the same "
        "drone reports <font face='Courier'>+9.81</font> on z, which is what REP-145 requires."))

    st.append(Paragraph("2.4  Timing is the quiet killer", H2))
    st.append(para(
        "Video and telemetry arrive on different sockets with different, unequal latencies, "
        "and neither is hardware-timestamped. At 1&nbsp;rad/s of body rate, "
        "<b>50&nbsp;ms of unmodelled camera-IMU offset injects 50&nbsp;mrad — about 3° — of "
        "rotation error into every visual measurement</b>, which is an order of magnitude "
        "larger than the measurement noise itself."))
    st += figure("timeline.png",
                 "Figure 2 — The timing problem. Frames are stamped when they arrive, not when "
                 "they were captured. The estimator therefore buffers telemetry and advances "
                 "only to each frame's offset-corrected capture time, rather than fusing "
                 "measurements in arrival order.")
    st.append(para(
        "Two things follow. First, the offset <i>t<sub>d</sub></i> must be estimated "
        "(Section&nbsp;9.3) and subtracted from every image stamp. Second, the estimator must "
        "run on the <i>image</i> clock: telemetry is buffered and replayed up to each frame's "
        "corrected time. The published odometry is consequently ~250&nbsp;ms old but "
        "self-consistent; a separate topic forward-propagates to now for controllers, clearly "
        "labelled as a prediction and with an inflated covariance."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 3. Review
# --------------------------------------------------------------------------- #

DEFECTS = [
    ("CRIT", "tello/node.py", "Blocking SDK commands in subscriber callbacks",
     "land()/takeoff()/flip() block up to 21 s retrying. On a single-threaded executor "
     "/emergency and the RC dead-man are unavailable for that whole window, and the Tello "
     "latches the last rc setpoint — a drone told to fly forward keeps flying forward."),
    ("CRIT", "tello/node.py", "Three blocking query_* round trips inside the 2 Hz status timer",
     "With /status and /id subscribed the executor blocked ~300 ms out of every 500 ms; on a "
     "degraded link a single tick stalled the node for up to 30 s."),
    ("CRIT", "tello/node.py", "/imu published angular_velocity = 0 with zero covariance",
     "ROS semantics: all-zero covariance means 'measured and exactly known'. The Tello has no "
     "gyro. Any consumer would integrate a confident zero rate."),
    ("CRIT", "tello/node.py", "IMU acceleration published in the drone's FRD frame",
     "Gravity pointed the wrong way: z ≈ −10 m/s² at rest instead of +9.81, with a 2 % scale "
     "error on top."),
    ("CRIT", "tello/node.py", "Attitude published without the NED→ENU conversion",
     "Yaw and pitch ran backwards relative to REP-103."),
    ("CRIT", "orbslam2/...node.cpp", "Tcw computed every frame and discarded",
     "No TF, no Odometry, no PoseStamped. The SLAM node localised nothing."),
    ("HIGH", "tello/node.py", "self._resize_cache read but never initialised",
     "AttributeError killed the node for any video_scale not within 1e-6 of 0.5 or 0.25."),
    ("HIGH", "tello/node.py", "camera_info not rescaled with video_scale",
     "At 0.5 the published intrinsics claimed twice the true focal length; every derived "
     "bearing was wrong by a factor of two."),
    ("HIGH", "tello/node.py", "tello_ip and connect_timeout silently ignored",
     "djitellopy binds the host in __init__ and freezes RESPONSE_TIMEOUT as a default argument "
     "at import time, so assigning the class attributes did nothing."),
    ("HIGH", "tello/node.py", "Frames published as bgr8 but decoded as RGB",
     "djitellopy returns np.array(frame.to_image()) — RGB. Red and blue were swapped for every "
     "colour consumer."),
    ("HIGH", "tello/node.py", "No new-frame detection",
     "The same frame was republished with a fresh timestamp, fabricating motion-free evidence "
     "and hiding a dead decoder behind a healthy-looking 30 Hz topic."),
    ("HIGH", "tello/node.py", "IMU timer-driven off a polled dict, not packet-driven",
     "Timestamps were the timer's firing instant, displaced from the measurement by an "
     "unbounded 0–100 ms. A fixed offset can be calibrated out; a varying one cannot."),
    ("HIGH", "tello/node.py", "Odometry twist 10× too small, FRD axes, empty child_frame_id",
     "robot_localization's transform lookup fails on an empty frame id and drops the "
     "measurement silently."),
    ("HIGH", "tello/node.py", "TF z from absolute barometric altitude",
     "Published field elevation (~100 m) as height, on the same TF edge as a static publisher "
     "asserting zero — two owners, one edge."),
    ("HIGH", "tello/node.py", "Camera and IMU shared one frame_id",
     "The camera-IMU extrinsic was literally inexpressible anywhere in the system."),
    ("HIGH", "orbslam2/...node.cpp", "Map points published in optical axes into 'map'",
     "The map appeared on its side in RViz and was wrong for every consumer."),
    ("HIGH", "orbslam2/...node.cpp", "Whole map re-serialised into Markers every frame",
     "O(map size) work on the tracking thread at 30 Hz."),
    ("HIGH", "orbslam2/CMakeLists.txt", "ORB_SLAM2_ENABLE=OFF forced the build",
     "The find_package was inside the guard but add_executable was not, so the documented "
     "off switch caused a hard compile failure."),
    ("HIGH", "orbslam2/CMakeLists.txt", "config.yaml never installed",
     "ORB-SLAM2's System constructor calls exit(-1) on a missing settings file, before any "
     "logging exists. README and launch both pointed at paths that do not exist."),
    ("HIGH", "workspace/src/launch.py", "Driver remapped /image_raw to /camera",
     "Every consumer — tello_control, the SLAM wrapper, all ROS image tools — subscribes to "
     "/image_raw. The camera window stayed black."),
    ("HIGH", "scripts/build.sh", "Relative cd followed by rm -rf",
     "'cd ../workspace && rm -rf build install log' deleted three directories relative to "
     "wherever the caller happened to be."),
    ("HIGH", "scripts/install.sh", "Installed ROS 2 Foxy into a Humble repository",
     "Also used the deprecated apt-key mechanism."),
    ("MED", "orbslam2/...node.cpp", "annotated_frame stamped with wall-clock publish time",
     "Could not be time-aligned with its own source image."),
    ("MED", "FindORB_SLAM2.cmake", "Accepted an include layout the sources cannot compile against",
     "Configure succeeded and the compile then failed — a confusing place to find a search-path "
     "problem."),
    ("MED", "tello_control/main.cpp", "1 ms timer wrapping a blocking waitKey(15)",
     "Spun a core at ~66 Hz to render a 30 Hz stream, and re-created the window every tick."),
    ("MED", "tello_msg/TelloStatus.msg", "No Header",
     "No timestamp, so it could not be time-aligned or used with message_filters."),
    ("MED", "tello/package.xml", "Declared rosidl generators for a package with no messages", ""),
    ("MED", "orbslam2/package.xml", "Eigen, OpenCV and Boost find_package'd but not declared",
     "rosdep reported success and the build then failed."),
    ("MED", "orbslam2/CMakeLists.txt", "CMAKE_BUILD_TYPE never defaulted",
     "The wrapper compiled at -O0. Measured cost ~2.4 ms/frame — real, but far smaller than "
     "first assumed, since the heavy work is inside a Release-built libORB_SLAM2.so."),
    ("MED", "tello/node.py", "video_backend documented in the README but never declared",
     "Passing it would have raised ParameterNotDeclaredException. Removed from the docs rather "
     "than faked."),
]


def sec3():
    st = h1("3", "Code review findings",
            "Thirty confirmed defects, ranked by what they do in flight")

    st.append(para(
        "Findings were produced by five independent review passes — driver runtime, sensor "
        "physics, the SLAM wrapper, the build system, and VIO readiness — and each candidate "
        "was then handed to a separate adversarial verifier whose brief was to <i>refute</i> "
        "it against the real code, real ROS 2 semantics and real djitellopy behaviour. Ten "
        "claims were rejected that way, including some plausible ones: the "
        "<font face='Courier'>cb_control</font> axis mapping is non-standard but "
        "self-consistent end to end, and the <font face='Courier'>-O0</font> build costs "
        "~2.4&nbsp;ms/frame rather than the catastrophe first claimed. Only what survived "
        "verification is listed here.", LEAD))

    rows = [["", "Location", "Defect", "Effect"]]
    for sev, loc, title, eff in DEFECTS:
        col = {"CRIT": "#b3261e", "HIGH": "#b8520f", "MED": "#5b6472"}[sev]
        rows.append([
            Paragraph(f'<font color="{col}" size="6.6"><b>{sev}</b></font>', TD),
            Paragraph(f'<font face="Courier" size="6.7">{loc}</font>', TD),
            Paragraph(f"<b>{title}</b>", TD),
            Paragraph(f'<font size="7.3">{eff}</font>', TD),
        ])
    st += table(rows, widths=[11 * mm, 33 * mm, 55 * mm, 65 * mm])

    st += callout(
        "The pattern worth noticing",
        "Almost none of these produce an error message. A frozen executor looks like a laggy "
        "link; an unscaled <font face='Courier'>camera_info</font> looks like SLAM being bad at "
        "its job; a fabricated zero gyro reading looks like a drifting filter. Every one of them "
        "degrades gracefully into something you would misattribute to tuning. That is why the "
        "rewrite leans so heavily on making the invalid states <i>representable</i> — "
        "<font face='Courier'>covariance[0] = −1</font> for unmeasured quantities, "
        "<font face='Courier'>+inf</font> for out-of-range ToF, "
        "<font face='Courier'>None</font> rather than zero for a missing angular rate.",
        "warn")
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 4. Foundations
# --------------------------------------------------------------------------- #

def sec4():
    st = h1("4", "Mathematical foundations",
            "Manifolds, error states, and why the convention is load-bearing")

    st.append(Paragraph("4.1  Rotations do not live in a vector space", H2))
    st.append(para(
        "Position and velocity are vectors: adding a small correction to them is meaningful. "
        "Orientation is not. SO(3) is a curved three-dimensional manifold with no global "
        "three-parameter chart, which is why every naive parameterisation fails somewhere — "
        "Euler angles at gimbal lock, a four-vector quaternion by carrying a singular "
        "covariance and needing constant re-normalisation.", LEAD))
    st.append(para(
        "The standard resolution, and the one used throughout this codebase, is to keep the "
        "<i>nominal</i> state on the manifold where it is free to be large, and to keep a "
        "<i>small</i> error in the tangent space where a Gaussian is actually a reasonable "
        "model. The two are connected by the exponential map:"))
    st += eq(
        "Exp : R^3 -> SO(3)      Exp(phi) = I + sin|phi|/|phi| [phi]x\n"
        "                                     + (1-cos|phi|)/|phi|^2 [phi]x^2\n"
        "\n"
        "Log : SO(3) -> R^3      the inverse, on (-pi, pi]")
    st.append(para(
        "<font face='Courier'>tello_vio/lie.py</font> implements these with Taylor branches "
        "below 10<super>−4</super>&nbsp;rad — where the closed forms lose precision to "
        "cancellation in (1&nbsp;−&nbsp;cos&nbsp;θ)/θ² — and a quaternion-based branch near "
        "θ&nbsp;=&nbsp;π where sin&nbsp;θ&nbsp;→&nbsp;0 makes the trace formula blow up. Both "
        "branches are exercised by tests."))

    st.append(Paragraph("4.2  The convention, stated once", H2))
    st += callout(
        "Right-multiplicative, body-frame error",
        ["<font face='Courier'>R = R̂ · Exp(δθ)</font> &nbsp;&nbsp; and &nbsp;&nbsp; "
         "<font face='Courier'>q = q̂ ⊗ Exp<sub>q</sub>(δθ)</font>, with δθ in the body frame.",
         "This is the choice made by Forster's preintegration and by GTSAM. The alternative — "
         "a left, world-frame error — yields <i>different</i> Jacobians for every single "
         "measurement. Mixing the two is the classic silent bug: the filter stays numerically "
         "well-behaved, the covariance still shrinks, and the estimate walks steadily away from "
         "the truth."], "note")
    st.append(para(
        "Because the convention is load-bearing, it is declared in exactly one place "
        "(<font face='Courier'>lie.py</font>'s module docstring) and every derived Jacobian is "
        "checked against a finite difference taken through the <i>same</i> retraction. That "
        "discipline is not decorative: it caught two genuine sign errors in the filter during "
        "development — the attitude update's <font face='Courier'>H</font> and the visual "
        "bearing residual — either of which would have made the filter correct in the wrong "
        "direction."))

    st.append(Paragraph("4.3  The right Jacobian", H2))
    st.append(para(
        "The one piece of Lie-group machinery that appears everywhere below is the right "
        "Jacobian, defined by how the exponential map responds to a perturbation of its "
        "argument:"))
    st += eq(
        "Exp(phi + dphi)  ~=  Exp(phi) · Exp( Jr(phi) dphi )\n"
        "\n"
        "Jr(phi) = I - (1-cos t)/t^2 [phi]x + (t - sin t)/t^3 [phi]x^2 ,   t = |phi|")
    st.append(para(
        "It is what converts “a small change in a rotation vector” into “a small body-frame "
        "rotation”, and it appears in the covariance propagation of preintegration, in the "
        "bias Jacobians, and in every rotation residual's derivative. Where a residual is "
        "itself a rotation vector, its inverse <font face='Courier'>Jr<super>−1</super></font> "
        "appears instead."))

    st.append(Paragraph("4.4  Frames used throughout", H2))
    st += table([
        ["Frame", "Convention", "Meaning"],
        ["<font face='Courier'>map</font>", "ENU, z up",
         "Globally consistent, loop-closed. May jump."],
        ["<font face='Courier'>odom</font>", "ENU, z up",
         "VIO origin, fixed at initialisation. Continuous; drifts slowly."],
        ["<font face='Courier'>base_link</font>", "FLU: x forward, y left, z up",
         "The body / IMU frame. All inertial quantities live here."],
        ["<font face='Courier'>camera_optical</font>", "z out of the lens, x right, y down",
         "REP-103 optical frame. All image geometry lives here."],
    ], widths=[36 * mm, 46 * mm, 82 * mm])
    st.append(para(
        "The camera-to-body rotation is the fixed <font face='Courier'>R<sub>z</sub>(−90°) "
        "R<sub>x</sub>(−90°)</font>, which maps camera&nbsp;+z (forward) to body&nbsp;+x, "
        "camera&nbsp;+x (right) to body&nbsp;−y, and camera&nbsp;+y (down) to body&nbsp;−z. "
        "The actual mount deviates from nominal by a few degrees of manufacturing tolerance, "
        "and a few degrees of extrinsic error looks exactly like scale drift in the output — "
        "hence Section&nbsp;9.2."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 5. Preintegration
# --------------------------------------------------------------------------- #

def sec5():
    st = h1("5", "IMU preintegration",
            "Compressing a burst of inertial data into a single constraint")

    st.append(para(
        "Preintegration answers one question: given a burst of IMU samples between two "
        "keyframes, what single constraint do they impose on the two keyframe states? Naively "
        "you would re-integrate the whole burst every time an optimiser touches either pose — "
        "ruinously expensive inside a Gauss-Newton loop, since the optimiser moves those poses "
        "on every iteration.", LEAD))

    st.append(Paragraph("5.1  The trick", H2))
    st.append(para(
        "Preintegration instead compresses the burst <i>once</i>, into rotation, velocity and "
        "position increments expressed <b>in the frame of the first keyframe</b>. Because they "
        "are relative, they stay valid as the optimiser moves both poses around:"))
    st += eq(
        "dR_ij = prod_k Exp( (w_k - b_g) dt )\n"
        "dv_ij = sum_k  dR_ik (a_k - b_a) dt\n"
        "dp_ij = sum_k  [ dv_ik dt + 1/2 dR_ik (a_k - b_a) dt^2 ]")
    st.append(para("These relate the two states through gravity and elapsed time:"))
    st += eq(
        "R_j = R_i dR_ij\n"
        "v_j = v_i + g dt_ij + R_i dv_ij\n"
        "p_j = p_i + v_i dt_ij + 1/2 g dt_ij^2 + R_i dp_ij")
    st.append(para(
        "Here <i>a<sub>k</sub></i> is <b>specific force</b>, not acceleration: a level, "
        "stationary IMU reads <font face='Courier'>[0, 0, +9.81]</font>, not zero. Conflating "
        "the two is a 1&nbsp;g error, which is not subtle but is remarkably easy to introduce "
        "when converting frames."))

    st.append(Paragraph("5.2  The one thing it cannot absorb: bias", H2))
    st.append(para(
        "The increments above are computed at a fixed bias linearisation point. When the "
        "optimiser changes its bias estimate, re-integrating from scratch would defeat the "
        "purpose, so analytic bias Jacobians are carried alongside and used for a first-order "
        "correction:"))
    st += eq(
        "dR~ = dR · Exp( ddR/db_g · db_g )\n"
        "dv~ = dv + ddv/db_a · db_a + ddv/db_g · db_g\n"
        "dp~ = dp + ddp/db_a · db_a + ddp/db_g · db_g")
    st += callout(
        "The ordering trap",
        "The position Jacobians must be updated using the <i>previous</i> step's velocity "
        "Jacobians, before those are themselves updated. Get the order backwards and the "
        "correction is subtly wrong in a way that only shows up as slow convergence. The test "
        "suite pins this down by re-preintegrating from scratch at a perturbed bias and "
        "demanding the first-order prediction match to 10<super>−8</super>.", "warn")

    st.append(Paragraph("5.3  Covariance, and why it is checked against Monte Carlo", H2))
    st.append(para(
        "The 9×9 covariance propagates as "
        "<font face='Courier'>Σ ← A Σ A<super>T</super> + B Q<sub>d</sub> B<super>T</super></font>. "
        "The discretisation constant is easy to get wrong: a continuous density σ acting over "
        "<i>dt</i> has discrete variance σ²/<i>dt</i> once the <i>dt</i> inside "
        "<font face='Courier'>B</font> is accounted for. A missing or extra <i>dt</i> changes "
        "the answer by an order of magnitude — and is <b>completely invisible in the mean</b>. "
        "The estimate looks fine; only the consistency is destroyed."))
    st.append(para(
        "So the implementation is validated by sampling the same noise through the same model "
        "4000 times and comparing per-axis standard deviations to the analytic ones. They agree "
        "within 20&nbsp;%, which at 4000 samples is a tight test."))

    st += callout(
        "On a Tello specifically",
        "The angular rate fed into this is the 10&nbsp;Hz attitude-difference surrogate from "
        "Section&nbsp;2.2, not a gyroscope. The mathematics is rate-agnostic and remains exactly "
        "correct — but the <i>information</i> the constraint carries is proportionally weaker, "
        "which is why the estimator leans on body velocity and height for anything "
        "long-horizon. The same code runs unchanged on a real IMU.", "note")

    st.append(Paragraph("5.4  Degenerate edges", H2))
    st.append(para(
        "Two cases produce an information matrix that asserts near-infinite confidence, which "
        "would weld two keyframes rigidly together and dominate every other factor in the graph:"))
    st += bullets([
        "<b>Empty increment.</b> The covariance is exactly zero, so its inverse is unbounded. "
        "Returns zero information — no constraint — rather than infinite.",
        "<b>Single sample.</b> Less obvious: the covariance is only rank 6, not 9. The noise "
        "mapping makes the position rows a fixed multiple of the velocity rows, so three "
        "directions carry no uncertainty at all and invert to infinity even though the matrix "
        "looks harmlessly non-zero. Handled by an eigenvalue cap, not just a diagonal floor — "
        "a floor merely converts “no information” into 1/floor of information.",
    ])
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 6. Front-end
# --------------------------------------------------------------------------- #

def sec6():
    st = h1("6", "The visual front-end",
            "Tracking, model selection, degeneracy, and honest uncertainty")

    st.append(para(
        "The front-end's job is to turn a soft, compressed, 250&nbsp;ms-late video stream into "
        "one relative-pose measurement per keyframe, inside a 33&nbsp;ms budget. It reports "
        "geometry only: rotation and translation <i>direction</i>, never a magnitude.", LEAD))
    st += figure("frontend.png", "Figure 3 — The per-frame pipeline.")

    st.append(Paragraph("6.1  Why KLT and not ORB every frame", H2))
    st.append(para(
        "ORB-SLAM2 detects and describes ~1000 ORB features on <i>every</i> frame and matches "
        "them by Hamming distance. That is the right architecture when the same features must "
        "also serve relocalisation and loop closure — which is exactly why ORB-SLAM2 remains "
        "the map backend in this project. It is the wrong architecture for a low-latency "
        "odometry front-end, where detect-and-describe dominates the per-frame cost."))
    st += table([
        ["", "ORB detect+describe+match", "Sparse KLT"],
        ["Per-frame cost, 480×360, ~150 features", "~25–30 ms", "<b>~3 ms</b>"],
        ["Sub-pixel accuracy between adjacent frames", "quantised binary descriptor",
         "direct photometric alignment"],
        ["Survives total tracking loss", "yes — descriptors allow relocalisation",
         "<b>no</b> — this is the trade"],
        ["Robust to large baselines", "yes", "no — needs small inter-frame motion"],
    ], widths=[62 * mm, 52 * mm, 50 * mm])
    st.append(para(
        "The weakness is real and is covered by design rather than denied: recovering from a "
        "total loss is ORB-SLAM2's job, running in parallel on the same stream."))

    st.append(Paragraph("6.2  Three details that decide whether it works", H2))
    st += bullets([
        "<b>Forward-backward validation.</b> Track a point forward, then track the result back; "
        "if it does not land where it started, the track is wrong. One extra KLT call removes "
        "most of the drift-inducing bad matches that RANSAC would otherwise have to absorb.",
        "<b>Grid bucketing on detection.</b> <font face='Courier'>goodFeaturesToTrack</font> "
        "piles features onto the highest-contrast corner of the image — typically one window or "
        "one poster. Features clustered in a small image region give a nearly degenerate "
        "geometry and a translation direction with enormous variance. A per-cell quota spreads "
        "them out and conditions the whole estimate.",
        "<b>Never undistort the image.</b> Undistorting a full frame every cycle is pure waste. "
        "Distortion is a smooth warp that KLT handles happily; the correction is applied only to "
        "the ~150 <i>points</i> that reach the geometry stage.",
    ])

    st.append(Paragraph("6.3  Two-view geometry and its degeneracies", H2))
    st.append(para(
        "Given matched points, the relative pose comes from the essential matrix. Three things "
        "break this, and indoor flying hits all three routinely:"))
    st += figure("degeneracy.png",
                 "Figure 4 — The three configurations that break monocular two-view geometry.")
    st += bullets([
        "<b>Planar scenes.</b> Floors, walls and ceilings are the normal indoor case, and they "
        "are exactly where <font face='Courier'>E</font> is not uniquely determined and "
        "<font face='Courier'>recoverPose</font> returns confident nonsense. The front-end "
        "scores a homography against the fundamental matrix (Torr's symmetric-transfer approach, "
        "as used by ORB-SLAM) and switches models when the scene is plane-dominated.",
        "<b>Low parallax.</b> With little baseline the translation direction is unobservable — "
        "<font face='Courier'>t</font> is essentially a random unit vector. Parallax is measured "
        "<i>after de-rotating the bearings</i>, which is the only measure that separates “the "
        "drone rotated” from “the drone moved”; the raw bearing change conflates them completely.",
        "<b>Pure rotation.</b> The translation is genuinely zero. The front-end still returns "
        "the rotation — which is well conditioned and valuable — and flags the translation "
        "unreliable, so the filter applies a rotation-only update. Failing the whole "
        "measurement here would discard a good attitude constraint every time the pilot yaws "
        "on the spot, which on a Tello is most of the flight.",
    ])

    st += callout(
        "The keyframe trigger must measure parallax, not pixel motion",
        ["A rotation induces pixel motion identical in character to translation, so a "
         "raw-displacement trigger fires on <i>turning</i>, not on <i>moving</i> — the worst "
         "possible moment to make a keyframe, because the baseline is near zero.",
         "Removing the prior rotation costs one 3×3 matrix and a projective divide over ~150 "
         "points. Measured effect on the rendered fly-through: median parallax at the keyframe "
         "rose from 0.6° to 1.5°, model selection stopped drifting to the homography, and "
         "rotation error fell from 2–4° to 0.2–1.7°.", "note"][:2] +
        ["The prior comes free from the drone's attitude stream — the one piece of inertial "
         "data a Tello provides in good shape."], "good")

    st.append(Paragraph("6.4  Honest per-measurement uncertainty", H2))
    st.append(para(
        "A fixed <font face='Courier'>R</font> matrix is wrong here. A 170-inlier essential "
        "solve at 3° of parallax and a 30-inlier homography at 0.3° are not the same "
        "measurement, and giving both the same confidence lets the bad ones dominate. Each "
        "measurement therefore reports its own σ:"))
    st += eq(
        "sigma_rot = 0.033 rad · sqrt(60/N) · (3.5 if homography else 1)\n"
        "sigma_dir = 0.128 rad · sqrt(60/N) · max(1, 3 deg / parallax_deg)")
    st.append(para(
        "The <font face='Courier'>1/√N</font> is averaging over correspondences; the "
        "<font face='Courier'>1/parallax</font> is the geometry of a long thin triangle, which "
        "degrades fast as the baseline shrinks. <b>The constants are fitted, not invented</b>: "
        "regressing observed errors from 64 keyframes across 8 rendered sequences onto that "
        "shape gives 1.87° and 7.3°, at which the normalised errors have a 90th percentile of "
        "1.4–1.7&nbsp;σ and only ~2&nbsp;% beyond 3&nbsp;σ. Those figures come from synthetic "
        "imagery with 0.3&nbsp;px feature noise; a real Tello feed is worse, so they are a "
        "floor and both are node parameters."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 7. ESKF
# --------------------------------------------------------------------------- #

def sec7():
    st = h1("7", "Fusion I — the error-state Kalman filter",
            "The real-time path, and where metric scale comes from")

    st.append(para(
        "The ESKF is what runs in flight. It is cheap, it is causal, and it produces a pose at "
        "every telemetry epoch. Its cost, relative to the smoother in Section&nbsp;8, is that "
        "it linearises once at the current estimate and never revisits that choice.", LEAD))

    st.append(Paragraph("7.1  State layout", H2))
    st += table([
        ["Index", "Error state", "Nominal quantity", "Why it is there"],
        ["0:3", "δp", "position of body in world (m)", ""],
        ["3:6", "δv", "velocity of body in world (m/s)", ""],
        ["6:9", "δθ", "attitude q_WB", "body-frame, right-multiplicative"],
        ["9:12", "δb<sub>a</sub>", "accelerometer bias (m/s²)",
         "observable through gravity and ZUPT"],
        ["12:15", "δb<sub>ω</sub>", "surrogate-gyro bias (rad/s)",
         "observable through ZARU"],
        ["15", "δb<sub>p</sub>", "barometer bias (m)",
         "the barometer is sea-level referenced and drifts"],
        ["16:19", "δp<sub>c</sub>", "<b>cloned</b> position at the last keyframe",
         "stochastic cloning"],
        ["19:22", "δθ<sub>c</sub>", "<b>cloned</b> attitude at the last keyframe",
         "stochastic cloning"],
    ], widths=[15 * mm, 20 * mm, 60 * mm, 69 * mm])
    st.append(para(
        "Twenty-two states. A 22×22 update is a few microseconds of NumPy, so the filter is "
        "nowhere near the compute budget — the front-end is."))

    st.append(Paragraph("7.2  Why there is a clone", H2))
    st.append(para(
        "Monocular vision naturally produces <i>relative</i> measurements: “between keyframe A "
        "and now, the camera rotated by this much and moved in this direction”. Feeding a "
        "relative measurement to a filter that only knows the current state is statistically "
        "wrong — the measurement is correlated with the past state error, and ignoring that "
        "correlation makes the filter over-confident and eventually divergent."))
    st.append(para(
        "<b>Stochastic cloning</b> (Roumeliotis &amp; Burdick, 2002) fixes it by carrying a copy "
        "of the state at the reference time inside the same covariance, so the "
        "cross-correlation is tracked explicitly. Cloning is not a copy of the mean: the clone "
        "inherits the full covariance <i>and</i> cross-covariance via "
        "<font face='Courier'>P ← J P J<super>T</super></font>."))
    st += callout(
        "A consequence that looks like a bug and is not",
        "Immediately after cloning, the live state and the clone are perfectly correlated, so "
        "the filter knows the relative pose <i>exactly</i> — it is zero — and correctly ignores "
        "any relative measurement. Only accumulated process noise makes the baseline uncertain "
        "and therefore correctable. A test asserts precisely this, because the first instinct on "
        "seeing “the visual update did nothing” is to go hunting for a sign error.", "note")

    st.append(Paragraph("7.3  The measurements, and what each one buys", H2))
    st += table([
        ["Update", "Model", "What it makes observable"],
        ["Attitude (AHRS)", "r = Log(q̂<super>−1</super> ⊗ q<sub>m</sub>)",
         "Roll and pitch absolutely (gravity-referenced). Yaw only as a smoothness hint — no "
         "magnetometer, so a large yaw variance is used, built in <i>world</i> axes and rotated "
         "into the body so it stays attached to world-z under tilt."],
        ["Body velocity (optical flow)", "h = R<super>T</super>v",
         "<b>Metric scale.</b> The <font face='Courier'>skew(R<super>T</super>v)</font> term in "
         "the Jacobian is what couples a velocity residual into an attitude correction — the "
         "mechanism by which sustained horizontal flight observes roll/pitch bias."],
        ["Barometer", "h = p<sub>z</sub> + b<sub>p</sub>",
         "Vertical scale, once the bias is anchored at initialisation. Without that anchor "
         "p<sub>z</sub> and b<sub>p</sub> are jointly unobservable and drift together forever "
         "while the residual sits happily at zero."],
        ["ToF", "h = (p<sub>z</sub> − z<sub>g</sub>) / R[2,2]",
         "Absolute height below ~1.2 m. The tilt division is not optional: skipping it injects "
         "6 % of height error at 20° of bank, an ordinary Tello manoeuvre."],
        ["ZUPT / ZARU", "v = 0, ω<sub>m</sub> = b<sub>ω</sub>",
         "The highest-information measurements available. While landed these are <i>exact</i>, "
         "turning bias from weakly to strongly observable in seconds."],
        ["Visual relative", "rotation (3) + bearing (2) against the clone",
         "Rotation and translation <i>direction</i>. Deliberately never magnitude."],
    ], widths=[34 * mm, 44 * mm, 86 * mm])

    st.append(Paragraph("7.4  The bearing residual", H2))
    st.append(para(
        "A bearing carries exactly two degrees of freedom, so its residual must live in the "
        "two-dimensional tangent plane of the unit sphere at the measured direction:"))
    st += eq(
        "u_pred = d(x) / |d(x)|          d(x) = R_BC^T ( w + (A - I) p_BC )\n"
        "r_dir  = B^T ( t_meas - u_pred )      B = 3x2 basis perpendicular to t_meas")
    st += callout(
        "Why not just use a 3-vector residual",
        "Because the radial direction carries no information. Treating the bearing as a "
        "3-vector residual — the common shortcut — silently adds a phantom constraint along "
        "that direction, which pulls on the translation <i>magnitude</i> and therefore biases "
        "the very scale that the flow sensor, barometer and ToF worked to establish. The 2-DoF "
        "residual is what keeps the split in Section&nbsp;1 honest.", "warn")

    st.append(Paragraph("7.5  Numerical hygiene", H2))
    st += bullets([
        "<b>Joseph-form update</b> — <font face='Courier'>(I−KH)P(I−KH)<super>T</super> + "
        "KRK<super>T</super></font>. It stays symmetric positive-definite under round-off, "
        "which the short form does not once you run for minutes at 30&nbsp;Hz with states "
        "scaled from metres to milliradians. One extra 22×22 product: free.",
        "<b>Chi-square innovation gating</b> at 3× the 95&nbsp;% quantile — loose enough not to "
        "reject honest measurements during fast manoeuvres, tight enough to catch a "
        "mis-associated visual match or a ToF reading off a table edge.",
        "<b>Covariance reset on injection</b> — after folding δθ into the nominal state, the "
        "<i>meaning</i> of the remaining attitude error changes. Two lines, a second-order "
        "effect, and it keeps the filter consistent during aggressive rotation.",
        "<b>Bias clamping</b> — biases are physically bounded, so clamping stops a temporarily "
        "unobservable direction from ratcheting off during a long stationary segment.",
        "<b>Missing data is missing</b> — when the surrogate cannot produce a rate (dropout, "
        "duplicate packet) the filter propagates with no rotation and inflates the covariance. "
        "Substituting zero would assert “the drone did not rotate”, a confident lie.",
    ])
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 8. Smoother
# --------------------------------------------------------------------------- #

def sec8():
    st = h1("8", "Fusion II — the fixed-lag smoother",
            "Factor graphs, and why marginalisation is not optional")

    st.append(para(
        "A filter linearises once, at the current estimate, and lives with it. An early "
        "linearisation error — made when the attitude or scale was still poorly known — is "
        "baked in forever. A smoother keeps the last <i>N</i> keyframes as free variables and "
        "re-solves the whole window as new data arrives, so an early mistake gets corrected "
        "once later measurements disambiguate it. That is why VINS-Mono, OKVIS and ORB-SLAM3 "
        "are all optimisation-based.", LEAD))
    st += figure("factor_graph.png",
                 "Figure 5 — The sliding window. Circles are keyframe states (R, p, v, bias); "
                 "squares are factors. When a keyframe leaves the window its information is "
                 "pushed into a dense prior on its neighbours rather than discarded.")

    st.append(Paragraph("8.1  The problem being solved", H2))
    st.append(para(
        "Maximum a posteriori over the window: minimise the sum of squared Mahalanobis "
        "residuals, on the manifold, by Levenberg-Marquardt with a Huber kernel on the visual "
        "factors."))
    st += eq(
        "min_x  sum_i  rho( || r_i(x) ||^2_{Lambda_i} )\n"
        "\n"
        "solve   (H + lambda diag H) delta = b      with H = sum J^T Lambda J\n"
        "then    x <- x [+] delta                   (retraction, not addition)")
    st.append(para(
        "Each keyframe contributes 15 error dimensions "
        "<font face='Courier'>[δθ, δp, δv, δb<sub>a</sub>, δb<sub>ω</sub>]</font>. At the "
        "default window of 10 that is a 150×150 dense Cholesky — a few hundred microseconds, "
        "comfortably inside a keyframe interval. Past roughly <i>N</i>&nbsp;=&nbsp;25 a sparse "
        "solver starts to pay for itself, and past that you want incremental factorisation "
        "(iSAM2) instead of re-solving from scratch."))
    st += callout(
        "A trap worth naming",
        "Preintegration orders its state blocks <font face='Courier'>[δθ, δv, δp]</font> "
        "(following Forster); the smoother orders them "
        "<font face='Courier'>[δθ, δp, δv, …]</font>. The IMU factor remaps between them "
        "explicitly rather than relying on the two happening to agree. Swapping <i>p</i> and "
        "<i>v</i> silently produces a graph that converges to the wrong trajectory.", "warn")

    st.append(Paragraph("8.2  Marginalisation is the whole point", H2))
    st.append(para(
        "When a keyframe leaves the window you cannot simply delete it. The measurements that "
        "touched it carry information about the keyframes that <i>remain</i>; dropping them "
        "throws that information away, which makes the estimate both worse and over-confident "
        "about being worse. The correct operation is a Schur complement:"))
    st += eq(
        "H_prior = H_kk - H_km H_mm^-1 H_mk\n"
        "b_prior = b_k  - H_km H_mm^-1 b_m")
    st.append(para(
        "Three implementation details are load-bearing. <b>Only factors that touch the departing "
        "keyframe are eliminated</b> — folding untouched factors into the prior double-counts "
        "them on the next solve. <b>The prior is held at a fixed linearisation point</b> "
        "(a first-estimates-Jacobian convention); re-linearising it around a moving estimate "
        "injects information that was never measured and is the standard route to an "
        "inconsistent estimator. And <b>a pseudo-inverse is used for "
        "H<sub>mm</sub></b>, so a partly-observed departing keyframe marginalises its observed "
        "subspace and leaves the rest alone."))

    st += callout(
        "Verified, not asserted",
        ["The property that matters is <b>batch equivalence</b>: solving the window <i>after</i> "
         "marginalisation must give the same answer as solving the full batch. The test solves "
         "to convergence, marginalises at the optimum, kicks the survivors, re-solves, and "
         "demands they return to within 2&nbsp;mm of the batch optimum.",
         "A companion test quantifies what it buys: naive deletion of the same keyframe leaves "
         "an error more than 5× larger. A sliding window that deletes instead of marginalising "
         "still runs, still converges, and still looks plausible on a plot — it is just quietly "
         "wrong. Only this test catches that."], "good")

    st.append(Paragraph("8.3  Observability, stated as a rank test", H2))
    st.append(para(
        "A visual-inertial graph measures relative geometry plus gravity. Gravity pins roll and "
        "pitch; nothing pins absolute position. So the Hessian of a window with no "
        "absolute-position measurement is rank deficient by exactly three, and the null space is "
        "“translate every keyframe identically”. The test suite asserts both the dimension and "
        "the direction of that null space, which documents why the position anchor exists and "
        "would catch a future factor that accidentally constrained something it cannot observe."))

    st.append(Paragraph("8.4  Which one should you run?", H2))
    st += table([
        ["", "ESKF (Section 7)", "Fixed-lag smoother"],
        ["Cost per update", "~µs (22×22)", "~100s of µs (150×150), per keyframe"],
        ["Linearisation", "once, at the current estimate", "repeatedly, over the window"],
        ["Recovers from a bad early estimate", "no", "yes"],
        ["Output rate", "every telemetry epoch", "every keyframe"],
        ["Use it for", "<b>flight, control, the default path</b>",
         "post-flight refinement, bag replay, initialisation, and as ground truth when tuning "
         "the filter"],
    ], widths=[42 * mm, 55 * mm, 67 * mm])
    st.append(para(
        "Both share the same measurement models and the same conventions, so a discrepancy "
        "between them is a real signal that something is wrong — which makes the smoother "
        "useful as a cross-check even when it is not in the flight path."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 9. Calibration
# --------------------------------------------------------------------------- #

def sec9():
    st = h1("9", "Calibration",
            "Three quantities you must measure on your own airframe")

    st.append(para(
        "None of these is published by the manufacturer, and none can be guessed. Skipping any "
        "of them produces a confident wrong answer rather than an obviously wrong one, which is "
        "considerably more expensive.", LEAD))

    st.append(Paragraph("9.1  IMU noise and biases", H2))
    st.append(para(
        "These set the process noise <font face='Courier'>Q</font>. Guess them too small and "
        "the filter over-trusts dead reckoning and diverges; too large and it ignores the IMU "
        "entirely. The tool is the <b>overlapping Allan deviation</b> of a static log, which "
        "separates white noise from bias instability rather than lumping them:"))
    st += figure("allan.png",
                 "Figure 6 — Reading an Allan deviation plot. The −1/2 slope gives the "
                 "white-noise density N at τ = 1 s; the flat minimum is bias instability; the "
                 "+1/2 tail gives the random walk K.", width=110 * mm)
    st += bullets([
        "<b>Two minutes, not ten seconds.</b> The estimator at cluster time τ averages over "
        "N&nbsp;−&nbsp;2τ/dt overlapping windows, so the bias-instability and random-walk "
        "regions — the ones that set how long the filter may trust dead reckoning — only become "
        "visible once the log is many times longer than the τ you care about.",
        "<b>The drone must be genuinely still</b>, on a surface that is not transmitting "
        "vibration. Real motion inflates the apparent noise, which quietly makes the filter "
        "sluggish forever after. The node reports detected non-stationarity as a warning rather "
        "than folding it into the answer.",
        "<b>The worst axis is used, not the mean.</b> The filter's noise model is isotropic, and "
        "being optimistic on the noisiest axis is what makes it diverge.",
        "<b>Sanity check built in.</b> If the mean acceleration magnitude is not ~9.81 m/s², the "
        "node says so — raw milli-g fed in by mistake is the common cause.",
    ])

    st.append(Paragraph("9.2  The camera-IMU rotation, from motion alone", H2))
    st.append(para(
        "Every visual measurement is in the camera frame and every inertial one in the body "
        "frame; without the rotation between them they cannot be combined at all. <b>A 5° error "
        "here couples rotation directly into apparent translation and produces a drift that "
        "looks exactly like scale error</b> — which is why people chase the wrong bug for days."))
    st.append(para(
        "No calibration target is needed. For a camera rigidly attached to a body:"))
    st += eq(
        "R_C1C2 = R_CB  R_B1B2  R_BC\n"
        "\n"
        "Conjugation preserves the rotation ANGLE and maps the AXIS, so with n_C, n_B\n"
        "the unit rotation axes of the two measured increments:\n"
        "\n"
        "    n_C = R_CB n_B          and        |theta_C| = |theta_B|")
    st.append(para(
        "That reduces hand-eye to Wahba's problem — find the rotation best aligning two sets of "
        "unit vectors — solved in closed form by SVD (Kabsch). No iteration, no initial guess, "
        "no local minimum."))
    st += callout(
        "Two screens, because one is not enough",
        ["Increments below a few degrees have noise-dominated axes and are dropped. Pairs whose "
         "two <i>angles</i> disagree violate the identity outright — almost always a time-sync "
         "error — and are dropped too.",
         "That cheap screen is <b>not sufficient</b>: a corrupted axis that happens to preserve "
         "the rotation angle sails straight through it and biases the closed-form fit by "
         "degrees. A second stage re-fits after rejecting pairs whose full conjugation residual "
         "exceeds a median+3·MAD cut. The test that exercises this deliberately includes a "
         "blunder with 0.1° of angle mismatch, which only the second stage removes."], "warn")
    st.append(para(
        "<b>Excitation is reported alongside the residual.</b> Rotating about a single axis "
        "leaves R<sub>BC</sub> undetermined about that axis; the fit will still converge and "
        "still report a small residual. The node computes the smallest singular value of the "
        "stacked rotation axes and says <font face='Courier'>POOR</font> below 0.25. A small "
        "residual with a low excitation score means the answer is confidently wrong."))
    st.append(para(
        "The translation <font face='Courier'>p<sub>BC</sub></font> is <i>not</i> fitted. It "
        "needs metric scale and well-excited translation; on a Tello the lever arm is a few "
        "centimetres and is far better taken from the mechanical layout than from noisy data."))

    st.append(Paragraph("9.3  The camera-IMU time offset", H2))
    st.append(para(
        "Recovered by cross-correlating angular-rate <i>magnitudes</i> — one from the visual "
        "front-end, one from the attitude surrogate. Magnitude is the right choice because it is "
        "frame-independent: it does not require R<sub>BC</sub> to be known yet, so the offset "
        "and the extrinsic can be solved in either order. In practice the offset is solved "
        "first and applied before the hand-eye fit, because an uncompensated offset makes the "
        "paired rotations disagree and would otherwise be absorbed into a wrong extrinsic."))
    st.append(para(
        "The correlation peak is refined to sub-sample resolution by fitting a parabola to the "
        "three values around the maximum. Without that the answer is quantised to the resampling "
        "period — 20&nbsp;ms at the default — which is the same order as the effect being "
        "measured. The reported correlation coefficient is the quality metric: below ~0.5, "
        "excite the drone more."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 10. ROS 2
# --------------------------------------------------------------------------- #

def sec10():
    st = h1("10", "ROS 2 concepts, as used here",
            "Executors, callback groups, QoS, TF — and the decisions they force")

    st.append(para(
        "ROS 2 concepts are covered here in the order they actually bite on this project, with "
        "the concrete decision each one forced.", LEAD))
    st += figure("architecture.png", "Figure 7 — Node and topic graph.")

    st.append(Paragraph("10.1  Executors and callback groups", H2))
    st.append(para(
        "A ROS 2 node does not run its callbacks; an <b>executor</b> does. The default "
        "<font face='Courier'>SingleThreadedExecutor</font> runs one callback at a time, so a "
        "callback that blocks blocks <i>everything</i> — which is exactly how the original "
        "driver could freeze <font face='Courier'>/emergency</font> for 21 seconds."))
    st.append(para(
        "<b>Callback groups</b> declare what may run concurrently. "
        "<font face='Courier'>MutuallyExclusiveCallbackGroup</font> guarantees its members "
        "never run simultaneously with each other; "
        "<font face='Courier'>ReentrantCallbackGroup</font> allows even a single callback to "
        "re-enter itself. Groups only take effect under a "
        "<font face='Courier'>MultiThreadedExecutor</font>."))
    st += table([
        ["Node", "Groups", "Why"],
        ["<font face='Courier'>tello</font> (driver)",
         "video · telemetry · control, plus a non-ROS command worker thread",
         "Video decode must not delay telemetry ingest, and neither may delay the RC dead-man. "
         "Blocking SDK round trips are pushed off the executor entirely — a worker thread is "
         "the right answer because the Tello's command socket is single-threaded by protocol."],
        ["<font face='Courier'>tello_vio</font>",
         "vision · telemetry",
         "The image callback runs feature detection and RANSAC — tens of milliseconds. "
         "Telemetry must not queue behind it. Mutual exclusivity <i>within</i> each group means "
         "no callback races itself; one lock guards the estimator, the only shared mutable "
         "state."],
    ], widths=[34 * mm, 46 * mm, 84 * mm])
    st += callout(
        "The rule of thumb",
        "Threads are not the fix for a blocking call — they are the fix for <i>concurrency</i>. "
        "A blocking call belongs off the executor entirely. Callback groups then let the "
        "genuinely independent work overlap.", "note")

    st.append(Paragraph("10.2  QoS: the silent no-connection", H2))
    st.append(para(
        "ROS 2 QoS profiles must be <i>compatible</i> for a subscription to connect at all, and "
        "an incompatible pair produces <b>no data and no error</b> — the most confusing failure "
        "mode in the middleware. Sensor streams here use "
        "<font face='Courier'>qos_profile_sensor_data</font> (best-effort, volatile, depth 1) "
        "on both sides."))
    st += table([
        ["Setting", "Choice", "Reason"],
        ["Reliability", "BEST_EFFORT for image/IMU; RELIABLE for commands",
         "Retransmitting a 250 ms-old video frame is worse than dropping it. Losing a takeoff "
         "command is not acceptable."],
        ["History / depth", "KEEP_LAST, depth 1 for sensors",
         "A deep queue on a latency-critical stream buys you stale data, not resilience."],
        ["Durability", "VOLATILE for sensors; TRANSIENT_LOCAL for static TF",
         "A node that starts late still needs the static transform; it does not need old frames."],
    ], widths=[30 * mm, 56 * mm, 78 * mm])

    st.append(Paragraph("10.3  TF and REP-105", H2))
    st += figure("tf_tree.png", "Figure 8 — The transform tree, and who owns each edge.")
    st.append(para(
        "REP-105 defines the <font face='Courier'>map → odom → base_link</font> chain and, "
        "crucially, the different guarantees each edge carries. "
        "<font face='Courier'>odom → base_link</font> must be <b>continuous</b> because "
        "controllers differentiate it — a discontinuity there is a step input to the aircraft. "
        "<font face='Courier'>map → odom</font> is <b>allowed to jump</b> because nothing "
        "differentiates it."))
    st.append(para(
        "That is what makes loop closure safe: when ORB-SLAM2 closes a loop and its trajectory "
        "snaps, the correction is published as a change to "
        "<font face='Courier'>map → odom</font>, and the VIO's "
        "<font face='Courier'>odom → base_link</font> is untouched. Publishing the loop closure "
        "straight into the odometry — the obvious-looking shortcut — is what makes drones lurch "
        "when a loop closes."))
    st += callout(
        "One owner per edge",
        "The original launch files ran a static publisher on <font face='Courier'>map → drone</font> "
        "while the driver published a barometer-derived transform on the <i>same</i> edge. TF "
        "does not merge those: lookups return whichever sample won the buffer, and RViz shows "
        "the drone flickering between them. Every edge in this system has exactly one publisher, "
        "and the launch files enforce it.", "warn")

    st.append(Paragraph("10.4  Message design", H2))
    st += bullets([
        "<b>Covariance conventions are semantics, not decoration.</b> "
        "<font face='Courier'>covariance[0] = −1</font> means “not measured”. All-zeros means "
        "“measured, and exactly known”. The driver now uses the former for the absent gyro and "
        "for the absent position, because the difference is what a downstream filter acts on.",
        "<b><font face='Courier'>child_frame_id</font> on Odometry is required.</b> The twist is "
        "expressed in it. Leaving it empty makes robot_localization's transform lookup fail and "
        "the measurement is dropped silently.",
        "<b>Every message needs a Header.</b> <font face='Courier'>TelloStatus</font> had none, "
        "so it could not be time-aligned with <font face='Courier'>/imu</font> or used with "
        "<font face='Courier'>message_filters</font>. Adding one was a prerequisite for fusing it.",
        "<b>Use the idiomatic type.</b> The downward rangefinder is now "
        "<font face='Courier'>sensor_msgs/Range</font>, which has a defined convention for "
        "out-of-range readings (<font face='Courier'>±inf</font>) — so “no reading” is "
        "representable instead of being encoded as a plausible number.",
    ])

    st.append(Paragraph("10.5  Parameters, launch and bags", H2))
    st += bullets([
        "<b>Everything measured is a parameter.</b> Sensor unit scales, noise densities, the "
        "extrinsic, the time offset. The defaults are documented with their provenance and "
        "flagged <font face='Courier'>CALIBRATE</font> in the config where they must be replaced.",
        "<b>Launch arguments over edited files.</b> One launch file covers bench testing, "
        "flight and bag replay.",
        "<b><font face='Courier'>use_sim_time</font> is wired through.</b> Without it, nodes "
        "stamp with wall time while a bag replays recorded stamps, and every time-based "
        "computation in the estimator silently becomes nonsense.",
        "<b>Record bags and iterate offline.</b> Recording "
        "<font face='Courier'>/image_raw /camera_info /imu /status /tof</font> lets you re-run "
        "calibration and re-tune the filter without flying — which matters more than usual here, "
        "because the Tello overheats sitting still and gives you ~10 minutes of battery.",
    ])
    st += code(
        "ros2 bag record -o flight1 /image_raw /camera_info /imu /status /tof\n"
        "ros2 bag play flight1 --clock\n"
        "ros2 launch tello_vio vio.launch.py driver:=false use_sim_time:=true")

    st.append(Paragraph("10.6  Diagnostics", H2))
    st.append(para(
        "The estimator publishes <font face='Courier'>diagnostic_msgs/DiagnosticArray</font> at "
        "2&nbsp;Hz with per-measurement acceptance rates and the last normalised innovation "
        "squared, current bias estimates, per-axis sigmas, front-end status and the last visual "
        "measurement's model, inlier count and parallax. Watch it in "
        "<font face='Courier'>rqt_robot_monitor</font>. Falling visual acceptance is the "
        "earliest visible sign of a bad extrinsic, a wrong time offset, or a scene the front-end "
        "cannot track — all of which otherwise present identically as “drift”."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 11. Results
# --------------------------------------------------------------------------- #

def sec11():
    st = h1("11", "Results", "Measured performance, and what the camera is actually worth")

    st.append(Paragraph("11.1  Test suite", H2))
    st.append(para(
        "116 tests, all passing, none requiring a drone or a ROS installation. They are "
        "organised around <i>failure modes</i> rather than around functions.", LEAD))
    st += table([
        ["Suite", "n", "What it pins down"],
        ["<font face='Courier'>test_lie</font>", "14",
         "Exp/Log round-trips including the θ→0 and θ→π branches; the right Jacobian against "
         "its defining identity; Euler axis semantics."],
        ["<font face='Courier'>test_preintegration</font>", "13",
         "Exactness against the generating trajectory; bias Jacobians against full "
         "re-integration; residual Jacobians against numerical differentiation; <b>covariance "
         "against 4000-sample Monte Carlo</b>; degenerate information matrices."],
        ["<font face='Courier'>test_eskf</font>", "15",
         "Every measurement Jacobian differentiated through the filter's own retraction; "
         "clone covariance and cross-terms; ZUPT/ZARU bias convergence; gating; PSD over a "
         "2000-step run."],
        ["<font face='Courier'>test_two_view</font>", "10",
         "General, planar and pure-rotation scenes; the translation-convention sign; outlier "
         "rejection at 25 % contamination; triangulation cheirality and parallax gating."],
        ["<font face='Courier'>test_frontend</font>", "7",
         "Tracking on rendered 3-D scenes; grid coverage; intrinsic rescaling; and "
         "<b>uncertainty calibration</b> — errors must lie inside the σ each measurement "
         "reports."],
        ["<font face='Courier'>test_smoother</font>", "8",
         "Convergence from a perturbed start; bias recovery; <b>marginalisation batch "
         "equivalence</b>; deletion measurably worse; the translation null space."],
        ["<font face='Courier'>test_calib</font>", "13",
         "Allan deviation against known densities; hand-eye exact and under noise; blunder "
         "rejection; single-axis excitation detection; time offset to 10 ms."],
        ["<font face='Courier'>test_sim3</font>", "7",
         "Umeyama exactness; the coplanar reflection case; RANSAC scale robustness."],
        ["<font face='Courier'>test_driver_conventions</font>", "5",
         "Cross-package: the driver's unit and frame conversions must match "
         "<font face='Courier'>tello_model</font>, since the two are duplicated by design."],
        ["<font face='Courier'>test_integration</font>", "7",
         "End-to-end simulated flight — the numbers below."],
    ], widths=[40 * mm, 9 * mm, 115 * mm])

    st.append(Paragraph("11.2  Simulated flight", H2))
    st.append(para(
        "A 60&nbsp;s figure-of-eight at ~1.5&nbsp;m and ~0.5&nbsp;m/s, with the drone's real "
        "limitations modelled: 10&nbsp;Hz telemetry with 12&nbsp;ms of jitter, whole-degree "
        "attitude quantisation, milli-g accelerometer quantisation, integer-decimetre velocity, "
        "a drifting barometer, a yaw free-running at 0.4°/s, and monocular vision that supplies "
        "bearing but never scale."))
    st += table([
        ["Metric", "Result", "Note"],
        ["Absolute trajectory error (RMS)", "<b>0.57 m</b>", "over 39.8 m of path"],
        ["Final drift", "<b>1.4 % of path</b>", ""],
        ["Metric scale (path-length ratio)", "<b>1.034</b>",
         "no GPS, no motion capture, no known landmark"],
        ["Velocity error (RMS)", "0.17 m/s", ""],
        ["Roll / pitch error (RMS)", "<b>1.8°</b>", "gravity-referenced, absolute"],
        ["Yaw error (RMS)", "13.0°", "free-running by design — no magnetometer"],
        ["Accelerometer bias recovered", "[0.134, −0.098, 0.055]",
         "true [0.12, −0.08, 0.05]"],
        ["Position NEES", "3.2 at 40 s, 3.8 at 60 s",
         "ideal 3.0 — mildly over-confident, growing with time"],
    ], widths=[52 * mm, 42 * mm, 70 * mm])

    st.append(Paragraph("11.3  What the camera is actually worth", H2))
    st += figure("vision_benefit.png",
                 "Figure 9 — ATE with and without the visual front-end, as the Tello's own "
                 "optical-flow velocity degrades. Mean of 5 seeds, 40 s each, log scale.",
                 width=120 * mm)
    st.append(para(
        "This is the most useful number in the report, and it is not the flattering one. "
        "<b>When the Tello's flow sensor is working well over textured ground, monocular VO adds "
        "essentially nothing to short-horizon odometry</b> — the flow estimate is already "
        "metric and already good, and vision only adds direction information that is largely "
        "redundant with it."))
    st.append(para("Vision earns its CPU in exactly two situations:"))
    st += bullets([
        "<b>When flow degrades.</b> Above roughly a metre of altitude, over a plain floor, in "
        "low light, or over a moving surface. At σ&nbsp;=&nbsp;0.8&nbsp;m/s vision halves the "
        "error; with the flow sensor dead it is the difference between 0.75&nbsp;m and "
        "15&nbsp;m — inertial dead reckoning on a 10&nbsp;Hz quantised accelerometer diverges "
        "within seconds.",
        "<b>For global consistency.</b> Loop closure, via the ORB-SLAM2 backend, is the only "
        "mechanism here that bounds drift over long runs rather than merely slowing it.",
    ])
    st += callout(
        "Why report this rather than tune it away",
        "It would have been easy to pick a scenario where vision looks essential and show only "
        "that. The honest framing is more useful: it tells you that if your flow sensor is "
        "healthy you should spend your effort on loop closure rather than on front-end tuning, "
        "and it tells you exactly which flight conditions make the camera load-bearing.", "good")

    st.append(Paragraph("11.4  Compute budget", H2))
    st += table([
        ["Stage", "Cost", "Notes"],
        ["KLT tracking + detection, 480×360, ~150 features", "~3 ms/frame",
         "measured; scales with feature count and pyramid levels"],
        ["Essential/homography estimation + model scoring", "~1–2 ms/keyframe",
         "only on keyframes, not every frame"],
        ["ESKF propagate + all updates", "&lt; 0.1 ms", "22×22"],
        ["Fixed-lag smoother, window 10", "~0.3 ms/keyframe", "150×150 dense Cholesky"],
        ["ORB-SLAM2 tracking (optional backend)", "~15–25 ms/frame",
         "separate process; this is what the budget is really spent on"],
    ], widths=[68 * mm, 25 * mm, 71 * mm])
    st.append(para(
        "The fast path leaves ample headroom inside a 33&nbsp;ms frame budget on one core. If "
        "you enable the ORB-SLAM2 backend, that is where the time goes — and it is worth it "
        "only for the loop closure, since the front-end here already covers odometry."))
    st.append(PageBreak())
    return st


# --------------------------------------------------------------------------- #
# 12-13 + appendix
# --------------------------------------------------------------------------- #

def sec12():
    st = h1("12", "Bring-up procedure", "The order to do things in, and why that order")

    st.append(para(
        "Each step depends on the previous one being right. Doing them out of order does not "
        "fail loudly — it produces plausible numbers that are wrong, which is worse.", LEAD))

    st.append(Paragraph("Step 0 — Build and verify without the drone", H3))
    st += code(
        "./scripts/install.sh                        # Ubuntu 22.04 + ROS 2 Humble\n"
        "./scripts/build.sh\n"
        "cd workspace/src/tello_vio && python3 -m pytest test/ -q     # 116 tests, no drone needed")

    st.append(Paragraph("Step 1 — Camera intrinsics", H3))
    st.append(para(
        "Everything geometric depends on these. The shipped "
        "<font face='Courier'>ost.yaml</font> is from another airframe and its tangential term "
        "(p<sub>1</sub>&nbsp;=&nbsp;−0.023) is large enough to suggest an under-constrained fit."))
    st += code("./scripts/cameracalib.sh          # SIZE=8x6 SQUARE=0.025 to override")
    st.append(para(
        "Fill the image corners, vary distance, and <b>tilt the board</b>. A target held flat "
        "and centred trades focal length against distance and yields a confident wrong "
        "f<sub>x</sub>. Copy the resulting <font face='Courier'>ost.yaml</font> over "
        "<font face='Courier'>workspace/src/tello/resource/ost.yaml</font> and rebuild."))

    st.append(Paragraph("Step 2 — IMU noise, drone still", H3))
    st += code("ros2 launch tello_vio calibrate.launch.py target:=imu duration:=120.0")
    st.append(para(
        "Motors off, on a solid non-vibrating surface. Point a fan at the drone: it overheats "
        "sitting powered on, and a two-minute log is long enough to hit thermal shutdown."))

    st.append(Paragraph("Step 3 — Extrinsic and time offset", H3))
    st += code("ros2 launch tello_vio calibrate.launch.py target:=camera_imu duration:=90.0")
    st.append(para(
        "Pick the drone up and rotate it smoothly about all three axes in front of a textured "
        "scene. Avoid jerks — motion blur breaks feature tracking. <b>Check the excitation "
        "score before believing the residual.</b> Merge both YAML fragments into "
        "<font face='Courier'>config/tello_vio.yaml</font>."))

    st.append(Paragraph("Step 4 — Record a bag, then iterate offline", H3))
    st += code(
        "ros2 launch tello tello.launch.py                 # driver + keyboard control\n"
        "ros2 bag record -o flight1 /image_raw /camera_info /imu /status /tof\n"
        "#  ... fly a loop that returns to its start ...\n"
        "ros2 bag play flight1 --clock\n"
        "ros2 launch tello_vio vio.launch.py driver:=false use_sim_time:=true rviz:=true")
    st.append(para(
        "Flying a loop that returns to its start gives you a free accuracy measurement: the "
        "final position error against the origin is your drift over the whole flight."))

    st.append(Paragraph("Step 5 — Watch the diagnostics, not the trajectory", H3))
    st += code("ros2 run rqt_robot_monitor rqt_robot_monitor\nros2 topic echo /diagnostics")
    st += table([
        ["Symptom in diagnostics", "Most likely cause"],
        ["Visual acceptance falling below ~70 %",
         "Wrong time offset, wrong extrinsic, or a scene the front-end cannot track "
         "(blank wall, low light, motion blur)."],
        ["<font face='Courier'>vel_body</font> rejections rising",
         "<font face='Courier'>speed_to_mps</font> is wrong for your firmware, or the drone is "
         "above the flow sensor's useful range."],
        ["<font face='Courier'>tof</font> rejections in bursts",
         "Normal over furniture and drop-offs — that is the gate doing its job."],
        ["Accel bias drifting steadily rather than settling",
         "No ZUPT is firing. Check the stationarity thresholds, and that the drone actually sits "
         "still before takeoff."],
        ["σ growing without bound",
         "No metric measurement is being accepted at all. Check <font face='Courier'>/status</font> "
         "is flowing and that <font face='Courier'>tello_msg</font> is built."],
    ], widths=[58 * mm, 106 * mm])
    st.append(PageBreak())
    return st


def sec13():
    st = h1("13", "Limitations and next steps", "What this does not do")

    st.append(Paragraph("Limitations", H2))
    st += bullets([
        "<b>Yaw is unbounded.</b> There is no magnetometer, so heading free-runs — ~13° RMS "
        "over 60&nbsp;s in simulation. Roll and pitch are gravity-referenced and stay bounded; "
        "yaw does not. Bounding it needs loop closure or an external reference.",
        "<b>Metric scale depends on the flow sensor.</b> Section&nbsp;11.3 quantifies exactly "
        "how much. Above 1–2&nbsp;m, or over a featureless floor, the metric anchor weakens.",
        "<b>Preintegration is fed a 10&nbsp;Hz surrogate rate, not a gyroscope.</b> The "
        "mathematics is exact and rate-agnostic; the information content is not what it would be "
        "with a real IMU.",
        "<b>No relocalisation in the fast path.</b> KLT has no descriptors. Recovering from total "
        "tracking loss is ORB-SLAM2's job, and requires running that backend.",
        "<b>The filter is mildly over-confident</b> — NEES ~3.8 against an ideal 3.0 at 60&nbsp;s, "
        "and growing. Most likely the unmodelled yaw drift. It is well inside the range where "
        "gating still behaves, but it is the direction of error that matters.",
        "<b>Simulation is not flight.</b> Every number in Section&nbsp;11 comes from a simulator "
        "built from the SDK's documented behaviour and community-reproduced unit conventions. "
        "Real H.264 artefacts, rolling shutter, motion blur, propeller vibration and WiFi "
        "dropouts are all worse than modelled. <b>Validate on a recorded bag before trusting "
        "any of it.</b>",
        "<b>Unit scales are firmware-dependent.</b> <font face='Courier'>speed_to_mps</font> in "
        "particular rests on a community-reproduced reading, not on documentation. It is a "
        "parameter for that reason, and it is the first thing to check if your scale is wrong "
        "by a clean factor of ten.",
    ])

    st.append(Paragraph("Next steps, in order of value", H2))
    st += table([
        ["", "What", "Why it is worth doing"],
        ["1", "Validate on real bags and re-fit the front-end noise constants",
         "Every number here is simulated. Fitting σ<sub>rot</sub> and σ<sub>dir</sub> against "
         "real imagery is the single highest-value hour available."],
        ["2", "Close the loop: run the ORB-SLAM2 backend and map_align in anger",
         "This is the only mechanism that <i>bounds</i> drift and yaw rather than slowing them. "
         "The wiring exists and is untested against a real map."],
        ["3", "Add a lightweight place-recognition layer to the fast path",
         "Would give relocalisation without paying ORB-SLAM2's full per-frame cost — the main "
         "structural gap in the current front-end."],
        ["4", "Estimate the time offset online as a filter state",
         "It is currently a calibrated constant. WiFi latency varies with congestion, and the "
         "residual variation is not absorbed by any fixed t<sub>d</sub>."],
        ["5", "Add an external IMU",
         "The only way to reach genuine tightly-coupled VIO on this airframe. Every module here "
         "is written to accept a real gyro unchanged — including preintegration, which is "
         "rate-agnostic by construction."],
        ["6", "Run the smoother in the flight path on a sliding window",
         "Currently built, tested and used offline. Promoting it would recover from bad early "
         "linearisations, at a cost the budget can afford."],
    ], widths=[8 * mm, 62 * mm, 94 * mm])
    st.append(PageBreak())
    return st


def appendix():
    st = h1("A", "Reference", "Files, parameters, topics")

    st.append(Paragraph("A.1  Where things live", H2))
    st += table([
        ["Path", "Contents"],
        ["<font face='Courier'>tello_vio/lie.py</font>",
         "SO(3)/SE(3). <b>All conventions declared here.</b>"],
        ["<font face='Courier'>tello_vio/tello_model.py</font>",
         "Units, frames, the surrogate gyro, the stationarity detector."],
        ["<font face='Courier'>tello_vio/preintegration.py</font>",
         "On-manifold IMU preintegration with bias Jacobians and covariance."],
        ["<font face='Courier'>tello_vio/eskf.py</font>",
         "22-state error-state KF with stochastic cloning. The flight path."],
        ["<font face='Courier'>tello_vio/smoother.py</font>",
         "Fixed-lag factor graph, factors, Schur-complement marginalisation."],
        ["<font face='Courier'>tello_vio/two_view.py</font>",
         "E/H model selection, relative pose, triangulation, PnP."],
        ["<font face='Courier'>tello_vio/frontend.py</font>",
         "KLT tracking, keyframing, per-measurement uncertainty."],
        ["<font face='Courier'>tello_vio/calib.py</font>",
         "Allan deviation, hand-eye, time offset."],
        ["<font face='Courier'>tello_vio/sim3.py</font>",
         "Umeyama alignment, RANSAC. map↔odom and monocular map scale."],
        ["<font face='Courier'>tello_vio/nodes/</font>",
         "<font face='Courier'>vio</font>, <font face='Courier'>imu_calib</font>, "
         "<font face='Courier'>camera_imu_calib</font>, <font face='Courier'>map_align</font>."],
        ["<font face='Courier'>tello/tello/node.py</font>",
         "The driver."],
        ["<font face='Courier'>slam/src/orbslam2/</font>",
         "ORB-SLAM2 wrapper: pose, TF, path, map, tracking state."],
    ], widths=[62 * mm, 102 * mm])

    st.append(Paragraph("A.2  Parameters you must set", H2))
    st += table([
        ["Parameter", "Default", "Source"],
        ["<font face='Courier'>extrinsic_rpy_deg</font>", "[0, 0, 0]",
         "<font face='Courier'>camera_imu_calib</font>"],
        ["<font face='Courier'>time_offset_s</font>", "0.0",
         "<font face='Courier'>camera_imu_calib</font>"],
        ["<font face='Courier'>accel_noise_density</font>", "0.08",
         "<font face='Courier'>imu_calib</font>"],
        ["<font face='Courier'>gyro_noise_density</font>", "0.03",
         "<font face='Courier'>imu_calib</font>"],
        ["<font face='Courier'>speed_to_mps</font>", "0.1",
         "verify against a known flown distance"],
        ["<font face='Courier'>extrinsic_xyz_m</font>", "[0.03, 0, −0.01]",
         "mechanical layout, not fitted"],
        ["camera intrinsics", "shipped <font face='Courier'>ost.yaml</font>",
         "<font face='Courier'>camera_calibration</font> — replace it"],
    ], widths=[52 * mm, 40 * mm, 72 * mm])

    st.append(Paragraph("A.3  Command reference", H2))
    st += code(
        "# build and test\n"
        "./scripts/build.sh                       # CLEAN=1 to wipe build/install/log\n"
        "cd workspace/src/tello_vio && python3 -m pytest test/ -q\n"
        "\n"
        "# calibrate\n"
        "./scripts/cameracalib.sh\n"
        "ros2 launch tello_vio calibrate.launch.py target:=imu        duration:=120.0\n"
        "ros2 launch tello_vio calibrate.launch.py target:=camera_imu duration:=90.0\n"
        "\n"
        "# fly\n"
        "ros2 launch tello_vio vio.launch.py rviz:=true\n"
        "ros2 launch tello_vio vio.launch.py video_scale:=0.5 video_target_fps:=20\n"
        "\n"
        "# with the ORB-SLAM2 loop-closing backend\n"
        "./scripts/orbslam.sh\n"
        "export ORB_SLAM2_ROOT_DIR=\"$PWD/libs/ORB_SLAM2\"\n"
        "./scripts/build.sh\n"
        "ros2 launch tello_vio vio.launch.py slam:=true \\\n"
        "    vocabulary:=$PWD/libs/ORB_SLAM2/Vocabulary/ORBvoc.txt \\\n"
        "    slam_config:=$(ros2 pkg prefix orbslam2)/share/orbslam2/config.yaml\n"
        "\n"
        "# offline\n"
        "ros2 bag record -o flight1 /image_raw /camera_info /imu /status /tof\n"
        "ros2 launch tello_vio vio.launch.py driver:=false use_sim_time:=true\n"
        "\n"
        "# runtime\n"
        "ros2 service call /tello_vio/reset std_srvs/srv/Trigger\n"
        "ros2 run rqt_robot_monitor rqt_robot_monitor")

    st.append(Paragraph("A.4  Selected references", H2))
    st += bullets([
        "Forster, Carlone, Dellaert, Scaramuzza — <i>On-Manifold Preintegration for Real-Time "
        "Visual-Inertial Odometry</i>, IEEE T-RO 2017. The preintegration in Section&nbsp;5.",
        "Solà — <i>Quaternion Kinematics for the Error-State Kalman Filter</i>, 2017. The ESKF "
        "formulation in Section&nbsp;7.",
        "Roumeliotis &amp; Burdick — <i>Stochastic Cloning</i>, ICRA 2002. Section&nbsp;7.2.",
        "Mur-Artal &amp; Tardós — <i>ORB-SLAM2</i>, IEEE T-RO 2017. The map backend, and the "
        "homography/fundamental model-selection score in Section&nbsp;6.3.",
        "Umeyama — <i>Least-Squares Estimation of Transformation Parameters Between Two Point "
        "Patterns</i>, IEEE PAMI 1991. Section&nbsp;8 and the map alignment.",
        "Qin, Li &amp; Shen — <i>VINS-Mono</i>, IEEE T-RO 2018. The marginalisation and "
        "first-estimates-Jacobian treatment in Section&nbsp;8.2.",
        "REP-103 (units and coordinate conventions) and REP-105 (coordinate frames for mobile "
        "platforms). Sections&nbsp;2.3 and&nbsp;10.3.",
    ])
    return st


# --------------------------------------------------------------------------- #

def build():
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN,
                          title="Tello VIO — Technical Report",
                          author="Roey Turgeman",
                          subject="Visual-Inertial Odometry on a DJI Tello (ROS 2 Humble)")
    frame = Frame(MARGIN, MARGIN, PW - 2 * MARGIN, PH - 2 * MARGIN, id="f")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_page),
        PageTemplate(id="body", frames=[frame], onPage=body_page),
    ])

    story = []
    story += build_cover()[:-1]
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story += build_toc()
    for fn in (sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec10,
               sec11, sec12, sec13, appendix):
        story += fn()
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
