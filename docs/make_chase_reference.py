#!/usr/bin/env python3
"""Build the drone-chase technical reference PDF.

Reuses the typographic system of make_report.py so the project's documents
look like one family. Figures come from make_chase_figures.py.
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer)

from make_report import (ACCENT, BODY, BUL, CAP, COVER_S, COVER_T, GOOD, H1,
                         H1SUB, H2, H3, INK, LEAD, LINE, MARGIN, MUTED, PH,
                         PW, TD, TDB, TDC, WARN, bullets, callout, code, eq,
                         h1, para, table)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, 'figures_chase')
OUT = os.path.join(HERE, 'Drone_Chase_RL_Reference.pdf')

CW = PW - 2 * MARGIN          # content width


def para(t, style=BODY):          # list-returning, unlike make_report's
    return [Paragraph(t, style)]


def figure(name, caption, width=None):
    from PIL import Image as PILImage
    path = os.path.join(FIGD, name)
    iw, ih = PILImage.open(path).size
    w = width or CW
    h = w * ih / iw
    max_h = PH - 2 * MARGIN - 70
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(path, width=w, height=h)
    img.hAlign = 'CENTER'
    return [Spacer(1, 5), img, Paragraph(caption, CAP)]


def h2(t):
    return [Paragraph(t, H2)]


def h3(t):
    return [Paragraph(t, H3)]


# --------------------------------------------------------------------------- #
# page furniture
# --------------------------------------------------------------------------- #

def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor('#0f1620'))
    canvas.rect(0, PH - 86 * mm, PW, 86 * mm, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PH - 86 * mm, PW, 2.2 * mm, stroke=0, fill=1)
    canvas.restoreState()


def body_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PH - MARGIN + 7 * mm, PW - MARGIN, PH - MARGIN + 7 * mm)
    canvas.setFont('Helvetica', 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PH - MARGIN + 9.5 * mm,
                      'Vision-Based Drone Chase  ·  Deep Reinforcement Learning '
                      'on a DJI Tello  ·  ROS 2 Humble')
    canvas.line(MARGIN, MARGIN - 5 * mm, PW - MARGIN, MARGIN - 5 * mm)
    canvas.drawRightString(PW - MARGIN, MARGIN - 9 * mm,
                           str(canvas.getPageNumber()))
    canvas.drawString(MARGIN, MARGIN - 9 * mm, 'Technical Reference')
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# cover + contents
# --------------------------------------------------------------------------- #

def build_cover():
    st = [Spacer(1, 30 * mm)]
    st.append(Paragraph(
        '<font color="#ffffff">Vision-Based<br/>Drone Chase</font>',
        ParagraphStyle('t', parent=COVER_T, fontSize=31, leading=37)))
    st.append(Spacer(1, 6 * mm))
    st.append(Paragraph(
        '<font color="#9fb3cd">Deep reinforcement learning on a DJI Tello:<br/>'
        'the algorithms, the three-tier simulation method,<br/>'
        'and the ROS&nbsp;2 flight stack</font>',
        ParagraphStyle('s', parent=COVER_S, fontSize=12.4, leading=17.5)))
    st.append(Spacer(1, 30 * mm))

    st.append(Paragraph(
        'A drone that chases another drone using nothing but a camera. The '
        'policy that decides where to point is <b>learned</b>; the rule that '
        'decides how fast to close is <b>hand-written</b>; and everything is '
        'proven in three simulators of increasing realism before a propeller '
        'turns. This document explains every piece of that, in order, with '
        'the numbers each stage actually produced.', LEAD))
    st.append(Spacer(1, 6 * mm))

    rows = [
        ['Reinforcement-learning verdict (20 runs, 4 algorithms &times; 5 seeds)',
         'ACCEPT'],
        ['Tier&nbsp;B physics agreement (A&harr;B replay gate, real Gazebo)',
         '8.0 px vs 40 px tolerance'],
        ['Tier&nbsp;C follow mission, vision in the loop', '100 % over 1,500 frames'],
        ['Detector precision (with drone-free frames in the exam)', '0.996'],
        ['Verified interception in the rendered warehouse', '3.6 m chase, caught at 0.48 m'],
        ['Flight-stack nodes implemented and integration-tested', '6 nodes, 7 tests'],
        ['Total automated tests passing without hardware', '94'],
    ]
    st += table([[Paragraph(a, TD), Paragraph(f'<b>{b}</b>', TD)] for a, b in rows],
                widths=[112 * mm, 52 * mm], header=False)
    st.append(Spacer(1, 4 * mm))
    st += callout(
        'What this document is careful about',
        'Every number here came from a run on this hardware. Where a value is '
        'a design choice rather than a measurement it is marked '
        '<b>[D]</b>; where it is a placeholder that real flight must replace '
        'it is marked <b>[!]</b>. Section&nbsp;9 states plainly what the '
        'system has <i>not</i> yet proven.', 'note')
    return st


def build_contents():
    st = [NextPageTemplate('body'), PageBreak()]
    st += h1('', 'Contents')
    rows = [
        ['1', 'The problem, and the shape of the answer', 'what it does, and who decides what'],
        ['2', 'Why three simulators', 'the method, and what each tier can and cannot prove'],
        ['3', 'The one contract', 'observation, action, forward law, checkpoint'],
        ['4', 'Tier A &mdash; learning the behaviour', 'the MDP, the reward, TD3 vs DDPG, the protocol'],
        ['5', 'Tier B &mdash; does the maths match physics?', 'Gazebo, the sign test, the A&harr;B gate'],
        ['6', 'Tier C &mdash; does it survive real vision?', 'warehouse, dataset, detector, closed loop'],
        ['7', 'The flight stack', 'six ROS 2 nodes, in detail'],
        ['8', 'Safety', 'what restrains the aircraft, and what cannot'],
        ['9', 'Results, and what is not yet proven', 'the honest ledger'],
        ['10', 'Bring-up procedure', 'the flight ladder, step by step'],
        ['A', 'Reference tables', 'files, topics, parameters, commands'],
    ]
    st += table([[Paragraph(f'<b>{n}</b>', TD), Paragraph(f'<b>{t}</b>', TD),
                  Paragraph(f'<font color="#5b6472">{s}</font>', TD)]
                 for n, t, s in rows],
                widths=[12 * mm, 74 * mm, 78 * mm], header=False)
    return st


# --------------------------------------------------------------------------- #
# 1. the problem
# --------------------------------------------------------------------------- #

def sec1():
    st = [PageBreak()]
    st += h1('1', 'The problem, and the shape of the answer',
             'One drone follows another using a single camera and nothing else.')
    st += para(
        'The task is deceptively simple to state: a DJI Tello watches the '
        'world through its forward camera, finds another drone in the image, '
        'and flies so as to keep it centred at a chosen distance &mdash; or, '
        'in the intercept variant, closes until it reaches it. There is no '
        'GPS, no motion capture, no depth sensor and no second camera. The '
        'only signal is a box around a small grey object in a 960&times;720 '
        'frame that arrives between 150 and 350&nbsp;ms late.')
    st += h2('1.1&nbsp;&nbsp;Two learned things, and one thing that is not learned')
    st += para(
        'It is worth being precise about this from the first page, because it '
        'is the most common misunderstanding of the system. There are '
        '<b>two trained models</b> and they do entirely different jobs:')
    st += bullets([
        '<b>The eyes</b> &mdash; a YOLO object detector. Input: a camera '
        'frame. Output: a box. It is trained by ordinary supervised learning '
        'on labelled images (Section&nbsp;6).',
        '<b>The brain</b> &mdash; a reinforcement-learning policy. Input: '
        'where the box is and how it has been moving. Output: how to move '
        'the aircraft. It is trained by trial and error against a simulated '
        'target (Section&nbsp;4).',
    ])
    st += para(
        'And there is a third component that is emphatically <i>not</i> '
        'learned: the <b>forward-speed law</b>. The policy controls only two '
        'axes &mdash; vertical speed and yaw rate, the two axes that keep the '
        'target in frame. How fast the aircraft moves toward or away from the '
        'target is computed by a short, auditable, hand-written rule.')
    st += callout(
        'Why the forward axis is deliberately not learned',
        ['Monocular range is weak: the only distance signal is how wide the '
         'box is, and that estimate degrades with the square of distance. The '
         'forward axis is also the one that decides collision energy. A '
         'monotonic, clamped rule on that axis can be read, reasoned about '
         'and bounded by a human in a way a network cannot.',
         'Meanwhile the genuinely hard perceptual-control problem &mdash; '
         'keeping a small, fast, intermittently-detected object inside the '
         'frame despite a third of a second of delay &mdash; is exactly what '
         'reinforcement learning is good at, and exactly what the reward '
         'measures.'], 'note')
    st += para(
        'This split also survives deployment unchanged: in the flight stack '
        'of Section&nbsp;7 the learned part is <font face="Courier">'
        'chase_policy</font> and the hand-written part is <font '
        'face="Courier">chase_depth_rule</font>, and they own disjoint axes '
        'of the command by construction.')

    st += h2('1.2&nbsp;&nbsp;The aircraft is part of the problem')
    st += para(
        'A stock Tello is a hostile platform for closed-loop vision, and '
        'every design decision downstream is shaped by its limits. These are '
        'measured properties of this airframe, not estimates:')
    rows = [
        ['Property', 'Value', 'Consequence for the design'],
        ['Video latency', '150&ndash;350 ms', 'The policy must be delay-aware; '
         'it sees the past, not the present'],
        ['Video stamp', 'applied at publish, not capture',
         'True latency is under-reported by the decode+poll time'],
        ['Command interface', 'normalised sticks in [&minus;1,&nbsp;1], scaled to &plusmn;100',
         'The policy output range and the aircraft input range coincide exactly'],
        ['Command dead-man', '0.35 s', 'Commands must be published continuously at '
         '&ge;&nbsp;10 Hz, including zeros'],
        ['Gyroscope', 'none', 'No angular-rate feedback; attitude only, in whole degrees at 10 Hz'],
        ['Position', 'none (no GPS, no mocap)', 'A lateral geofence is impossible &mdash; '
         'see Section&nbsp;8'],
        ['Target size', '0.098 m body / 0.180 m prop span',
         'At 4 m the target is ~23 px wide: a small-object detection problem'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TDC), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[36 * mm, 42 * mm, 86 * mm])
    st += callout(
        'The dead-man is a feature, not an obstacle',
        'Because the Tello latches its last stick command forever, the driver '
        'zeroes the sticks if no command has arrived for 0.35&nbsp;s. That '
        'means a crashed or hung policy node <i>stops</i> the aircraft rather '
        'than leaving it flying. The flight stack is built around this: a '
        'zero command is a message that must be <i>sent</i>, never an absence '
        'of messages.', 'good')
    return st


# --------------------------------------------------------------------------- #
# 2. three tiers
# --------------------------------------------------------------------------- #

def sec2():
    st = [PageBreak()]
    st += h1('2', 'Why three simulators',
             'Three different questions, three different fidelities, one shared contract.')
    st += para(
        'Before the aircraft flies, three separate questions must be '
        'answered, and no single simulator answers them well. Learning needs '
        'millions of attempts, which rules out anything that renders. '
        'Checking the physics needs a rigid-body engine. Checking the vision '
        'needs photorealistic pixels. So the work is split into three tiers, '
        'each of which is fast at exactly the thing it must prove.')
    st += figure('tiers.png',
                 'The three tiers. They never talk to each other at run time: '
                 'they are connected by shared code and by files (a trained '
                 'checkpoint, a dataset, a detector).')
    rows = [
        ['Tier', 'What it is', 'Speed', 'The question it answers'],
        ['<b>A</b>', 'Pure-Python kinematics and a simulated detector. No '
         'renderer, no physics engine, no ROS.',
         'thousands of steps/min', '<b>Can the behaviour be learned at all?</b>'],
        ['<b>B</b>', 'Gazebo Harmonic: real rotors, inertia, drag. Headless, '
         'stepped in lockstep. Measurements come from projecting ground '
         'truth, not from a camera.',
         '~real time', '<b>Does the maths match real physics?</b>'],
        ['<b>C</b>', 'Unreal Engine 5.7 via Project AirSim: rendered '
         '960&times;720 frames, the real YOLO detector in the loop.',
         '~2 decisions/s', '<b>Does it survive a real camera and a real detector?</b>'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD),
                              Paragraph(c, TDC), Paragraph(d, TD)]
                             for a, b, c, d in rows[1:]],
                widths=[10 * mm, 66 * mm, 25 * mm, 63 * mm])
    st += h2('2.1&nbsp;&nbsp;Why no tier has a window')
    st += para(
        'Tier&nbsp;A and Tier&nbsp;B render nothing, deliberately. Tier&nbsp;A '
        'is arithmetic; there is nothing to draw. Tier&nbsp;B has no camera at '
        'all &mdash; its "detector" is an oracle that projects the true '
        'relative position into image coordinates through the same pinhole '
        'model the real camera obeys, then delays and corrupts it. Rendering '
        'would add cost and no information, because Tier&nbsp;B is not the '
        'tier that tests vision. Only Tier&nbsp;C draws pixels, because in '
        'Tier&nbsp;C the pixels <i>are</i> the experiment.')
    st += h2('2.2&nbsp;&nbsp;What each tier cannot prove')
    st += para(
        'Being explicit about the ceiling of each tier is what makes the '
        'ladder meaningful:')
    st += bullets([
        '<b>Tier A cannot prove its own realism.</b> It can only show that a '
        'policy learns to solve the world it was given. Whether that world '
        'resembles a flying machine is precisely what Tier&nbsp;B tests.',
        '<b>Tier B cannot say anything about perception.</b> Its measurements '
        'are derived from truth. A detector that hallucinates, or misses '
        'small targets, is invisible here.',
        '<b>Tier C cannot prove the real world.</b> Its images come from a '
        'renderer, and a detector trained on them has only ever seen that '
        'renderer. This is the gap that remains open at the end of this '
        'document (Section&nbsp;9).',
    ])
    return st


# --------------------------------------------------------------------------- #
# 3. the contract
# --------------------------------------------------------------------------- #

def sec3():
    st = [PageBreak()]
    st += h1('3', 'The one contract',
             'What makes a checkpoint trained in Tier A run unchanged in Tier B, '
             'Tier C and on the aircraft.')
    st += para(
        'The tiers are interchangeable only because they all speak the same '
        'four interfaces, and because those interfaces exist once in code and '
        'are <i>imported</i> rather than reimplemented. This is the single '
        'most important structural decision in the project: it makes '
        'train/deploy skew impossible rather than merely unlikely.')

    st += h2('3.1&nbsp;&nbsp;The observation &mdash; 14 numbers')
    st += para(
        'Every tier and the aircraft build the policy\'s input with the same '
        'function, <font face="Courier">ObservationAssembler.assemble()</font>:')
    st += figure('observation.png',
                 'The observation vector. Errors are normalised to '
                 '[&minus;1,&nbsp;1] by half the frame, so the policy is '
                 'resolution-independent.')
    st += para(
        'Two details in that layout carry a great deal of weight:')
    st += bullets([
        '<b>On a miss, the errors are zeroed &mdash; never held.</b> It is '
        'tempting to repeat the last known position when the detector fails. '
        'That trains the policy to chase ghosts: it learns to act confidently '
        'on stale information, which is exactly the behaviour that flies a '
        'drone into a wall after the target leaves the frame. Instead the '
        'errors go to zero and <font face="Courier">visible</font> goes to 0, '
        'so "I cannot see it" is a distinct, learnable state.',
        '<b>The last four actions are part of the input.</b> This is what '
        'makes the policy <i>latency-aware</i>. The image describes the world '
        'as it was 150&ndash;350&nbsp;ms ago; the action history describes '
        'what the aircraft has done since. With a 10&nbsp;Hz control rate, '
        'k&nbsp;=&nbsp;4 ticks spans 400&nbsp;ms &mdash; slightly more than '
        'the worst-case delay, by construction.',
    ])
    st += figure('timeline.png',
                 'Where the delay lives. The policy never sees the present; '
                 'the action history is how it compensates.')

    st += h2('3.2&nbsp;&nbsp;The action &mdash; two sticks')
    st += para(
        'The actor emits two numbers in [&minus;1,&nbsp;1]. They map onto the '
        'aircraft with <b>no conversion layer whatsoever</b>, which is a rare '
        'piece of luck worth protecting:')
    st += eq(
        'action[0] = a_v    ->  linear.z   (vertical stick, +up)\n'
        'action[1] = a_yaw  ->  angular.z  (yaw stick, +counter-clockwise)\n'
        '\n'
        'sign convention, verified by unit test and re-tested per tier:\n'
        '   +angular.z (CCW)  moves the target box RIGHT in the image (+u)\n'
        '   +linear.z  (up)   moves the target box DOWN  in the image (+v)')
    st += para(
        'Because <font face="Courier">/cmd_vel</font> already takes normalised '
        'sticks in [&minus;1,&nbsp;1] and the driver scales them to the SDK\'s '
        '&plusmn;100 internally, the network\'s output range and the '
        'aircraft\'s input range are the same interval. Inserting any '
        'rescaling between them would silently change what the policy\'s '
        'learned magnitudes mean, so none exists, and the checkpoint loader '
        'refuses any policy whose declared action scale is not exactly 1.0.')
    st += callout(
        'Signs are tested, never assumed',
        'Three simulators plus one aircraft is four independent chances for a '
        'silently flipped axis, and a flipped yaw turns a chase into an '
        'escape. Each tier therefore runs a sign test that commands one axis '
        'and asserts the direction the target moves in the image. The flight '
        'stack ships two such tests as automated integration tests '
        '(Section&nbsp;7.7).', 'warn')

    st += h2('3.3&nbsp;&nbsp;The forward law &mdash; range from a box width')
    st += para(
        'Because the target is a known object, the width of its box is a '
        'distance measurement through the pinhole relation:')
    st += eq('d  =  f_x * W / w        f_x = 919.42 px,  W = 0.098 m (Tello body width)')
    st += para(
        'The source work commands forward motion from the <i>fraction of the '
        'frame</i> the box covers, with thresholds at 20&nbsp;% and '
        '55&nbsp;%. Against a Tello-sized target and this camera those '
        'thresholds are catastrophic: 20&nbsp;% of the frame width is not '
        'reached until roughly 0.47&nbsp;m, and 55&nbsp;% corresponds to '
        'about 0.17&nbsp;m. A faithful port would therefore command '
        '"too far &mdash; move forward" at <i>every safe separation</i> and '
        'stop only when the aircraft were a few tens of centimetres apart.')
    st += figure('depth_law.png',
                 'Left: the published ratio thresholds mapped onto real range '
                 'for this target &mdash; a collision policy. Right: the '
                 'metric law actually used.')
    st += para('The law that is used instead, identical in all four places it runs:')
    st += eq(
        'FOLLOW      err = d - standoff                      (standoff = 2.0 m)\n'
        '            |err| < 0.25 m            ->  v = 0     (deadband)\n'
        '            otherwise                  ->  v = clip(0.8 * err, -0.8, +0.8)  m/s\n'
        '\n'
        'INTERCEPT   d <= r_cap (0.5 m)         ->  v = 0     (inside: terminal)\n'
        '            otherwise                  ->  v = clip(1.0 * (d - r_cap), 0.15, 1.2)  m/s\n'
        '\n'
        'NO DETECTION                           ->  v = 0     (hold; never coast)')

    st += h2('3.4&nbsp;&nbsp;The checkpoint &mdash; auditable or it is folklore')
    st += para(
        'A saved policy is not just weights. Each bundle carries the exact '
        'observation specification and its hash, the action map, the control '
        'rate, the resolved environment config, the git commit, the seed, the '
        'environment-step count and the library versions. Any consumer that '
        'cannot reproduce the observation hash <b>refuses to load it</b>. '
        'That refusal is the entire sim-to-real audit trail in one '
        'mechanism: an actor is meaningless without the exact input contract '
        'it trained under, and a silent mismatch would fly a network on '
        'numbers that mean something different from what it learned.')
    st += code(
        "obs_spec_hash   = '3e902aea0712a3c9'\n"
        "action_map      = {'order': ['linear.z', 'angular.z'],\n"
        "                   'scale': 1.0, 'stick_range': [-1.0, 1.0],\n"
        "                   'signs': '+angular.z (CCW) moves box +u; "
        "+linear.z (up) moves box +v'}\n"
        "control_rate_hz = 10.0\n"
        "git_sha         = 'a4c71c7111b76eb662cf9eddc26bb3ff94e6493b'")
    return st


# --------------------------------------------------------------------------- #
# 4. Tier A
# --------------------------------------------------------------------------- #

def sec4():
    st = [PageBreak()]
    st += h1('4', 'Tier A &mdash; learning the behaviour',
             'The Markov decision process, the reward, and the algorithm that solves it.')
    st += para(
        'Tier&nbsp;A is where the policy is actually trained. It is a pure '
        'Python environment: no renderer, no physics engine, no ROS. That '
        'austerity is the point &mdash; it runs thousands of steps per minute, '
        'which is what makes twenty independent training runs affordable.')

    st += h2('4.1&nbsp;&nbsp;The decision problem')
    rows = [
        ['Element', 'Definition'],
        ['State (observed)', 'The 14-dim vector of Section&nbsp;3.1 &mdash; '
         'delayed, noisy, sometimes absent. The policy never sees true geometry.'],
        ['Action', 'a &isin; [&minus;1,&nbsp;1]&sup2; &rarr; vertical stick, yaw stick. '
         'Scaled by v_max = 1.5 m/s and &omega;_max = 1.5 rad/s.'],
        ['Step', '&Delta;t = 0.1 s &mdash; the deployment control period, not a tuning knob.'],
        ['Episode', '300 steps (30 s), or until the target leaves frame, or capture.'],
        ['Discount', '&gamma; = 0.99'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD)] for a, b in rows[1:]],
                widths=[34 * mm, 130 * mm])
    st += callout(
        'The simulator is deliberately unkind',
        'Measurements are delayed by a per-episode base drawn from '
        'U(150,&nbsp;350)&nbsp;ms plus per-frame jitter; boxes are jittered by '
        '~2&nbsp;px; detections drop out with a probability that rises from '
        '2&nbsp;% for a large box to 30&nbsp;% for a small one; and a two-state '
        'Markov chain produces multi-frame <i>burst</i> dropouts averaging half '
        'a second. A policy that only works on clean data never survives '
        'training here &mdash; which is the entire purpose.', 'note')

    st += h2('4.2&nbsp;&nbsp;The reward')
    st += para(
        'The reward is computed from <b>truth</b>, not from the detection. '
        'This is a deliberate asymmetry: the agent must act on a corrupted '
        'view but is graded on what actually happened, so it cannot earn '
        'reward by fooling its own sensor.')
    st += eq(
        'd_px  = pixel distance from the frame centre to the TRUE box centre\n'
        '\n'
        'tracking     r_track = (100 - d_px)/100          if d_px <= 100\n'
        '                     = -0.25 * (d_px - 100)/100  otherwise\n'
        '\n'
        'smoothness   - 0.05 * || a_t - a_{t-1} ||^2      (anti-judder)\n'
        'loss         - 5.0                               on the step the target exits frame\n'
        '\n'
        'INTERCEPT additionally:\n'
        '  shaping    + gamma * Phi(s_{t+1}) - Phi(s_t),  Phi(s) = -0.1 * range_m\n'
        '  capture    + 10.0                              on reaching 0.5 m in frame')
    st += para(
        'Two of those terms deserve comment. The <b>tracking</b> term is '
        'continuous and crosses zero at exactly 100&nbsp;px: inside that disc '
        'the agent is paid, outside it is charged, and the slope outside is '
        'gentler than inside so that a briefly-off-centre target does not '
        'dominate the return. The published version of this reward has a '
        '25-point <i>cliff</i> at the threshold &mdash; the reward jumps '
        'discontinuously from 0 to &minus;25 as the target crosses '
        '100&nbsp;px. That discontinuity is reproduced exactly in the '
        '<font face="Courier">faithful</font> variant, for comparison, and '
        'repaired in the one that flies.')
    st += para(
        'The <b>shaping</b> term uses the potential-based form '
        'F&nbsp;=&nbsp;&gamma;&Phi;(s\')&nbsp;&minus;&nbsp;&Phi;(s), which is '
        'provably unable to change the optimal policy &mdash; it only makes '
        'the gradient easier to find. The capture bonus, by contrast, '
        '<i>does</i> change the optimum, deliberately: capture is the task.')

    st += h2('4.3&nbsp;&nbsp;One environment step, in order')
    st += para(
        'The order of operations inside a step is itself a contract, because '
        'it determines what the policy can possibly know:')
    st += code(
        "1. the target moves                       (it acts first; we react)\n"
        "2. forward speed from the DELAYED measurement of the PREVIOUS step\n"
        "   -- the deployed depth rule also acts on detections, never on truth\n"
        "3. the follower integrates a first-order velocity lag (T_lag ~ 0.25 s)\n"
        "4. TRUE geometry is projected through our calibration -> u, v, w_px\n"
        "5. if the target is in frame, truth enters the delay line\n"
        "6. a measurement is SAMPLED from the delay line, Delta seconds old\n"
        "7. that measurement is corrupted (jitter / dropout / burst)\n"
        "8. record_action(a) THEN assemble(meas)  -- the ordering contract\n"
        "9. terminals are decided from TRUTH; the step budget is truncation")
    st += callout(
        'Truncation is not termination',
        'When an episode ends merely because the 300-step budget ran out, the '
        'value bootstrap must continue through the cut &mdash; the world did '
        'not end, the clock did. Conflating the two teaches the agent that '
        'time itself is fatal and systematically depresses the value of every '
        'long-horizon action. The two flags are mutually exclusive by '
        'construction here.', 'warn')

    st += h2('4.4&nbsp;&nbsp;The algorithm: TD3, and why not plain DDPG')
    st += para(
        'The source work uses DDPG. DDPG is an actor-critic method for '
        'continuous actions: a <i>critic</i> learns the value of a '
        'state&ndash;action pair, and an <i>actor</i> is trained to output the '
        'action the critic likes most. Its well-known failure is that the '
        'critic systematically <b>overestimates</b>: the actor exploits '
        'whatever the critic happens to over-value, the critic then trains on '
        'those inflated targets, and the pair can spiral. TD3 is DDPG plus '
        'three specific repairs.')
    st += figure('td3.png', 'The three differences between TD3 and DDPG.')
    st += para('As implemented, the update is exactly:')
    st += eq(
        "target action   a' = clip( mu'(s') + clip(eps, -c*s, +c*s), -s, +s )\n"
        "                     eps ~ N(0, (sigma_t * s)^2),  sigma_t = 0.2, c = 0.5\n"
        "\n"
        "label           y  = r + gamma * (1 - d) * min( Q'_1(s',a'), Q'_2(s',a') )\n"
        "\n"
        "critic loss     L_c = SUM_i  MSE( Q_i(s,a), y )        (sum, not mean)\n"
        "actor  loss     L_a = - mean( Q_1(s, mu(s)) )          (Q1 only, never the min)\n"
        "\n"
        "every d = 2 gradient steps:  actor step, then\n"
        "                theta' <- (1 - tau) theta' + tau theta,  tau = 0.005")
    st += para(
        'Three details in that listing are easy to get wrong and are worth '
        'naming. The smoothing noise exists <b>only inside the label</b> '
        '&mdash; it is never added to the action the aircraft takes. The '
        'actor is trained against <b>Q&#8321; alone</b>, not the minimum, '
        'because the minimum is a pessimistic estimate appropriate for '
        'labelling, not for ascent. And the soft update is counted in '
        '<i>gradient</i> steps and applied only on delayed steps, so the '
        'targets move at half the critic\'s rate.')
    st += para(
        'Three algorithm variants were trained head to head, so that the '
        'choice is a measurement rather than a preference:')
    rows = [
        ['Arm', 'What it is', 'Why it is in the comparison'],
        ['<b>td3</b>', 'All three repairs active.',
         'The candidate for deployment.'],
        ['<b>ddpg_repaired</b>', 'DDPG in the same environment, same '
         'observation, same reward &mdash; without the TD3 repairs.',
         'Isolates the algorithm from the environment: any difference is '
         'attributable to the repairs alone.'],
        ['<b>ddpg_faithful</b>', 'An exact reproduction of the published '
         'method, including its raw-pixel observation, its &plusmn;60 action '
         'scale, its reward cliff and its treatment of timeouts as terminal.',
         'Establishes what the original actually does, so improvements are '
         'measured against it rather than asserted.'],
        ['<b>sb3_td3</b>', 'An independent third-party TD3 implementation.',
         'Guards against a bug in our own trainer flattering our own result.'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[26 * mm, 66 * mm, 72 * mm])
    st += para(
        'Critically, <b>the arms are configuration, not separate code</b>. '
        'There is one trainer class; DDPG is TD3 with the twin critic, the '
        'smoothing and the delay switched off. Only five places in the whole '
        'package branch on the arm name. That is what makes the comparison '
        'honest: any difference between arms cannot be an accident of two '
        'divergent implementations.')
    rows = [
        ['Setting', 'td3', 'ddpg_repaired', 'ddpg_faithful'],
        ['Critics', '2', '1', '1'],
        ['Target smoothing &sigma;&#771; / clip', '0.2 / 0.5', 'off', 'off'],
        ['Policy delay', '2', '1', '1'],
        ['&tau; (soft update)', '0.005', '0.005', '0.001'],
        ['Learning rate (actor/critic)', '3e-4', '3e-4', '1e-3, clipnorm 1'],
        ['Hidden layers', '[256, 256]', '[256, 256]', '[16,16,16] / [32,32,32]'],
        ['Observation', '14-dim', '14-dim', '<b>2-dim raw pixels</b>'],
        ['Action scale', '1.0', '1.0', '<b>60.0</b>'],
        ['Exploration noise', 'Gaussian 0.3&rarr;0.05', 'same',
         'OU, post-scale, unclipped'],
        ['Timeout treated as terminal', 'no', 'no', '<b>yes</b>'],
        ['Curriculum / prefill', 'yes / 20 eps', 'yes / 20 eps', 'no / none'],
    ]
    st += table([rows[0]] + [[Paragraph(r[0], TD), Paragraph(r[1], TDC),
                              Paragraph(r[2], TDC), Paragraph(r[3], TDC)]
                             for r in rows[1:]],
                widths=[52 * mm, 28 * mm, 34 * mm, 50 * mm])
    st += callout(
        'A quantified explanation of why the faithful arm struggles',
        'Its Ornstein&ndash;Uhlenbeck exploration noise has a stationary '
        'standard deviation of about 0.55, and it is added <i>after</i> the '
        '&times;60 action scaling and never clipped. So its exploration '
        'perturbs an action range of &plusmn;60 by well under one per cent. '
        'It is not a bug in this reproduction &mdash; it is faithful to the '
        'original, and it is a measurable reason the method behaves as it '
        'does.', 'note')
    st += callout(
        'The faithful arm is quarantined, on purpose',
        'Because it emits a 2-dimensional raw-pixel observation and an action '
        'map of (image_x, image_y) scaled by 60, it is <i>structurally</i> '
        'incompatible with the flight stack. The flight loader refuses it by '
        'name. It exists to be measured, never to be flown.', 'warn')

    st += h2('4.5&nbsp;&nbsp;Training protocol')
    st += para(
        '<b>Curriculum.</b> Training begins against a stationary target and '
        'widens in four stages. The gate is a <i>rolling</i> one that can '
        'also step back down, and &mdash; importantly &mdash; it is measured '
        'on a frozen full-difficulty evaluation suite, not on the restricted '
        'training distribution, so "progress" cannot be an artefact of an '
        'easier exam.')
    rows = [
        ['Stage', 'Target motion families', 'Speed cap', 'Advance / demote'],
        ['0', 'static', '0.0 m/s', '&ge; 0.90 / &le; 0.70'],
        ['1', 'static, constant velocity', '0.4 m/s', '&ge; 0.90 / &le; 0.70'],
        ['2', 'constant velocity, vertical + horizontal oscillation',
         '0.6 m/s', '&ge; 0.90 / &le; 0.70'],
        ['3', 'all five, including aggressive', '1.0 m/s', '&mdash;'],
    ]
    st += table([rows[0]] + [[Paragraph(r[0], TDB), Paragraph(r[1], TD),
                              Paragraph(r[2], TDC), Paragraph(r[3], TDC)]
                             for r in rows[1:]],
                widths=[14 * mm, 88 * mm, 22 * mm, 40 * mm])
    st += bullets([
        '<b>Warm start.</b> The replay buffer is pre-filled with 20 episodes '
        'flown by the P-controller baseline &mdash; a few thousand '
        'transitions &mdash; so the agent begins from "someone who can '
        'already roughly chase" rather than from noise. The demonstration '
        'seeds are drawn from a block disjoint from both training and '
        'evaluation.',
        '<b>Exploration.</b> Gaussian noise decaying linearly from '
        '&sigma;&nbsp;=&nbsp;0.3 to 0.05, added <i>before</i> the clip to '
        '[&minus;1,&nbsp;1], on top of a 1,000-step uniform-random warm-up.',
        '<b>Selection.</b> The saved checkpoint is the one with the best '
        'evaluation time-in-view <i>on the moving families</i> &mdash; not '
        'the most recent, and not one flattered by static-target scores. '
        'Return breaks ties within a 0.005 band.',
        '<b>Stopping.</b> Training halts on an evaluation plateau: five '
        'consecutive evaluations without improvement, and only after the '
        'curriculum has reached its final stage. In practice runs stopped '
        'between 60,000 and 95,000 steps against a 200,000-step budget.',
        '<b>Reporting.</b> The headline numbers come from a <b>held-out seed '
        'block</b> 500,000 away from the selection suite, scored on the '
        'selected checkpoint rather than the final weights &mdash; because '
        'selection, the curriculum and the plateau rule all consumed the '
        'selection suite.',
    ])
    st += h2('4.6&nbsp;&nbsp;The acceptance rule')
    st += para(
        'A result is accepted only if the <b>lower</b> bound of the arm\'s '
        'confidence interval exceeds the <b>upper</b> bound of the '
        'P-controller baseline\'s interval, on the <i>moving</i> target '
        'families. Comparing means would let noise decide; comparing on '
        'static targets would flatter everything. Five seeds per arm make the '
        'interval meaningful.')
    st += callout(
        'Result: ACCEPT',
        'All reinforcement-learning arms beat the classical P-controller on '
        'moving targets under the interval rule. The deployable follow policy '
        'scored a held-out time-in-view of 99.7&nbsp;%; the intercept policy '
        '98.1&nbsp;%.', 'good')
    return st


# --------------------------------------------------------------------------- #
# 5. Tier B
# --------------------------------------------------------------------------- #

def sec5():
    st = [PageBreak()]
    st += h1('5', 'Tier B &mdash; does the maths match physics?',
             'Gazebo Harmonic, headless and in lockstep.')
    st += para(
        'Tier&nbsp;A approximates flight with a first-order velocity lag. '
        'Gazebo simulates rotors, inertia and drag. If a policy only works in '
        'the approximation it is worthless, so Tier&nbsp;B is the referee '
        'between the two worlds. It runs the same environment interface, the '
        'same observation module and the same forward law &mdash; only the '
        'body dynamics are replaced.')
    st += h2('5.1&nbsp;&nbsp;The three checks')
    rows = [
        ['Check', 'What it does', 'Result'],
        ['<b>Sign test</b>', 'Commands one axis at a time and asserts the '
         'direction the target moves in the image, through the entire chain.',
         '<font color="#1f8a4c"><b>PASS</b></font>'],
        ['<b>Lag calibration</b>', 'Measures the simulated aircraft\'s '
         'velocity step response.', '0.54&ndash;0.84 s measured'],
        ['<b>A&harr;B replay gate</b>', 'Replays identical command sequences '
         'in Tier&nbsp;A and in Gazebo and compares the resulting image-space '
         'trajectories pixel by pixel.',
         '<font color="#1f8a4c"><b>8.0 px median vs 40 px tolerance</b></font>'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[30 * mm, 90 * mm, 44 * mm])
    st += para(
        'The replay gate is the certificate that makes the whole ladder '
        'meaningful: it says the world the policy trained in agrees with '
        'rigid-body physics to within a few pixels of image-space truth.')
    st += callout(
        'Two defects this gate found',
        ['The gate could not run at all until a reset bug was fixed: after '
         'teleporting both aircraft into place, the world was given only '
         '<b>12 milliseconds</b> to settle before the first comparison, so '
         'the very first sample was taken mid-transient. It now holds for '
         '1.5&nbsp;s while re-arming the controllers.',
         'Fixing that exposed a second, subtler defect. The delay queue '
         'compared timestamps with a bare <font face="Courier">&lt;=</font>. '
         'When the latency is an exact multiple of the control period, the '
         'comparison lands precisely on a stamp and its outcome depends on '
         'accumulated floating-point error &mdash; error that <i>grows</i> '
         'with simulated time. Starting the clock later silently flipped the '
         'comparison from inclusive to exclusive and froze the observation at '
         'its pre-rolled value. A nanosecond of slack makes the knife edge '
         'deterministic.'], 'warn')
    st += para(
        'Tier&nbsp;B has no window and no camera, by design. Its measurements '
        'come from projecting ground truth through the same pinhole model the '
        'real camera obeys, then delaying and corrupting it exactly as '
        'Tier&nbsp;A does. It also runs faster than real time in lockstep, so '
        'a rendered view would be an unwatchable stutter as well as an '
        'irrelevance.')
    return st


# --------------------------------------------------------------------------- #
# 6. Tier C
# --------------------------------------------------------------------------- #

def sec6():
    st = [PageBreak()]
    st += h1('6', 'Tier C &mdash; does it survive real vision?',
             'Unreal Engine, a rendered warehouse, and a detector trained on it.')
    st += para(
        'Tier&nbsp;C is the only tier with pixels, because in Tier&nbsp;C the '
        'pixels are the experiment. It answers the question the other two '
        'cannot: does the policy still work when its input comes from an '
        'actual object detector looking at an actual image?')

    st += h2('6.1&nbsp;&nbsp;A warehouse, built at run time')
    st += para(
        'The deployment target is indoor flight, so the scene is an enclosed '
        'warehouse &mdash; floor, ceiling, four walls, shelving racks down '
        'both sides of a central aisle, loaded with crates, boxes and '
        'barrels, lit by ceiling lights whose intensity is genuinely driven '
        'by the requested lighting class. This matters more than it sounds: '
        'a detector inherits the domain it is trained on, and a detector '
        'trained on an outdoor test level will not find a drone in a '
        'warehouse.')

    st += h2('6.2&nbsp;&nbsp;A dataset with perfect labels')
    st += para(
        'The target is placed on a grid of ranges (0.5&ndash;6 m), azimuths, '
        'elevations and headings, in three lighting classes. Each frame is '
        'saved with a label taken from the engine\'s own ground truth &mdash; '
        'so the labels are exact by construction, with no human error and no '
        'annotation cost. That produced <b>3,003 labelled frames</b>.')

    st += h2('6.3&nbsp;&nbsp;The phantom lesson')
    st += para(
        'The first detector (v1) scored a precision of 0.995 and looked '
        'excellent. It was not. Its validation set contained only images '
        '<i>with a drone in them</i>, so "precision" had never been tested '
        'against the one thing that matters in flight: an empty room. The '
        'moment the chase flew off the aisle centre line, v1 produced large, '
        'high-confidence "drone" boxes on dark walls and shelf shadows &mdash; '
        'and because range is inferred from box width, a huge phantom box '
        'reads as a target a few centimetres away. Every hallucination became '
        'a fake capture.')
    st += callout(
        'A detector that has never seen a drone-free image cannot say "no drone"',
        'The fix was to add <b>360 verified target-free frames</b> (~11&nbsp;% '
        'of the set) captured by a patrol that hides the target under the '
        'floor and photographs walls, racks and dark corners. Retrained, v2 '
        'holds precision 0.996 <i>with those frames inside the exam</i>, and '
        'all three frames that fooled v1 come back clean.', 'good')
    rows = [
        ['Detector', 'Precision', 'Recall', 'mAP50', 'Backgrounds in validation?'],
        ['v1', '0.995', '0.711', '0.732', '<font color="#b8520f"><b>none</b></font>'],
        ['v2', '0.996', '0.705', '0.724', '<font color="#1f8a4c"><b>72 frames</b></font>'],
    ]
    st += table([rows[0]] + [[Paragraph(c, TD) for c in r] for r in rows[1:]],
                widths=[26 * mm, 26 * mm, 24 * mm, 24 * mm, 64 * mm])
    st += para(
        'Recall of ~0.70 is not a defect to be tuned away: the misses are '
        'targets below roughly 30 px wide, i.e. beyond about 4 m at extreme '
        'angles. That is a genuine physical limit of a 0.098 m object in a '
        '960 px frame, and the chase policy\'s job is to keep the target '
        'larger and more central than that.')

    st += h2('6.4&nbsp;&nbsp;Vision in the loop')
    st += para(
        'The final Tier&nbsp;C test closes the loop with nothing mocked: '
        'rendered frame &rarr; the trained YOLO &rarr; the same observation '
        'assembler &rarr; the trained actor &rarr; stick commands &rarr; the '
        'engine flies &rarr; next frame.')
    rows = [
        ['Mission', 'Result', 'Reading'],
        ['<b>Follow</b> (the deployment task)',
         '<font color="#1f8a4c"><b>100 % detection, 5 &times; 300 steps</b></font>',
         'The policy held the target inside the detectable envelope for 1,500 '
         'consecutive frames. This is the mission the aircraft will fly.'],
        ['<b>Intercept</b>', 'verified capture: 3.6 m chase, caught at 0.48 m',
         'Genuine &mdash; the target\'s rotors are inside the detection box on '
         'the saved frame. Other episodes tracked perfectly but ran out the '
         '300-step clock.'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[38 * mm, 48 * mm, 78 * mm])
    st += callout(
        'Why the intercept capture rate is a lower bound, not a score',
        'The evaluation loop is rendering-bound: each tick costs a render plus '
        'a YOLO inference, so the aircraft receives roughly two commands per '
        'second and spends most of the wall clock decelerating. The closing '
        'law also tapers to 0.15&nbsp;m/s near the target. The result is that '
        'the last metre eats the step budget. The deployed stack streams '
        'commands continuously at 10&nbsp;Hz and does not have this problem.',
        'note')
    st += para(
        'Getting an honest number here took three separate corrections, each '
        'of which had produced a <i>convincing</i> false positive: episodes '
        'that "captured" on their first frame (the previous episode\'s '
        'momentum carried the aircraft onto a freshly-placed target), '
        'captures on giant phantom boxes (Section&nbsp;6.3), and an early '
        '80&nbsp;% capture rate that was all three artefacts stacked. The '
        'working rule that emerged is simple: <b>a number is not a result '
        'until the frame behind it has been looked at.</b>')
    return st


# --------------------------------------------------------------------------- #
# 7. flight stack
# --------------------------------------------------------------------------- #

def sec7():
    st = [PageBreak()]
    st += h1('7', 'The flight stack',
             'Six ROS 2 nodes that put the trained policy on the aircraft.')
    st += para(
        'The flight software is deliberately split into six small nodes with '
        'one responsibility each. The split is not decoration: it means the '
        'observation definition can change without touching perception, the '
        'policy can be swapped for a hand-written controller without touching '
        'anything else, and &mdash; most importantly &mdash; the component '
        'that restrains the aircraft runs in a <i>different process</i> from '
        'the components that command it.')
    st += figure('flight_graph.png',
                 'The flight graph. Exactly one node publishes /cmd_vel.')

    st += h2('7.1&nbsp;&nbsp;chase_state &mdash; the observation clock')
    st += para(
        'Owns the 10&nbsp;Hz control tick. On each tick it records the action '
        'that was actually <i>sent</i>, assembles the 14-dim observation from '
        'the newest detection, and publishes it. Two details are load-bearing:')
    st += bullets([
        '<b>The tick is a timer, not a detection callback.</b> Detections '
        'stop arriving precisely when the target is lost &mdash; which is '
        'exactly when the staleness counter must keep climbing. A '
        'detection-driven clock freezes staleness at the moment it matters.',
        '<b>Detections are consumed once.</b> Deciding freshness by age alone '
        'lets a single detection be re-used on many ticks, which both '
        'double-counts it as "visible" and stalls staleness at zero. This was '
        'a real defect, caught by the integration tests in Section&nbsp;7.7.',
    ])

    st += h2('7.2&nbsp;&nbsp;chase_policy &mdash; inference only')
    st += para(
        'Loads the actor and runs one forward pass per tick. No noise, no '
        'learning, no exploration: the aircraft flies a frozen checkpoint, '
        'because an updating policy on a live aircraft has no safety '
        'argument. Inference uses the exported ONNX graph; the torch bundle '
        'is still read, because it is the only artifact carrying the '
        'observation hash and action map that authorise the graph to fly at '
        'all. A mismatch is a hard refusal, not a warning.')
    st += para(
        'The same node can fly a hand-written P-controller instead '
        '(<font face="Courier">baseline:=p_controller</font>), which is how '
        'the flight ladder begins &mdash; see Section&nbsp;10.')

    st += h2('7.3&nbsp;&nbsp;chase_depth_rule &mdash; the forward axis')
    st += para(
        'Imports <font face="Courier">chase_gym.env.forward_command</font> '
        'directly: the same function Tier&nbsp;A stepped and Tier&nbsp;B '
        'replayed. It publishes metres per second; converting to sticks is '
        'the mixer\'s job. It ignores the detector\'s own published range '
        'entirely, for a reason worth stating plainly.')
    st += callout(
        'A silent 1.84&times; range error, designed out',
        'The detector node defaults its reference width to 0.180&nbsp;m (the '
        '<i>propeller span</i>) while the trained policy\'s config uses '
        '0.098&nbsp;m (the <i>body width</i>). Both are correct measurements '
        'of the same aircraft, and mixing them scales every range estimate by '
        '1.84. This stack therefore derives range from the box width with the '
        'checkpoint\'s own value, and the launch file sets the detector to '
        'match so that a human reading the topic is not misled either.', 'warn')

    st += h2('7.4&nbsp;&nbsp;chase_behaviour &mdash; the mission state machine')
    st += figure('behaviour.png',
                 'SEARCH / TRACK / REACQUIRE / PATROL. REACQUIRE holds '
                 'position before it rotates &mdash; it does not coast toward '
                 'a target it can no longer see.')
    st += para(
        'Every timeout in that diagram is a <b>[D] design choice</b>. The '
        'source material names exactly these four knobs &mdash; lost-frame '
        'count, loss timeout, search rate, patrol pattern &mdash; and fixes '
        'none of them. The values shipped are conservative starting points, '
        'and they are the first thing to tune in flight. One of them does '
        'have provenance: the 1.0&nbsp;s loss timeout matches the staleness '
        'normaliser baked into the checkpoint\'s observation spec, i.e. the '
        'value the policy was actually trained against.')
    st += para(
        'The search rate is chosen against the <i>detector</i>, not against '
        'the airframe: at a quarter stick the aircraft sweeps its '
        '55&deg; field of view in about 2.6&nbsp;s, which gives the detector '
        'several frames on anything it crosses. Sweeping faster than the '
        'detector can see is the classic way to rotate straight past the '
        'thing you are looking for.')

    st += h2('7.5&nbsp;&nbsp;chase_mixer &mdash; exclusive axis ownership')
    st += para(
        'Two controllers writing one axis is the classic route to an '
        'uncontrollable aircraft, so ownership is structural rather than a '
        'matter of discipline:')
    st += eq(
        'linear.z   (up/down)       chase_policy        learned\n'
        'angular.z  (yaw)           chase_policy        learned  [see precedence]\n'
        'linear.x   (forward/back)  chase_depth_rule    hand-coded\n'
        'linear.y   (lateral)       UNUSED -- always exactly 0.0')
    st += para(
        '<b>Precedence.</b> The design set never defined how the behaviour '
        'mode combines with the policy, so this stack defines it: outside '
        'TRACK the behaviour node owns yaw and the policy is ignored. The '
        'reasoning is that outside TRACK there is no target in frame, the '
        'observation errors are therefore zeroed, and the policy\'s output '
        'carries no information about where to point.')
    st += para(
        '<b>Units.</b> The policy axes pass through <i>unscaled</i>: the '
        'checkpoint declares a stick range of [&minus;1,&nbsp;1], '
        '<font face="Courier">/cmd_vel</font> consumes exactly that, and the '
        'driver scales to the SDK\'s &plusmn;100 itself. Only the forward '
        'axis is converted, from m/s to sticks &mdash; and that conversion '
        'needs a number this project does not yet have.')
    st += callout(
        '[!] The one unmeasured number in the control path',
        'Converting the depth rule\'s metres per second into a stick requires '
        'the airframe\'s full-stick forward speed. It has never been measured '
        'on this aircraft; the simulator assumed 1.5&nbsp;m/s and the launch '
        'default repeats that assumption. Setting it <i>below</i> the truth '
        'makes the aircraft fly faster than intended, so it must err high '
        'until a step-response test replaces it. This is item&nbsp;1 of the '
        'bring-up procedure.', 'warn')

    st += h2('7.6&nbsp;&nbsp;chase_safety &mdash; the supervisor')
    st += para(
        'The only publisher of <font face="Courier">/cmd_vel</font>, running '
        'in its own process so that a controller bug cannot disable the thing '
        'that restrains it. Covered in full in Section&nbsp;8.')

    st += h2('7.7&nbsp;&nbsp;How it is tested without an aircraft')
    st += para(
        'Seven integration tests start all six real nodes in one process, '
        'inject synthetic detections and assert on what reaches '
        '<font face="Courier">/cmd_vel</font>:')
    st += bullets([
        'A good detection while <b>disarmed</b> produces exactly zero on every axis.',
        'The lateral axis is <b>always</b> zero, structurally.',
        'No axis ever exceeds the speed cap, even at detector saturation.',
        '<b>Sign test, yaw:</b> a target right of centre yaws right.',
        '<b>Sign test, vertical:</b> a target below centre descends.',
        'A lost target holds the forward axis at zero &mdash; no coasting.',
        'A far target commands forward; a close one backs off.',
    ])
    st += callout(
        'These tests earned their keep immediately',
        'The target-loss test failed on first run and exposed the '
        'consume-once defect described in Section&nbsp;7.1 &mdash; a bug that '
        'would have made the aircraft believe it could still see a target it '
        'had lost, for as long as the timeout allowed.', 'good')
    return st


# --------------------------------------------------------------------------- #
# 8. safety
# --------------------------------------------------------------------------- #

def sec8():
    st = [PageBreak()]
    st += h1('8', 'Safety',
             'What actually restrains this aircraft &mdash; and what cannot.')
    st += para(
        'The supervisor is the last node before the driver and the only '
        'publisher of the command topic. It is a separate process from the '
        'mixer on purpose: isolation means a bug in a controller cannot '
        'switch off the thing that limits it.')
    rows = [
        ['Mechanism', 'Behaviour', 'Default'],
        ['<b>Engage gate</b>', 'Until an operator arms the stack the command '
         'on the wire is exactly zero. The policy runs, publishes and is '
         'recorded from the first second &mdash; it simply does not reach the '
         'aircraft.', 'starts <b>DISARMED</b>'],
        ['<b>Speed cap</b>', 'Every axis clamped before it is sent. On this '
         'airframe this <i>is</i> the safety envelope.', '0.6 of full stick [D]'],
        ['<b>Dead-man feed</b>', 'Publishes at 20 Hz with hold-last-command '
         'semantics, so the driver\'s 0.35 s dead-man is always fed and a '
         'stalled upstream node stops the aircraft.', '20 Hz [V]'],
        ['<b>Auto-disarm</b>', 'Latches disarmed and commands a land on any '
         'of: link loss, low battery, prolonged target loss, altitude breach, '
         'or a stale mixer.', 'see below'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[30 * mm, 100 * mm, 34 * mm])
    st += callout(
        'The honest limitation: there is no geofence',
        ['This method estimates no absolute position. The aircraft knows its '
         '<i>height</i> &mdash; from the time-of-flight sensor and the SDK\'s '
         'own estimate &mdash; but nothing whatsoever about where it is '
         'horizontally. A lateral geofence is therefore not implementable, '
         'and this stack does not pretend to offer one.',
         'What actually keeps the aircraft in the room is the speed cap, a '
         'short flight volume chosen by the operator, and a human holding the '
         'abort. Those should be described that way rather than overclaimed. '
         'The altitude limits <i>are</i> real, because height is measured. '
         'There is also no hardware kill line on a Tello: the equivalents are '
         'the emergency motor-cut and pulling the battery.'], 'warn')
    st += h2('8.1&nbsp;&nbsp;Thresholds, and where they come from')
    rows = [
        ['Threshold', 'Default', 'Provenance'],
        ['Speed cap', '0.6 stick', '[D] conservative first-flight value'],
        ['Battery floor', '20 %', '[D] &mdash; the source material names the rule, no number'],
        ['Link-loss timeout', '1.0 s', '[D]'],
        ['Stale-command timeout', '0.30 s', '[V] must stay under the driver\'s 0.35 s'],
        ['Prolonged target loss', '30 s', '[D]'],
        ['Altitude window', '0.3 &ndash; 2.5 m', '[D], but enforceable &mdash; height is measured'],
        ['Forward stick scale', '1.5 m/s', '<b>[!] UNMEASURED</b> &mdash; see Section 7.5'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TDC), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[46 * mm, 30 * mm, 88 * mm])
    st += para(
        'Not one of these numbers is inherited from the source material, '
        'which names every rule and fixes no value. They are stated here as '
        'choices so that they can be argued with, rather than buried as '
        'defaults.')
    return st


# --------------------------------------------------------------------------- #
# 9 + 10 + appendix
# --------------------------------------------------------------------------- #

def sec9():
    st = [PageBreak()]
    st += h1('9', 'Results, and what is not yet proven',
             'The honest ledger.')
    st += figure('results.png', 'The measured results across the three tiers.')
    rows = [
        ['Claim', 'Evidence'],
        ['The behaviour is learnable, and RL beats the classical baseline',
         'Tier A matrix, 20 runs, interval rule &rarr; <b>ACCEPT</b>'],
        ['The training world matches real physics',
         'Tier B A&harr;B gate on real Gazebo, <b>8.0 px</b> median vs 40 px'],
        ['A detector can be built entirely in simulation, and made honest',
         'v2 precision <b>0.996</b> with drone-free frames in the exam'],
        ['The full vision loop works with nothing mocked',
         'Follow: <b>100 %</b> detection over 1,500 frames'],
        ['The intercept mission can genuinely complete',
         'Verified capture at <b>0.48 m</b>, frame on record'],
        ['The flight stack behaves correctly without hardware',
         '<b>7</b> integration tests on all six nodes; <b>94</b> tests total'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD)] for a, b in rows[1:]],
                widths=[92 * mm, 72 * mm])
    st += h2('9.1&nbsp;&nbsp;What is NOT proven')
    st += para(
        'This is the more important half of the section, and it is the reason '
        'the system is not yet cleared to fly the mission.')
    rows = [
        ['Gap', 'Why it matters', 'What closes it'],
        ['<b>The detector has never seen reality.</b> All 3,363 training '
         'images are rendered.',
         'Rendered-to-real is the single most common failure surface for '
         'drone detectors. A precision of 0.996 on Unreal frames says nothing '
         'about a real warehouse camera.',
         'Record real footage of the target, mix it in, retrain, re-validate.'],
        ['<b>The aircraft\'s real dynamics are unmeasured.</b> The policy '
         'trained against a 0.25 s velocity lag placeholder.',
         'Tier B <i>measured</i> 0.54&ndash;0.84 s in physics alone. A chase '
         'policy is a tight feedback loop; feeding it two to three times the '
         'delay it learned on can oscillate.',
         'Step-response tests, then recalibrate Tier B and fine-tune if the '
         'gap demands it.'],
        ['<b>The forward stick scale is a guess.</b>',
         'It sets the magnitude of the only axis that decides closing energy.',
         'The same step-response test.'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[50 * mm, 62 * mm, 52 * mm])
    st += callout(
        'The summary judgement',
        'What exists is a <b>validated policy and a validated physics model</b>, '
        'plus a detector and a command path that have <b>not yet met '
        'reality</b>. That is a strong place to begin the hardware phase &mdash; '
        'and not a licence to skip it.', 'note')
    return st


def sec10():
    st = [PageBreak()]
    st += h1('10', 'Bring-up procedure',
             'The flight ladder, in the order it must be climbed.')
    st += para(
        'Each rung produces a measurement that the next rung depends on. The '
        'safety supervisor and a human on the abort are live throughout.')
    rows = [
        ['#', 'Step', 'Produces', 'Needs'],
        ['1', '<b>Measure the aircraft.</b> Step-response tests for command '
         'lag and full-stick speed on each axis.',
         'The two numbers every placeholder in this document is waiting for.',
         'The drone, a tape measure, no RL'],
        ['2', '<b>Recalibrate Tier B</b> to the measured lag, re-run the '
         'A&harr;B gate, fine-tune the policy in Gazebo if the gap is large.',
         'A policy trained against the real delay.', 'Gazebo only'],
        ['3', '<b>Collect real frames</b> of the target at known ranges; mix '
         'into the dataset; retrain the detector; re-validate on real images.',
         'A detector that has seen reality.', 'Manual flight'],
        ['4', '<b>Fly the P-controller first</b> '
         '(<font face="Courier">baseline:=p_controller</font>). Same graph, '
         'same safety chain, no learned policy.',
         'Proof the whole chain works, plus the baseline number that makes '
         'the RL result interpretable.', 'Full stack, tethered'],
        ['5', '<b>Fly the learned policy</b> against a stationary target, '
         'then a slow one, then the full scenarios.',
         'The result.', 'Full stack'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TDB), Paragraph(b, TD),
                              Paragraph(c, TD), Paragraph(d, TD)]
                             for a, b, c, d in rows[1:]],
                widths=[7 * mm, 62 * mm, 55 * mm, 40 * mm])
    st += h2('10.1&nbsp;&nbsp;Before every flight')
    st += bullets([
        'Launch with <font face="Courier">control:=false</font> so the '
        'keyboard teleop node cannot fight the policy for the same command '
        'register &mdash; they write the same register and last writer wins.',
        'Confirm the stack comes up <b>DISARMED</b> and that '
        '<font face="Courier">/cmd_vel</font> is exactly zero before arming.',
        'Confirm <font face="Courier">/emergency</font> is wired to something '
        'you can physically reach.',
        'Check the detector\'s reference width matches the policy\'s '
        '(Section&nbsp;7.3) &mdash; the launch file does this, but verify it.',
        'Run the sign test for the tier you are about to use. Three '
        'simulators plus one aircraft is four chances for a silent flip.',
    ])
    st += h2('10.2&nbsp;&nbsp;Commands')
    st += code(
        "# build\n"
        "colcon build --packages-select chase_msgs chase_flight tello_msg\n"
        "\n"
        "# fly the hand-written baseline first (ladder step 4)\n"
        "ros2 launch chase_flight chase_flight.launch.py \\\n"
        "     baseline:=p_controller weights:=<detector.pt>\n"
        "\n"
        "# fly the learned policy (ladder step 5)\n"
        "ros2 launch chase_flight chase_flight.launch.py \\\n"
        "     checkpoint:=<bundle.pt> weights:=<detector.pt>\n"
        "\n"
        "# arm / abort / motor cut\n"
        "ros2 topic pub -1 /chase/arm   std_msgs/Bool  'data: true'\n"
        "ros2 topic pub -1 /chase/abort std_msgs/Empty '{}'\n"
        "ros2 topic pub -1 /emergency   std_msgs/Empty '{}'\n"
        "\n"
        "# watch what the supervisor is deciding\n"
        "ros2 topic echo /chase/safety")
    return st


def appendix():
    st = [PageBreak()]
    st += h1('A', 'Reference tables')
    st += h2('A.1&nbsp;&nbsp;Where the code lives')
    rows = [
        ['Package', 'Role'],
        ['<font face="Courier">chase_gym</font>', 'THE contract: observation '
         'assembler, forward law, reward, kinematics, corruption, target '
         'motion, baselines, constants. Imported by every tier.'],
        ['<font face="Courier">chase_train</font>', 'The trainer: TD3/DDPG '
         'arms, checkpoint bundles, ONNX export, the run matrix and its '
         'acceptance report.'],
        ['<font face="Courier">chase_eval</font>', 'Held-out evaluation and '
         'the selection metric.'],
        ['<font face="Courier">chase_sim_gz</font>', 'Tier B: Gazebo backend, '
         'oracle detector, latency shim, sign test, A&harr;B replay gate.'],
        ['<font face="Courier">sim/tier_c_airsim</font>', 'Tier C: engine '
         'adapter, warehouse builder, dataset factory, detector trainer, '
         'vision-in-the-loop evaluation.'],
        ['<font face="Courier">chase_detector</font>', 'The deployed YOLO node.'],
        ['<font face="Courier">chase_flight</font>', '<b>The flight stack</b> '
         '&mdash; the six nodes of Section 7.'],
        ['<font face="Courier">chase_msgs</font>', 'Interface messages: '
         'DroneDetection, TrackingState, PolicyAction, FlightMode, SafetyStatus.'],
        ['<font face="Courier">tello</font>', 'The driver: video, telemetry, '
         'RC output, the 0.35 s dead-man.'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TD), Paragraph(b, TD)] for a, b in rows[1:]],
                widths=[44 * mm, 120 * mm])

    st += h2('A.2&nbsp;&nbsp;Topics')
    rows = [
        ['Topic', 'Type', 'Published by'],
        ['/image_raw', 'sensor_msgs/Image', 'tello driver, 960&times;720 bgr8'],
        ['/chase/detection', 'chase_msgs/DroneDetection', 'chase_detector, every frame'],
        ['/chase/observation', 'chase_msgs/TrackingState', 'chase_state, 10 Hz'],
        ['/chase/action', 'chase_msgs/PolicyAction', 'chase_policy'],
        ['/chase/forward_cmd', 'geometry_msgs/TwistStamped', 'chase_depth_rule, m/s'],
        ['/chase/mode', 'chase_msgs/FlightMode', 'chase_behaviour'],
        ['/chase/cmd_raw', 'geometry_msgs/TwistStamped', 'chase_mixer, sticks'],
        ['/cmd_vel', 'geometry_msgs/Twist', '<b>chase_safety ONLY</b>, 20 Hz'],
        ['/chase/action_sent', 'chase_msgs/PolicyAction', 'chase_safety &rarr; chase_state'],
        ['/chase/safety', 'chase_msgs/SafetyStatus', 'chase_safety'],
        ['/chase/arm, /chase/abort', 'std_msgs/Bool, Empty', 'the operator'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TDC), Paragraph(b, TDC), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[42 * mm, 52 * mm, 70 * mm])

    st += h2('A.3&nbsp;&nbsp;Key constants')
    rows = [
        ['Constant', 'Value', 'Meaning'],
        ['FX, FY', '919.42 px', 'Focal length, from the calibration file'],
        ['FRAME_W &times; H', '960 &times; 720', 'Calibration frame'],
        ['HFOV / VFOV', '55.1&deg; / 42.8&deg;', 'Derived, not the ~82&deg; usually quoted'],
        ['DT', '0.1 s', 'Control period = simulator step'],
        ['K_ACTION_HISTORY', '4', 'ceil(0.35 s &times; 10 Hz)'],
        ['TELLO_BODY_W', '0.098 m', 'The reference width the policy uses'],
        ['TELLO_PROP_SPAN', '0.180 m', 'The detector\'s default &mdash; 1.84&times; larger'],
        ['LATENCY_RANGE_S', '0.150 &ndash; 0.350 s', 'Measured video-link latency'],
        ['DEADMAN_S', '0.35 s', 'Driver zeroes a staler command'],
    ]
    st += table([rows[0]] + [[Paragraph(a, TDC), Paragraph(b, TDC), Paragraph(c, TD)]
                             for a, b, c in rows[1:]],
                widths=[42 * mm, 34 * mm, 88 * mm])

    st.append(Spacer(1, 6))
    st.append(HRFlowable(width='100%', thickness=1.0, color=LINE,
                         spaceBefore=6, spaceAfter=8))
    st += para(
        '<font color="#5b6472"><i>Every figure in this document was generated '
        'from source by <font face="Courier">docs/make_chase_figures.py</font>, '
        'and the document itself by <font face="Courier">'
        'docs/make_chase_reference.py</font>. Numbers marked [D] are design '
        'choices, [!] are unmeasured placeholders, and [V] are verified '
        'against code or measurement.</i></font>')
    return st


def build():
    doc = BaseDocTemplate(
        OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='Vision-Based Drone Chase — Technical Reference',
        author='Roey Turgeman')
    frame = Frame(MARGIN, MARGIN, CW, PH - 2 * MARGIN, id='f')
    doc.addPageTemplates([
        PageTemplate(id='cover', frames=[frame], onPage=cover_page),
        PageTemplate(id='body', frames=[frame], onPage=body_page),
    ])
    story = []
    for fn in (build_cover, build_contents, sec1, sec2, sec3, sec4, sec5,
               sec6, sec7, sec8, sec9, sec10, appendix):
        story += fn()
    doc.build(story)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
