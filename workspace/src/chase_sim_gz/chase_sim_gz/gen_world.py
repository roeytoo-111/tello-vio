"""Generates worlds/chase.sdf -- two Tello-parameterised quadrotors.

One source of truth for the vehicle parameterisation instead of two
hand-synchronised 150-line SDF blocks. Run after changing any constant:

    python3 -m chase_sim_gz.gen_world

Plugin structure follows the VERIFIED gz-sim8 example world
(multicopter_velocity_control.sdf, read at source 2026-09-17):
MulticopterMotorModel per rotor (ccw/ccw/cw/cw diagonal pairs) +
MulticopterVelocityControl (subscribes <ns>/cmd_vel, gz.msgs.Twist,
body-frame linear + yaw rate -- semantically the Tello stick contract) +
OdometryPublisher (sim-time throttled: THE deterministic lockstep pose
source; pose/info is wall-clock throttled and unusable at high RTF).

CALIBRATION STATUS (sim_training_architecture.md 3.1/3.2): mass and body
extent are [V manufacturer]; inertia is the box estimate; motor and
controller constants are DERIVED STARTING POINTS (hover at ~half
maxRotVelocity; gains scaled from the X3 example by mass/inertia -- the
controller normalises internally, RotorS heritage). They are calibrated
against the measured Phase-1 step response, never presented as truth.
Run sign_test.py before anything (gz issue #2657: quadrotor yaw sign
reported inverted, open and unconfirmed upstream).
"""
import math
import os

from chase_gym import constants as C

# ---- Tello parameterisation (one place) ---------------------------------
MASS_BASE = 0.060          # kg; + 4 rotors = 0.080 total [V]
MASS_ROTOR = 0.005
# Body extents come from the shared constants (depth ~= width [V]).
BODY = (C.TELLO_BODY_W, C.TELLO_BODY_W, C.TELLO_BODY_H)
ARM = 0.045                # rotor offset on x and y, m (prop span 0.18 [V])
ROTOR_Z = 0.024
ROTOR_RADIUS = 0.038
MAX_ROT_VEL = 2500.0       # rad/s
HOVER_FRACTION = 0.5       # hover at ~half throttle -> motorConstant
MOTOR_CONSTANT = (0.080 * 9.81 / 4.0) / (HOVER_FRACTION * MAX_ROT_VEL) ** 2
MOMENT_CONSTANT = 0.016    # X3's thrust->moment ratio (start value)
TIME_CONSTANT_UP = 0.0125  # X3 (start values; calibrate to T_lag, 3.2)
TIME_CONSTANT_DOWN = 0.025
# Rotor drag force = coeff * rotor speed * airspeed. The X3's 8.06428e-5
# copied unchanged gives this 80 g vehicle ~36x the X3's drag deceleration
# (hover rotor speed 1250 vs 656 rad/s, mass 0.08 vs 1.5 kg): measured on
# gz 8.15, yaw rate stalled at 0.118 of 0.5 rad/s and forward speed at
# ~0.04 of 0.3 m/s. Scale to keep the X3's coeff*speed/mass per rotor.
X3_DRAG_PER_MASS = 8.06428e-05 * 656.0 / 1.5
ROTOR_DRAG = X3_DRAG_PER_MASS * 0.080 / (HOVER_FRACTION * MAX_ROT_VEL)
ROLLING_MOMENT = 1e-06 * ROTOR_DRAG / 8.06428e-05
ROTOR_VELOCITY_SLOWDOWN = 10
# Controller gains: X3's scaled by mass ratio (0.080/1.5) for the linear
# loop and by box-inertia ratio (~1/350) for the attitude loops.
VELOCITY_GAIN = (0.15, 0.15, 0.15)
ATTITUDE_GAIN = (0.006, 0.009, 0.0004)
ANGULAR_RATE_GAIN = (0.0012, 0.0016, 0.0005)
MAX_LIN_ACC = (2.0, 2.0, 2.0)

Ixx = MASS_BASE / 12.0 * (BODY[1] ** 2 + BODY[2] ** 2)
Iyy = MASS_BASE / 12.0 * (BODY[0] ** 2 + BODY[2] ** 2)
Izz = MASS_BASE / 12.0 * (BODY[0] ** 2 + BODY[1] ** 2)
ROTOR_IZZ = 0.5 * MASS_ROTOR * ROTOR_RADIUS ** 2
ROTOR_IXX = MASS_ROTOR * (3 * ROTOR_RADIUS ** 2) / 12.0

# X3 pattern: rotors 0,1 ccw on one diagonal; 2,3 cw on the other.
ROTORS = [
    (0, +ARM, +ARM, 'ccw', 1),
    (1, -ARM, -ARM, 'ccw', 1),
    (2, +ARM, -ARM, 'cw', -1),
    (3, -ARM, +ARM, 'cw', -1),
]


def rotor_link(ns: str, i: int, x: float, y: float) -> str:
    return f"""
      <link name="rotor_{i}">
        <pose>{x} {y} {ROTOR_Z} 0 0 0</pose>
        <inertial>
          <mass>{MASS_ROTOR}</mass>
          <inertia>
            <ixx>{ROTOR_IXX:.3e}</ixx><iyy>{ROTOR_IXX:.3e}</iyy>
            <izz>{ROTOR_IZZ:.3e}</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <visual name="rotor_{i}_visual">
          <geometry><cylinder><radius>{ROTOR_RADIUS}</radius><length>0.003</length></cylinder></geometry>
        </visual>
      </link>
      <joint name="rotor_{i}_joint" type="revolute">
        <parent>base_link</parent>
        <child>rotor_{i}</child>
        <axis>
          <xyz>0 0 1</xyz>
          <limit><lower>-1e16</lower><upper>1e16</upper></limit>
          <dynamics><spring_reference>0</spring_reference><spring_stiffness>0</spring_stiffness></dynamics>
        </axis>
      </joint>"""


def motor_plugin(ns: str, i: int, direction: str) -> str:
    return f"""
      <plugin filename="gz-sim-multicopter-motor-model-system"
              name="gz::sim::systems::MulticopterMotorModel">
        <robotNamespace>{ns}</robotNamespace>
        <jointName>rotor_{i}_joint</jointName>
        <linkName>rotor_{i}</linkName>
        <turningDirection>{direction}</turningDirection>
        <timeConstantUp>{TIME_CONSTANT_UP}</timeConstantUp>
        <timeConstantDown>{TIME_CONSTANT_DOWN}</timeConstantDown>
        <maxRotVelocity>{MAX_ROT_VEL}</maxRotVelocity>
        <motorConstant>{MOTOR_CONSTANT:.4e}</motorConstant>
        <momentConstant>{MOMENT_CONSTANT}</momentConstant>
        <commandSubTopic>command/motor_speed</commandSubTopic>
        <actuator_number>{i}</actuator_number>
        <rotorDragCoefficient>{ROTOR_DRAG}</rotorDragCoefficient>
        <rollingMomentCoefficient>{ROLLING_MOMENT}</rollingMomentCoefficient>
        <motorSpeedPubTopic>motor_speed/{i}</motorSpeedPubTopic>
        <rotorVelocitySlowdownSim>{ROTOR_VELOCITY_SLOWDOWN}</rotorVelocitySlowdownSim>
        <motorType>velocity</motorType>
      </plugin>"""


def model(ns: str, pose: str) -> str:
    rotors = ''.join(rotor_link(ns, i, x, y) for i, x, y, _, _ in ROTORS)
    motors = ''.join(motor_plugin(ns, i, d) for i, _, _, d, _ in ROTORS)
    rotor_config = ''.join(f"""
          <rotor>
            <jointName>rotor_{i}_joint</jointName>
            <forceConstant>{MOTOR_CONSTANT:.4e}</forceConstant>
            <momentConstant>{MOMENT_CONSTANT}</momentConstant>
            <direction>{sgn}</direction>
          </rotor>""" for i, _, _, _, sgn in ROTORS)
    return f"""
    <model name="{ns}">
      <pose>{pose}</pose>
      <link name="base_link">
        <inertial>
          <mass>{MASS_BASE}</mass>
          <inertia>
            <ixx>{Ixx:.3e}</ixx><iyy>{Iyy:.3e}</iyy><izz>{Izz:.3e}</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <collision name="base_collision">
          <geometry><box><size>{BODY[0]} {BODY[1]} {BODY[2]}</size></box></geometry>
        </collision>
        <visual name="base_visual">
          <geometry><box><size>{BODY[0]} {BODY[1]} {BODY[2]}</size></box></geometry>
        </visual>
      </link>{rotors}{motors}
      <plugin filename="gz-sim-multicopter-control-system"
              name="gz::sim::systems::MulticopterVelocityControl">
        <robotNamespace>{ns}</robotNamespace>
        <commandSubTopic>cmd_vel</commandSubTopic>
        <enableSubTopic>enable</enableSubTopic>
        <comLinkName>base_link</comLinkName>
        <velocityGain>{' '.join(map(str, VELOCITY_GAIN))}</velocityGain>
        <attitudeGain>{' '.join(map(str, ATTITUDE_GAIN))}</attitudeGain>
        <angularRateGain>{' '.join(map(str, ANGULAR_RATE_GAIN))}</angularRateGain>
        <maximumLinearAcceleration>{' '.join(map(str, MAX_LIN_ACC))}</maximumLinearAcceleration>
        <rotorConfiguration>{rotor_config}
        </rotorConfiguration>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system"
              name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
        <odom_publish_frequency>100</odom_publish_frequency>
      </plugin>
    </model>"""


HEADER = """<?xml version="1.0" ?>
<!-- GENERATED by chase_sim_gz/gen_world.py; edit THAT file, then rerun
     python3 -m chase_sim_gz.gen_world. See its docstring for the verified
     plugin structure and the calibration status of every constant. -->
<sdf version="1.9">
  <world name="chase">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </visual>
      </link>
    </model>
"""

FOOTER = """
  </world>
</sdf>
"""


def generate() -> str:
    return (HEADER + model('follower', '0 0 1 0 0 0')
            + model('target', '2 0 1 0 0 0') + FOOTER)


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'worlds', 'chase.sdf')
    with open(out, 'w') as f:
        f.write(generate())
    print(f'wrote {out}')
    print(f'  motorConstant {MOTOR_CONSTANT:.4e} (hover at '
          f'{HOVER_FRACTION:.0%} of {MAX_ROT_VEL} rad/s)')
    print(f'  inertia base Ixx {Ixx:.3e} Iyy {Iyy:.3e} Izz {Izz:.3e}')
    print('  REMINDER: sign_test.py first, then calibrate_lag.py (3.2)')


if __name__ == '__main__':
    main()
