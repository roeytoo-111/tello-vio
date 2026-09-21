"""The full real-time chase graph on the aircraft.

    /image_raw ──▶ chase_detector ──▶ /chase/detection
                                            │
                                            ▼
                                      chase_state  (10 Hz clock)
                                            │ /chase/observation
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
                  chase_policy      chase_depth_rule     chase_behaviour
                  /chase/action     /chase/forward_cmd   /chase/mode
                        └───────────────────┼───────────────────┘
                                            ▼
                                      chase_mixer
                                            │ /chase/cmd_raw
                                            ▼
                                      chase_safety ──▶ /cmd_vel ──▶ driver
                                            └──────▶ /chase/action_sent
                                                       (back to chase_state)

Launched with `control:=false` by default: the keyboard teleop node writes
the SAME rc register in the driver as /cmd_vel does, and last writer wins,
so leaving it running would let it fight the policy. Keep /emergency wired
to something you can reach regardless.

The stack starts DISARMED. Nothing reaches the aircraft until:

    ros2 topic pub -1 /chase/arm std_msgs/Bool "data: true"

and it can be stopped at any moment with:

    ros2 topic pub -1 /chase/abort std_msgs/Empty "{}"      # land
    ros2 topic pub -1 /emergency  std_msgs/Empty "{}"       # cut motors
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint = LaunchConfiguration('checkpoint')
    baseline = LaunchConfiguration('baseline')
    params = LaunchConfiguration('params')
    default_params = os.path.join(
        get_package_share_directory('chase_flight'), 'config',
        'flight.yaml')

    args = [
        DeclareLaunchArgument(
            'checkpoint', default_value='',
            description='Trained bundle (.pt). Its .onnx sibling is used '
                        'for inference when present.'),
        DeclareLaunchArgument(
            'baseline', default_value='',
            description="Fly a hand-written controller instead: "
                        "'p_controller' or 'pn'. The documented flight "
                        "ladder starts here, before any learned policy."),
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('driver', default_value='true',
                              description='Also launch the Tello driver.'),
        DeclareLaunchArgument('detector', default_value='true',
                              description='Also launch the YOLO detector.'),
        DeclareLaunchArgument('weights', default_value='',
                              description='Detector weights (.pt).'),
        DeclareLaunchArgument('conf', default_value='0.25'),
        DeclareLaunchArgument('target_classes', default_value=''),
        DeclareLaunchArgument('tello_ip', default_value='192.168.10.1'),
    ]

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('tello'), 'launch',
            'tello.launch.py')),
        condition=IfCondition(LaunchConfiguration('driver')),
        launch_arguments={
            'tello_ip': LaunchConfiguration('tello_ip'),
            # video_scale 1.0 keeps the published boxes in the calibration
            # frame the observation spec is expressed in. Any other value
            # still works (chase_flight rescales), but 1.0 keeps the
            # detector's own range_m honest too.
            'video_scale': '1.0',
            'video_target_fps': '30.0',
            'control': 'false',      # never let teleop fight the policy
            'rviz': 'false',
        }.items())

    detector = Node(
        package='chase_detector', executable='detector',
        name='chase_detector', output='screen',
        condition=IfCondition(LaunchConfiguration('detector')),
        parameters=[{
            'weights': LaunchConfiguration('weights'),
            'conf': LaunchConfiguration('conf'),
            'target_classes': LaunchConfiguration('target_classes'),
            # MUST match the policy's EnvConfig.ref_width_m (0.098 m body
            # width). The detector's own default is 0.180 m (prop span),
            # which would make its published range_m read 1.84x this
            # stack's range. chase_flight ignores that field and derives
            # range from the box width itself, but an inconsistent value
            # here would mislead every human reading the topic.
            'ref_width_m': 0.098,
            'publish_annotated': True,
        }])

    common = [params, {'checkpoint': checkpoint}]
    nodes = [
        Node(package='chase_flight', executable='chase_state',
             name='chase_state', output='screen', parameters=common),
        Node(package='chase_flight', executable='chase_policy',
             name='chase_policy', output='screen',
             parameters=common + [{'baseline': baseline}]),
        Node(package='chase_flight', executable='chase_depth_rule',
             name='chase_depth_rule', output='screen', parameters=common),
        Node(package='chase_flight', executable='chase_behaviour',
             name='chase_behaviour', output='screen', parameters=[params]),
        Node(package='chase_flight', executable='chase_mixer',
             name='chase_mixer', output='screen', parameters=[params]),
        Node(package='chase_flight', executable='chase_safety',
             name='chase_safety', output='screen', parameters=[params]),
    ]

    return LaunchDescription(args + [driver, detector] + nodes)
