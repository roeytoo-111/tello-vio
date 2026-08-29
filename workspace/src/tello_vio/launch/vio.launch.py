"""Full VIO stack: Tello driver + tello_vio, with optional ORB-SLAM2 + RViz.

    ros2 launch tello_vio vio.launch.py
    ros2 launch tello_vio vio.launch.py video_scale:=0.5 rviz:=true
    ros2 launch tello_vio vio.launch.py slam:=true            # + ORB-SLAM2 backend

Everything is a LaunchArgument so the same file covers bench testing, flight and
bag replay without editing. ``use_sim_time`` is wired through for bag replay:
without it, nodes stamp with wall time while the bag replays recorded stamps,
and every time-based computation in the estimator quietly becomes nonsense.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = get_package_share_directory('tello_vio')
    default_config = os.path.join(share, 'config', 'tello_vio.yaml')

    args = [
        DeclareLaunchArgument('config', default_value=default_config,
                              description='tello_vio parameter YAML'),
        DeclareLaunchArgument('tello_ip', default_value='192.168.10.1'),
        DeclareLaunchArgument('video_scale', default_value='0.5',
                              description='Driver-side downscale (camera_info is '
                                          'rescaled to match). 0.5 is the measured '
                                          'sweet spot: the front-end works at 480 px '
                                          'anyway, so full resolution only buys 4x '
                                          'the DDS bandwidth (2 MB vs 0.5 MB per '
                                          'frame) and a 2 ms RGB->BGR copy per '
                                          'frame in the driver.'),
        DeclareLaunchArgument('video_target_fps', default_value='30.0'),
        DeclareLaunchArgument('driver', default_value='true',
                              description='Launch the Tello driver (false for bag replay)'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('control', default_value='true',
                              description='Keyboard control GUI. Its OpenCV '
                                          'window must have FOCUS for keys to '
                                          'register - the terminal will not do.'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='Also run ORB-SLAM2 + map_align'),
        DeclareLaunchArgument('vocabulary', default_value='',
                              description='Path to ORBvoc.txt (required if slam:=true)'),
        DeclareLaunchArgument('slam_config', default_value='',
                              description='ORB-SLAM2 settings YAML (defaults to the '
                                          'one installed by the orbslam2 package)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Set true when replaying a bag'),
        DeclareLaunchArgument('mission_pad', default_value='false',
                              description='Enable Mission Pad detection for '
                                          'ground truth (Tello EDU / RoboMaster '
                                          'TT only; a standard Tello reports it '
                                          'as unsupported and carries on).'),
        DeclareLaunchArgument('mission_pad_direction', default_value='0',
                              description='0 = downward only (20 Hz), '
                                          '1 = forward only (20 Hz), '
                                          '2 = both (10 Hz).'),
    ]

    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    driver = Node(
        package='tello', executable='tello', name='tello', output='screen',
        condition=IfCondition(LaunchConfiguration('driver')),
        parameters=[{
            'tello_ip': LaunchConfiguration('tello_ip'),
            'video_scale': LaunchConfiguration('video_scale'),
            'video_target_fps': LaunchConfiguration('video_target_fps'),
            # The driver must NOT publish odom->base_link: tello_vio owns that
            # edge. Two publishers on one TF edge is a broken tree.
            'tf_pub': False,
            'mission_pad_enable': LaunchConfiguration('mission_pad'),
            'mission_pad_direction': LaunchConfiguration('mission_pad_direction'),
        }, use_sim_time],
        respawn=False,
    )

    vio = Node(
        package='tello_vio', executable='vio', name='tello_vio', output='screen',
        parameters=[LaunchConfiguration('config'), use_sim_time],
    )

    control = Node(
        package='tello_control', executable='tello_control', name='control',
        output='screen', respawn=False,
        condition=IfCondition(LaunchConfiguration('control')),
        parameters=[use_sim_time],
    )

    slam_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('slam')),
        actions=[
            Node(
                package='orbslam2', executable='mono', name='orbslam', output='screen',
                arguments=[LaunchConfiguration('vocabulary'),
                           LaunchConfiguration('slam_config')],
                parameters=[{'image_topic': '/image_raw',
                             'map_frame': 'map',
                             'camera_frame': 'camera_optical'}, use_sim_time],
            ),
            Node(
                package='tello_vio', executable='map_align', name='tello_map_align',
                output='screen',
                parameters=[LaunchConfiguration('config'), use_sim_time],
            ),
        ],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', PathJoinSubstitution([FindPackageShare('tello_vio'),
                                               'rviz', 'vio.rviz'])],
        parameters=[use_sim_time],
    )

    return LaunchDescription(args + [driver, vio, control, slam_group, rviz])
