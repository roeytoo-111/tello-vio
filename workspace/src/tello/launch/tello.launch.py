"""Tello driver + keyboard control GUI (no estimator).

    ros2 launch tello tello.launch.py
    ros2 launch tello tello.launch.py video_scale:=0.5 rviz:=true

For the full VIO stack use `ros2 launch tello_vio vio.launch.py` instead --
that one also starts the estimator, which is what publishes odom -> base_link.

Note what this file deliberately does NOT do: it publishes no odom -> base_link
transform. The previous version started a static_transform_publisher on that
edge *and* let the driver publish a barometer-derived transform on the same
edge, so two publishers fought over one TF link and lookups returned whichever
sample happened to win. A TF edge has exactly one owner.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('tello_ip', default_value='192.168.10.1'),
        DeclareLaunchArgument('video_scale', default_value='1.0',
                              description='Downscale before publishing. '
                                          'camera_info is rescaled to match.'),
        DeclareLaunchArgument('video_target_fps', default_value='30.0'),
        DeclareLaunchArgument('control', default_value='true',
                              description='Keyboard control GUI'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('rqt', default_value='false',
                              description='rqt is off by default: a loaded topic '
                                          'monitor subscribes to /status, which '
                                          'costs the driver a WiFi round trip.'),
    ]

    driver = Node(
        package='tello', executable='tello', name='tello', output='screen',
        parameters=[{
            'tello_ip': LaunchConfiguration('tello_ip'),
            'video_scale': LaunchConfiguration('video_scale'),
            'video_target_fps': LaunchConfiguration('video_target_fps'),
            'tf_base': 'odom',
            'tf_drone': 'base_link',
            'camera_frame': 'camera_optical',
            # The static base_link -> camera_optical edge; harmless on its own,
            # and superseded by tello_vio's calibrated version when that runs.
            'tf_pub': True,
        }],
        # Do not respawn: a driver that fails because djitellopy is missing, or
        # because the drone is off, will fail identically on every restart and
        # the log fills with the same traceback forever.
        respawn=False,
    )

    control = Node(
        package='tello_control', executable='tello_control', name='control',
        output='screen', respawn=False,
        condition=IfCondition(LaunchConfiguration('control')),
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        # Without -d, RViz opens with no displays at all and looks broken.
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('tello'), 'rviz', 'tello.rviz'])],
    )

    rqt = Node(
        package='rqt_gui', executable='rqt_gui', name='rqt', output='screen',
        condition=IfCondition(LaunchConfiguration('rqt')),
    )

    return LaunchDescription(args + [driver, control, rviz, rqt])
