"""Calibration helpers. Run these once per airframe, in this order.

    # 1. Camera intrinsics (external tool, see README):
    #    ros2 run camera_calibration cameracalibrator \
    #        --size 8x6 --square 0.025 image:=/image_raw camera:=/camera

    # 2. IMU noise + biases -- drone still, motors off, on a solid surface:
    ros2 launch tello_vio calibrate.launch.py target:=imu duration:=120.0

    # 3. Camera-IMU rotation + time offset -- rotate the drone by hand about all
    #    three axes in front of a textured scene:
    ros2 launch tello_vio calibrate.launch.py target:=camera_imu duration:=90.0

Each writes a YAML fragment; merge the results into config/tello_vio.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('target', default_value='imu',
                              choices=['imu', 'camera_imu'],
                              description='Which calibration to run'),
        DeclareLaunchArgument('duration', default_value='120.0'),
        DeclareLaunchArgument('tello_ip', default_value='192.168.10.1'),
        DeclareLaunchArgument('driver', default_value='true'),
        DeclareLaunchArgument('output', default_value=''),
    ]

    driver = Node(
        package='tello', executable='tello', name='tello', output='screen',
        parameters=[{'tello_ip': LaunchConfiguration('tello_ip'), 'tf_pub': False}],
        respawn=False,
    )

    imu = Node(
        package='tello_vio', executable='imu_calib', name='tello_imu_calib',
        output='screen',
        condition=LaunchConfigurationEquals('target', 'imu'),
        parameters=[{'duration_s': LaunchConfiguration('duration'),
                     'output': 'imu_calibration.yaml'}],
    )

    cam_imu = Node(
        package='tello_vio', executable='camera_imu_calib',
        name='tello_camera_imu_calib', output='screen',
        condition=LaunchConfigurationEquals('target', 'camera_imu'),
        parameters=[{'duration_s': LaunchConfiguration('duration'),
                     'output': 'camera_imu_calibration.yaml'}],
    )

    return LaunchDescription(args + [driver, imu, cam_imu])
