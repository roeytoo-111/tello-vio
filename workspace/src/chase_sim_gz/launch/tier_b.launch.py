"""Tier B ROS-rehearsal graph (sim_training_architecture.md 3.2-3.3):

  gz server (headless, chase.sdf, RUNNING free -- this launch is the
  wall-clock rehearsal mode; the lockstep RL mode is GzChaseEnv, which
  owns its own server)
  + ros_gz bridge (twists in, odometry + clock out)
  + tello_sim_shim x2 (sticks -> metric, dead-man)
  + sim_oracle_detector (truth -> DroneDetection, our calibration)
  + latency_shim (measured-delay queue on sim time)

Everything above the shims -- chase_state, chase_policy, chase_mixer,
chase_safety -- attaches to /tello1/cmd_vel and /chase/detection_delayed
exactly as it would on hardware.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('chase_sim_gz')
    world = os.path.join(share, 'worlds', 'chase.sdf')
    bridge_cfg = os.path.join(share, 'config', 'bridge.yaml')
    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        DeclareLaunchArgument('corruption_file', default_value=''),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', world], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': bridge_cfg}, sim_time],
             output='screen'),
        Node(package='chase_sim_gz', executable='tello_sim_shim',
             namespace='tello1', parameters=[sim_time],
             remappings=[('cmd_vel_metric', '/follower/cmd_vel_metric')],
             output='screen'),
        Node(package='chase_sim_gz', executable='tello_sim_shim',
             namespace='tello2', parameters=[sim_time],
             remappings=[('cmd_vel_metric', '/target/cmd_vel_metric')],
             output='screen'),
        Node(package='chase_sim_gz', executable='sim_oracle_detector',
             parameters=[sim_time,
                         {'corruption_file':
                          LaunchConfiguration('corruption_file')}],
             output='screen'),
        Node(package='chase_sim_gz', executable='latency_shim',
             parameters=[sim_time], output='screen'),
    ])
