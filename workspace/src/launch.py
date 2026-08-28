"""Deprecated top-level launch file, kept working for muscle memory.

Prefer one of the installed launch files, which are discoverable and take
arguments:

    ros2 launch tello_vio vio.launch.py      # driver + VIO (what you want)
    ros2 launch tello tello.launch.py        # driver + keyboard control only

This file remains because `ros2 launch launch.py` from `workspace/src` is in
the README's history. Two things it used to get wrong are fixed here:

* it remapped the driver's `/image_raw` to `/camera`, while `tello_control`,
  the ORB-SLAM2 wrapper and every ROS image tool subscribe to `/image_raw` --
  so the camera window stayed black and SLAM never received a frame;
* it started a `map -> drone` static transform that collided with the driver's
  own transform on the same edge.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tello', executable='tello', name='tello', output='screen',
            parameters=[{
                'tello_ip': '192.168.10.1',
                'tf_base': 'odom',
                'tf_drone': 'base_link',
                'camera_frame': 'camera_optical',
                'tf_pub': True,
            }],
            respawn=False,
        ),
        Node(
            package='tello_control', executable='tello_control', name='control',
            output='screen', respawn=False,
        ),
    ])
