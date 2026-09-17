"""Tello driver + YOLO detector + live viewer, one command.

    ros2 launch chase_detector chase.launch.py weights:=/abs/path/yolov8n.pt
    ros2 launch chase_detector chase.launch.py weights:=... target_classes:=drone
    ros2 launch chase_detector chase.launch.py driver:=false   # replaying a bag
    ros2 launch chase_detector chase.launch.py capture:=true   # + dataset recorder

The driver include reuses tello.launch.py exactly as-is (its camera_info
path, its no-TF-fight rules); `control` stays off by default because
detection does not need the keyboard GUI -- pass control:=true when flying.
Relative csv/capture directories resolve against the shell's cwd, so run
from the repo root and logs land in <repo>/chase_logs.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = LaunchConfiguration
    args = [
        # --- driver ---
        DeclareLaunchArgument('driver', default_value='true',
                              description='Start the tello driver too. Set '
                                          'false when replaying a rosbag.'),
        DeclareLaunchArgument('tello_ip', default_value='192.168.10.1'),
        DeclareLaunchArgument('video_scale', default_value='1.0'),
        DeclareLaunchArgument('video_target_fps', default_value='30.0'),
        DeclareLaunchArgument('control', default_value='false',
                              description='Keyboard control GUI (flying).'),
        # --- detector ---
        DeclareLaunchArgument('weights', default_value='yolov8n.pt',
                              description='Path to .pt weights, resolved '
                                          'against the launch cwd. Stock '
                                          'yolov8n.pt has NO drone class -- '
                                          'smoke tests only.'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('conf', default_value='0.15',
                              description='Low on purpose while the '
                                          'stand-in airplane class is '
                                          'used; raise to 0.25+ with '
                                          'real drone weights.'),
        # Stand-in until the one-class drone weights exist: keep only
        # COCO 'airplane', the nearest class. Expect weak, intermittent
        # hits (0.19 conf on this repo's own Tello close-up) -- that is the
        # weights gap, not a pipeline fault. With fine-tuned weights pass
        # target_classes:=drone. Empty string keeps ALL classes.
        DeclareLaunchArgument('target_classes', default_value='airplane',
                              description="Comma-separated classes to keep; "
                                          "empty keeps all. 'drone' with "
                                          "one-class fine-tuned weights."),
        DeclareLaunchArgument('ref_width_m', default_value='0.180',
                              description='Real width of the boxed object: '
                                          '0.180 Tello prop span, 0.098 body.'),
        DeclareLaunchArgument('csv_dir', default_value='chase_logs',
                              description='Per-frame CSV directory; empty '
                                          'string disables logging.'),
        # --- extras ---
        DeclareLaunchArgument('view', default_value='true',
                              description='Live window (video + traces).'),
        DeclareLaunchArgument('capture', default_value='false',
                              description='Also save raw frames for the '
                                          'training dataset.'),
        DeclareLaunchArgument('capture_dir', default_value='chase_dataset'),
        DeclareLaunchArgument('capture_rate_hz', default_value='2.0'),
    ]

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('tello'), 'launch', 'tello.launch.py'])),
        condition=IfCondition(cfg('driver')),
        launch_arguments={
            'tello_ip': cfg('tello_ip'),
            'video_scale': cfg('video_scale'),
            'video_target_fps': cfg('video_target_fps'),
            'control': cfg('control'),
        }.items(),
    )

    detector = Node(
        package='chase_detector', executable='detector',
        name='chase_detector', output='screen',
        parameters=[{
            'weights': cfg('weights'),
            'device': cfg('device'),
            'imgsz': cfg('imgsz'),
            'conf': cfg('conf'),
            'target_classes': cfg('target_classes'),
            'ref_width_m': cfg('ref_width_m'),
            'csv_dir': cfg('csv_dir'),
        }],
        respawn=False,
    )

    viewer = Node(
        package='chase_detector', executable='viewer',
        name='chase_viewer', output='screen',
        condition=IfCondition(cfg('view')),
        respawn=False,
    )

    capture = Node(
        package='chase_detector', executable='capture',
        name='chase_capture', output='screen',
        condition=IfCondition(cfg('capture')),
        parameters=[{
            'out_dir': cfg('capture_dir'),
            'rate_hz': cfg('capture_rate_hz'),
        }],
        respawn=False,
    )

    return LaunchDescription(args + [driver, detector, viewer, capture])
