import os

from launch          import LaunchDescription
from launch.actions   import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    garlileo_share = get_package_share_directory('garlileo')

    default_cfg_path   = os.path.join(garlileo_share, 'config', 'RAGNAROK', 'config_RVI.yaml')
    default_bag_path   = "/root/code/RAGNAROK_dataset/Garden/ros2_bag"

    default_start_time = '3.0'
    # Delay bag play slightly so GaRLILEO subscribes before the first IMU arrives.
    default_bag_play_delay = '0.5'
    default_okvis_sync_stamp_topic = '/okvis/synchronized_stamp'
    default_cam_stamp_queue_max = '10000'
    default_cam_stamp_max_publishes_per_tick = '200'


    cfg_arg   = DeclareLaunchArgument('config_path', default_value=default_cfg_path)
    bag_arg   = DeclareLaunchArgument('rosbag_path', default_value=default_bag_path)
    start_arg = DeclareLaunchArgument('start_time',  default_value=default_start_time)
    delay_arg = DeclareLaunchArgument('bag_play_delay', default_value=default_bag_play_delay)
    okvis_sync_stamp_topic_arg = DeclareLaunchArgument(
        'okvis_sync_stamp_topic', default_value=default_okvis_sync_stamp_topic)
    cam_stamp_queue_max_arg = DeclareLaunchArgument(
        'cam_stamp_queue_max', default_value=default_cam_stamp_queue_max)
    cam_stamp_max_publishes_per_tick_arg = DeclareLaunchArgument(
        'cam_stamp_max_publishes_per_tick', default_value=default_cam_stamp_max_publishes_per_tick)

    cfg_path   = LaunchConfiguration('config_path')
    bag_path   = LaunchConfiguration('rosbag_path')
    start_time = LaunchConfiguration('start_time')
    bag_play_delay = LaunchConfiguration('bag_play_delay')
    okvis_sync_stamp_topic = LaunchConfiguration('okvis_sync_stamp_topic')
    cam_stamp_queue_max = LaunchConfiguration('cam_stamp_queue_max')
    cam_stamp_max_publishes_per_tick = LaunchConfiguration('cam_stamp_max_publishes_per_tick')


    garlileo_node = Node(
        package='garlileo',
        executable='garlileo_node_exe',
        name='garlileo_node',
        output='screen',
        parameters=[{
            'config_path': cfg_path,
            'okvis_sync_stamp_topic': okvis_sync_stamp_topic,
            'foot_msg_format': 'ragnarok',
            'cam_stamp_queue_max': ParameterValue(cam_stamp_queue_max, value_type=int),
            'cam_stamp_max_publishes_per_tick': ParameterValue(cam_stamp_max_publishes_per_tick, value_type=int),
        }]
    )

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path,
             '--clock',                      
             '--start-offset', start_time],
        output='screen'
    )
    bag_play_after_garlileo = RegisterEventHandler(
        OnProcessStart(
            target_action=garlileo_node,
            on_start=[TimerAction(period=bag_play_delay, actions=[bag_play])],
        )
    )

    rviz_cfg = PathJoinSubstitution(
        [garlileo_share, 'config', 'rviz.rviz'])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_cfg]
    )


    return LaunchDescription([
        cfg_arg, bag_arg, start_arg, delay_arg, okvis_sync_stamp_topic_arg,
        cam_stamp_queue_max_arg, cam_stamp_max_publishes_per_tick_arg,
        garlileo_node,
        bag_play_after_garlileo,
        rviz,
    ])
