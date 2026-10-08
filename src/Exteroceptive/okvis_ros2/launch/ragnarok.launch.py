"""Start depth-fusion OKVIS, then GaRLILEO and bag playback when OKVIS is ready.

make_launch_description() is shared with the variant launch files
ragnarok_RVI.launch.py and ragnarok_VKI.launch.py.
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


READY_MARKER = b'RAGNAROK_OKVIS_READY'


def make_launch_description(okvis_executable='okvis2x_depthfusion_network_node_subscriber',
                            okvis_config='okvis2.yaml', garlileo_config=None,
                            okvis_label='depth-fusion OKVIS'):
    """Build the launch: OKVIS first, then GaRLILEO and bag playback once OKVIS is ready.

    okvis_config and garlileo_config are the default config file names under
    okvis config/rsD455 and garlileo config/RAGNAROK; garlileo_config=None keeps
    the default of garlileo.launch.py.
    """
    okvis_share = get_package_share_directory('okvis')
    garlileo_share = get_package_share_directory('garlileo')

    arguments = [
        DeclareLaunchArgument('config_filename', default_value=os.path.join(
            okvis_share, 'config', 'rsD455', okvis_config)),
        DeclareLaunchArgument('se_config_filename', default_value=os.path.join(
            okvis_share, 'config', 'rsD455', 'se2.yaml')),
        DeclareLaunchArgument('csv_path', default_value='/root/code/RAGNAROK/results'),
        DeclareLaunchArgument('mesh_cutoff_z', default_value='2.5'),
        DeclareLaunchArgument('save_submap_meshes', default_value='true'),
        DeclareLaunchArgument('mesh', default_value='realsense.dae'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('sigterm_timeout', default_value='300.0', description=(
            'Seconds to allow final BA and map saving after Ctrl+C before sending SIGTERM.')),
        DeclareLaunchArgument('warmup_timeout', default_value='180.0', description=(
            'Maximum seconds to wait for OKVIS initialization, including GPU warmup.')),
    ]

    # Reuse GaRLILEO's argument declarations and defaults, including rosbag_path,
    # config_path, start_time, play_rate, and bag_play_delay. Expose them to
    # `ros2 launch ... --show-args` even though the launch itself is deferred.
    # GaRLILEO's own rviz argument is left out: the rviz argument above controls the
    # OKVIS RViz, and GaRLILEO's RViz stays closed.
    garlileo_source = PythonLaunchDescriptionSource(os.path.join(
        garlileo_share, 'launch', 'garlileo.launch.py'))
    garlileo_description = garlileo_source.get_launch_description(LaunchContext())
    garlileo_arguments = [arg for arg in garlileo_description.get_launch_arguments()
                          if arg.name != 'rviz']
    if garlileo_config is not None:
        garlileo_arguments = [
            DeclareLaunchArgument('config_path', default_value=os.path.join(
                garlileo_share, 'config', 'RAGNAROK', garlileo_config))
            if arg.name == 'config_path' else arg
            for arg in garlileo_arguments]
    arguments.extend(garlileo_arguments)
    garlileo = IncludeLaunchDescription(
        garlileo_source,
        launch_arguments=[(arg.name, LaunchConfiguration(arg.name))
                          for arg in garlileo_arguments] + [('rviz', 'false')],
    )

    okvis = Node(
        package='okvis', executable=okvis_executable,
        name='okvis', namespace='okvis', output='screen',
        parameters=[{
            'config_filename': LaunchConfiguration('config_filename'),
            'se_config_filename': LaunchConfiguration('se_config_filename'),
            'csv_path': LaunchConfiguration('csv_path'),
            'mesh_cutoff_z': ParameterValue(LaunchConfiguration('mesh_cutoff_z'), value_type=float),
            'save_submap_meshes': ParameterValue(
                LaunchConfiguration('save_submap_meshes'), value_type=bool),
            'mesh_file': ['file://', PathJoinSubstitution([
                okvis_share, 'resources', 'meshes', LaunchConfiguration('mesh')])],
        }],
        remappings=[
            ('cam0/image_raw', '/camera/camera/infra1/image_rect_raw'),
            ('cam1/image_raw', '/camera/camera/infra2/image_rect_raw'),
            ('imu0', '/imu/data'),
            ('radar0', '/ti_mmwave_0/radar_scan_pcl'),
            ('radar1', '/ti_mmwave_1/radar_scan_pcl'),
        ],
    )
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', namespace='okvis',
        arguments=['-d', os.path.join(
            okvis_share, 'config', 'rviz2', 'rviz2_okvis2x_config.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    started = False
    tails = {True: b'', False: b''}

    def start_garlileo(context):
        nonlocal started
        if started or context.is_shutdown:
            return []
        started = True
        watchdog.cancel()
        return [LogInfo(msg=f'{okvis_label} initialization complete; starting GaRLILEO.'),
                garlileo]

    def on_output(event):
        if started:
            return None
        # ProcessIO may split the marker across reads. Keep stdout/stderr separate
        # and retain only enough bytes to recognize a split marker.
        stream = event.from_stdout
        text = tails[stream] + event.text
        tails[stream] = text[-(len(READY_MARKER) - 1):]
        if READY_MARKER in text:
            return [OpaqueFunction(function=start_garlileo)]
        return None

    def on_timeout(context):
        if not started and not context.is_shutdown:
            raise RuntimeError(
                'OKVIS did not become ready before warmup_timeout. GaRLILEO was not started. '
                'Check the OKVIS/CUDA logs and rebuild OKVIS if using an older executable.')
        return []

    def on_okvis_exit(event, context):
        if context.is_shutdown:
            return []
        watchdog.cancel()
        if not started:
            raise RuntimeError(
                f'OKVIS exited before becoming ready (code {event.returncode}); '
                'GaRLILEO was not started.')
        return [EmitEvent(event=Shutdown(reason='OKVIS exited; stopping RAGNAROK.'))]

    def validate_timeout(context):
        timeout = float(LaunchConfiguration('warmup_timeout').perform(context))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('warmup_timeout must be a finite positive number of seconds.')
        return []

    watchdog = TimerAction(
        period=LaunchConfiguration('warmup_timeout'),
        actions=[OpaqueFunction(function=on_timeout)],
    )
    return LaunchDescription(arguments + [
        OpaqueFunction(function=validate_timeout),
        SetEnvironmentVariable('OMP_NUM_THREADS', '2'),
        RegisterEventHandler(OnProcessIO(
            target_action=okvis, on_stdout=on_output, on_stderr=on_output)),
        RegisterEventHandler(OnProcessExit(target_action=okvis, on_exit=on_okvis_exit)),
        watchdog,
        LogInfo(msg=f'Starting {okvis_label}; waiting for initialization.'),
        okvis,
        rviz,
    ])


def generate_launch_description():
    return make_launch_description()
