"""Exercise launch ordering and failure paths with real child processes, without a GPU.

Run after sourcing ROS 2:
  python3 -m unittest discover -s src/Exteroceptive/okvis_ros2/test -v
"""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, SetLaunchConfiguration, TimerAction
from launch.events import Shutdown


class RagnarOKLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ROS launch caches its log directory across LaunchService instances.
        cls.logs = tempfile.TemporaryDirectory(prefix='ragnarok-launch-test-logs-')
        cls.addClassCleanup(cls.logs.cleanup)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ragnarok-launch-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.started = self.root / 'garlileo-starts.txt'
        launch_dir = self.root / 'garlileo' / 'launch'
        launch_dir.mkdir(parents=True)
        record_start = (
            f'from pathlib import Path; import sys; '
            f'p = Path({str(self.started)!r}); '
            f'p.open("a").write(sys.argv[1] + "\\n")'
        )
        (launch_dir / 'garlileo.launch.py').write_text(
            'import sys\n'
            'from launch import LaunchDescription\n'
            'from launch.actions import DeclareLaunchArgument, ExecuteProcess\n'
            'from launch.substitutions import LaunchConfiguration\n'
            'def generate_launch_description():\n'
            '    return LaunchDescription([\n'
            '        DeclareLaunchArgument("rosbag_path", default_value="/default/bag"),\n'
            f'        ExecuteProcess(cmd=[sys.executable, "-c", {record_start!r},\n'
            '                            LaunchConfiguration("rosbag_path")]),\n'
            '    ])\n'
        )
        path = Path(__file__).resolve().parents[1] / 'launch' / 'ragnarok.launch.py'
        spec = importlib.util.spec_from_file_location('ragnarok_launch', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def run_launch(self, okvis_code, timeout='2.0', shutdown_after=None):
        def fake_node(**kwargs):
            if kwargs['package'] == 'okvis':
                return ExecuteProcess(
                    cmd=[sys.executable, '-u', '-c', okvis_code], output='log',
                    sigterm_timeout='1.0', sigkill_timeout='1.0')
            return ExecuteProcess(
                cmd=['/bin/true'], condition=kwargs.get('condition'), output='log')

        with patch.object(self.module, 'Node', side_effect=fake_node), patch.object(
            self.module, 'get_package_share_directory',
            side_effect=lambda package: str(self.root / package),
        ), patch.dict(os.environ, {'ROS_LOG_DIR': self.logs.name}):
            description = self.module.generate_launch_description()
            if shutdown_after is not None:
                description.add_action(TimerAction(
                    period=shutdown_after,
                    actions=[EmitEvent(event=Shutdown(reason='test shutdown'))],
                ))
            service = LaunchService()
            service.include_launch_description(LaunchDescription([
                SetLaunchConfiguration('rviz', 'false'),
                SetLaunchConfiguration('warmup_timeout', timeout),
                SetLaunchConfiguration('rosbag_path', '/custom/bag'),
                description,
            ]))
            return service.run()

    def test_split_marker_starts_once_and_forwards_bag_argument(self):
        code = (
            'import os, sys, time\n'
            'print("GPU warmup for the MVS network iteration 9", flush=True)\n'
            'time.sleep(0.1)\n'
            f'assert not os.path.exists({str(self.started)!r})\n'
            'sys.stderr.write("RAGNAROK_OK"); sys.stderr.flush()\n'
            'print("unrelated stdout", flush=True)\n'
            'time.sleep(0.05)\n'
            'sys.stderr.write("VIS_READY\\n"); sys.stderr.flush()\n'
            'print("RAGNAROK_OKVIS_READY", flush=True)\n'
            'time.sleep(0.5)\n'
        )
        self.assertEqual(self.run_launch(code), 0)
        self.assertEqual(self.started.read_text().splitlines(), ['/custom/bag'])

    def test_exit_before_ready_does_not_start_garlileo(self):
        self.assertNotEqual(self.run_launch('import sys; sys.exit(17)'), 0)
        self.assertFalse(self.started.exists())

    def test_timeout_does_not_start_garlileo(self):
        self.assertNotEqual(self.run_launch('import time; time.sleep(10)', timeout='0.15'), 0)
        self.assertFalse(self.started.exists())

    def test_marker_cannot_be_assembled_from_different_streams(self):
        code = (
            'import sys, time\n'
            'sys.stderr.write("RAGNAROK_OK"); sys.stderr.flush()\n'
            'sys.stdout.write("VIS_READY"); sys.stdout.flush()\n'
            'time.sleep(10)\n'
        )
        self.assertNotEqual(self.run_launch(code, timeout='0.2'), 0)
        self.assertFalse(self.started.exists())

    def test_shutdown_before_ready_does_not_start_garlileo(self):
        code = (
            'import signal, time\n'
            'signal.signal(signal.SIGINT, lambda *_: None)\n'
            'time.sleep(0.3)\n'
            'print("RAGNAROK_OKVIS_READY", flush=True)\n'
            'time.sleep(0.1)\n'
        )
        self.assertEqual(self.run_launch(code, shutdown_after=0.15), 0)
        self.assertFalse(self.started.exists())


if __name__ == '__main__':
    unittest.main()
