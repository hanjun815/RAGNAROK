"""Start RVI OKVIS, then GaRLILEO with the RVI config and bag playback when OKVIS is ready.

RVI uses only the left camera, so OKVIS runs on the VIO node instead of
the depth-fusion node. Everything else follows ragnarok.launch.py.
"""

import importlib.util
import os


def _ragnarok_launch():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ragnarok.launch.py')
    spec = importlib.util.spec_from_file_location('ragnarok_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_launch_description():
    return _ragnarok_launch().make_launch_description(
        okvis_executable='okvis2x_node_subscriber',
        okvis_config='okvis2_RVI.yaml',
        garlileo_config='config_RVI.yaml',
        okvis_label='VIO OKVIS')
