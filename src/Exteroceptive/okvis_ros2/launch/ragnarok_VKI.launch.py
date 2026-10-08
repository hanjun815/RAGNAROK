"""Start VKI OKVIS, then GaRLILEO with the VKI config and bag playback when OKVIS is ready.

VKI leaves out the radars: OKVIS runs on the depth-fusion node with the VKI
config, and GaRLILEO uses only the legs and the IMU. Everything else follows
ragnarok.launch.py.
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
        okvis_config='okvis2_VKI.yaml',
        garlileo_config='config_VKI.yaml')
