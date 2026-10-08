#!/usr/bin/env bash
# Source this file in each container shell, including after building the workspace.
source /opt/ros/humble/setup.bash
if [[ -f /root/code/RAGNAROK/install/local_setup.bash ]]; then
    source /root/code/RAGNAROK/install/local_setup.bash
fi
export Torch_DIR=/opt/libtorch-cu121/share/cmake/Torch
export LD_LIBRARY_PATH="/root/code/RAGNAROK/install/garlileo/lib:/opt/libtorch-cu121/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
