#!/usr/bin/env bash
set -eo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$workspace_dir"
source /opt/ros/humble/setup.bash
build_jobs=${BUILD_JOBS:-2}
export CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"
export Torch_DIR=${Torch_DIR:-/opt/libtorch-cu121/share/cmake/Torch}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-'8.6;8.9'}
cuda_architectures=${CUDA_ARCHITECTURES:-'86;89'}

thirdparty_dir="$workspace_dir/src/Proprioceptive/thirdparty"
ctraj_dir="$thirdparty_dir/ctraj"
viewer_dir="$ctraj_dir/thirdparty/tiny-viewer"
cmake -S "$viewer_dir" -B "$viewer_dir-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$viewer_dir-install"
cmake --build "$viewer_dir-build" --target tiny-viewer --parallel "$build_jobs"
cmake --install "$viewer_dir-build"
cmake -S "$ctraj_dir" -B "$ctraj_dir-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ctraj_dir-install"
cmake --build "$ctraj_dir-build" --target ctraj --parallel "$build_jobs"
cmake --install "$ctraj_dir-build"

colcon build --base-paths src --executor sequential \
    --packages-select spot_msgs garlileo okvis \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_ROS2=ON \
    -DUSE_NN=ON -DUSE_GPU=ON -DHAVE_LIBREALSENSE=OFF \
    -DBUILD_APPS=OFF -DBUILD_TESTS=OFF -DSE_APP=OFF -DSE_TEST=OFF \
    "-DTorch_DIR=$Torch_DIR" "-DCMAKE_CUDA_ARCHITECTURES=$cuda_architectures"
