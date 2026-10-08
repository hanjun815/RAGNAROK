FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG BUILD_JOBS=4
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg locales software-properties-common \
    && locale-gen en_US.UTF-8 && add-apt-repository universe

ARG UBUNTU_MIRROR=
RUN if [[ -n $UBUNTU_MIRROR ]]; then \
        sed -i -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
               -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list; \
    fi \
    && echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git git-lfs unzip pkg-config \
    python3-pip python3-yaml python3-colcon-common-extensions python3-rosdep \
    libeigen3-dev libboost-all-dev libopencv-dev libpcl-dev \
    libgoogle-glog-dev libgflags-dev libsuitesparse-dev libmetis-dev \
    liblapack-dev libblas-dev libtbb-dev libyaml-cpp-dev libcereal-dev \
    libfmt-dev libspdlog-dev libgeographic-dev \
    libepoxy-dev libglew-dev libgl1-mesa-dev libegl1-mesa-dev \
    libwayland-dev libxkbcommon-dev libx11-dev libjpeg-dev libpng-dev \
    ros-humble-ros-base ros-humble-rviz2 ros-humble-pcl-ros \
    ros-humble-pcl-conversions ros-humble-cv-bridge ros-humble-image-transport \
    ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs \
    ros-humble-eigen3-cmake-module ros-humble-rosbag2-storage-mcap \
    ros-humble-rosidl-default-generators ros-humble-ament-cmake-python \
    && rm -rf /var/lib/apt/lists/*

# Pinned versions. Build dependencies independently of the workspace.
RUN python3 -m pip install --no-cache-dir cmake==3.26.3
RUN git clone --depth 1 --branch 1.24.6 https://github.com/strasdat/Sophus.git /tmp/sophus \
    && cmake -S /tmp/sophus -B /tmp/sophus/build -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF \
    && cmake --build /tmp/sophus/build --parallel "$BUILD_JOBS" \
    && cmake --install /tmp/sophus/build && rm -rf /tmp/sophus
RUN git clone --depth 1 --branch v0.9.7 https://github.com/Neargye/magic_enum.git /tmp/magic_enum \
    && cmake -S /tmp/magic_enum -B /tmp/magic_enum/build \
       -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF \
    && cmake --install /tmp/magic_enum/build && rm -rf /tmp/magic_enum
RUN git clone --depth 1 --branch v0.9 https://github.com/stevenlovegrove/Pangolin.git /tmp/pangolin \
    && cmake -S /tmp/pangolin -B /tmp/pangolin/build -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
    && cmake --build /tmp/pangolin/build --parallel "$BUILD_JOBS" \
    && cmake --install /tmp/pangolin/build && rm -rf /tmp/pangolin
# Static, position-independent Ceres: CTraj and GaRLILEO link it into shared libraries.
RUN git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git /tmp/ceres \
    && cmake -S /tmp/ceres -B /tmp/ceres/build -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
       -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_CUDA=OFF \
    && cmake --build /tmp/ceres/build --parallel "$BUILD_JOBS" \
    && cmake --install /tmp/ceres/build && rm -rf /tmp/ceres

# ROS/OpenCV use the C++11 ABI; the pre-C++11 LibTorch archive is incompatible.
RUN curl -fL --retry 3 \
       'https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip' \
       -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && mv /opt/libtorch /opt/libtorch-cu121 && rm /tmp/libtorch.zip \
    && printf '/usr/local/lib\n/opt/libtorch-cu121/lib\n' > /etc/ld.so.conf.d/ragnarok.conf \
    && ldconfig

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
COPY setup_env.sh /usr/local/share/ragnarok/setup_env.sh
COPY docker/docker-entrypoint.sh /usr/local/bin/ragnarok-entrypoint
RUN chmod +x /usr/local/bin/ragnarok-entrypoint \
    && echo 'source /usr/local/share/ragnarok/setup_env.sh' >> /root/.bashrc
WORKDIR /root/code/RAGNAROK
ENTRYPOINT ["/usr/local/bin/ragnarok-entrypoint"]
CMD ["bash"]
