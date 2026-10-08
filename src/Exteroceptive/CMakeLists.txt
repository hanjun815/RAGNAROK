cmake_minimum_required(VERSION 3.8...3.30)
project(okvis VERSION 2.0.0 LANGUAGES CXX C)

# options for build configuration
option (USE_SYSTEM_BRISK
        "Use brisk via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_DBOW2
        "Use DBoW2 via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_CERES
        "Use ceres via find_package rather than downloading it as part of okvis" OFF)
option (BUILD_APPS
        "Builds a demo app (which require boost and OpenCV highgui)" ON)
option (BUILD_TESTS
        "Builds all gtests" OFF)
option (BUILD_ROS2
        "Builds ros2 wrapper" ON)
option (HAVE_LIBREALSENSE
        "Use realsense as part of okvis" ON)
option (USE_NN
        "Use keypoint classification as part of okvis, requires torch" ON)
option (USE_GPU
        "Use keypoint classification with GPU inference, requires torch with Cuda" ON)
option (DO_TIMING
        "Enable timing" ON)
option (BUILD_SHARED_LIBS
        "Whether to build okvis as a shared library" OFF)

option (USE_COLIDMAP "Whether to build okvis supportin colour and object's id 'for language based OKVIS' or regular one" ON)

set(OKVIS_CXX_STANDARD 17)
set(OKVIS_PUBLIC_CXX_FLAGS)
if(APPLE)
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra -Wno-dtor-name)
else()
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra)
endif()

set(BLA_VENDOR Generic)
find_package(LAPACK REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/geographiclib")

# Make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
endif()

if(USE_NN AND USE_GPU)
  if(APPLE)
    # MPS enabled by default
    add_compile_definitions(OKVIS_USE_MPS)
  else()
    # If CUDA isn't enabled explicitly here it's not detected correctly by Torch.
    enable_language(CUDA)
    add_compile_definitions(OKVIS_USE_CUDA)
  endif()
endif()

if(USE_NN)
  find_package(Torch REQUIRED)

  # As in OKVIS2-X, depth-model.pt and fast-scnn.pt are committed in resources/, and
  # mvs-model.pt, too large for Git, is downloaded from the OKVIS2-X model server on the
  # first configure. All three are checked against their SHA256.
  function(require_model filename expected_hash)
    set(model_path "${PROJECT_SOURCE_DIR}/resources/${filename}")
    if(NOT EXISTS "${model_path}")
      message(FATAL_ERROR "Missing ${model_path}.")
    endif()
    file(SHA256 "${model_path}" actual_hash)
    if(NOT actual_hash STREQUAL expected_hash)
      message(FATAL_ERROR "${model_path} does not match its expected SHA256.")
    endif()
  endfunction()
  function(download_model filename url expected_hash)
    set(model_path "${PROJECT_SOURCE_DIR}/resources/${filename}")
    if(EXISTS "${model_path}")
      file(SHA256 "${model_path}" actual_hash)
      if(actual_hash STREQUAL expected_hash)
        return()
      endif()
    endif()
    # Download into the build tree first, so an interrupted download never
    # leaves a broken model in resources/.
    set(download_dir "${CMAKE_BINARY_DIR}/model_download")
    message(STATUS "Downloading ${filename} from ${url}")
    file(DOWNLOAD "${url}" "${download_dir}/${filename}"
         INACTIVITY_TIMEOUT 60 STATUS download_status SHOW_PROGRESS)
    list(GET download_status 0 download_code)
    set(actual_hash "")
    if(download_code EQUAL 0)
      file(SHA256 "${download_dir}/${filename}" actual_hash)
    endif()
    if(NOT actual_hash STREQUAL expected_hash)
      file(REMOVE_RECURSE "${download_dir}")
      message(FATAL_ERROR "Could not download ${filename} (${download_status}). "
                          "Download ${url} to ${model_path} and configure again.")
    endif()
    file(COPY "${download_dir}/${filename}" DESTINATION "${PROJECT_SOURCE_DIR}/resources")
    file(REMOVE_RECURSE "${download_dir}")
  endfunction()
  require_model(depth-model.pt 75d3a68b0a18e8ff1b20ac1e4b75a094d595c57cf52482485d4e8049782fe4c8)
  require_model(fast-scnn.pt fcef94092f5f0f6425fc479bd32f0a57b05be2fdbdd896a3c3941c1532a861e4)
  download_model(mvs-model.pt https://cvg.cit.tum.de/webshare/g/srl-models/mvs/mvs-sigma.pt
                 1f37ee7499b1b38dfdb11208b859cf02c127e5f120e9db3ed34847ff14eb6633)

endif()

if(NOT DO_TIMING)
  add_compile_definitions(DEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers")
endif()

if(BUILD_TESTS)
  # Must be called before adding any tests for them to be registered with CMake
  enable_testing()
endif()

add_subdirectory(external)
# SE2 Stuff
option(SE_TEST "Compile the supereight tests" OFF)
option(SE_APP "Compile the supereight application" OFF)
add_subdirectory(supereight2)


add_subdirectory(okvis_util)
add_subdirectory(okvis_kinematics)
add_subdirectory(okvis_time)
add_subdirectory(okvis_cv)
add_subdirectory(okvis_common)
add_subdirectory(okvis_timing)
add_subdirectory(okvis_ceres)
add_subdirectory(okvis_frontend)
add_subdirectory(okvis_mapping)
add_subdirectory(okvis_multisensor_processing)

# Include stereo from depth network.
if(USE_NN)
  add_subdirectory(okvis_deep_learning)
endif()

if(BUILD_APPS)
  find_package(Boost REQUIRED COMPONENTS filesystem)
  find_package(OpenCV REQUIRED core imgcodecs highgui calib3d)
  if(USE_SYSTEM_BRISK)
    find_package(brisk REQUIRED)
  endif()
  if(USE_SYSTEM_DBOW2)
    find_package(DBoW2 REQUIRED)
  endif()
  
  # DBoW2 test app
  add_executable(dbow2_test okvis_apps/src/dbow2_test.cpp)
  target_include_directories(dbow2_test
    PRIVATE
      ${DBoW2_INCLUDE_DIRS}
  )
  target_link_libraries(dbow2_test PRIVATE
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::Frontend
  )
  target_compile_options(dbow2_test
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )


  # require GeographicLib
  find_package(GeographicLib REQUIRED)

  # OKVIS2 app
  add_executable(okvis_app_synchronous okvis_apps/src/okvis_app_synchronous.cpp)
  target_link_libraries(okvis_app_synchronous
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::MultisensorProcessing
  )
  target_compile_options(okvis_app_synchronous
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  install(TARGETS okvis_app_synchronous EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(okvis2x_app_synchronous
          okvis_apps/src/okvis2x_app_synchronous.cpp
          )
  target_link_libraries(okvis2x_app_synchronous
          okvis::Util
          okvis::Kinematics
          okvis::Time
          okvis::CV
          okvis::Common
          okvis::Ceres
          okvis::Timing
          okvis::Frontend
          okvis::MultisensorProcessing
          pthread
          ${Boost_LIBRARIES}
          ${TORCH_LIBRARIES}
          ${GeographicLib_LIBRARIES}
          SRL::Supereight2
          SRL::Projection
  )
  target_compile_options(okvis2x_app_synchronous
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )

  target_compile_features(okvis2x_app_synchronous
          PUBLIC cxx_std_${OKVIS_CXX_STANDARD}
  )

  if(HAVE_LIBREALSENSE)
    # OKIVS RealSense app
    add_executable(okvis_app_realsense okvis_apps/src/okvis_app_realsense.cpp)
    target_link_libraries(okvis_app_realsense
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  
    # OKIVS RealSense recorder app
    add_executable(okvis_app_realsense_recorder okvis_apps/src/okvis_app_realsense_recorder.cpp)
    target_link_libraries(okvis_app_realsense_recorder
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense_recorder
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense_recorder EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  
  if(USE_NN)
    find_package(Torch REQUIRED)
  
    # Torch test app
    add_executable(nn_test okvis_apps/src/nn_test.cpp)
    target_compile_options(nn_test PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(nn_test
      PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(nn_test
      ${OpenCV_LIBS}
      ${TORCH_LIBRARIES}
    )
    target_compile_features(nn_test PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
    target_compile_options(nn_test
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    target_compile_features(nn_test
      PUBLIC cxx_std_${OKVIS_CXX_STANDARD}
    )
    
    # Submapping App with Stereo Depth NN
    add_executable(okvis2x_app_snetwork_synchronous
                   okvis_apps/src/okvis2x_app_network_synchronous.cpp)
    target_compile_definitions(okvis2x_app_snetwork_synchronous PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR)
    target_link_libraries(okvis2x_app_snetwork_synchronous
      okvis_util
      okvis_kinematics
      okvis_time
      okvis_cv
      okvis_common
      okvis_ceres
      okvis_timing
      okvis_frontend
      okvis_multisensor_processing
      okvis_deep_learning
      okvis_mapping
      pthread
      ${Boost_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${GeographicLib_LIBRARIES}
      SRL::Supereight2
      SRL::Projection
    )

    add_executable(okvis2x_app_depthfusion_synchronous
                   okvis_apps/src/okvis2x_app_network_synchronous.cpp)
    target_compile_definitions(okvis2x_app_depthfusion_synchronous PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR)
    target_link_libraries(okvis2x_app_depthfusion_synchronous
      okvis_util
      okvis_kinematics
      okvis_time
      okvis_cv
      okvis_common
      okvis_ceres
      okvis_timing
      okvis_frontend
      okvis_multisensor_processing
      okvis_deep_learning
      pthread
      ${Boost_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${GeographicLib_LIBRARIES}
      SRL::Supereight2
      SRL::Projection
    )

  endif()               
endif()

# ros2
if(BUILD_ROS2)
  if (NOT "$ENV{ROS_VERSION}" STREQUAL "2")
    message(WARNING "ROS2 not found, will build without")
    set(BUILD_ROS2 OFF)
  endif()
endif()
if(BUILD_ROS2)
  include_directories(okvis_ros2/include/)
  message(STATUS "Building ROS2 wrapper")

  # list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  # ROS2 dependencies
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(eigen3_cmake_module REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(Boost REQUIRED)
  find_package(tf2_eigen REQUIRED)
  find_package(tf2_geometry_msgs REQUIRED)

  find_package(PCL REQUIRED QUIET COMPONENTS common)
  include_directories(${PCL_INCLUDE_DIRS} include)
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  find_package(pcl_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(rclpy REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosbag2_cpp REQUIRED)
  find_package(rosbag2_storage REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(image_transport REQUIRED)
  find_package(message_filters REQUIRED)

  # incldues and lib directories
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${Eigen3_INCLUDE_DIRS})
  include_directories( include ${PCL_INCLUDE_DIRS} )

  # okvis::Ros2 library
  set(LIB_NAME okvis_ros2)
  add_library(${LIB_NAME}
    okvis_ros2/src/Subscriber.cpp
    okvis_ros2/src/RosbagReader.cpp
    okvis_ros2/src/Publisher.cpp
    okvis_ros2/src/RePublisher.cpp
  )
  target_include_directories(${LIB_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/okvis_ros2/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${GLOG_INCLUDE_DIRS}
  )
  if(USE_NN)
    target_link_libraries(${LIB_NAME}
      okvis::DeepLearning
    )
  endif()

  target_link_libraries(${LIB_NAME}
      ${OpenCV_LIBS}
      Ceres::ceres
      Eigen3::Eigen
      SRL::Supereight2
      okvis::Common
      okvis::Util
      okvis::Timing
      okvis::Mapping
      okvis::MultisensorProcessing
      ${GLOG_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
  )
  target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
  target_compile_options(${LIB_NAME}
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  if(OPENMP_FOUND)
    target_compile_definitions(${LIB_NAME} PRIVATE USE_OPENMP)
  endif()
  add_library(okvis::Ros2 ALIAS ${LIB_NAME})

  # target dependencies
  ament_target_dependencies(
    ${LIB_NAME}
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    eigen3_cmake_module
    Eigen3
    sensor_msgs
    visualization_msgs
    std_srvs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
  )

  # export dependencies
  ament_export_dependencies(
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    eigen3_cmake_module
    Eigen3
    std_msgs
    std_srvs
    sensor_msgs
    visualization_msgs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
  )

  install(TARGETS ${LIB_NAME} EXPORT okvisTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

  # OKVIS2X nodes - Wo/ Deep Learning

  # Subscriber
  add_executable(okvis2x_node_subscriber okvis_ros2/src/okvis2x_node.cpp)
  target_link_libraries(
    okvis2x_node_subscriber
    okvis::Ros2
    ${OpenCV_LIBS}
    ${SUITESPARSE_LIBRARIES}
    ${Boost_LIBRARIES} 
    ${PCL_LIBRARIES}
  )
  install(TARGETS okvis2x_node_subscriber DESTINATION lib/${PROJECT_NAME})
  # Realsense
  if(HAVE_LIBREALSENSE)

    add_executable(okvis_node_realsense_publisher okvis_ros2/src/okvis_node_realsense_publisher.cpp)
    target_link_libraries(
      okvis_node_realsense_publisher
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    install(TARGETS okvis_node_realsense_publisher DESTINATION lib/${PROJECT_NAME})
    add_executable(okvis2x_node_realsense okvis_ros2/src/okvis2x_node.cpp)
    target_compile_definitions(okvis2x_node_realsense PUBLIC SRL_NAV_USE_REALSENSE)
    target_link_libraries(
      okvis2x_node_realsense
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    install(TARGETS okvis2x_node_realsense DESTINATION lib/${PROJECT_NAME})
  endif()

  # OKVIS2X nodes - Wo/ Deep Learning W/Deep Learning
  if(USE_NN)
    # W/ Stereo Network Processor
    add_executable(okvis2x_stereo_network_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
    target_compile_definitions(okvis2x_stereo_network_node_subscriber PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR)
    target_link_libraries(
      okvis2x_stereo_network_node_subscriber
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
      okvis::DeepLearning
    )
    install(TARGETS okvis2x_stereo_network_node_subscriber DESTINATION lib/${PROJECT_NAME})

    # W/ DFusion Processor
    add_executable(okvis2x_depthfusion_network_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
    target_compile_definitions(okvis2x_depthfusion_network_node_subscriber PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR)
    target_link_libraries(
      okvis2x_depthfusion_network_node_subscriber
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
      okvis::DeepLearning
    )
    install(TARGETS okvis2x_depthfusion_network_node_subscriber DESTINATION lib/${PROJECT_NAME})


    if(HAVE_LIBREALSENSE)
      # W/ DepthFusion
      add_executable(okvis2x_depthfusion_network_node_realsense okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_depthfusion_network_node_realsense PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR SRL_NAV_USE_REALSENSE)
      target_link_libraries(
        okvis2x_depthfusion_network_node_realsense
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES} 
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      install(TARGETS okvis2x_depthfusion_network_node_realsense DESTINATION lib/${PROJECT_NAME})

      # W/ StereoNetwork
      add_executable(okvis2x_stereo_network_node_realsense okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_stereo_network_node_realsense PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR SRL_NAV_USE_REALSENSE)
      target_link_libraries(
        okvis2x_stereo_network_node_realsense
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES} 
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      install(TARGETS okvis2x_stereo_network_node_realsense DESTINATION lib/${PROJECT_NAME})
    endif()

  endif()

  # Install launch files
  install(DIRECTORY okvis_ros2/launch DESTINATION share/${PROJECT_NAME}/)
  install(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

  # Form ROS2 Package
  AMENT_PACKAGE()

endif(BUILD_ROS2)

# The directory where the .cmake files will be installed
include(GNUInstallDirs)
if(WIN32 AND NOT CYGWIN)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
    set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()
# Generate okvisConfig.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/okvisConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
# Generate okvisConfigVersion.cmake
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(EXPORT okvisTargets
  FILE okvisTargets.cmake
  NAMESPACE okvis::
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)