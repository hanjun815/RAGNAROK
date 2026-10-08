/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file okvis_node_realsense_publisher.cpp
 * @brief This file includes the ROS node implementation: publish only.
 * @author Stefan Leutenegger
 */

#include <functional>
#include <iostream>
#include <stdlib.h>
#include <iostream>
#include <signal.h>

#include <glog/logging.h>

#include <okvis/DatasetWriter.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/Realsense.hpp>
#include <okvis/RealsenseRgbd.hpp>
#include <okvis/ThreadedSlam.hpp>

#include <okvis/ros2/RePublisher.hpp>
#include <okvis/ThreadedPublisher.hpp>

std::atomic_bool shtdown; ///< Shutdown requested?

/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  // ros2 setup
  rclcpp::init(argc, argv);

  // set up the node
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis_node_realsense_publisher");
  
  // Setting up paramaters
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  std::string configFilename("");
  node->declare_parameter("config_filename", "");
  node->get_parameter("config_filename", configFilename);
  if (configFilename.compare("")==0){
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate", imu_propagated_state_publishing_rate);

  try {

    // determine RGB/Depth modes
    bool rgb = false;
    bool depth = false;
    bool alignDepthToRgb = false;
    if(parameters.nCameraSystem.numCameras() == 3) {
      if(parameters.nCameraSystem.cameraType(2).isColour) {
        rgb = true;
      }
    }
    if(parameters.nCameraSystem.isDepthCamera(0)) {
      depth = true;
      alignDepthToRgb = false;
    } else if (parameters.nCameraSystem.numCameras() == 3
               && parameters.nCameraSystem.isDepthCamera(2)) {
      depth = true;
      alignDepthToRgb = true;
    }

    // realsense sensor
    std::unique_ptr<okvis::Realsense> realsense;
    if(!depth) {
      LOG(INFO) << "No depth camera enabled";
      realsense.reset(new okvis::Realsense(okvis::Realsense::SensorType::D455, rgb));
    } else {
      LOG(INFO) << "Depth camera enabled";
      realsense.reset(
        new okvis::RealsenseRgbd(okvis::Realsense::SensorType::D455, rgb, alignDepthToRgb));
      realsense->setIrSize(parameters.nCameraSystem.cameraGeometry(0)->imageWidth(),
                           parameters.nCameraSystem.cameraGeometry(0)->imageHeight());
    }
    if(rgb) {
      realsense->setRgbSize(parameters.nCameraSystem.cameraGeometry(2)->imageWidth(),
                            parameters.nCameraSystem.cameraGeometry(2)->imageHeight());
      LOG(INFO) << "RGB camera enabled";
    }
    realsense->setHasDeviceTimestamps(false);
                    
    // re-publishing
    auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
    auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
    okvis::RePublisher rePublisher(node, parameters, threadedImagePublisher, threadedPublisher);
    std::string rgbString = "";
    if(rgb) {
      rgbString = "rgb";
    }
    std::string depthString = "";
    if(depth) {
      depthString = "depth";
    }
    rePublisher.setTopics("imu0", "cam", rgbString, depthString);

    // connect sensor to re-publishing
    realsense->setImuCallback(
          std::bind(&okvis::RePublisher::publishImuMeasurement, &rePublisher,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImagesCallback(
          std::bind(&okvis::RePublisher::publishImages, &rePublisher, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));

    // start streaming
    if(!realsense->startStreaming()) {
      return EXIT_FAILURE;
    }
    
    threadedImagePublisher->startThread();
    threadedPublisher->startThread();

    // require a special termination handler to properly close
    shtdown = false;
    signal(SIGINT, [](int) { shtdown = true; });

    // Main loop
    while (true) {
      rclcpp::spin_some(node);
      if(shtdown) {
        break;
      }
    }
    // Stop the pipeline
    realsense->stopStreaming();
  }

  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
