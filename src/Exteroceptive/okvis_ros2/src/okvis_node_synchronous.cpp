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
 * @file okvis_node_synchronous.cpp
 * @brief This file includes the ROS node implementation: synchronous (blocking) dataset processing.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <execinfo.h>
#include <Eigen/Core>
#include <fstream>

#include <boost/filesystem.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma GCC diagnostic pop

#include <okvis/TrajectoryOutput.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetReader.hpp>
#include <okvis/RpgDatasetReader.hpp>
#include <okvis/ros2/RosbagReader.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <okvis/ros2/Publisher.hpp>
#include <okvis/ThreadedPublisher.hpp>

/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv)
{

  // ros2 setup
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis_node_synchronous");

  // publisher
  auto threadedOdometryPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(
    node,
    threadedOdometryPublisher,
    threadedImagePublisher,
    threadedPublisher);

  // logging
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  // Setting up paramaters
  okvis::Duration deltaT(0.0);
  bool rpg = false;
  bool rgb = false;
  bool rosbag = false;
  std::string configFilename("");
  std::string path("");

  node->declare_parameter("rpg", false);
  node->declare_parameter("rgb", false);
  node->declare_parameter("config_filename", "");
  node->declare_parameter("path", "");
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);

  node->get_parameter("rpg", rpg);
  node->get_parameter("rgb", rgb);
  node->get_parameter("config_filename", configFilename);
  node->get_parameter("path", path);
  if (configFilename.compare("")==0){
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  if (path.compare("")==0){
    LOG(ERROR) << "ros parameter 'path' not set";
    return EXIT_FAILURE;
  }
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate", imu_propagated_state_publishing_rate);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // dataset reader
  std::shared_ptr<okvis::DatasetReaderBase> dataset_reader;
  
  // check if ros2 bag
  std::ifstream f(path+"/metadata.yaml");
  if(f.good()) {
    rosbag = true;
  }
  if(rosbag) {
    dataset_reader.reset(new okvis::RosbagReader(
      path, int(parameters.nCameraSystem.numCameras()),
      parameters.camera.sync_cameras, deltaT));
  } else if(rpg) {
    dataset_reader.reset(new okvis::RpgDatasetReader(
      path, deltaT, int(parameters.nCameraSystem.numCameras())));
  } else {
    dataset_reader.reset(new okvis::DatasetReader(
      path, int(parameters.nCameraSystem.numCameras()),
      parameters.camera.sync_cameras, deltaT));
  }

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string() + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR)<<"DBoW2 vocaublary " << dBowVocDir << "/small_voc.yml.gz not found.";
    return EXIT_FAILURE;
  }

  okvis::ThreadedSlam estimator(parameters, dBowVocDir);
  estimator.setBlocking(true);

  // write logs
  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }

  // setup publishing
  publisher.setCsvFile(path + "/okvis2-" + mode + "-live_trajectory.csv", rpg);
  estimator.setFinalTrajectoryCsvFile(path+"/okvis2-" + mode + "-final_trajectory.csv", rpg);
  estimator.setMapCsvFile(path+"/okvis2-" + mode + "-final_map.csv");
  estimator.setOptimisedGraphCallback(
    std::bind(&okvis::Publisher::publishEstimatorUpdate, &publisher,
              std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4));
  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
  publisher.setupImageTopics(parameters.nCameraSystem);

  threadedOdometryPublisher->startThread();
  threadedImagePublisher->startThread();
  threadedPublisher->startThread();

  // connect reader to estimator
  dataset_reader->setImuCallback(
    std::bind(&okvis::ThreadedSlam::addImuMeasurement, &estimator,
              std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  dataset_reader->setImagesCallback(
    std::bind(&okvis::ThreadedSlam::addImages, &estimator, std::placeholders::_1,
              std::placeholders::_2, std::placeholders::_3));

  // start
  okvis::Time startTime = okvis::Time::now();
  dataset_reader->startStreaming();
  int progress = 0;
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    estimator.processFrame();
    std::map<std::string, cv::Mat> images;
    estimator.display(images);
    publisher.publishImages(images);

    // check if done
    if(!dataset_reader->isStreaming()) {
      estimator.stopThreading();
      std::cout << "\rFinished!" << std::endl;
      if(parameters.estimator.do_final_ba) {
        LOG(INFO) << "final full BA...";
        cv::Mat topView;
        estimator.doFinalBa();
      }
      estimator.writeFinalTrajectoryCsv();
      if(parameters.estimator.do_final_ba) {
        estimator.saveMap();
      }
      LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
      break;
    }

    // display progress
    int newProgress = int(dataset_reader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
    if (newProgress>progress) {
      LOG(INFO) << okvis::timing::Timing::print();
    }
#endif
    if (newProgress>progress) {
      progress = newProgress;
      LOG(INFO) << "Progress: "
                << progress << "% "
                << std::flush;
    }
  }
  return EXIT_SUCCESS;
}
