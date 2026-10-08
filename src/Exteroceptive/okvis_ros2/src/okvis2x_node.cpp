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
 * @file okvis2x_node.cpp
 * @brief This file includes the ROS node implementation.
 * @author Stefan Leutenegger
 * @author Jaehyung Jung
 * @author Simon Boche
 */

#include <functional>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <signal.h>

#include <glog/logging.h>

#include <okvis/DatasetWriter.hpp>
#include <okvis/ViParametersReader.hpp>
#ifdef SRL_NAV_USE_REALSENSE
  #include <okvis/Realsense.hpp>
  #include <okvis/RealsenseRgbd.hpp>
  #include <okvis/ros2/RePublisher.hpp>
#else
  #include <okvis/ros2/Subscriber.hpp>
#endif

#include <okvis/ThreadedSlam.hpp>
#include <okvis/ros2/Publisher.hpp>
#include <okvis/ThreadedPublisher.hpp>

#include <std_srvs/srv/set_bool.hpp>


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
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis_node_realsense");
  
  // Setting up parameters
  std::string configFilename("");
  std::string seConfigFilename("");
  node->declare_parameter("config_filename", "");
  node->declare_parameter("se_config_filename", "");
  node->declare_parameter("image_transport", "raw");
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  node->declare_parameter("mesh_cutoff_z", std::numeric_limits<float>::max());
  node->declare_parameter("save_submap_meshes", false);
  node->declare_parameter("csv_path", "/tmp/");

  node->get_parameter("config_filename", configFilename);
  if (configFilename.compare("")==0){
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  node->get_parameter("se_config_filename", seConfigFilename);
  if (seConfigFilename.compare("")==0){
    LOG(ERROR) << "ros parameter 'se_config_filename' not set";
    return EXIT_FAILURE;
  }
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  std::string image_transport("raw");
  node->get_parameter("image_transport", image_transport);
  LOG(INFO) << "[Node] image_transport parameter=" << image_transport;
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate", imu_propagated_state_publishing_rate);

  // Read supereight2 config file.
  se::SubMapConfig submapConfig(seConfigFilename);
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataConfigType dataConfig;
  dataConfig.readYaml(seConfigFilename);

  bool save_meshes = false;
  if(!node->get_parameter("save_submap_meshes", save_meshes)){
    LOG(WARNING) << "save_submap_meshes not defined. Will not save submap meshes.";
  }
  std::string csv_path = "/tmp/";
  if(!node->get_parameter("csv_path", csv_path)){
    LOG(INFO) << "csv_path not defined. Will save trajectory to /tmp/.";
  }

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string()
                          + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR)<<"DBoW2 vocabulary " << dBowVocDir << "/small_voc.yml.gz not found.";
    return EXIT_FAILURE;
  }

  // publisher
  auto threadedOdometryPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher    = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher         = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(node, threadedOdometryPublisher, threadedImagePublisher, threadedPublisher);

  // Check for mesh cutoff
  double mesh_cutoff_z = std::numeric_limits<float>::max();
  if(node->get_parameter("mesh_cutoff_z", mesh_cutoff_z)){
    publisher.setMeshCutoffZ(mesh_cutoff_z);
  }

  okvis::ThreadedSlam estimator(parameters, dBowVocDir, submapConfig);
  estimator.setBlocking(false);

  std::shared_ptr<okvis::SubmappingInterface> seInterface(nullptr);
  #ifdef SRL_NAV_USE_REALSENSE

    // Read Exposure settings and emitter power from launch file
    float rs_rgb_exposure = 100.0f;
    node->declare_parameter("realsense_rgb_exposure", 100.0f);
    if(!node->get_parameter("realsense_rgb_exposure", rs_rgb_exposure)) {
      LOG(WARNING) << "Realsense RGB Exposure not specified. Set to a default of " << std::to_string(rs_rgb_exposure) << " [ms].";
    } else {
      LOG(INFO) << "Realsense RGB Exposure set to " << std::to_string(rs_rgb_exposure) << " [ms]";
    }

    float rs_emitter_power = 360.0f;
    node->declare_parameter("realsense_emitter_power", 360.0f);
    if(!node->get_parameter("realsense_emitter_power", rs_emitter_power)) {
      LOG(WARNING) << "Realsense Emitter Power not specified. Set to a default of " << std::to_string(rs_emitter_power);
    } else {
      LOG(INFO) << "Realsense Emitter Power set to " << std::to_string(rs_emitter_power);
    }

    bool republish_images = true;
    node->declare_parameter("republish_images", true);
    if(!node->get_parameter("republish_images", republish_images)) {
      LOG(INFO) << "Republishing not specified. Will republish images.";
    }

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
      realsense.reset(new okvis::Realsense(okvis::Realsense::SensorType::D455, rgb, rs_rgb_exposure));
    } else {
      LOG(INFO) << "Depth camera enabled";
      realsense.reset(
        new okvis::RealsenseRgbd(okvis::Realsense::SensorType::D455, rgb, alignDepthToRgb, rs_emitter_power, rs_rgb_exposure));
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
    okvis::RePublisher rePublisher(node, parameters, &estimator, threadedImagePublisher, threadedPublisher);
    std::string rgbString = "";
    if(rgb) {
      rgbString = "rgb";
    }
    std::string depthString = "";
    if(depth) {
      depthString = "depth";
    }
    rePublisher.setTopics("imu0", "cam", rgbString, depthString);

    // connect sensor to estimator/re-publishing
    realsense->setImuCallback(
          std::bind(&okvis::ThreadedSlam::addImuMeasurement, &estimator,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImuCallback(
          std::bind(&okvis::Publisher::realtimePredictAndPublish, &publisher,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImuCallback(
          std::bind(&okvis::RePublisher::publishImuMeasurement, &rePublisher,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    if(republish_images){
      realsense->setImagesCallback(
            std::bind(&okvis::RePublisher::publishImages, &rePublisher, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3));
    }
    realsense->setImagesCallback(
          std::bind(&okvis::RePublisher::addImages, &rePublisher, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));

    if(parameters.output.enable_submapping){

      seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters));
      seInterface->setT_BS(parameters.imu.T_BS);
      seInterface->setBlocking(false);

      rePublisher.setMappingCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        bool estimatorAdd = false;
        for(const auto& idx_images : frames) {
          for(const auto& measurements : idx_images.second) {
            if(!measurements.measurement.depthImage.empty()){
              estimatorAdd = estimator.addDepthMeasurement(measurements.timeStamp, measurements.measurement.depthImage);
            }
          }
        }
        bool mapAdd = seInterface->addDepthMeasurement(frames);
        return (estimatorAdd && mapAdd);
      });
    }

  #else
    std::shared_ptr<okvis::Subscriber> subscriber;
    if(parameters.output.enable_submapping){
      const bool isDepth = parameters.lidar ? false : true;
      const bool isLidar = !isDepth;
      seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters));
      seInterface->setT_BS(parameters.imu.T_BS);
      seInterface->setBlocking(false);
      
      subscriber.reset(new okvis::Subscriber(node, &estimator, &publisher, parameters,
                                              seInterface.get(), isDepth, isLidar));
    }
    else {
      subscriber.reset(new okvis::Subscriber(node, &estimator, &publisher, parameters));
    }
  #endif

  try {

    // output publishing
    publisher.setBodyTransform(parameters.imu.T_BS);
    publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
    publisher.setupImageTopics(parameters.nCameraSystem);

    if(parameters.output.enable_submapping){
      estimator.setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                              std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                              std::shared_ptr<const okvis::MapPointVector> _4){
        publisher.publishEstimatorUpdate(_1,_2,_3,_4);
        seInterface->stateUpdateCallback(_1,_2,_3);
      });

      estimator.setAlignmentPublishCallback([&](const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
        const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud, bool isMapFrame)
        {publisher.publishAlignmentPointsAsCallback(timestamp, T_WS, alignPointCloud, isMapFrame);}
      );

      seInterface->setSubmapCallback(std::bind(
        &okvis::Publisher::publishSubmapsAsCallback, 
        &publisher, 
        std::placeholders::_1, std::placeholders::_2)
      );

      seInterface->setAlignCallback(std::bind(&okvis::ThreadedSlam::addSubmapAlignmentConstraints, &estimator,
                                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                    std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));

      seInterface->setAlignmentPublishCallback([&](const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
        const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud, bool isMapFrame)
        {publisher.publishAlignmentPointsAsCallback(timestamp, T_WS, alignPointCloud, isMapFrame);}
      );

      // Start submapping interface.
      seInterface->start();
    }
    else {
      estimator.setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                              std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                              std::shared_ptr<const okvis::MapPointVector> _4){
        publisher.publishEstimatorUpdate(_1,_2,_3,_4);
      });
    }
                    
    
    #ifdef SRL_NAV_USE_REALSENSE
      // start streaming
      if(!realsense->startStreaming()) {
        return EXIT_FAILURE;
      }
    #endif

    // require a special termination handler to properly close
    shtdown = false;
    signal(SIGINT, [](int) { shtdown = true; });

    // Setup Serivce for proper shutdown including post-processing steps
    auto service = node->create_service<std_srvs::srv::SetBool>(
      "shutdown",
      [&](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
          std::shared_ptr<std_srvs::srv::SetBool::Response> response) {

          (void)request; // if you don't use the request data
          shtdown = true;
          response->success = true;
          response->message = "Requested Shutdown. Starting offline processing.";
    });

    threadedOdometryPublisher->startThread();
    threadedImagePublisher->startThread();
    threadedPublisher->startThread();

    // Launch uses this marker only after all subscribers, callbacks, and
    // publishing threads are ready to receive data.
    LOG(INFO) << "RAGNAROK_OKVIS_READY";

    // Main loop
    while (true) {
      rclcpp::spin_some(node);
      estimator.processFrame();
      std::map<std::string, cv::Mat> images;
      estimator.display(images);
      publisher.publishImages(images);
      if(shtdown) {
        break;
      }
    }

    #ifdef SRL_NAV_USE_REALSENSE
      // Stop the pipeline
      realsense->stopStreaming();
    #else
      subscriber->shutdown();
    #endif

    // final BA if needed
    if(parameters.estimator.do_final_ba) {
      LOG(INFO) << "Final full BA...";
      estimator.doFinalBa();
    }

    // Finish up
    estimator.stopThreading();
    if(seInterface) {
      while(!seInterface->finishedIntegrating());
      seInterface->setFinished();
    }

    // Write CSV
    estimator.setFinalTrajectoryCsvFile(csv_path + "/okvis2-final_trajectory.csv", false);
    estimator.writeFinalTrajectoryCsv();
    estimator.writeEwgUtilityMinCsv(csv_path + "/okvis2-ewg_utility_min.csv");

    // Save Meshes if requested
    if(save_meshes && seInterface){
      LOG(INFO) << "Saving the submap meshes of the submapping interface";
      seInterface->saveAllSubmapMeshes();
    } else {
      LOG(INFO) << "Not saving the submap meshes of the submapping interface";
    }
  }

  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
