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
 * @file okvis2x_node_subscriber.cpp
 * @brief This file includes the ROS node implementation -- subscribe to sensor topics.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 * @author Jaehyung Jung
 */

#include <functional>
#include <iostream>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdlib.h>

#include <glog/logging.h>

#ifdef SRL_NAV_USE_REALSENSE
  #include <okvis/Realsense.hpp>
  #include <okvis/RealsenseRgbd.hpp>
  #include <okvis/ros2/RePublisher.hpp>
#else
  #include <okvis/ros2/Subscriber.hpp>
#endif
#include <okvis/ros2/Publisher.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetWriter.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/SubmappingInterface.hpp>
#include <okvis/Processor.hpp>
#if defined(OKVIS_STEREO_NETWORK_PROCESSOR) || defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
  #if defined(OKVIS_STEREO_NETWORK_PROCESSOR)
    #include <okvis/Stereo2DepthProcessor.hpp>
  #endif
  #if defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
    #include <okvis/DepthFusionProcessor.hpp>
  #endif
#else
  #error "One of (OKVIS_STEREO_NETWORK_PROCESSOR || OKVIS_DFUSION_NETWORK_PROCESSOR) must be defined"
#endif

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
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis_node_subscriber");
  
  // Setting up paramaters
  std::string configFilename("");
  std::string seConfigFilename("");

  node->declare_parameter("config_filename", "");
  node->declare_parameter("se_config_filename", "");
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
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate", imu_propagated_state_publishing_rate);


  bool save_meshes = false;
  if(!node->get_parameter("save_submap_meshes", save_meshes)){
    LOG(WARNING) << "save_submap_meshes not defined. Will not save submap meshes.";
  }
  std::string csv_path = "/tmp/";
  if(!node->get_parameter("csv_path", csv_path)){
    LOG(INFO) << "csv_path not defined. Will save trajectory to /tmp/.";
  }

  // publisher
  auto threadedOdometryPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(
    node,
    threadedOdometryPublisher,
    threadedImagePublisher,
    threadedPublisher);
    
  // Check for mesh cutoff
  double mesh_cutoff_z = std::numeric_limits<float>::max();
  if(node->get_parameter("mesh_cutoff_z", mesh_cutoff_z)){
    publisher.setMeshCutoffZ(mesh_cutoff_z);
  }

  // construct OKVIS side
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // Debug: make it explicit at runtime which frontend prediction mode is configured.
  LOG(INFO) << "[GaRLILEO][Frontend] runtime config: frontend_use_garlileo="
            << std::boolalpha << parameters.garlileo.frontend_use_garlileo
            << " frontend_use_rotation=" << parameters.garlileo.frontend_use_rotation
            << " frontend_use_translation=" << parameters.garlileo.frontend_use_translation
            << " frontend_use_garlileo_gravity=" << parameters.garlileo.frontend_use_garlileo_gravity
            << " spline_state_topic=/garlileo/spline_state"
            << " frontend_max_age_sec=" << parameters.garlileo.frontend_max_age_sec
            << " keyframe_stationary_speed_threshold=" << parameters.garlileo.keyframe_stationary_speed_threshold
            << " place_recognition_use_radar=" << parameters.garlileo.place_recognition_use_radar
            << " place_recognition_radar0_enable=" << parameters.garlileo.place_recognition_radar0_enable
            << " place_recognition_radar1_enable=" << parameters.garlileo.place_recognition_radar1_enable
            << " place_recognition_radar_max_time_diff=" << parameters.garlileo.place_recognition_radar_max_time_diff
            << " place_recognition_radar_scans_per_radar="
            << parameters.garlileo.place_recognition_radar_scans_per_radar
            << " place_recognition_radar_alpha=" << parameters.garlileo.place_recognition_radar_alpha
            << " place_recognition_radar_min_intensity=" << parameters.garlileo.place_recognition_radar_min_intensity
            << " apply_imu_extrinsic=" << parameters.garlileo.apply_imu_extrinsic;

  // Read supereight2 config file.
  se::SubMapConfig submapConfig(seConfigFilename);
  okvis::SupereightMapType::Config  mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataConfigType dataConfig;
  dataConfig.readYaml(seConfigFilename);

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string() + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR)<<"DBoW2 vocabulary " << dBowVocDir << "/small_voc.yml.gz not found.";
    return EXIT_FAILURE;
  }

  // Setereo or depth fusion or vision-language fusion
  okvis::DeepLearningProcessor* dlProcessor;
  #if defined(OKVIS_STEREO_NETWORK_PROCESSOR)
    dlProcessor = new okvis::Stereo2DepthProcessor(parameters, dBowVocDir);
  #elif defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
    dlProcessor = new okvis::DepthFusionProcessor(parameters, dBowVocDir);
  #endif
  
  okvis::Processor processor(
    parameters, 
    dlProcessor,
    dBowVocDir,
    mapConfig, 
    dataConfig, 
    submapConfig
  );
  processor.setBlocking(false);
  processor.setT_BS(parameters.imu.T_BS);

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
    #if defined(OKVIS_STEREO_NETWORK_PROCESSOR) || defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
      realsense.reset(new okvis::Realsense(okvis::Realsense::SensorType::D455, rgb, rs_rgb_exposure));
      realsense->setIrSize(parameters.nCameraSystem.cameraGeometry(0)->imageWidth(),
                             parameters.nCameraSystem.cameraGeometry(0)->imageHeight());
      if(depth) {
        LOG(WARNING) << "Specified Depth Camera. but currently not supported with stereo or depth fusion network";
      }
      if(rgb) {
        realsense->setRgbSize(parameters.nCameraSystem.cameraGeometry(2)->imageWidth(),
                              parameters.nCameraSystem.cameraGeometry(2)->imageHeight());
        LOG(INFO) << "RGB camera enabled";
      }
    #else
      realsense.reset(new okvis::RealsenseRgbd(okvis::Realsense::SensorType::D455, true, false, rs_emitter_power, rs_rgb_exposure));
    #endif
  #else
    // subscriber
    #if defined(OKVIS_STEREO_NETWORK_PROCESSOR) || defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
        std::shared_ptr<okvis::Subscriber> subscriber(new okvis::Subscriber(node, &processor, &publisher, parameters, &processor.se_interface_, false, false));
    #endif
  #endif


  // output publishing
  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
  publisher.setupImageTopics(parameters.nCameraSystem);
  #ifdef OKVIS_STEREO_NETWORK_PROCESSOR
  publisher.setupNetworkTopics("stereo");
  #endif
  #ifdef OKVIS_DFUSION_NETWORK_PROCESSOR
  publisher.setupNetworkTopics("fuse");
  #endif

  #ifdef SRL_NAV_USE_REALSENSE
    bool use_rgb = false;
    for(size_t i = 0; i < parameters.nCameraSystem.numCameras(); i++) {
      if(parameters.nCameraSystem.cameraType(i).isColour) {
        use_rgb = true;
      }
    }
    // re-publishing
    okvis::RePublisher rePublisher(node, parameters, &processor, threadedImagePublisher, threadedPublisher);
    realsense->setImuCallback(
            std::bind(&okvis::Processor::addImuMeasurement, &processor,
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImuCallback(
          std::bind(&okvis::Publisher::realtimePredictAndPublish, &publisher,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImuCallback(
          std::bind(&okvis::RePublisher::publishImuMeasurement, &rePublisher,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImagesCallback(
          std::bind(&okvis::RePublisher::addImages, &rePublisher, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));
    if(republish_images){
      realsense->setImagesCallback(
            std::bind(&okvis::RePublisher::publishImages, &rePublisher, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3));
    }
    std::string depthTopic = "";

    LOG(INFO) << "Depth topic is " << depthTopic;
    if(use_rgb) {
      rePublisher.setTopics("imu0", "cam", "rgb", depthTopic);
    } else {
      rePublisher.setTopics("imu0", "cam", "", depthTopic);
    }
    
  #endif
      

  // Setup processor callbacks
  processor.setOptimizedGraphCallback(
        std::bind(&okvis::Publisher::publishEstimatorUpdate, &publisher,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));
  processor.setAlignmentPublishCallback([&](const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud, bool isMapFrame)
    {publisher.publishAlignmentPointsAsCallback(timestamp, T_WS, alignPointCloud, isMapFrame);}
  );
  processor.setSubmapCallback(std::bind(
    &okvis::Publisher::publishSubmapsAsCallback, 
    &publisher, 
    std::placeholders::_1, std::placeholders::_2)
  );

  // require a special termination handler to properly close
  shtdown = false;
  signal(SIGINT, [](int) { 
    shtdown = true; 
  });

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

  // Start streaming (if Realsense)
  #ifdef SRL_NAV_USE_REALSENSE
    if(!realsense->startStreaming()) {
      return EXIT_FAILURE;
    }
  #endif
  
  // Launch uses this marker only after successful GPU warmup and after all
  // subscribers, callbacks, and publishing threads are ready to receive data.
  LOG(INFO) << "RAGNAROK_OKVIS_READY";

  // Main loop
  while (true) {
    rclcpp::spin_some(node);
    processor.processFrame();
    std::map<std::string, cv::Mat> images;
    processor.display(images);
    publisher.publishImages(images);
    if(shtdown) {
      break;
    }
  }
  #ifdef SRL_NAV_USE_REALSENSE
    realsense->stopStreaming();
  #else
    subscriber->shutdown();
  #endif

  // final BA if needed
  if(parameters.estimator.do_final_ba) {
    LOG(INFO) << "Final full BA...";
    processor.slam_.doFinalBa();
  }

  processor.finish();
  // SLAM callbacks are joined; stop depth workers while their Processor targets
  // are still alive. Both network variants join their workers in the virtual dtor.
  delete dlProcessor;
  dlProcessor = nullptr;
  processor.slam_.setFinalTrajectoryCsvFile(csv_path + "/okvis2-final_trajectory.csv", false);
  processor.slam_.writeFinalTrajectoryCsv();
  processor.slam_.writeEwgUtilityMinCsv(csv_path + "/okvis2-ewg_utility_min.csv");
  processor.slam_.writeImuBiasCsv(csv_path + "/okvis2-imu_bias.csv");
  if(save_meshes){
    LOG(WARNING) << "Saving the submap meshes of the submapping interface";
    processor.se_interface_.saveAllSubmapMeshes();
  } else {
    LOG(WARNING) << "Not saving the submap meshes of the submapping interface";
  }
  cv::destroyAllWindows();

  LOG(INFO) << "Closing program";

  return EXIT_SUCCESS;
}
