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
 * @file okvis2x_app_network_synchronous.cpp
 * @brief A synchronous app with neural networks
 * @author Simon Boche
 * @author Jaehyung Jung
 * @author Sebastian Barbas Laina
 */

#include <iostream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <utility>

#include <Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// okvis & supereightinterface
#include <okvis/XDatasetReader.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <boost/filesystem.hpp>
#include <okvis/SubmappingInterface.hpp>

#if defined(OKVIS_STEREO_NETWORK_PROCESSOR)
#include <okvis/Stereo2DepthProcessor.hpp>
#elif defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
#include <okvis/DepthFusionProcessor.hpp>
#endif

// Rewrite without SubmapInterfacer class
int main(int argc, char **argv)
{

  // argv[1] --> okvis config
  // argv[2] --> mapping config file
  // argv[3] --> dataset path
  // argv[4] --> (optional) save directory

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // read configuration file
  std::string configFilename(argv[1]);
  std::string seConfigFilename(argv[2]);
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  se::SubMapConfig submapConfig(seConfigFilename);
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataType::Config dataConfig;
  dataConfig.readYaml(seConfigFilename);

  const bool isSubmapping = parameters.output.enable_submapping;

  if(!isSubmapping) {
    LOG(WARNING) << "You are running a depth network app with enable_submapping: false. Setting it to true. If you do not want submapping use okvis_app_synchronous.";
    parameters.output.enable_submapping = true;
  }

  // dataset reader
  std::string path(argv[3]);
  std::shared_ptr<okvis::XDatasetReader> datasetReader;
  okvis::Duration deltaT(0.0); // time tolerance to callbacks
  #if defined(OKVIS_STEREO_NETWORK_PROCESSOR) || defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
  datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, false, true, false));
  #endif

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string();
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR) << "The dbow file is not valid: " 
      << dBowVocDir << "/small_voc.yml.gz";
    return EXIT_FAILURE;
  }

  // Output Folder
  std::string savePath;
  if(argc == 4) {
    savePath = path;
  }
  else if(argc == 5){
    savePath = std::string(argv[4]);
    submapConfig.resultsDirectory = savePath;
  }
  else {
    LOG(ERROR) << "Usesage: ./" << argv[0] << 
      "[config-okvis2.yaml] [config-se2.yaml] [input-directory] (optional)[output-directory]";
    return EXIT_FAILURE;
  }

  // Setup OKVIS estimatpr
  std::shared_ptr<okvis::ThreadedSlam> estimator(nullptr);
  estimator.reset(new okvis::ThreadedSlam(parameters, dBowVocDir, submapConfig));
  estimator->setBlocking(true);
  
  // Setup the submapping interface
  std::shared_ptr<okvis::SubmappingInterface> seInterface(nullptr);
  seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters));
  seInterface->setT_BS(parameters.imu.T_BS);
  seInterface->setBlocking(true);

  // Setup stereo network
  std::shared_ptr<okvis::DeepLearningProcessor> processor;

  #ifdef OKVIS_STEREO_NETWORK_PROCESSOR
  processor.reset(new okvis::Stereo2DepthProcessor(parameters, dBowVocDir));
  #elif defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
  processor.reset(new okvis::DepthFusionProcessor(parameters, dBowVocDir));
  #endif
                                              
  processor->setBlocking(true);

  // Select a proper output name
  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }
  estimator->setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final_trajectory.csv");
  estimator->setMapCsvFile(savePath+"/okvis2-" + mode + "-final_map.csv");

  // Setup the trajectory output writer
  std::shared_ptr<okvis::TrajectoryOutput> writer;
  writer.reset(new okvis::TrajectoryOutput(savePath+"/okvis2-" + mode + "_trajectory.csv", false, parameters.output.display_topview));
  estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                  std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                  std::shared_ptr<const okvis::MapPointVector> _4){
    writer->processState(_1,_2,_3,_4);
    seInterface->stateUpdateCallback(_1,_2,_3);
    #ifdef OKVIS_DFUSION_NETWORK_PROCESSOR
    processor->stateUpdateCallback(_1,_2,_3,_4);
    #endif
  });

  // Set a callback in the submapping interface
  if(submapConfig.useMap2MapFactors || submapConfig.useMap2LiveFactors) {
    seInterface->setAlignCallback(std::bind(&okvis::ThreadedSlam::addSubmapAlignmentConstraints, estimator,
                                              std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                              std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  }

  // connect reader to estimator
  datasetReader->setImuCallback(
          std::bind(&okvis::ThreadedSlam::addImuMeasurement, estimator,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  datasetReader->setImagesCallback(
          std::bind(&okvis::ThreadedSlam::addImages, estimator, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));
  
  if(parameters.gps){
      if((*parameters.gps).type == "cartesian"){
          datasetReader->setGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
      }
      else if((*parameters.gps).type == "geodetic" || (*parameters.gps).type == "geodetic-leica"){
          datasetReader->setGeodeticGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGeodeticGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
      }
  }

  datasetReader->setImagesNetworkCallback(
        std::bind(&okvis::DeepLearningProcessor::addImages, processor, std::placeholders::_1,
                  std::placeholders::_2));
  #if defined(OKVIS_STEREO_NETWORK_PROCESSOR)
  processor->setImageCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
    bool estimatorAdd = false;
    bool mapAdd = false;
    for(const auto& cam_idx_frames : frames) {
      for(const auto& cam_measurement : cam_idx_frames.second) {
        if(!cam_measurement.measurement.depthImage.empty() && !cam_measurement.measurement.sigmaImage.empty()) {
          estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                              cam_measurement.measurement.depthImage,
                                                              cam_measurement.measurement.sigmaImage);
        } else if (!cam_measurement.measurement.depthImage.empty()) {
          estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                              cam_measurement.measurement.depthImage);
        }
      }
    }
    mapAdd = seInterface->addDepthMeasurement(frames);
    return (estimatorAdd && mapAdd);
  });
  #elif defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
  processor->setLiveDepthImageCallback(
          std::bind(&okvis::ThreadedSlam::addDepthMeasurement, estimator,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  processor->setImageCallback(
          std::bind(&okvis::SubmappingInterface::addDepthMeasurement, seInterface, std::placeholders::_1));
  #endif 
      
  // Start submapping interface
  seInterface->start();

  // Start streaming
  if(!datasetReader->startStreaming()) {
    LOG(ERROR) << "Failure with datasetReader streaming.";
    return EXIT_FAILURE;
  }

  // Estimator Loop
  okvis::Time startTime = okvis::Time::now();

  int progress = 0;
  bool datasetreaderFinished = false;
  while(true){

    if(!datasetreaderFinished){
      estimator->processFrame();      
      #if defined(OKVIS_STEREO_NETWORK_PROCESSOR)
      std::map<std::string, cv::Mat> images;
      processor->display(images);

      if (!images["leftImage"].empty() && !images["rightImage"].empty()) {
          cv::Mat stereoImages, depthSigma, leftImageRgb, rightImageRgb, stereoDebug;
          cv::cvtColor(images["leftImage"], leftImageRgb, cv::COLOR_GRAY2BGR);
          cv::cvtColor(images["rightImage"], rightImageRgb, cv::COLOR_GRAY2BGR);

          cv::hconcat(leftImageRgb, rightImageRgb, stereoImages);
          auto it = images.find("stereoDepth");
          auto it_sigma = images.find("stereoSigma");
          if(it != images.end() && it_sigma != images.end()) {
          cv::hconcat(it->second, it_sigma->second, depthSigma);
          cv::vconcat(stereoImages, depthSigma, stereoDebug);
          } else {
            stereoDebug = stereoImages;
          }

          cv::imshow("Left & Right images; Depth & Sigma images", stereoDebug);
      }
      #elif defined(OKVIS_DFUSION_NETWORK_PROCESSOR)
      std::map<std::string, cv::Mat> debugImages;
      estimator->display(debugImages);
      for(const auto & image : debugImages) {
        cv::imshow(image.first, image.second);
      }
      std::map<std::string, cv::Mat> networkImages;
      processor->display(networkImages);
      if (!networkImages["srcImage0"].empty()) {
        std::vector<cv::Mat> rgbImages;
        for (int ii = 0; ii < 7; ii++) {
          cv::Mat tmp_ii;
          std::string srcName = "srcImage" + std::to_string(ii);
          cv::cvtColor(networkImages[srcName], tmp_ii, cv::COLOR_GRAY2RGB);
          rgbImages.push_back(tmp_ii);
        }
        std::vector<cv::Mat> rowVec0 = {rgbImages[0], rgbImages[1], rgbImages[2], rgbImages[3]};
        std::vector<cv::Mat> rowVec1 = {rgbImages[4], rgbImages[5], rgbImages[6], networkImages["sparseDepth"]};
        cv::Mat rowImage0, rowImage1, inputImages;
        cv::hconcat(rowVec0, rowImage0);
        cv::hconcat(rowVec1, rowImage1);
        cv::vconcat(rowImage0, rowImage1, inputImages);
        cv::imshow("Input images", inputImages);
      }

      if (networkImages.count("stereoDepth") != 0 && networkImages.count("mvsDepth") != 0 && networkImages.count("fuseDepth") != 0) {
        cv::Mat invDepthOutput, invSigmaOutput, networkOutput;
        std::vector<cv::Mat> visDepth = {networkImages["stereoDepth"], networkImages["mvsDepth"], networkImages["fuseDepth"]};
        std::vector<cv::Mat> visSigma = {networkImages["stereoSigma"], networkImages["mvsSigma"], networkImages["fuseSigma"]};
        cv::hconcat(visDepth, invDepthOutput);
        cv::hconcat(visSigma, invSigmaOutput);
        cv::vconcat(invDepthOutput, invSigmaOutput, networkOutput);
        cv::imshow("Depth stereo/mvs/fuse; Inverse depth sigma stereo/mvs/fuse", networkOutput);
      }
      #endif
      
      cv::Mat topView;
      writer->drawTopView(topView);
      if(!topView.empty()) {
        cv::imshow("OKVIS 2 Top View", topView);
      }
      cv::waitKey(2);
    }

    // check if done
    if(!datasetReader->isStreaming()) {
      datasetreaderFinished = true;
      std::cout << "\r DatasetReader Finished!" << std::endl;

      if(datasetreaderFinished){
        while(!(processor->finishedProcessing() && seInterface->finishedIntegrating())){
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          LOG(INFO) << "waiting for integrator to finish!";
        }

        estimator->writeFinalTrajectoryCsv();
        if(parameters.estimator.do_final_ba) {
          LOG(INFO) << "final full BA...";
          estimator->doFinalBa();
          estimator->setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final-ba_trajectory.csv");
          estimator->writeFinalTrajectoryCsv();
          
          // Wait until other processors finish again from the finalBA stateUpdate callback.
          while(!(processor->finishedProcessing() && seInterface->finishedIntegrating())){
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
            LOG(INFO) << "waiting for integrator to finish!";
          }
        }
        LOG(INFO) <<"total processing time OKVIS only " << (okvis::Time::now() - startTime) << " s" << std::endl;

        seInterface->setFinished();
        seInterface->printAssociations();
        if (submapConfig.write_mesh_output) {
          seInterface->saveAllSubmapMeshes();
        }
        cv::Mat finalSubmapPlot;
        if(seInterface->publishSubmapTopView(finalSubmapPlot)){
          if(!finalSubmapPlot.empty()){
              cv::imwrite(savePath+"submaps.png", finalSubmapPlot);
          }
        }
        estimator->saveMap(); // This saves landmarks map
        LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
        break;
      }
    }

    // display progress
    int newProgress = int(datasetReader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
    if (newProgress>progress) {
      LOG(INFO) << okvis::timing::Timing::print();
    }
#endif
    if (newProgress>progress) {
      progress = newProgress;
      std::cout << "\rProgress: "
                << progress << "% "
                << std::flush;
    }
  }
  return EXIT_SUCCESS;
}