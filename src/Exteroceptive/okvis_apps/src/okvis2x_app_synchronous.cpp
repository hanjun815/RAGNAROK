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
 * @file okvis2x_app_synchronous.cpp
 * @brief A synchronous app without neural networks
 * @author Simon Boche
 * @author Jaehyung Jung
 * @author Sebastian Barbas Laina
 */

#include <iostream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <utility>

#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <opencv2/highgui/highgui.hpp>

#include <okvis/XDatasetReader.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <okvis/SubmappingInterface.hpp>


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
  const bool isDepth = parameters.lidar ? false : true;
  const bool isLidar = !isDepth;
  const bool isSubmapping = parameters.output.enable_submapping;

  // dataset reader
  std::string path(argv[3]);
  std::shared_ptr<okvis::XDatasetReader> datasetReader;
  okvis::Duration deltaT(0.0); // time tolerance to callbacks
  if (isSubmapping) {
    datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, isLidar, false, isDepth));
  }
  else {
    datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, false, false, false));
  }
  

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

  // Setup OKVIS estimator
  std::shared_ptr<okvis::ThreadedSlam> estimator(nullptr);
  estimator.reset(new okvis::ThreadedSlam(parameters, dBowVocDir, submapConfig));
  estimator->setBlocking(true);

  // Setup the submapping interface
  std::shared_ptr<okvis::SubmappingInterface> seInterface(nullptr);
  if (isSubmapping) {
    seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters));
    seInterface->setT_BS(parameters.imu.T_BS);
    seInterface->setBlocking(true);
  }

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

  if (isSubmapping) {
    // Set callbacks in the estimator
    estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      writer->processState(_1,_2,_3,_4);
      seInterface->stateUpdateCallback(_1,_2,_3);
    });

    // Set a callback in the submapping interface
    if(submapConfig.useMap2MapFactors || submapConfig.useMap2LiveFactors) {
      seInterface->setAlignCallback(std::bind(&okvis::ThreadedSlam::addSubmapAlignmentConstraints, estimator,
                                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
    }
  }
  else {
    estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      writer->processState(_1,_2,_3,_4);
    });
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
  if (isSubmapping) {
    if (isLidar) {
      datasetReader->setLidarCallback([&] (const okvis::Time& _1, const Eigen::Vector3d& _2){
        bool estimatorAdd = true;
        if(submapConfig.useMap2LiveFactors) {
            estimatorAdd = estimator->addLidarMeasurement(_1, _2);
        }
        bool mapAdd = seInterface->addLidarMeasurement(_1, _2);
        return (estimatorAdd && mapAdd);
      });
    }
    else if (isDepth) {
      datasetReader->setDepthImageCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        bool estimatorAdd = false;
        bool mapAdd = false;
        for(const auto& cam_idx_frames : frames) {
          for(const auto& cam_measurement : cam_idx_frames.second) {
            if(!cam_measurement.measurement.depthImage.empty() && !cam_measurement.measurement.sigmaImage.empty()) {
              estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                                  cam_measurement.measurement.depthImage,
                                                                  cam_measurement.measurement.sigmaImage);
            } 
            else if(!cam_measurement.measurement.depthImage.empty()) {
              estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                                  cam_measurement.measurement.depthImage);
            }
          }
        }
        mapAdd = seInterface->addDepthMeasurement(frames);
        return (estimatorAdd && mapAdd);    
      });
    }

    // Start submapping interface
    seInterface->start();
  }

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
          std::map<std::string, cv::Mat> images;
          estimator->display(images);
          for(const auto & image : images) {
            cv::imshow(image.first, image.second);
          }
          cv::Mat topView;
          writer->drawTopView(topView);
          if(!topView.empty()) {
            cv::imshow("OKVIS 2 Top View", topView);
          }
          cv::Mat submapPlot;
          if(isSubmapping && seInterface->publishSubmapTopView(submapPlot)){
            if(!submapPlot.empty()) {
              cv::imshow("Top View Submaps", submapPlot);
            }
          }
          if(!images.empty() || !topView.empty() || !submapPlot.empty()) {
            cv::waitKey(2);
          }
      }

      // check if done
      if(!datasetReader->isStreaming()) {
          datasetreaderFinished = true;
          std::cout << "\r DatasetReader Finished!" << std::endl;

          if(datasetreaderFinished){
              if (isSubmapping) {
                while(!seInterface->finishedIntegrating()){
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  LOG(INFO) << "waiting for integrator to finish!";
                }
              }
              estimator->writeFinalTrajectoryCsv();
              if(parameters.estimator.do_final_ba) {
                LOG(INFO) << "final full BA...";
                estimator->doFinalBa();
                estimator->setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final-ba_trajectory.csv");
                estimator->writeFinalTrajectoryCsv();

                // Wait until the other processor finishes again from the finalBA stateUpdate callback.
                if (isSubmapping) {
                  while(!seInterface->finishedIntegrating()){
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    LOG(INFO) << "waiting for integrator to finish!";
                  }
                }
              }
              LOG(INFO) <<"total processing time OKVIS only " << (okvis::Time::now() - startTime) << " s" << std::endl;

              if (isSubmapping) {
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