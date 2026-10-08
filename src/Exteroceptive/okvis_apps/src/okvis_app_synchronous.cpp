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
 * @file okvis_app_synchronous.cpp
 * @brief This file processes a dataset.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma GCC diagnostic pop
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetReader.hpp>
#include <okvis/RpgDatasetReader.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <boost/filesystem.hpp>

#include <execinfo.h>


/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv)
{

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  if (argc != 4 && argc != 5) {
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file dataset-folder [-rpg]";
    return EXIT_FAILURE;
  }

  okvis::Duration deltaT(0.0);
  bool rpg = false;
  std::string savePath;
  savePath = std::string(argv[2]);
  if (argc == 5) {
    savePath = std::string(argv[3]);
    if(strcmp(argv[4], "-rpg")==0) {
      rpg = true;
    }
  }
  else if (argc == 4) {
    if(strcmp(argv[3], "-rpg")!=0)  {
      savePath = std::string(argv[3]);
    }
  }

  // read configuration file
  std::string configFilename(argv[1]);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // dataset reader
  // the folder path
  std::string path(argv[2]);
  std::shared_ptr<okvis::DatasetReaderBase> datasetReader;
  if(rpg){
    datasetReader.reset(new okvis::RpgDatasetReader(
                          path, deltaT, int(parameters.nCameraSystem.numCameras())));
  } else {
    datasetReader.reset(new okvis::DatasetReader(
                          path, int(parameters.nCameraSystem.numCameras()),
                          parameters.camera.sync_cameras, deltaT, parameters.gps));
  }

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string();
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

  const bool isWriteRpg = false;
  okvis::TrajectoryOutput writer(savePath+"/okvis2-" + mode + "_trajectory.csv", isWriteRpg, parameters.output.display_topview);
  estimator.setOptimisedGraphCallback(
        std::bind(&okvis::TrajectoryOutput::processState, &writer,
                  std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                  std::placeholders::_4));
  estimator.setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final_trajectory.csv", isWriteRpg);
  estimator.setMapCsvFile(savePath+"/okvis2-" + mode + "-final_map.csv");

  // connect reader to estimator
  datasetReader->setImuCallback(
        std::bind(&okvis::ThreadedSlam::addImuMeasurement, &estimator,
                  std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  datasetReader->setImagesCallback(
        std::bind(&okvis::ThreadedSlam::addImages, &estimator, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));
  if(parameters.gps) {
    if ((*parameters.gps).type == "cartesian") {
      datasetReader->setGpsCallback(
              std::bind(&okvis::ThreadedSlam::addGpsMeasurement, &estimator,
                        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    } else if ((*parameters.gps).type == "geodetic" || (*parameters.gps).type == "geodetic-leica") {
      datasetReader->setGeodeticGpsCallback(
              std::bind(&okvis::ThreadedSlam::addGeodeticGpsMeasurement, &estimator,
                        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
    } else {
      LOG(ERROR) << "Unknown GPS data type.";
      return EXIT_FAILURE;
    }
  }

  // start
  okvis::Time startTime = okvis::Time::now();
  datasetReader->startStreaming();
  int progress = 0;
  while (true) {
    estimator.processFrame();
    std::map<std::string, cv::Mat> images;
    estimator.display(images);
    for(const auto & image : images) {
      cv::imshow(image.first, image.second);
    }
    cv::Mat topView;
    writer.drawTopView(topView);
    if(!topView.empty()) {
      cv::imshow("OKVIS 2 Top View", topView);
    }
    if(!images.empty() || !topView.empty()) {
      char b = cv::waitKey(2);
      if (b == 's') {
        cv::imwrite("saved.png", topView);
      }
    }

    // check if done
    if(!datasetReader->isStreaming()) {
      estimator.stopThreading();
      LOG(INFO) << "Finished!" << std::endl;
      estimator.writeFinalTrajectoryCsv();
      if(parameters.gps){
        estimator.writeGlobalTrajectoryCsv(savePath+"/okvis2-" + mode + "-global-final_trajectory.csv");
      }
      estimator.setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final-ba_trajectory.csv", isWriteRpg);
      if(parameters.estimator.do_final_ba) {
        LOG(INFO) << "final full BA...";
        cv::Mat topView;
        estimator.doFinalBa();
        writer.drawTopView(topView);
        if (!topView.empty()) {
          cv::imshow("OKVIS 2 Top View Final", topView);
          cv::imwrite("okvis2_final_ba.png", topView);
        }
        cv::waitKey(1000);
      }
      estimator.writeFinalTrajectoryCsv();
      if(parameters.gps){
        estimator.writeGlobalTrajectoryCsv(savePath+"/okvis2-" + mode + "-global-final-ba_trajectory.csv");
      }
      if(parameters.estimator.do_final_ba) {
        estimator.saveMap();
      }
      LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
      break;
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
      LOG(INFO) << "Progress: " << progress << "% ";
    }
  }
  return EXIT_SUCCESS;
}