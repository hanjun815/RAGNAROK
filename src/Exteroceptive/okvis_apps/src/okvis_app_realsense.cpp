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
 * @file okvis_app_realsense.cpp
 * @brief This file processes a dataset.
 
 Compatible and tested with D435i and D455

 * @author Stefan Leutenegger
 */

#include <iostream>
#include <signal.h>

#include <opencv2/highgui/highgui.hpp>

#include <okvis/TrajectoryOutput.hpp>
#include <okvis/Realsense.hpp>
#include <okvis/RealsenseRgbd.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetWriter.hpp>
#include <okvis/ViParametersReader.hpp>

static std::atomic_bool shtdown; ///< Shutdown requested?

/// \brief Main.
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // command line arguments
  if (argc != 2) {
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file";
    return EXIT_FAILURE;
  }

  try {

    // construct OKVIS side
    std::string configFilename(argv[1]);
    okvis::ViParametersReader viParametersReader(configFilename);
    okvis::ViParameters parameters;
    viParametersReader.getParameters(parameters);

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

    // also check DBoW2 vocabulary
    boost::filesystem::path executable(argv[0]);
    std::string dBowVocDir = executable.remove_filename().string();
    std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
    if(!infile.good()) {
       LOG(ERROR)<<"DBoW2 vocaublary " << dBowVocDir << "/small_voc.yml.gz not found.";
       return EXIT_FAILURE;
    }

    okvis::ThreadedSlam estimator(parameters, dBowVocDir);
    estimator.setBlocking(false);

    // output visualisation
    okvis::TrajectoryOutput writer;
    estimator.setOptimisedGraphCallback(
          std::bind(&okvis::TrajectoryOutput::processState, &writer,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                    std::placeholders::_4));

    // connect sensor to estimator
    realsense->setImuCallback(
          std::bind(&okvis::ThreadedSlam::addImuMeasurement, &estimator,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImuCallback(
          std::bind(&okvis::TrajectoryOutput::addImuMeasurement, &writer,
                     std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    realsense->setImagesCallback(
          std::bind(&okvis::ThreadedSlam::addImages, &estimator, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));

    // start streaming
    if(!realsense->startStreaming()) {
      return EXIT_FAILURE;
    }

    // require a special termination handler to properly close
    shtdown = false;
    signal(SIGINT, [](int) { shtdown = true; });

    // Main loop
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
      int pressed = cv::waitKey(2);
      if(pressed=='q' || shtdown) {
        break;
      }
    }
    // Stop the pipeline
    realsense->stopStreaming();
    estimator.stopThreading();

    // final BA if needed
    if(parameters.estimator.do_final_ba) {
      LOG(INFO) << "final full BA...";
      cv::Mat topView;
      estimator.doFinalBa();
      writer.drawTopView(topView);
      if (!topView.empty()) {
        cv::imshow("OKVIS 2 Top View Final", topView);
      }
      cv::waitKey(1000);
    }
  }

  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
