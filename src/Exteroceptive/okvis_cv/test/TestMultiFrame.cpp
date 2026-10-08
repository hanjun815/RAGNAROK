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
 * @file TestMultiFrame.cpp
 * @brief Runs MultiFrame tests.
 * @author Stefan Leutenegger
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <brisk/brisk.h>
#pragma GCC diagnostic pop

#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "okvis/cameras/RadialTangentialDistortion.hpp"
#include "okvis/cameras/EquidistantDistortion.hpp"
#include "okvis/MultiFrame.hpp"

TEST(MulitFrame, functions)
{

  // instantiate all possible versions of test cameras
  std::vector<std::shared_ptr<const okvis::cameras::CameraBase> > cameras;
  std::vector<okvis::cameras::NCameraSystem::DistortionType> distortions;
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>::createTestObject());
  distortions.push_back(okvis::cameras::NCameraSystem::NoDistortion);
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>
        ::createTestObject());
  distortions.push_back(okvis::cameras::NCameraSystem::RadialTangential);
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
  distortions.push_back(okvis::cameras::NCameraSystem::Equidistant);

  // the mounting transformations. The third one is opposite direction
  std::vector<std::shared_ptr<const okvis::kinematics::Transformation>> T_SC;
  T_SC.push_back(
      std::shared_ptr<okvis::kinematics::Transformation>(
          new okvis::kinematics::Transformation(Eigen::Vector3d(0.1, 0.1, 0.1),
                                                Eigen::Quaterniond(1, 0, 0, 0))));
  T_SC.push_back(
      std::shared_ptr<okvis::kinematics::Transformation>(
          new okvis::kinematics::Transformation(
              Eigen::Vector3d(0.1, -0.1, -0.1), Eigen::Quaterniond(1, 0, 0, 0))));
  T_SC.push_back(
      std::shared_ptr<okvis::kinematics::Transformation>(
          new okvis::kinematics::Transformation(
              Eigen::Vector3d(0.1, -0.1, -0.1), Eigen::Quaterniond(0, 0, 1, 0))));

  okvis::cameras::NCameraSystem nCameraSystem;
  okvis::cameras::NCameraSystem::CameraType cameraType;
  cameraType.isColour = false;
  for(size_t i=0; i<3; ++i) {
    nCameraSystem.addCamera(T_SC.at(i), cameras.at(i), distortions.at(i),
                            true, cameraType); // comp. overlaps
  }
  okvis::MultiFrame multiFrame(nCameraSystem, okvis::Time::now(), 1);

  for (size_t c = 0; c < cameras.size(); ++c) {
#ifdef __ARM_NEON__
   std::shared_ptr<cv::FeatureDetector> detector(
        new brisk::BriskFeatureDetector(34, 2));
#else
   std::shared_ptr<cv::FeatureDetector> detector(
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
            34, 2, 800, 450));
#endif

    std::shared_ptr<cv::DescriptorExtractor> extractor(
        new cv::BriskDescriptorExtractor(true, false));

    // create a stupid random image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigenImage(
        752, 480);
    eigenImage.setRandom();
    cv::Mat image(480, 752, CV_8UC1, eigenImage.data());

    // setup multifrmae
    multiFrame.setDetector(c,detector);
    multiFrame.setExtractor(c,extractor);
    multiFrame.setImage(c,image);

    // run
    multiFrame.detect(c);
    multiFrame.describe(c);
  }
}

