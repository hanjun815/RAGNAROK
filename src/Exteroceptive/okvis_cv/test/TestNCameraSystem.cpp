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
 * @file TestNCameraSystem.cpp
 * @brief Runs NCameraSystem tests.
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
#include "okvis/cameras/NCameraSystem.hpp"

TEST(NCameraSystem, functions)
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

  // verify self overlaps
  OKVIS_ASSERT_TRUE(std::runtime_error, nCameraSystem.hasOverlap(0, 0), "No self overlap?")
  OKVIS_ASSERT_TRUE(std::runtime_error, nCameraSystem.hasOverlap(1, 1), "No self overlap?")
  OKVIS_ASSERT_TRUE(std::runtime_error, nCameraSystem.hasOverlap(2, 2), "No self overlap?")

  // verify 0 and 1 overlap
  OKVIS_ASSERT_TRUE(std::runtime_error, nCameraSystem.hasOverlap(0, 1), "No overlap?")
  OKVIS_ASSERT_TRUE(std::runtime_error, nCameraSystem.hasOverlap(1, 0), "No overlap?")

  // verify 1 and 2 do not overlap
  OKVIS_ASSERT_TRUE(std::runtime_error, !nCameraSystem.hasOverlap(1, 2), "Overlap?")
  OKVIS_ASSERT_TRUE(std::runtime_error, !nCameraSystem.hasOverlap(2, 1), "Overlap?")

  // verify 0 and 2 do not overlap
  OKVIS_ASSERT_TRUE(std::runtime_error, !nCameraSystem.hasOverlap(0, 2), "Overlap?")
  OKVIS_ASSERT_TRUE(std::runtime_error, !nCameraSystem.hasOverlap(2, 0), "Overlap?")

}

