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

#ifndef TEST_DATA_GENERATOR_HPP
#define TEST_DATA_GENERATOR_HPP

#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
class TestDataGenerator {
 public:
  static const okvis::cameras::NCameraSystem getTestCameraSystem(int num_cams) {
    okvis::cameras::NCameraSystem nCameraSystem;
    for (int i = 0; i < num_cams; ++i) {
      Eigen::Matrix4d T_SC_e;
      T_SC_e << 1, 0, 0, i*0.01, // TODO(gohlp): does that make sense?
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0;
      std::shared_ptr<const okvis::kinematics::Transformation> T_SC_ok_ptr(
            new okvis::kinematics::Transformation(T_SC_e));
      nCameraSystem.addCamera(
            T_SC_ok_ptr,
            std::shared_ptr<const okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion>(
                  752,480,
                  615.14, 613.03,
                  378.27, 220.05,
                  okvis::cameras::RadialTangentialDistortion(
                    -0.40675345816443564, 0.1685454248925922, -0.00046944024734783504,
                    0.00026638950478762756))),
            okvis::cameras::NCameraSystem::RadialTangential);
    }
    return nCameraSystem;
  }
};
}

#endif// TEST_DATA_GENERATOR_HPP
