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

#ifndef MOCK_FRONTEND_HPP_
#define MOCK_FRONTEND_HPP_

#define GTEST_USE_OWN_TR1_TUPLE 0
#include "gmock/gmock.h"
#include "gtest/gtest.h"

/// \brief okvis Main namespace of this package.
namespace okvis {

class MockVioFrontendInterface{
 public:
  MOCK_METHOD4(detectAndDescribe,
      bool(size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut, const okvis::kinematics::Transformation& T_WC, const std::vector<cv::KeyPoint> * keypoints));
  MOCK_METHOD6(dataAssociationAndInitialization,
      bool(okvis::VioBackendInterface& estimator, okvis::kinematics::Transformation& T_WS_propagated, const okvis::VioParameters & params, const std::shared_ptr<okvis::MapPointVector> map, std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe));
  MOCK_CONST_METHOD8(propagation,
      bool(const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParams, okvis::kinematics::Transformation& T_WS_propagated, okvis::SpeedAndBias & speedAndBiases, const okvis::Time& t_start, const okvis::Time& t_end, Eigen::Matrix<double, 15, 15>* covariance, Eigen::Matrix<double, 15, 15>* jacobian));
  MOCK_METHOD1(setBriskDetectionOctaves,
      void(size_t octaves));
  MOCK_METHOD1(setBriskDetectionThreshold,
      void(double threshold));
};

}  // namespace okvis


#endif /* MOCK_DUMMYVIO_HPP_ */
