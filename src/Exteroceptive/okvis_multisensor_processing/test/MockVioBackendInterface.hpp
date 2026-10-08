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

#ifndef MockVioBackendInterface_HPP_
#define MockVioBackendInterface_HPP_

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <okvis/VioBackendInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

class MockVioBackendInterface : public VioBackendInterface {
 public:
  MOCK_METHOD1(addCamera,
      int(const ExtrinsicsEstimationParameters & extrinsicsEstimationParameters));
  MOCK_METHOD1(addImu,
      int(const ImuParameters & imuParameters));
  MOCK_METHOD1(addGps,
      int(const GpsParameters & gpsParameters));
  MOCK_METHOD0(clearCameras,
      void());
  MOCK_METHOD0(clearImus,
      void());
  MOCK_METHOD0(clearGpss,
      void());

  MOCK_METHOD3(addStates,
               bool(okvis::MultiFramePtr multiFrame, const okvis::ImuMeasurementDeque & imuMeasurements, bool asKeyframe));

  MOCK_METHOD2(addLandmark,
      bool(uint64_t landmarkId, const Eigen::Vector4d & landmark));
  MOCK_METHOD4(addObservation,
      ::ceres::ResidualBlockId(uint64_t landmarkId, uint64_t poseId, size_t camIdx, size_t keypointIdx));
  MOCK_METHOD4(removeObservation,
      bool(uint64_t landmarkId, uint64_t poseId, size_t camIdx, size_t keypointIdx));
  MOCK_METHOD7(addTargetMeasurement,
      bool(int targetId, uint64_t poseId, size_t camIdx,
           const okvis::kinematics::Transformation &T_SCi,
           const cameras::PinholeCamera<cameras::NoDistortion> & camera,
           const std::vector<std::pair<size_t,cv::Point2f>> & matches,
           const okvis::kinematics::Transformation *T_CT));
  MOCK_METHOD3(applyMarginalizationStrategy,
      bool(size_t numKeyframes, size_t numImuFrames, okvis::MapPointVector& removedLandmarks));
  MOCK_METHOD3(optimize,
      void(size_t, size_t, bool));
  MOCK_METHOD2(setOptimizationTimeLimit,
      bool(double timeLimit, int minIterations));
  MOCK_CONST_METHOD1(isLandmarkAdded,
      bool(uint64_t landmarkId));
  MOCK_CONST_METHOD1(isLandmarkInitialized,
      bool(uint64_t landmarkId));
  MOCK_CONST_METHOD2(getLandmark,
      bool(uint64_t landmarkId, MapPoint& mapPoint));
  MOCK_CONST_METHOD1(getLandmarks,
      size_t(PointMap & landmarks));
  MOCK_CONST_METHOD1(getLandmarks,
      size_t(okvis::MapPointVector& landmarks));
  MOCK_CONST_METHOD1(multiFrame,
                     okvis::MultiFramePtr(uint64_t frameId));
  MOCK_CONST_METHOD2(get_T_WS,
      bool(uint64_t poseId, okvis::kinematics::Transformation & T_WS));
  MOCK_CONST_METHOD6(get_T_WT,
      bool(uint64_t poseId, uint64_t targetId, okvis::kinematics::Transformation & T_WS,
           Eigen::Vector3d & v_W, Eigen::Vector3d & omega_W, double & velocityUncertainty));
  MOCK_CONST_METHOD3(getSpeedAndBias,
      bool(uint64_t poseId, uint64_t imuIdx, okvis::SpeedAndBias & speedAndBias));
  MOCK_CONST_METHOD3(getCameraSensorStates,
      bool(uint64_t poseId, size_t cameraIdx, okvis::kinematics::Transformation & T_SCi));
  MOCK_CONST_METHOD0(numFrames,
                     size_t());
  MOCK_CONST_METHOD0(numLandmarks,
      size_t());
  MOCK_CONST_METHOD0(currentKeyframeId,
      uint64_t());
  MOCK_CONST_METHOD1(frameIdByAge,
      uint64_t(size_t age));
  MOCK_CONST_METHOD0(currentFrameId,
      uint64_t());
  MOCK_CONST_METHOD1(isKeyframe,
      bool(uint64_t frameId));
  MOCK_CONST_METHOD1(isInImuWindow,
      bool(uint64_t frameId));
  MOCK_CONST_METHOD1(timestamp,
      okvis::Time(uint64_t frameId));
  MOCK_METHOD2(set_T_WS,
      bool(uint64_t poseId, const okvis::kinematics::Transformation & T_WS));
  MOCK_METHOD3(setSpeedAndBias,
      bool(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias & speedAndBias));
  MOCK_METHOD3(setCameraSensorStates,
      bool(uint64_t poseId, size_t cameraIdx, const okvis::kinematics::Transformation & T_SCi));
  MOCK_METHOD2(setLandmark,
      bool(uint64_t landmarkId, const Eigen::Vector4d & landmark));
  MOCK_METHOD2(setLandmarkInitialized,
      void(uint64_t landmarkId, bool initialized));
  MOCK_METHOD2(setKeyframe,
      void(uint64_t frameId, bool isKeyframe));
  MOCK_METHOD1(setMap,
      void(std::shared_ptr<okvis::ceres::Map> mapPtr));
  MOCK_CONST_METHOD0(initializationStatus,
      VioBackendInterface::InitializationStatus());
};

}  // namespace okvis

#endif /* MockVioBackendInterface_HPP_ */
