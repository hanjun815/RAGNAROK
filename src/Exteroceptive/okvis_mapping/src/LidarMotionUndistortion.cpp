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

#include <okvis/assert_macros.hpp>
#include <okvis/LidarMotionUndistortion.hpp>
#include <okvis/VoxelGridFilter.hpp>
#include <okvis/SubmappingUtils.hpp>

namespace okvis{
LidarMotionUndistortion::LidarMotionUndistortion(const State& initialState, const kinematics::Transformation& T_WS_live,
                                                 const kinematics::Transformation& T_SL,
                                                 const LidarMeasurementDeque& lidarScan,
                                                 const ImuMeasurementDeque& imuMeasurementDeque)
                                                 : initialState_(initialState),
                                                 T_WS_live_(T_WS_live), T_SW_live_(T_WS_live.inverse()), T_SL_(T_SL),
                                                 lidarScan_(lidarScan), imuMeasurementDeque_(imuMeasurementDeque){};

bool LidarMotionUndistortion::deskew() {
  if(initialState_.id.value() == 0){
    return false;
  }
  if(lidarScan_.empty()){
    return false;
  }
  OKVIS_ASSERT_TRUE(std::runtime_error, imuMeasurementDeque_.front().timeStamp <= initialState_.timestamp, "IMU Measurement Mismatch Motion Undistortion with State");
  OKVIS_ASSERT_TRUE(std::runtime_error, imuMeasurementDeque_.back().timeStamp >= lidarScan_.back().timeStamp, "IMU Measurement Mismatch Motion Undistortion with Lidar");

  lidarScanDeskewed_.reserve(lidarScan_.size());

  // do the propagation...
  kinematics::Transformation T_WS_0 = initialState_.T_WS;
  SpeedAndBias speedAndBias_0;
  speedAndBias_0.head<3>() = initialState_.v_W;
  speedAndBias_0.segment<3>(3) = initialState_.b_g;
  speedAndBias_0.tail<3>() = initialState_.b_a;
  kinematics::Transformation T_WS_1;
  SpeedAndBias speedAndBias_1;

  State state;
  BatchedLidarPropagator propagator(initialState_.timestamp, imuMeasurementDeque_);

  size_t i = 0;
  if(lidarScan_.front().timeStamp == initialState_.timestamp){
    state = initialState_;
    Eigen::Vector3d undistortedRay = T_SW_live_.T3x4() * state.T_WS.T() * T_SL_.T() * lidarScan_[i].measurement.rayMeasurement.homogeneous();
    lidarScanDeskewed_.push_back(undistortedRay.cast<float>());
    i++;
  }

  // Iterate measurements
  for(/*i*/; i < lidarScan_.size(); i++) {
    if(lidarScan_[i].timeStamp < initialState_.timestamp ) continue;

    propagator.appendTo(imuMeasurementDeque_,
                        T_WS_0, speedAndBias_0, lidarScan_[i].timeStamp);
    propagator.getState(T_WS_0, speedAndBias_0,
                        T_WS_1, speedAndBias_1, state.omega_S);

    state.T_WS = T_WS_1; // Transformation between World W and Sensor S.
    state.v_W = speedAndBias_1.head<3>(); // Velocity in frame W [m/s].
    state.b_g = speedAndBias_1.segment<3>(3); // Gyro bias [rad/s].
    state.b_a = speedAndBias_1.tail<3>(); // Accelerometer bias [m/s^2].
    // ToDo: The rest from here should not be needed?
    state.timestamp = lidarScan_[i].timeStamp; // Timestamp corresponding to this state.
    state.id = StateId(); // set invalid.
    state.previousImuMeasurements = initialState_.previousImuMeasurements;
    state.isKeyframe = false;

    // Write into return vector
    Eigen::Vector3d undistortedRay =
            T_SW_live_.T3x4() * state.T_WS.T() * T_SL_.T() * lidarScan_[i].measurement.rayMeasurement.homogeneous();
    lidarScanDeskewed_.push_back(undistortedRay.cast<float>());
  }
  return true;

}

bool LidarMotionUndistortion::downsample(size_t num_output_points, double voxel_grid_resolution)
{
  if(lidarScanDeskewed_.empty()){
    return false;
  }
  lidarScanDeskewedDownsampled_ = okvis::VoxelDownSample(lidarScanDeskewed_, voxel_grid_resolution);

  // Check if voxel grid filtering already decreased number sufficiently, otherwise random subsampling
  if(lidarScanDeskewedDownsampled_.size() > num_output_points){
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> tmp;
    okvis::downsamplePointCloud(lidarScanDeskewedDownsampled_, tmp, num_output_points);//todo:
    lidarScanDeskewedDownsampled_.clear();
    lidarScanDeskewedDownsampled_.insert(lidarScanDeskewedDownsampled_.end(), tmp.begin(), tmp.end());

  }
  return true;
}

size_t LidarMotionUndistortion::filterObserved(const SupereightMapType* map, const okvis::kinematics::Transformation& T_WM)
{
  if(lidarScanDeskewed_.size() == 0)
  {
    return 0;
  }

  size_t observed_counter = 0;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPoints;
  observedPoints.reserve(lidarScanDeskewed_.size());

  for (auto measurementIter = lidarScanDeskewed_.begin(); measurementIter != lidarScanDeskewed_.end(); ){
    // Transform se frame measurement into submap: T_WM.inverse() * T_WD * ray == world_to_submap*depth_to_submap * measurement_in_depth_sensor_frame
    Eigen::Vector3f pt = T_WM.inverse().T3x4().cast<float>() * T_WS_live_.T().cast<float>() * measurementIter->homogeneous();
    std::optional<float> occ = map->interpField<se::Safe::On>(pt);

    if (occ){
      observed_counter++;
      observedPoints.push_back(*measurementIter);
    }
    measurementIter++;
  }
  lidarScanDeskewed_.clear();
  lidarScanDeskewed_.insert(lidarScanDeskewed_.end(), observedPoints.begin(), observedPoints.end());

  return observed_counter;
}

}
