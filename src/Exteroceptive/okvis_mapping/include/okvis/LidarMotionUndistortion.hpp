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

#ifndef INCLUDE_OKVIS_LIDARMOTIONUNDISTORTION_HPP
#define INCLUDE_OKVIS_LIDARMOTIONUNDISTORTION_HPP

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/mapTypedefs.hpp>
#include <se/map/map.hpp>

namespace okvis{
    class LidarMotionUndistortion{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LidarMotionUndistortion() = delete;

        /// \brief Default move Constructor.
        LidarMotionUndistortion(LidarMotionUndistortion && ) = default;

        /// \brief Default copy Constructor.
        LidarMotionUndistortion(const LidarMotionUndistortion & ) = default;

        /// \brief Constructor
        LidarMotionUndistortion(const State& initialState, const kinematics::Transformation& T_WS_live,
                                const kinematics::Transformation& T_SL,
                                const LidarMeasurementDeque& lidarMeasurementDeque,
                                const ImuMeasurementDeque& imuMeasurementDeque);

        /// \brief Function doing motion undistortion based on latest state and imu measurements
        bool deskew();

        /// \brief Return reference to deskewed point cloud
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> & deskewedPointCloud(){
          return lidarScanDeskewed_;
        }

        /// \brief Function doing voxel hash grid based downsampling
        bool downsample(size_t num_output_points, double voxel_grid_resolution=0.1);

        /// \brief Return reference to downsampled, deskewed point cloud
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& deskewedDownsampledPointCloud(){
          return lidarScanDeskewedDownsampled_;
        }

        /// \brief Check which points are actually observed
        /// @param map
        /// @param T_WM
        size_t filterObserved(const SupereightMapType* map, const okvis::kinematics::Transformation& T_WM);

    private:

        State initialState_; /// < T_WS0, sb_0, biases_0, state propagated from
        kinematics::Transformation T_WS_live_; ///< Pose of the frame propagated to
        kinematics::Transformation T_SW_live_; ///< Inverse
        kinematics::Transformation T_SL_; ///< Extrinsics IMU (S) <-> Lidar (L)
        LidarMeasurementDeque lidarScan_;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> lidarScanDeskewed_;
        //LidarMeasurementDeque lidarScanDeskewed_;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> lidarScanDeskewedDownsampled_;
        //LidarMeasurementDeque lidarScanDeskewedDownsampled_;
        ImuMeasurementDeque imuMeasurementDeque_;

    };
}

#endif //INCLUDE_OKVIS_LIDARMOTIONUNDISTORTION_HPP
