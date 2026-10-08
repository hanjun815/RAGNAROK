// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
//
// Copyright (c) 2024
//   School of Geodesy and Geomatics (SGG), Wuhan University, China
//   Based on: "River: A Tightly-Coupled Radar-Inertial Velocity Estimator
//              Based on Continuous-Time Optimization"
//   Upstream: https://github.com/Unsigned-Long/River
//   Author:   Shuolong Chen
//
// Copyright (c) 2025
//   Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
//
// See LICENSE for the full MIT License text.

#ifndef GARLILEO_CALIB_PARAM_MANAGER_H
#define GARLILEO_CALIB_PARAM_MANAGER_H

#include "ctraj/view/traj_viewer.h"
#include "cereal/types/map.hpp"
#include "spdlog/spdlog.h"
#include "util/cereal_archive_helper.hpp"
#include "ctraj/utils/sophus_utils.hpp"

namespace garlileo {
    class CalibParamManager {
    public:
        using Ptr = std::shared_ptr<CalibParamManager>;

    public:
        // trans radian angle to degree angle
        constexpr static double RAD_TO_DEG = 180.0 / M_PI;
        // trans degree angle to radian angle
        constexpr static double DEG_TO_RAD = M_PI / 180.0;

    public:
        // Radar0 extrinsics
        Sophus::SO3d SO3_RtoB;
        Eigen::Vector3d POS_RinB;
        // Radar1 extrinsics (second radar, used when UseRadar1 is true)
        Sophus::SO3d SO3_R1toB;
        Eigen::Vector3d POS_R1inB;

        Sophus::SO3d SO3_BtoL; //R_LI
        Eigen::Vector3d POS_BinL; //P_LI

        // Static + online time offsets (seconds).
        //   timestamp_in_imu_clock = timestamp_sensor + TIME_OFFSET_*
        //
        // The static portion (initial value loaded from yaml) is applied once
        // when the messages are buffered (see DataManager::HandleRadarMessage
        // and friends), purely to align timestamps coarsely between sensors.
        // When online temporal calibration is enabled, the *same* parameter is
        // additionally registered as a 1-d ceres parameter block so that the
        // optimiser can refine it; the corresponding factor (RadarFactor /
        // LegVelFactor) evaluates the spline at `frame.t + TIME_OFFSET_*` so a
        // non-zero update directly compensates for residual sensor delay.
        // A PriorTimeOffsetFactor anchors the parameter near the user-provided
        // value during this online refinement.
        double TIME_OFFSET_RtoB{0.0};
        double TIME_OFFSET_R1toB{0.0};
        double TIME_OFFSET_BtoL{0.0};

    public:

        // the constructor
        explicit CalibParamManager();

        // the creator
        static CalibParamManager::Ptr Create();

        // save the parameters to file using cereal library
        void
        Save(const std::string &filename, CerealArchiveType::Enum archiveType = CerealArchiveType::Enum::YAML) const;

        // load the parameters from file using cereal library
        static CalibParamManager::Ptr
        Load(const std::string &filename, CerealArchiveType::Enum archiveType = CerealArchiveType::Enum::YAML);

        // print the parameters in the console
        void ShowParamStatus();

        void VisualizationSensors(ns_viewer::Viewer &viewer) const;

        // lie algebra vector space se3
        [[nodiscard]] Sophus::SE3d SE3_RtoB() const;

        [[nodiscard]] Eigen::Quaterniond Q_RtoB() const;

        // the euler angles [radian and degree format]
        [[nodiscard]] Eigen::Vector3d EULER_RtoB_RAD() const;

        [[nodiscard]] Eigen::Vector3d EULER_RtoB_DEG() const;

    public:
        // Serialization
        template<class Archive>
        void serialize(Archive &archive) {
            archive(CEREAL_NVP(SO3_RtoB), CEREAL_NVP(POS_RinB),
                    CEREAL_NVP(SO3_R1toB), CEREAL_NVP(POS_R1inB),
                    CEREAL_NVP(SO3_BtoL), CEREAL_NVP(POS_BinL),
                    CEREAL_NVP(TIME_OFFSET_RtoB),
                    CEREAL_NVP(TIME_OFFSET_R1toB),
                    CEREAL_NVP(TIME_OFFSET_BtoL));
        }

    };
}


#endif //GARLILEO_CALIB_PARAM_MANAGER_H
