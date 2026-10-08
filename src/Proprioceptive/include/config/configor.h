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

#ifndef GARLILEO_CONFIGOR_H
#define GARLILEO_CONFIGOR_H

#include "cereal/archives/json.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/set.hpp"
#include "util/utils.hpp"
#include "util/enum_cast.hpp"
#include "core/calib_param_manager.h"
#include "sensor/imu_data_loader.h"
#include "sensor/radar_data_loader.h"
#include "sensor/leg_data_loader.h"
#include "sensor/foot_data_loader.h"

namespace garlileo {
    // Converts a three-element configuration list, e.g. AcceBiasInit, to a vector.
    inline Eigen::Vector3d ToVector3d(const std::vector<double> &v, const std::string &name) {
        if (v.size() != 3) {
            throw std::invalid_argument(name + " must have three elements");
        }
        return {v[0], v[1], v[2]};
    }

    // Loads a field that configuration files may omit; the default is kept then.
    template<class Archive, class T>
    void OptionalNVP(Archive &ar, const char *name, T &value) {
        if constexpr (Archive::is_loading::value) {
            try {
                ar(cereal::make_nvp(name, value));
            } catch (const cereal::Exception &) {}
        } else {
            ar(cereal::make_nvp(name, value));
        }
    }

    struct Configor {
    public:
        using Ptr = std::shared_ptr<Configor>;

    public:
        struct DataStream {
            std::string IMUTopic;
            std::string IMUMsgType;

            std::string RadarTopic;
            std::string RadarTopic1;
            // Per-radar enable flags. Set independently in yaml. Both may be false
            // (radar-less mode: only IMU + leg are used to optimize the splines).
            bool UseRadar0{true};
            bool UseRadar1{false};
            std::string RadarMsgType;

            // Leg enable flag. When false, leg/foot/joint_state subscribers are not
            // created, leg measurement factors are not added, and leg-IMU online
            // calibration / marginalisation hooks are skipped. Combined with
            // UseRadar0/UseRadar1 it allows radar-only, leg-only, IMU-only, or
            // any subset to be used at runtime via yaml.
            bool UseLeg{true};
            std::string FootTopic;
            std::string JointStateTopic{"/joint_states"};
            // Deprecated/compatibility field. Not used in current RAGNAROK mode.
            std::string LegSplineTopic{"/garlileo/spline_state"};

            CalibParamManager CalibParam;

            std::string OutputPath;

        public:
            template<class Archive>
            void serialize(Archive &ar) {
                ar(CEREAL_NVP(IMUTopic), CEREAL_NVP(IMUMsgType),
                   CEREAL_NVP(RadarTopic), CEREAL_NVP(RadarTopic1),
                   CEREAL_NVP(UseRadar0), CEREAL_NVP(UseRadar1),
                   CEREAL_NVP(UseLeg),
                   CEREAL_NVP(FootTopic),
                   CEREAL_NVP(JointStateTopic), CEREAL_NVP(RadarMsgType),
                   CEREAL_NVP(LegSplineTopic),
                   CEREAL_NVP(CalibParam), CEREAL_NVP(OutputPath));
            }
        } dataStream;

        struct Prior {
            static constexpr int SplineOrder = 3;

            double GravityNorm;
            double GravityDirection;

            double SO3SplineKnotDist;
            double VelSplineKnotDist;
            double GravSplineKnotDist;

            double AcceWeight;
            double AcceBiasRandomWalk;
            double GyroWeight;
            double GyroBiasRandomWalk;
            double RadarWeight;
            double LegWeight;
            // Parameters of the VILENS-like leg velocity model (LegOdometryProcessor).
            double LegEncoderPosSigma{0.003};
            double LegEncoderVelSigma{0.050};
            // Per-leg pairwise inter-leg disagreement scale (mean_{j!=i} (v_i-v_j)(v_i-v_j)^T).
            double LegInterLegCovarianceScale{2.0};
            // Per-leg effort discontinuity scale (each leg uses its OWN |Δτ_i/dt|).
            double LegForceDiscontinuityScale{0.001};
            // Per-leg stance-age covariance inflation:
            //   Σ_i += I * (LegStanceAgeCovarianceScale / max(stance_age_i, LegStanceAgeFloorSec))
            // The covariance is therefore large immediately after a leg lands and
            // decays the longer the leg has been in contact.
            double LegStanceAgeCovarianceScale{0.05};
            double LegStanceAgeFloorSec{0.02};
            double LegVelocityCovarianceFloor{1.0e-4};
            double LegOmegaStaleSec{0.05};
            // Dynamic leg weighting in GaRLILEO optimization based on measured leg covariance.
            bool LegDynamicWeightEnable{false};
            double LegDynamicWeightReferenceCovTrace{0.03};
            double LegDynamicWeightMinScale{0.2};
            double LegDynamicWeightMaxScale{5.0};
            bool LegRollingContactEnable{false};
            double LegRollingContactFootRadius{0.01};

            double CauchyLossForRadarFactor;

            // K-AOC (Kinematics-Aware Orthogonal Covariance) radar noise parameters
            double RadarDopplerNoiseStd{0.05};
            double RadarAzimuthNoiseBase{0.05};
            double RadarElevationNoiseBase{0.066};
            double RadarSpatialDegradationCoeff{0.5};
            bool RadarUseKAOC{true};

            // Normalize the K-AOC weight to the Doppler-only noise floor
            // (sigma_vd * sqrt(Omega)) and clamp it to [min, max].
            bool RadarKAOCClipEnable{false};
            double RadarKAOCSqrtOmegaMin{0.8};
            double RadarKAOCSqrtOmegaMax{1.2};

            double RotGravWeight;
            double GravityWeight;

            // Constant IMU biases. AcceBiasInit seeds b_a; with UseGyroBias,
            // GyroBiasInit is subtracted from every gyroscope sample.
            std::vector<double> AcceBiasInit{0.0, 0.0, 0.0};
            double AcceBiasPriorVariance{1.0e-8};
            std::vector<double> GyroBiasInit{0.0, 0.0, 0.0};
            bool UseGyroBias{false};

            // -----------------------------
            // Online radar-IMU calibration
            // -----------------------------
            bool RadarImuOnlineCalibEnable{false};
            bool RadarImuOnlineCalibOptimizeSO3_RtoB{false};
            bool RadarImuOnlineCalibOptimizePOS_RinB{false};
            // iKalibr-style temporal calibration: refine TIME_OFFSET_RtoB online.
            bool RadarImuOnlineCalibOptimizeTO_RtoB{false};
            // Soft priors to keep extrinsics near initial guess (weight is applied directly to residual).
            double RadarImuOnlineCalibPriorWeightSO3{0.0};
            double RadarImuOnlineCalibPriorWeightPOS{0.0};
            double RadarImuOnlineCalibPriorWeightTO{0.0};
            // Hard search bound for TIME_OFFSET_RtoB (seconds, symmetric around 0).
            // Should be at most ~half of SO3SplineKnotDist to avoid pushing the
            // jet-typed spline-meta query outside its single-knot segment, which
            // would silently disable the factor (see RadarFactor::QueryInsideSplineMeta).
            // Recommended: ~half of one radar period (radar 20Hz -> 0.025s).
            double RadarImuOnlineCalibTOMaxAbs{0.020};

            // Second radar online calibration (only effective when UseRadar1: true)
            bool Radar1ImuOnlineCalibEnable{false};
            bool Radar1ImuOnlineCalibOptimizeSO3_R1toB{false};
            bool Radar1ImuOnlineCalibOptimizePOS_R1inB{false};
            bool Radar1ImuOnlineCalibOptimizeTO_R1toB{false};
            double Radar1ImuOnlineCalibPriorWeightSO3{0.0};
            double Radar1ImuOnlineCalibPriorWeightPOS{0.0};
            double Radar1ImuOnlineCalibPriorWeightTO{0.0};
            double Radar1ImuOnlineCalibTOMaxAbs{0.020};

            // -----------------------------
            // Online leg-IMU calibration
            // -----------------------------
            bool LegImuOnlineCalibEnable{false};
            bool LegImuOnlineCalibOptimizeSO3_BtoL{false};
            bool LegImuOnlineCalibOptimizePOS_BinL{false};
            bool LegImuOnlineCalibOptimizeTO_BtoL{false};
            // Soft priors to keep extrinsics near initial guess (weight is applied directly to residual).
            double LegImuOnlineCalibPriorWeightSO3{0.0};
            double LegImuOnlineCalibPriorWeightPOS{0.0};
            double LegImuOnlineCalibPriorWeightTO{0.0};
            // Hard search bound for TIME_OFFSET_BtoL (seconds, symmetric around 0).
            // For leg @ 150Hz keep this small (e.g. 0.005s) so the optimizer
            // cannot shift more than a couple of leg sample periods.
            double LegImuOnlineCalibTOMaxAbs{0.005};

            // -----------------------------
            // Online calibration logging
            // -----------------------------
            // Period (in incremental optimisation windows) at which the latest
            // calibration estimate (extrinsic + time offset) is dumped via
            // spdlog::info. Set to 0 to disable. Default = every 20 windows.
            int OnlineCalibLogEveryNWindows{20};

        public:
            template<class Archive>
            void serialize(Archive &ar) {
                ar(
                        CEREAL_NVP(GravityNorm), CEREAL_NVP(GravityDirection),
                        CEREAL_NVP(SO3SplineKnotDist), CEREAL_NVP(VelSplineKnotDist), CEREAL_NVP(GravSplineKnotDist),
                        CEREAL_NVP(AcceWeight), CEREAL_NVP(AcceBiasRandomWalk),
                        CEREAL_NVP(GyroWeight), CEREAL_NVP(GyroBiasRandomWalk),
                        CEREAL_NVP(RadarWeight), CEREAL_NVP(LegWeight),
                        CEREAL_NVP(LegEncoderPosSigma), CEREAL_NVP(LegEncoderVelSigma),
                        CEREAL_NVP(LegInterLegCovarianceScale), CEREAL_NVP(LegForceDiscontinuityScale),
                        CEREAL_NVP(LegStanceAgeCovarianceScale), CEREAL_NVP(LegStanceAgeFloorSec),
                        CEREAL_NVP(LegVelocityCovarianceFloor), CEREAL_NVP(LegOmegaStaleSec),
                        CEREAL_NVP(LegDynamicWeightEnable), CEREAL_NVP(LegDynamicWeightReferenceCovTrace),
                        CEREAL_NVP(LegDynamicWeightMinScale), CEREAL_NVP(LegDynamicWeightMaxScale),
                        CEREAL_NVP(LegRollingContactEnable), CEREAL_NVP(LegRollingContactFootRadius),
                        CEREAL_NVP(CauchyLossForRadarFactor),
                        CEREAL_NVP(RadarDopplerNoiseStd), CEREAL_NVP(RadarAzimuthNoiseBase),
                        CEREAL_NVP(RadarElevationNoiseBase), CEREAL_NVP(RadarSpatialDegradationCoeff),
                        CEREAL_NVP(RadarUseKAOC),
                        CEREAL_NVP(RotGravWeight), CEREAL_NVP(GravityWeight),
                        CEREAL_NVP(RadarImuOnlineCalibEnable),
                        CEREAL_NVP(RadarImuOnlineCalibOptimizeSO3_RtoB),
                        CEREAL_NVP(RadarImuOnlineCalibOptimizePOS_RinB),
                        CEREAL_NVP(RadarImuOnlineCalibOptimizeTO_RtoB),
                        CEREAL_NVP(RadarImuOnlineCalibPriorWeightSO3),
                        CEREAL_NVP(RadarImuOnlineCalibPriorWeightPOS),
                        CEREAL_NVP(RadarImuOnlineCalibPriorWeightTO),
                        CEREAL_NVP(RadarImuOnlineCalibTOMaxAbs),
                        CEREAL_NVP(Radar1ImuOnlineCalibEnable),
                        CEREAL_NVP(Radar1ImuOnlineCalibOptimizeSO3_R1toB),
                        CEREAL_NVP(Radar1ImuOnlineCalibOptimizePOS_R1inB),
                        CEREAL_NVP(Radar1ImuOnlineCalibOptimizeTO_R1toB),
                        CEREAL_NVP(Radar1ImuOnlineCalibPriorWeightSO3),
                        CEREAL_NVP(Radar1ImuOnlineCalibPriorWeightPOS),
                        CEREAL_NVP(Radar1ImuOnlineCalibPriorWeightTO),
                        CEREAL_NVP(Radar1ImuOnlineCalibTOMaxAbs),
                        CEREAL_NVP(LegImuOnlineCalibEnable),
                        CEREAL_NVP(LegImuOnlineCalibOptimizeSO3_BtoL),
                        CEREAL_NVP(LegImuOnlineCalibOptimizePOS_BinL),
                        CEREAL_NVP(LegImuOnlineCalibOptimizeTO_BtoL),
                        CEREAL_NVP(LegImuOnlineCalibPriorWeightSO3),
                        CEREAL_NVP(LegImuOnlineCalibPriorWeightPOS),
                        CEREAL_NVP(LegImuOnlineCalibPriorWeightTO),
                        CEREAL_NVP(LegImuOnlineCalibTOMaxAbs),
                        CEREAL_NVP(OnlineCalibLogEveryNWindows)
                );
                OptionalNVP(ar, "RadarKAOCClipEnable", RadarKAOCClipEnable);
                OptionalNVP(ar, "RadarKAOCSqrtOmegaMin", RadarKAOCSqrtOmegaMin);
                OptionalNVP(ar, "RadarKAOCSqrtOmegaMax", RadarKAOCSqrtOmegaMax);
                OptionalNVP(ar, "AcceBiasInit", AcceBiasInit);
                OptionalNVP(ar, "AcceBiasPriorVariance", AcceBiasPriorVariance);
                OptionalNVP(ar, "GyroBiasInit", GyroBiasInit);
                OptionalNVP(ar, "UseGyroBias", UseGyroBias);
            }
        } prior{};

        struct Preference {
            /**
             * when the mode is 'DEBUG_MODE', then:
             * 1. the solving information from ceres would be output on the console
             * 2.
             */
            static bool DEBUG_MODE;

            static std::string SO3Spline;
            static std::string VelSpline;

            // these two splines are not used currently
            static std::string BaSpline;
            static std::string BgSpline;
            
            static std::string GravitySpline;

            static std::string PublishTopic;

            static double StatePublishDelay;

            std::uint32_t IMUMsgQueueSize;
            std::uint32_t RadarMsgQueueSize;
            std::uint32_t FootMsgQueueSize;
            std::uint32_t JointStateMsgQueueSize{800};
            std::uint32_t IncrementalOptRate;
            std::uint32_t PublishRate = 50;  ///< Publishing rate for spline state and odometry [Hz]. Independent of IncrementalOptRate.
            double SplineStateSafetyLagSec{0.30};  ///< Delay /garlileo/spline_state publication until this much stable spline history is available.

            bool OutputResultsWithTimeAligned;


        public:
            template<class Archive>
            void serialize(Archive &ar) {
                ar(
                        CEREAL_NVP(IMUMsgQueueSize), CEREAL_NVP(RadarMsgQueueSize), CEREAL_NVP(FootMsgQueueSize),
                        CEREAL_NVP(JointStateMsgQueueSize),
                        CEREAL_NVP(IncrementalOptRate), CEREAL_NVP(PublishRate),
                        CEREAL_NVP(SplineStateSafetyLagSec), CEREAL_NVP(OutputResultsWithTimeAligned)
                );
            }
        } preference{};

    public:
        Configor();

        static Ptr Create();

        // load configure information from the xml file
        static Configor::Ptr
        Load(const std::string &filename, CerealArchiveType::Enum archiveType = CerealArchiveType::Enum::YAML);

        // load configure information from the xml file
        void Save(const std::string &filename, CerealArchiveType::Enum archiveType = CerealArchiveType::Enum::YAML);

        // print the main fields
        void PrintMainFields();

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(
                    cereal::make_nvp("DataStream", dataStream),
                    cereal::make_nvp("Prior", prior),
                    cereal::make_nvp("Preference", preference)
            );
        }
    };
}


#endif //GARLILEO_CONFIGOR_H
