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

#include "core/state_manager.h"
#include "core/incremental_window_readiness.h"
#include <mutex>
#include <set>

#include <utility>
#include <vector>
#include <string>
#include "core/estimator.h"
#include "spdlog/stopwatch.h"
#include "spdlog/fmt/fmt.h"
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <fstream>
 

namespace garlileo {

    // -------------------
    // static member field
    // -------------------
    std::mutex StateManager::StatesMutex = {};
    double st = 0;
    long update_start_index = -1; 
    long update_end_index = -1; 

    StateManager::StateManager(DataManager::Ptr dataMagr, Configor::Ptr configor)
            : dataMagr(std::move(dataMagr)), configor(std::move(configor)), splines(nullptr),
              gravity(std::make_shared<Eigen::Vector3d>(0.0, 0.0, -this->configor->prior.GravityNorm)),
              ba(std::make_shared<Eigen::Vector3d>(ToVector3d(this->configor->prior.AcceBiasInit, "AcceBiasInit"))),
              bg(std::make_shared<Eigen::Vector3d>(0.0, 0.0, 0.0)), 
              bv(std::make_shared<Eigen::Vector2d>(0.0, 0.0)), margInfo(nullptr) {}

    StateManager::Ptr StateManager::Create(const DataManager::Ptr &dataMagr, const Configor::Ptr &configor) {
        return std::make_shared<StateManager>(dataMagr, configor);
    }

    void StateManager::Run() {
        spdlog::info("'StateManager::Run' has been booted, thread id: {}.", GARLILEO_TO_STR(std::this_thread::get_id()));

        rclcpp::Rate rate(configor->preference.IncrementalOptRate);

        while (rclcpp::ok()) {
            auto status = GaRLILEOStatus::GetStatusPackSafely();
            if (GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::ShouldQuit, status.StateMagr)) {
                spdlog::warn("'StateManager::Run' quits normally.");
                break;
            }

            if (!GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::HasInitialized, status.StateMagr)) { 
                // if this system has not been initialized
                if (GaRLILEOStatus::IsWith(GaRLILEOStatus::DataManager::Status::DataReadyForInit, status.DataMagr)) {
                    // if this system is ready for initialization
                    {
                        LOCK_GARLILEO_STATUS
                        GaRLILEOStatus::DataManager::CurStatus ^= GaRLILEOStatus::DataManager::Status::DataReadyForInit;
                     }
                    spdlog::stopwatch sw;
                    auto valid = TryPerformInitialization();
                    if (valid) { spdlog::info("total time elapsed in initialization: {} (s).", sw); }
                } else {
                    spdlog:: info("preparing initialization...");
                }
            } else {
                spdlog::stopwatch sw;
                auto valid = IncrementalOptimization(status);
                if (valid) {
                    // spdlog::info("total time elapsed in current incremental optimization: {} (s). \n", sw);
                }
            }

            rate.sleep();
        }
    }

    // obtain the rotation from aligned {w} to {ref} based on 'SO3_B0ToRef' and 'gravityInRef'
    Sophus::SO3d StateManager::ObtainAlignedWtoRef(const Sophus::SO3d &SO3_B0ToRef, const Eigen::Vector3d &gravityInRef) {
        // Robustly compute the minimal rotation that aligns "down axis" (zNegAxis) with gravity direction.
        // The previous implementation could produce NaNs when the vectors are nearly parallel/anti-parallel.
        constexpr double kEps = 1.0e-12;
        constexpr double kAxisEps = 1.0e-8;
        constexpr double kPi = 3.14159265358979323846;

        const Eigen::Vector3d zNegAxis = configor->prior.GravityDirection * SO3_B0ToRef.matrix().col(2);
        const double zn = zNegAxis.norm();
        const double gn = gravityInRef.norm();
        if (zn < kEps || gn < kEps) {
            // Fallback: no alignment possible.
            return SO3_B0ToRef;
        }

        const Eigen::Vector3d z_dir = zNegAxis / zn;
        const Eigen::Vector3d g_dir = gravityInRef / gn;

        double cos_ang = z_dir.dot(g_dir);
        // Clamp to avoid acos domain errors due to numeric precision.
        if (cos_ang > 1.0) cos_ang = 1.0;
        if (cos_ang < -1.0) cos_ang = -1.0;

        Eigen::Vector3d axis = z_dir.cross(g_dir);
        const double axis_norm = axis.norm();

        Sophus::SO3d R_align;
        if (axis_norm < kAxisEps) {
            // z_dir and g_dir are (nearly) aligned or anti-aligned.
            if (cos_ang > 0.0) {
                // Already aligned: no rotation.
                R_align = Sophus::SO3d();
            } else {
                // 180deg rotation around any axis orthogonal to z_dir.
                axis = z_dir.cross(Eigen::Vector3d::UnitX());
                if (axis.norm() < kAxisEps) {
                    axis = z_dir.cross(Eigen::Vector3d::UnitY());
                }
                axis.normalize();
                R_align = Sophus::SO3d(Eigen::AngleAxisd(kPi, axis).toRotationMatrix());
            }
        } else {
            axis /= axis_norm;
            const double angRad = std::acos(cos_ang);
            R_align = Sophus::SO3d(Eigen::AngleAxisd(angRad, axis).toRotationMatrix());
        }

        return R_align * SO3_B0ToRef;
    }

    bool StateManager::TryPerformInitialization() {
        const bool useRadar0 = configor->dataStream.UseRadar0;

        // Decide initialization window [sTime, eTime] (in internal/IMU-aligned time).
        //   - radar0 enabled : derive from accumulated radar frames (legacy behaviour)
        //   - radar0 disabled: derive from currently buffered IMU sequence (radar-less mode)
        std::list<RadarTargetArray::Ptr> radarTarAryForInit;
        double sTimeRadar = 0.0, eTimeRadar = 0.0;
        if (useRadar0) {
            radarTarAryForInit = dataMagr->GetRadarTarAryForInitSafely();
            if (radarTarAryForInit.empty()) {
                spdlog::warn("[Init] DataReadyForInit fired but radarTarAryForInit is empty; skip.");
                return false;
            }
            sTimeRadar = radarTarAryForInit.front()->GetTargets().front()->GetTimestamp();
            eTimeRadar = radarTarAryForInit.back()->GetTargets().back()->GetTimestamp();
        }

        // Pull IMU/leg pieces.
        auto imuDataSeq = useRadar0
                              ? dataMagr->ExtractIMUDataPieceSafely(sTimeRadar, eTimeRadar)
                              : dataMagr->ExtractIMUDataPieceSafely(0.0);
        if (imuDataSeq.empty()) {
            spdlog::warn("[Init] no IMU frames available for initialization; skip.");
            return false;
        }

        const double sTimeImu = imuDataSeq.front()->GetTimestamp();
        const double eTimeImu = imuDataSeq.back()->GetTimestamp();
        std::list<LegFrame::Ptr> legDataSeq;
        if (configor->dataStream.UseLeg) {
            legDataSeq = dataMagr->ExtractLegDataPieceSafely(sTimeImu, eTimeImu);
        }

        // ----------------
        // static condition
        // ----------------
        const double sTime = useRadar0 ? std::max(sTimeRadar, sTimeImu) : sTimeImu;
        const double eTime = useRadar0 ? std::min(eTimeRadar, eTimeImu) : eTimeImu;

        spdlog::info(
                "'initialization': start time: {:.6f}, end time: {:.6f}, duration: {:.6f} "
                "(radar0={}, radar1={}, leg={})",
                sTime, eTime, eTime - sTime,
                useRadar0, configor->dataStream.UseRadar1, configor->dataStream.UseLeg
        );

        if (Configor::Preference::DEBUG_MODE) {
            if (useRadar0) {
                spdlog::info(
                        "'radarTarAryForInit' size: {}, start time: {:.6f}, end time: {:.6f}, duration: {:.6f}",
                        radarTarAryForInit.size(), sTimeRadar, eTimeRadar, eTimeRadar - sTimeRadar
                );
            }
            spdlog::info(
                    "involved IMU frame size: {}, start time: {:.6f}, end time: {:.6f}, duration: {:.6f}",
                    imuDataSeq.size(), imuDataSeq.front()->GetTimestamp(), imuDataSeq.back()->GetTimestamp(),
                    imuDataSeq.back()->GetTimestamp() - imuDataSeq.front()->GetTimestamp()
            );
            if (!configor->dataStream.UseLeg) {
                spdlog::info("involved Leg frame size: 0 (UseLeg=false, leg disabled)");
            } else if (!legDataSeq.empty()) {
                spdlog::info(
                        "involved Leg frame size: {}, start time: {:.6f}, end time: {:.6f}, duration: {:.6f}",
                        legDataSeq.size(), legDataSeq.front()->GetTimestamp(), legDataSeq.back()->GetTimestamp(),
                        legDataSeq.back()->GetTimestamp() - legDataSeq.front()->GetTimestamp()
                );
            } else {
                spdlog::info("involved Leg frame size: 0 (no leg measurements yet)");
            }
        }

        // create splines
        splines = CreateSplines(sTime, eTime);
        if (Configor::Preference::DEBUG_MODE) { ShowSplineStatus(); }

        // init filters
        InitializeBiasFilters(sTime);

        // ---------------------
        // initialize so3 spline
        // ---------------------
        spdlog::stopwatch sw;
        InitializeSO3Spline(imuDataSeq);
        spdlog::info("initialized rotation spline time elapsed: {} (s).", sw);

        // -------------------------
        // initialize global gravity vector
        // -------------------------
        spdlog::stopwatch sw2;
        // OKVIS2-X compatible: use IMU samples around the first camera timestamp t0
        // with overlap ±0.02s (imuTemporalOverlap in OKVIS2-X).
        std::list<IMUFrame::Ptr> imuForGravity = imuDataSeq;
        if (const auto t0_opt = dataMagr->GetOkvisInitTimeSafely(); t0_opt) {
            const double overlap = 0.02;
            imuForGravity = dataMagr->ExtractIMUDataPieceSafely(*t0_opt - overlap, *t0_opt + overlap);
            if (imuForGravity.empty()) {
                spdlog::warn(
                        "[OKVIS-like] IMU window for gravity init is empty for t0={:.6f} (internal). "
                        "Falling back to radar init IMU segment.",
                        *t0_opt
                );
                imuForGravity = imuDataSeq;
            }
        } else {
            spdlog::warn(
                    "[OKVIS-like] t0 (first camera timestamp with IMU coverage) not available. "
                    "Falling back to radar init IMU segment for gravity init."
            );
        }
        InitializeGravity(radarTarAryForInit, imuForGravity);

        // observability condition is good, the gravity has been initialized
        spdlog::info(
                "initialized gravity vector: 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f}",
                (*gravity)(0), (*gravity)(1), (*gravity)(2)
        );

        // -----------------------
        // recover velocity spline
        // -----------------------
        spdlog::stopwatch sw3;
        
        auto estimator = InitializeVelSpline(radarTarAryForInit, imuDataSeq, legDataSeq);

        spdlog::info("initialize velocity spline time elapsed: {} (s)", sw3);


        spdlog::info(
                "refined gravity vector: 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f}",
                (*gravity)(0), (*gravity)(1), (*gravity)(2)
        );

        // ------------------
        // update bias filter
        // ------------------
        UpdateBiasFilters(estimator, eTime);

        // -----------------------
        // align states to gravity
        // -----------------------
        AlignInitializedStates();

        spdlog::info(
                "aligned gravity vector: 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f}",
                (*gravity)(0), (*gravity)(1), (*gravity)(2)
        );

        // ---------------------------------
        // marginalization in initialization
        // ---------------------------------
        spdlog::stopwatch sw4;
        
        MarginalizationInInit(estimator);
        
        spdlog::info("marginalization in initialization time elapsed: {} (s)", sw4);
        
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        std::size_t num_knots = 0;
        while (num_knots<so3Spline.GetKnots().size()) {
            gravSpline.GetKnot(num_knots) = (so3Spline.GetKnot(num_knots).matrix().transpose()) * (*gravity);
            num_knots++;
        }

        LOCK_GARLILEO_STATUS
        GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::HasInitialized;
        GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::NewStateNeedToDraw;
        GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::NewStateNeedToPublish;
        GaRLILEOStatus::StateManager::ValidStateEndTime = eTime;

        return true;
    }

    StateManager::SplineBundleType::Ptr StateManager::CreateSplines(double sTime, double eTime) const {
        // create four splines
        auto so3SplineInfo = ns_ctraj::SplineInfo(
                Configor::Preference::SO3Spline, ns_ctraj::SplineType::So3Spline,
                sTime, eTime, configor->prior.SO3SplineKnotDist
        );
        auto velSplineInfo = ns_ctraj::SplineInfo(
                Configor::Preference::VelSpline, ns_ctraj::SplineType::RdSpline,
                sTime, eTime, configor->prior.VelSplineKnotDist
        );

        auto gravitySplineInfo = ns_ctraj::SplineInfo(
                Configor::Preference::GravitySpline, ns_ctraj::SplineType::RdSpline,
                sTime, eTime, configor->prior.GravSplineKnotDist
        );
        return SplineBundleType::Create({so3SplineInfo, velSplineInfo, gravitySplineInfo});
    }

    void StateManager::InitializeSO3Spline(const std::list<IMUFrame::Ptr> &imuData) {
        // add gyro factors and fit so3 spline
        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);
        for (const auto &frame: imuData) {
            estimator->AddGyroMeasurementWithConstBias(frame, GaRLILEO_Opt::OPT_SO3, configor->prior.GyroWeight);
        } 

        // make this problem fun rank
        estimator->SetParameterBlockConstant(
                splines->GetSo3Spline(Configor::Preference::SO3Spline).KnotsFront().data()
        );

        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, Configor::Preference::DEBUG_MODE, false));

        if (Configor::Preference::DEBUG_MODE) {
            spdlog::info("here is the summary of 'InitializeSO3Spline':\n{}\n", sum.BriefReport());
        }
    }

    void StateManager::InitializeGravity(const std::list<RadarTargetArray::Ptr> &radarTarAryVec,
                                         const std::list<IMUFrame::Ptr> &imuData) {
        (void)radarTarAryVec;

        // OKVIS2-X style: use raw IMU accelerometer mean (body frame) to initialise gravity direction.
        // Do NOT rotate with any spline here to keep the world definition consistent with OKVIS2-X.
        Eigen::Vector3d acc_sum_B = Eigen::Vector3d::Zero();
        int acc_cnt = 0;
        for (const auto &frame: imuData) {
            acc_sum_B += frame->GetAcce();
            ++acc_cnt;
        }

        if (acc_cnt <= 0) {
            spdlog::warn(
                    "InitializeGravity(OKVIS-style): no IMU samples. "
                    "Falling back to default gravity [0,0,-g]."
            );
            *gravity = Eigen::Vector3d(0.0, 0.0, -configor->prior.GravityNorm);
            return;
        }

        const Eigen::Vector3d acc_mean_B = acc_sum_B / static_cast<double>(acc_cnt);
        if (acc_mean_B.norm() < 1.0e-6) {
            spdlog::warn(
                    "InitializeGravity(OKVIS-style): mean accelerometer norm too small ({:.3e}). "
                    "Falling back to default gravity [0,0,-g].",
                    acc_mean_B.norm()
            );
            *gravity = Eigen::Vector3d(0.0, 0.0, -configor->prior.GravityNorm);
            return;
        }

        // Accelerometer measures specific force. At rest it points opposite to gravity.
        // Therefore gravity direction is the opposite of the mean accelerometer direction.
        // We initialise gravity in the (initial) reference frame consistently with OKVIS2-X.
        *gravity = -acc_mean_B.normalized() * configor->prior.GravityNorm;

        spdlog::info(
                "initialized gravity vector (OKVIS-style): 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f} "
                "(from {} imu frames, |mean_f_B|={:.3f})",
                (*gravity)(0), (*gravity)(1), (*gravity)(2),
                acc_cnt, acc_mean_B.norm()
        );

        // // obtain extrinsics
        // const auto &SO3_RtoB = configor->dataStream.CalibParam.SO3_RtoB;
        // const Eigen::Vector3d &POS_RinB = configor->dataStream.CalibParam.POS_RinB;

        // // obtain the fitted so3 spline
        // const auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);

        // // timestamp, imu velocity computed from radar measurements
        // std::vector<std::pair<double, Eigen::Vector3d>> LIN_VEL_BtoB0inB0_VEC;

        // // compute the velocity of imu (with respect to the reference frame expressed in the reference frame)
        // for (const auto &tarAry: radarTarAryVec) {
        //     double t = tarAry->GetTimestamp();

        //     if (t < so3Spline.MinTime() || t >= so3Spline.MaxTime()) { continue; }

        //     auto SO3_BtoB0 = so3Spline.Evaluate(t);
        //     Eigen::Vector3d ANG_VEL_BtoB0inB0 = SO3_BtoB0 * so3Spline.VelocityBody(t);

        //     Sophus::SO3d SO3_RtoB0 = SO3_BtoB0 * SO3_RtoB;
        //     Eigen::Vector3d LIN_VEL_RtoB0InB0 = tarAry->RadarVelocityFromStaticTargetArray(SO3_RtoB0);

        //     Eigen::Vector3d LIN_VEL_BtoB0inB0 =
        //             LIN_VEL_RtoB0InB0 + Sophus::SO3d::hat(SO3_BtoB0 * POS_RinB) * ANG_VEL_BtoB0inB0;

        //     LIN_VEL_BtoB0inB0_VEC.emplace_back(t, LIN_VEL_BtoB0inB0);
        // }
        // Eigen::Vector3d mean;
        // Eigen::MatrixXd var;
        // Eigen::MatrixXd matrix(LIN_VEL_BtoB0inB0_VEC.size(), 3);
        // auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);
        // for (int i = 0; i < static_cast<int>(LIN_VEL_BtoB0inB0_VEC.size()) - 1; ++i) {
        //     int j = i + 1;
        //     const auto &[ti, vi] = LIN_VEL_BtoB0inB0_VEC.at(i);
        //     const auto &[tj, vj] = LIN_VEL_BtoB0inB0_VEC.at(j);
        //     matrix.row(i++) = vi;
        //     auto [sIter, eIter] = ExtractRange(imuData, ti, tj);
        //     std::vector<std::pair<double, Eigen::Vector3d>> velData;
        //     for (auto iter = sIter; iter != eIter; ++iter) {
        //         const auto &frame = *iter;
        //         double t = frame->GetTimestamp();
        //         velData.emplace_back(t, so3Spline.Evaluate(t) * frame->GetAcce());
        //     }
        //     Eigen::Vector3d velPIM = TrapIntegrationOnce(velData);
        //     estimator->AddVelPIMForGravityRecovery(tj - ti, vj - vi, velPIM, GaRLILEO_Opt::OPT_GRAVITY, 1.0);
        // }

        // if (LIN_VEL_BtoB0inB0_VEC.size() != 0) {
        //     mean = matrix.colwise().mean();
        //     var = ((matrix.rowwise() - matrix.colwise().mean()).transpose() *
        //                         (matrix.rowwise() - matrix.colwise().mean())) / static_cast<double>(LIN_VEL_BtoB0inB0_VEC.size() - 1);
        // }
        
        // if (mean.norm()<0.01 && var.diagonal().norm() < 0.01){
        //     auto [acceMean, acceVar] = DataManager::AcceMeanVar(imuData);
        //     estimator->AddStationaryGravity(acceMean);
        // }
        // auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, Configor::Preference::DEBUG_MODE, false));

        // if (Configor::Preference::DEBUG_MODE) {
        //     spdlog::info("here is the summary of 'InitializeGravity':\n{}\n", sum.BriefReport());
        // }
    }

    Estimator::Ptr StateManager::InitializeVelSpline(const std::list<RadarTargetArray::Ptr> &radarTarAryVec,
                                                     const std::list<IMUFrame::Ptr> &imuData,
                                                     const std::list<LegFrame::Ptr> &legData) {

        // add gyro factors and fit so3 spline
        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);

        // optimization option in initialization
        //
        // IMPORTANT: To keep GaRLILEO's initial world definition (SO3_RefToW, gravity alignment)
        // independent of online extrinsic calibration, we ALWAYS keep extrinsics fixed during
        // initialisation. Online calibration is only enabled in incremental optimisation.
        GaRLILEO_Opt option = GaRLILEO_Opt::OPT_SO3 | GaRLILEO_Opt::OPT_VEL | GaRLILEO_Opt::OPT_GRAVITY;

        // acce and gyro factors
        for (const auto &frame: imuData) {
            estimator->AddGyroMeasurementWithConstBias(frame, option, configor->prior.GyroWeight);
            estimator->AddAcceMeasurementWithConstBias(frame, option, configor->prior.AcceWeight);
        }

        // radar factors (radar0). Skip when radar0 is disabled by yaml.
        if (configor->dataStream.UseRadar0) {
            for (const auto &ary: radarTarAryVec) {
                for (const auto &tar: ary->GetTargets()) {
                    estimator->AddRadarMeasurement(tar, option, SO3_RefToW, configor->prior.RadarWeight);
                }
            }
        }

        // leg factors. Skip when leg is disabled by yaml.
        if (configor->dataStream.UseLeg) {
            for (const auto &frame: legData) {
                estimator->AddLegMeasurement(frame, option, SO3_RefToW, configor->prior.LegWeight);
            }
        }

        // NOTE: Extrinsics are always fixed during initialisation. Online calibration
        // priors are only added in incremental optimisation.

        // add a tail constraint to handle the poor observability of the last knot
        auto velTailIdVec = estimator->AddVelSplineTailConstraint(option, 1000.0);
        auto so3TailIdVec = estimator->AddSo3SplineTailConstraint(option, 1000.0);

        // make this problem fun rank
        estimator->SetParameterBlockConstant(
                splines->GetSo3Spline(Configor::Preference::SO3Spline).KnotsFront().data()
        );

        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(4, Configor::Preference::DEBUG_MODE, false));

        option |= GaRLILEO_Opt::OPT_BA;
        sum = estimator->Solve(Estimator::DefaultSolverOptions(4, Configor::Preference::DEBUG_MODE, false));

        if (Configor::Preference::DEBUG_MODE) {
            spdlog::info("here is the summary of 'InitializeVelSpline':\n{}\n", sum.BriefReport());
        }

        for (const auto &id: velTailIdVec) { estimator->RemoveResidualBlock(id); }
        for (const auto &id: so3TailIdVec) { estimator->RemoveResidualBlock(id); }

        return estimator;
    }

    void StateManager::InitializeBiasFilters(double sTime) {
        // ShowSplineStatus();
        {
            // -----------------
            // acceleration bias
            // -----------------
            // the initial variance is AcceBiasPriorVariance (0.0001^2 by default)
            BiasFilter::StatePack initState(
                    sTime, *ba, Eigen::Vector3d::Ones() * configor->prior.AcceBiasPriorVariance
            );
            baFilter = BiasFilter::Create(initState, configor->prior.AcceBiasRandomWalk);
            spdlog::info("initial state of ba filter: {}", GARLILEO_TO_STR(baFilter->GetCurState()));

        }

        {

            VelBiasFilter::StatePack initState(
                    sTime, *bv, Eigen::Vector2d::Ones() * 0.0001 * 0.0001
            );
            bvFilter = VelBiasFilter::Create(initState, configor->prior.AcceBiasRandomWalk);
            spdlog::info("initial state of bv filter: {}", GARLILEO_TO_STR(bvFilter->GetCurState()));
        }
    }

    void StateManager::AlignInitializedStates() {
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        // current gravity, velocities, and rotations are expressed in the reference frame
        // align them to the world frame whose negative z axis is aligned with the gravity vector
        spdlog::info(
                "aligned gravity vector1: 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f}",
                (*gravity)(0), (*gravity)(1), (*gravity)(2)
        );
        SO3_RefToW = ObtainAlignedWtoRef(so3Spline.Evaluate(so3Spline.MinTime()), *gravity).inverse();
        *gravity = SO3_RefToW * *gravity;
        spdlog::info(
                "aligned gravity vector2: 'gx': {:.6f}, 'gy': {:.6f}, 'gz': {:.6f}",
                (*gravity)(0), (*gravity)(1), (*gravity)(2)
        );
        for (int i = 0; i < static_cast<int>(so3Spline.GetKnots().size()); ++i) {
            so3Spline.GetKnot(i) = SO3_RefToW * so3Spline.GetKnot(i);
        }
        // for (int i = 0; i < static_cast<int>(velSpline.GetKnots().size()); ++i) {
        //     velSpline.GetKnot(i) = SO3_RefToW * velSpline.GetKnot(i);
        // }
        // const auto &SO3_RtoB = configor->dataStream.CalibParam.SO3_RtoB;
        // const Eigen::Vector3d &POS_RinB = configor->dataStream.CalibParam.POS_RinB;

        // const auto &SO3_BtoL = configor->dataStream.CalibParam.SO3_BtoL;

        // configor->dataStream.CalibParam.SO3_RtoB = SO3_RefToW * SO3_RtoB;
        // configor->dataStream.CalibParam.POS_RinB = SO3_RefToW * POS_RinB;

        // configor->dataStream.CalibParam.SO3_BtoL = SO3_BtoL * SO3_RefToW.inverse();
        // configor->dataStream.CalibParam.POS_BinL = SO3_RefToW * configor->dataStream.CalibParam.POS_BinL;

        // Cache the aligned extrinsics as the fixed prior for online calibration (iKalibr-style).
        if (!hasRadarImuExtriPrior_) {
            radarImuExtriSO3Prior_ = configor->dataStream.CalibParam.SO3_RtoB;
            radarImuExtriPOSPrior_ = configor->dataStream.CalibParam.POS_RinB;
            radarImuExtriTOPrior_  = configor->dataStream.CalibParam.TIME_OFFSET_RtoB;
            hasRadarImuExtriPrior_ = true;
            spdlog::info(
                    "[OnlineCalib] cached radar-IMU extrinsic priors (after alignment): "
                    "EULER_RtoB(deg)=[{:.3f},{:.3f},{:.3f}] POS_RinB=[{:.3f},{:.3f},{:.3f}] "
                    "TO_RtoB={:.6f}s",
                    configor->dataStream.CalibParam.EULER_RtoB_DEG()(0),
                    configor->dataStream.CalibParam.EULER_RtoB_DEG()(1),
                    configor->dataStream.CalibParam.EULER_RtoB_DEG()(2),
                    radarImuExtriPOSPrior_(0), radarImuExtriPOSPrior_(1), radarImuExtriPOSPrior_(2),
                    radarImuExtriTOPrior_
            );
        }

        if (configor->dataStream.UseRadar1 && !hasRadar1ImuExtriPrior_) {
            radar1ImuExtriSO3Prior_ = configor->dataStream.CalibParam.SO3_R1toB;
            radar1ImuExtriPOSPrior_ = configor->dataStream.CalibParam.POS_R1inB;
            radar1ImuExtriTOPrior_  = configor->dataStream.CalibParam.TIME_OFFSET_R1toB;
            hasRadar1ImuExtriPrior_ = true;
            spdlog::info("[OnlineCalib] cached radar1-IMU extrinsic priors (after alignment) "
                         "(TO_R1toB={:.6f}s).",
                         radar1ImuExtriTOPrior_);
        }

        if (configor->dataStream.UseLeg && !hasLegImuExtriPrior_) {
            legImuExtriSO3Prior_ = configor->dataStream.CalibParam.SO3_BtoL;
            legImuExtriPOSPrior_ = configor->dataStream.CalibParam.POS_BinL;
            legImuExtriTOPrior_  = configor->dataStream.CalibParam.TIME_OFFSET_BtoL;
            hasLegImuExtriPrior_ = true;

            const auto q = legImuExtriSO3Prior_.unit_quaternion();
            Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
            constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
            euler *= kRadToDeg;

            spdlog::info(
                    "[OnlineCalib] cached leg-IMU extrinsic priors (after alignment): "
                    "EULER_BtoL(deg)=[{:.3f},{:.3f},{:.3f}] POS_BinL=[{:.3f},{:.3f},{:.3f}] "
                    "TO_BtoL={:.6f}s",
                    euler(0), euler(1), euler(2),
                    legImuExtriPOSPrior_(0), legImuExtriPOSPrior_(1), legImuExtriPOSPrior_(2),
                    legImuExtriTOPrior_
            );
        }
    }

    namespace {
        constexpr double kRadToDeg_ = 180.0 / 3.14159265358979323846;

        std::string FormatExtrinsicLine(const char *tag,
                                        const Sophus::SO3d &so3,
                                        const Eigen::Vector3d &pos,
                                        const Sophus::SO3d &so3_prior,
                                        const Eigen::Vector3d &pos_prior,
                                        double time_offset,
                                        double time_offset_prior) {
            const Eigen::Vector3d euler =
                    so3.unit_quaternion().toRotationMatrix().eulerAngles(2, 1, 0) * kRadToDeg_;
            const Eigen::Vector3d euler_prior =
                    so3_prior.unit_quaternion().toRotationMatrix().eulerAngles(2, 1, 0) * kRadToDeg_;

            const Sophus::SO3d delta = so3_prior.inverse() * so3;
            const double rot_dev_deg = delta.log().norm() * kRadToDeg_;
            const double pos_dev_m   = (pos - pos_prior).norm();
            const double to_dev_s    = time_offset - time_offset_prior;

            return fmt::format(
                    "  [{}] euler(deg)=[{:+8.3f},{:+8.3f},{:+8.3f}] (Δ={:+6.3f}deg)  "
                    "pos(m)=[{:+7.4f},{:+7.4f},{:+7.4f}] (Δ={:+6.4f}m)  "
                    "TO(s)={:+10.6f} (Δ={:+10.6f}s)",
                    tag,
                    euler(0), euler(1), euler(2), rot_dev_deg,
                    pos(0), pos(1), pos(2), pos_dev_m,
                    time_offset, to_dev_s);
            (void)euler_prior;  // currently unused, kept for future absolute logging
        }

        void AppendOnlineCalibCsvRow(const std::string &outputDir,
                                    const char *filename,
                                    double windowEndTimeSec,
                                    std::uint64_t windowIndex,
                                    const Sophus::SO3d &so3,
                                    const Eigen::Vector3d &pos,
                                    double time_offset,
                                    const Sophus::SO3d &so3_prior,
                                    const Eigen::Vector3d &pos_prior,
                                    double time_offset_prior) {
            if (outputDir.empty()) {
                return;
            }
            namespace fs = std::filesystem;
            const fs::path dirPath(outputDir);
            std::error_code ec;
            fs::create_directories(dirPath, ec);
            if (ec) {
                return;
            }
            const fs::path filePath = dirPath / filename;
            // Start each file afresh on its first row of this run; later rows append.
            static std::mutex startedFilesMutex;
            static std::set<std::string> startedFiles;
            bool write_header;
            {
                std::lock_guard<std::mutex> lock(startedFilesMutex);
                write_header = startedFiles.insert(filePath.string()).second;
            }

            const Eigen::Vector3d euler =
                    so3.unit_quaternion().toRotationMatrix().eulerAngles(2, 1, 0) * kRadToDeg_;

            const Sophus::SO3d delta = so3_prior.inverse() * so3;
            const double rot_dev_deg = delta.log().norm() * kRadToDeg_;
            const double pos_dev_m   = (pos - pos_prior).norm();
            const double to_dev_s    = time_offset - time_offset_prior;

            std::ofstream ofs(filePath, write_header ? std::ios::trunc : std::ios::app);
            if (!ofs.is_open()) {
                return;
            }
            const double wall_unix_s = std::chrono::duration<double>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count();

            if (write_header) {
                ofs << "timestamp_window_end_s,wall_time_unix_s,incremental_window_index,"
                       "euler_z_deg,euler_y_deg,euler_x_deg,"
                       "pos_x_m,pos_y_m,pos_z_m,time_offset_s,"
                       "rot_delta_deg,pos_delta_m,time_offset_delta_s\n";
            }
            ofs << fmt::format(
                    "{:.9f},{:.6f},{},{},{},{},{},{},{},{},{},{},{}\n",
                    windowEndTimeSec,
                    wall_unix_s,
                    windowIndex,
                    euler(0),
                    euler(1),
                    euler(2),
                    pos(0),
                    pos(1),
                    pos(2),
                    time_offset,
                    rot_dev_deg,
                    pos_dev_m,
                    to_dev_s);
        }
    }  // namespace

    void StateManager::LogOnlineCalibrationStatus(double windowEndTimeSec) const {
        const int period = configor->prior.OnlineCalibLogEveryNWindows;
        if (period <= 0) {
            return;
        }
        if (incrementalOptCounter_ == 0 ||
            (incrementalOptCounter_ % static_cast<std::uint64_t>(period)) != 0) {
            return;
        }

        const auto &calib  = configor->dataStream.CalibParam;
        const auto &priors = configor->prior;

        std::vector<std::string> lines;
        lines.reserve(3);

        const std::string outRoot = configor->dataStream.OutputPath + "/proprioceptive";

        if (configor->dataStream.UseRadar0 && priors.RadarImuOnlineCalibEnable) {
            lines.emplace_back(FormatExtrinsicLine(
                    "Radar0->IMU",
                    calib.SO3_RtoB, calib.POS_RinB,
                    hasRadarImuExtriPrior_ ? radarImuExtriSO3Prior_ : calib.SO3_RtoB,
                    hasRadarImuExtriPrior_ ? radarImuExtriPOSPrior_ : calib.POS_RinB,
                    calib.TIME_OFFSET_RtoB,
                    hasRadarImuExtriPrior_ ? radarImuExtriTOPrior_  : calib.TIME_OFFSET_RtoB));
            AppendOnlineCalibCsvRow(
                    outRoot,
                    "online_calib_radar0_imu.csv",
                    windowEndTimeSec,
                    incrementalOptCounter_,
                    calib.SO3_RtoB,
                    calib.POS_RinB,
                    calib.TIME_OFFSET_RtoB,
                    hasRadarImuExtriPrior_ ? radarImuExtriSO3Prior_ : calib.SO3_RtoB,
                    hasRadarImuExtriPrior_ ? radarImuExtriPOSPrior_ : calib.POS_RinB,
                    hasRadarImuExtriPrior_ ? radarImuExtriTOPrior_ : calib.TIME_OFFSET_RtoB);
        }
        if (configor->dataStream.UseRadar1 && priors.Radar1ImuOnlineCalibEnable) {
            lines.emplace_back(FormatExtrinsicLine(
                    "Radar1->IMU",
                    calib.SO3_R1toB, calib.POS_R1inB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriSO3Prior_ : calib.SO3_R1toB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriPOSPrior_ : calib.POS_R1inB,
                    calib.TIME_OFFSET_R1toB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriTOPrior_ : calib.TIME_OFFSET_R1toB));
            AppendOnlineCalibCsvRow(
                    outRoot,
                    "online_calib_radar1_imu.csv",
                    windowEndTimeSec,
                    incrementalOptCounter_,
                    calib.SO3_R1toB,
                    calib.POS_R1inB,
                    calib.TIME_OFFSET_R1toB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriSO3Prior_ : calib.SO3_R1toB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriPOSPrior_ : calib.POS_R1inB,
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriTOPrior_ : calib.TIME_OFFSET_R1toB);
        }
        if (configor->dataStream.UseLeg && priors.LegImuOnlineCalibEnable) {
            lines.emplace_back(FormatExtrinsicLine(
                    "Leg(B->L)",
                    calib.SO3_BtoL, calib.POS_BinL,
                    hasLegImuExtriPrior_ ? legImuExtriSO3Prior_ : calib.SO3_BtoL,
                    hasLegImuExtriPrior_ ? legImuExtriPOSPrior_ : calib.POS_BinL,
                    calib.TIME_OFFSET_BtoL,
                    hasLegImuExtriPrior_ ? legImuExtriTOPrior_ : calib.TIME_OFFSET_BtoL));
            AppendOnlineCalibCsvRow(
                    outRoot,
                    "online_calib_leg_imu.csv",
                    windowEndTimeSec,
                    incrementalOptCounter_,
                    calib.SO3_BtoL,
                    calib.POS_BinL,
                    calib.TIME_OFFSET_BtoL,
                    hasLegImuExtriPrior_ ? legImuExtriSO3Prior_ : calib.SO3_BtoL,
                    hasLegImuExtriPrior_ ? legImuExtriPOSPrior_ : calib.POS_BinL,
                    hasLegImuExtriPrior_ ? legImuExtriTOPrior_ : calib.TIME_OFFSET_BtoL);
        }

        if (lines.empty()) {
            return;
        }

        std::string body;
        for (const auto &line : lines) {
            body += "\n" + line;
        }
        spdlog::info("[OnlineCalib] window #{} (every {}):{}",
                     incrementalOptCounter_, period, body);
    }

    void StateManager::AppendExtrinsicMargBlocks(std::set<double *> &margParBlocks,
                                                 const Estimator::Ptr &estimator) const {
        // Spatial / temporal extrinsics. A parameter block is only inserted in
        // the marg set if (a) the corresponding sensor is enabled in yaml AND
        // (b) the block has actually been registered with the ceres problem
        // by some measurement factor. Without (b) ns_ctraj's map_util will
        // abort during MarginalizationInfo::Create because it cannot resolve
        // the Manifold/parameter metadata for a pointer the problem has never
        // seen.
        //
        // Concrete cases this guards against:
        //   - InitializeVelSpline does NOT add radar1 measurements (radar1 is
        //     only used during incremental optimisation), so SO3_R1toB et al.
        //     are absent from the problem in MarginalizationInInit even when
        //     UseRadar1=true.
        //   - With UseLeg=false the leg blocks are never touched.
        //   - Sliding-window iterations may temporarily have no radar/leg
        //     measurements for a given window if the buffer is starved.
        auto &calib = configor->dataStream.CalibParam;
        auto *problem = static_cast<const ceres::Problem *>(estimator.get());

        auto try_insert = [&](double *ptr) {
            if (problem->HasParameterBlock(ptr)) {
                margParBlocks.insert(ptr);
            }
        };

        if (configor->dataStream.UseRadar0) {
            try_insert(calib.SO3_RtoB.data());
            try_insert(calib.POS_RinB.data());
            try_insert(&calib.TIME_OFFSET_RtoB);
        }
        if (configor->dataStream.UseRadar1) {
            try_insert(calib.SO3_R1toB.data());
            try_insert(calib.POS_R1inB.data());
            try_insert(&calib.TIME_OFFSET_R1toB);
        }
        if (configor->dataStream.UseLeg) {
            try_insert(calib.SO3_BtoL.data());
            try_insert(calib.POS_BinL.data());
            try_insert(&calib.TIME_OFFSET_BtoL);
        }
    }

    void StateManager::MarginalizationInInit(const Estimator::Ptr &estimator) {
        // perform marginalization
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        int so3KnotSize = static_cast<int>(so3Spline.GetKnots().size());
        int velKnotSize = static_cast<int>(velSpline.GetKnots().size());

        std::set<double *> margParBlocks{gravity->data()};       
        for (int i = 0; i < so3KnotSize; ++i) {
            if (i <= so3KnotSize - 2 * Configor::Prior::SplineOrder) {
                margParBlocks.insert(so3Spline.GetKnot(i).data());
            } else {
                lastKeepSo3KnotAdd.insert({i, so3Spline.GetKnot(i).data()});
            }
        }
        for (int i = 0; i < velKnotSize; ++i) {
            if (i <= velKnotSize - 2 * Configor::Prior::SplineOrder) {
                margParBlocks.insert(velSpline.GetKnot(i).data());
            } else {
                lastKeepVelKnotAdd.insert({i, velSpline.GetKnot(i).data()});
            }
        }

        // Always marginalize all extrinsic / time-offset parameter blocks here.
        //
        // Rationale: during initialisation extrinsics are forced to be constant
        // (see InitializeVelSpline). If we leave them in the keep set, the marg
        // factor that is recreated every incremental window keeps growing because
        // it has to maintain a residual for each extrinsic block, and the cross
        // correlation accumulates indefinitely with the spline knots.  Putting
        // them in the marg set means: their (zero-Jacobian) contribution is
        // schur-complemented out and the marg factor of the next incremental
        // window only carries spline-knot keep blocks, exactly like a clean
        // VINS-style sliding window.  When online calibration is later turned
        // on, the extrinsics are re-introduced as fresh free variables in every
        // incremental optimisation, anchored only by their YAML prior plus the
        // current window's measurements.
        AppendExtrinsicMargBlocks(margParBlocks, estimator);

        if (Configor::Preference::DEBUG_MODE) {
            spdlog::info("rotation knots: {}, velocity knots: {}", so3KnotSize, velKnotSize);
            spdlog::info(
                    "marginalize rotation knots from 0-th to {}-th", so3KnotSize - Configor::Prior::SplineOrder - 1
            );
            spdlog::info(
                    "marginalize velocity knots from 0-th to {}-th", velKnotSize - Configor::Prior::SplineOrder - 1
            );
        }
        margInfo = ns_ctraj::MarginalizationInfo::Create(estimator.get(), margParBlocks, {}, 2);
    }

    void StateManager::PreOptimization(const std::list<IMUFrame::Ptr> &imuData, double stime, double etime) {
        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);

        for (const auto &frame: imuData) {
            estimator->AddGyroMeasurementWithConstBias(frame, GaRLILEO_Opt::OPT_SO3, configor->prior.GyroWeight);
        }
        // maintain observability of the last few control points
        estimator->AddSo3SplineTailConstraint(GaRLILEO_Opt::OPT_SO3, 1000.0);

        double dt = 0.05;
        for (double t=stime; t<etime-0.05; t+=0.01){
            estimator->AddRotPrior(t, dt, 1000);
        }
        
        // solving
        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, Configor::Preference::DEBUG_MODE, false));
    }

    void StateManager::InitGravitySpline(const std::list<IMUFrame::Ptr> &imuData){
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);
        GaRLILEO_Opt option = GaRLILEO_Opt::OPT_SO3 | GaRLILEO_Opt::OPT_VEL | GaRLILEO_Opt::OPT_BA;
        if (configor->prior.RadarImuOnlineCalibEnable) {
            if (configor->prior.RadarImuOnlineCalibOptimizeSO3_RtoB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_SO3_RtoB;
            }
            if (configor->prior.RadarImuOnlineCalibOptimizePOS_RinB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_POS_RinB;
            }
        }
        if (configor->prior.LegImuOnlineCalibEnable) {
            if (configor->prior.LegImuOnlineCalibOptimizeSO3_BtoL) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_SO3_BtoL;
            }
            if (configor->prior.LegImuOnlineCalibOptimizePOS_BinL) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_POS_BinL;
            }
        }
        for (auto iter1 = imuData.cbegin(); iter1 != imuData.cend(); ++iter1){
            estimator->AddGravitySplineConstraint(*iter1, 100);
        }
        auto gravTailIdVec = estimator->AddGravSplineTailConstraint(option, gravSpline.GetKnots().size()-3, 1000.0);

        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, Configor::Preference::DEBUG_MODE, false));

        for (const auto &id: gravTailIdVec) { estimator->RemoveResidualBlock(id); }

    }

    void StateManager::PostOptimization(long start_idx, long end_idx) {
        
        if(start_idx < 5){
            spdlog::info("not enough knots for update Rotation using Gravity");
            return;
        }

        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);

        for (long idx = start_idx-2; idx < end_idx-2; idx++){
            estimator->AddRotRefine(idx, 100);
        }

        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, false, false)); 

        // The local problem is discarded here. Adding a prior after Solve
        // cannot affect this result and only allocates unused residual blocks.

    }

    bool StateManager::IncrementalOptimization(const GaRLILEOStatus::StatusPack &status) { 
        const double sTime = status.ValidStateEndTime, eTime = dataMagr->GetEndTimeSafely();
        const bool useRadar0 = configor->dataStream.UseRadar0;
        const bool useRadar1 = configor->dataStream.UseRadar1;
        const bool useAnyRadar = useRadar0 || useRadar1;
        const double minimumWindowDurationSec = 1.0 / configor->preference.IncrementalOptRate;
        const double latestImuTime = useAnyRadar ? dataMagr->GetIMUEndTimeSafely() : eTime;
        const auto stopIfRadarRecoveryExpired = [&]() {
            if (!radarRecoveryHorizonExceeded(
                    sTime, eTime, latestImuTime, useAnyRadar, radarWindowDeferred_,
                    minimumWindowDurationSec)) {
                return false;
            }
            spdlog::error(
                    "[RadarWindow] Recovery exceeded {:.3f}s sensor-time horizon: "
                    "start={:.6f}, common_end={:.6f}, latest_imu={:.6f}, deferred={}. "
                    "Stopping without advancing estimator state or extending the solve window.",
                    kMaxRadarRecoveryHorizonSec, sTime, eTime, latestImuTime,
                    radarWindowDeferred_);
            LOCK_GARLILEO_STATUS
            GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::ShouldQuit;
            return true;
        };
        // Check before the short-window return: an empty/absent radar can freeze
        // the common end time while IMU/leg buffers continue to grow.
        if (stopIfRadarRecoveryExpired()) { return false; }

        if (eTime - sTime < minimumWindowDurationSec) { return false; }

        // spdlog::info(
        //         "'incremental optimization': start time: {:.6f}, end time: {:.6f}, duration: {:.6f}",
        //         sTime, eTime, eTime - sTime
        // );
        
        const bool useLeg = configor->dataStream.UseLeg;

        auto imuData = dataMagr->ExtractIMUDataPieceSafely(sTime, eTime);
        std::list<RadarTarget::Ptr> radarData;
        std::list<RadarTarget::Ptr> radar1Data;
        if (useRadar0) {
            radarData = dataMagr->ExtractRadarDataPieceSafely(sTime, eTime);
        }
        if (useRadar1) {
            radar1Data = dataMagr->ExtractRadar1DataPieceSafely(sTime, eTime);
        }
        std::list<LegFrame::Ptr> legDataSeq;
        if (useLeg) {
            legDataSeq = dataMagr->ExtractLegDataPieceSafely(sTime, eTime);
        }

        const double dt = eTime - sTime;
        const auto readiness = classifyIncrementalWindow(
                dt, imuData.size(), useRadar0, radarData.size(),
                useRadar1, radar1Data.size());
        if (readiness == IncrementalWindowReadiness::FatalImuRate) {
            // Preserve the existing fatal IMU-rate condition.
            LOCK_GARLILEO_STATUS
            GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::ShouldQuit;
            return false;
        }
        if (readiness == IncrementalWindowReadiness::WaitForImuSamples) {
            // Retry from the unchanged ValidStateEndTime with a longer window.
            return false;
        }
        if (readiness == IncrementalWindowReadiness::WaitForRadarTargets) {
            radarWindowDeferred_ = true;
            // A first insufficient window may already exceed the retry budget.
            if (stopIfRadarRecoveryExpired()) { return false; }
            // These are target densities, not radar message rates. One empty
            // scan can lower a short window below the threshold. Retry from the
            // unchanged ValidStateEndTime; extraction does not consume data and
            // cleanup retains everything newer than ValidStateEndTime - 0.2.
            static thread_local auto lastRadarWaitLog = std::chrono::steady_clock::time_point{};
            const auto now = std::chrono::steady_clock::now();
            if (lastRadarWaitLog == std::chrono::steady_clock::time_point{}
                || now - lastRadarWaitLog >= std::chrono::seconds(1)) {
                lastRadarWaitLog = now;
                spdlog::warn(
                        "[RadarWindow] Deferring update [{:.6f}, {:.6f}): "
                        "radar0 enabled={} targets={} density={:.3f}/s; "
                        "radar1 enabled={} targets={} density={:.3f}/s. "
                        "Waiting for at least 10 targets/s per enabled radar; "
                        "estimator state and window start are unchanged.",
                        sTime, eTime, useRadar0, radarData.size(),
                        static_cast<double>(radarData.size()) / dt,
                        useRadar1, radar1Data.size(),
                        static_cast<double>(radar1Data.size()) / dt);
            }
            return false;
        }
        radarWindowDeferred_ = false;

        // ----------------
        // static condition
        // ----------------
        LOCK_STATES
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        auto [acceMean, acceVar] = DataManager::AcceMeanVar(imuData);
        double velocity = velSpline.Evaluate(status.ValidStateEndTime - 1E-3).norm();

        for (auto iter1 = imuData.cbegin(); iter1 != imuData.cend(); ++iter1){
            Eigen::Quaternion<double> ori = (*iter1)->GetOrientation();
            Sophus::SO3d so3(ori);
            Sophus::SO3d gravity_align = SO3_RefToW * so3;
            Eigen::Quaternion<double> quat_g_align = gravity_align.unit_quaternion();
            (*iter1)->SetOrientation(quat_g_align);

            // Eigen::Vector3d gyro((*iter1)->GetGyro());
            // Eigen::Vector3d gravity_align_gyro = SO3_RefToW * gyro;

            // (*iter1)->SetGyro(gravity_align_gyro);
        }
        

        // --------------
        // extend splines
        // --------------
        int so3KnotOldSize = static_cast<int>(so3Spline.GetKnots().size());
        int velKnotOldSize = static_cast<int>(velSpline.GetKnots().size());
        int gravKnotOldSize = static_cast<int>(gravSpline.GetKnots().size());

        if (Configor::Preference::DEBUG_MODE) { ShowSplineStatus(); }

        // --------------------
        // linear extrapolation
        // --------------------
        LinearExtendKnotTo(so3Spline, eTime + 1E-6);
        LinearExtendKnotTo(velSpline, eTime + 1E-6);
        

        int old_grav_idx  = gravSpline.GetKnots().size();
        while (gravSpline.GetKnots().size()<so3Spline.GetKnots().size()) {
            gravSpline.KnotsPushBack((so3Spline.GetKnot(gravSpline.GetKnots().size()).matrix().transpose()) * (*gravity));
        }

        // obtain the updated address
        std::map<double *, double *> updatedLastKeepKnotAdd;
        for (const auto &[idx, add]: lastKeepSo3KnotAdd) {
            updatedLastKeepKnotAdd.insert({add, so3Spline.GetKnot(idx).data()});
        }
        for (const auto &[idx, add]: lastKeepVelKnotAdd) {
            updatedLastKeepKnotAdd.insert({add, velSpline.GetKnot(idx).data()});
        }
        for (const auto &[idx, add]: lastKeepGravKnotAdd) {
            updatedLastKeepKnotAdd.insert({add, gravSpline.GetKnot(idx).data()});
        }
        margInfo->ShiftKeepParBlockAddress(updatedLastKeepKnotAdd);

        // ------------------------
        // incremental optimization
        // ------------------------

        PreOptimization(imuData, sTime, eTime);

        // add gyro factors and fit so3 spline
        auto estimator = Estimator::Create(configor, splines, gravity, ba, bg, bv);
        // // marginalization factor
        ns_ctraj::MarginalizationFactor::AddToProblem(estimator.get(), margInfo, 1.0);

        GaRLILEO_Opt option = GaRLILEO_Opt::OPT_SO3 | GaRLILEO_Opt::OPT_VEL | GaRLILEO_Opt::OPT_BA;
        if (useRadar0 && configor->prior.RadarImuOnlineCalibEnable) {
            if (configor->prior.RadarImuOnlineCalibOptimizeSO3_RtoB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_SO3_RtoB;
            }
            if (configor->prior.RadarImuOnlineCalibOptimizePOS_RinB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_POS_RinB;
            }
            if (configor->prior.RadarImuOnlineCalibOptimizeTO_RtoB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_TO_RtoB;
            }
        }
        if (useRadar1 && configor->prior.Radar1ImuOnlineCalibEnable) {
            if (configor->prior.Radar1ImuOnlineCalibOptimizeSO3_R1toB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_SO3_R1toB;
            }
            if (configor->prior.Radar1ImuOnlineCalibOptimizePOS_R1inB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_POS_R1inB;
            }
            if (configor->prior.Radar1ImuOnlineCalibOptimizeTO_R1toB) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_TO_R1toB;
            }
        }
        if (useLeg && configor->prior.LegImuOnlineCalibEnable) {
            if (configor->prior.LegImuOnlineCalibOptimizeSO3_BtoL) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_SO3_BtoL;
            }
            if (configor->prior.LegImuOnlineCalibOptimizePOS_BinL) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_POS_BinL;
            }
            if (configor->prior.LegImuOnlineCalibOptimizeTO_BtoL) {
                option = option | GaRLILEO_Opt::OPT_EXTRI_TO_BtoL;
            }
        }
        std::set<ceres::ResidualBlockId> gravIdSet;
        std::set<double*> gravParamSet;
        std::set<int> gravIDX;
        for (auto iter1 = imuData.cbegin(); iter1 != imuData.cend(); ++iter1){
            double t1 = (*iter1)->GetTimestamp();
            auto id1 = estimator->AddGravitySplineConstraint(*iter1, 1000);
            Eigen::Quaternion<double> ori1 = (*iter1)->GetOrientation();
            Sophus::SO3d rot1(ori1);
            estimator->AddGyroMeasurementWithConstBias(*iter1, option, configor->prior.GyroWeight);
            const auto pairBegin = std::next(iter1);
            if (pairBegin == imuData.cend()) {
                continue;
            }
            // Every pair with this start shares the same ordered trapezoid prefix.
            // Append one interval using the original transform expression.
            Eigen::Vector3d previousAcceleration =
                so3Spline.Evaluate(t1).matrix().transpose() * so3Spline.Evaluate(t1).matrix() * ((*iter1)->GetAcce() - *ba);
            double previousTime = t1;
            Eigen::Vector3d velPIM = Eigen::Vector3d::Zero();
            for (auto iter2 = pairBegin; iter2 != imuData.cend(); ++iter2) {
                Eigen::Quaternion<double> ori2 = (*iter2)->GetOrientation();
                Sophus::SO3d rot2(ori2);
                double t2 = (*iter2)->GetTimestamp();
                const Eigen::Vector3d acceleration =
                    so3Spline.Evaluate(t1).matrix().transpose() * so3Spline.Evaluate(t2).matrix() * ((*iter2)->GetAcce() - *ba);
                velPIM += (previousAcceleration + acceleration) * (t2 - previousTime) * double(0.5);
                previousAcceleration = acceleration;
                previousTime = t2;
                auto id1 = estimator->AddGravityRotConstraint(t1, t2, velPIM, rot1, rot2, configor->prior.GravityWeight);
                gravIdSet.insert(id1);
            }
        }

        if (!useLeg) {
            for (const auto &frame : imuData) {
                estimator->AddAcceMeasurementWithConstBias(frame, option, configor->prior.AcceWeight);
            }
        }

        auto baPrioriId = estimator->AddAcceBiasPriori(baFilter->Prediction(eTime), option);

        auto bvPrioriId = estimator->AddVelBiasPriori(bvFilter->Prediction(eTime), option);

        // radar0 factors (skip when radar0 is disabled by yaml)
        if (useRadar0) {
            for (const auto &tar: radarData) {
                estimator->AddRadarMeasurement(tar, option, SO3_RefToW, configor->prior.RadarWeight);
            }
        }
        // radar1 factors (skip when radar1 is disabled by yaml)
        if (useRadar1) {
            for (const auto &tar: radar1Data) {
                estimator->AddRadar1Measurement(tar, option, SO3_RefToW, configor->prior.RadarWeight);
            }
        }

        // leg factors. Skip when leg is disabled by yaml.
        if (useLeg) {
            for (const auto &frame: legDataSeq) {
                estimator->AddLegMeasurement(frame, option, SO3_RefToW, configor->prior.LegWeight);
            }
        }

        // iKalibr-style soft priors for online radar-IMU extrinsic calibration.
        if (useRadar0 && configor->prior.RadarImuOnlineCalibEnable) {
            const Sophus::SO3d so3_prior =
                    hasRadarImuExtriPrior_ ? radarImuExtriSO3Prior_ : configor->dataStream.CalibParam.SO3_RtoB;
            const Eigen::Vector3d pos_prior =
                    hasRadarImuExtriPrior_ ? radarImuExtriPOSPrior_ : configor->dataStream.CalibParam.POS_RinB;
            const double to_prior =
                    hasRadarImuExtriPrior_ ? radarImuExtriTOPrior_ : configor->dataStream.CalibParam.TIME_OFFSET_RtoB;

            if (configor->prior.RadarImuOnlineCalibPriorWeightSO3 > 0.0) {
                estimator->AddRadarImuExtriSO3Prior(
                        so3_prior, configor->prior.RadarImuOnlineCalibPriorWeightSO3
                );
            }
            if (configor->prior.RadarImuOnlineCalibPriorWeightPOS > 0.0) {
                estimator->AddRadarImuExtriPOSPrior(
                        pos_prior, configor->prior.RadarImuOnlineCalibPriorWeightPOS
                );
            }
            if (configor->prior.RadarImuOnlineCalibOptimizeTO_RtoB &&
                configor->prior.RadarImuOnlineCalibPriorWeightTO > 0.0) {
                estimator->AddRadarImuExtriTOPrior(
                        to_prior, configor->prior.RadarImuOnlineCalibPriorWeightTO
                );
            }
        }

        // Soft priors for online radar1-IMU extrinsic calibration (second radar).
        if (useRadar1 && configor->prior.Radar1ImuOnlineCalibEnable) {
            const Sophus::SO3d so3_prior =
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriSO3Prior_ : configor->dataStream.CalibParam.SO3_R1toB;
            const Eigen::Vector3d pos_prior =
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriPOSPrior_ : configor->dataStream.CalibParam.POS_R1inB;
            const double to_prior =
                    hasRadar1ImuExtriPrior_ ? radar1ImuExtriTOPrior_ : configor->dataStream.CalibParam.TIME_OFFSET_R1toB;

            if (configor->prior.Radar1ImuOnlineCalibPriorWeightSO3 > 0.0) {
                estimator->AddRadar1ImuExtriSO3Prior(
                        so3_prior, configor->prior.Radar1ImuOnlineCalibPriorWeightSO3
                );
            }
            if (configor->prior.Radar1ImuOnlineCalibPriorWeightPOS > 0.0) {
                estimator->AddRadar1ImuExtriPOSPrior(
                        pos_prior, configor->prior.Radar1ImuOnlineCalibPriorWeightPOS
                );
            }
            if (configor->prior.Radar1ImuOnlineCalibOptimizeTO_R1toB &&
                configor->prior.Radar1ImuOnlineCalibPriorWeightTO > 0.0) {
                estimator->AddRadar1ImuExtriTOPrior(
                        to_prior, configor->prior.Radar1ImuOnlineCalibPriorWeightTO
                );
            }
        }

        // iKalibr-style soft priors for online leg-IMU extrinsic calibration.
        // Only meaningful when leg measurements are part of the optimisation.
        if (useLeg && configor->prior.LegImuOnlineCalibEnable) {
            const Sophus::SO3d so3_prior =
                    hasLegImuExtriPrior_ ? legImuExtriSO3Prior_ : configor->dataStream.CalibParam.SO3_BtoL;
            const Eigen::Vector3d pos_prior =
                    hasLegImuExtriPrior_ ? legImuExtriPOSPrior_ : configor->dataStream.CalibParam.POS_BinL;
            const double to_prior =
                    hasLegImuExtriPrior_ ? legImuExtriTOPrior_ : configor->dataStream.CalibParam.TIME_OFFSET_BtoL;

            if (configor->prior.LegImuOnlineCalibPriorWeightSO3 > 0.0) {
                estimator->AddLegImuExtriSO3Prior(
                        so3_prior, configor->prior.LegImuOnlineCalibPriorWeightSO3
                );
            }
            if (configor->prior.LegImuOnlineCalibPriorWeightPOS > 0.0) {
                estimator->AddLegImuExtriPOSPrior(
                        pos_prior, configor->prior.LegImuOnlineCalibPriorWeightPOS
                );
            }
            if (configor->prior.LegImuOnlineCalibOptimizeTO_BtoL &&
                configor->prior.LegImuOnlineCalibPriorWeightTO > 0.0) {
                estimator->AddLegImuExtriTOPrior(
                        to_prior, configor->prior.LegImuOnlineCalibPriorWeightTO
                );
            }
        }

        // // ---------------------------------------------------
        // // handle the poor observability of the last few knots
        // // ---------------------------------------------------
        std::size_t old_grav = 0;
        std::map<long, std::map<std::size_t, int>> knotRecoder = estimator->getKnotRecoder();
        for (const auto &[knotId, count]: knotRecoder[reinterpret_cast<long>(&gravSpline)]){
            if(old_grav == 0) old_grav = knotId;
            auto *data = const_cast<double *>(gravSpline.GetKnot(knotId).data());
            gravParamSet.insert(data);
            gravIDX.insert(knotId);
        }

        old_grav_idx = static_cast<int>(old_grav);
        auto gravTailIdVec = estimator->AddGravSplineTailConstraint(option, old_grav_idx, 1000.0);

        auto velTailIdVec = estimator->AddVelSplineTailConstraint(option, 1000.0);
        auto so3TailIdVec = estimator->AddSo3SplineTailConstraint(option, 1000.0);
        auto sum = estimator->Solve(Estimator::DefaultSolverOptions(1, Configor::Preference::DEBUG_MODE, false));

        // // ------------------
        // // update bias filter
        // // ------------------

        update_start_index = update_end_index;
        for (const auto &[knotId, count]: knotRecoder[reinterpret_cast<long>(&so3Spline)]){
            update_end_index = knotId;
            break;
        }
        UpdateBiasFilters(estimator, eTime);

        PostOptimization(update_start_index, update_end_index);

        // // -----------------------
        // // perform marginalization
        // // -----------------------

        int so3KnotSize = static_cast<int>(so3Spline.GetKnots().size());
        int velKnotSize = static_cast<int>(velSpline.GetKnots().size());
        int gravKnotSize = static_cast<int>(gravSpline.GetKnots().size());
        
        std::set<double *> margParBlocks{gravity->data()};
        lastKeepSo3KnotAdd.clear(), lastKeepVelKnotAdd.clear(), lastKeepGravKnotAdd.clear();
        for (int i = std::max(0, so3KnotOldSize - 2 * Configor::Prior::SplineOrder + 1); i < so3KnotSize; ++i) {
            if (i <= so3KnotSize - 2 * Configor::Prior::SplineOrder) {
                margParBlocks.insert(so3Spline.GetKnot(i).data());
            } else {
                lastKeepSo3KnotAdd.insert({i, so3Spline.GetKnot(i).data()});
            }
        }
        for (int i = std::max(0, velKnotOldSize - 2 * Configor::Prior::SplineOrder + 1); i < velKnotSize; ++i) {
            if (i <= velKnotSize - 2 * Configor::Prior::SplineOrder) {
                margParBlocks.insert(velSpline.GetKnot(i).data());
            } else {
                lastKeepVelKnotAdd.insert({i, velSpline.GetKnot(i).data()});
            }
        }

        for (int i = std::max(0, gravKnotOldSize - 2 * Configor::Prior::SplineOrder + 1); i < gravKnotSize; ++i) {
            if (i <= gravKnotSize - 2 * Configor::Prior::SplineOrder) {
                margParBlocks.insert(gravSpline.GetKnot(i).data());
            } else {
                lastKeepGravKnotAdd.insert({i, gravSpline.GetKnot(i).data()});
            }
        }

        // Always marginalise extrinsic / time-offset blocks so that they do not
        // accumulate in the keep set of the marginalisation factor across
        // sliding-window iterations. See StateManager::AppendExtrinsicMargBlocks
        // for the rationale (and for why we only insert blocks that ceres
        // already knows about).
        AppendExtrinsicMargBlocks(margParBlocks, estimator);

        estimator->RemoveResidualBlock(baPrioriId);
        estimator->RemoveResidualBlock(bvPrioriId);
        // remove linear extrapolation factor as they are not come from real-sensor measurements
        for (const auto &id: velTailIdVec) { estimator->RemoveResidualBlock(id); }
        for (const auto &id: so3TailIdVec) { estimator->RemoveResidualBlock(id); }

        st = sTime;

        margInfo = ns_ctraj::MarginalizationInfo::Create(estimator.get(), margParBlocks, {}, 1);

        ++incrementalOptCounter_;
        LogOnlineCalibrationStatus(eTime);

        LOCK_GARLILEO_STATUS
        GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::NewStateNeedToDraw;
        GaRLILEOStatus::StateManager::CurStatus |= GaRLILEOStatus::StateManager::Status::NewStateNeedToPublish;
        GaRLILEOStatus::StateManager::ValidStateEndTime = eTime;

        return true;
    }

    std::optional<std::pair<double, double>> StateManager::GetSplineTimeRangeSafely() const {
        LOCK_STATES
        if (!splines) { return {}; }
        const auto &so3 = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        const auto &vel = splines->GetRdSpline(Configor::Preference::VelSpline);
        const auto &grav = splines->GetRdSpline(Configor::Preference::GravitySpline);
        const auto order = static_cast<std::size_t>(Configor::Prior::SplineOrder);
        if (so3.GetKnots().size() < order || vel.GetKnots().size() < order
                || grav.GetKnots().size() < order) { return {}; }
        const double start = std::max(std::max(so3.MinTime(), vel.MinTime()), grav.MinTime());
        const double end = std::min(std::min(so3.MaxTime(), vel.MaxTime()), grav.MaxTime());
        if (!(start < end)) { return {}; }
        return std::make_pair(start, end);
    }

    std::optional<StateManager::StatePack> StateManager::GetStatePackSafely(
            double t, Eigen::Vector3d *angularVelocity) const {
        LOCK_STATES
        if (splines == nullptr) { return {}; }

        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        if (!splines->TimeInRange(t, so3Spline)) { return {}; }

        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        if (!splines->TimeInRange(t, velSpline)) { return {}; }
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        if (!splines->TimeInRange(t, gravSpline)) { return {}; }

        StatePack pack;
        pack.timestamp = t;
        pack.ba = *ba;
        pack.bg = *bg;
        pack.bv = *bv;
        pack.gravity = gravSpline.Evaluate(t);
        pack.SO3_CurToRef = so3Spline.Evaluate(t);
        pack.LIN_VEL_CurToRefInCur = velSpline.Evaluate(t);
        if (angularVelocity) {
            *angularVelocity = so3Spline.VelocityBody(t);
        }

        return pack;
    }

    std::vector<std::optional<StateManager::StatePack>>
    StateManager::GetStatePackSafely(const std::vector<double> &times) const {
        std::vector<std::optional<StateManager::StatePack>> packs(times.size());
        LOCK_STATES
        if (splines == nullptr) { return packs; }
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);

        for (int i = 0; i < static_cast<int>(times.size()); ++i) {
            double t = times.at(i);
            if (!splines->TimeInRange(t, so3Spline)
                || !splines->TimeInRange(t, velSpline)
                || !splines->TimeInRange(t, gravSpline)) {
                packs.at(i) = {};
            } else {
                StatePack pack;
                pack.timestamp = t;
                pack.ba = *ba;
                pack.bg = *bg;
                pack.bv = *bv;
                pack.gravity = gravSpline.Evaluate(t);
                pack.SO3_CurToRef = so3Spline.Evaluate(t);
                pack.LIN_VEL_CurToRefInCur = velSpline.Evaluate(t);

                packs.at(i) = pack;
            }
        }
        return packs;
    }

    const StateManager::SplineBundleType::Ptr &StateManager::GetSplines() const {
        return splines;
    }

    void StateManager::ShowSplineStatus() const {
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        spdlog::info(
                "[Rotation Spline] min: '{:.6f}', max: '{:.6f}', dt: '{:.6f}', size: '{}'",
                so3Spline.MinTime(), so3Spline.MaxTime(), so3Spline.GetTimeInterval(), so3Spline.GetKnots().size()
        );
        spdlog::info(
                "[Velocity Spline] min: '{:.6f}', max: '{:.6f}', dt: '{:.6f}', size: '{}'",
                velSpline.MinTime(), velSpline.MaxTime(), velSpline.GetTimeInterval(), velSpline.GetKnots().size()
        );
        spdlog::info(
                "[Gravity Spline] min: '{:.6f}', max: '{:.6f}', dt: '{:.6f}', size: '{}'",
                gravSpline.MinTime(), gravSpline.MaxTime(), gravSpline.GetTimeInterval(), gravSpline.GetKnots().size()
        );
    }

    void StateManager::UpdateBiasFilters(const Estimator::Ptr &estimator, double eTime) {
        // Unregistered blocks (for example bv with UseLeg=false) must never
        // reach Ceres covariance lookup. Both requested blocks share the same
        // problem Jacobian, so factor it only once before updating the filters.
        const auto *problem = static_cast<const ceres::Problem *>(estimator.get());
        const bool hasBa = problem->HasParameterBlock(ba->data());
        const bool hasBv = problem->HasParameterBlock(bv->data());
        std::vector<std::pair<const double *, const double *>> blocks;
        blocks.reserve(2);
        if (hasBa) {
            blocks.emplace_back(ba->data(), ba->data());
        }
        if (hasBv) {
            blocks.emplace_back(bv->data(), bv->data());
        }
        if (blocks.empty()) {
            return;
        }

        ceres::Covariance covariance({});
        const bool covarianceValid = covariance.Compute(blocks, estimator.get());
        // Ceres returns zero covariance for a constant block even when the
        // variable part is singular. Preserve those independent filter updates
        // if the combined request fails because of another variable block.
        if (hasBa && (covarianceValid || problem->IsParameterBlockConstant(ba->data()))) {
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> cov =
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
            if (!covarianceValid || covariance.GetCovarianceBlock(ba->data(), ba->data(), cov.data())) {
                baFilter->UpdateByEstimator(BiasFilter::StatePack(eTime, *ba, cov.diagonal()));
            }
        }
        if (hasBv && (covarianceValid || problem->IsParameterBlockConstant(bv->data()))) {
            Eigen::Matrix<double, 2, 2, Eigen::RowMajor> cov =
                Eigen::Matrix<double, 2, 2, Eigen::RowMajor>::Zero();
            if (!covarianceValid || covariance.GetCovarianceBlock(bv->data(), bv->data(), cov.data())) {
                bvFilter->UpdateByEstimator(VelBiasFilter::StatePack(eTime, *bv, cov.diagonal()));
            }
        }
    }

    const BiasFilter::Ptr &StateManager::GetBaFilter() const {
        return baFilter;
    }

    const VelBiasFilter::Ptr &StateManager::GetBvFilter() const {
        return bvFilter;
    }

    void StateManager::LinearExtendKnotTo(SplineBundleType::RdSplineType &spline, double t) {
        Eigen::Vector3d delta = spline.GetKnots().back() - spline.GetKnots().at(spline.GetKnots().size() - 2);
        while ((spline.GetKnots().size() < SplineBundleType::N) || (spline.MaxTime() < t)) {
            spline.KnotsPushBack(spline.GetKnots().back() + delta);
        }
    }

    void StateManager::LinearExtendKnotTo(SplineBundleType::So3SplineType &spline, double t) {
        Sophus::SO3d delta =
                spline.GetKnots().at(spline.GetKnots().size() - 2).inverse() * spline.GetKnots().back();
        while ((spline.GetKnots().size() < SplineBundleType::N) || (spline.MaxTime() < t)) {
            spline.KnotsPushBack(spline.GetKnots().back() * delta);
        }
    }

    StateManager::StatePack::StatePack(double timestamp, const Sophus::SO3d &so3CurToRef,
                                       Eigen::Vector3d linVelCurToRefInCur, Eigen::Vector3d gravity,
                                       Eigen::Vector3d ba, Eigen::Vector3d bg, Eigen::Vector2d bv)
            : timestamp(timestamp), SO3_CurToRef(so3CurToRef), LIN_VEL_CurToRefInCur(std::move(linVelCurToRefInCur)),
              gravity(std::move(gravity)), ba(std::move(ba)), bg(std::move(bg)), bv(std::move(bv)) {}

    Eigen::Vector3d StateManager::StatePack::LIN_VEL_CurToRefInRef() const {
        return SO3_CurToRef * LIN_VEL_CurToRefInCur;
    }

    StateManager::StatePack::StatePack() = default;

}
