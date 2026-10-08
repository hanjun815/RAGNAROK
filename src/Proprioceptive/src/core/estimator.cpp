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

#include <utility>
#include <algorithm>
#include "core/estimator.h"
#include "factor/imu_gyro_factor.hpp"
#include "factor/gravity_factor.hpp"
#include "factor/imu_acce_factor.hpp"
#include "factor/radar_factor.hpp"
#include "factor/bias_factor.hpp"
#include "factor/tail_factor.hpp"
#include "factor/spline_factor.hpp"
#include "factor/gravity_update_factor.hpp"
#include "factor/gravity_spline_factor.hpp"
#include "factor/rot_refine_factor.hpp"
#include "factor/rot_prior_factor.hpp"
#include "factor/leg_vel_factor.hpp"
#include "factor/vel_bias_factor.hpp"
#include "factor/gravity_stationary_factor.hpp"
#include "factor/prior_extri_so3_factor.hpp"
#include "factor/prior_extri_pos_factor.hpp"
#include "factor/prior_time_offset_factor.hpp"

namespace garlileo {

    std::shared_ptr<ceres::EigenQuaternionManifold> Estimator::QUATER_MANIFOLD(new ceres::EigenQuaternionManifold());
    std::shared_ptr<ceres::SphereManifold<3>> Estimator::GRAVITY_MANIFOLD(new ceres::SphereManifold<3>());

    ceres::Problem::Options Estimator::DefaultProblemOptions() {
        return ns_ctraj::TrajectoryEstimator<Configor::Prior::SplineOrder>::DefaultProblemOptions();
    }

    ceres::Solver::Options Estimator::DefaultSolverOptions(int threadNum, bool toStdout, bool useCUDA) {
        auto defaultSolverOptions = ns_ctraj::TrajectoryEstimator<Configor::Prior::SplineOrder>::DefaultSolverOptions(
                threadNum, toStdout, useCUDA
        );
        if (!useCUDA) {
            defaultSolverOptions.linear_solver_type = ceres::DENSE_SCHUR;
        }
        defaultSolverOptions.trust_region_strategy_type = ceres::DOGLEG;
        return defaultSolverOptions;
    }

    Estimator::Estimator(Configor::Ptr configor, SplineBundleType::Ptr splines,
                         const std::shared_ptr<Eigen::Vector3d> &gravity, const std::shared_ptr<Eigen::Vector3d> &ba,
                         const std::shared_ptr<Eigen::Vector3d> &bg,
                         const std::shared_ptr<Eigen::Vector2d> &bv)
            : ceres::Problem(DefaultProblemOptions()), configor(std::move(configor)),
              splines(std::move(splines)), gravity(gravity), ba(ba), bg(bg), bv(bv) {}

    Estimator::Ptr Estimator::Create(const Configor::Ptr &configor, const SplineBundleType::Ptr &splines,
                                     const std::shared_ptr<Eigen::Vector3d> &gravity,
                                     const std::shared_ptr<Eigen::Vector3d> &ba,
                                     const std::shared_ptr<Eigen::Vector3d> &bg,
                                     const std::shared_ptr<Eigen::Vector2d> &bv) {
        return std::make_shared<Estimator>(configor, splines, gravity, ba, bg, bv);
     }

    ceres::Solver::Summary Estimator::Solve(const ceres::Solver::Options &options) {
        ceres::Solver::Summary summary;
        ceres::Solve(options, this, &summary);
        return summary;
    }

    void Estimator::AddGravKnotsData(std::vector<double *> &paramBlockVec,
                                   const Estimator::SplineBundleType::RdSplineType &spline,
                                   const Estimator::SplineMetaType &splineMeta, bool setToConst) {
        // for each segment
        for (const auto &seg: splineMeta.segments) {
            // the factor 'seg.dt * 0.5' is the treatment for numerical accuracy
            auto idxMaster = spline.ComputeTIndex(seg.t0 + seg.dt * 0.5).second;
            
            // from the first control point to the last control point
            for (std::size_t i = idxMaster; i < idxMaster + seg.NumParameters(); ++i) {
                auto *data = const_cast<double *>(spline.GetKnot(static_cast<int>(i)).data());
                this->AddParameterBlock(data, 3, GRAVITY_MANIFOLD.get());
                paramBlockVec.push_back(data);
                // set this param block to be constant
                if (setToConst) { this->SetParameterBlockConstant(data); }

                // knot recoder
                knotRecoder[reinterpret_cast<long>(&spline)][i]++;
            }
        }
    }

    void Estimator::AddRdKnotsData(std::vector<double *> &paramBlockVec,
                                   const Estimator::SplineBundleType::RdSplineType &spline,
                                   const Estimator::SplineMetaType &splineMeta, bool setToConst) {
        // for each segment
        for (const auto &seg: splineMeta.segments) {
            // the factor 'seg.dt * 0.5' is the treatment for numerical accuracy
            auto idxMaster = spline.ComputeTIndex(seg.t0 + seg.dt * 0.5).second;
            
            // from the first control point to the last control point
            for (std::size_t i = idxMaster; i < idxMaster + seg.NumParameters(); ++i) {
                auto *data = const_cast<double *>(spline.GetKnot(static_cast<int>(i)).data());

                this->AddParameterBlock(data, 3);
                paramBlockVec.push_back(data);
                // set this param block to be constant
                if (setToConst) { this->SetParameterBlockConstant(data); }

                // knot recoder
                knotRecoder[reinterpret_cast<long>(&spline)][i]++;
            }
        }
    }

    void Estimator::AddSo3KnotsData(std::vector<double *> &paramBlockVec,
                                    const Estimator::SplineBundleType::So3SplineType &spline,
                                    const Estimator::SplineMetaType &splineMeta, bool setToConst) {
        // for each segment
        for (const auto &seg: splineMeta.segments) {
            // the factor 'seg.dt * 0.5' is the treatment for numerical accuracy
            auto idxMaster = spline.ComputeTIndex(seg.t0 + seg.dt * 0.5).second;

            // from the first control point to the last control point
            for (std::size_t i = idxMaster; i < idxMaster + seg.NumParameters(); ++i) {
                auto *data = const_cast<double *>(spline.GetKnot(static_cast<int>(i)).data());
                // the local parameterization is very important!!!
                this->AddParameterBlock(data, 4, QUATER_MANIFOLD.get());

                paramBlockVec.push_back(data);
                // set this param block to be constant
                if (setToConst) { this->SetParameterBlockConstant(data); }

                // knot recoder
                knotRecoder[reinterpret_cast<long>(&spline)][i]++;
            }
        }
    }


    void Estimator::AddVelPIMForGravityRecovery(const double dt, const Eigen::Vector3d &deltaVel,
                                                const Eigen::Vector3d &velPim, GaRLILEO_Opt option, double weight) {
        auto costFunc = GravityFactor::Create(dt, deltaVel, velPim, weight);

        // gravity
        costFunc->AddParameterBlock(3);

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.push_back(gravity->data());

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        this->SetManifold(gravity->data(), GRAVITY_MANIFOLD.get());

        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_GRAVITY, option)) {
            this->SetParameterBlockConstant(gravity->data());
        }
    }

    Eigen::MatrixXd Estimator::CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobian_crs_matrix) {
        Eigen::MatrixXd J(jacobian_crs_matrix->num_rows, jacobian_crs_matrix->num_cols);
        J.setZero();

        std::vector<int> jacobian_crs_matrix_rows, jacobian_crs_matrix_cols;
        std::vector<double> jacobian_crs_matrix_values;
        jacobian_crs_matrix_rows = jacobian_crs_matrix->rows;
        jacobian_crs_matrix_cols = jacobian_crs_matrix->cols;
        jacobian_crs_matrix_values = jacobian_crs_matrix->values;

        int cur_index_in_cols_and_values = 0;
        // rows is a num_rows + 1 sized array
        int row_size = static_cast<int>(jacobian_crs_matrix_rows.size()) - 1;
        // outer loop traverse rows, inner loop traverse cols and values
        for (int row_index = 0; row_index < row_size; ++row_index) {
            while (cur_index_in_cols_and_values < jacobian_crs_matrix_rows[row_index + 1]) {
                J(row_index, jacobian_crs_matrix_cols[cur_index_in_cols_and_values]) =
                        jacobian_crs_matrix_values[cur_index_in_cols_and_values];
                cur_index_in_cols_and_values++;
            }
        }
        return J;
    }

    Eigen::MatrixXd Estimator::GetHessianMatrix() {
        ceres::Problem::EvaluateOptions EvalOpts;
        ceres::CRSMatrix jacobian_crs_matrix;
        this->Evaluate(EvalOpts, nullptr, nullptr, nullptr, &jacobian_crs_matrix);
        Eigen::MatrixXd J = CRSMatrix2EigenMatrix(&jacobian_crs_matrix);
        Eigen::MatrixXd H = J.transpose() * J;
        return H;
    }

    void Estimator::AddRadarMeasurement(const RadarTarget::Ptr &radarTar, GaRLILEO_Opt option, Sophus::SO3d& SO3_RefToW, double weight) {
        auto time = radarTar->GetTimestamp();

        if (!splines->TimeInRangeForSo3(time, Configor::Preference::SO3Spline)) { return; }
        if (!splines->TimeInRangeForRd(time, Configor::Preference::VelSpline)) { return; }

        SplineMetaType so3Meta, velMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);
        splines->CalculateRdSplineMeta(Configor::Preference::VelSpline, {{time, time}}, velMeta);

        const bool useKAOC = configor->prior.RadarUseKAOC;
        KAOCParams kaocP{};
        if (useKAOC) {
            kaocP.sigma_vd = configor->prior.RadarDopplerNoiseStd;
            kaocP.sigma_az_base = configor->prior.RadarAzimuthNoiseBase;
            kaocP.sigma_el_base = configor->prior.RadarElevationNoiseBase;
            kaocP.k_degrad = configor->prior.RadarSpatialDegradationCoeff;
            kaocP.clip = configor->prior.RadarKAOCClipEnable;
            kaocP.sqrt_omega_min = configor->prior.RadarKAOCSqrtOmegaMin;
            kaocP.sqrt_omega_max = configor->prior.RadarKAOCSqrtOmegaMax;
        }

        auto costFunc = RadarFactor<Configor::Prior::SplineOrder>::Create(
                SO3_RefToW, radarTar, so3Meta, velMeta, weight, useKAOC, kaocP
        );

        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }
        for (int i = 0; i < static_cast<int>(velMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }

        costFunc->AddParameterBlock(4); // SO3_RtoB
        costFunc->AddParameterBlock(3); // POS_RinB
        costFunc->AddParameterBlock(1); // TIME_OFFSET_RtoB

        costFunc->SetNumResiduals(1);
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters() + velMeta.NumParameters() + 3);

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_SO3, option)
        );
        AddRdKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::VelSpline), velMeta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)
        );

        auto *SO3_RtoB = configor->dataStream.CalibParam.SO3_RtoB.data();
        auto *POS_RinB = configor->dataStream.CalibParam.POS_RinB.data();
        auto *TO_RtoB  = &configor->dataStream.CalibParam.TIME_OFFSET_RtoB;
        this->AddParameterBlock(SO3_RtoB, 4, QUATER_MANIFOLD.get());
        this->AddParameterBlock(POS_RinB, 3);
        this->AddParameterBlock(TO_RtoB, 1);
        paramBlockVec.push_back(SO3_RtoB);
        paramBlockVec.push_back(POS_RinB);
        paramBlockVec.push_back(TO_RtoB);

        this->AddResidualBlock(
                costFunc, new ceres::CauchyLoss(configor->prior.CauchyLossForRadarFactor * weight), paramBlockVec
        );

        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_SO3_RtoB, option)) {
            this->SetParameterBlockConstant(SO3_RtoB);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_POS_RinB, option)) {
            this->SetParameterBlockConstant(POS_RinB);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_TO_RtoB, option)) {
            this->SetParameterBlockConstant(TO_RtoB);
        } else {
            // Hard search bound around 0 (the YAML "TIME_OFFSET_RtoB" applies the
            // coarse offset at message buffering, so the optimisable delta must
            // stay within ±RadarImuOnlineCalibTOMaxAbs of zero).  Bounding here
            // also keeps the jet-shifted spline-meta query inside its segment.
            const double max_abs = std::abs(configor->prior.RadarImuOnlineCalibTOMaxAbs);
            if (max_abs > 0.0) {
                this->SetParameterLowerBound(TO_RtoB, 0, -max_abs);
                this->SetParameterUpperBound(TO_RtoB, 0,  max_abs);
            }
        }
    }

    void Estimator::AddRadar1Measurement(const RadarTarget::Ptr &radarTar, GaRLILEO_Opt option, Sophus::SO3d& SO3_RefToW, double weight) {
        auto time = radarTar->GetTimestamp();

        if (!splines->TimeInRangeForSo3(time, Configor::Preference::SO3Spline)) { return; }
        if (!splines->TimeInRangeForRd(time, Configor::Preference::VelSpline)) { return; }

        SplineMetaType so3Meta, velMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);
        splines->CalculateRdSplineMeta(Configor::Preference::VelSpline, {{time, time}}, velMeta);

        const bool useKAOC = configor->prior.RadarUseKAOC;
        KAOCParams kaocP{};
        if (useKAOC) {
            kaocP.sigma_vd = configor->prior.RadarDopplerNoiseStd;
            kaocP.sigma_az_base = configor->prior.RadarAzimuthNoiseBase;
            kaocP.sigma_el_base = configor->prior.RadarElevationNoiseBase;
            kaocP.k_degrad = configor->prior.RadarSpatialDegradationCoeff;
            kaocP.clip = configor->prior.RadarKAOCClipEnable;
            kaocP.sqrt_omega_min = configor->prior.RadarKAOCSqrtOmegaMin;
            kaocP.sqrt_omega_max = configor->prior.RadarKAOCSqrtOmegaMax;
        }

        auto costFunc = RadarFactor<Configor::Prior::SplineOrder>::Create(
                SO3_RefToW, radarTar, so3Meta, velMeta, weight, useKAOC, kaocP
        );

        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }
        for (int i = 0; i < static_cast<int>(velMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }

        costFunc->AddParameterBlock(4); // SO3_R1toB
        costFunc->AddParameterBlock(3); // POS_R1inB
        costFunc->AddParameterBlock(1); // TIME_OFFSET_R1toB

        costFunc->SetNumResiduals(1);
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters() + velMeta.NumParameters() + 3);

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_SO3, option)
        );
        AddRdKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::VelSpline), velMeta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)
        );

        auto *SO3_R1toB = configor->dataStream.CalibParam.SO3_R1toB.data();
        auto *POS_R1inB = configor->dataStream.CalibParam.POS_R1inB.data();
        auto *TO_R1toB  = &configor->dataStream.CalibParam.TIME_OFFSET_R1toB;
        this->AddParameterBlock(SO3_R1toB, 4, QUATER_MANIFOLD.get());
        this->AddParameterBlock(POS_R1inB, 3);
        this->AddParameterBlock(TO_R1toB, 1);
        paramBlockVec.push_back(SO3_R1toB);
        paramBlockVec.push_back(POS_R1inB);
        paramBlockVec.push_back(TO_R1toB);

        this->AddResidualBlock(
                costFunc, new ceres::CauchyLoss(configor->prior.CauchyLossForRadarFactor * weight), paramBlockVec
        );

        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_SO3_R1toB, option)) {
            this->SetParameterBlockConstant(SO3_R1toB);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_POS_R1inB, option)) {
            this->SetParameterBlockConstant(POS_R1inB);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_TO_R1toB, option)) {
            this->SetParameterBlockConstant(TO_R1toB);
        } else {
            const double max_abs = std::abs(configor->prior.Radar1ImuOnlineCalibTOMaxAbs);
            if (max_abs > 0.0) {
                this->SetParameterLowerBound(TO_R1toB, 0, -max_abs);
                this->SetParameterUpperBound(TO_R1toB, 0,  max_abs);
            }
        }
    }


    void Estimator::AddGyroMeasurementWithConstBias(const IMUFrame::Ptr &frame, GaRLILEO_Opt option, double weight) {
        auto time = frame->GetTimestamp();

        if (!splines->TimeInRangeForSo3(time, Configor::Preference::SO3Spline)) {
            // if this frame is not in range
            return;
        }

        // prepare metas for splines
        SplineMetaType so3Meta; 
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);

        auto costFunc = IMUGyroFactorWithConstBias<Configor::Prior::SplineOrder>::Create(so3Meta, frame, weight);
        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) { 
            costFunc->AddParameterBlock(4);
        }


        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters());

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_SO3, option)
        );

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    void Estimator::AddAcceMeasurementWithConstBias(const IMUFrame::Ptr &frame, GaRLILEO_Opt option, double weight) {
        auto time = frame->GetTimestamp();

        if (!splines->TimeInRangeForSo3(time, Configor::Preference::SO3Spline)) {
            // if this frame is not in range
            return;
        }
        if (!splines->TimeInRangeForRd(time, Configor::Preference::VelSpline)) {
            // if this frame is not in range
            return;
        }

        SplineMetaType so3Meta, velMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);
        splines->CalculateRdSplineMeta(Configor::Preference::VelSpline, {{time, time}}, velMeta);

        auto costFunc = IMUAcceFactorWithConstBias<Configor::Prior::SplineOrder>::Create(
                configor->dataStream.CalibParam, so3Meta, velMeta, frame, weight
        );
        

        // knots of so3 spline
        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }
        // knots of vel spline
        for (int i = 0; i < static_cast<int>(velMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }
        // BIAS OF ACCE
        costFunc->AddParameterBlock(3);
        // gravity
        costFunc->AddParameterBlock(3);

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters() + velMeta.NumParameters() + 2);
        // knots of so3 spline
        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_SO3, option)
        );
        // knots of vel spline
        AddRdKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::VelSpline), velMeta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)
        );
        // BIAS OF ACCE
        paramBlockVec.push_back(ba->data());
        // gravity
        paramBlockVec.push_back(gravity->data());

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        this->SetManifold(gravity->data(), GRAVITY_MANIFOLD.get());

        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_BA, option)) {
            this->SetParameterBlockConstant(ba->data());
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_GRAVITY, option)) {
            this->SetParameterBlockConstant(gravity->data());
        }
        this->SetParameterBlockConstant(gravity->data());
    }

    void Estimator::ShowKnotStatus() const {
        for (const auto &[splineAddress, knotInfo]: knotRecoder) {
            std::stringstream stream;
            stream << "spline: " << splineAddress << ", ";
            for (const auto &[knotId, count]: knotInfo) {
                stream << '[' << knotId << ": " << count << "] ";
            }
            spdlog::info("{}", stream.str());
        }
    }

    ceres::ResidualBlockId Estimator::AddAcceBiasPriori(const BiasFilter::StatePack &priori, GaRLILEO_Opt option) {
        auto costFunc = BiasFactor::Create(priori.state, priori.var);
        costFunc->AddParameterBlock(3);
        costFunc->SetNumResiduals(3);
        auto id = this->AddResidualBlock(costFunc, nullptr, ba->data());
        
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_BA, option)) {
            this->SetParameterBlockConstant(ba->data());
        }
        return id;
    }

    ceres::ResidualBlockId Estimator::AddVelBiasPriori(const VelBiasFilter::StatePack &priori, GaRLILEO_Opt option) {
        auto costFunc = VelBiasFactor::Create(priori.state, priori.var);
        costFunc->AddParameterBlock(2);
        costFunc->SetNumResiduals(2);
        auto id = this->AddResidualBlock(costFunc, nullptr, bv->data());

        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_BA, option)) {
            this->SetParameterBlockConstant(bv->data());
        }
        return id;
    }

    std::vector<ceres::ResidualBlockId> Estimator::AddVelSplineTailConstraint(GaRLILEO_Opt option, double weight) {
        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        std::vector<ceres::ResidualBlockId> idVec;
        for (int j = 0; j < Configor::Prior::SplineOrder - 2; ++j) {
            auto costFunc = RdTailFactor::Create(weight);
            costFunc->AddParameterBlock(3);
            costFunc->AddParameterBlock(3);
            costFunc->AddParameterBlock(3);
            costFunc->SetNumResiduals(3);

            // organize the param block vector
            std::vector<double *> paramBlockVec(3);
            for (int i = 0; i < 3; ++i) {
                paramBlockVec.at(i) = velSpline.GetKnot(
                        j + i + static_cast<int>(velSpline.GetKnots().size()) - Configor::Prior::SplineOrder
                ).data();

            }

            auto id = this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
            idVec.push_back(id);

            if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)) {
                for (auto &knot: paramBlockVec) { this->SetParameterBlockConstant(knot); }
            }
        }
        return idVec;
    }

    std::vector<ceres::ResidualBlockId> Estimator::AddGravSplineTailConstraint(GaRLILEO_Opt option, int old_size, double weight){
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        std::vector<ceres::ResidualBlockId> idVec;
        for (int j = 0; j < Configor::Prior::SplineOrder - 2; ++j) {
            auto costFunc = RdTailFactor::Create(weight);
            costFunc->AddParameterBlock(3);
            costFunc->AddParameterBlock(3);
            costFunc->AddParameterBlock(3);
            costFunc->SetNumResiduals(3);

            // organize the param block vector
            std::vector<double *> paramBlockVec(3);
            for (int i = 0; i < 3; ++i) {
                paramBlockVec.at(i) = gravSpline.GetKnot(
                        old_size + i
                ).data();

            }

            auto id = this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
            idVec.push_back(id);

            if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)) {
                for (auto &knot: paramBlockVec) { this->SetParameterBlockConstant(knot); }
            }

        }
        return idVec;
    }


    std::vector<ceres::ResidualBlockId> Estimator::AddSo3SplineTailConstraint(GaRLILEO_Opt option, double weight) {
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        std::vector<ceres::ResidualBlockId> idVec;
        for (int j = 0; j < Configor::Prior::SplineOrder - 2; ++j) {
            auto costFunc = So3TailFactor::Create(weight);
            costFunc->AddParameterBlock(4);
            costFunc->AddParameterBlock(4);
            costFunc->AddParameterBlock(4);
            costFunc->SetNumResiduals(3);

            // organize the param block vector
            std::vector<double *> paramBlockVec(3);
            for (int i = 0; i < 3; ++i) {
                paramBlockVec.at(i) = so3Spline.GetKnot(
                        j + i + static_cast<int>(so3Spline.GetKnots().size()) - Configor::Prior::SplineOrder
                ).data();
            }

            auto id = this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
            idVec.push_back(id);

            for (const auto &item: paramBlockVec) { this->SetManifold(item, QUATER_MANIFOLD.get()); }

            if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_SO3, option)) {
                for (auto &knot: paramBlockVec) { this->SetParameterBlockConstant(knot); }
            }
        }
        return idVec;
    }

    ceres::ResidualBlockId Estimator::AddGravityRotConstraint(double st, double et, const Eigen::Vector3d &alpha, 
                                            Sophus::SO3d& rot1, Sophus::SO3d& rot2, double weight) {
        if (!splines->TimeInRangeForRd(st, Configor::Preference::VelSpline) ||
            !splines->TimeInRangeForRd(et, Configor::Preference::VelSpline) ||
            !splines->TimeInRangeForRd(st, Configor::Preference::GravitySpline) ||
            !splines->TimeInRangeForRd(et, Configor::Preference::GravitySpline)) {
            return nullptr;
        }

        SplineMetaType gravMeta, velMeta;
        splines->CalculateRdSplineMeta(Configor::Preference::GravitySpline, {{st, st}}, gravMeta);
        splines->CalculateRdSplineMeta(Configor::Preference::VelSpline, {{st, st},{et, et}}, velMeta);

        auto costFunc = GravityUpdateFactor<Configor::Prior::SplineOrder>::Create(
                gravMeta, velMeta, st, et, alpha, rot1, rot2, weight
        );

        // local grav
        // knots of vel spline
        for (int i = 0; i < static_cast<int>(gravMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }

        for (int i = 0; i < static_cast<int>(velMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(gravMeta.NumParameters() + velMeta.NumParameters());

        AddGravKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::GravitySpline), gravMeta,
                false
        );

        // knots of vel spline
        AddRdKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::VelSpline), velMeta,
                false
        );
        // knots of vel spline
        

        auto id = this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        
        return id;
    }

    ceres::ResidualBlockId Estimator::AddGravitySplineConstraint(const IMUFrame::Ptr &frame, double weight){

        auto time = frame->GetTimestamp();
        if (!splines->TimeInRangeForRd(time, Configor::Preference::GravitySpline)) {
            return nullptr;
        }

        SplineMetaType so3Meta, gravMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);
        splines->CalculateRdSplineMeta(Configor::Preference::GravitySpline, {{time, time}}, gravMeta);

        auto costFunc = GravitySplineFactor<Configor::Prior::SplineOrder>::Create(gravMeta, so3Meta, frame, weight);
        

        for (int i = 0; i < static_cast<int>(gravMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }

        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }


        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(gravMeta.NumParameters() + so3Meta.NumParameters());

        AddGravKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::GravitySpline), gravMeta,
                false
        );

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                false
        );

        auto id = this->AddResidualBlock(costFunc, nullptr, paramBlockVec);

        return id;

    }

    void Estimator::AddRotRefine(long idx, double weight) {
        
        auto costFunc = RotRefineFactor<Configor::Prior::SplineOrder>::Create(weight);
        
        // Param for rotation   
        costFunc->AddParameterBlock(4);
        // knots of local gravity spline
        costFunc->AddParameterBlock(3);
        // Param for global gravity
        costFunc->AddParameterBlock(3);

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(3);

        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        
        auto *data = const_cast<double *>(so3Spline.GetKnot(static_cast<int>(idx)).data());
        this->AddParameterBlock(data, 4, QUATER_MANIFOLD.get());
        paramBlockVec.push_back(data);
        // knot recoder
        knotRecoder[reinterpret_cast<long>(&so3Spline)][idx]++;

        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        auto *grav_data = const_cast<double *>(gravSpline.GetKnot(static_cast<int>(idx)).data());
        this->AddParameterBlock(grav_data, 3, GRAVITY_MANIFOLD.get());
        paramBlockVec.push_back(grav_data);
        this->SetParameterBlockConstant(grav_data);
        // knot recoder
        knotRecoder[reinterpret_cast<long>(&grav_data)][idx]++;
  
        paramBlockVec.push_back(gravity->data());

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);

        this->SetManifold(gravity->data(), GRAVITY_MANIFOLD.get());
        this->SetParameterBlockConstant(gravity->data());

        return;       
        
    }

    void Estimator::AddRotPrior(double st, double dt, double weight) {

        if (!splines->TimeInRangeForSo3(st, Configor::Preference::SO3Spline)) {
            return;
        }
        
        // prepare metas for splines
        SplineMetaType so3Meta, gravMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{st,st},
                                                                          {st+dt, st+dt}}, so3Meta);

        auto costFunc = RotPriorFactor<Configor::Prior::SplineOrder>::Create(st, dt, so3Meta, weight);

        // knots of so3 spline
        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters());

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                false
        );

        
        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        return;       
        
    }

    void Estimator::AddLegMeasurement(const LegFrame::Ptr &frame, GaRLILEO_Opt option, Sophus::SO3d& SO3_RefToW, double weight) {
        Eigen::Vector3d leg_vel = frame->GetVel();
        if(leg_vel(2)>0.75){
            return;
        }


        auto time = frame->GetTimestamp();

        if (!splines->TimeInRangeForRd(time, Configor::Preference::VelSpline)) {
            return;
        }

        SplineMetaType so3Meta, velMeta;
        splines->CalculateSo3SplineMeta(Configor::Preference::SO3Spline, {{time, time}}, so3Meta);
        splines->CalculateRdSplineMeta(Configor::Preference::VelSpline, {{time, time}}, velMeta);

        double effective_weight = weight;
        if (configor->prior.LegDynamicWeightEnable) {
            const Eigen::Matrix3d vel_cov = frame->GetVelCov();
            if (vel_cov.allFinite()) {
                const double cov_trace = std::max(1.0e-12, vel_cov.trace());
                const double cov_ref =
                    std::max(1.0e-12, configor->prior.LegDynamicWeightReferenceCovTrace);
                double scale = cov_ref / cov_trace;
                scale = std::clamp(scale,
                                   configor->prior.LegDynamicWeightMinScale,
                                   configor->prior.LegDynamicWeightMaxScale);
                effective_weight *= scale;
            }
        }

        auto costFunc = LegVelFactor<Configor::Prior::SplineOrder>::Create(
            so3Meta, velMeta, frame, SO3_RefToW, effective_weight);

        for (int i = 0; i < static_cast<int>(so3Meta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(4);
        }

        for (int i = 0; i < static_cast<int>(velMeta.NumParameters()); ++i) {
            costFunc->AddParameterBlock(3);
        }
        //add velocity bias
        costFunc->AddParameterBlock(2);
        // leg-imu extrinsics (online calibration)
        costFunc->AddParameterBlock(4); // SO3_BtoL
        costFunc->AddParameterBlock(3); // POS_BinL
        costFunc->AddParameterBlock(1); // TIME_OFFSET_BtoL

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.reserve(so3Meta.NumParameters()+velMeta.NumParameters()+4);

        AddSo3KnotsData(
                paramBlockVec, splines->GetSo3Spline(Configor::Preference::SO3Spline), so3Meta,
                false
        );

        AddRdKnotsData(
                paramBlockVec, splines->GetRdSpline(Configor::Preference::VelSpline), velMeta,
                !GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_VEL, option)
        );

        paramBlockVec.push_back(bv->data());

        // extrinsics parameter blocks
        auto *SO3_BtoL = configor->dataStream.CalibParam.SO3_BtoL.data();
        auto *POS_BinL = configor->dataStream.CalibParam.POS_BinL.data();
        auto *TO_BtoL  = &configor->dataStream.CalibParam.TIME_OFFSET_BtoL;
        this->AddParameterBlock(SO3_BtoL, 4, QUATER_MANIFOLD.get());
        this->AddParameterBlock(POS_BinL, 3);
        this->AddParameterBlock(TO_BtoL, 1);
        paramBlockVec.push_back(SO3_BtoL);
        paramBlockVec.push_back(POS_BinL);
        paramBlockVec.push_back(TO_BtoL);

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_SO3_BtoL, option)) {
            this->SetParameterBlockConstant(SO3_BtoL);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_POS_BinL, option)) {
            this->SetParameterBlockConstant(POS_BinL);
        }
        if (!GaRLILEO_OptOption::IsOptionWith(GaRLILEO_Opt::OPT_EXTRI_TO_BtoL, option)) {
            this->SetParameterBlockConstant(TO_BtoL);
        } else {
            const double max_abs = std::abs(configor->prior.LegImuOnlineCalibTOMaxAbs);
            if (max_abs > 0.0) {
                this->SetParameterLowerBound(TO_BtoL, 0, -max_abs);
                this->SetParameterUpperBound(TO_BtoL, 0,  max_abs);
            }
        }
        return;
    }

    void Estimator::AddStationaryGravity(Eigen::Vector3d &acceMean) {
        auto costFunc = GravityStationaryFactor::Create(acceMean, 10);
        // // organize the param block vector
        // gravity
        costFunc->AddParameterBlock(3);

        costFunc->SetNumResiduals(3);

        // organize the param block vector
        std::vector<double *> paramBlockVec;
        paramBlockVec.push_back(gravity->data());

        this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
        this->SetManifold(gravity->data(), GRAVITY_MANIFOLD.get());
        return;
    }

    ceres::ResidualBlockId Estimator::AddRadarImuExtriSO3Prior(const Sophus::SO3d &SO3_prior, double weight) {
        auto costFunc = PriorExtriSO3Factor::Create(SO3_prior, weight);
        costFunc->AddParameterBlock(4);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *SO3_RtoB = configor->dataStream.CalibParam.SO3_RtoB.data();
        this->AddParameterBlock(SO3_RtoB, 4, QUATER_MANIFOLD.get());
        paramBlockVec.push_back(SO3_RtoB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddRadarImuExtriPOSPrior(const Eigen::Vector3d &POS_prior, double weight) {
        auto costFunc = PriorExtriPOSFactor::Create(POS_prior, weight);
        costFunc->AddParameterBlock(3);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *POS_RinB = configor->dataStream.CalibParam.POS_RinB.data();
        this->AddParameterBlock(POS_RinB, 3);
        paramBlockVec.push_back(POS_RinB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddRadar1ImuExtriSO3Prior(const Sophus::SO3d &SO3_prior, double weight) {
        auto costFunc = PriorExtriSO3Factor::Create(SO3_prior, weight);
        costFunc->AddParameterBlock(4);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *SO3_R1toB = configor->dataStream.CalibParam.SO3_R1toB.data();
        this->AddParameterBlock(SO3_R1toB, 4, QUATER_MANIFOLD.get());
        paramBlockVec.push_back(SO3_R1toB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddRadar1ImuExtriPOSPrior(const Eigen::Vector3d &POS_prior, double weight) {
        auto costFunc = PriorExtriPOSFactor::Create(POS_prior, weight);
        costFunc->AddParameterBlock(3);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *POS_R1inB = configor->dataStream.CalibParam.POS_R1inB.data();
        this->AddParameterBlock(POS_R1inB, 3);
        paramBlockVec.push_back(POS_R1inB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddLegImuExtriSO3Prior(const Sophus::SO3d &SO3_prior, double weight) {
        auto costFunc = PriorExtriSO3Factor::Create(SO3_prior, weight);
        costFunc->AddParameterBlock(4);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *SO3_BtoL = configor->dataStream.CalibParam.SO3_BtoL.data();
        this->AddParameterBlock(SO3_BtoL, 4, QUATER_MANIFOLD.get());
        paramBlockVec.push_back(SO3_BtoL);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddLegImuExtriPOSPrior(const Eigen::Vector3d &POS_prior, double weight) {
        auto costFunc = PriorExtriPOSFactor::Create(POS_prior, weight);
        costFunc->AddParameterBlock(3);
        costFunc->SetNumResiduals(3);

        std::vector<double *> paramBlockVec;
        auto *POS_BinL = configor->dataStream.CalibParam.POS_BinL.data();
        this->AddParameterBlock(POS_BinL, 3);
        paramBlockVec.push_back(POS_BinL);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    // -------------------------
    // Time-offset prior factors
    // -------------------------
    ceres::ResidualBlockId Estimator::AddRadarImuExtriTOPrior(double TO_prior, double weight) {
        auto costFunc = PriorTimeOffsetFactor::Create(TO_prior, weight);
        costFunc->AddParameterBlock(1);
        costFunc->SetNumResiduals(1);

        std::vector<double *> paramBlockVec;
        auto *TO_RtoB = &configor->dataStream.CalibParam.TIME_OFFSET_RtoB;
        this->AddParameterBlock(TO_RtoB, 1);
        paramBlockVec.push_back(TO_RtoB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddRadar1ImuExtriTOPrior(double TO_prior, double weight) {
        auto costFunc = PriorTimeOffsetFactor::Create(TO_prior, weight);
        costFunc->AddParameterBlock(1);
        costFunc->SetNumResiduals(1);

        std::vector<double *> paramBlockVec;
        auto *TO_R1toB = &configor->dataStream.CalibParam.TIME_OFFSET_R1toB;
        this->AddParameterBlock(TO_R1toB, 1);
        paramBlockVec.push_back(TO_R1toB);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    ceres::ResidualBlockId Estimator::AddLegImuExtriTOPrior(double TO_prior, double weight) {
        auto costFunc = PriorTimeOffsetFactor::Create(TO_prior, weight);
        costFunc->AddParameterBlock(1);
        costFunc->SetNumResiduals(1);

        std::vector<double *> paramBlockVec;
        auto *TO_BtoL = &configor->dataStream.CalibParam.TIME_OFFSET_BtoL;
        this->AddParameterBlock(TO_BtoL, 1);
        paramBlockVec.push_back(TO_BtoL);

        return this->AddResidualBlock(costFunc, nullptr, paramBlockVec);
    }

    const std::map<long, std::map<std::size_t, int>>& Estimator::getKnotRecoder() const {
        return knotRecoder;
    }
}
