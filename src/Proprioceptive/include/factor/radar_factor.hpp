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

#ifndef GARLILEO_RADAR_FACTOR_HPP
#define GARLILEO_RADAR_FACTOR_HPP

#include <utility>
#include <cmath>
#include <type_traits>
#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/utils/sophus_utils.hpp"
#include "ctraj/spline/ceres_spline_helper_jet.h"
#include <Eigen/Geometry>
#include "sensor/imu.h"

namespace garlileo {

    // =========================================================================
    // K-AOC helper: computes per-point Doppler information weight from geometry
    // =========================================================================
    struct KAOCParams {
        double sigma_vd;
        double sigma_az_base;
        double sigma_el_base;
        double k_degrad;
        // Normalize sqrt(Omega) by the Doppler-only information and clamp it.
        bool clip = false;
        double sqrt_omega_min = 0.8;
        double sqrt_omega_max = 1.2;
    };

    inline double ComputeKAOCWeight(const KAOCParams &p,
                                    const Eigen::Vector3d &target_xyz,
                                    const Eigen::Matrix3d &R_BtoR,
                                    const Eigen::Vector3d &v_radar_B) {
        const double r = target_xyz.norm();
        if (r < 1e-6) return 1.0 / (p.sigma_vd * p.sigma_vd);

        const double x = target_xyz.x(), y = target_xyz.y(), z = target_xyz.z();
        const double r_xy = std::sqrt(x * x + y * y);

        // azimuth (theta) and elevation (phi) in radar frame
        const double theta = std::atan2(y, x);
        const double phi = std::atan2(z, r_xy);

        // 1) Spatial angle variance with degradation
        const double sig_th2 = p.sigma_az_base * p.sigma_az_base * (1.0 + p.k_degrad * theta * theta);
        const double sig_ph2 = p.sigma_el_base * p.sigma_el_base * (1.0 + p.k_degrad * phi * phi);

        // 2) Jacobian of unit direction vector w.r.t. (theta, phi)
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double ct = std::cos(theta), st = std::sin(theta);
        Eigen::Matrix<double, 3, 2> J_u;
        J_u << -cp * st, -sp * ct,
                cp * ct, -sp * st,
                0.0,      cp;

        // C_Theta = diag(sig_theta^2, sig_phi^2)
        Eigen::Matrix2d C_Theta;
        C_Theta << sig_th2, 0.0,
                   0.0, sig_ph2;

        // 3) Unit vector covariance in radar frame, then transform to body
        const Eigen::Matrix3d Sigma_u_R = J_u * C_Theta * J_u.transpose();
        const Eigen::Matrix3d R_RtoB = R_BtoR.transpose();
        const Eigen::Matrix3d Sigma_u_B = R_RtoB * Sigma_u_R * R_RtoB.transpose();

        // 4) Kinematics-aware Doppler variance
        const double sigma_d2 = p.sigma_vd * p.sigma_vd
                                + v_radar_B.transpose() * Sigma_u_B * v_radar_B;

        return 1.0 / std::max(sigma_d2, 1e-12);
    }

    // =========================================================================
    // RadarFactor: per-point Doppler factor with K-AOC dynamic weighting and
    // optional online temporal calibration.
    //
    // Parameter block layout (DynamicAutoDiff):
    //   [ SO3 knots... | VEL knots... | SO3_RtoB(4) | POS_RinB(3) | TIME_OFFSET(1) ]
    //
    // The factor evaluates the SO3/velocity splines at  t_eval = t_radar + TO
    // so that a non-zero TO update directly compensates for residual delay
    // between the radar timestamp (already pre-corrected by the static
    // TIME_OFFSET_RtoB applied in DataManager) and the IMU clock that drives
    // the spline knots. This mirrors iKalibr's RadarFactor.
    // =========================================================================
    template<int Order>
    struct RadarFactor {
    public:
        using SplineMetaType = ns_ctraj::SplineMeta<Configor::Prior::SplineOrder>;

    private:
        SplineMetaType so3Meta_;
        SplineMetaType velMeta_;
        RadarTarget::Ptr target;
        Sophus::SO3d SO3_RefToW;

        double weight;
        double so3DtInv, velDtInv;

        bool useKAOC;
        KAOCParams kaocParams;

        // Constant per-factor offsets into parBlocks. The SO3/VEL knot offsets
        // depend on the (time-shifted) spline index and are computed inside
        // operator() at every evaluation.
        std::size_t EXTRI_SO3_OFFSET, EXTRI_POS_OFFSET, EXTRI_TO_OFFSET;

    public:
        RadarFactor(Sophus::SO3d& SO3_RefToW, RadarTarget::Ptr radarTar, const SplineMetaType &so3Meta,
                    const SplineMetaType &velMeta, double weight,
                    bool useKAOC = false, const KAOCParams &kaocP = {})
                : so3Meta_(so3Meta), velMeta_(velMeta),
                  target(std::move(radarTar)), SO3_RefToW(SO3_RefToW), weight(weight),
                  so3DtInv(1.0 / so3Meta.segments.front().dt), velDtInv(1.0 / velMeta.segments.front().dt),
                  useKAOC(useKAOC), kaocParams(kaocP) {
            EXTRI_SO3_OFFSET = so3Meta.NumParameters() + velMeta.NumParameters();
            EXTRI_POS_OFFSET = EXTRI_SO3_OFFSET + 1;
            EXTRI_TO_OFFSET  = EXTRI_POS_OFFSET + 1;
        }

        static auto Create(Sophus::SO3d& SO3_RefToW, const RadarTarget::Ptr &radarTar,
                           const SplineMetaType &so3Meta, const SplineMetaType &velMeta, double weight,
                           bool useKAOC = false, const KAOCParams &kaocP = {}) {
            return new ceres::DynamicAutoDiffCostFunction<RadarFactor>(
                    new RadarFactor(SO3_RefToW, radarTar, so3Meta, velMeta, weight, useKAOC, kaocP)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(RadarFactor).hash_code();
        }

    public:
        // Helper: clamp the time-shifted query into the cached spline-meta
        // segment so that ComputeSplineIndex does not assert.
        //
        // The spline meta is built from a single (target_t, target_t) range
        // and therefore covers exactly one knot interval (dt wide).  Two
        // benign situations can push the query outside that interval:
        //   - the target timestamp itself is right at the segment border and
        //     small numerical noise tips it past MaxTime();
        //   - online temporal calibration (TO) shifts the query by an amount
        //     that exceeds the segment width.
        // In both cases the assertion in ctraj's SplineMeta::ComputeSplineIndex
        // would crash the whole node.  We instead disable the factor for this
        // evaluation by returning a zero residual.
        template<class T>
        bool QueryInsideSplineMeta(const T &t_eval) const {
            const auto &so3Seg = so3Meta_.segments.front();
            const auto &velSeg = velMeta_.segments.front();
            const T so3Min = T(so3Seg.MinTime());
            const T so3Max = T(so3Seg.MaxTime() - 1e-6);
            const T velMin = T(velSeg.MinTime());
            const T velMax = T(velSeg.MaxTime() - 1e-6);
            return (t_eval >= so3Min) && (t_eval < so3Max) &&
                   (t_eval >= velMin) && (t_eval < velMax);
        }

        template<class T>
        bool operator()(T const *const *parBlocks, T *sResiduals) const {
            Eigen::Map<Eigen::Vector1<T>> residuals(sResiduals);

            // ---- Time-shifted spline indexing (iKalibr-style) ----
            const T TO = parBlocks[EXTRI_TO_OFFSET][0];
            const T t_eval = T(target->GetTimestamp()) + TO;

            if (!QueryInsideSplineMeta(t_eval)) {
                residuals.setZero();
                return true;
            }

            std::pair<std::size_t, T> so3IU, velIU;
            so3Meta_.template ComputeSplineIndex(t_eval, so3IU.first, so3IU.second);
            velMeta_.template ComputeSplineIndex(t_eval, velIU.first, velIU.second);

            const std::size_t SO3_OFFSET = so3IU.first;
            const std::size_t VEL_OFFSET = so3Meta_.NumParameters() + velIU.first;

            // The CeresSplineHelperJet variant accepts a Jet-typed time argument,
            // which is required because we shift the spline lookup time by the
            // (jet-valued) TIME_OFFSET parameter above.
            Sophus::SO3<T> SO3_CurToRef;
            Sophus::SO3Tangent<T> ANG_VEL_CurToRefInCur;
            ns_ctraj::CeresSplineHelperJet<T, Order>::template EvaluateLie<Sophus::SO3>(
                    parBlocks + SO3_OFFSET, so3IU.second, so3DtInv, &SO3_CurToRef, &ANG_VEL_CurToRefInCur
            );

            Eigen::Vector3<T> LIN_VEL_CurToRefInCur;
            ns_ctraj::CeresSplineHelperJet<T, Order>::template Evaluate<3, 0>(
                    parBlocks + VEL_OFFSET, velIU.second, velDtInv, &LIN_VEL_CurToRefInCur
            );

            Eigen::Vector3<T> LIN_VEL_RtoRefInCur =
                    Sophus::SO3<T>::hat(ANG_VEL_CurToRefInCur) * Eigen::Map<const Eigen::Vector3<T>>(parBlocks[EXTRI_POS_OFFSET]) +
                    LIN_VEL_CurToRefInCur;

            T v1 = -target->GetTargetXYZ().cast<T>().dot(
                    Sophus::SO3<T>(Eigen::Map<const Eigen::Quaternion<T>>(parBlocks[EXTRI_SO3_OFFSET]))
                        .matrix().transpose() * LIN_VEL_RtoRefInCur
            );

            T v2 = static_cast<T>(target->GetRadialVelocity());

            // Compute effective weight: base weight * sqrt(Omega_kaoc)
            T eff_weight = T(weight);
            if (useKAOC) {
                auto toDouble = [](const T &val) -> double {
                    if constexpr (std::is_same_v<T, double>) return val;
                    else return val.a;
                };

                const Eigen::Quaterniond q_extri(
                        toDouble(parBlocks[EXTRI_SO3_OFFSET][3]),
                        toDouble(parBlocks[EXTRI_SO3_OFFSET][0]),
                        toDouble(parBlocks[EXTRI_SO3_OFFSET][1]),
                        toDouble(parBlocks[EXTRI_SO3_OFFSET][2]));
                const Eigen::Matrix3d R_BtoR_d = Sophus::SO3d(q_extri).matrix().transpose();

                Eigen::Vector3d omega_d, vel_d, pos_d;
                for (int j = 0; j < 3; ++j) {
                    omega_d(j) = toDouble(ANG_VEL_CurToRefInCur(j));
                    vel_d(j)   = toDouble(LIN_VEL_CurToRefInCur(j));
                    pos_d(j)   = toDouble(parBlocks[EXTRI_POS_OFFSET][j]);
                }
                const Eigen::Vector3d v_radar_B = omega_d.cross(pos_d) + vel_d;

                const double Omega_i = ComputeKAOCWeight(kaocParams, target->GetTargetXYZ(), R_BtoR_d, v_radar_B);
                double sqrtOmega = std::sqrt(Omega_i);
                if (kaocParams.clip) {
                    sqrtOmega = std::clamp(kaocParams.sigma_vd * sqrtOmega,
                                           kaocParams.sqrt_omega_min, kaocParams.sqrt_omega_max);
                }
                eff_weight = T(weight * sqrtOmega);
            }

            residuals(0, 0) = eff_weight * (target->GetInvRange() * v1 - v2);

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}
#endif
