// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#ifndef LEG_VEL_FACTOR_HPP
#define LEG_VEL_FACTOR_HPP

#include <utility>
#include <Eigen/Geometry>
#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/utils/sophus_utils.hpp"
#include "ctraj/spline/ceres_spline_helper_jet.h"
#include "sensor/leg_data_loader.h"

namespace garlileo {
    // Leg velocity factor for continuous-time velocity spline optimisation.
    //
    // Pre-computed quantities stored in LegFrame (in the kinematic L frame):
    //   v_kin_fused  = information-weighted combination of per-leg  -J_i * qdot_i
    //   F_omega      = information-weighted omega coupling matrix:
    //                  C * Σ(W_i * [f_p_i]×)   where [·]× = skew()
    //   T_i          = tangent basis (3x2) spanning the plane orthogonal to the
    //                  information-weighted average contact normal n̄_i. When
    //                  the foot-state terrain normals are unavailable or
    //                  degenerate, LegOdometryProcessor stores the canonical
    //                  horizontal basis, i.e. T_i = [[1,0,0]^T, [0,1,0]^T],
    //                  which reproduces the previous fixed-2D behavior.
    //
    // Full ego-centric velocity in L frame:
    //   v_leg_L = v_kin_fused + F_omega * omega_L
    //   where  omega_L = R_BtoL * omega_B   (omega_B from SO3 spline)
    //
    // Residual (3-dimensional):
    //   r = v_spline_B  −  R_LtoB * v_leg_L  −  omega_B × (R_LtoB * POS_BinL)  −  T_i * b_v,i
    //
    // When the optimiser drives r → 0 the spline velocity matches the
    // leg-derived body velocity with proper lever-arm and omega coupling
    // from the jointly-optimised SO3 spline, with the 2D bias b_v,i living
    // in the (contact-plane / horizontal) tangent plane of the terrain.
    template<int Order>
    struct LegVelFactor {
    public:
        using SplineMetaType = ns_ctraj::SplineMeta<Configor::Prior::SplineOrder>;

    private:
        SplineMetaType so3Meta_;
        SplineMetaType velMeta_;
        LegFrame::Ptr frame;
        Sophus::SO3d& SO3_RefToW;

        double weight;
        double so3DtInv, velDtInv;

        // Constant per-factor offsets. SO3/VEL knot offsets depend on the
        // (time-shifted) spline index and are computed in operator() below.
        std::size_t BV_OFFSET, EXTRI_SO3_OFFSET, EXTRI_POS_OFFSET, EXTRI_TO_OFFSET;
    public:
        LegVelFactor(const SplineMetaType &so3Meta, const SplineMetaType &velMeta, LegFrame::Ptr legFrame, Sophus::SO3d& SO3_RefToW, double weight)
                : so3Meta_(so3Meta), velMeta_(velMeta),
                  frame(std::move(legFrame)), SO3_RefToW(SO3_RefToW), weight(weight),
                  so3DtInv(1.0 / so3Meta.segments.front().dt), velDtInv(1.0 / velMeta.segments.front().dt){
            BV_OFFSET = so3Meta.NumParameters() +  velMeta.NumParameters();
            EXTRI_SO3_OFFSET = BV_OFFSET + 1;
            EXTRI_POS_OFFSET = EXTRI_SO3_OFFSET + 1;
            EXTRI_TO_OFFSET  = EXTRI_POS_OFFSET + 1;
        }

        static auto Create(const SplineMetaType &so3Meta, const SplineMetaType &velMeta, LegFrame::Ptr legFrame, Sophus::SO3d& SO3_RefToW, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<LegVelFactor>(
                    new LegVelFactor(so3Meta, velMeta, legFrame, SO3_RefToW, weight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(LegVelFactor).hash_code();
        }

    public:
        // See garlileo::RadarFactor::QueryInsideSplineMeta for rationale.
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
            Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);

            // ---- Time-shifted spline indexing (iKalibr-style) ----
            const T TO = parBlocks[EXTRI_TO_OFFSET][0];
            const T t_eval = T(frame->GetTimestamp()) + TO;

            if (!QueryInsideSplineMeta(t_eval)) {
                residuals.setZero();
                return true;
            }

            std::pair<std::size_t, T> so3IU, velIU;
            so3Meta_.template ComputeSplineIndex(t_eval, so3IU.first, so3IU.second);
            velMeta_.template ComputeSplineIndex(t_eval, velIU.first, velIU.second);

            const std::size_t SO3_OFFSET = so3IU.first;
            const std::size_t VEL_OFFSET = so3Meta_.NumParameters() + velIU.first;

            // ---- Spline evaluation (jet-aware: time argument may be a Jet) ----
            Eigen::Vector3<T> LIN_VEL_B;
            ns_ctraj::CeresSplineHelperJet<T, Order>::template Evaluate<3, 0>(
                    parBlocks + VEL_OFFSET, velIU.second, velDtInv, &LIN_VEL_B
            );

            Sophus::SO3<T> SO3_CurToRef;
            Sophus::SO3Tangent<T> ANG_VEL_B;
            ns_ctraj::CeresSplineHelperJet<T, Order>::template EvaluateLie<Sophus::SO3>(
                    parBlocks + SO3_OFFSET, so3IU.second, so3DtInv, &SO3_CurToRef, &ANG_VEL_B
            );

            // ---- Parameter blocks ----
            Eigen::Map<const Eigen::Vector2<T>> BV(parBlocks[BV_OFFSET]);
            const Sophus::SO3<T> SO3_BtoL(Eigen::Map<const Eigen::Quaternion<T>>(parBlocks[EXTRI_SO3_OFFSET]));
            const Eigen::Vector3<T> POS_BinL = Eigen::Map<const Eigen::Vector3<T>>(parBlocks[EXTRI_POS_OFFSET]);

            const Eigen::Matrix<T, 3, 3> R_LtoB = SO3_BtoL.matrix().transpose();
            const Eigen::Matrix<T, 3, 3> R_BtoL = SO3_BtoL.matrix();

            // ---- Pre-computed kinematic data from LegFrame (in L frame) ----
            const Eigen::Vector3<T> v_kin    = frame->GetVel().cast<T>();
            const Eigen::Matrix<T, 3, 3> F_omega = frame->GetOmegaCoupling().template cast<T>();

            // ---- Omega coupling: v_leg_L = v_kin + F_omega * omega_L ----
            const Eigen::Vector3<T> omega_L = R_BtoL * ANG_VEL_B;
            const Eigen::Vector3<T> v_leg_L = v_kin + F_omega * omega_L;

            // ---- Transform to B frame ----
            const Eigen::Vector3<T> v_leg_B = R_LtoB * v_leg_L;

            // ---- Lever arm: velocity at B origin due to rotation about L origin ----
            const Eigen::Vector3<T> p_LtoB_B = R_LtoB * POS_BinL;
            const Eigen::Vector3<T> lever_arm = Sophus::SO3<T>::hat(ANG_VEL_B) * p_LtoB_B;

            // ---- Bias embedding: T_i * b_v,i  (tangent plane of n̄_i) ----
            // LegOdometryProcessor guarantees T_bias is a 3x2 real matrix; in
            // the fallback case it equals [[1,0,0]^T, [0,1,0]^T], so the
            // residual collapses to the legacy [b_vx, b_vy, 0]^T form.
            const Eigen::Matrix<T, 3, 2> T_bias = frame->GetBiasTangent().template cast<T>();
            const Eigen::Vector3<T> bias_emb = T_bias * BV;

            // ---- Residual: v_spline_B - v_leg_B - lever_arm - T_i * b_v,i ----
            residuals = LIN_VEL_B - v_leg_B - lever_arm - bias_emb;

            residuals = T(weight) * residuals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif
