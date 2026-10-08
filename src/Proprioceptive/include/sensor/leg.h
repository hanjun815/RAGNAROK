// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#ifndef LEG_H
#define LEG_H

#include "filesystem"
#include "fstream"
#include "memory"
#include "ostream"
#include "ctraj/utils/macros.hpp"
#include "ctraj/utils/eigen_utils.hpp"
#include "utility"
#include "ctraj/utils/sophus_utils.hpp"

namespace ns_leg {

struct LegFrame {
public:
    using Ptr = std::shared_ptr<LegFrame>;
    // Tangent-plane basis used by the leg-velocity factor to embed the
    // residual's 2D velocity bias b_v (see LegVelFactor). Each column is a
    // 3-vector in the same frame the residual is evaluated in (body B).
    using BiasTangent = Eigen::Matrix<double, 3, 2>;

private:
    double _timestamp;
    Eigen::Vector3d _linearLegVel;
    Eigen::Matrix3d _linearLegVelCov;
    // Omega coupling matrix F_omega (3x3, in L frame).
    // Full fused leg velocity = _linearLegVel + _omegaCoupling * omega_L
    // where omega_L = R_BtoL * omega_B (angular velocity from the SO3 spline).
    Eigen::Matrix3d _omegaCoupling;

    // Tangent basis T_i \in R^{3x2} spanning the plane orthogonal to the
    // information-weighted average contact normal nbar_i. The residual
    // embeds the 2D bias as  T_i * b_v,i.  When contact-normal data is
    // unavailable/degenerate, LegOdometryProcessor falls back to the
    // canonical horizontal basis T_i = [[1,0,0]^T, [0,1,0]^T], which
    // reproduces the previous [b_vx, b_vy, 0]^T behavior exactly.
    BiasTangent _biasTangent = (BiasTangent() << 1.0, 0.0,
                                                 0.0, 1.0,
                                                 0.0, 0.0).finished();
    // True iff _biasTangent was derived from a valid contact-normal average
    // (i.e., at least one contacting leg had a valid, non-degenerate normal).
    // Purely informational; the factor itself uses _biasTangent directly.
    bool _biasTangentValid = false;


public:
    explicit LegFrame(double timestamp = INVALID_TIME_STAMP,
                      Eigen::Vector3d linearLegVel = Eigen::Vector3d::Zero(),
                      Eigen::Matrix3d linearLegVelCov = Eigen::Matrix3d::Identity() * 1.0e-2,
                      Eigen::Matrix3d omegaCoupling = Eigen::Matrix3d::Zero());

    static LegFrame::Ptr Create(double timestamp = INVALID_TIME_STAMP,
                                const Eigen::Vector3d &linearLegVel = Eigen::Vector3d::Zero(),
                                const Eigen::Matrix3d &linearLegVelCov = Eigen::Matrix3d::Identity() * 1.0e-2,
                                const Eigen::Matrix3d &omegaCoupling = Eigen::Matrix3d::Zero());

    [[nodiscard]] double GetTimestamp() const;

    [[nodiscard]] const Eigen::Vector3d &GetVel() const;
    [[nodiscard]] const Eigen::Matrix3d &GetVelCov() const;
    [[nodiscard]] const Eigen::Matrix3d &GetOmegaCoupling() const;
    [[nodiscard]] const BiasTangent &GetBiasTangent() const;
    [[nodiscard]] bool IsBiasTangentValid() const;
    void SetVel(Eigen::Vector3d &linearLegVel);
    void SetVelCov(const Eigen::Matrix3d &linearLegVelCov);
    void SetOmegaCoupling(const Eigen::Matrix3d &omegaCoupling);
    void SetBiasTangent(const BiasTangent &biasTangent, bool valid);


    void SetTimestamp(double timestamp);

    friend std::ostream &operator<<(std::ostream &os, const LegFrame &frame);

    static bool SaveFramesToDisk(const std::string &filename,
                                 const std::vector<LegFrame::Ptr> &frames,
                                 int precision = 10);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    template <class Archive>
    void serialize(Archive &ar) {
        ar(cereal::make_nvp("timestamp", _timestamp),
           cereal::make_nvp("linear_vel", _linearLegVel),
           cereal::make_nvp("linear_vel_cov", _linearLegVelCov),
           cereal::make_nvp("omega_coupling", _omegaCoupling),
           cereal::make_nvp("bias_tangent", _biasTangent),
           cereal::make_nvp("bias_tangent_valid", _biasTangentValid));
    }
};

} 

#endif
