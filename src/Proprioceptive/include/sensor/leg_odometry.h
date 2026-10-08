// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#ifndef LEG_ODOMETRY_H
#define LEG_ODOMETRY_H

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>
#include "sensor/leg.h"
#include "sensor/spot_kinematics.h"

namespace garlileo {

// Processes raw JointState + foot contact to produce a fused LegFrame.
//
// Each call to process() computes, for every contacting leg i, a *per-leg*
// covariance Σ_i composed of:
//   (a) base encoder noise  :  J_i Σ_{q̇} J_i^T  (+ finite-diff term if no q̇)
//   (b) stance-age inflation:  I · stance_age_scale / max(age_i, floor)
//       - age_i is the time since leg i transitioned into contact, so the
//         covariance is inflated right after landing and decays the longer
//         the leg has been on the ground.
//   (c) per-leg effort discontinuity:  I · force_scale · ‖(τ_i − τ_i^prev)/dt‖
//       - each leg contributes only its OWN effort discontinuity (no averaging
//         over legs).
//   (d) per-leg inter-leg disagreement:  scale · mean_{j≠i} (v_i − v_j)(v_i − v_j)^T
//       - pairwise disagreement of leg i against every other contacting leg
//         (no average-velocity reference).
//
// Information-weighted fusion of Σ_i gives v_kin_fused, Σ_fused, and the
// omega-coupling matrix  F_ω = Σ_fused · Σ (W_i [f_p_i]×).  The spline angular
// velocity ω is applied inside LegVelFactor, not here.
class LegOdometryProcessor {
public:
    using Ptr = std::shared_ptr<LegOdometryProcessor>;

    struct Config {
        double encoder_pos_sigma    = 0.001;
        double encoder_vel_sigma    = 0.005;
        // Scale on per-leg pairwise inter-leg disagreement  (see (d) above).
        double inter_leg_cov_scale  = 1.0;
        // Scale on per-leg effort discontinuity  (see (c) above).
        double force_disc_scale     = 0.001;
        // Numerator of the per-leg stance-age inflation  (see (b) above).
        // Larger → more uncertainty right after a leg lands.
        double stance_age_scale     = 0.01;
        // Lower bound on stance age [s] when computing (b) to avoid division
        // by ~0 in the first few frames after touch-down.
        double stance_age_floor_sec = 0.02;
        double cov_floor            = 1.0e-4;
        bool rolling_contact_enable = false;
        double rolling_contact_foot_radius = 0.01;
        // Use the z-axis of the kinematic L frame as the contact normal when a
        // recording carries no terrain normals (GaRLILEO and Co-RaL foot messages).
        bool rolling_contact_body_normal = false;
        // Rotation from body B frame contact normals into the kinematic L frame.
        Eigen::Matrix3d SO3_BtoL = Eigen::Matrix3d::Identity();
    };

    explicit LegOdometryProcessor(const Config &cfg) : cfg_(cfg) {}

    static Ptr Create(const Config &cfg) {
        return std::make_shared<LegOdometryProcessor>(cfg);
    }

    // Default "no contact-normal info" arguments for the overload below.
    // Using const references to static locals keeps the API minimal for
    // callers that do not (yet) forward contact normals.
    static const std::array<Eigen::Vector3d, 4> &kNoContactNormals() {
        static const std::array<Eigen::Vector3d, 4> v{{
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
        return v;
    }
    static const std::array<bool, 4> &kNoNormalValid() {
        static const std::array<bool, 4> v{{false, false, false, false}};
        return v;
    }

    // Process one synchronized (JointState, FootContact) pair.
    // Returns a LegFrame if at least one contact leg is available, otherwise std::nullopt.
    //
    // joint_pos:       12-element joint position vector  (FL0..2, FR0..2, RL0..2, RR0..2)
    // joint_vel:       12-element joint velocity vector  (may be empty/invalid → finite-diff fallback)
    // joint_eff:       12-element joint effort vector    (may be empty)
    // contact:         4-element boolean (FL, FR, RL, RR)
    // contact_normals: per-leg ground contact normals (assumed expressed in
    //                  the body B frame; see FootFrame docstring). They are
    //                  transformed into the kinematic L frame for rolling
    //                  contact and used directly for the bias tangent basis.
    //                  Only entries with the matching normal_valid flag set
    //                  to true are considered.
    // normal_valid:    validity flag per leg.
    // timestamp:       message timestamp (seconds, after epoch subtraction)
    std::optional<ns_leg::LegFrame::Ptr> process(
            const std::vector<double> &joint_pos,
            const std::vector<double> &joint_vel,
            const std::vector<double> &joint_eff,
            const Eigen::Vector4d &contact,
            double timestamp,
            const std::array<Eigen::Vector3d, 4> &contact_normals = kNoContactNormals(),
            const std::array<bool, 4> &normal_valid = kNoNormalValid())
    {
        const double dt = timestamp - prev_time_;
        const bool first = !initialized_ || dt <= 0.0 || dt > 0.2;

        // Per-leg estimates
        struct PerLeg {
            Eigen::Vector3d foot_pos  = Eigen::Vector3d::Zero();
            Eigen::Vector3d v_kin     = Eigen::Vector3d::Zero();
            Eigen::Matrix3d cov       = Eigen::Matrix3d::Identity() * 1.0e-2;
            bool in_contact           = false;
        };
        std::array<PerLeg, 4> legs;

        const Eigen::Matrix3d sigma_qdot = Eigen::Matrix3d::Identity() * (cfg_.encoder_vel_sigma * cfg_.encoder_vel_sigma);
        const Eigen::Matrix3d sigma_q    = Eigen::Matrix3d::Identity() * (cfg_.encoder_pos_sigma * cfg_.encoder_pos_sigma);
        const Eigen::Matrix3d floor_cov  = Eigen::Matrix3d::Identity() * cfg_.cov_floor;

        const bool has_vel = (joint_vel.size() >= 12);
        const bool has_eff = (joint_eff.size() >= 12);

        for (int i = 0; i < 4; ++i) {
            const int off = i * 3;
            const Eigen::Vector3d q(joint_pos[off], joint_pos[off + 1], joint_pos[off + 2]);
            // contact(i) is pre-converted boolean (1.0=contact, 0.0=no contact).
            // Guard against raw enum leak: only exactly 1.0 means CONTACT_MADE.
            const bool in_contact = (std::abs(contact(i) - 1.0) < 0.1);
            const Eigen::Vector3d fp = SpotKinematics::footPosition(i, q);
            const Eigen::Matrix3d J  = SpotKinematics::footJacobian(i, q);

            Eigen::Vector3d contact_pos = fp;
            Eigen::Matrix3d J_contact = J;
            if (in_contact && cfg_.rolling_contact_enable
                && (normal_valid[i] || cfg_.rolling_contact_body_normal)
                && cfg_.rolling_contact_foot_radius > 0.0) {
                Eigen::Vector3d n_L = normal_valid[i] ? Eigen::Vector3d(cfg_.SO3_BtoL * contact_normals[i])
                                                      : Eigen::Vector3d::UnitZ();
                const double nn = n_L.norm();
                if (n_L.allFinite() && nn > 1.0e-6) {
                    n_L /= nn;

                    // Spherical-foot contact: express the ground normal in the
                    // foot frame, choose the support point on the sphere, then
                    // map it back to L. For a perfect sphere this equals
                    // -r*n_L, but keeping the frame transform makes the
                    // geometry explicit and easy to extend to non-spherical feet.
                    const Eigen::Matrix3d R_LF = SpotKinematics::footRotation(i, q);
                    const Eigen::Vector3d n_F = R_LF.transpose() * n_L;
                    const Eigen::Vector3d rho_L =
                        R_LF * (-cfg_.rolling_contact_foot_radius * n_F);
                    contact_pos = fp + rho_L;

                    // v_contact = Jp*qdot + (Jw*qdot) x rho
                    //           = (Jp - hat(rho)*Jw) qdot.
                    const Eigen::Matrix3d Jw = SpotKinematics::footAngularJacobian(i, q);
                    J_contact = J - hat3(rho_L) * Jw;
                }
            }

            Eigen::Vector3d v_kin;
            if (has_vel && isFinite3(joint_vel, off)) {
                const Eigen::Vector3d qdot(joint_vel[off], joint_vel[off + 1], joint_vel[off + 2]);
                v_kin = -J_contact * qdot;
            } else if (!first) {
                v_kin = -(contact_pos - prev_foot_pos_[i]) / dt;
            } else {
                v_kin.setZero();
            }

            Eigen::Matrix3d cov = floor_cov;
            cov += J_contact * sigma_qdot * J_contact.transpose();
            if (!has_vel) {
                const double dt_safe = std::max(1.0e-3, dt);
                cov += (1.0 / (dt_safe * dt_safe)) * (J_contact * sigma_q * J_contact.transpose());
            }

            legs[i].foot_pos   = contact_pos;
            legs[i].v_kin      = v_kin;
            legs[i].cov        = cov;
            legs[i].in_contact = in_contact;
        }

        if (first) {
            // Initialise per-leg contact start times for legs already on the
            // ground at the first frame so stance_age is well-defined.
            for (int i = 0; i < 4; ++i) {
                if (legs[i].in_contact) contact_start_time_[i] = timestamp;
            }
            storePrev(legs, contact, joint_eff, has_eff, timestamp);
            return std::nullopt;
        }

        // Per-leg stance-age bookkeeping: reset on the 0 → 1 transition.
        for (int i = 0; i < 4; ++i) {
            const bool cur_in  = legs[i].in_contact;
            const bool prev_in = (std::abs(prev_contact_(i) - 1.0) < 0.1);
            if (cur_in && !prev_in) {
                contact_start_time_[i] = timestamp;
            }
        }

        // Collect contact legs
        std::vector<int> contact_idx;
        contact_idx.reserve(4);
        for (int i = 0; i < 4; ++i) {
            if (legs[i].in_contact) contact_idx.push_back(i);
        }
        if (contact_idx.empty()) {
            storePrev(legs, contact, joint_eff, has_eff, timestamp);
            return std::nullopt;
        }

        // ---- Per-leg covariance augmentation ----
        Eigen::Matrix3d sum_info      = Eigen::Matrix3d::Zero();
        Eigen::Vector3d sum_info_v    = Eigen::Vector3d::Zero();
        Eigen::Matrix3d sum_info_skew = Eigen::Matrix3d::Zero();
        Eigen::Vector3d mean_v        = Eigen::Vector3d::Zero();
        for (int i : contact_idx) mean_v += legs[i].v_kin;
        mean_v /= static_cast<double>(contact_idx.size());
        int used = 0;

        // Weighted accumulation of per-leg contact normals (valid legs only).
        // Weight for leg i is 1 / trace(Σ_i) so that legs with lower velocity
        // uncertainty dominate the averaged tangent plane, mirroring the
        // existing information-weighted velocity fusion.
        Eigen::Vector3d normal_acc = Eigen::Vector3d::Zero();
        double          normal_w_sum = 0.0;
        int             normal_count = 0;

        for (int i : contact_idx) {
            Eigen::Matrix3d cov_i = legs[i].cov;  // base encoder noise

            // (b) Per-leg stance-age inflation.  Uses THIS leg's own age
            //     since touch-down, not a global contact-time.
            const double stance_age = std::max(
                cfg_.stance_age_floor_sec,
                timestamp - contact_start_time_[i]);
            cov_i += Eigen::Matrix3d::Identity() *
                     (cfg_.stance_age_scale / stance_age);

            // (c) Per-leg effort discontinuity (no cross-leg average).
            double impact_i = 0.0;
            if (has_eff && has_prev_effort_ && isFinite3(joint_eff, i * 3)) {
                const Eigen::Vector3d cur_eff(
                    joint_eff[i * 3], joint_eff[i * 3 + 1], joint_eff[i * 3 + 2]);
                const Eigen::Vector3d d_eff =
                    (cur_eff - prev_effort_[i]) / std::max(1.0e-3, dt);
                impact_i = d_eff.norm();
            }
            cov_i += Eigen::Matrix3d::Identity() *
                     (cfg_.force_disc_scale * impact_i);

            // (d) Per-leg inter-leg disagreement using pairwise differences
            //     against every OTHER contacting leg (no mean reference).
            Eigen::Matrix3d inter_cov = Eigen::Matrix3d::Zero();
            int n_others = 0;
            for (int j : contact_idx) {
                if (j == i) continue;
                const Eigen::Vector3d diff_ij = legs[i].v_kin - legs[j].v_kin;
                inter_cov += diff_ij * diff_ij.transpose();
                ++n_others;
            }
            if (n_others > 0) {
                inter_cov /= static_cast<double>(n_others);
                cov_i += cfg_.inter_leg_cov_scale * inter_cov;
            }

            // Symmetrise + floor
            cov_i = 0.5 * (cov_i + cov_i.transpose());
            cov_i += floor_cov;

            const Eigen::Matrix3d info_i = cov_i.inverse();
            if (!info_i.allFinite()) continue;

            sum_info      += info_i;
            sum_info_v    += info_i * legs[i].v_kin;
            sum_info_skew += info_i * hat3(legs[i].foot_pos);
            ++used;

            // Accumulate a valid contact normal for this leg into the
            // weighted average (weight = 1 / trace(Σ_i)).
            if (normal_valid[i]) {
                const Eigen::Vector3d &n = contact_normals[i];
                const double nn = n.norm();
                if (n.allFinite() && nn > 1.0e-6) {
                    const double tr_cov = cov_i.trace();
                    if (std::isfinite(tr_cov) && tr_cov > 1.0e-12) {
                        const double w = 1.0 / tr_cov;
                        normal_acc += (w / nn) * n;  // normalize then weight
                        normal_w_sum += w;
                        ++normal_count;
                    }
                }
            }
        }

        if (used == 0 || std::abs(sum_info.determinant()) < 1.0e-12) {
            storePrev(legs, contact, joint_eff, has_eff, timestamp);
            return std::nullopt;
        }

        const Eigen::Matrix3d cov_fused = sum_info.inverse();
        Eigen::Vector3d v_kin_fused;
        Eigen::Matrix3d F_omega;

        if (cov_fused.allFinite()) {
            v_kin_fused = cov_fused * sum_info_v;
            F_omega     = cov_fused * sum_info_skew;
        } else {
            v_kin_fused = mean_v;
            F_omega     = Eigen::Matrix3d::Zero();
        }

        // Build the bias tangent basis T_i from the weighted average normal.
        // Fallback to the canonical horizontal basis whenever the contact
        // normal information is unavailable, degenerate, or non-finite — in
        // that case T_i * b_v reduces to [b_vx, b_vy, 0]^T, preserving the
        // legacy leg-factor bias behavior bit-exactly.
        ns_leg::LegFrame::BiasTangent T_bias;
        bool T_valid = false;
        if (normal_count > 0 && normal_w_sum > 0.0) {
            Eigen::Vector3d nbar = normal_acc / normal_w_sum;
            const double nb = nbar.norm();
            if (nbar.allFinite() && nb > 1.0e-3) {
                nbar /= nb;
                if (tangentBasisFromNormal(nbar, T_bias)) {
                    T_valid = true;
                }
            }
        }
        if (!T_valid) {
            T_bias = canonicalHorizontalBasis();
        }

        auto leg_frame = ns_leg::LegFrame::Create(timestamp, v_kin_fused, cov_fused, F_omega);
        leg_frame->SetBiasTangent(T_bias, T_valid);

        storePrev(legs, contact, joint_eff, has_eff, timestamp);

        return leg_frame;
    }

private:
    Config cfg_;
    bool initialized_ = false;
    bool has_prev_effort_ = false;
    double prev_time_ = 0.0;
    // Per-leg time at which the leg most recently transitioned into contact.
    // Used to compute each leg's own stance age (see Config::stance_age_scale).
    std::array<double, 4> contact_start_time_ = {{0.0, 0.0, 0.0, 0.0}};
    Eigen::Vector4d prev_contact_ = Eigen::Vector4d::Ones();
    std::array<Eigen::Vector3d, 4> prev_foot_pos_ = {{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
    std::array<Eigen::Vector3d, 4> prev_effort_ = {{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};

    struct PerLegSimple {
        Eigen::Vector3d foot_pos;
        bool in_contact;
    };

    template<typename Legs>
    void storePrev(const Legs &legs, const Eigen::Vector4d &contact,
                   const std::vector<double> &joint_eff, bool has_eff, double t) {
        for (int i = 0; i < 4; ++i) prev_foot_pos_[i] = legs[i].foot_pos;
        prev_contact_ = contact;
        if (has_eff) {
            for (int i = 0; i < 4; ++i) {
                const int off = i * 3;
                if (isFinite3(joint_eff, off)) {
                    prev_effort_[i] << joint_eff[off], joint_eff[off + 1], joint_eff[off + 2];
                } else {
                    prev_effort_[i].setZero();
                }
            }
            has_prev_effort_ = true;
        }
        prev_time_ = t;
        initialized_ = true;
    }

    static bool isFinite3(const std::vector<double> &v, int off) {
        return static_cast<int>(v.size()) > off + 2
               && std::isfinite(v[off]) && std::isfinite(v[off + 1]) && std::isfinite(v[off + 2]);
    }

    static Eigen::Matrix3d hat3(const Eigen::Vector3d &w) {
        Eigen::Matrix3d m;
        m <<  0.0, -w.z(),  w.y(),
              w.z(),  0.0, -w.x(),
             -w.y(),  w.x(),  0.0;
        return m;
    }

    // Canonical fallback basis: columns span the body-frame XY plane, so
    // T * [b_vx, b_vy]^T = [b_vx, b_vy, 0]^T (exactly the pre-refactor model).
    static ns_leg::LegFrame::BiasTangent canonicalHorizontalBasis() {
        ns_leg::LegFrame::BiasTangent T;
        T << 1.0, 0.0,
             0.0, 1.0,
             0.0, 0.0;
        return T;
    }

    // Build an orthonormal 3x2 tangent basis spanning the plane whose
    // normal is the unit vector n. Columns are chosen by cross-product
    // Gram-Schmidt against the world axis least parallel to n, which is
    // numerically stable for any n with |n| ~ 1.
    // Returns false iff the resulting basis fails the orthonormality
    // sanity check, in which case the caller should fall back.
    static bool tangentBasisFromNormal(const Eigen::Vector3d &n,
                                       ns_leg::LegFrame::BiasTangent &T) {
        if (!n.allFinite()) return false;
        const double nn = n.norm();
        if (!(nn > 0.9 && nn < 1.1)) return false;  // expect pre-normalized

        // Pick the world axis most orthogonal to n for the seed vector.
        Eigen::Vector3d seed;
        const double ax = std::abs(n.x());
        const double ay = std::abs(n.y());
        const double az = std::abs(n.z());
        if (ax <= ay && ax <= az) seed = Eigen::Vector3d::UnitX();
        else if (ay <= ax && ay <= az) seed = Eigen::Vector3d::UnitY();
        else seed = Eigen::Vector3d::UnitZ();

        Eigen::Vector3d e1 = seed - seed.dot(n) * n;
        const double e1n = e1.norm();
        if (!(e1n > 1.0e-6)) return false;
        e1 /= e1n;
        Eigen::Vector3d e2 = n.cross(e1);  // already unit since e1 ⟂ n, |n|=|e1|=1
        const double e2n = e2.norm();
        if (!(e2n > 1.0e-6)) return false;
        e2 /= e2n;

        T.col(0) = e1;
        T.col(1) = e2;

        // Sanity: columns orthonormal and perpendicular to n.
        const double c01 = std::abs(e1.dot(e2));
        const double cn0 = std::abs(n.dot(e1));
        const double cn1 = std::abs(n.dot(e2));
        if (!(c01 < 1.0e-5 && cn0 < 1.0e-5 && cn1 < 1.0e-5)) return false;
        if (!T.allFinite()) return false;
        return true;
    }
};

}  // namespace garlileo

#endif  // LEG_ODOMETRY_H
