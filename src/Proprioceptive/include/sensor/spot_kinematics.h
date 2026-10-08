// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#ifndef SPOT_KINEMATICS_H
#define SPOT_KINEMATICS_H

#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace garlileo {

// Spot robot 3-DOF leg kinematics (hip-x, hip-y, knee-y).
// Leg index convention:  0=FL, 1=FR, 2=RL, 3=RR.
// All quantities expressed in the robot kinematic body frame (L frame).
class SpotKinematics {
public:
    static constexpr int kNumLegs = 4;
    static constexpr int kJointsPerLeg = 3;

    // Forward kinematics: foot position in L frame given joint angles q = [q0, q1, q2].
    static Eigen::Vector3d footPosition(int leg_idx, const Eigen::Vector3d &q) {
        const bool left = (leg_idx == 0 || leg_idx == 2);
        const bool rear = (leg_idx >= 2);
        const double y_sign = left ? 1.0 : -1.0;
        const double x_base = rear ? kHipXRear : 0.0;

        const Eigen::Matrix3d A12 = rodrigues(Eigen::Vector3d(q(0), 0.0, 0.0));
        const Eigen::Matrix3d A23 = rodrigues(Eigen::Vector3d(0.0, q(1), 0.0));
        const Eigen::Matrix3d A34 = rodrigues(Eigen::Vector3d(0.0, q(2), 0.0));

        const Eigen::Vector3d t1(x_base, y_sign * kHipYOffset, 0.0);
        const Eigen::Vector3d t2(0.0, y_sign * kUpperLegYOffset, 0.0);
        const Eigen::Vector3d t3(kUpperLegXOffset, 0.0, -kUpperLegLength);
        const Eigen::Vector3d t4(0.0, 0.0, -kLowerLegLength);

        return A12 * A23 * A34 * t4 + A12 * A23 * t3 + A12 * t2 + t1;
    }

    // Analytical Jacobian dp/dq (3x3) at joint angles q.
    // Column k = ∂p/∂q_k.
    static Eigen::Matrix3d footJacobian(int leg_idx, const Eigen::Vector3d &q) {
        const bool left = (leg_idx == 0 || leg_idx == 2);
        const double y_sign = left ? 1.0 : -1.0;

        const Eigen::Matrix3d A12 = rodrigues(Eigen::Vector3d(q(0), 0.0, 0.0));
        const Eigen::Matrix3d A23 = rodrigues(Eigen::Vector3d(0.0, q(1), 0.0));
        const Eigen::Matrix3d A34 = rodrigues(Eigen::Vector3d(0.0, q(2), 0.0));

        const Eigen::Vector3d t2(0.0, y_sign * kUpperLegYOffset, 0.0);
        const Eigen::Vector3d t3(kUpperLegXOffset, 0.0, -kUpperLegLength);
        const Eigen::Vector3d t4(0.0, 0.0, -kLowerLegLength);

        // dA/dθ = hat(axis) * A  for single-axis rotation matrices
        static const Eigen::Matrix3d hat_ex = hat(Eigen::Vector3d::UnitX());
        static const Eigen::Matrix3d hat_ey = hat(Eigen::Vector3d::UnitY());

        const Eigen::Vector3d inner2 = A34 * t4 + t3;
        const Eigen::Vector3d inner1 = A23 * inner2 + t2;

        Eigen::Matrix3d J;
        J.col(0) = hat_ex * A12 * inner1;
        J.col(1) = A12 * hat_ey * A23 * inner2;
        J.col(2) = A12 * A23 * hat_ey * A34 * t4;
        return J;
    }

    // Foot-frame orientation in L frame. The returned rotation maps vectors
    // from the distal foot frame to the robot kinematic body frame.
    static Eigen::Matrix3d footRotation(int /*leg_idx*/, const Eigen::Vector3d &q) {
        const Eigen::Matrix3d A12 = rodrigues(Eigen::Vector3d(q(0), 0.0, 0.0));
        const Eigen::Matrix3d A23 = rodrigues(Eigen::Vector3d(0.0, q(1), 0.0));
        const Eigen::Matrix3d A34 = rodrigues(Eigen::Vector3d(0.0, q(2), 0.0));
        return A12 * A23 * A34;
    }

    // Angular Jacobian Jw where omega_foot_L = Jw(q) * qdot.
    static Eigen::Matrix3d footAngularJacobian(int /*leg_idx*/, const Eigen::Vector3d &q) {
        const Eigen::Matrix3d A12 = rodrigues(Eigen::Vector3d(q(0), 0.0, 0.0));
        const Eigen::Matrix3d A23 = rodrigues(Eigen::Vector3d(0.0, q(1), 0.0));

        Eigen::Matrix3d Jw;
        Jw.col(0) = Eigen::Vector3d::UnitX();
        Jw.col(1) = A12 * Eigen::Vector3d::UnitY();
        Jw.col(2) = A12 * A23 * Eigen::Vector3d::UnitY();
        return Jw;
    }

private:
    // Spot geometry constants (metres)
    static constexpr double kHipXRear         = -0.5957;
    static constexpr double kHipYOffset       =  0.055;
    static constexpr double kUpperLegYOffset  =  0.110945;
    static constexpr double kUpperLegXOffset  =  0.025;
    static constexpr double kUpperLegLength   =  0.3205;
    static constexpr double kLowerLegLength   =  0.34;

    static Eigen::Matrix3d hat(const Eigen::Vector3d &w) {
        Eigen::Matrix3d m;
        m <<  0.0, -w.z(),  w.y(),
              w.z(),  0.0, -w.x(),
             -w.y(),  w.x(),  0.0;
        return m;
    }

    static Eigen::Matrix3d rodrigues(const Eigen::Vector3d &w) {
        const double theta = w.norm();
        const Eigen::Matrix3d W = hat(w);
        if (theta < 1.0e-12) {
            return Eigen::Matrix3d::Identity() + W;
        }
        return Eigen::Matrix3d::Identity()
               + W * std::sin(theta) / theta
               + W * W * (1.0 - std::cos(theta)) / (theta * theta);
    }
};

}  // namespace garlileo

#endif  // SPOT_KINEMATICS_H
