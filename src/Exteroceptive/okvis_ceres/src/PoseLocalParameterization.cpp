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

/**
 * @file PoseLocalParameterization.cpp
 * @brief Source file for the PoseLocalParameterization class.
 * @author Stefan Leutenegger
 */

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

bool PoseManifold::plus(
    const double* x, const double* delta, double* x_plus_delta)
{
  Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);

  // call oplus operator in okvis::kinematis
  const Eigen::Vector4d q =
      (kinematics::deltaQ(delta_.tail<3>())
       *Eigen::Quaterniond(x[6], x[3], x[4], x[5]).normalized()).normalized().coeffs();

  // copy back
  const Eigen::Vector3d r = Eigen::Vector3d(x[0], x[1], x[2]) + delta_.head<3>();
  x_plus_delta[0] = r[0];
  x_plus_delta[1] = r[1];
  x_plus_delta[2] = r[2];
  x_plus_delta[3] = q[0];
  x_plus_delta[4] = q[1];
  x_plus_delta[5] = q[2];
  x_plus_delta[6] = q[3];

  return true;
}
bool PoseManifold::Plus(
    const double* x, const double* delta, double* x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

bool PoseManifold::plusJacobian(const double* x, double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Jp.setZero();
  Jp.topLeftCorner<3, 3>().setIdentity();
  Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
  S(0, 0) = 0.5;
  S(1, 1) = 0.5;
  S(2, 2) = 0.5;
  Jp.bottomRightCorner<4, 3>() =
      okvis::kinematics::oplus(Eigen::Quaterniond(x[6], x[3], x[4], x[5])) * S;
  return true;
}
bool PoseManifold::PlusJacobian(const double* x, double* jacobian) const {
  return plusJacobian(x, jacobian);
}

bool PoseManifold::minus(
    const double* y, const double* x, double* y_minus_x)
{
  y_minus_x[0] = y[0] - x[0];
  y_minus_x[1] = y[1] - x[1];
  y_minus_x[2] = y[2] - x[2];
  const Eigen::Quaterniond q_plus_delta_(y[6], y[3], y[4], y[5]);
  const Eigen::Quaterniond q_(x[6], x[3], x[4], x[5]);
  Eigen::Map<Eigen::Vector3d> delta_q_(&y_minus_x[3]);
  delta_q_ = 2 * (q_plus_delta_ * q_.inverse()).coeffs().template head<3>();
  return true;
}
bool PoseManifold::Minus(
    const double* y, const double* x, double* y_minus_x) const {
  return minus(y, x, y_minus_x);
}

bool PoseManifold::minusJacobian(const double* x, double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);
  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();
  Eigen::Matrix4d Qplus = okvis::kinematics::oplus(q_inv);
  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;
  J_lift.bottomRightCorner<3, 4>() = Jq_pinv * Qplus;

  return true;
}

bool PoseManifold::MinusJacobian(const double* x, double* jacobian) const {
  return minusJacobian(x, jacobian);
}

bool PoseManifold::verifyJacobianNumDiff(const double* x,
                                         double* jacobian,
                                         double* jacobianNumDiff) const {
  PlusJacobian(x, jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jpn(
      jacobianNumDiff);
  double dx = 1e-9;
  Eigen::Matrix<double, 7, 1> xp;
  Eigen::Matrix<double, 7, 1> xm;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta;
    delta.setZero();
    delta[i] = dx;
    Plus(x, delta.data(), xp.data());
    delta[i] = -dx;
    Plus(x, delta.data(), xm.data());
    Jpn.col(i) = (xp - xm) / (2 * dx);
  }
  if (Jp.isApprox(Jpn,1.0e-6)) {
    return true;
  } else {
    std::cout << "Manifold Jacobian check failed" << std::endl;
    std::cout << Jp << std::endl;
    std::cout << "vs" << std::endl;
    std::cout << Jpn << std::endl;
    return false;
  }
}

bool PoseManifold4d::plus(
        const double *x, const double *delta, double *x_plus_delta) {
  Eigen::Matrix<double, 6, 1> delta_;
  delta_.setZero();
  delta_[0] = delta[0];
  delta_[1] = delta[1];
  delta_[2] = delta[2];
  delta_[5] = delta[3];

  // transform to sm::kinematics framework
  okvis::kinematics::Transformation T(
          Eigen::Vector3d(x[0], x[1], x[2]),
          Eigen::Quaterniond(x[6], x[3], x[4], x[5]));

  // call oplus operator in sm::kinematis
  T.oplus(delta_);

  // copy back
  x_plus_delta[0] = T.r()[0];
  x_plus_delta[1] = T.r()[1];
  x_plus_delta[2] = T.r()[2];
  x_plus_delta[3] = T.q().coeffs()[0];
  x_plus_delta[4] = T.q().coeffs()[1];
  x_plus_delta[5] = T.q().coeffs()[2];
  x_plus_delta[6] = T.q().coeffs()[3];

  OKVIS_ASSERT_TRUE_DBG(std::runtime_error, T.q().norm() - 1.0 < 1e-15, "damn.");

  return true;
}

bool PoseManifold4d::Plus(
        const double *x, const double *delta, double *x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

bool PoseManifold4d::plusJacobian(const double *x, double *jacobian) {

  Eigen::Map<Eigen::Matrix<double, 7, 4, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> Jp_full;
  okvis::kinematics::Transformation T(
          Eigen::Vector3d(x[0], x[1], x[2]),
          Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
  T.oplusJacobian(Jp_full);
  Jp.topLeftCorner<7,3>()=Jp_full.topLeftCorner<7,3>();
  Jp.bottomRightCorner<7,1>()=Jp_full.bottomRightCorner<7,1>();
  return true;
}

bool PoseManifold4d::PlusJacobian(const double *x, double *jacobian) const {
  return plusJacobian(x, jacobian);
}

bool PoseManifold4d::minus(
        const double *y, const double *x, double *y_minus_x) {

  y_minus_x[0] = y[0] - x[0];
  y_minus_x[1] = y[1] - x[1];
  y_minus_x[2] = y[2] - x[2];
  const Eigen::Quaterniond q_plus_delta_(y[6], y[3],
                                         y[4], y[5]);
  const Eigen::Quaterniond q_(x[6], x[3], x[4], x[5]);
  y_minus_x[3] = 2 * (q_plus_delta_ * q_.inverse()).coeffs()[2];

  return true;

}

bool PoseManifold4d::Minus(
        const double *y, const double *x, double *y_minus_x) const {
  return minus(y, x, y_minus_x);
}

bool PoseManifold4d::minusJacobian(const double *x, double *jacobian) {

  Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor> > J_lift(jacobian);
  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();
  Eigen::Matrix4d Qplus = okvis::kinematics::oplus(q_inv);
  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;
  J_lift.bottomRightCorner<1, 4>() = (Jq_pinv * Qplus).bottomRightCorner<1, 4>();

  return true;
}

bool PoseManifold4d::MinusJacobian(const double *x, double *jacobian) const {
  return minusJacobian(x, jacobian);
}
}  // namespace ceres
}  // namespace okvis
