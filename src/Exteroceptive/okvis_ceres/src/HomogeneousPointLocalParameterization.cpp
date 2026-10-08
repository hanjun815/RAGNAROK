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
 * @file HomogeneousPointLocalParameterization.cpp
 * @brief Source file for the HomogeneousPointLocalParameterization class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

bool HomogeneousPointManifold::plus(
    const double* x, const double* delta, double* x_plus_delta)
{
  Eigen::Map<const Eigen::Vector3d> delta_(delta);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<Eigen::Vector4d> x_plus_delta_(x_plus_delta);

  // Euclidean style
  x_plus_delta_ = x_ + Eigen::Vector4d(delta_[0], delta_[1], delta_[2], 0);

  return true;
}
bool HomogeneousPointManifold::Plus(
    const double* x, const double* delta, double* x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

bool HomogeneousPointManifold::plusJacobian(const double* /*x*/, double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > Jp(jacobian);

  // Euclidean-style
  Jp.setZero();
  Jp.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

  return true;
}
bool HomogeneousPointManifold::PlusJacobian(const double* x, double* jacobian) const {
  return plusJacobian(x, jacobian);
}

bool HomogeneousPointManifold::minus(
    const double* y, const double* x, double* y_minus_x)
{

  Eigen::Map<Eigen::Vector3d> delta_(y_minus_x);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<const Eigen::Vector4d> x_plus_delta_(y);

  // Euclidean style
  delta_ = (x_plus_delta_ - x_).head<3>();

  return true;
}
bool HomogeneousPointManifold::Minus(
    const double* y, const double* x, double* y_minus_x) const {
  return minus(y, x, y_minus_x);
}


bool HomogeneousPointManifold::minusJacobian(const double* /*x*/, double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > Jp(jacobian);

  // Euclidean-style
  Jp.setZero();
  Jp.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

  return true;
}
bool HomogeneousPointManifold::MinusJacobian(const double* x, double* jacobian) const {
  return minusJacobian(x, jacobian);
}

}  // namespace ceres
}  // namespace okvis
