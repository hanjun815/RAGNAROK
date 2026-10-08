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
 * @file HomogeneousPointError.cpp
 * @brief Source file for the HomogeneousPointError class.
 * @author Stefan Leutenegger
 */

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <okvis/ceres/HomogeneousPointError.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix.
HomogeneousPointError::HomogeneousPointError(
    const Eigen::Vector4d & measurement, const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
HomogeneousPointError::HomogeneousPointError(
    const Eigen::Vector4d & measurement, double variance) {
  setMeasurement(measurement);
  setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
}

// Construct with measurement and variance.
void HomogeneousPointError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  _squareRootInformation = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool HomogeneousPointError::Evaluate(double const* const * parameters,
                                     double* residuals,
                                     double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool HomogeneousPointError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  Eigen::Vector4d hp(parameters[0][0], parameters[0][1], parameters[0][2],
                     parameters[0][3]);
  // delta
  Eigen::Vector3d error;
  HomogeneousPointManifold::minus(&parameters[0][0], &measurement_[0], &error[0]);

  // weigh it
  Eigen::Map<Eigen::Vector3d> weighted_error(residuals);
  weighted_error = _squareRootInformation * error;

  // compute Jacobian...
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_lift;
      HomogeneousPointManifold::minusJacobian(parameters[0], J_lift.data());
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_plus;
      HomogeneousPointManifold::plusJacobian(parameters[0], J_plus.data());

      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > J0(
          jacobians[0]);
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J0_minimal = J_lift * J_plus;
      J0_minimal = (_squareRootInformation * J0_minimal).eval();

      // hallucinate Jacobian w.r.t. state
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > J0_minimal_mapped(
              jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis

