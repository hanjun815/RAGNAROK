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
 * @file PoseError.cpp
 * @brief Source file for the PoseError class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix.
PoseError::PoseError(const okvis::kinematics::Transformation & measurement,
                     const Eigen::Matrix<double, 6, 6> & information) {
  setMeasurement(measurement);
  setInformation(information);
}

PoseError::PoseError(const okvis::kinematics::Transformation & measurement,
          const Eigen::Matrix<double,6,1> & informationDiagonal) {
  setMeasurement(measurement);
  information_ = informationDiagonal.asDiagonal();
  squareRootInformation_ = informationDiagonal.array().sqrt().matrix().asDiagonal();
}

// Construct with measurement and variance.
PoseError::PoseError(const okvis::kinematics::Transformation & measurement,
                     double translationVariance, double rotationVariance) {
  setMeasurement(measurement);

  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / translationVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / rotationVariance;

  setInformation(information);
}

// Set the information.
void PoseError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool PoseError::Evaluate(double const* const * parameters, double* residuals,
                         double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool PoseError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                             double* residuals,
                                             double** jacobians,
                                             double** jacobiansMinimal) const {

  // compute error
  okvis::kinematics::Transformation T_WS(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]).normalized());
  // delta pose
  okvis::kinematics::Transformation dp = measurement_ * T_WS.inverse();
  // get the error
  Eigen::Matrix<double, 6, 1> error;
  const Eigen::Vector3d dtheta = 2 * dp.q().coeffs().head<3>();
  error.head<3>() = measurement_.r() - T_WS.r();
  error.tail<3>() = dtheta;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian...
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J0(
          jacobians[0]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setIdentity();
      J0_minimal *= -1.0;
      J0_minimal.block<3, 3>(3, 3) = -okvis::kinematics::plus(dp.q())
          .topLeftCorner<3, 3>();
      J0_minimal = (squareRootInformation_ * J0_minimal).eval();

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[0], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J0_minimal_mapped(
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
