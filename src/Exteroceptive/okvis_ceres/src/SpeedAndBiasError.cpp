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
 * @file SpeedAndBiasError.cpp
 * @brief Source file for the SpeedAndBiasError class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <cmath>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix
SpeedAndBiasError::SpeedAndBiasError(const okvis::SpeedAndBias & measurement,
                                     const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
SpeedAndBiasError::SpeedAndBiasError(const okvis::SpeedAndBias& measurement,
                                     double speedVariance,
                                     double gyrBiasVariance,
                                     double accBiasVariance) {
  setMeasurement(measurement);

  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / speedVariance;
  information.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1.0
      / gyrBiasVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / accBiasVariance;

  setInformation(information);
}

// Set the information.
void SpeedAndBiasError::setInformation(const information_t & information) {
  OKVIS_ASSERT_TRUE(Exception, information.allFinite(), "Prior information must be finite")
  information_ = information;
  if (information_.isDiagonal(0.0)) {
    OKVIS_ASSERT_TRUE(Exception, (information_.diagonal().array() >= 0.0).all(),
                      "Diagonal prior information must be nonnegative")
    if ((information_.diagonal().array() == 0.0).any()) {
      // A selective bias prior has exactly zero information on velocity and on
      // estimated biases. Singular inverse/LLT cannot represent those zero rows.
      covariance_.setZero();  // Moore-Penrose inverse on unconstrained components.
      squareRootInformation_.setZero();
      for (int i = 0; i < 9; ++i) {
        const double weight = information_(i, i);
        if (weight > 0.0) {
          covariance_(i, i) = 1.0 / weight;
          squareRootInformation_(i, i) = std::sqrt(weight);
        }
      }
      return;
    }
  }
  // Preserve the existing factorization for positive-definite information.
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool SpeedAndBiasError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool SpeedAndBiasError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  Eigen::Map<const okvis::SpeedAndBias> estimate(parameters[0]);
  okvis::SpeedAndBias error = measurement_ - estimate;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 9, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = -squareRootInformation_ * Eigen::Matrix<double, 9, 9>::Identity();
    }
  }
  if (jacobiansMinimal != nullptr) {
    if (jacobiansMinimal[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J0min(
          jacobiansMinimal[0]);
      J0min = -squareRootInformation_ * Eigen::Matrix<double, 9, 9>::Identity();
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
