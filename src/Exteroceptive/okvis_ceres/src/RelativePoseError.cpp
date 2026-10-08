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
 * @file RelativePoseError.cpp
 * @brief Source file for the RelativePoseError class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix.
RelativePoseError::RelativePoseError(const Eigen::Matrix<double, 6, 6> & information,
                                     const kinematics::Transformation &T_AB) {
  T_AB_ = T_AB;
  setInformation(information);
}

// Construct with measurement and variance.
RelativePoseError::RelativePoseError(
    double translationVariance, double rotationVariance, const kinematics::Transformation & T_AB) {
  T_AB_ = T_AB;
  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / translationVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / rotationVariance;

  setInformation(information);
}

// Set the information.
void RelativePoseError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool RelativePoseError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool RelativePoseError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  okvis::kinematics::Transformation T_WA(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]).normalized());
  okvis::kinematics::Transformation T_WB(
      Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]),
      Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4],
                         parameters[1][5]).normalized());
  // delta pose
  // Compute the error vector
  const kinematics::Transformation T_AB = T_WA.inverse()*T_WB;
  Eigen::Matrix<double, 6, 1> error;
  error.head<3>() = T_AB_.r() - T_AB.r();
  // q and -q encode the same rotation. Choose the shortest-arc residual so
  // translation/rotation cross information cannot make its cost sign-dependent.
  const Eigen::Quaterniond rotationError = T_AB_.q() * T_AB.q().inverse();
  const double rotationSign = rotationError.w() < 0.0 ? -1.0 : 1.0;
  error.tail<3>() = 2.0 * rotationSign * rotationError.vec();

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian...
  const kinematics::Transformation T_AW = T_WA.inverse();
  const kinematics::Transformation T_BW = T_WB.inverse();
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J0(
          jacobians[0]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setIdentity();
      J0_minimal.block<3, 3>(0, 0) = T_AW.C();
      J0_minimal.block<3, 3>(0, 3) = -T_AW.C() * (kinematics::crossMx(T_WB.r() - T_WA.r()));
      J0_minimal.block<3, 3>(3, 3) = (kinematics::plus(T_AB_.q() * T_BW.q())
                                      * kinematics::oplus(T_WA.q())).topLeftCorner<3, 3>();
      J0_minimal.block<3, 3>(3, 3) *= rotationSign;
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
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J1(
          jacobians[1]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J1_minimal;
      J1_minimal.setIdentity();
      J1_minimal.block<3, 3>(0, 0) = -T_AW.C();
      J1_minimal.block<3, 3>(3, 3) = -(kinematics::plus(T_AB_.q() * T_BW.q())
                                     * kinematics::oplus(T_WA.q())).topLeftCorner<3, 3>();
      J1_minimal.block<3, 3>(3, 3) *= rotationSign;
      J1_minimal = (squareRootInformation_ * J1_minimal).eval();

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[1], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      J1 = J1_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[1] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J1_minimal_mapped(
              jacobiansMinimal[1]);
          J1_minimal_mapped = J1_minimal;
        }
      }
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis

