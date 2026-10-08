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
 * @file SubmapIcpError.cpp
 * @brief Source file for the Submap ICP error class.
 * @author Simon Boche
 */

#include <okvis/ceres/SubmapIcpError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/SubmappingUtils.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

SubmapIcpError::SubmapIcpError(const SupereightMapType &submap,
                               const Eigen::Vector3d &point_b,
                               const double sigma_measurement) : submapPtr_(submap), p_b_(point_b), sigma_measurement_(sigma_measurement)
{
}

void SubmapIcpError::setPb(const Eigen::Vector3d& point_b) {
  p_b_ = point_b;
}

bool SubmapIcpError::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool SubmapIcpError::EvaluateWithMinimalJacobians(const double *const *parameters, double *residuals,
                                                  double **jacobians, double **jacobiansMinimal) const {
  // if(!jacobians){
  //   residuals[0] = 0.0;
  //   return true;
  // }

  // T_WS_a
  Eigen::Map<const Eigen::Vector3d> t_WS_a(&parameters[0][0]);
  const Eigen::Quaterniond q_WS_a(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Matrix3d C_WS_a = q_WS_a.toRotationMatrix();
  okvis::kinematics::Transformation T_WS_a(t_WS_a,q_WS_a);

  // T_WS_b
  Eigen::Map<const Eigen::Vector3d> t_WS_b(&parameters[1][0]);
  const Eigen::Quaterniond q_WS_b(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
  Eigen::Matrix3d C_WS_b = q_WS_b.toRotationMatrix();
  okvis::kinematics::Transformation T_WS_b(t_WS_b,q_WS_b);

  // current T_SaSb
  okvis::kinematics::Transformation T_SaSb = T_WS_a.inverse() * T_WS_b;
  // Eigen::Vector3d p_a = (T_SaSb.T() * p_b_.homogeneous()).head<3>();
  Eigen::Vector3d p_a = T_SaSb.C() * p_b_ + T_SaSb.r();

  // Assess Occupancy and Gradient in submap
  std::optional<Eigen::Vector3f> gradf = okvis::gradFieldMeanOccup<se::Safe::On>(submapPtr_, p_a.cast<float>());
  if(!gradf || gradf.value().norm() < 1e-03){
    residuals[0] = 0.;
    if(jacobians!= nullptr && jacobians[0] != nullptr){
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
      J0.setZero();
    }
    if(jacobians!= nullptr && jacobians[1] != nullptr){
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J1(jacobians[1]);
      J1.setZero();
    }
    return true;
  }
  Eigen::Vector3d grad = gradf.value().cast<double>();
  double grad_norm = grad.norm();

  std::optional<float> occf = okvis::interpFieldMeanOccup<se::Safe::On>(submapPtr_, p_a.cast<float>());
  if(!occf){
    residuals[0] = 0.;
    if(jacobians!= nullptr && jacobians[0] != nullptr){
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
      J0.setZero();
    }
    if(jacobians!= nullptr && jacobians[1] != nullptr){
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J1(jacobians[1]);
      J1.setZero();
    }
    return true;
  }
  double occ = static_cast<double>(occf.value());
  // Define covariances with that
  sigma_map_ = fabs(submapPtr_.getDataConfig().field.log_odd_min) / (3.0f * grad_norm);

  double sigma_tot = std::sqrt(sigma_map_ * sigma_map_ + sigma_measurement_ * sigma_measurement_);
  double squareRootInformation = 1. / sigma_tot;

  // Error
  double weighted_error = squareRootInformation * occ / grad_norm;
  // assign:
  residuals[0] = weighted_error;

  // calculate jacobians, if required
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {

      // Jacobian w.r.t. p_a
      Eigen::Matrix<double,1,3> de_dp_a = squareRootInformation * grad / grad_norm;
      //std::cout << "--- de_dp_a: ---\n" << de_dp_a << std::endl;

      // Chain rule: w.r.t. T_WS_a
      Eigen::Matrix<double, 3, 6> dpa_dTwsa;
      dpa_dTwsa.setZero();
      dpa_dTwsa.topLeftCorner<3,3>() = -C_WS_a.transpose();
      dpa_dTwsa.topRightCorner<3,3>() = C_WS_a.transpose() * okvis::kinematics::crossMx(C_WS_b * p_b_ + t_WS_b - t_WS_a);
      //std::cout << "--- dpa_dTwsa: ---\n" << dpa_dTwsa << std::endl;
      //std::cout << "--- p_b_: ---\n" << p_b_.transpose() << std::endl;

      Eigen::Matrix<double,1,6> J0_minimal;
      J0_minimal = de_dp_a * dpa_dTwsa;

      // Lift Jacobian: minimal to non-minimal representation
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J0_lift;
      PoseManifold::minusJacobian(parameters[0], J0_lift.data());
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
      J0 =  J0_minimal * J0_lift;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;  // this is for Euclidean-style perturbation only.
        }
      }

    }
    if (jacobians[1] != nullptr) {

      // Jacobian w.r.t. p_a
      Eigen::Matrix<double,1,3> de_dp_a = squareRootInformation * grad / grad_norm;

      // Chain rule: w.r.t. T_WS_b
      Eigen::Matrix<double, 3, 6> dpa_dTwsb;
      dpa_dTwsb.setZero();
      dpa_dTwsb.topLeftCorner<3,3>() = C_WS_a.transpose();
      dpa_dTwsb.topRightCorner<3,3>() = -C_WS_a.transpose() * okvis::kinematics::crossMx(C_WS_b * p_b_);

      Eigen::Matrix<double,1,6> J1_minimal;
      J1_minimal = de_dp_a * dpa_dTwsb;

      // Lift Jacobian: minimal to non-minimal representation
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J1_lift;
      PoseManifold::minusJacobian(parameters[1], J1_lift.data());
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J1(jacobians[1]);
      J1 = J1_minimal * J1_lift;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[1] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
          J1_minimal_mapped = J1_minimal;
        }
      }
    }
  }
  return true;
}

std::pair<bool, bool> SubmapIcpError::nonZeroFieldAndGradient(const double *const *parameters, float& fieldVal, float& gradNorm){
  
  bool nonZeroField = true;
  bool nonZeroGradient = true;


  // T_WS_a
  Eigen::Map<const Eigen::Vector3d> t_WS_a(&parameters[0][0]);
  const Eigen::Quaterniond q_WS_a(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  okvis::kinematics::Transformation T_WS_a(t_WS_a,q_WS_a);

  // T_WS_b
  Eigen::Map<const Eigen::Vector3d> t_WS_b(&parameters[1][0]);
  const Eigen::Quaterniond q_WS_b(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
  okvis::kinematics::Transformation T_WS_b(t_WS_b,q_WS_b);

  // current T_SaSb
  okvis::kinematics::Transformation T_SaSb = T_WS_a.inverse() * T_WS_b;
  // Eigen::Vector3d p_a = (T_SaSb.T() * p_b_.homogeneous()).head<3>();
  Eigen::Vector3d p_a = T_SaSb.C() * p_b_ + T_SaSb.r();

  // Assess Occupancy and Gradient in submap
  std::optional<Eigen::Vector3f> gradf = okvis::gradFieldMeanOccup<se::Safe::On>(submapPtr_, p_a.cast<float>());
  if(!gradf || gradf.value().norm() < 1e-03){
    gradNorm = 0.0;
    nonZeroGradient = false;
  }
  else{
    gradNorm = gradf.value().norm();
  }

  std::optional<float> occf = okvis::interpFieldMeanOccup<se::Safe::On>(submapPtr_, p_a.cast<float>());
  if(!occf){
    fieldVal = 0.0f;
    nonZeroField = false;
  }
  else{
    fieldVal = occf.value();
  }

  return std::pair<bool, bool> (nonZeroField, nonZeroGradient);
}

}//namespace ceres
}//namespace okvis