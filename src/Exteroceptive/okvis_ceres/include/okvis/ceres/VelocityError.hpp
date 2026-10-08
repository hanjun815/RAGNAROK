/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */
/**
 * @file VelocityError.hpp
 * @brief Ego-frame velocity prior error term.
 */

#ifndef INCLUDE_OKVIS_CERES_VELOCITYERROR_HPP_
#define INCLUDE_OKVIS_CERES_VELOCITYERROR_HPP_

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <ceres/sized_cost_function.h>

#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>  // SpeedAndBias typedef
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

namespace okvis {
namespace ceres {

/// \brief Velocity prior error term for one state.
/// Residual compares measured ego velocity (sensor frame) against
/// the current state's velocity transformed from world to sensor frame:
///   r = v_S_meas - C_SW * v_W_est
class VelocityError : public ::ceres::SizedCostFunction<
    3 /* residuals */, 7 /* Pose */, 9 /* SpeedAndBias */>,
    public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  using base_t = ::ceres::SizedCostFunction<3, 7, 9>;
  static const int kNumResiduals = 3;

  using information_t = Eigen::Matrix3d;
  using covariance_t = Eigen::Matrix3d;

  VelocityError() = default;

  VelocityError(const Eigen::Vector3d& v_S_meas, const information_t& information) {
    v_S_meas_ = v_S_meas;
    setInformation(information);
  }

  void setInformation(const information_t& information) {
    information_ = information;
    covariance_ = information.inverse();
    Eigen::LLT<information_t> lltOfInformation(information_);
    squareRootInformation_ = lltOfInformation.matrixL().transpose();
  }

  const information_t& information() const { return information_; }
  const covariance_t& covariance() const { return covariance_; }
  const Eigen::Vector3d& measurement() const { return v_S_meas_; }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override final {
    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
  }

  bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians,
                                    double** jacobiansMinimal) const override final {
    const Eigen::Quaterniond q_WS(
        parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    const Eigen::Matrix3d C_WS = q_WS.normalized().toRotationMatrix();
    const Eigen::Matrix3d C_SW = C_WS.transpose();

    Eigen::Map<const okvis::SpeedAndBias> estimate(parameters[1]);
    const Eigen::Vector3d v_W_est = estimate.head<3>();
    const Eigen::Vector3d v_S_est = C_SW * v_W_est;
    const Eigen::Vector3d error = v_S_meas_ - v_S_est;

    Eigen::Map<Eigen::Matrix<double, 3, 1>> weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

    // Jacobian wrt pose (minimal 6D: [dr, dtheta]) and speed/bias (9D).
    // r = v_S_meas - C_SW*v_W, and with left-multiplicative perturbation on q_WS:
    // d(C_SW*v_W)/d(dtheta) = +C_SW*[v_W]x, so dr/d(dtheta) = -C_SW*[v_W]x.
    const Eigen::Matrix3d J_rot = -C_SW * skew(v_W_est);

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setZero();
      J0_minimal.block<3,3>(0,3) = J_rot;
      J0_minimal = (squareRootInformation_ * J0_minimal).eval();

      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[0], J_lift.data());

      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != nullptr && jacobiansMinimal[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J0min(jacobiansMinimal[0]);
        J0min = J0_minimal;
      }
    }

    if (jacobians != nullptr && jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J1(jacobians[1]);
      J1.setZero();
      J1.block<3,3>(0,0) = -(squareRootInformation_ * C_SW);
    }

    if (jacobiansMinimal != nullptr && jacobiansMinimal[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J1min(jacobiansMinimal[1]);
      J1min.setZero();
      J1min.block<3,3>(0,0) = -(squareRootInformation_ * C_SW);
    }

    return true;
  }

  int residualDim() const override final { return kNumResiduals; }
  int parameterBlocks() const override final { return int(base_t::parameter_block_sizes().size()); }
  int parameterBlockDim(int parameterBlockId) const override final {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }
  std::string typeInfo() const override final { return "VelocityError"; }

 private:
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
  }

  Eigen::Vector3d v_S_meas_ = Eigen::Vector3d::Zero();
  information_t information_ = information_t::Identity();
  information_t squareRootInformation_ = information_t::Identity();
  covariance_t covariance_ = information_t::Identity();
};

}  // namespace ceres
}  // namespace okvis

#endif  // INCLUDE_OKVIS_CERES_VELOCITYERROR_HPP_

