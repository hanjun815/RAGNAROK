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
 * @file RelativePoseError.hpp
 * @brief Header file for the RelativePoseError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_RELATIVEPOSEERROR_HPP_
#define INCLUDE_OKVIS_CERES_RELATIVEPOSEERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Relative error between two poses.
class RelativePoseError : public ::ceres::SizedCostFunction<
    6 /* number of residuals */,
    7, /* size of first parameter */
    7 /* size of second parameter */>, public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  /// \brief Number of residuals (6).
  static const int kNumResiduals = 6;

  /// \brief The information matrix type (6x6).
  typedef Eigen::Matrix<double, 6, 6> information_t;

  /// \brief The covariance matrix type (same as information).
  typedef Eigen::Matrix<double, 6, 6> covariance_t;

  /// \brief Default constructor.
  RelativePoseError();

  /// \brief Construct with measurement and information matrix
  /// @param[in] information The information (weight) matrix.
  /// @param T_AB The relative transformation (if not identity).
  RelativePoseError(
      const Eigen::Matrix<double, 6, 6> & information,
      const kinematics::Transformation & T_AB = kinematics::Transformation::Identity());

  /// \brief Construct with measurement and variance.
  /// @param[in] translationVariance The (relative) translation variance.
  /// @param[in] rotationVariance The (relative) rotation variance.
  /// @param T_AB The relative transformation (if not identity).
  RelativePoseError(
      double translationVariance, double rotationVariance,
      const kinematics::Transformation & T_AB = kinematics::Transformation::Identity());

  /// \brief Trivial destructor.
  virtual ~RelativePoseError() override = default;

  // setters
  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const information_t & information);

  // getters
  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const {
    return covariance_;
  }

  // error term and Jacobian implementation
  /**
    * @brief This evaluates the error term and additionally computes the Jacobians.
    * @param parameters Pointer to the parameters (see ceres)
    * @param residuals Pointer to the residual vector (see ceres)
    * @param jacobians Pointer to the Jacobians (see ceres)
    * @return success of th evaluation.
    */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  int parameterBlockDim(int parameterBlockId) const override final {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const override final {
    return "RelativePoseError";
  }

  /// @brief Access the underlying relative transformation T_AB.
  /// @return T_AB.
  const kinematics::Transformation& T_AB() const {return T_AB_;}

 protected:

  // weighting related
  kinematics::Transformation T_AB_; ///< The relative transformation between the poses.
  information_t information_; ///< The 6x6 information matrix.
  information_t squareRootInformation_; ///< The 6x6 square root information matrix.
  covariance_t covariance_; ///< The 6x6 covariance matrix.

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_RELATIVEPOSEERROR_HPP_ */
