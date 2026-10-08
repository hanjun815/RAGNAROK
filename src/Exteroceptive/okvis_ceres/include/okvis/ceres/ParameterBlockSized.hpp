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
 * @file ParameterBlockSized.hpp
 * @brief Header file for the ParameterBlockSized class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_PARAMETERBLOCKSIZED_HPP_
#define INCLUDE_OKVIS_CERES_PARAMETERBLOCKSIZED_HPP_

#include <okvis/ceres/ParameterBlock.hpp>
#include <okvis/assert_macros.hpp>
#include <Eigen/Core>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// @brief Base class providing the interface for parameter blocks.
/// @tparam Dim     Dimension of parameter block
/// @tparam MinDim  Minimal dimension of parameter block
/// @tparam T       The type of the estimate
template<int Dim, int MinDim, class T>
class ParameterBlockSized : public okvis::ceres::ParameterBlock {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// @brief Dimension of the parameter block.
  static const int Dimension = Dim;

  /// @brief Internal (minimal) dimension.
  static const int MinimalDimension = MinDim;

  /// \brief Make the parameter type accessible.
  typedef T parameter_t;

  /// \brief Default constructor -- initialises elements in parametes_ to zero.
  ParameterBlockSized() = default;

  /// \brief Trivial destructor.
  virtual ~ParameterBlockSized() override = default;

  /// @name Setters
  /// @{

  /// @brief Set estimate of this parameter block.
  /// @param[in] estimate The estimate to set this to.
  void setEstimate(const parameter_t& estimate) {
    estimate_ = estimate;
  }

  /// @}

  /// @name Getters
  /// @{

  /// @brief Get estimate.
  /// \return The estimate.
  const parameter_t& estimate() const {
    return estimate_;
  }

  /// @brief Get the parameter dimension.
  /// \return The parameter dimension.
  virtual int dimension() const override final {
    return Dimension;
  }

  /// @brief Get the internal minimal parameter dimension.
  /// \return The internal minimal parameter dimension.
  virtual int minimalDimension() const override final {
    return MinimalDimension;
  }

  /// @}

 protected:
  /// @brief Parameters
  parameter_t estimate_;
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_PARAMETERBLOCKSIZED_HPP_ */
