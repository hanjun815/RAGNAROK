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
 * @file PoseLocalParameterization.hpp
 * @brief Header file for the PoseLocalParemerization class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_POSELOCALPARAMETERIZATION_HPP_
#define INCLUDE_OKVIS_CERES_POSELOCALPARAMETERIZATION_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/manifold.h>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/operators.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief A simple pose manifold parameterisation using ambient space R^3 x SO(3).
class PoseManifold : public ::ceres::Manifold {
 public:
  /// \brief Default destructor.
  virtual ~PoseManifold() override = default;

  /// \brief Dimension of the ambient space (R^3 x SO(3)) in which the manifold is embedded.
  /// \return  The dimension (7).
  virtual int AmbientSize() const override {return 7;}

  /// \brief Dimension of the tangent space (R^6).
  /// \return  The dimension (6).
  virtual int TangentSize() const override {return 6;}

  /// \brief Generalised addition: x_plus_delta = Plus(x, delta).
  /// @param[in] x x.
  /// @param[in] delta delta.
  /// @param[out] x_plus_delta x_plus_delta.
  /// \return true on success.
  static bool plus(const double* x,
                   const double* delta,
                   double* x_plus_delta);

  /// \brief Generalised addition: x_plus_delta = Plus(x, delta).
  /// @param[in] x x.
  /// @param[in] delta delta.
  /// @param[out] x_plus_delta x_plus_delta.
  /// \return true on success.
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const override;

  /// \brief Jacobian of the plus operation (generalised addition: x_plus_delta = Plus(x, delta)).
  /// @param[in] x x.
  /// @param[out] jacobian The jacobian.
  /// \return true on success.
  static bool plusJacobian(const double* x, double* jacobian);

  /// \brief Jacobian of the plus operation (generalised addition: x_plus_delta = Plus(x, delta)).
  /// @param[in] x x.
  /// @param[out] jacobian The jacobian.
  /// \return true on success.
  virtual bool PlusJacobian(const double* x, double* jacobian) const override;

  /// \brief Generalised subtraction: y_minus_x = Minus(y, x).
  /// @param[in] y y.
  /// @param[in] x x.
  /// @param[out] y_minus_x y_minus_x.
  /// \return true on success.
  static bool minus(const double* y,
                    const double* x,
                    double* y_minus_x);

  /// \brief Generalised subtraction: y_minus_x = Minus(y, x).
  /// @param[in] y y.
  /// @param[in] x x.
  /// @param[out] y_minus_x y_minus_x.
  /// \return true on success.
  virtual bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override;

  /// \brief Jacobian of the minus operation (generalised subtraction: y_minus_x = Minus(y, x)).
  /// @param[in] x x.
  /// @param[out] jacobian The Jacobian.
  /// \return true on success.
  static bool minusJacobian(const double* x, double* jacobian);

  /// \brief Jacobian of the minus operation (generalised subtraction: y_minus_x = Minus(y, x)).
  /// @param[in] x x.
  /// @param[out] jacobian The Jacobian.
  /// \return true on success.
  virtual bool MinusJacobian(const double* x, double* jacobian) const override;

  /// \brief Plus jacobian verification using numeric differences for convenience.
  /// @param[in] x x.
  /// @param[out] jacobian The Jacobian.
  /// @param[out] jacobianNumDiff The Jacobian with numeric differences.
  /// \return true on success.
  bool verifyJacobianNumDiff(const double* x, double* jacobian,
                        double* jacobianNumDiff) const;
};

/// \brief A simple pose manifold parameterisation for 4DoF transformation.
/// Here, we only perturb the translation and yaw though.
class PoseManifold4d : public ::ceres::Manifold {
 public:
  /// \brief Default destructor.
  virtual ~PoseManifold4d() override = default;

  /// \brief Dimension of the ambient space (R^3 x SO(3)) in which the manifold is embedded.
  /// \return  The dimension (7).
  virtual int AmbientSize() const override {return 7;}

  /// \brief Dimension of the tangent space (R^6).
  /// \return  The dimension (4).
  virtual int TangentSize() const override {return 4;}

  /// \brief Generalised addition: x_plus_delta = Plus(x, delta).
  /// @param[in] x x.
  /// @param[in] delta delta.
  /// @param[out] x_plus_delta x_plus_delta.
  /// \return true on success.
  static bool plus(const double* x,
                   const double* delta,
                   double* x_plus_delta);

  /// \brief Generalised addition: x_plus_delta = Plus(x, delta).
  /// @param[in] x x.
  /// @param[in] delta delta.
  /// @param[out] x_plus_delta x_plus_delta.
  /// \return true on success.
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const override;

  /// \brief Jacobian of the plus operation (generalised addition: x_plus_delta = Plus(x, delta)).
  /// @param[in] x x.
  /// @param[out] jacobian The jacobian.
  /// \return true on success.
  static bool plusJacobian(const double* x, double* jacobian);

  /// \brief Jacobian of the plus operation (generalised addition: x_plus_delta = Plus(x, delta)).
  /// @param[in] x x.
  /// @param[out] jacobian The jacobian.
  /// \return true on success.
  virtual bool PlusJacobian(const double* x, double* jacobian) const override;

  /// \brief Generalised subtraction: y_minus_x = Minus(y, x).
  /// @param[in] y y.
  /// @param[in] x x.
  /// @param[out] y_minus_x y_minus_x.
  /// \return true on success.
  static bool minus(const double* y,
                    const double* x,
                    double* y_minus_x);

  /// \brief Generalised subtraction: y_minus_x = Minus(y, x).
  /// @param[in] y y.
  /// @param[in] x x.
  /// @param[out] y_minus_x y_minus_x.
  /// \return true on success.
  virtual bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override;

  /// \brief Jacobian of the minus operation (generalised subtraction: y_minus_x = Minus(y, x)).
  /// @param[in] x x.
  /// @param[out] jacobian The Jacobian.
  /// \return true on success.
  static bool minusJacobian(const double* x, double* jacobian);

  /// \brief Jacobian of the minus operation (generalised subtraction: y_minus_x = Minus(y, x)).
  /// @param[in] x x.
  /// @param[out] jacobian The Jacobian.
  /// \return true on success.
  virtual bool MinusJacobian(const double* x, double* jacobian) const override;


};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_POSELOCALPARAMETERIZATION_HPP_ */
