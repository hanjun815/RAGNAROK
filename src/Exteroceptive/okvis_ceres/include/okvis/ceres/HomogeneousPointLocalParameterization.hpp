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
 * @file HomogeneousPointLocalParameterization.hpp
 * @brief Header file for the HomogeneousPointLocalParameterization class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTLOCALPARAMETERIZATION_HPP_
#define INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTLOCALPARAMETERIZATION_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/manifold.h>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief This is a simple parameterisation for homogeneous points: we assume they are represented
/// as hPoint = [x,y,z,c]^T, where c is a constant for scaling. Then, adding [dx,dy,dz] simply
/// becomes [x+dx,y+dx,z+dz,c]^T. Note that to keep things simple, the minus operation assumes
/// equal constants c...!
class HomogeneousPointManifold : public ::ceres::Manifold {
 public:
  /// \brief Default destructor.
  virtual ~HomogeneousPointManifold() override = default;

  /// \brief Dimension of the ambient space (R^3 x const) in which the manifold is embedded.
  /// \return  The dimension (4).
  virtual int AmbientSize() const override {return 4;}

  /// \brief Dimension of the tangent space (R^3).
  /// \return  The dimension (3).
  virtual int TangentSize() const override {return 3;}

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

#endif /* INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTLOCALPARAMETERIZATION_HPP_ */
