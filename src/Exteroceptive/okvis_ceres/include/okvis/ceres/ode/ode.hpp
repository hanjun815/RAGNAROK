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
 * @file ode.hpp
 * @brief File for ODE integration functionality.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_ODE_ODE_HPP_
#define INCLUDE_OKVIS_CERES_ODE_ODE_HPP_

#include <math.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
/// \brief ode Namespace for functionality related to ODE integration implemented in okvis.
namespace ode {

/// \brief sin(x)/x: to make things a bit faster than using angle-axis conversion.
/// \param x Input x.
/// \return Result of sin(x)/x.
__inline__ double sinc(double x) {
  if (fabs(x) > 1e-6) {
   return sin(x) / x;
   } else{
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}

}

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_ODE_ODE_HPP_ */
