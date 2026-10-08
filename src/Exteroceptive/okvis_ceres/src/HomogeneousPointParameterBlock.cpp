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
 * @file HomogeneousPointParameterBlock.cpp
 * @brief Source file for the HomogeneousPointParameterBlock class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
HomogeneousPointParameterBlock::HomogeneousPointParameterBlock()
    : base_t::ParameterBlockSized(),
      initialized_(false) {
  setFixed(false);
}
// Trivial destructor.
HomogeneousPointParameterBlock::~HomogeneousPointParameterBlock() {
}

// Constructor with estimate and time.
HomogeneousPointParameterBlock::HomogeneousPointParameterBlock(
    const Eigen::Vector4d& point, uint64_t id, bool initialized) {
  setEstimate(point);
  setId(id);
  setInitialized(initialized);
  setFixed(false);
}

// Constructor with estimate and time.
HomogeneousPointParameterBlock::HomogeneousPointParameterBlock(
    const Eigen::Vector3d& point, uint64_t id, bool initialized) {
  setEstimate(Eigen::Vector4d(point[0], point[1], point[2], 1.0));
  setId(id);
  setInitialized(initialized);
  setFixed(false);
}

}  // namespace ceres
}  // namespace okvis
