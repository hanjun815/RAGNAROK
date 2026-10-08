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
 * @file PoseParameterBlock.cpp
 * @brief Source file for the PoseParameterBlock class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/PoseParameterBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
PoseParameterBlock::PoseParameterBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

// Trivial destructor.
PoseParameterBlock::~PoseParameterBlock() {
}

// Constructor with estimate and time.
PoseParameterBlock::PoseParameterBlock(
    const okvis::kinematics::Transformation& T_WS, uint64_t id,
    const okvis::Time& timestamp) {
  setEstimate(T_WS);
  setId(id);
  setTimestamp(timestamp);
  setFixed(false);
}

}  // namespace ceres
}  // namespace okvis
