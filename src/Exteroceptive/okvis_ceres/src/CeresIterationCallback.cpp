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
 * @file CeresIterationCallback.cpp
 * @brief Source file for the CeresIterationCallback class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/CeresIterationCallback.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

CeresIterationCallback::CeresIterationCallback(double timeLimit, int iterationMinimum)
  : timeLimit_(timeLimit),
    iterationMinimum_(iterationMinimum) {
}

::ceres::CallbackReturnType CeresIterationCallback::operator()(
    const ::ceres::IterationSummary &summary) {
  // assume next iteration takes the same time as current iteration
  if (summary.iteration >= iterationMinimum_
      && summary.cumulative_time_in_seconds
      + summary.iteration_time_in_seconds > timeLimit_) {
    return ::ceres::SOLVER_TERMINATE_SUCCESSFULLY;
  }
  return ::ceres::SOLVER_CONTINUE;
}

void CeresIterationCallback::setTimeLimit(double timeLimit) {
  timeLimit_ = timeLimit;
}

void CeresIterationCallback::setMinimumIterations(int iterationMinimum) {
  iterationMinimum_ = iterationMinimum;
}


}  // namespace ceres
}  // namespace okvis

