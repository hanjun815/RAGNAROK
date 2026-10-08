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
 * @file CeresIterationCallback.hpp
 * @brief Header file for the CeresIterationCallback class.
 *        Used to enforce a time limit on the ceres optimization.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_CERESITERATIONCALLBACK_HPP
#define INCLUDE_OKVIS_CERES_CERESITERATIONCALLBACK_HPP

#include <ceres/iteration_callback.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/**
 * @brief The CeresIterationCallback class tries to enforce a time limit on the
 *        optimization. It does not guarantee to stay within the time budget as
 *        it assumes the next iteration takes as long as the previous iteration.
 */
class CeresIterationCallback : public ::ceres::IterationCallback {
 public:

  /**
   * @brief The constructor.
   * @param[in] timeLimit Time budget for the optimization.
   * @param[in] iterationMinimum Minimum iterations the optimization should perform
   *            disregarding the time.
   */
  CeresIterationCallback(double timeLimit, int iterationMinimum);

  /// \brief Trivial Destructor.
  virtual inline ~CeresIterationCallback() override {
  }

  /// @brief This method is called after every iteration in ceres.
  /// @param[in] summary The iteration summary.
  ::ceres::CallbackReturnType operator()(
      const ::ceres::IterationSummary& summary) override;

  /**
   * @brief setTimeLimit changes time limit of optimization.
   *        If you want to disable the time limit, either set it to a large value,
   *        delete the callback in the ceres options or set the minimum iterations
   *        to the maximum iteration.
   * @param[in] timeLimit desired time limit in seconds
   */
  void setTimeLimit(double timeLimit);

  /**
   * @brief setMinimumIterations changes the minimum iterations the optimization
   *        goes through disregarding the time limit
   * @param iterationMinimum
   */
  void setMinimumIterations(int iterationMinimum);

 private:
  double timeLimit_; ///< The set time limit.
  int iterationMinimum_; ///< The set maximum no. iterations.
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_CERESITERATIONCALLBACK_HPP */
