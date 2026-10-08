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
 * @file QueuedTrajectory.hpp
 * @brief Header file for the QueuedTrajectory class.
 * @author Simon Schaefer
 */

#ifndef OKVIS_QUEUEDTRAJECTORY_HPP
#define OKVIS_QUEUEDTRAJECTORY_HPP

#include <vector>
#include <deque>
#include <Eigen/Core>

#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ViInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief A class to store measurements in a queue and get the according state when available.
///        The class is templated with the measurement type, e.g. cv::Mat for images.
template<class MEASUREMENT_T>
class QueuedTrajectory {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef Measurement<MEASUREMENT_T> MeasurementTyped; ///< Helper Measurem. typed on MEASUREMENT_T.

  /// @brief For each measurement in the internal queue, get the corresponding state from the
  ///        trajectory based on the measurement's timestamp. If the state is available, return the
  ///        state-measurement pair and remove the measurement from the queue. Otherwise, return an
  ///        empty vector.
  /// @param trajectory The trajectory to get the states from. 
  std::vector<std::pair<MeasurementTyped, okvis::State>> getStates(okvis::Trajectory& trajectory);

  /// @brief Enqueues a new measurement to the measurement queue.
  /// @param measurement measurement to queue.
  void enqueue(const MeasurementTyped& measurement);

  /// @brief Enqueues new raw measurement data and timestamp. Therefore, first a MeasurementTyped
  ///        object is created from the timestamp and measurement data. Then, the MeasurementTyped
  ///        object is enqueued.
  /// @param data measurement data.
  /// @param timestamp measurement timestamp.
  void enqueue(const MEASUREMENT_T& data, const okvis::Time& timestamp);

private:
  /// \brief The internal measurement queue.
  std::deque<MeasurementTyped, Eigen::aligned_allocator<MeasurementTyped> > measurementQueue_;

};

#include "implementation/QueuedTrajectory.hpp"

}

#endif // OKVIS_QUEUEDTRAJECTORY_HPP
