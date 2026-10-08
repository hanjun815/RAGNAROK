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
 * @brief Source file for the QueuedTrajectory class.
 * @author Simon Schaefer
 */

#pragma once

#include <okvis/QueuedTrajectory.hpp>


template<class MEASUREMENT_T>
std::vector<std::pair<okvis::Measurement<MEASUREMENT_T>, okvis::State>>
okvis::QueuedTrajectory<MEASUREMENT_T>::getStates(okvis::Trajectory& trajectory) {
  std::vector<std::pair<MeasurementTyped, okvis::State>> states;
  while (!measurementQueue_.empty()) {
    okvis::State state;
    MeasurementTyped measurement = measurementQueue_.back();
    bool valid_transform = trajectory.getState(measurement.timeStamp, state);

    // The queue is sorted in time. If the current iteration's transform is invalid, the next
    // iterations with an even larger timestamp will be invalid, too.
    if (!valid_transform) {
      break;
    }

    // Otherwise, push the state-measurement-pair to the output vector and remove the measurement
    // from the queue.
    measurementQueue_.pop_back();
    std::pair<MeasurementTyped, okvis::State> state_measurement_pair(measurement, state);
    states.push_back(state_measurement_pair);
  }
  return states;
}

template<class MEASUREMENT_T>
void okvis::QueuedTrajectory<MEASUREMENT_T>::enqueue(
   const okvis::Measurement<MEASUREMENT_T>& measurement) {
  measurementQueue_.push_front(measurement);
  while (measurementQueue_.size() > 20) {
    LOG(WARNING) << "RGB Image queue overloaded, popping last element";
    measurementQueue_.pop_back();
  }
}

template<class MEASUREMENT_T>
void okvis::QueuedTrajectory<MEASUREMENT_T>::enqueue(
  const MEASUREMENT_T& data, const okvis::Time& timestamp) {
  MeasurementTyped measurement(timestamp, data);
  enqueue(measurement);
}
