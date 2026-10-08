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

#ifndef OKVIS_DEEP_LEARNING_PROCESSOR_HPP
#define OKVIS_DEEP_LEARNING_PROCESSOR_HPP

#include <map>
#include <thread>
#include <atomic>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <okvis/Measurements.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/QueuedTrajectory.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViSensorBase.hpp>

namespace okvis {

class DeepLearningProcessor {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief Callback for receiving depth measurements
  typedef std::function<bool(std::map<size_t, std::vector<okvis::CameraMeasurement>> &)> ImageCallback;

  /// \brief Callback for receiving depth & 1-sigma measurements
  typedef std::function<bool(const okvis::Time&, const cv::Mat&, const std::optional<cv::Mat>&)> LiveDepthCallback;

  DeepLearningProcessor() = default;
  virtual ~DeepLearningProcessor() = default;

  /**
   * \brief Set the blocking variable that indicates whether the addMeasurement() functions
   *        should return immediately (blocking=false), or only when the processing is complete.
   */
  void setBlocking(bool blocking) {
    blocking_ = blocking;
  }

  /// @brief Display rgb image, depth image and language features
  virtual void display(std::map<std::string, cv::Mat> &images) = 0;

  /// \name Add measurements to the algorithm.
  /**
   * \brief              Add a set of new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \return             Returns true normally. False, if the previous one has not been processed yet.
   */
  virtual bool addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
                         const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages) = 0;

  /// @brief Set the images callback before OKVIS2 graph optimization
  /// @param imagesCallback The depth and 1-sigma images callback to register.
  virtual void setLiveDepthImageCallback(const LiveDepthCallback& /*liveDepthCallback*/) {;};

  /**
   * @brief      Stores the state and keyframe updates provided by OKVIS
   *
   * @param[in]  latestState          The current OKVIS state
   * @param[in]  latestTrackingState  The current tracking state
   * @param[in]  keyframeStates       The state of the updated Keyframes
   * @param[in]  landmarks All the landmarks that the estimator updated.
   * @return     True when successful
  */
  virtual bool stateUpdateCallback(const okvis::State& /*latestState*/,
    const okvis::TrackingState& /*latestTrackingState*/,
    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> /*updatedStates*/,
    std::shared_ptr<const okvis::MapPointVector> /*landmarks*/) {return false;};

  /// @brief Set the images callback
  /// @param imagesCallback The images callback to register.
  void setImageCallback(const ImageCallback & imageCallback) {
    imageCallback_ = imageCallback;
  }

  /// @brief Check whether the processor is finished.
  virtual bool finishedProcessing() = 0;

protected:

  /// @brief Processing loops and according threads.
  virtual void processing() = 0;

  std::atomic_bool shutdown_{}; ///< True if shutdown requested.
  std::thread processingThread_;
  ImageCallback imageCallback_;

  okvis::threadsafe::Queue<std::map<size_t, std::vector<okvis::CameraMeasurement>>> cameraMeasurementsQueue_;
  std::atomic_bool blocking_;

  // A flag to control finishing.
  std::atomic_bool isProcessing_ = true;
};

} // namespace srl

#endif //OKVIS_DEEP_LEARNING_PROCESSOR_HPP