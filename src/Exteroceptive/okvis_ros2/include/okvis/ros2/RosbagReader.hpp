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
 * @file RosbagReader.hpp
 * @brief Header file for the DatasetReader class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_ROSBAGREADER_HPP_
#define INCLUDE_OKVIS_ROSBAGREADER_HPP_

#include <atomic>
#include <thread>

#include <glog/logging.h>

#include <rclcpp/serialization.hpp>

#include <rosbag2_cpp/readers/sequential_reader.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ViSensorBase.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief Reader class acting like a VI sensor.
/// @warning Make sure to use this in combination with synchronous
/// processing, as there is no throttling of the reading process.
class RosbagReader : public DatasetReaderBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// @brief Disallow default construction.
  RosbagReader() = delete;

  /// @brief Construct pointing to dataset.
  /// @param path The absolute or relative path to the dataset.
  /// @param numCameras The total number of cameras.
  /// @param syncCameras Camera group to force synchronisation.
  /// @param deltaT Duration [s] to skip in the beginning.
  RosbagReader(const std::string& path, size_t numCameras, const std::set<size_t> & syncCameras,
                const Duration & deltaT = Duration(0.0));

  /// @brief Destructor: stops streaming.
  virtual ~RosbagReader();

  /// @brief (Re-)setting the dataset path.
  /// @param path The absolute or relative path to the dataset.
  /// @return True, if the dateset folder structure could be created.
  virtual bool setDatasetPath(const std::string & path) final;

  /// @brief Setting skip duration in the beginning.
  /// @param deltaT Duration [s] to skip in the beginning.
  /// @return True on success.
  virtual bool setStartingDelay(const okvis::Duration & deltaT) final;

  /// @brief Starts reading the dataset.
  /// @return True, if successful
  virtual bool startStreaming() final;

  /// @brief Stops reading the dataset.
  /// @return True, if successful
  virtual bool stopStreaming() final;

  /// @brief Check if currently reading the dataset.
  /// @return True, if reading.
  virtual bool isStreaming() final;

  /// @brief Get the completion fraction read already.
  /// @return Fraction read already.
  virtual double completion() const final;

private:

  /// @brief Main processing loop.
  void processing();
  std::thread processingThread_; ///< Thread running processing loop.
  rosbag2_cpp::readers::SequentialReader reader_; ///< The ros2 bag reader.

  std::string path_; ///< Dataset path.

  std::atomic_bool streaming_; ///< Are we streaming?
  std::atomic_int counter_; ///< Number of images read yet.
  size_t numImages_ = 0; ///< Number of images to read.

  size_t numCameras_ = 0; ///< Total number of cameras.
  std::set<size_t> syncCameras_; ///< Camera group to force synchronisation.

  Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].

};

}

#endif // INCLUDE_OKVIS_ROSBAGREADER_HPP_
