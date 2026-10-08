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
 * @file RpgDatasetReader.hpp
 * @brief Header file for the DatasetReader class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_RPGDATASETREADER_HPP_
#define INCLUDE_OKVIS_RPGDATASETREADER_HPP_

#include <fstream>
#include <atomic>
#include <thread>

#include <glog/logging.h>

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
class RpgDatasetReader : public DatasetReaderBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// @brief Disallow default construction.
  RpgDatasetReader() = delete;

  /// @brief Construct pointing to dataset.
  /// @param path The absolute or relative path to the dataset.
  /// @param deltaT Duration [s] to skip in the beginning.
  /// @param numCameras Number of cameras. If -1, use all that are available in the dataset.
  RpgDatasetReader(const std::string& path, const Duration & deltaT = Duration(0.0),
                   int numCameras = -1);

  /// @brief Destructor: stops streaming.
  virtual ~RpgDatasetReader();

  /// @brief (Re-)setting the dataset path.
  /// @param path The absolute or relative path to the dataset.
  /// @return True, if the dateset folder structure could be created.
  virtual bool setDatasetPath(const std::string & path) final;

  /// @brief Setting skip duration in the beginning.
  /// deltaT Duration [s] to skip in the beginning.
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

  std::string path_; ///< Dataset path.

  std::atomic_bool streaming_; ///< Are we streaming?
  std::atomic_int counter_; ///< Number of images read yet.
  size_t numImages_ = 0; ///< Number of images to read.

  std::ifstream imuFile_; ///< Imu csv file.

  Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].

  /// @brief The times and names of all images by camera.
  std::vector < std::vector < std::pair<Time, std::string> > > allImageNames_;

  int numCameras_ = -1; ///< The number of cameras to consider (-1 = all).

};

}

#endif // INCLUDE_OKVIS_RPGDATASETREADER_HPP_
