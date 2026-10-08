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
 * @file RealsenseRgbd.hpp
 * @brief Header file for the Realsense Rgbd class.
 * @author Simon Schaefer
 */

#ifndef INCLUDE_OKVIS_REALSENSE_RGBD_HPP_
#define INCLUDE_OKVIS_REALSENSE_RGBD_HPP_

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Realsense.hpp>

namespace okvis {

/// @brief RealsenseRGBD (D455, for now) sensor interface.
class RealsenseRgbd : public Realsense {
public:
  /// @brief Constructor. Won't start the streaming.
  /// @param sensorType Specify the sensor type. Only D455 supported for now.
  /// @param enableRgb Enable RGB camera?
  /// @param alignDepthToRgb If true, will align to RGB camera, else to left IR.
  /// @param emitterPower Set the emitter power. Unfortunately, this is not adaptive...
  /// @param rgbExposure The exposure for the RGB camera. Set to 0 to enable auto-exposure.
  RealsenseRgbd(SensorType sensorType, bool enableRgb = false,
                bool alignDepthToRgb = false, double emitterPower = 150.0,
                float rgbExposure = 0.0f);

  /// @brief Starts streaming.
  /// @return True, if successful
  virtual bool startStreaming() override final;

  /// \brief Process a frame.
  /// \param frame The frame.
  void processFrame(const rs2::frame &frame) override final;

protected:

  /// @brief Process a depth frame.
  /// @param[in] frame The frame.
  /// @param[inout] outFrame The processed output (imserted depth image).
  /// @param[out] timestamp The corresponding timestamp.
  /// \return True on success.
    bool processDepth_(const rs2::frameset &frame,
                       std::map<size_t, cv::Mat> &outFrame,
                       Time &timestamp);

private:
  double depth_scale_; ///< Device specific depth scale. depth_meters = depth_scale * pixel_value;
  bool imuStreamStarted_ = false; ///< Data stream has started.
  bool supportsMetadata_ = true; ///< Supports metadata for reading emitter mode.
  bool alignDepthToRgb_ = false; ///< Will align to RGB camera, else to left IR.
  double emitterPower_ = 150.0; ///< The emitter power. Unfortunately, this is not adaptive...
};

}

#endif // INCLUDE_OKVIS_REALSENSE_RGBD_HPP_
