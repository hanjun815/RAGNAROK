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
 * @file implementation/CameraBase.hpp
 * @brief Header implementation file for the CameraBase class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <okvis/cameras/CameraBase.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

// Set the mask. It must be the same size as the image and
bool CameraBase::setMask(const cv::Mat & mask)
{
  // check type
  if (mask.type() != CV_8UC1) {
    return false;
  }
  // check size
  if (mask.rows != imageHeight_) {
    return false;
  }
  if (mask.cols != imageWidth_) {
    return false;
  }
  mask_ = mask;
  return true;
}

/// Was a nonzero mask set?
bool CameraBase::removeMask()
{
  mask_.resize(0);
  return true;
}

// Was a nonzero mask set?
bool CameraBase::hasMask() const
{
  return (mask_.data);
}

// Get the mask.
const cv::Mat & CameraBase::mask() const
{
  return mask_;
}

bool CameraBase::isMasked(const Eigen::Vector2d& imagePoint) const
{
  if (!isInImage(imagePoint)) {
    return true;
  }
  if (!hasMask()) {
    return false;
  }
  return mask_.at<uchar>(int(imagePoint[1]), int(imagePoint[0]));
}

// Check if the keypoint is in the image.
bool CameraBase::isInImage(const Eigen::Vector2d& imagePoint) const
{
  if (imagePoint[0] < 0.0 || imagePoint[1] < 0.0) {
    return false;
  }
  if (imagePoint[0] >= imageWidth_ || imagePoint[1] >= imageHeight_) {
    return false;
  }
  return true;
}

} // namespace cameras
} // namespace okvis

