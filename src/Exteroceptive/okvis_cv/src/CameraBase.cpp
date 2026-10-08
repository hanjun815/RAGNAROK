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
 * @file CameraBase.cpp
 * @brief Source file for the CameraBase class.
 * @author Stefan Leutenegger
 */

#include <okvis/cameras/CameraBase.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

// Creates a random (uniform distribution) image point.
Eigen::Vector2d CameraBase::createRandomImagePoint() const
{
  // Uniform random sample in image coordinates.
  // Add safety boundary for later inaccurate backprojection
  Eigen::Vector2d outPoint = Eigen::Vector2d::Random();
  outPoint += Eigen::Vector2d::Ones();
  outPoint *= 0.5;
  outPoint[0] *= double(imageWidth_-0.022);
  outPoint[0] += 0.011;
  outPoint[1] *= double(imageHeight_-0.022);
  outPoint[1] += 0.011;
  return outPoint;
}

// Creates a random visible point in Euclidean coordinates.
Eigen::Vector3d CameraBase::createRandomVisiblePoint(double minDist,
                                                     double maxDist) const
{
  // random image point first:
  Eigen::Vector2d imagePoint = createRandomImagePoint();
  // now sample random depth:
  Eigen::Vector2d depth = Eigen::Vector2d::Random();
  Eigen::Vector3d ray;
  backProject(imagePoint, &ray);
  ray.normalize();
  ray *= (0.5 * (maxDist - minDist) * (depth[0] + 1.0) + minDist);  // rescale and offset
  return ray;
}

// Creates a random visible point in homogeneous coordinates.
Eigen::Vector4d CameraBase::createRandomVisibleHomogeneousPoint(
    double minDist, double maxDist) const
{
  Eigen::Vector3d point = createRandomVisiblePoint(minDist, maxDist);
  return Eigen::Vector4d(point[0], point[1], point[2], 1.0);
}

}  // namespace cameras
}  // namespace okvis
