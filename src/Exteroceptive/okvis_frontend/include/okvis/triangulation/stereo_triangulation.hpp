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
 * @file stereo_triangulation.hpp
 * @brief File containing the triangulateFast method definition.
 * @author Stefan Leutenegger
 */


#ifndef INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_
#define INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_

#include <Eigen/Core>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief
namespace triangulation {

/**
 * @brief Triangulate the intersection of two rays.
 * @warning The rays e1 and e2 need to be normalized!
 * @param[in]  p1 Camera center position of frame A in coordinate frame A
 * @param[in]  e1 Ray through keypoint of frame A in coordinate frame A.
 * @param[in]  p2 Camera center position of frame B in coordinate frame A.
 * @param[in]  e2 Ray through keypoint of frame B in coordinate frame A.
 * @param[in]  sigma Ray uncertainty.
 * @param[out] isValid Is the triangulation valid.
 * @param[out] isParallel Are the rays parallel?
 * @return Homogeneous coordinates of triangulated point.
 */
Eigen::Vector4d triangulateFast(const Eigen::Vector3d & p1,
                                const Eigen::Vector3d & e1,
                                const Eigen::Vector3d & p2,
                                const Eigen::Vector3d & e2, double sigma,
                                bool & isValid, bool & isParallel);
}
}

#endif /* INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_ */
