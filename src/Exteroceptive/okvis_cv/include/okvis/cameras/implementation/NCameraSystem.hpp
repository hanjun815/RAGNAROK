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
 * @file implementation/NCameraSystem.hpp
 * @brief Header implementation file for the NCameraSystem class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#pragma once

#include <okvis/cameras/NCameraSystem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

// Default constructor
NCameraSystem::NCameraSystem() : numUsedCameras_(0)
{
}

NCameraSystem::~NCameraSystem()
{
}

// Add a camera
void NCameraSystem::addCamera(
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
    std::shared_ptr<const cameras::CameraBase> cameraGeometry,
    DistortionType distortionType,
    bool computeOverlaps,
    const CameraType & cameraType)
{
  T_SC_.push_back(T_SC);
  cameraGeometries_.push_back(cameraGeometry);
  distortionTypes_.push_back(distortionType);
  cameraTypes_.push_back(cameraType); ///< is it a depth camera etc

  if(cameraType.isUsed) {
    numUsedCameras_++;
  }

  if(cameraType.depthType.isDepthCamera && cameraType.depthType.createVirtual) {
    // create an additional virtual camera
    okvis::kinematics::Transformation T_CCvirtual(
        cameraType.depthType.baseline,
        Eigen::Quaterniond::Identity());
    okvis::kinematics::Transformation T_SCvirtual(*T_SC*T_CCvirtual);
    virtual_T_SC_.push_back(std::shared_ptr<const okvis::kinematics::Transformation>(
                      new okvis::kinematics::Transformation(T_SCvirtual)));
    virtual_cameraGeometries_.push_back(cameraGeometry); // TODO: check if this is thread-safe.
    virtual_distortionTypes_.push_back(distortionType);
  }

  // recompute overlaps if requested
  if (computeOverlaps) {
    this->computeOverlaps();
  }
}

// get the pose of the IMU frame S with respect to the camera cameraIndex
std::shared_ptr<const okvis::kinematics::Transformation> NCameraSystem::T_SC(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return T_SC_[cameraIndex];
}

// get the pose of the IMU frame S with respect to the camera cameraIndex
std::shared_ptr<const okvis::kinematics::Transformation> NCameraSystem::rectifyT_SC(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, rectify_T_SC_.find(cameraIndex) != rectify_T_SC_.end(),
                        "Rectified camera index " << cameraIndex << "out of range.");
  try {
    return rectify_T_SC_.at(cameraIndex);
  }
  catch(const std::out_of_range& ex) {
    throw std::runtime_error("The camera ID is out of range in the rectified camera.");
  }
}

void NCameraSystem::setExtrinsics(size_t cameraIndex, kinematics::Transformation T_SCi)
{
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  T_SC_.at(cameraIndex).reset(new kinematics::Transformation(T_SCi));
}

//get the camera geometry of camera cameraIndex
std::shared_ptr<const cameras::CameraBase> NCameraSystem::cameraGeometry(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return cameraGeometries_[cameraIndex];
}

//get the camera geometry of camera cameraIndex
std::shared_ptr<const cameras::CameraBase> NCameraSystem::rectifyCameraGeometry(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, rectify_cameraGeometries_.find(cameraIndex) != rectify_cameraGeometries_.end(),
                        "Rectified camera index " << cameraIndex << "out of range.");
  try {
    return rectify_cameraGeometries_.at(cameraIndex);
  }
  catch(const std::out_of_range& ex) {
    throw std::runtime_error("The camera ID is out of range in the rectified camera.");
  }
}

// get the distortion type of cmaera cameraIndex
inline NCameraSystem::DistortionType NCameraSystem::distortionType(size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return distortionTypes_[cameraIndex];
}

// Get the overlap mask
const cv::Mat NCameraSystem::overlap(size_t cameraIndexSeenBy,
                                      size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception, cameraIndexSeenBy < T_SC_.size(),
      "Camera index " << cameraIndexSeenBy << "out of range.");
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");

  OKVIS_ASSERT_TRUE_DBG(Exception, overlapComputationValid(),
                            "Overlap computation not performed or incorrectly computed!");

  return overlapMats_[cameraIndexSeenBy][cameraIndex];
}

// Can the first camera see parts of the FOV of the second camera?
bool NCameraSystem::hasOverlap(size_t cameraIndexSeenBy,
                                      size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception, cameraIndexSeenBy < T_SC_.size(),
      "Camera index " << cameraIndexSeenBy << "out of range.");
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");

  OKVIS_ASSERT_TRUE_DBG(Exception, overlapComputationValid(),
                          "Overlap computation not performed or incorrectly computed!");

  return overlaps_[cameraIndexSeenBy][cameraIndex];
}

bool NCameraSystem::overlapComputationValid() const {
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC_.size() == cameraGeometries_.size(),
      "Number of extrinsics must match number of camera models!");

  if(overlaps_.size() != cameraGeometries_.size()) {
    return false;
  }
  if(overlapMats_.size() != cameraGeometries_.size()) {
    return false;
  }

  // also check for each element
  for(size_t i= 0; i<overlaps_.size(); ++i){
    if(overlaps_[i].size() != cameraGeometries_.size()) {
      return false;
    }
    if(overlapMats_[i].size() != cameraGeometries_.size()) {
      return false;
    }
  }
  return true;
}

size_t NCameraSystem::numCameras() const {
  return cameraGeometries_.size();
}

size_t NCameraSystem::numUsedCameras() const {
  return numUsedCameras_;
}

void NCameraSystem::addRectifyCamera(
    size_t cameraIndex,
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
    std::shared_ptr<const cameras::CameraBase> cameraGeometry)
{
  rectify_T_SC_[cameraIndex] = T_SC;
  rectify_cameraGeometries_[cameraIndex] = cameraGeometry;
}

}  // namespace cameras
}  // namespace okvis
