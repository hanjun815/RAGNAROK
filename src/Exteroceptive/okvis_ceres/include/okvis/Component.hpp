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
 * @file okvis/Component.hpp
 * @brief Header file for the Component class. Load/save/multisession stuff.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_COMPONENT_HPP_
#define INCLUDE_OKVIS_COMPONENT_HPP_

#include <okvis/ViGraphEstimator.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief Class containing a SLAM run.
class Component
{
  public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Constructor with graph and frames
  Component(const ImuParameters &imuParameters,
            const cameras::NCameraSystem& nCameraSystem,
            ViGraphEstimator &fullGraph,
            std::map<StateId, MultiFramePtr> multiFrames);

  /// \brief Default constructor
  Component(const ImuParameters &imuParameters, const cameras::NCameraSystem& nCameraSystem);

  /// \brief Load this component.
  /// \return True on success.
  bool load(const std::string &path);

  /// \brief Save this component.
  /// \return True on success.
  bool save(const std::string &path);

  //private:
  ImuParameters imuParameters_; ///< IMU parameters of this Component.
  cameras::NCameraSystem nCameraSystem_; ///< Multi-camera configuration of this Component.
  ViGraphEstimator *fullGraph_ = nullptr; ///< The full graoh for asynchronous optimisation.
  std::unique_ptr<ViGraphEstimator> fullGraphOwn_; ///< Owned full graph.
  std::map<StateId, MultiFramePtr> multiFrames_; ///< All the multiframes added so far.
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_COMPONENT_HPP_ */
