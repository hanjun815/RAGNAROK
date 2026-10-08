/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Feb 2, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file NoDistortion.hpp
 * @brief Header file for the NoDistortion class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_SRL_PROJECTION_NODISTORTION_HPP_
#define INCLUDE_SRL_PROJECTION_NODISTORTION_HPP_

#include <memory>
#include <Eigen/Core>
#include "srl/projection/DistortionBase.hpp"

/// \brief Main namespace of this package.
namespace srl {
/// \brief Namespace for camera-related functionality.
namespace projection {

/// \class NoDistortion
/// \brief This trivially doesn't do anything in terms of distortion.
/// This is useful for testing, or working with pre-undistorted images.
class NoDistortion : public DistortionBase
{
 public:
  /// \brief Destructor, not doing anything
  inline ~NoDistortion()
  {
  }

  //////////////////////////////////////////////////////////////
  /// \name Methods related to generic parameters
  /// @{

  /// \brief set the generic parameters
  /// @param[in] parameters Parameter vector -- length must correspond numDistortionIntrinsics().
  /// @return    True if the requirements were followed.
  bool setParameters(const VectorXf & parameters)
  {
    (void)parameters;
    return true;
  }

  /// \brief Obtain the generic parameters.
  bool getParameters(VectorXf & parameters) const
  {
    parameters.resize(0);
    return true;
  }

  /// \brief The class type.
  std::string type() const
  {
    return "NoDistortion";
  }

  /// \brief Number of derived class distortion parameters
  int numDistortionIntrinsics() const
  {
    return 0;
  }

  static const int NumDistortionIntrinsics = 0; ///< THis class does not contain any distortion intrinsics.
  /// @}

  /// \brief Unit test support -- create a test distortion object
  static std::shared_ptr<DistortionBase> createTestObject() {
    return std::shared_ptr<DistortionBase>(new NoDistortion());
  }
  /// \brief Unit test support -- create a test distortion object
  static NoDistortion testObject() {
    return NoDistortion();
  }

  //////////////////////////////////////////////////////////////
  /// \name Distortion functions
  /// @{

  /// \brief Distortion only
  /// @param[in]  pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointDistorted   The distorted normalised (!) image point.
  /// @return     True on success (no singularity)
  bool distort(const Vector2f & pointUndistorted,
                       Vector2f * pointDistorted) const
  {
    *pointDistorted = pointUndistorted;
    return true;
  }

  /// \brief Distortion and Jacobians.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  bool distort(const Vector2f & pointUndistorted,
                       Vector2f * pointDistorted,
                       Matrix2f * pointJacobian,
                       Matrix2Xf * parameterJacobian = nullptr) const
  {
    *pointDistorted = pointUndistorted;
    *pointJacobian = Matrix2f::Identity();
    if (parameterJacobian) {
      parameterJacobian->resize(2, 0);
    }
    return true;
  }

  /// \brief Distortion and Jacobians using external distortion intrinsics parameters.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[in]  parameters        The distortion intrinsics vector.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  bool distortWithExternalParameters(
      const Vector2f & pointUndistorted,
      const VectorXf & parameters, Vector2f * pointDistorted,
      Matrix2f * pointJacobian = nullptr,
      Matrix2Xf * parameterJacobian = nullptr) const
  {
    (void)parameters;
    *pointDistorted = pointUndistorted;
    if (pointJacobian) {
      *pointJacobian = Matrix2f::Identity();
    }
    if (parameterJacobian) {
      parameterJacobian->resize(2, 0);
    }
    return true;
  }
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Undistortion functions
  /// @{

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @return     True on success (no singularity)
  bool undistort(const Vector2f & pointDistorted,
                 Vector2f * pointUndistorted) const
  {
    *pointUndistorted = pointDistorted;
    return true;
  }

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointJacobian    The Jacobian w.r.t. changes on the image point.
  /// @return     True on success (no singularity)
  bool undistort(const Vector2f & pointDistorted,
                         Vector2f * pointUndistorted,
                         Matrix2f * pointJacobian) const
  {
    *pointUndistorted = pointDistorted;
    *pointJacobian = Matrix2f::Identity();
    return true;
  }
  /// @}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace projection
}  // namespace srl

#endif /* INCLUDE_SRL_PROJECTION_NODISTORTION_HPP_ */
