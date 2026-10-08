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
 *  Created on: Feb 3, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file projection/EquidistantDistortion.hpp
 * @brief Header file for the EquidistantDistortion class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_SRL_PROJECTION_EQUIDISTANTDISTORTION_HPP_
#define INCLUDE_SRL_PROJECTION_EQUIDISTANTDISTORTION_HPP_

#include <memory>
#include <Eigen/Core>
#include "srl/projection/DistortionBase.hpp"

/// \brief Main namespace of this package.
namespace srl {
/// \brief Namespace for camera-related functionality.
namespace projection {

/// \class EquidistantDistortion
/// \brief Implements equidistant distortion.
class EquidistantDistortion : public DistortionBase
{
 public:
  /// \brief The default constructor with all zero ki
  inline EquidistantDistortion();

  /// \brief Constructor initialising ki
  /// @param[in] k1 equidistant parameter 1
  /// @param[in] k2 equidistant parameter 2
  /// @param[in] k3 equidistant parameter 3
  /// @param[in] k4 equidistant parameter 4
  inline EquidistantDistortion(float_t k1, float_t k2, float_t k3, float_t k4);
  //////////////////////////////////////////////////////////////
  /// \name Methods related to generic parameters
  /// @{

  /// \brief set the generic parameters
  /// @param[in] parameters Parameter vector -- length must correspond numDistortionIntrinsics().
  /// @return    True if the requirements were followed.
  inline bool setParameters(const VectorXf & parameters);

  /// \brief Obtain the generic parameters.
  inline bool getParameters(VectorXf & parameters) const
  {
    parameters = parameters_;
    return true;
  }
  /// \brief The class type.
  inline std::string type() const
  {
    return "EquidistantDistortion";
  }

  /// \brief Number of distortion parameters.
  inline int numDistortionIntrinsics() const
  {
    return NumDistortionIntrinsics;
  }

  static const int NumDistortionIntrinsics = 4;  ///< The Number of distortion parameters.
  /// @}

  /// \brief Unit test support -- create a test distortion object
  static std::shared_ptr<DistortionBase> createTestObject()
  {
    return std::shared_ptr<DistortionBase>(
        new EquidistantDistortion(-0.21, 0.14, 0.0006, 0.0003));
  }
  /// \brief Unit test support -- create a test distortion object
  static EquidistantDistortion testObject()
  {
    return EquidistantDistortion(-0.21, 0.14, 0.0006, 0.0003);
  }

  //////////////////////////////////////////////////////////////
  /// \name Distortion functions
  /// @{

  /// \brief Distortion only
  /// @param[in]  pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointDistorted   The distorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool distort(const Vector2f & pointUndistorted,
                      Vector2f * pointDistorted) const;

  /// \brief Distortion and Jacobians.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distort(const Vector2f & pointUndistorted,
                      Vector2f * pointDistorted,
                      Matrix2f * pointJacobian,
                      Matrix2Xf * parameterJacobian = nullptr) const;

  /// \brief Distortion and Jacobians using external distortion intrinsics parameters.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[in]  parameters        The distortion intrinsics vector.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distortWithExternalParameters(
      const Vector2f & pointUndistorted,
      const VectorXf & parameters, Vector2f * pointDistorted,
      Matrix2f * pointJacobian = nullptr,
      Matrix2Xf * parameterJacobian = nullptr) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Undistortion functions
  /// @{

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool undistort(const Vector2f & pointDistorted,
                        Vector2f * pointUndistorted) const;

  /// \brief Undistortion only
    /// @param[in]  pointDistorted   The distorted normalised (!) image point.
    /// @param[out] pointUndistorted The undistorted normalised (!) image point.
    /// @param[out] pointJacobian    The Jacobian w.r.t. changes on the image point.
    /// @return     True on success (no singularity)
  inline bool undistort(const Vector2f & pointDistorted,
                        Vector2f * pointUndistorted,
                        Matrix2f * pointJacobian) const;
  /// @}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  Matrixf<NumDistortionIntrinsics, 1> parameters_;  ///< all distortion parameters

  float_t k1_;  ///< radial parameter 1
  float_t k2_;  ///< radial parameter 2
  float_t k3_;  ///< tangential parameter 1
  float_t k4_;  ///< tangential parameter 2
};

}  // namespace projection
}  // namespace srl

#include "implementation/EquidistantDistortion.hpp"

#endif /* INCLUDE_SRL_PROJECTION_EQUIDISTANTDISTORTION_HPP_ */
