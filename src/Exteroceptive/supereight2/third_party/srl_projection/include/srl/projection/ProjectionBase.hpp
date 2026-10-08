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
 *  Created on: Jan 28, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file projection/ProjectionBase.hpp
 * @brief Header file for the ProjectionBase class.
 * @author Stefan Leutenegger
 */


#ifndef INCLUDE_SRL_PROJECTION_PROJECTIONBASE_HPP_
#define INCLUDE_SRL_PROJECTION_PROJECTIONBASE_HPP_

#include <vector>
#include <memory>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include <srl/typedefs.hpp>
#include <srl/projection/DistortionBase.hpp>

/// \brief Main namespace of this package.
namespace srl {
/// \brief Namespace for camera-related functionality.
namespace projection {

/// \class ProjectionStatus
/// \brief Indicates what happened when applying any of the project functions.
enum class ProjectionStatus
{
  Successful,
  OutsideImage,
  Masked,
  Behind,
  Invalid
};


/// \class ProjectionBase
/// \brief Base class for all camera models.
class ProjectionBase
{
 public:
  /// \brief default Constructor -- does nothing serious
  inline ProjectionBase()
      : imageWidth_(0),
        imageHeight_(0)
  {
  }

  /// \brief Constructor for width, height and Id
  inline ProjectionBase(int imageWidth, int imageHeight)
        : imageWidth_(imageWidth),
          imageHeight_(imageHeight)
    {
    }

  /// \brief Destructor -- does nothing
  inline virtual ~ProjectionBase()
  {
  }

  //////////////////////////////////////////////////////////////
  /// \name Methods related to masking a certain image area as invalid.
  /// @{

  /// \brief Set the mask. It must be the same size as the image and
  /// comply with OpenCV: 0 == masked, nonzero == valid.
  /// Type must be CV_8U1C.
  /// @param[in] mask The actual mask.
  /// @return True if the requirements were followed.
  inline bool setMask(const cv::Mat & mask);

  /// \brief Was a nonzero mask set?
  inline bool hasMask() const;

  /// \brief stop masking
  /// @return Always true.
  inline bool removeMask();

  /// \brief Get the mask.
  inline const cv::Mat & mask() const;

  /// @}

  /// \brief The width of the image in pixels.
  inline int imageWidth() const
  {
    return imageWidth_;
  }
  /// \brief The height of the image in pixels.
  inline int imageHeight() const
  {
    return imageHeight_;
  }

  /// \brief obtain all intrinsics
  virtual void getIntrinsics(VectorXf & intrinsics) const = 0;

  /// \brief overwrite all intrinsics - use with caution !
  virtual bool setIntrinsics(const VectorXf & intrinsics) = 0;

  //////////////////////////////////////////////////////////////
  /// \name Methods to project points
  /// @{

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Euclidean coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus project(const Vector3f & point,
                                   Vector2f * imagePoint) const = 0;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus project(
      const Vector3f & point, Vector2f * imagePoint,
      Matrixf<2, 3> * pointJacobian,
      Matrix2Xf * intrinsicsJacobian = nullptr) const = 0;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus projectWithExternalParameters(
      const Vector3f & point, const VectorXf & parameters,
      Vector2f * imagePoint, Matrixf<2, 3> * pointJacobian = nullptr,
      Matrix2Xf * intrinsicsJacobian = nullptr) const = 0;

  /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in Euclidean coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  virtual void projectBatch(const Matrix3Xf & points,
                            Matrix2Xf * imagePoints,
                            std::vector<ProjectionStatus> * stati) const = 0;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Homogeneous coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus projectHomogeneous(
      const Vector4f & point, Vector2f * imagePoint) const = 0;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus projectHomogeneous(
      const Vector4f & point, Vector2f * imagePoint,
      Matrixf<2, 4> * pointJacobian,
      Matrix2Xf * intrinsicsJacobian = nullptr) const = 0;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  virtual ProjectionStatus projectHomogeneousWithExternalParameters(
      const Vector4f & point, const VectorXf & parameters,
      Vector2f * imagePoint,
      Matrixf<2, 4> * pointJacobian = nullptr,
      Matrix2Xf * intrinsicsJacobian = nullptr) const = 0;

  /// \brief Projects points in homogenous coordinates to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in homogeneous coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  virtual void projectHomogeneousBatch(
      const Matrix4Xf & points, Matrix2Xf * imagePoints,
      std::vector<ProjectionStatus> * stati) const = 0;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to backproject points
  /// @{

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The Euclidean direction vector.
  /// @return     true on success.
  virtual bool backProject(const Vector2f & imagePoint,
                           Vector3f * direction) const = 0;

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The Euclidean direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
  /// @return     true on success.
  virtual bool backProject(
      const Vector2f & imagePoint, Vector3f * direction,
      Matrixf<3, 2> * pointJacobian) const = 0;

  /// \brief Back-project 2d image points into Euclidean space (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The Euclidean direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  virtual bool backProjectBatch(const Matrix2Xf & imagePoints,
                                Matrix3Xf * directions,
                                std::vector<bool> * success) const = 0;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The homogeneous point as direction vector.
  /// @return     true on success.
  virtual bool backProjectHomogeneous(const Vector2f & imagePoint,
                                      Vector4f * direction) const = 0;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The homogeneous point as direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function.
  /// @return     true on success.
  virtual bool backProjectHomogeneous(
      const Vector2f & imagePoint, Vector4f * direction,
      Matrixf<4, 2> * pointJacobian) const = 0;

  /// \brief Back-project 2d image points into homogeneous points (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The homogeneous points as direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  virtual bool backProjectHomogeneousBatch(
      const Matrix2Xf & imagePoints, Matrix4Xf * directions,
      std::vector<bool> * success) const = 0;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to facilitate unit testing
  /// @{

  /// \brief Creates a random (uniform distribution) image point.
  /// @return A random image point.
  virtual inline Vector2f createRandomImagePoint() const;

  /// \brief Creates a random visible point in Euclidean coordinates.
  /// @param[in] minDist The minimal distance of this point.
  /// @param[in] maxDist The maximum distance of this point.
  /// @return    A random Euclidean point.
  virtual inline Vector3f createRandomVisiblePoint(float_t minDist = 0.0,
                                                   float_t maxDist = 10.0) const;

  /// \brief Creates a random visible point in homogeneous coordinates.
  /// @param[in] minDist The minimal distance of this point.
  /// @param[in] maxDist The maximum distance of this point.
  /// @return    A random homogeneous point.
  virtual inline Vector4f createRandomVisibleHomogeneousPoint(float_t minDist = 0.0,
                                                              float_t maxDist = 10.0) const;
  /// @}

  /// \brief Obtain the number of intrinsics parameters.
  virtual int numIntrinsicsParameters() const = 0;

  /// \brief Check if the keypoint is in the image.
  inline bool isInImage(const Vector2f& imagePoint) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:

  /// \brief Check if the keypoint is masked.
  inline bool isMasked(const Vector2f& imagePoint) const;
  
  cv::Mat mask_;  ///< The mask -- empty by default

  int imageWidth_;  ///< image width in pixels
  int imageHeight_;  ///< image height in pixels
};

}  // namespace projection
}  // namespace srl

#include "implementation/ProjectionBase.hpp"

#endif /* INCLUDE_SRL_PROJECTION_PROJECTIONBASE_HPP_ */
