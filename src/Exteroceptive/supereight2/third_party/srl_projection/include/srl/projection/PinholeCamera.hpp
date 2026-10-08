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
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file projection/PinholeCamera.hpp
 * @brief Header file for the PinholeCamera class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_SRL_PROJECTION_PINHOLECAMERA_HPP_
#define INCLUDE_SRL_PROJECTION_PINHOLECAMERA_HPP_

#include <vector>
#include <memory>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "srl/projection/ProjectionBase.hpp"
#include "srl/projection/DistortionBase.hpp"
#include "srl/projection/NoDistortion.hpp"

/// \brief Main namespace of this package.
namespace srl {
/// \brief Namespace for camera-related functionality.
namespace projection {

template<class DISTORTION_T>
class PinholeCamera; // forward declaration

/// \class PinholeCameraBase
/// \brief This is an interface for all the different distortion versions, allowing generic undistortion.
class PinholeCameraBase : public ProjectionBase {
 public:
  /// \brief Constructor for width, height and Id
  inline PinholeCameraBase(int imageWidth, int imageHeight)
        : ProjectionBase(imageWidth, imageHeight)
  {
  }

  /// \brief Destructor.
  virtual ~PinholeCameraBase()
  {
  }

  /// \brief Initialise undistort maps to defaults, i.e.
  /// undistortedFocalLengh = 0.5 * (focalLengthU() + focalLengthV()) (same for U and V),
  /// same image dimensions and center in the middle, i.e
  /// undistortedImageCenterU() = 0.5 * imageWith() + 0.5.
  /// \return True on success.
  virtual bool initialiseUndistortMaps() = 0;

  /// \brief Initialise undistort maps, provide custom parameters for the undistorted cam.
  /// @param[in] undistortedImageWidth The width in pixels.
  /// @param[in] undistortedImageHeight The height in pixels.
  /// @param[in] undistortedFocalLengthU The horizontal focal length in pixels.
  /// @param[in] undistortedFocalLengthV The vertical focal length in pixels.
  /// @param[in] undistortedImageCenterU The horizontal centre in pixels.
  /// @param[in] undistortedImageCenterV The vertical centre in pixels.
  /// \return True on success.
  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight,
      float_t undistortedFocalLengthU, float_t undistortedFocalLengthV,
      float_t undistortedImageCenterU, float_t undistortedImageCenterV) = 0;

  /// \brief Get undistorted image -- assumes initialiseUndistortMaps was called
  /// @param[in] srcImg The distorted input image.
  /// @param[out] destImg The undistorted output image.
  /// \return True on success.
  virtual bool undistortImage(const cv::Mat & srcImg, cv::Mat & destImg) const = 0;

  /// \brief Get the model of the undistorted camera.
  /// \return The PinholeCamera without distortion associated with the undistorted image.
  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const = 0;

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  virtual float_t focalLengthU() const = 0;

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  virtual float_t focalLengthV() const = 0;

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  virtual float_t imageCenterU() const = 0;

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  virtual float_t imageCenterV() const = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// \class PinholeCamera<DISTORTION_T>
/// \brief This implements a standard pinhole camera projection model.
/// \tparam DISTORTION_T the distortion type, e.g. okvis::cameras::RadialTangentialDistortion
template<class DISTORTION_T>
class PinholeCamera : public PinholeCameraBase
{
 public:
  typedef DISTORTION_T distortion_t; ///< Makes the distortion type accessible.

  /// \brief Constructor that will figure out the type of distortion
  /// @param[in] imageWidth The width in pixels.
  /// @param[in] imageHeight The height in pixels.
  /// @param[in] focalLengthU The horizontal focal length in pixels.
  /// @param[in] focalLengthV The vertical focal length in pixels.
  /// @param[in] imageCenterU The horizontal centre in pixels.
  /// @param[in] imageCenterV The vertical centre in pixels.
  /// @param[in] distortion The distortion object to be used.
  PinholeCamera(int imageWidth, int imageHeight, float_t focalLengthU,
                float_t focalLengthV, float_t imageCenterU, float_t imageCenterV,
                const distortion_t & distortion);

  /// \brief Destructor.
  virtual ~PinholeCamera()
  {
  }

  static const int NumProjectionIntrinsics = 4;  ///< optimisable projection intrinsics
  static const int NumIntrinsics = NumProjectionIntrinsics
      + distortion_t::NumDistortionIntrinsics; ///< total number of intrinsics

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  virtual float_t focalLengthU() const
  {
    return fu_;
  }

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  virtual float_t focalLengthV() const
  {
    return fv_;
  }

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  virtual float_t imageCenterU() const
  {
    return cu_;
  }

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  virtual float_t imageCenterV() const
  {
    return cv_;
  }

  /// \brief Get the intrinsics as a concatenated vector.
  /// \return The intrinsics as a concatenated vector.
  inline void getIntrinsics(VectorXf & intrinsics) const ;

  /// \brief overwrite all intrinsics - use with caution !
  /// \param[in] intrinsics The intrinsics as a concatenated vector.
  inline bool setIntrinsics(const VectorXf & intrinsics);

  /// \brief Get the total number of intrinsics.
  /// \return Number of intrinsics parameters.
  inline int numIntrinsicsParameters() const
  {
    return NumIntrinsics;
  }

  /// \brief Initialise undistort maps to defaults, i.e.
  /// undistortedFocalLengh = 0.5 * (focalLengthU() + focalLengthV()) (same for U and V),
  /// same image dimensions and center in the middle, i.e
  /// undistortedImageCenterU() = 0.5 * imageWith() + 0.5.
  /// \return True on success.
  virtual bool initialiseUndistortMaps();

  /// \brief Initialise undistort maps, provide custom parameters for the undistorted cam.
  /// @param[in] undistortedImageWidth The width in pixels.
  /// @param[in] undistortedImageHeight The height in pixels.
  /// @param[in] undistortedFocalLengthU The horizontal focal length in pixels.
  /// @param[in] undistortedFocalLengthV The vertical focal length in pixels.
  /// @param[in] undistortedImageCenterU The horizontal centre in pixels.
  /// @param[in] undistortedImageCenterV The vertical centre in pixels.
  /// \return True on success.
  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight,
      float_t undistortedFocalLengthU, float_t undistortedFocalLengthV,
      float_t undistortedImageCenterU, float_t undistortedImageCenterV);

  /// \brief Get the model of the undistorted camera.
  /// \return The PinholeCamera without distortion associated with the undistorted image.
  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const;

  /// \brief Get undistorted image -- assumes initialiseUndistortMaps was called
  /// @param[in] srcImg The distorted input image.
  /// @param[out] destImg The undistorted output image.
  /// \return True on success.
  virtual bool undistortImage(const cv::Mat & srcImg, cv::Mat & destImg) const;

  //////////////////////////////////////////////////////////////
  /// \name Methods to project points
  /// @{

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Euclidean coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus project(
      const Vector3f & point, Vector2f * imagePoint) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus project(
      const Vector3f & point, Vector2f * imagePoint,
      Matrixf<2, 3> * pointJacobian,
      Matrix2Xf * intrinsicsJacobian = nullptr) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectWithExternalParameters(
      const Vector3f & point, const VectorXf & parameters,
      Vector2f * imagePoint, Matrixf<2, 3> * pointJacobian,
      Matrix2Xf * intrinsicsJacobian = nullptr) const;

  /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in Euclidean coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectBatch(
      const Matrix3Xf & points, Matrix2Xf * imagePoints,
      std::vector<ProjectionStatus> * stati) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Homogeneous coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneous(
      const Vector4f & point, Vector2f * imagePoint) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneous(
      const Vector4f & point, Vector2f * imagePoint,
      Matrixf<2, 4> * pointJacobian,
      Matrix2Xf * intrinsicsJacobian = nullptr) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneousWithExternalParameters(
      const Vector4f & point, const VectorXf & parameters,
      Vector2f * imagePoint,
      Matrixf<2, 4> * pointJacobian = nullptr,
      Matrix2Xf * intrinsicsJacobian = nullptr) const;

  /// \brief Projects points in homogenous coordinates to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in homogeneous coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectHomogeneousBatch(
      const Matrix4Xf & points, Matrix2Xf * imagePoints,
      std::vector<ProjectionStatus> * stati) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to backproject points
  /// @{

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The Euclidean direction vector.
  /// @return     true on success.
  inline bool backProject(const Vector2f & imagePoint,
                          Vector3f * direction) const;

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The Euclidean direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
  /// @return     true on success.
  inline bool backProject(const Vector2f & imagePoint,
                          Vector3f * direction,
                          Matrixf<3, 2> * pointJacobian) const;

  /// \brief Back-project 2d image points into Euclidean space (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The Euclidean direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectBatch(const Matrix2Xf & imagePoints,
                               Matrix3Xf * directions,
                               std::vector<bool> * success) const;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The homogeneous point as direction vector.
  /// @return     true on success.
  inline bool backProjectHomogeneous(const Vector2f & imagePoint,
                                     Vector4f * direction) const;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The homogeneous point as direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function.
  /// @return     true on success.
  inline bool backProjectHomogeneous(
      const Vector2f & imagePoint, Vector4f * direction,
      Matrixf<4, 2> * pointJacobian) const;

  /// \brief Back-project 2d image points into homogeneous points (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The homogeneous points as direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectHomogeneousBatch(const Matrix2Xf & imagePoints,
                                          Matrix4Xf * directions,
                                          std::vector<bool> * success) const;
  /// @}

  /// \brief get a test instance
  static std::shared_ptr<ProjectionBase> createTestObject()
  {
    return std::shared_ptr<ProjectionBase>(new PinholeCamera(752, 480, 350, 360, 378, 238,
                         distortion_t::testObject()));
  }
  /// \brief get a test instance
  static PinholeCamera testObject()
  {
    return PinholeCamera(752, 480, 350, 360, 378, 238,
                          distortion_t::testObject());
  }

  /// \brief Obtain the projection type
  std::string type() const
  {
    return "PinholeCamera<" + distortion_.type() + ">";
  }

  /// \brief Obtain the projection type
  const std::string distortionType() const
  {
    return distortion_.type();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:

  /// \brief No default constructor.
  PinholeCamera() = delete;

  distortion_t distortion_;  ///< the distortion to be used

  Matrixf<NumIntrinsics, 1> intrinsics_;  ///< summary of all intrinsics parameters
  float_t fu_;  ///< focalLengthU
  float_t fv_;  ///< focalLengthV
  float_t cu_;  ///< imageCenterU
  float_t cv_;  ///< imageCenterV
  float_t one_over_fu_;  ///< 1.0 / fu_
  float_t one_over_fv_;  ///< 1.0 / fv_
  float_t fu_over_fv_;  ///< fu_ / fv_

  cv::Mat map_x_fast_; ///< OpenCV undistort fast map x-coordinates
  cv::Mat map_y_fast_; ///< OpenCV undistort fast map x-coordinates

  int undistortedImageWidth_ = 0;  ///< undistortedImageWidth The width in pixels.
  int undistortedImageHeight_ = 0;  ///< undistortedImageHeight The height in pixels.
  float_t undistortedFocalLengthU_ = 0.0;  ///< undistortedFocalLengthU The horizontal focal length in pixels.
  float_t undistortedFocalLengthV_ = 0.0;  ///< undistortedFocalLengthV The vertical focal length in pixels.
  float_t undistortedImageCenterU_ = 0.0;  ///< undistortedImageCenterU The horizontal centre in pixels.
  float_t undistortedImageCenterV_ = 0.0;  ///< undistortedImageCenterV The vertical centre in pixels.

};

}  // namespace projection
}  // namespace srl

#include "implementation/PinholeCamera.hpp"

#endif /* INCLUDE_SRL_PROJECTION_PINHOLECAMERA_HPP_ */
