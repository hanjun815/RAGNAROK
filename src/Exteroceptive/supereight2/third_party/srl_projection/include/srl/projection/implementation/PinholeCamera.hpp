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
 * @file implementation/PinholeCamera.hpp
 * @brief Header implementation file for the PinholeCamera class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <Eigen/Geometry>

// \brief Main namespace of this package.
namespace srl {
// \brief Namespace for camera-related functionality.
namespace projection {

template<class DISTORTION_T>
PinholeCamera<DISTORTION_T>::PinholeCamera(int imageWidth,
                                           int imageHeight,
                                           float_t focalLengthU,
                                           float_t focalLengthV,
                                           float_t imageCenterU,
                                           float_t imageCenterV,
                                           const distortion_t & distortion)
    : PinholeCameraBase(imageWidth, imageHeight),
    distortion_(distortion),
    fu_(focalLengthU),
    fv_(focalLengthV),
    cu_(imageCenterU),
    cv_(imageCenterV)
{
  intrinsics_[0] = fu_;  //< focalLengthU
  intrinsics_[1] = fv_;  //< focalLengthV
  intrinsics_[2] = cu_;  //< imageCenterU
  intrinsics_[3] = cv_;  //< imageCenterV
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
}

// overwrite all intrinsics - use with caution !
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::setIntrinsics(
    const VectorXf & intrinsics)
{
  if (intrinsics.cols() != NumIntrinsics) {
    return false;
  }
  intrinsics_ = intrinsics;
  fu_ = intrinsics[0];  //< focalLengthU
  fv_ = intrinsics[1];  //< focalLengthV
  cu_ = intrinsics[2];  //< imageCenterU
  cv_ = intrinsics[3];  //< imageCenterV
  distortion_.setParameters(
      intrinsics.tail<distortion_t::NumDistortionIntrinsics>());
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
  return true;
}

// Initialise undistort maps to defaults
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::initialiseUndistortMaps() {
  const float_t f = (fu_ + fv_) * 0.5;
  return initialiseUndistortMaps(imageWidth(), imageHeight(), f, f,
      float_t(imageWidth())*0.5+0.5, float_t(imageHeight())*0.5+0.5);
}

// Initialise undistort maps, provide custom parameters for the undistorted cam.
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::initialiseUndistortMaps(
    int undistortedImageWidth, int undistortedImageHeight,
    float_t undistortedFocalLengthU, float_t undistortedFocalLengthV,
    float_t undistortedImageCenterU, float_t undistortedImageCenterV) {

  // store parameters
  undistortedImageWidth_ = undistortedImageWidth;
  undistortedImageHeight_ = undistortedImageHeight;
  undistortedFocalLengthU_ = undistortedFocalLengthU;
  undistortedFocalLengthV_ = undistortedFocalLengthV;
  undistortedImageCenterU_ = undistortedImageCenterU;
  undistortedImageCenterV_ = undistortedImageCenterV;

  // some preparation of the actual and undistorted projections
  Matrix2f undistortedK_inv, actualK;
  undistortedK_inv << 1.0/undistortedFocalLengthU, 0.0, 0.0, 1.0/undistortedFocalLengthV;
  actualK << fu_, 0.0, 0.0, fv_;
  Vector2f actualCenter(cu_, cv_);
  Vector2f undistortedCenter(undistortedImageCenterU, undistortedImageCenterV);

  // Create the maps and vectors for points
  cv::Mat map_x(undistortedImageHeight, undistortedImageWidth, CV_32F);
  cv::Mat map_y(undistortedImageHeight, undistortedImageWidth, CV_32F);
  Vector2f pixel, imgPlane, projectedPoint, distortedPoint, mappedPixel;
  Vector3f ray, rayTransformed;
  Vector4f hrayTransformed;
  const float_t rayLength = 0.25;

  for (int y = 0; y < undistortedImageHeight; y++) {

    pixel(1) = y;

    float *pmap_x = map_x.ptr<float>(y); // for the yth row in the map
    float *pmap_y = map_y.ptr<float>(y);

    for (int x = 0; x < undistortedImageWidth; x++) {

      pixel(0) = x;

      // Convert from pixels to image plane using ideal camera intrinsics
      imgPlane = undistortedK_inv * (pixel - undistortedCenter);

      // Shoot a ray through the (x,y) point
      ray = rayLength * imgPlane.homogeneous();

      // Transform that ray into the frame of the actual camera
      hrayTransformed = /*T_LC * */ ray.homogeneous();
      rayTransformed = hrayTransformed.hnormalized();

      // Project the transformed ray into the actual camera
      projectedPoint = rayTransformed.hnormalized();

      // Apply the distortion model to the projection
      distortion_.distort(projectedPoint,&distortedPoint);

      // Apply the intrinsics model to get a pixel location
      mappedPixel = (actualK * distortedPoint) + actualCenter;

      // Assign that pixel location to the map
      pmap_x[x] = mappedPixel(0); // assign a value to the (x,y) position in the map
      pmap_y[x] = mappedPixel(1);

    }
  }

  // Apply convertMaps for actual fast remapping later when calling undistortImage
  cv::convertMaps(map_x, map_y, map_x_fast_, map_y_fast_, CV_16SC2);

  return true;
}

// Get the model of the undistorted camera.
template<class DISTORTION_T>
PinholeCamera<NoDistortion> PinholeCamera<DISTORTION_T>::undistortedPinholeCamera()
    const {
  assert(map_x_fast_.cols !=0);
  return PinholeCamera<NoDistortion>(undistortedImageWidth_, undistortedImageHeight_,
      undistortedFocalLengthU_, undistortedFocalLengthV_,
      undistortedImageCenterU_, undistortedImageCenterV_,
      NoDistortion());
}

// Get undistorted image -- assumes initialiseUndistortMaps was called
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::undistortImage(const cv::Mat & srcImg,
                                                 cv::Mat & destImg) const {
  cv::remap(srcImg, destImg, map_x_fast_, map_y_fast_,
      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::getIntrinsics(VectorXf & intrinsics) const
  {
    intrinsics = intrinsics_;
    VectorXf distortionIntrinsics;
    if(distortion_t::NumDistortionIntrinsics > 0) {
      distortion_.getParameters(distortionIntrinsics);
      intrinsics.tail<distortion_t::NumDistortionIntrinsics>() = distortionIntrinsics;
    }
  }

//////////////////////////////////////////
// Methods to project points

// Projects a Euclidean point to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::project(
    const Vector3f & point, Vector2f * imagePoint) const
{
  // handle singularity
  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // projection
  Vector2f imagePointUndistorted;
  const float_t rz = 1.0 / point[2];
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  // distortion
  Vector2f imagePoint2;
  if (!distortion_.distort(imagePointUndistorted, &imagePoint2)) {
    return ProjectionStatus::Invalid;
  }

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!ProjectionBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (ProjectionBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

// Projects a Euclidean point to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::project(
    const Vector3f & point, Vector2f * imagePoint,
    Matrixf<2, 3> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  // handle singularity
  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // projection
  Vector2f imagePointUndistorted;
  const float_t rz = 1.0 / point[2];
  float_t rz2 = rz * rz;
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  Matrixf<2, 3> pointJacobianProjection;
  Matrix2Xf intrinsicsJacobianProjection;
  Matrix2f distortionJacobian;
  Matrix2Xf intrinsicsJacobianDistortion;
  Vector2f imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {
    // get both Jacobians
    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
                                            &distortionJacobian,
                                            &intrinsicsJacobianDistortion);
    // compute the intrinsics Jacobian
    intrinsicsJacobian->template topLeftCorner<2, 2>() =
        imagePoint2.asDiagonal();
    intrinsicsJacobian->template block<2, 2>(0,2) = Matrix2f::Identity();

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian
          ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Vector2f(fu_, fv_).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
    }
  } else {
    // only get point Jacobian
    distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
                                            &distortionJacobian);
  }

  // compute the point Jacobian in any case
  Matrixf<2, 3> & J = *pointJacobian;
  J(0, 0) = fu_ * distortionJacobian(0, 0) * rz;
  J(0, 1) = fu_ * distortionJacobian(0, 1) * rz;
  J(0, 2) = -fu_
      * (point[0] * distortionJacobian(0, 0)
          + point[1] * distortionJacobian(0, 1)) * rz2;
  J(1, 0) = fv_ * distortionJacobian(1, 0) * rz;
  J(1, 1) = fv_ * distortionJacobian(1, 1) * rz;
  J(1, 2) = -fv_
      * (point[0] * distortionJacobian(1, 0)
          + point[1] * distortionJacobian(1, 1)) * rz2;

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!distortionSuccess) {
    return ProjectionStatus::Invalid;
  }
  if (!ProjectionBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (ProjectionBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

// Projects a Euclidean point to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectWithExternalParameters(
    const Vector3f & point, const VectorXf & parameters,
    Vector2f * imagePoint, Matrixf<2, 3> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  // handle singularity
  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // parse parameters into human readable form
  const float_t fu = parameters[0];
  const float_t fv = parameters[1];
  const float_t cu = parameters[2];
  const float_t cv = parameters[3];

  VectorXf distortionParameters;
  if (distortion_t::NumDistortionIntrinsics > 0) {
    distortionParameters = parameters
        .template tail<distortion_t::NumDistortionIntrinsics>();
  }

  // projection
  Vector2f imagePointUndistorted;
  const float_t rz = 1.0 / point[2];
  float_t rz2 = rz * rz;
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  Matrixf<2, 3> pointJacobianProjection;
  Matrix2Xf intrinsicsJacobianProjection;
  Matrix2f distortionJacobian;
  Matrix2Xf intrinsicsJacobianDistortion;
  Vector2f imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {
    // get both Jacobians
    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
                                            distortionParameters, &imagePoint2,
                                            &distortionJacobian,
                                            &intrinsicsJacobianDistortion);
    // compute the intrinsics Jacobian
    intrinsicsJacobian->template topLeftCorner<2, 2>() =
        imagePoint2.asDiagonal();
    intrinsicsJacobian->template block<2, 2>(0,2) = Matrix2f::Identity();

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian
          ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Vector2f(fu, fv).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
    }
  } else {
    // only get point Jacobian
    distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
                                            distortionParameters, &imagePoint2,
                                            &distortionJacobian);
  }

  // compute the point Jacobian, if requested
  if(pointJacobian) {
    Matrixf<2, 3> & J = *pointJacobian;
    J(0, 0) = fu * distortionJacobian(0, 0) * rz;
    J(0, 1) = fu * distortionJacobian(0, 1) * rz;
    J(0, 2) = -fu
        * (point[0] * distortionJacobian(0, 0)
            + point[1] * distortionJacobian(0, 1)) * rz2;
    J(1, 0) = fv * distortionJacobian(1, 0) * rz;
    J(1, 1) = fv * distortionJacobian(1, 1) * rz;
    J(1, 2) = -fv
        * (point[0] * distortionJacobian(1, 0)
            + point[1] * distortionJacobian(1, 1)) * rz2;
  }

  // scale and offset
  (*imagePoint)[0] = fu * imagePoint2[0] + cu;
  (*imagePoint)[1] = fv * imagePoint2[1] + cv;

  if (!distortionSuccess) {
    return ProjectionStatus::Invalid;
  }
  if (!ProjectionBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (ProjectionBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

// Projects Euclidean points to 2d image points (projection) in a batch.
template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::projectBatch(
    const Matrix3Xf & points, Matrix2Xf * imagePoints,
    std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Vector3f point = points.col(i);
    Vector2f imagePoint;
    ProjectionStatus status = project(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if(stati)
      stati->push_back(status);
  }
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneous(
    const Vector4f & point, Vector2f * imagePoint) const
{
  Vector3f head = point.head<3>();
  if (point[3] < 0) {
    return project(-head, imagePoint);
  } else {
    return project(head, imagePoint);
  }
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneous(
    const Vector4f & point, Vector2f * imagePoint,
    Matrixf<2, 4> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  Vector3f head = point.head<3>();
  Matrixf<2, 3> pointJacobian3;
  ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  } else {
    status = project(head, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Vector2f::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneousWithExternalParameters(
    const Vector4f & point, const VectorXf & parameters,
    Vector2f * imagePoint, Matrixf<2, 4> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  Vector3f head = point.head<3>();
  Matrixf<2, 3> pointJacobian3;
  ProjectionStatus status;
  if (point[3] < 0) {
    status = projectWithExternalParameters(-head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  } else {
    status = projectWithExternalParameters(head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Vector2f::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

// Projects points in homogenous coordinates to 2d image points (projection) in a batch.
template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::projectHomogeneousBatch(
    const Matrix4Xf & points, Matrix2Xf * imagePoints,
    std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Vector4f point = points.col(i);
    Vector2f imagePoint;
    ProjectionStatus status = projectHomogeneous(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if(stati)
      stati->push_back(status);
  }
}

//////////////////////////////////////////
// Methods to backproject points

// Back-project a 2d image point into Euclidean space (direction vector).
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProject(
    const Vector2f & imagePoint, Vector3f * direction) const
{
  // unscale and center
  Vector2f imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  // undistort
  Vector2f undistortedImagePoint;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint);

  // project 1 into z direction
  (*direction)[0] = undistortedImagePoint[0];
  (*direction)[1] = undistortedImagePoint[1];
  (*direction)[2] = 1.0;

  return success;
}

// Back-project a 2d image point into Euclidean space (direction vector).
template<class DISTORTION_T>
inline bool PinholeCamera<DISTORTION_T>::backProject(
    const Vector2f & imagePoint, Vector3f * direction,
    Matrixf<3, 2> * pointJacobian) const
{
  // unscale and center
  Vector2f imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  // undistort
  Vector2f undistortedImagePoint;
  Matrix2f pointJacobianUndistortion;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint,
                                       &pointJacobianUndistortion);

  // project 1 into z direction
  (*direction)[0] = undistortedImagePoint[0];
  (*direction)[1] = undistortedImagePoint[1];
  (*direction)[2] = 1.0;

  // Jacobian w.r.t. imagePoint
  Matrixf<3, 2> outProjectJacobian =
      Matrixf<3, 2>::Zero();
  outProjectJacobian(0, 0) = one_over_fu_;
  outProjectJacobian(1, 1) = one_over_fv_;

  (*pointJacobian) = outProjectJacobian * pointJacobianUndistortion;  // chain rule

  return success;
}

// Back-project 2d image points into Euclidean space (direction vectors).
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectBatch(
    const Matrix2Xf & imagePoints, Matrix3Xf * directions,
    std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  for (int i = 0; i < numPoints; ++i) {
    Vector2f imagePoint = imagePoints.col(i);
    Vector3f point;
    bool suc = backProject(imagePoint, &point);
    if(success)
      success->push_back(suc);
    directions->col(i) = point;
  }
  return true;
}

// Back-project a 2d image point into homogeneous point (direction vector).
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneous(
    const Vector2f & imagePoint, Vector4f * direction) const
{
  Vector3f ray;
  bool success = backProject(imagePoint, &ray);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  // arbitrary
  return success;
}

// Back-project a 2d image point into homogeneous point (direction vector).
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneous(
    const Vector2f & imagePoint, Vector4f * direction,
    Matrixf<4, 2> * pointJacobian) const
{
  Vector3f ray;
  Matrixf<3, 2> pointJacobian3;
  bool success = backProject(imagePoint, &ray, &pointJacobian3);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  // arbitrary
  pointJacobian->template bottomRightCorner<1,2>() = Vector2f::Zero();
  pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
  return success;
}

// Back-project 2d image points into homogeneous points (direction vectors).
template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneousBatch(
    const Matrix2Xf & imagePoints, Matrix4Xf * directions,
    std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  directions->row(3) = VectorXf::Ones(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    Vector2f imagePoint = imagePoints.col(i);
    Vector3f point;
    bool suc = backProject(imagePoint, &point);
    if(success)
      success->push_back(suc);
    directions->template block<3, 1>(0, i) = point;
  }
  return true;
}

}  // namespace projection
}  // namespace srl
