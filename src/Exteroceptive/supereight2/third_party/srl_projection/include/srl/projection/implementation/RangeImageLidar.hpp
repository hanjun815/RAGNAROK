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
 *  Created on: March, 2020
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file implementation/RangeImageLidar.hpp
 * @brief Header implementation file for the RangeImageLidar class.
 * @author Stefan Leutenegger
 */

#include <Eigen/Geometry>

// \brief Main namespace of this package.
namespace srl {
// \brief Namespace for camera-related functionality.
namespace projection {

RangeImageLidar::RangeImageLidar(int imageWidth, int imageHeight, const VectorXf & beamAzimuthAngles,
                         const VectorXf & beamElevationAngles)
    : ProjectionBase(imageWidth, imageHeight),
      beamAzimuthAngles_(beamAzimuthAngles), beamElevationAngles_(beamElevationAngles)
{
}

//////////////////////////////////////////
// Methods to project points

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus RangeImageLidar::project(
    const Vector3f & point, Vector2f * imagePoint) const
{
  // handle singularity
  if (point.norm() < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // compute azimuth and elevation angles (projection) [rad]
  const float_t R = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
  const float_t azimuth = (2.0 * M_PI - std::atan2(point[1], point[0])) * 360.0 / (2 * M_PI);
  const float_t elevation = std::asin(point[2]/R) * 360.0 / (2 * M_PI);

  // check bounds
  if(elevation > beamElevationAngles_[0]) {
    (*imagePoint)[0] = (azimuth - beamAzimuthAngles_[0])/360.0 * imageWidth_;
    // azimuthal wrap-around
    if((*imagePoint)[0]<-0.5) {
      (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
    }
    if((*imagePoint)[0]>imageWidth_-0.5) {
      (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
    }
    (*imagePoint)[1] = -1;
    return ProjectionStatus::OutsideImage;
  }
  if(elevation < beamElevationAngles_[beamElevationAngles_.rows()-1]) {
    (*imagePoint)[0] = (azimuth - beamAzimuthAngles_[beamElevationAngles_.rows()-1])/360.0 * imageWidth_;
    // azimuthal wrap-around
    if((*imagePoint)[0]<-0.5) {
      (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
    }
    if((*imagePoint)[0]>imageWidth_-0.5) {
      (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
    }
    (*imagePoint)[1] = imageHeight_;
    return ProjectionStatus::OutsideImage;
  }
  // find the right row
  /// \todo make this more elegant with binary search
  bool found = false;
  int i = 1;
  for(; i < beamElevationAngles_.rows(); ++i) {
    if(float_t(beamElevationAngles_[i]) < elevation) {
      found = true;
      break;
    }
  }
  if(!found) {
    return ProjectionStatus::OutsideImage;
  }

  // for interpolation
  const float_t r = (elevation-beamElevationAngles_[i])
      /(beamElevationAngles_[i-1] - beamElevationAngles_[i]);
  const float_t azimuthOffset = (1.0 - r) * beamAzimuthAngles_[i] + r * beamAzimuthAngles_[i-1];

  // scale and offset
  (*imagePoint)[0] = (azimuth - azimuthOffset)/360.0 * imageWidth_;
  (*imagePoint)[1] = i - r;

  // azimuthal wrap-around
  if((*imagePoint)[0]<-0.5) {
    (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
  }
  if((*imagePoint)[0]>imageWidth_-0.5) {
    (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
  }

  // checks
  if (ProjectionBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  return ProjectionStatus::Successful;
}

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus RangeImageLidar::project(
    const Vector3f & point, Vector2f * imagePoint,
    Matrixf<2, 3> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  // handle singularity
  if (point.norm() < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // compute azimuth and elevation angles (projection) [rad]
  const float_t R = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
  const float_t azimuth = (2.0 * M_PI - std::atan2(point[1], point[0])) * 360.0 / (2 * M_PI);
  const float_t elevation = std::asin(point[2]/R) * 360.0 / (2 * M_PI);

  // check bounds
  if(elevation > beamElevationAngles_[0]) {
    return ProjectionStatus::OutsideImage;
  }
  if(elevation < beamElevationAngles_[beamElevationAngles_.rows()-1]) {
    return ProjectionStatus::OutsideImage;
  }
  // find the right row
  /// \todo make this more elegant with binary search
  bool found = false;
  int i = 1;
  for(; i < beamElevationAngles_.rows(); ++i) {
    if(float_t(beamElevationAngles_[i]) < elevation) {
      found = true;
      break;
    }
  }
  if(!found) {
    return ProjectionStatus::OutsideImage;
  }

  // for interpolation
  const float_t scalingElevation = 1.0/(beamElevationAngles_[i-1] - beamElevationAngles_[i]);
  const float_t r = (elevation-beamElevationAngles_[i]) * scalingElevation;
  const float_t azimuthOffset = (1.0 - r) * beamAzimuthAngles_[i] + r * beamAzimuthAngles_[i-1];

  // scale and offset
  const float_t scalingAzimuth = imageWidth_/360.0;
  (*imagePoint)[0] = (azimuth - azimuthOffset) * scalingAzimuth;
  (*imagePoint)[1] = i - r;

  // azimuthal wrap-around
  if((*imagePoint)[0]<-0.5) {
    (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
  }
  if((*imagePoint)[0]>imageWidth_-0.5) {
    (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
  }

  // Jacobians
  if(pointJacobian) {
    pointJacobian->setZero();
    const float_t R2 = point[0]*point[0] + point[1]*point[1];
    const float_t sqrt_R2 = sqrt(R2);
    const float_t D2 = R2 + point[2]*point[2];
    //(*pointJacobian)(0,0) = -(point[1] * scalingAzimuth)/R2;
    //(*pointJacobian)(0,1) = (point[0] * scalingAzimuth)/R2;
    //(*pointJacobian)(0,2) = 0.0; /// \todo
    //(*pointJacobian)(1,0) = -scalingElevation*(point[0]*point[2])/(sqrt_R2*D2);
    //(*pointJacobian)(1,1) = -scalingElevation*(point[1]*point[2])/(sqrt_R2*D2);
    //(*pointJacobian)(1,2) = scalingElevation*sqrt_R2/D2;
    (*pointJacobian)(0,0) = imageWidth_*((point[1]*180.0)/(M_PI*R2)+(beamAzimuthAngles_[i]*point[0]*point[2]/sqrt_R2*180.0)/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2)-(beamAzimuthAngles_[i-1]*point[0]*point[2]*1.0/sqrt_R2*180.0)/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2))*(1.0/360.0);
    (*pointJacobian)(0,1) = imageWidth_*((point[0]*180.0)/(M_PI*R2)-(beamAzimuthAngles_[i]*point[1]*point[2]/sqrt_R2*180.0)/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2)+(beamAzimuthAngles_[i-1]*point[1]*point[2]*1.0/sqrt_R2*180.0)/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2))*(-1.0/360.0);
    (*pointJacobian)(0,2) = (imageWidth_*sqrt_R2*(beamAzimuthAngles_[i]-beamAzimuthAngles_[i-1])*(-0.5))/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2);
    (*pointJacobian)(1,0) = (point[0]*point[2]*1.0/sqrt_R2*(-180.0))/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2);
    (*pointJacobian)(1,1) = (point[1]*point[2]*1.0/sqrt_R2*(-180.0))/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2);
    (*pointJacobian)(1,2) = (180.0*sqrt_R2)/(M_PI*(beamElevationAngles_[i]-beamElevationAngles_[i-1])*D2);

  }
  if(intrinsicsJacobian) {
    throw std::runtime_error("RangeImageLidar does not support intrinsics Jacobian");
  }

  // checks
  if (ProjectionBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  return ProjectionStatus::Successful;
}

ProjectionStatus RangeImageLidar::projectSphere(
    const Vector3f& center,
    float radius,
    Vector2f* imageCenter,
    float& imageRadius) const
{
  // handle singularity
  if (center.norm() < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  // compute azimuth and elevation angles (projection) [rad]
  const float_t R = sqrt(center[0]*center[0] + center[1]*center[1] + center[2]*center[2]);
  const float_t azimuth = (2.0 * M_PI - std::atan2(center[1], center[0])) * 360.0 / (2 * M_PI);
  const float_t elevation = std::asin(center[2]/R) * 360.0 / (2 * M_PI);

  // check bounds
  if(elevation > beamElevationAngles_[0]) {
    return ProjectionStatus::OutsideImage;
  }
  if(elevation < beamElevationAngles_[beamElevationAngles_.rows()-1]) {
    return ProjectionStatus::OutsideImage;
  }
  // find the right row
  /// \todo make this more elegant with binary search
  bool found = false;
  int i = 1;
  for(; i < beamElevationAngles_.rows(); ++i) {
    if(float_t(beamElevationAngles_[i]) < elevation) {
      found = true;
      break;
    }
  }
  if(!found) {
    return ProjectionStatus::OutsideImage;
  }

  // for interpolation
  const float_t r = (elevation-beamElevationAngles_[i])
      /(beamElevationAngles_[i-1] - beamElevationAngles_[i]);
  const float_t azimuthOffset = (1.0 - r) * beamAzimuthAngles_[i] + r * beamAzimuthAngles_[i-1];

  // scale and offset
  (*imageCenter)[0] = (azimuth - azimuthOffset)/360.0 * imageWidth_;
  (*imageCenter)[1] = i - r;

  // azimuthal wrap-around
  if((*imageCenter)[0]<-0.5) {
    (*imageCenter)[0] = (*imageCenter)[0] + imageWidth_;
  }
  if((*imageCenter)[0]>imageWidth_-0.5) {
    (*imageCenter)[0] = (*imageCenter)[0] - imageWidth_;
  }

  // checks
  if (ProjectionBase::isMasked(*imageCenter)) {
    return ProjectionStatus::Masked;
  }

  // Compute the projected radius
  // Sine of the angle between the ray through the sphere's centre and a ray tangent to it
  const float_t sinTheta = radius / center.norm();
  if (sinTheta > 1) {
    // The sensor is inside the sphere
    return ProjectionStatus::OutsideImage;
  }
  // Compute radian to degree conversion once
  static const float_t radToDeg = 180 / M_PI;
  // Get the angle in degrees from the sine
  const float_t azimuthRightOffset = radToDeg * std::asin(sinTheta);
  // NOTE asin ~= x + x^3 / 6
  //const float_t azimuthRightOffset = radToDeg * (sinTheta + sinTheta * sinTheta * sinTheta / 6);
  // Azimuth angle of a ray tangent to the right side of the sphere
  const float_t azimuthRight = azimuth + azimuthRightOffset - azimuthOffset;
  // Horizontal pixel coordinate of the pixel at the right edge of the projected circle
  float_t uRadius = azimuthRight / 360.0 * imageWidth_;
  // azimuthal wrap-around
  if (uRadius < -0.5) {
    uRadius = uRadius + imageWidth_;
  }
  if (uRadius > imageWidth_ - 0.5) {
    uRadius = uRadius - imageWidth_;
  }
  // Comput the horizontal pixel distane with wrap-around
  imageRadius = uRadius - (*imageCenter)[0];
  if (imageRadius < 0) {
    imageRadius += imageWidth_;
  }

  return ProjectionStatus::Successful;
}


// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus RangeImageLidar::projectWithExternalParameters(
    const Vector3f &, const VectorXf &,
    Vector2f *, Matrixf<2, 3> *,
    Matrix2Xf *) const
{
  throw std::runtime_error("external parameters projection for RangeImageLidar not implemented");
  return ProjectionStatus::Invalid;
}

// Projects Euclidean points to 2d image points (projection) in a batch.
void RangeImageLidar::projectBatch(
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
ProjectionStatus RangeImageLidar::projectHomogeneous(
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
ProjectionStatus RangeImageLidar::projectHomogeneous(
    const Vector4f & point, Vector2f * imagePoint,
    Matrixf<2, 4> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
  Vector3f head = point.head<3>();
  Matrixf<2, 3> pointJacobian3;
  ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  } else {
    status = project(head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Vector2f::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
ProjectionStatus RangeImageLidar::projectHomogeneousWithExternalParameters(
    const Vector4f &, const VectorXf &,
    Vector2f *, Matrixf<2, 4> *,
    Matrix2Xf *) const
{
  throw std::runtime_error("intrinsics Jacobian for RangeImageLidar not implemented");
  return ProjectionStatus::Invalid;
}

// Projects points in homogenous coordinates to 2d image points (projection) in a batch.
void RangeImageLidar::projectHomogeneousBatch(
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
bool RangeImageLidar::backProject(
    const Vector2f & imagePoint, Vector3f * direction) const
{
  // adapted from
  // https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/os1_util.cpp

  // compute elevation and azimuth angles from pixel coordinates with interpolation
  const uint32_t v0 = std::floor(imagePoint[1]);
  const uint32_t v1 = std::ceil(imagePoint[1]);
  const float_t r = imagePoint[1] - v0;

  // azimuth angle with interpolated offset
  const float_t azimuth0 = 2.0 * M_PI * imagePoint[0] / float_t(imageWidth_);
  const float_t azimuth = ((1.0-r) * beamAzimuthAngles_[v0] +  r * beamAzimuthAngles_[v1])
      * 2.0 * M_PI / 360.0 + azimuth0;

  // interpolate elevation angle
  float_t elevation = ((1.0-r) * beamElevationAngles_[v0] + r * beamElevationAngles_[v1])
      * 2.0 * M_PI / 360.0;

  // project as ray
  (*direction)[0] = std::cos(elevation) * std::cos(azimuth);
  (*direction)[1] = -std::cos(elevation) * std::sin(azimuth);
  (*direction)[2] = std::sin(elevation);

  return true;
}

// Back-project a 2d image point into Euclidean space (direction vector).
inline bool RangeImageLidar::backProject(
    const Vector2f & imagePoint, Vector3f * direction,
    Matrixf<3, 2> * pointJacobian) const
{
  // adapted from
  // https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/os1_util.cpp

  // compute elevation and azimuth angles from pixel coordinates with interpolation
  const uint32_t v0 = std::floor(imagePoint[1]);
  const uint32_t v1 = std::ceil(imagePoint[1]);
  const float_t r = imagePoint[1] - v0;

  // azimuth angle with interpolated offset
  const float_t azimuth0 = 2.0 * M_PI * imagePoint[0] / float_t(imageWidth_);
  const float_t azimuth = ((1.0-r) * beamAzimuthAngles_[v0] +  r * beamAzimuthAngles_[v1])
      * 2.0 * M_PI / 360.0 + azimuth0;

  // interpolate elevation angle
  float_t elevation = ((1.0-r) * beamElevationAngles_[v0] + r * beamElevationAngles_[v1])
      * 2.0 * M_PI / 360.0;

  // project as ray
  (*direction)[0] = std::cos(elevation) * std::cos(azimuth);
  (*direction)[1] = -std::cos(elevation) * std::sin(azimuth);
  (*direction)[2] = std::sin(elevation);

  // Jacobian w.r.t. imagePoint
  pointJacobian->setZero(); /// \todo

  return true;
}

// Back-project 2d image points into Euclidean space (direction vectors).
bool RangeImageLidar::backProjectBatch(
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
bool RangeImageLidar::backProjectHomogeneous(
    const Vector2f & imagePoint, Vector4f * direction) const
{
  Vector3f ray;
  bool success = backProject(imagePoint, &ray);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  // arbitrary
  return success;
}

// Back-project a 2d image point into homogeneous point (direction vector).
bool RangeImageLidar::backProjectHomogeneous(
    const Vector2f & imagePoint, Vector4f * direction,
    Matrixf<4, 2> * pointJacobian) const
{
  Vector3f ray;
  Matrixf<3, 2> pointJacobian3;
  bool success = backProject(imagePoint, &ray, &pointJacobian3);
  direction->template head<3>() = ray;
  (*direction)[4] = 1.0;  // arbitrary
  pointJacobian->template bottomRightCorner<1,2>() = Vector2f::Zero();
  pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
  return success;
}

// Back-project 2d image points into homogeneous points (direction vectors).
bool RangeImageLidar::backProjectHomogeneousBatch(
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

VectorXf RangeImageLidar::beamAzimuthAngles() const
{
  return beamAzimuthAngles_;
}

void RangeImageLidar::setBeamAzimuthAngles(const VectorXf &beamAzimuthAngles)
{
  beamAzimuthAngles_ = beamAzimuthAngles;
}

VectorXf RangeImageLidar::beamElevationAngles() const
{
  return beamElevationAngles_;
}

void RangeImageLidar::setBeamElevationAngles(const VectorXf &beamElevationAngles)
{
  beamElevationAngles_ = beamElevationAngles;
}

// needed for test instance:
const VectorXf beam_altitude_angles = (VectorXf(64) <<
    16.611,  16.084,  15.557,  15.029,  14.502,  13.975,  13.447,  12.920,
    12.393,  11.865,  11.338,  10.811,  10.283,  9.756,   9.229,   8.701,
    8.174,   7.646,   7.119,   6.592,   6.064,   5.537,   5.010,   4.482,
    3.955,   3.428,   2.900,   2.373,   1.846,   1.318,   0.791,   0.264,
    -0.264,  -0.791,  -1.318,  -1.846,  -2.373,  -2.900,  -3.428,  -3.955,
    -4.482,  -5.010,  -5.537,  -6.064,  -6.592,  -7.119,  -7.646,  -8.174,
    -8.701,  -9.229,  -9.756,  -10.283, -10.811, -11.338, -11.865, -12.393,
    -12.920, -13.447, -13.975, -14.502, -15.029, -15.557, -16.084, -16.611).finished();

const VectorXf beam_azimuth_angles = (VectorXf(64) <<
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164,
    3.164, 1.055, -1.055, -3.164, 3.164, 1.055, -1.055, -3.164).finished();

// get a test instance
std::shared_ptr<ProjectionBase> RangeImageLidar::createTestObject()
{
  return std::shared_ptr<ProjectionBase>(new RangeImageLidar(2048, 64, beam_azimuth_angles,
                                                     beam_altitude_angles));
}
// \brief get a test instance
RangeImageLidar RangeImageLidar::testObject()
{
  return RangeImageLidar(2048, 64, beam_azimuth_angles, beam_altitude_angles);
}

// \brief Get the intrinsics as a concatenated vector.
// \param[out] intrinsics The intrinsics as a concatenated vector.
void RangeImageLidar::getIntrinsics(VectorXf &) const {
  throw std::runtime_error("not implemented");
}

// \brief overwrite all intrinsics - use with caution !
// \param[in] intrinsics The intrinsics as a concatenated vector.
bool RangeImageLidar::setIntrinsics(const VectorXf &) {
  throw std::runtime_error("not implemented");
}

}  // namespace projection
}  // namespace srl
