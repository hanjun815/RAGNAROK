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
 * @file implementation/EucmCamera.hpp
 * @brief Header implementation file for the EucmCamera class.
 * @author Simon Boche
 */

#include <Eigen/Geometry>
#include <glog/logging.h>

// \brief okvis Main namespace of this package.
namespace okvis {
// \brief cameras Namespace for camera-related functionality.
namespace cameras {

EucmCamera::EucmCamera(int imageWidth,
                       int imageHeight,
                       double focalLengthU,
                       double focalLengthV,
                       double imageCenterU,
                       double imageCenterV,
                       double alpha,
                       double beta,
                       uint64_t id)
         : CameraBase(imageWidth, imageHeight, id),
          fu_(focalLengthU),
          fv_(focalLengthV),
          cu_(imageCenterU),
          cv_(imageCenterV),
          alpha_(alpha),
          beta_(beta)
{
  intrinsics_[0] = fu_;  //< focalLengthU
  intrinsics_[1] = fv_;  //< focalLengthV
  intrinsics_[2] = cu_;  //< imageCenterU
  intrinsics_[3] = cv_;  //< imageCenterV
  intrinsics_[4] = alpha_;  //< alpha
  intrinsics_[5] = beta_;  //< beta
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
}

// overwrite all intrinsics - use with caution !
bool EucmCamera::setIntrinsics(
        const Eigen::VectorXd & intrinsics)
{
  if (intrinsics.cols() != NumIntrinsics && intrinsics.rows() != NumIntrinsics) {
    return false;
  }
  intrinsics_ = intrinsics;
  fu_ = intrinsics[0];  //< focalLengthU
  fv_ = intrinsics[1];  //< focalLengthV
  cu_ = intrinsics[2];  //< imageCenterU
  cv_ = intrinsics[3];  //< imageCenterV
  alpha_ = intrinsics[4];  //< alpha
  beta_ = intrinsics[5];  //< beta
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
  return true;
}

bool EucmCamera::initialiseCameraAwarenessMaps() { // ToDO: Implement / change this!
  rays_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC3);
  imageJacobians_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC(6));
  for (int v=0; v<imageHeight_; ++v) {
    for (int u=0; u<imageWidth_; ++u) {
      Eigen::Vector3d ray;
      Eigen::Matrix<double, 2, 3> jacobian;
      if(backProject(Eigen::Vector2d(u,v), &ray)) {
        ray.normalize();
      } else {
        ray.setZero();
      }
      rays_.at<cv::Vec3f>(v,u) = cv::Vec3f(ray[0],ray[1],ray[2]);
      Eigen::Vector2d pt;
      if(project(ray, &pt, &jacobian)
         ==cameras::ProjectionStatus::Successful) {
        cv::Vec6f j;
        j[0]=jacobian(0,0);
        j[1]=jacobian(0,1);
        j[2]=jacobian(0,2);
        j[3]=jacobian(1,0);
        j[4]=jacobian(1,1);
        j[5]=jacobian(1,2);
        imageJacobians_.at<cv::Vec6f>(v,u) = j;
      }
    }
  }
  return true;
}

bool EucmCamera::getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const {// ToDO: Implement / change this!
  rays = rays_;
  imageJacobians = imageJacobians_;
  if(rays_.empty() || imageJacobians_.empty()) {
    return false;
  }
  return true;
}

void EucmCamera::getIntrinsics(Eigen::VectorXd & intrinsics) const
{
  intrinsics = intrinsics_;
}

//////////////////////////////////////////
// Methods to project points

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus EucmCamera::project(
        const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint) const
{

  // projection
  Eigen::Vector2d imagePoint2;
  double rho = std::sqrt( beta_ * (point[0]*point[0] + point[1]*point[1]) + point[2]*point[2]);
  const double rz = 1.0 / (alpha_*rho + (1.0-alpha_) * point[2]);
  imagePoint2[0] = point[0] * rz;
  imagePoint2[1] = point[1] * rz;

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  return ProjectionStatus::Successful;
}

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus EucmCamera::project(
        const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
        Eigen::Matrix<double, 2, 3> * pointJacobian,
        Eigen::Matrix2Xd * intrinsicsJacobian) const
{

  // projection
  Eigen::Vector2d imagePoint2;
  double rho = std::sqrt( beta_ * (point[0]*point[0] + point[1]*point[1]) + point[2]*point[2]);
  const double rz = 1.0 / (alpha_ * rho + (1.0-alpha_) * point[2]);
  imagePoint2[0] = point[0] * rz;
  imagePoint2[1] = point[1] * rz;

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (intrinsicsJacobian) {
    // get both Jacobians
    double den = alpha_ * rho + (1. - alpha_) * point[2];
    intrinsicsJacobian->resize(2, NumIntrinsics);
    intrinsicsJacobian->setZero();
    (*intrinsicsJacobian)(0,0) = point[0] / den;
    (*intrinsicsJacobian)(1,1) = point[1] / den;
    (*intrinsicsJacobian)(0,2) = 1.0;
    (*intrinsicsJacobian)(1,3) = 1.0;
    (*intrinsicsJacobian)(0,4) = - fu_ * point[0] * (rho - point[2])/(den*den);
    (*intrinsicsJacobian)(1,4) = - fv_ * point[1] * (rho - point[2])/(den*den);
    (*intrinsicsJacobian)(0,5) = - alpha_ * fu_ * point[0] * (point[0]*point[0] + point[1]*point[1])/(2.0*rho*den*den);
    (*intrinsicsJacobian)(1,5) = - alpha_ * fv_ * point[1] * (point[0]*point[0] + point[1]*point[1])/(2.0*rho*den*den);
  }

  // compute the point Jacobian in any case
  Eigen::Matrix<double, 2, 3> & J = *pointJacobian;
  J(0, 0) = rz - (alpha_ * beta_ * point[0] * point[0]) * rz * rz / rho;
  J(0, 0) *= fu_;
  J(0, 1) = - (alpha_ * beta_ * point[0] * point[1]) * rz * rz / rho;
  J(0, 1) *= fu_;
  J(0, 2) = - point[0] * rz * rz * (1. - alpha_ + (alpha_ * point[2]) / rho);
  J(0, 2) *= fu_;

  J(1, 0) = - (alpha_ * beta_ * point[0] * point[1]) * rz * rz / rho;
  J(1, 0) *= fv_;
  J(1, 1) = rz - (alpha_ * beta_ * point[1] * point[1]) * rz * rz / rho;
  J(1, 1) *= fv_;
  J(1,2) = - point[1] * rz * rz * (1. - alpha_ + (alpha_ * point[2]) / rho);
  J(1, 2) *= fv_;


  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  return ProjectionStatus::Successful;

}

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus EucmCamera::projectWithExternalParameters(
        const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
        Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian,
        Eigen::Matrix2Xd * intrinsicsJacobian) const
{

  // parse parameters into human readable form
  const double fu = parameters[0];
  const double fv = parameters[1];
  const double cu = parameters[2];
  const double cv = parameters[3];
  const double alpha = parameters[4];
  const double beta = parameters[5];

  // projection
  Eigen::Vector2d imagePoint2;
  double rho = std::sqrt( beta * (point[0]*point[0] + point[1]*point[1]) + point[2]*point[2]);
  const double rz = 1.0 / (alpha * rho + (1.0-alpha) * point[2]);
  imagePoint2[0] = point[0] * rz;
  imagePoint2[1] = point[1] * rz;

  // scale and offset
  (*imagePoint)[0] = fu * imagePoint2[0] + cu;
  (*imagePoint)[1] = fv * imagePoint2[1] + cv;

  if (intrinsicsJacobian) {
    // get both Jacobians
    double den = alpha * rho + (1. - alpha) * point[2];
    intrinsicsJacobian->resize(2, NumIntrinsics);
    intrinsicsJacobian->setZero();
    (*intrinsicsJacobian)(0,0) = point[0] / den;
    (*intrinsicsJacobian)(1,1) = point[1] / den;
    (*intrinsicsJacobian)(0,2) = 1.0;
    (*intrinsicsJacobian)(1,3) = 1.0;
    (*intrinsicsJacobian)(0,4) = - fu * point[0] * (rho - point[2])/(den*den);
    (*intrinsicsJacobian)(1,4) = - fv * point[1] * (rho - point[2])/(den*den);
    (*intrinsicsJacobian)(0,5) = - alpha * fu * point[0] * (point[0]*point[0] + point[1]*point[1])/(2.0*rho*den*den);
    (*intrinsicsJacobian)(1,5) = - alpha * fv * point[1] * (point[0]*point[0] + point[1]*point[1])/(2.0*rho*den*den);
  }

  // compute the point Jacobian in any case
  Eigen::Matrix<double, 2, 3> & J = *pointJacobian;
  J(0, 0) = rz - (alpha * beta * point[0] * point[0]) * rz * rz / rho;
  J(0, 0) *= fu;
  J(0, 1) = - (alpha * beta * point[0] * point[1]) * rz * rz / rho;
  J(0, 1) *= fu;
  J(0, 2) = - point[0] * rz * rz * (1. - alpha + (alpha * point[2]) / rho);
  J(0, 2) *= fu;

  J(1, 0) = - (alpha * beta * point[0] * point[1]) * rz * rz / rho;
  J(1, 0) *= fv;
  J(1, 1) = rz - (alpha * beta * point[1] * point[1]) * rz * rz / rho;
  J(1, 1) *= fv;
  J(1,2) = - point[1] * rz * rz * (1. - alpha + (alpha * point[2]) / rho);
  J(1, 2) *= fv;


  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  return ProjectionStatus::Successful;
}

// Projects Euclidean points to 2d image points (projection) in a batch.
void EucmCamera::projectBatch(
        const Eigen::Matrix3Xd & points, Eigen::Matrix2Xd * imagePoints,
        std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector3d point = points.col(i);
    Eigen::Vector2d imagePoint;
    ProjectionStatus status = project(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    stati->push_back(status);
  }
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
ProjectionStatus EucmCamera::projectHomogeneous(
        const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const
{
  Eigen::Vector3d point3d = point.head<3>() / point[3];
  return project(point3d, imagePoint);
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
ProjectionStatus EucmCamera::projectHomogeneous(
        const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
        Eigen::Matrix<double, 2, 4> * pointJacobian,
        Eigen::Matrix2Xd * intrinsicsJacobian) const
{
  Eigen::Vector3d point3d = point.head<3>() / point[3];
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  ProjectionStatus status;
  status = project(point3d, imagePoint, &pointJacobian3, intrinsicsJacobian);

  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3 / point[3]; // scale also Jacobian by homogeneous coordinate
  return status;
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
ProjectionStatus EucmCamera::projectHomogeneousWithExternalParameters(
        const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
        Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 4> * pointJacobian,
        Eigen::Matrix2Xd * intrinsicsJacobian) const
{
  Eigen::Vector3d point3d = point.head<3>() / point[3];
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  ProjectionStatus status;
  status = projectWithExternalParameters(point3d, parameters, imagePoint, &pointJacobian3, intrinsicsJacobian);

  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3 / point[3];
  return status;
}

// Projects points in homogenous coordinates to 2d image points (projection) in a batch.
void EucmCamera::projectHomogeneousBatch(
        const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
        std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector4d point = points.col(i);
    Eigen::Vector2d imagePoint;
    ProjectionStatus status = projectHomogeneous(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    stati->push_back(status);
  }
}

//////////////////////////////////////////
// Methods to backproject points

// Back-project a 2d image point into Euclidean space (direction vector).
bool EucmCamera::backProject(
        const Eigen::Vector2d & imagePoint, Eigen::Vector3d * direction) const
{
  double mx = (imagePoint[0] - cu_) / fu_;
  double my = (imagePoint[1] - cv_) / fv_;
  double rSquared = mx * mx + my * my;
  double mz =  (1. - beta_ * alpha_*alpha_ * rSquared) / (alpha_ * std::sqrt( 1. - (2.*alpha_ - 1.) * beta_ * rSquared) + (1. - alpha_));

  double scale =  1. / std::sqrt( mx*mx + my*my + mz*mz);
  (*direction)[0] = scale * mx;
  (*direction)[1] = scale * my;
  (*direction)[2] = scale * mz;

  // consider domain
  if ((alpha_ > 0.5) && (rSquared > 1./(beta_ * (2. * alpha_ - 1.)))) {
    return false;
  }

  return true;
}

// Back-project a 2d image point into Euclidean space (direction vector).
inline bool EucmCamera::backProject(
        const Eigen::Vector2d & imagePoint, Eigen::Vector3d * direction,
        Eigen::Matrix<double, 3, 2> * pointJacobian) const
{

  double mx = (imagePoint[0] - cu_) / fu_;
  double my = (imagePoint[1] - cv_) / fv_;
  double rSquared = mx * mx + my * my;
  double mz =  (1. - beta_ * alpha_*alpha_ * rSquared) / (alpha_ * std::sqrt(1. - (2.*alpha_ -1.) * beta_ * rSquared) + (1. - alpha_));

  double scale =  1. / std::sqrt( mx*mx + my*my + mz*mz);
  (*direction)[0] = scale * mx;
  (*direction)[1] = scale * my;
  (*direction)[2] = scale * mz;

  if ((alpha_ > 0.5) && (rSquared > beta_ * (2. * alpha_ - 1.))) {
    return false;
  }

  // ToDo: Compute actual Jacobians
  // Jacobian w.r.t. imagePoint
  Eigen::Matrix<double, 3, 2> outProjectJacobian = Eigen::Matrix<double, 3, 2>::Zero();
  *pointJacobian = outProjectJacobian;
  LOG(ERROR) << "pointJacobian for EUCM Camera back-projection not implemented.";

  return true;
}

// Back-project 2d image points into Euclidean space.
bool EucmCamera::backProjectBatch(
        const Eigen::Matrix2Xd & imagePoints, Eigen::Matrix3Xd * directions,
        std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector3d point;
    bool suc = backProject(imagePoint, &point);
    success->push_back(suc);
    directions->col(i) = point;
  }
  return true;
}

// Back-project a 2d image point into homogeneous point (direction vector).
bool EucmCamera::backProjectHomogeneous(
        const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction) const
{
  Eigen::Vector3d ray;
  bool success = backProject(imagePoint, &ray);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  // arbitrary
  return success;
}

// Back-project a 2d image point into homogeneous point (direction vector).
bool EucmCamera::backProjectHomogeneous(
        const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
        Eigen::Matrix<double, 4, 2> * pointJacobian) const
{
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 3, 2> pointJacobian3;
  bool success = backProject(imagePoint, &ray, &pointJacobian3);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  // arbitrary
  pointJacobian->template bottomRightCorner<1,2>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
  return success;
}

// Back-project 2d image points into homogeneous points (direction vectors).
bool EucmCamera::backProjectHomogeneousBatch(
        const Eigen::Matrix2Xd & imagePoints, Eigen::Matrix4Xd * directions,
        std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  directions->row(3) = Eigen::VectorXd::Ones(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector4d point;
    bool suc = backProjectHomogeneous(imagePoint, &point);
    success->push_back(suc);
    directions->col(i) = point;
  }
  return true;
}

}  // namespace cameras
}  // namespace okvis

