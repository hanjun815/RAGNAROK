//
// Created by boche on 5/5/22.
//

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
 * @file implementation/Lidar.hpp
 * @brief Header implementation file for the Lidar class.
 * @author Simon Boche
 */

#include <Eigen/Geometry>

// \brief Main namespace of this package.
namespace srl {
// \brief Namespace for camera-related functionality.
namespace projection {

Lidar::Lidar(const int imageWidth, const int imageHeight)
    : ProjectionBase(imageWidth, imageHeight)
{
    azimuthResolution_ = 360.0f / imageWidth;
    elevationResolution_ = 180.0f / imageHeight;
}

//////////////////////////////////////////
// Methods to project points

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus Lidar::project(
    const Vector3f & point, Vector2f * imagePoint) const
{
    // handle singularity
    if (point.norm() < 1.0e-12) {
        return ProjectionStatus::Invalid;
    }

    // Compute azimuth and elevation angles (projection) [deg]
    float rad2deg = 180.0 / M_PI;
    const float_t R = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
    const float_t azimuth = std::atan2(point[0], point[2]); // ToDo: 2.0 * M_PI - ... ???
    const float_t elevation = std::asin(point[1]/R);
    /*** note that z is in forward direction for BLK2Fly LiDAR sensor
     * This differs from commonly used conventions!
     */

    // Check Bounds not needed? Full Coverage ToDo: Reason about this

    // Determine row and column in image
    //float_t v = (elevation + 90.0) / elevationResolution_;
    //float_t u = (azimuth + 180.0) / azimuthResolution_;

    float_t u = (azimuth * M_PI/180.0 * std::cos(elevation * M_PI/180.0) + M_PI)*rad2deg / azimuthResolution_;
    float_t v = (elevation * M_PI/180.0 + 0.5f * M_PI)*rad2deg / elevationResolution_;
    u = (azimuth + M_PI)*rad2deg  / azimuthResolution_;
    v = (elevation + 0.5*M_PI)*rad2deg / elevationResolution_;

    (*imagePoint)[0] = u;
    (*imagePoint)[1] = v;

    // azimuthal wrap-around
    if((*imagePoint)[0]<-0.5) {
        (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
    }
    if((*imagePoint)[0]>imageWidth_-0.5) {
        (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
    }
    // elevation wrap-around
    if((*imagePoint)[1]<-0.5) {
        (*imagePoint)[1] = (*imagePoint)[1] + imageHeight_;
    }
    if((*imagePoint)[1]>imageHeight_-0.5) {
        (*imagePoint)[1] = (*imagePoint)[1] - imageHeight_;
    }

    // checks
    if (ProjectionBase::isMasked(*imagePoint)) {
        return ProjectionStatus::Masked;
    }
    return ProjectionStatus::Successful;
}

// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus Lidar::project(
    const Vector3f & point, Vector2f * imagePoint,
    Matrixf<2, 3> * pointJacobian,
    Matrix2Xf * intrinsicsJacobian) const
{
    // handle singularity
    if (point.norm() < 1.0e-12) {
        return ProjectionStatus::Invalid;
    }

    // Compute azimuth and elevation angles (projection) [deg]
    float rad2deg = 180.0 / M_PI;
    const float_t R = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
    const float_t azimuth = std::atan2(point[0], point[2]); // ToDo: 2.0 * M_PI - ... ???
    const float_t elevation = std::asin(point[1]/R);
    /*** note that z is in forward direction for BLK2Fly LiDAR sensor
     * This differs from commonly used conventions!
     */

    // Check Bounds not needed? Full Coverage ToDo: Reason about this

    // Determine row and column in image
    //float_t v = (elevation + 90.0) / elevationResolution_;
    //float_t u = (azimuth + 180.0) / azimuthResolution_;
    float_t u = (azimuth * std::cos(elevation) + M_PI)*rad2deg / azimuthResolution_;
    float_t v = (elevation + 0.5f * M_PI)*rad2deg / elevationResolution_;
    u = (azimuth + M_PI)*rad2deg  / azimuthResolution_;
    v = (elevation + 0.5*M_PI)*rad2deg / elevationResolution_;

    (*imagePoint)[0] = u;
    (*imagePoint)[1] = v;

    // azimuthal wrap-around
    if((*imagePoint)[0]<-0.5) {
        (*imagePoint)[0] = (*imagePoint)[0] + imageWidth_;
    }
    if((*imagePoint)[0]>imageWidth_-0.5) {
        (*imagePoint)[0] = (*imagePoint)[0] - imageWidth_;
    }
    // elevation wrap-around
    if((*imagePoint)[1]<-0.5) {
        (*imagePoint)[1] = (*imagePoint)[1] + imageHeight_;
    }
    if((*imagePoint)[1]>imageHeight_-0.5) {
        (*imagePoint)[1] = (*imagePoint)[1] - imageHeight_;
    }

    // Jacobians
    if(pointJacobian) {
        const float_t deg2rad = M_PI / 180.0;
        pointJacobian->setZero();
        const float_t R3 = R*R*R * std::sqrt(1 - (point[1]*point[1])/(R*R));
        const float_t XZ = 1.0 + ( point[0] * point[0] ) / ( point[2] * point[2] );

        (*pointJacobian)(0,0) = 1.0 / (deg2rad*azimuthResolution_ * point[2] * XZ);
        (*pointJacobian)(0,1) = 0.0;
        (*pointJacobian)(0,2) = -point[0] / (deg2rad*azimuthResolution_ * point[2]*point[2] * XZ);

        (*pointJacobian)(1,0) = -(point[0]*point[1]) / (deg2rad*elevationResolution_ * R3);
        (*pointJacobian)(1,1) = std::sqrt(1. - (point[1]*point[1])/(R*R)) / (deg2rad*elevationResolution_ * R);
        (*pointJacobian)(1,2) = -(point[1]*point[2]) / (deg2rad*elevationResolution_ * R3);

    }
    if(intrinsicsJacobian) {
        throw std::runtime_error("Lidar does not support intrinsics Jacobian");
    }

    // checks
    if (ProjectionBase::isMasked(*imagePoint)) {
        return ProjectionStatus::Masked;
    }
    return ProjectionStatus::Successful;
}

ProjectionStatus Lidar::projectSphere(
    const Vector3f& center,
    float radius,
    Vector2f* imageCenter,
    float& imageRadius) const
{
    // handle singularity
    if (center.norm() < 1.0e-12) {
        return ProjectionStatus::Invalid;
    }

    // Compute azimuth and elevation angles (projection) [deg]
    float rad2deg = 180.0 / M_PI;
    const float_t R = sqrt(center[0]*center[0] + center[1]*center[1] + center[2]*center[2]);
    const float_t azimuth = std::atan2(center[0], center[2]); // ToDo: 2.0 * M_PI - ... ???
    const float_t elevation = std::asin(center[1]/R);
    /*** note that z is in forward direction for BLK2Fly LiDAR sensor
     * This differs from commonly used conventions!
     */

    // Check Bounds not needed? Full Coverage ToDo: Reason about this

    // Determine row and column in image
    float u = (azimuth + M_PI)*rad2deg  / azimuthResolution_;
    float v = (elevation + 0.5*M_PI)*rad2deg / elevationResolution_;

    (*imageCenter)[0] = u;
    (*imageCenter)[1] = v;

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
        // The sensor is inside the sphere ToDo: how to handle for full coverage?
        return ProjectionStatus::OutsideImage;
    }
    // Compute radian to degree conversion once
    static const float_t radToDeg = 180. / M_PI;
    // Get the angle in degrees from the sine
    const float_t azimuthRightOffset = radToDeg * std::asin(sinTheta);
    // NOTE asin ~= x + x^3 / 6
    //const float_t azimuthRightOffset = radToDeg * (sinTheta + sinTheta * sinTheta * sinTheta / 6);
    // Azimuth angle of a ray tangent to the right side of the sphere
    const float_t azimuthRight = azimuth + azimuthRightOffset;
    // Horizontal pixel coordinate of the pixel at the right edge of the projected circle
    float_t uRadius = (azimuthRight + 180.0) / 360.0 * imageWidth_; // ToDo: think about this
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


// ToDo: Check further projection fcts (batch, hom., ...) and corresponding backprojections
// Projects a Euclidean point to a 2d image point (projection).
ProjectionStatus Lidar::projectWithExternalParameters(
    const Vector3f &, const VectorXf &,
    Vector2f *, Matrixf<2, 3> *,
    Matrix2Xf *) const
{
    throw std::runtime_error("external parameters projection for Lidar not implemented");
    return ProjectionStatus::Invalid;
}

// Projects Euclidean points to 2d image points (projection) in a batch.
void Lidar::projectBatch(
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
ProjectionStatus Lidar::projectHomogeneous(
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
ProjectionStatus Lidar::projectHomogeneous(
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
ProjectionStatus Lidar::projectHomogeneousWithExternalParameters(
    const Vector4f &, const VectorXf &,
    Vector2f *, Matrixf<2, 4> *,
    Matrix2Xf *) const
{
    throw std::runtime_error("intrinsics Jacobian for Lidar not implemented");
    return ProjectionStatus::Invalid;
}

// Projects points in homogenous coordinates to 2d image points (projection) in a batch.
void Lidar::projectHomogeneousBatch(
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
bool Lidar::backProject(
    const Vector2f & imagePoint, Vector3f * direction) const
{
    // adapted from
    // https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/os1_util.cpp

    // Interpolation not needed as azimuth and elevation both continuously in our case ToDo: resason about this

    // azimuth angle with interpolated offset
    static const float_t radToDeg = 180.0 / M_PI;
    const float_t az = imagePoint[0] * azimuthResolution_/radToDeg - M_PI; // ToDo. reason about offsets of azimut / elevation (is -180, -90 needed)
    const float_t el = imagePoint[1] * elevationResolution_/radToDeg - 0.5 * M_PI;
    //const float_t el = imagePoint[1] * elevationResolution_/radToDeg - 0.5 * M_PI;
    //const float_t az = (imagePoint[0] * azimuthResolution_/radToDeg - M_PI) / std::cos(el);

    // project as ray
    (*direction)[0] = std::cos(el) * std::sin(az);
    (*direction)[1] = std::sin(el);
    (*direction)[2] = std::cos(el) * std::cos(az);

    return true;
}

// Back-project a 2d image point into Euclidean space (direction vector).
inline bool Lidar::backProject(
    const Vector2f & imagePoint, Vector3f * direction,
    Matrixf<3, 2> * pointJacobian) const
{
    // adapted from
    // https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/os1_util.cpp

    // Interpolation not needed as azimuth and elevation both continuously in our case ToDo: resason about this

    // azimuth angle with interpolated offset
    static const float_t radToDeg = 180.0 / M_PI;
    const float_t az = imagePoint[0] * azimuthResolution_/radToDeg - M_PI; // ToDo. reason about offsets of azimut / elevation (is -180, -90 needed)
    const float_t el = imagePoint[1] * elevationResolution_/radToDeg - 0.5 * M_PI;
    //const float_t el = imagePoint[1] * elevationResolution_/radToDeg - 0.5 * M_PI;
    //const float_t az = (imagePoint[0] * azimuthResolution_/radToDeg - M_PI) / std::cos(el);

    // project as ray
    (*direction)[0] = std::cos(el) * std::sin(az);
    (*direction)[1] = std::sin(el);
    (*direction)[2] = std::cos(el) * std::cos(az);

    // Jacobian w.r.t. imagePoint
    pointJacobian->setZero(); /// \todo

    return true;
}

// Back-project 2d image points into Euclidean space (direction vectors).
bool Lidar::backProjectBatch(
    const Matrix2Xf & imagePoints, Matrix3Xf * directions,
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
        directions->col(i) = point;
    }
    return true;
}

// Back-project a 2d image point into homogeneous point (direction vector).
bool Lidar::backProjectHomogeneous(
    const Vector2f & imagePoint, Vector4f * direction) const
{
    Vector3f ray;
    bool success = backProject(imagePoint, &ray);
    direction->template head<3>() = ray;
    (*direction)[3] = 1.0;  // arbitrary
    return success;
}

// Back-project a 2d image point into homogeneous point (direction vector).
bool Lidar::backProjectHomogeneous(
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
bool Lidar::backProjectHomogeneousBatch(
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

float Lidar::azimuthResolutionAngle() const
{
    return azimuthResolution_;
}

void Lidar::setAzimuthResolutionAngle(const float azimuthResolutionAngle)
{
    azimuthResolution_ = azimuthResolutionAngle;
}

float Lidar::elevationResolutionAngle() const
{
    return elevationResolution_;
}

void Lidar::setElevationResolutionAngle(const float elevationResolutionAngle)
{
    elevationResolution_ = elevationResolutionAngle;
}


// get a test instance
std::shared_ptr<ProjectionBase> Lidar::createTestObject()
{
    return std::shared_ptr<ProjectionBase>(new Lidar(3600, 1800));
}
// \brief get a test instance
Lidar Lidar::testObject()
{
    return Lidar(3600, 1800);
}

// \brief Get the intrinsics as a concatenated vector.
// \param[out] intrinsics The intrinsics as a concatenated vector.
void Lidar::getIntrinsics(VectorXf &) const {
    throw std::runtime_error("not implemented");
}

// \brief overwrite all intrinsics - use with caution !
// \param[in] intrinsics The intrinsics as a concatenated vector.
bool Lidar::setIntrinsics(const VectorXf &) {
    throw std::runtime_error("not implemented");
}

}  // namespace projection
}  // namespace srl
