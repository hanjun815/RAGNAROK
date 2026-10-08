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
 * @file cameras/EucmCamera.hpp
 * @brief Header file for the EUCM (Enhanced Unified Camera Model) class.
 * @author Simon Boche
 */


#ifndef INCLUDE_OKVIS_CAMERAS_EUCMCAMERA_HPP_
#define INCLUDE_OKVIS_CAMERAS_EUCMCAMERA_HPP_


#include <vector>
#include <memory>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "okvis/cameras/CameraBase.hpp"
#include "okvis/cameras/DistortionBase.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {


/// \class EucmCamera

class EucmCamera : public CameraBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Constructor
    /// @param[in] imageWidth The width in pixels.
    /// @param[in] imageHeight The height in pixels.
    /// @param[in] focalLengthU The horizontal focal length in pixels.
    /// @param[in] focalLengthV The vertical focal length in pixels.
    /// @param[in] imageCenterU The horizontal centre in pixels.
    /// @param[in] imageCenterV The vertical centre in pixels.
    /// @param[in] alpha Parameter alpha in [0,1] of the EUCM model
    /// @param[in] beta Parameter beta > 0 of the EUCM model
    /// @param[in] id Assign a generic ID, if desired.
    inline EucmCamera(int imageWidth, int imageHeight, double focalLengthU,
                  double focalLengthV, double imageCenterU, double imageCenterV,
                  double alpha, double beta, uint64_t id=-1);

    /// \brief Destructor.
    virtual ~EucmCamera()
    {
    }

    static const int NumProjectionIntrinsics = 6;  ///< optimisable projection intrinsics
    static const int NumIntrinsics = NumProjectionIntrinsics; ///< total number of intrinsics, distortion is part of projection in EUCM

    /// \brief Get the focal length along the u-dimension.
    /// \return The horizontal focal length in pixels.
    virtual double focalLengthU() const
    {
      return fu_;
    }

    /// \brief Get the focal length along the v-dimension.
    /// \return The vertical focal length in pixels.
    virtual double focalLengthV() const
    {
      return fv_;
    }

    /// \brief Get the image centre along the u-dimension.
    /// \return The horizontal centre in pixels.
    virtual double imageCenterU() const
    {
      return cu_;
    }

    /// \brief Get the focal image centre along the v-dimension.
    /// \return The vertical centre in pixels.
    virtual double imageCenterV() const
    {
      return cv_;
    }

    /// \brief Get alpha in [0,1] of EUCM model
    /// \return alpha
    double alpha() const
    {
      return alpha_;
    }

    /// \brief Get beta >0 of EUCM model
    /// \return beta
    double beta() const
    {
      return beta_;
    }

    /// \brief Get the intrinsics as a concatenated vector.
    /// \return The intrinsics as a concatenated vector.
    inline void getIntrinsics(Eigen::VectorXd & intrinsics) const ;

    /// \brief overwrite all intrinsics - use with caution !
    /// \param[in] intrinsics The intrinsics as a concatenated vector.
    inline bool setIntrinsics(const Eigen::VectorXd & intrinsics);

    /// \brief Not used
    virtual void setRectifyMap(const cv::Mat & /*map1*/, const cv::Mat & /*map2*/) {};

    /// \brief Not used
    virtual void getRectifyMap(cv::Mat & /*map1*/, cv::Mat & /*map2*/) const {};

    /// \brief Get the total number of intrinsics.
    /// \return Number of intrinsics parameters.
    inline int noIntrinsicsParameters() const
    {
      return NumIntrinsics;
    }

    inline virtual bool initialiseCameraAwarenessMaps();
    inline virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const;
    const std::string distortionType() const
    {
      return "NoDistortion";
    }

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
            const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint) const;

    /// \brief Projects a Euclidean point to a 2d image point (projection).
    ///        Uses projection including distortion models.
    /// @param[in]  point              The point in Euclidean coordinates.
    /// @param[out] imagePoint         The image point.
    /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
    /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
    /// @return     Get information about the success of the projection. See
    ///             \ref ProjectionStatus for more information.
    inline ProjectionStatus project(
            const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 3> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

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
            const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
            Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
    ///        Uses projection including distortion models.
    /// @param[in]  points      The points in Euclidean coordinates (one point per column).
    /// @param[out] imagePoints The image points (one point per column).
    /// @param[out] stati       Get information about the success of the projections. See
    ///                         \ref ProjectionStatus for more information.
    inline void projectBatch(
            const Eigen::Matrix3Xd & points, Eigen::Matrix2Xd * imagePoints,
            std::vector<ProjectionStatus> * stati) const;

    /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
    ///        Uses projection including distortion models.
    /// @param[in]  point      The point in Homogeneous coordinates.
    /// @param[out] imagePoint The image point.
    /// @return     Get information about the success of the projection. See
    ///             \ref ProjectionStatus for more information.
    inline ProjectionStatus projectHomogeneous(
            const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const;

    /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
    ///        Uses projection including distortion models.
    /// @param[in]  point              The point in Homogeneous coordinates.
    /// @param[out] imagePoint         The image point.
    /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
    /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
    /// @return     Get information about the success of the projection. See
    ///             \ref ProjectionStatus for more information.
    inline ProjectionStatus projectHomogeneous(
            const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 4> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

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
            const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
            Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 4> * pointJacobian = NULL,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    /// \brief Projects points in homogenous coordinates to 2d image points (projection) in a batch.
    ///        Uses projection including distortion models.
    /// @param[in]  points      The points in homogeneous coordinates (one point per column).
    /// @param[out] imagePoints The image points (one point per column).
    /// @param[out] stati       Get information about the success of the projections. See
    ///                         \ref ProjectionStatus for more information.
    inline void projectHomogeneousBatch(
            const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
            std::vector<ProjectionStatus> * stati) const;
    /// @}

    //////////////////////////////////////////////////////////////
    /// \name Methods to backproject points
    /// @{

    /// \brief Back-project a 2d image point into Euclidean space (direction vector).
    /// @param[in]  imagePoint The image point.
    /// @param[out] direction  The Euclidean direction vector.
    /// @return     true on success.
    inline bool backProject(const Eigen::Vector2d & imagePoint,
                            Eigen::Vector3d * direction) const;

    /// \brief Back-project a 2d image point into Euclidean space (direction vector).
    /// @param[in]  imagePoint         The image point.
    /// @param[out] direction          The Euclidean direction vector.
    /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
    /// @return     true on success.
    inline bool backProject(const Eigen::Vector2d & imagePoint,
                            Eigen::Vector3d * direction,
                            Eigen::Matrix<double, 3, 2> * pointJacobian) const;

    /// \brief Back-project 2d image points into Euclidean space (direction vectors).
    /// @param[in]  imagePoints The image points (one point per column).
    /// @param[out] directions  The Euclidean direction vectors (one point per column).
    /// @param[out] success     Success of each of the back-projection
    inline bool backProjectBatch(const Eigen::Matrix2Xd & imagePoints,
                                 Eigen::Matrix3Xd * directions,
                                 std::vector<bool> * success) const;

    /// \brief Back-project a 2d image point into homogeneous point (direction vector).
    /// @param[in]  imagePoint The image point.
    /// @param[out] direction  The homogeneous point as direction vector.
    /// @return     true on success.
    inline bool backProjectHomogeneous(const Eigen::Vector2d & imagePoint,
                                       Eigen::Vector4d * direction) const;

    /// \brief Back-project a 2d image point into homogeneous point (direction vector).
    /// @param[in]  imagePoint         The image point.
    /// @param[out] direction          The homogeneous point as direction vector.
    /// @param[out] pointJacobian      Jacobian of the back-projection function.
    /// @return     true on success.
    inline bool backProjectHomogeneous(
            const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
            Eigen::Matrix<double, 4, 2> * pointJacobian) const;

    /// \brief Back-project 2d image points into homogeneous points (direction vectors).
    /// @param[in]  imagePoints The image points (one point per column).
    /// @param[out] directions  The homogeneous points as direction vectors (one point per column).
    /// @param[out] success     Success of each of the back-projection
    inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd & imagePoints,
                                            Eigen::Matrix4Xd * directions,
                                            std::vector<bool> * success) const;
    /// @}

    /// \brief get a test instance
    static std::shared_ptr<CameraBase> createTestObject()
    {
      return std::shared_ptr<CameraBase>(new EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99));
    }
    /// \brief get a test instance
    static EucmCamera testObject()
    {
      return EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99);
    }

    /// \brief Obtain the projection type
    std::string type() const
    {
      return "EUCMCamera";
    }

  protected:

    /// \brief No default constructor.
    EucmCamera() = delete;
    Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  ///< summary of all intrinsics parameters
    double fu_;  ///< focalLengthU
    double fv_;  ///< focalLengthV
    double cu_;  ///< imageCenterU
    double cv_;  ///< imageCenterV
    double alpha_; ///< alpha parameter
    double beta_; ///< beta parameter
    double one_over_fu_;  ///< 1.0 / fu_
    double one_over_fv_;  ///< 1.0 / fv_
    double fu_over_fv_;  ///< fu_ / fv_

    cv::Mat rays_;
    cv::Mat imageJacobians_;

};

}  // namespace cameras
}  // namespace okvis

#include "implementation/EucmCamera.hpp"

#endif /*INCLUDE_OKVIS_CAMERAS_EUCMCAMERA_HPP_*/
