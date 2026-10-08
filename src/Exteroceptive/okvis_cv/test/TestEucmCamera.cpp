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

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <random>
#include "okvis/cameras/EucmCamera.hpp"
#include <okvis/assert_macros.hpp>

TEST(EucmCamera, project)
{
  const size_t NUM_POINTS = 100;

  // instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Try some points
  for(size_t i = 0; i < NUM_POINTS; ++i){

    // create a random point in the field of view:
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

    // backProject
    Eigen::Vector3d point3d;
    bool backStat = camera->backProject(imagePoint, &point3d);
    OKVIS_ASSERT_TRUE(std::runtime_error,  backStat,"unsuccessful back projection");

    // re-project
    Eigen::Vector2d imagePoint2;
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Matrix2Xd J_intrinsics;
    okvis::cameras::ProjectionStatus stat = camera->project(point3d, &imagePoint2, &J, &J_intrinsics);
    OKVIS_ASSERT_TRUE(std::runtime_error,stat == okvis::cameras::ProjectionStatus::Successful,"unsuccessful projection");

    // Check Point Jacobian vs. NumDiff
    const double dp = 1.0e-7;
    Eigen::Matrix<double, 2, 3> J_numDiff;
    for (size_t d = 0; d < 3; ++d) {
      Eigen::Vector3d point_p = point3d + Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector3d point_m = point3d - Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->project(point_p, &imagePoint_p);
      camera->project(point_m, &imagePoint_m);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    //std::cout << "==========\n J:\n" << J << std::endl;
    //std::cout << "==========\n J_numDiff:\n" << J_numDiff << std::endl;
    // Check Jacobian Error Norm
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_numDiff - J).norm() < 0.0001,"Point Jacobian Verification failed");

    // check intrinsics Jacobian by numeric differences
    const int numIntrinsics = camera->noIntrinsicsParameters();
    Eigen::Matrix2Xd J_intrinsics_numDiff;
    J_intrinsics_numDiff.resize(2,numIntrinsics);
    // back-up original intrinsics
    Eigen::VectorXd original_intrinsics;
    camera->getIntrinsics(original_intrinsics);

    for (size_t d = 0; d < 6; ++d) {
      Eigen::VectorXd dpVec;
      dpVec.resize(numIntrinsics);
      dpVec.setZero();
      dpVec[d] = dp;
      Eigen::VectorXd intrinsics_p = original_intrinsics + dpVec;
      Eigen::VectorXd intrinsics_m = original_intrinsics - dpVec;
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->setIntrinsics(intrinsics_p);
      camera->project(point3d, &imagePoint_p);
      camera->setIntrinsics(intrinsics_m);
      camera->project(point3d, &imagePoint_m);
      J_intrinsics_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
      // restore original intrinsics
      camera->setIntrinsics(original_intrinsics);
    }

    // Assert correctness of intrinsics jacobian
    //std::cout << "==========\n J_intrinsics:\n" << J_intrinsics << std::endl;
    //std::cout << "==========\n J_intrinsics_numDiff:\n" << J_intrinsics_numDiff << std::endl;
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_intrinsics_numDiff - J_intrinsics).norm() < 0.0001,"Intrinsics Jacobian Verification failed");

    // check re-projection error
    OKVIS_ASSERT_TRUE(std::runtime_error, (imagePoint2 - imagePoint).norm() < 0.01,"project/unproject failure");
  }
}
TEST(EucmCamera, projectWithExternalParameters){
  const size_t NUM_POINTS = 100;

  // Instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Get Intrinsics
  Eigen::VectorXd intrinsics;
  camera->getIntrinsics(intrinsics);

  for(size_t i = 0; i < NUM_POINTS; ++i){

    // create random visible point
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

    // backProject
    Eigen::Vector3d point3d;
    bool backStat = camera->backProject(imagePoint, &point3d);
    OKVIS_ASSERT_TRUE(std::runtime_error, backStat,"unsuccessful back projection");

    // re-project (with external parameters)
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Matrix2Xd J_intrinsics;
    Eigen::Vector2d imagePoint2;
    okvis::cameras::ProjectionStatus stat = camera->projectWithExternalParameters(point3d, intrinsics, &imagePoint2, &J, &J_intrinsics);
    OKVIS_ASSERT_TRUE(std::runtime_error,stat == okvis::cameras::ProjectionStatus::Successful,"unsuccessful projection");

    // check point Jacobian by numeric differences
    const double dp = 1.0e-7;
    Eigen::Matrix<double, 2, 3> J_numDiff;
    for (size_t d = 0; d < 3; ++d) {
      Eigen::Vector3d point_p = point3d + Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector3d point_m = point3d - Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->projectWithExternalParameters(point_p, intrinsics, &imagePoint_p, &J);
      camera->projectWithExternalParameters(point_m, intrinsics, &imagePoint_m, &J);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    // assert correctness of point Jacobian
    //std::cout << "==========\n J:\n" << J << std::endl;
    //std::cout << "==========\n J_numDiff:\n" << J_numDiff << std::endl;"
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_numDiff - J).norm() < 0.0001,"Point Jacobian Verification failed");

    // check intrinsics Jacobian by numeric differences
    const int numIntrinsics = camera->noIntrinsicsParameters();
    Eigen::Matrix2Xd J_intrinsics_numDiff;
    J_intrinsics_numDiff.resize(2,numIntrinsics);
    for (size_t d = 0; d < 6; ++d) {
      Eigen::VectorXd dpVec;
      dpVec.resize(numIntrinsics);
      dpVec.setZero();
      dpVec[d] = dp;
      Eigen::VectorXd intrinsics_p = intrinsics + dpVec;
      Eigen::VectorXd intrinsics_m = intrinsics - dpVec;
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->projectWithExternalParameters(point3d, intrinsics_p, &imagePoint_p, &J);
      camera->projectWithExternalParameters(point3d, intrinsics_m, &imagePoint_m, &J);

      J_intrinsics_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    // Assert correctness of intrinsics jacobian
    //std::cout << "==========\n J_intrinsics:\n" << J_intrinsics << std::endl;
    //std::cout << "==========\n J_intrinsics_numDiff:\n" << J_intrinsics_numDiff << std::endl;"
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_intrinsics_numDiff - J_intrinsics).norm() < 0.0001,"Intrinsics Jacobian Verification failed");

    // check also re-projection error
    OKVIS_ASSERT_TRUE(std::runtime_error, (imagePoint2 - imagePoint).norm() < 0.01,"project/unproject failure");
  }
}
TEST(EucmCamera, batchProjections){

  const size_t NUM_POINTS = 100;
  Eigen::Matrix2Xd imagePointsBatch;
  imagePointsBatch.resize(2, NUM_POINTS);
  Eigen::Matrix3Xd backProjectBatch;
  backProjectBatch.resize(3, NUM_POINTS);
  Eigen::Matrix2Xd reProjectBatch;
  reProjectBatch.resize(2,NUM_POINTS);

  // instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Create batch of image points
  for(size_t i = 0; i < NUM_POINTS; ++i) {

    // create a random point in the field of view:
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();
    imagePointsBatch.col(i) = imagePoint;
  }

  // backProject Batch
  std::vector<bool> success;
  camera->backProjectBatch(imagePointsBatch, &backProjectBatch, &success);


  // Re-Project batch
  std::vector<okvis::cameras::ProjectionStatus> stati;
  camera->projectBatch(backProjectBatch, &reProjectBatch, &stati);

  // Test reprojection error
  for(size_t i = 0; i < NUM_POINTS; ++i){
    double reProjectError = (imagePointsBatch.col(i)-reProjectBatch.col(i)).norm();
    OKVIS_ASSERT_TRUE(std::runtime_error, reProjectError < 0.01,"project/unproject failure for batchwise projections");
  }
}
TEST(EucmCamera, projectHomogeneous){

  std::srand(1); // set seed
  const size_t NUM_POINTS = 100;

  // instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Try some points
  for(size_t i = 0; i < NUM_POINTS; ++i){

    // create a random point in the field of view:
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

    // backProject
    Eigen::Vector4d point4d;
    bool backStat = camera->backProjectHomogeneous(imagePoint, &point4d);
    // Randomize homogeneous coordinate
    double factor = (double)(std::rand()) / RAND_MAX;
    point4d *= factor;

    OKVIS_ASSERT_TRUE(std::runtime_error,  backStat,"unsuccessful back projection");

    // re-project
    Eigen::Vector2d imagePoint2;
    Eigen::Matrix<double, 2, 4> J;
    Eigen::Matrix2Xd J_intrinsics;
    okvis::cameras::ProjectionStatus stat = camera->projectHomogeneous(point4d, &imagePoint2, &J, &J_intrinsics);
    OKVIS_ASSERT_TRUE(std::runtime_error,stat == okvis::cameras::ProjectionStatus::Successful,"unsuccessful projection");

    // Check Point Jacobian vs. NumDiff
    const double dp = 1.0e-7;
    Eigen::Matrix<double, 2, 4> J_numDiff;
    for (size_t d = 0; d < 4; ++d) {
      Eigen::Vector4d point_p = point4d + Eigen::Vector4d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0, d == 3 ? dp : 0);
      Eigen::Vector4d point_m = point4d - Eigen::Vector4d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0, d == 3 ? dp : 0);
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->projectHomogeneous(point_p, &imagePoint_p);
      camera->projectHomogeneous(point_m, &imagePoint_m);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    //std::cout << "==========\n J:\n" << J << std::endl;
    //std::cout << "==========\n J_numDiff:\n" << J_numDiff << std::endl;
    // Check Jacobian Error Norm
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_numDiff - J).norm() < 0.0001,"Point Jacobian Verification failed");

    // check intrinsics Jacobian by numeric differences
    const int numIntrinsics = camera->noIntrinsicsParameters();
    Eigen::Matrix2Xd J_intrinsics_numDiff;
    J_intrinsics_numDiff.resize(2,numIntrinsics);
    // back-up original intrinsics
    Eigen::VectorXd original_intrinsics;
    camera->getIntrinsics(original_intrinsics);

    for (size_t d = 0; d < 6; ++d) {
      Eigen::VectorXd dpVec;
      dpVec.resize(numIntrinsics);
      dpVec.setZero();
      dpVec[d] = dp;
      Eigen::VectorXd intrinsics_p = original_intrinsics + dpVec;
      Eigen::VectorXd intrinsics_m = original_intrinsics - dpVec;
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->setIntrinsics(intrinsics_p);
      camera->projectHomogeneous(point4d, &imagePoint_p);
      camera->setIntrinsics(intrinsics_m);
      camera->projectHomogeneous(point4d, &imagePoint_m);
      J_intrinsics_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
      // restore original intrinsics
      camera->setIntrinsics(original_intrinsics);
    }

    // Assert correctness of intrinsics jacobian
    //std::cout << "==========\n J_intrinsics:\n" << J_intrinsics << std::endl;
    //std::cout << "==========\n J_intrinsics_numDiff:\n" << J_intrinsics_numDiff << std::endl;
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_intrinsics_numDiff - J_intrinsics).norm() < 0.0001,"Intrinsics Jacobian Verification failed");

    // check re-projection error
    OKVIS_ASSERT_TRUE(std::runtime_error, (imagePoint2 - imagePoint).norm() < 0.01,"project/unproject failure");
  }
}
TEST(EucmCamera, projectHomogeneousWithExternalParameters){
  const size_t NUM_POINTS = 100;

  // Instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Get Intrinsics
  Eigen::VectorXd intrinsics;
  camera->getIntrinsics(intrinsics);

  for(size_t i = 0; i < NUM_POINTS; ++i){

    // create random visible point
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

    // backProject
    Eigen::Vector4d point4d;
    bool backStat = camera->backProjectHomogeneous(imagePoint, &point4d);
    // Randomize homogeneous coordinate
    double factor = (double)(std::rand()) / RAND_MAX;
    point4d *= factor;

    OKVIS_ASSERT_TRUE(std::runtime_error, backStat,"unsuccessful back projection");

    // re-project (with external parameters)
    Eigen::Matrix<double, 2, 4> J;
    Eigen::Matrix2Xd J_intrinsics;
    Eigen::Vector2d imagePoint2;
    okvis::cameras::ProjectionStatus stat = camera->projectHomogeneousWithExternalParameters(point4d, intrinsics, &imagePoint2, &J, &J_intrinsics);
    OKVIS_ASSERT_TRUE(std::runtime_error,stat == okvis::cameras::ProjectionStatus::Successful,"unsuccessful projection");

    // check point Jacobian by numeric differences
    const double dp = 1.0e-7;
    Eigen::Matrix<double, 2, 4> J_numDiff;
    for (size_t d = 0; d < 4; ++d) {
      Eigen::Vector4d point_p = point4d + Eigen::Vector4d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0, d == 3 ? dp : 0);
      Eigen::Vector4d point_m = point4d - Eigen::Vector4d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0, d == 3 ? dp : 0);
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->projectHomogeneousWithExternalParameters(point_p, intrinsics, &imagePoint_p, &J);
      camera->projectHomogeneousWithExternalParameters(point_m, intrinsics, &imagePoint_m, &J);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    // assert correctness of point Jacobian
    //std::cout << "==========\n J:\n" << J << std::endl;
    //std::cout << "==========\n J_numDiff:\n" << J_numDiff << std::endl;"
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_numDiff - J).norm() < 0.0001,"Point Jacobian Verification failed");

    // check intrinsics Jacobian by numeric differences
    const int numIntrinsics = camera->noIntrinsicsParameters();
    Eigen::Matrix2Xd J_intrinsics_numDiff;
    J_intrinsics_numDiff.resize(2,numIntrinsics);
    for (size_t d = 0; d < 6; ++d) {
      Eigen::VectorXd dpVec;
      dpVec.resize(numIntrinsics);
      dpVec.setZero();
      dpVec[d] = dp;
      Eigen::VectorXd intrinsics_p = intrinsics + dpVec;
      Eigen::VectorXd intrinsics_m = intrinsics - dpVec;
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->projectHomogeneousWithExternalParameters(point4d, intrinsics_p, &imagePoint_p, &J);
      camera->projectHomogeneousWithExternalParameters(point4d, intrinsics_m, &imagePoint_m, &J);

      J_intrinsics_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }

    // Assert correctness of intrinsics jacobian
    //std::cout << "==========\n J_intrinsics:\n" << J_intrinsics << std::endl;
    //std::cout << "==========\n J_intrinsics_numDiff:\n" << J_intrinsics_numDiff << std::endl;"
    OKVIS_ASSERT_TRUE(std::runtime_error, (J_intrinsics_numDiff - J_intrinsics).norm() < 0.0001,"Intrinsics Jacobian Verification failed");

    // check also re-projection error
    OKVIS_ASSERT_TRUE(std::runtime_error, (imagePoint2 - imagePoint).norm() < 0.01,"project/unproject failure");
  }
}
TEST(EucmCamera, batchHomogeneousProjections){
  const size_t NUM_POINTS = 100;
  Eigen::Matrix2Xd imagePointsBatch;
  imagePointsBatch.resize(2, NUM_POINTS);
  Eigen::Matrix4Xd backProjectBatch;
  backProjectBatch.resize(4, NUM_POINTS);
  Eigen::Matrix2Xd reProjectBatch;
  reProjectBatch.resize(2,NUM_POINTS);

  // instantiate camera
  std::shared_ptr<okvis::cameras::CameraBase> camera = okvis::cameras::EucmCamera::createTestObject();

  // Create batch of image points
  for(size_t i = 0; i < NUM_POINTS; ++i) {

    // create a random point in the field of view:
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();
    imagePointsBatch.col(i) = imagePoint;
  }

  // backProject Batch
  std::vector<bool> success;
  camera->backProjectHomogeneousBatch(imagePointsBatch, &backProjectBatch, &success);


  // Re-Project batch
  std::vector<okvis::cameras::ProjectionStatus> stati;
  camera->projectHomogeneousBatch(backProjectBatch, &reProjectBatch, &stati);

  // Test reprojection error
  for(size_t i = 0; i < NUM_POINTS; ++i){
    double reProjectError = (imagePointsBatch.col(i)-reProjectBatch.col(i)).norm();
    OKVIS_ASSERT_TRUE(std::runtime_error, reProjectError < 0.01,"project/unproject failure for batchwise projections");
  }
}