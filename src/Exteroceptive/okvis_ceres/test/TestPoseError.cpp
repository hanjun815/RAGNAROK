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

#include <memory>
#include <sys/time.h>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/cost_function.h>
#include <ceres/crs_matrix.h>
#include <ceres/evaluation_callback.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/ordered_groups.h>
#include <ceres/problem.h>
#include <ceres/product_manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <ceres/version.h>
#pragma GCC diagnostic pop

#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

TEST(okvisTestSuite, PoseError)
{
  // initialize random number generator
  //srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  ::ceres::Problem problem;

  // let's use our own local quaternion perturbation
  okvis::ceres::PoseManifold* poseLocalParameterization = new okvis::ceres::PoseManifold;

  // create pose and measurement of it
  okvis::kinematics::Transformation T;
  T.setRandom(); // get a random transformation
  okvis::kinematics::Transformation T_measured = T;
  okvis::kinematics::Transformation dT;
  dT.setRandom(0.1,0.01); // a realistic disturbance
  T_measured = T_measured * dT; // create a disturbed measurement
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
        new okvis::ceres::PoseParameterBlock(T, 1, okvis::Time(0)));
  problem.AddParameterBlock(poseParameterBlock->parameters(), 7, poseLocalParameterization);

  // Set up the only cost function.
  Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Identity();
  ::ceres::CostFunction* costFunction = new okvis::ceres::PoseError(T_measured, information);
  auto id = problem.AddResidualBlock(costFunction, nullptr, poseParameterBlock->parameters());

  // check Jacobian
  OKVIS_ASSERT_TRUE(
        Exception, okvis::ceres::jacobiansCorrect(&problem, id), "Jacobian verification failed")

  // run the optimiser just for fun
  ::ceres::Solver::Options options;
  //options.minimizer_progress_to_stdout = true;
  //options.check_gradients = true;
  //options.gradient_check_relative_precision = 1.0e-6;
  options.max_num_iterations = 10;
  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << std::endl;

  // check
  OKVIS_ASSERT_TRUE(Exception,
                    summary.termination_type != ::ceres::TerminationType::FAILURE,
                    "optimisation failed")

}

