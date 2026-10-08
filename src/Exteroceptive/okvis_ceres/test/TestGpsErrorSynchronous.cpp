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

#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ceres/GpsErrorSynchronous.hpp>
#include <okvis/ceres/RelativePoseError.hpp>

TEST(okvisTestSuite, GpsErrorSynchronous){

    // Build the problem.
    ::ceres::Problem problem;

    // GPS parameters
    okvis::GpsParameters gpsParameters;
    Eigen::Vector3d r_SA(0.3,0.3,0.3);
    //r_SA.setZero(); base case where antenna is in IMU sensor frame
    gpsParameters.r_SA = r_SA;

    // Set up random transformation GPS <-> world
    std::cout << "Set up random trafo GPS <-> world" << std::flush;
    okvis::kinematics::Transformation T_GW; // GPS to workd
    T_GW.setRandom(10.0, M_PI);
    okvis::kinematics::Transformation T_disturb;
    T_disturb.setRandom(1, 0.01);
    okvis::kinematics::Transformation T_GW_init=T_GW*T_disturb; // initial estimate for ceres solver
    // Create parameter block for pose GPS <-> world and add to problem
    okvis::ceres::PoseParameterBlock poseParameterBlock(T_GW_init,1,okvis::Time(0));
    problem.AddParameterBlock(poseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
    problem.SetParameterBlockVariable(poseParameterBlock.parameters()); // Optimize this!
    // Set local parametrization
    ::ceres::LocalParameterization* poseLocalParameterization = new okvis::ceres::PoseLocalParameterization;
    problem.SetParameterization(poseParameterBlock.parameters(),poseLocalParameterization);

    std::cout << " [OK] " << std::endl;

    // Create random points
    const size_t N = 100;
    std::cout << "create N=" << N << " visible points and add respective error terms... " << std::flush;
    for(int i = 0; i < N; i++){


        // Initialize Random Robot Pose
        okvis::kinematics::Transformation T_WS;
        T_WS.setRandom(10.0, M_PI);
        // disturb robot pose
        okvis::kinematics::Transformation T_dist_WS;
        T_dist_WS.setRandom(0.5, M_PI);
        okvis::kinematics::Transformation T_WS_init(T_dist_WS*T_WS);

        // True GPS Signal
        Eigen::Vector3d position_G = T_GW.C() * (T_WS.r() + T_WS.C()*r_SA)+ T_GW.r();

        // add a parameter block for robots position
        okvis::ceres::PoseParameterBlock* robotPoseParameterBlock = new okvis::ceres::PoseParameterBlock(T_WS_init,1,okvis::Time(0));
        problem.AddParameterBlock(robotPoseParameterBlock->parameters(), okvis::ceres::PoseParameterBlock::Dimension);
        // problem.SetParameterBlockConstant(robotPoseParameterBlock->parameters()); // do not optimize this
        problem.SetParameterBlockVariable(robotPoseParameterBlock->parameters());
        problem.SetParameterization(robotPoseParameterBlock->parameters(),poseLocalParameterization);

        // Set up the only cost function (also known as residual).
        Eigen::Matrix3d information=Eigen::Matrix3d::Identity();
        ::ceres::CostFunction* cost_function = new okvis::ceres::GpsErrorSynchronous(1,position_G,information, gpsParameters);
        problem.AddResidualBlock(cost_function, NULL, robotPoseParameterBlock->parameters(), poseParameterBlock.parameters());

        // Relative Pose Error to be considered
        okvis::ceres::PoseParameterBlock* refParameterBlock = new okvis::ceres::PoseParameterBlock(T_WS,1,okvis::Time(0));
        problem.AddParameterBlock(refParameterBlock->parameters(), okvis::ceres::PoseParameterBlock::Dimension);
        problem.SetParameterBlockConstant(refParameterBlock->parameters()); // keep ref point constant
        problem.SetParameterization(refParameterBlock->parameters(),poseLocalParameterization);
        // Minimize relative error
        Eigen::Matrix<double,6,6> information_rel;
        information_rel.setIdentity();
        okvis::kinematics::Transformation T_rel;
        T_rel.setIdentity();

        ::ceres::CostFunction* cost_function_rel = new okvis::ceres::RelativePoseError(information_rel,T_rel);
        problem.AddResidualBlock(cost_function_rel,NULL,robotPoseParameterBlock->parameters(), refParameterBlock->parameters());


    }

    std::cout << " [OK] " << std::endl;

    // Run the solver!
    std::cout << "run the solver... " << std::endl;
    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::FLAGS_stderrthreshold=google::WARNING; // enable console warnings (Jacobian verification)
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // print some infos about the optimization
    //std::cout << summary.BriefReport() << "\n";
    std::cout << "initial T_GW : " << T_GW_init.T() << "\n"
                    << "optimized T_GW : " << poseParameterBlock.estimate().T() << "\n"
                    << "correct T_GW : " << T_GW.T() << "\n"
                    << " where Translation between antenna and sensor is: " << r_SA << std::endl;

    // make sure it converged
    EXPECT_TRUE(2*(T_GW.q()*poseParameterBlock.estimate().q().inverse()).vec().norm()<1e-2);
    EXPECT_TRUE((T_GW.r()-poseParameterBlock.estimate().r()).norm()<1e-1);

}
