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
#include "ceres/covariance.h"
#include <gtest/gtest.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ceres/GpsErrorAsynchronous.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

// Define error thresholds
const double jacobianThresh = 1e-03; // Tolerance for error of analytic Jacobian compared to numerical Jacobian
const double posThresh = 1e-01; // Position estimate tolerance [m]
const double rotThresh = 1e-02; // Orientation estimate tolerance [°] ; 1e-02 rad = 0.57 °
const double velThresh = 5e-02; // Velocity estimate threshold [m/s]
const double bgThresh = 1e-03; // Gyroscope bias estimate threshold [rad/s]
const double baThresh = 1e-03; // Acclerometer bias estimate threshold [m/s²]

// SpeedAndBias Signal disturbances
const double bg_dist_std = 2e-03; // Gyroscope bias disturbance (standard deviation) 2e-03 [rad/s] => 0.57 [°]
const double ba_dist_std = 1e-02; // Accelerometer bias disturbance (standard deviation) [m/s²]
const double v_dist_std = 0.1;  // SpeedAndBias Velocity disturbance [m/s]

// GPS uncertainty
const double gps_std = 0.02; // GPS accuracy uncertainty [m] ==> 2cm = RTK GPS accuracy

// Odometry uncertainties (relative Pose errors)
const double odometry_trans_std = 2e-02; // 2 cm odometry error
const double odometry_rot_std = 4e-02;  // 2.3 degree odometry error

// Trajectory parameters (circular motion with constant speed)
const double omega = 0.2; // rotation rate 0.2 [rad/s] => 11.5 [°/s]
const double radius = 5.0; // radius

// sampling rates
const double gpsRate = 100.0; // 50 Hz
const double frameRate = 80.0; // 40 Hz


TEST(okvisTestSuite, GpsError){

    Eigen::Vector3d omegaVec(omega,omega,omega); // in world frame
    Eigen::Vector3d p_c(0.0,0.0,1.0); // center point of circle
    Eigen::Vector3d p0 = p_c + Eigen::Vector3d(0,radius,0); // starting point at (0,r,1)

    // set the imu parameters
    const double imuRate = 1000; // 1 kHz
    const double duration = 10.0; // duration
    okvis::ImuMeasurementDeque imuMeasurements;
    // time increment
    const double dt=1.0/double(imuRate); // time discretization

    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;

    // Set the gps parameters
    okvis::GpsParameters gpsParameters;
    Eigen::Vector3d r_SA(0.2,0.3,0.1);
    //r_SA.setZero();
    gpsParameters.r_SA = r_SA;

    // counter variables
    size_t countGps=0;
    size_t countFrames=0;

    // Build the problem.
    ::ceres::Problem problem;
    std::cout << "Setting up test case (circular movement) with \n"
              << "Duration: " << duration << "\n"
              << "IMU rate: " << imuRate << "\n"
              << "GPS rate: " << gpsRate << "\n"
              << "Frame rate: " << frameRate << "\n"
              << "Verifying Jacobians and State Estimation Results..." << std::endl;

    // Set up random transformation GPS <-> world
    okvis::kinematics::Transformation T_GW;
    T_GW.setRandom(10.0, M_PI); // Ground Truth
    okvis::kinematics::Transformation T_disturb;
    T_disturb.setRandom(2.0, M_PI/2.0);
    okvis::kinematics::Transformation T_GW_init=T_GW*T_disturb; // initial estimate for ceres solver

    // Create parameter block for pose GPS <-> world and add to problem
    okvis::ceres::PoseParameterBlock gpsPoseParameterBlock(T_GW_init,0,okvis::Time(0));
    // Set local parametrization
    auto* poseLocalParameterization = new okvis::ceres::PoseManifold();
    problem.AddParameterBlock(gpsPoseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension, poseLocalParameterization);

    // initialize states
    okvis::kinematics::Transformation T_WS(p0, Eigen::Quaterniond::Identity());
    Eigen::Quaterniond q = T_WS.q();
    Eigen::Vector3d r = T_WS.r();

    Eigen::Vector3d v_w = omegaVec.cross(r);
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    speedAndBias.head<3>() = v_w; // speed and biases are in world frame


    // DO THE PROPAGATION AND INTEGRATION
    double time = 0;
    okvis::Time tkPrev(time);
    okvis::Time tk(time); // Time of camera frame
    okvis::Time tg(time);
    okvis::kinematics::Transformation T_kPrev;
    okvis::kinematics::Transformation T_k;
    okvis::kinematics::Transformation T_k_dist;
    okvis::SpeedAndBias sb_k;
    okvis::SpeedAndBias sb_k_dist;

    // Vectors of parameter blocks
    std::vector< std::shared_ptr< okvis::ceres::PoseParameterBlock> > robotPoseParameterBlocks;
    std::vector< std::shared_ptr< okvis::ceres::SpeedAndBiasParameterBlock> > speedAndBiasParameterBlocks;
    std::vector< okvis::kinematics::Transformation> gtRobotPoses;
    std::vector< okvis::SpeedAndBias> gtSbs;

    for(size_t i=0; i<size_t(duration*imuRate); ++i){

        time = double(i)/imuRate; // current time in interval

        // --- IMU Signal creation ---

        // Compute velocity vector
        v_w = omegaVec.cross(r); // v = omega x r
        // Compute acceleration vector
        Eigen::Vector3d a_W = omegaVec.cross(v_w);

        // Propagate position
        r += v_w*dt;

        // Propagate Orientation
        Eigen::Quaterniond dq;
        const double alpha_half = omegaVec.norm()*dt*0.5;
        const double sinc_alpha_half = okvis::kinematics::sinc(alpha_half);
        const double cos_alpha_half = cos(alpha_half);
        dq.vec() = sinc_alpha_half * 0.5 * omegaVec * dt;
        dq.w() = cos_alpha_half;
        q = q * dq;

        // update GT T_WS
        T_WS = okvis::kinematics::Transformation(r,q);

        // speedAndBias - v only, obviously, since this is the Ground Truth
        speedAndBias.head<3>() = v_w;// velocity in world frame

        // generate measurements
        Eigen::Vector3d gyr = omegaVec + imuParameters.sigma_g_c/sqrt(dt)*Eigen::Vector3d::Random();
        Eigen::Vector3d acc = T_WS.inverse().C()*(a_W+Eigen::Vector3d(0,0,imuParameters.g)) + imuParameters.sigma_a_c/sqrt(dt)*Eigen::Vector3d::Random();
        imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(time),okvis::ImuSensorReadings(gyr,acc)));

        if (time > countFrames/frameRate /*std::fmod(time,1.0)==0*/){

            // Save previous pose and previous time (needed for relative pose error)
            T_kPrev = T_k;
            T_k = T_WS;
            tkPrev = tk;
            tk = okvis::Time(time);
            sb_k = speedAndBias;

            // Disturb states
            T_disturb.setRandom(1, M_PI);
            T_k_dist = T_k*T_disturb;
            sb_k_dist = speedAndBias;
            sb_k_dist.head<3>() += v_dist_std * Eigen::Vector3d::Random();
            sb_k_dist.segment<3>(3) += bg_dist_std * Eigen::Vector3d::Random();
            sb_k_dist.tail<3>() += ba_dist_std * Eigen::Vector3d::Random();

            // Create parameter blocks (poses and speed and biases
            std::shared_ptr<okvis::ceres::PoseParameterBlock> robotPoseParameterBlock(new okvis::ceres::PoseParameterBlock(T_k_dist,0,tk));
            robotPoseParameterBlocks.push_back(robotPoseParameterBlock);
            gtRobotPoses.push_back(T_k); // save gt pose
            std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(sb_k_dist,0,tk));
            speedAndBiasParameterBlocks.push_back(speedAndBiasParameterBlock);
            gtSbs.push_back(sb_k); // save gt sb

            // increase counter
            ++countFrames;
        }
        if(time > countGps/gpsRate/*std::fmod(time,1.0) == 0.5*/){

            // GPS signal
            tg = okvis::Time(time);
            // create GPS position measurement
            Eigen::Vector3d r_G = T_GW.C() * (T_WS.r() + T_WS.C()*r_SA) + T_GW.r();

            // Disturb measurement
            r_G += gps_std * Eigen::Vector3d::Random();

            // add RobotPose Parameter block
            robotPoseParameterBlocks.back()->setLocalParameterizationPtr(poseLocalParameterization); 
            problem.AddParameterBlock(robotPoseParameterBlocks.back()->parameters(), okvis::ceres::PoseParameterBlock::Dimension, poseLocalParameterization);
            problem.SetParameterBlockVariable(robotPoseParameterBlocks.back()->parameters()); // Optimize this
            // add SpeedAndBias Parameter block
            problem.AddParameterBlock(speedAndBiasParameterBlocks.back()->parameters(),okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
            problem.SetParameterBlockVariable(speedAndBiasParameterBlocks.back()->parameters()); // Optimize this
            // Add Residual Block
            //Eigen::Matrix3d cov_gps = pow(gps_std,2.0)*Eigen::Matrix3d::Identity();
            ::ceres::CostFunction* cost_function = new okvis::ceres::GpsErrorAsynchronous(r_G, gps_std, gps_std, gps_std, imuMeasurements, imuParameters, tk, tg, gpsParameters);
            problem.AddResidualBlock(cost_function, NULL, robotPoseParameterBlocks.back()->parameters(), speedAndBiasParameterBlocks.back()->parameters(), gpsPoseParameterBlock.parameters());

            // increase counter
            ++countGps;



            // JACOBIAN CHECK FOR GPS ERROR ASYNCHRONOUS CLASS
            if(true){

                double* parameters[3];
                parameters[0]=robotPoseParameterBlocks.back()->parameters();
                parameters[1]=speedAndBiasParameterBlocks.back()->parameters();
                parameters[2]=gpsPoseParameterBlock.parameters();
                double* jacobians[3];
                Eigen::Matrix<double,3,7,Eigen::RowMajor> J0;
                Eigen::Matrix<double,3,9,Eigen::RowMajor> J1;
                Eigen::Matrix<double,3,7,Eigen::RowMajor> J2;
                jacobians[0]=J0.data();
                jacobians[1]=J1.data();
                jacobians[2]=J2.data();
                Eigen::Matrix<double,3,1> residuals;
                // evaluate twice to be sure that we will be using the linearisation of the biases (i.e. no preintegrals redone)
                static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,NULL);
                static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,NULL);

                // and now num-diff:
                double dx=1e-6;

                // w.r.t. robot pose
                Eigen::Matrix<double,3,6> J0_numDiff;
                for(size_t i=0; i<6; ++i){
                  Eigen::Matrix<double,6,1> dp_0;
                  Eigen::Matrix<double,3,1> residuals_p;
                  Eigen::Matrix<double,3,1> residuals_m;
                  dp_0.setZero();
                  dp_0[i]=dx;
                  poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_p.data(),NULL);
                  robotPoseParameterBlocks.back()->setEstimate(T_k_dist); // reset
                  dp_0[i]=-dx;
                  poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_m.data(),NULL);
                  robotPoseParameterBlocks.back()->setEstimate(T_k_dist); // reset
                  J0_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

                }

                // Use lift Jacobian for non-minimal Jacobian
                Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0_numDiff_lift;
                Eigen::Matrix<double, 6, 7, Eigen::RowMajor> liftJac0;
                okvis::ceres::PoseManifold::minusJacobian(parameters[0], liftJac0.data());
                J0_numDiff_lift = J0_numDiff * liftJac0;

                EXPECT_TRUE((J0_numDiff_lift-J0).norm() < jacobianThresh) << " Jacobian Evaluation leads error  " << (J0_numDiff_lift-J0).norm() << " > " << jacobianThresh << std::endl;

                // w.r.t. SpeedAndBias
                Eigen::Matrix<double,3,9> J1_numDiff;
                for(size_t i=0; i<9; ++i){
                  Eigen::Matrix<double,9,1> dp_1;
                  Eigen::Matrix<double,3,1> residuals_p;
                  Eigen::Matrix<double,3,1> residuals_m;
                  dp_1.setZero();
                  dp_1[i]=dx;
                  speedAndBiasParameterBlocks.back()->plus(parameters[1],dp_1.data(),parameters[1]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_p.data(),NULL);
                  speedAndBiasParameterBlocks.back()->setEstimate(sb_k_dist); // reset
                  dp_1[i]=-dx;
                  speedAndBiasParameterBlocks.back()->plus(parameters[1],dp_1.data(),parameters[1]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_m.data(),NULL);
                  speedAndBiasParameterBlocks.back()->setEstimate(sb_k_dist); // reset
                  J1_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

                }

//                std::cout << "Jacobian evaluates to: \n" << J1 << std::endl;
//                std::cout << "Numerical Jacobian evaluates to: \n" << J1_numDiff << std::endl;

                EXPECT_TRUE((J1_numDiff-J1).norm() < jacobianThresh) << " Jacobian Evaluation leads error  " << (J1_numDiff-J1).norm() << " > " << jacobianThresh << std::endl;

                // w.r.t GPS trafo
                Eigen::Matrix<double,3,6> J2_numDiff;
                for(size_t i=0; i<6; ++i){
                  Eigen::Matrix<double,6,1> dp_2;
                  Eigen::Matrix<double,3,1> residuals_p;
                  Eigen::Matrix<double,3,1> residuals_m;
                  dp_2.setZero();
                  dp_2[i]=dx;
                  poseLocalParameterization->Plus(parameters[2],dp_2.data(),parameters[2]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_p.data(),NULL);
                  gpsPoseParameterBlock.setEstimate(T_GW_init); // reset
                  dp_2[i]=-dx;
                  poseLocalParameterization->Plus(parameters[2],dp_2.data(),parameters[2]);
                  static_cast<okvis::ceres::GpsErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_m.data(),NULL);
                  gpsPoseParameterBlock.setEstimate(T_GW_init); // reset
                  J2_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

                }

                // Use lift Jacobian for non-minimal Jacobian
                Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J2_numDiff_lift;
                Eigen::Matrix<double, 6, 7, Eigen::RowMajor> liftJac2;
                okvis::ceres::PoseManifold::minusJacobian(parameters[2], liftJac2.data());
                J2_numDiff_lift = J2_numDiff * liftJac2;

                EXPECT_TRUE((J2_numDiff_lift-J2).norm() < jacobianThresh) << " Jacobian Evaluation leads error  " << (J2_numDiff_lift-J2).norm() << " > " << jacobianThresh << std::endl;

            }


            // Add constraints between successive poses
            if(robotPoseParameterBlocks.size() > 1){

                // Constrain relative poses
                okvis::kinematics::Transformation T_rel;
                T_rel = T_k.inverse() * T_kPrev;
                okvis::kinematics::Transformation T_rel_dist;
                T_rel_dist.setRandom(odometry_trans_std, odometry_rot_std);
                ::ceres::CostFunction* cost_function_rel = new okvis::ceres::RelativePoseError(pow(odometry_trans_std,2.0), pow(odometry_rot_std,2.0),T_rel*T_rel_dist);
                problem.AddResidualBlock(cost_function_rel,NULL, robotPoseParameterBlocks.back()->parameters(), robotPoseParameterBlocks.at(robotPoseParameterBlocks.size()-2)->parameters());


            }
            else if(robotPoseParameterBlocks.size() == 1){

                // For first pose add prior knowledge of position
                ::ceres::CostFunction* cost_function_abs = new okvis::ceres::PoseError(T_k,1e-06,1e-06);
                problem.AddResidualBlock(cost_function_abs, NULL, robotPoseParameterBlocks.back()->parameters());
            }

            // Add constraints for speed and biases --> IMU error
            if(speedAndBiasParameterBlocks.size() > 1){

                okvis::ceres::ImuError* cost_function_imu = new okvis::ceres::ImuError(imuMeasurements, imuParameters, tkPrev, tk);
                problem.AddResidualBlock(cost_function_imu, NULL,
                                         robotPoseParameterBlocks.at(robotPoseParameterBlocks.size()-2)->parameters(),
                                         speedAndBiasParameterBlocks.at(speedAndBiasParameterBlocks.size()-2)->parameters(),
                                         robotPoseParameterBlocks.back()->parameters(),
                                         speedAndBiasParameterBlocks.back()->parameters());
            }
            else if(speedAndBiasParameterBlocks.size() == 1){
                ::ceres::CostFunction* cost_function_speedAndBias = new okvis::ceres::SpeedAndBiasError(sb_k,1e-06,1e-06,1e-06); // speed and biases prior...
                problem.AddResidualBlock(cost_function_speedAndBias, NULL,speedAndBiasParameterBlocks.back()->parameters());
            }

        }


      }



    // Compute covariances
//    ::ceres::Covariance::Options covOptions;
//    //covOptions.algorithm_type = ::ceres::DENSE_SVD;
//    ::ceres::Covariance covariance(covOptions);
//    std::vector<std::pair<const double*, const double*> > covariance_blocks;
//    covariance_blocks.push_back(std::make_pair(gpsPoseParameterBlock.parameters() , gpsPoseParameterBlock.parameters()));
//    covariance.Compute(covariance_blocks,&problem);
//    std::cout << "Covariances computed." << std::endl;
//    double covariance_gg[6*6];
//    std::cout << "Trying to assess covariance blocks" << std::endl;
//    covariance.GetCovarianceBlockInTangentSpace(gpsPoseParameterBlock.parameters() , gpsPoseParameterBlock.parameters() , covariance_gg);
//    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> covOutput(covariance_gg);
//    std::cout << "covariance has been computed and is: \n" << covOutput << std::endl;

    // Run the solver!
    std::cout << "run the solver... " << std::endl;

    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::FLAGS_stderrthreshold=google::WARNING; // enable console warnings (Jacobian verification)
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Verify correctness of estimated poses
    for(size_t j = 0; j<gtRobotPoses.size(); j++){
        EXPECT_TRUE((gtRobotPoses.at(j).r() - robotPoseParameterBlocks.at(j)->estimate().r()).norm() < posThresh)
                << " Estimated robot position has an error   "
                << (gtRobotPoses.at(j).r() - robotPoseParameterBlocks.at(j)->estimate().r()).norm()
                << " > " << posThresh << std::endl;
        EXPECT_TRUE(2*(gtRobotPoses.at(j).q() * robotPoseParameterBlocks.at(j)->estimate().q().inverse()).vec().norm() < rotThresh)
                << " Estimated robot orientation has an error   "
                << 2*(gtRobotPoses.at(j).q() * robotPoseParameterBlocks.at(j)->estimate().q().inverse()).vec().norm()
                << " > " << rotThresh << std::endl;
    }
    // Verify correctness of estimated speed and biases
    for(size_t jj = 0; jj<speedAndBiasParameterBlocks.size(); jj++){
        double velocityError = (gtSbs.at(jj).head<3>() - speedAndBiasParameterBlocks.at(jj)->estimate().head<3>()).norm();
        double bgError = (gtSbs.at(jj).segment<3>(3) - speedAndBiasParameterBlocks.at(jj)->estimate().segment<3>(3)).norm();
        double baError = (gtSbs.at(jj).tail<3>() - speedAndBiasParameterBlocks.at(jj)->estimate().tail<3>()).norm();
        EXPECT_TRUE(velocityError < velThresh)
                << " Velocity error of   " << velocityError << " > " << velThresh << std::endl;
        EXPECT_TRUE(bgError < bgThresh)
                << " Gyr bias error of   " << bgError << " > " << bgThresh << std::endl;
        EXPECT_TRUE(baError <baThresh)
                << " Acc bias error of   " << baError << " > " << baThresh << std::endl;
    }
    // Verify correctness of estimated trafo GPS <-> world
    EXPECT_TRUE(2*(T_GW.q()*gpsPoseParameterBlock.estimate().q().inverse()).vec().norm() < rotThresh)
            << " Rotational component of trafo GPS <-> world has error   "
            << 2*(T_GW.q()*gpsPoseParameterBlock.estimate().q().inverse()).vec().norm()
            << " > " << rotThresh << std::endl;
    EXPECT_TRUE((T_GW.r()-gpsPoseParameterBlock.estimate().r()).norm() < posThresh)
            << " Translational component of trafo GPS <-> world has error   "
            << (T_GW.r()-gpsPoseParameterBlock.estimate().r()).norm()
            << " > " << posThresh << std::endl;



}
