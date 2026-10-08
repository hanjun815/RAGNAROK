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

#include <gtest/gtest.h>
#include <okvis/ViGraphEstimator.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/GpsErrorAsynchronous.hpp>
#include <okvis/assert_macros.hpp>


TEST(okvisTestSuite, EstimatorGpsError) {
//    srand((unsigned int) time(NULL));
    srand(1);

    OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

    const double DURATION = 10.0;  // 10 seconds motion
    const double IMU_RATE = 1000.0;  // 1 kHz
    const double DT = 1.0 / IMU_RATE;  // time increments
    const double OMEGA = 0.2; // rotation rate 0.2 [rad/s] => 11.5 [°/s]
    const double RADIUS = 5.0; // radius
    // const size_t K = 8; // number of camera frames
    //const size_t G = 10; // number of GPS Measurements
    const double GPS_RATE = 4.5; // Hz 4.5 for best test case
    const double FRAME_RATE = 4; // Hz 4 for best test case
    // for 1 (GPS) and 0.8 (FRAME) T_GW will never be observable
    const double GPS_STD = 5e-02;

    std::cout << "Simulating circular movement with IMU Rate: " << IMU_RATE << ", Frame Rate: " << FRAME_RATE << " and GPS Rate: " << GPS_RATE
              << " with a GPS uncertainty: " << GPS_STD << " for a total of " << DURATION << " seconds" << std::endl;

    // set the imu parameters
    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;
    imuParameters.sigma_bg = 0.001;
    imuParameters.sigma_ba = 0.001;

    // Set the gps parameters
    okvis::GpsParameters gpsParameters;
    Eigen::Vector3d r_SA(0.2,0.3,0.1);
    //r_SA.setZero();
    gpsParameters.r_SA = r_SA;
    // Set up random transformation GPS <-> world
    // Set up a random yaw rotation (only 4D parametrization of T_GW considered)
    Eigen::Vector3d randomTrans;
    randomTrans = 10*Eigen::Vector3d::Random();
    Eigen::Vector3d yawAxis(0.0, 0.0, 1.0);
    Eigen::AngleAxisd randomRot(M_PI / 4.0, yawAxis);
    Eigen::Matrix4d randomT;
    randomT.block<3,3>(0,0) = randomRot.toRotationMatrix();
    randomT.block<3,1>(0,3) = randomTrans;
    okvis::kinematics::Transformation T_GW;
    //T_GW.setRandom(10.0, 0.0); // Ground Truth
    T_GW.set(randomT);


    T_GW.C() = randomRot.toRotationMatrix();
    std::cout << "Ground-Truth T_GW : \n" << T_GW.T() << std::endl;

    // Initialize Measurement Queues
    okvis::ImuMeasurementDeque imuMeasurements;
    okvis::GpsMeasurementDeque gpsMeasurements;
    imuMeasurements.clear();
    gpsMeasurements.clear();


    // camera extrinsics:
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(new okvis::kinematics::Transformation(Eigen::Vector3d(0,-0.1,0.1),Eigen::Quaterniond(-sqrt(0.5),0,0,sqrt(0.5))));
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_1(new okvis::kinematics::Transformation(Eigen::Vector3d(0,0.1,0.1),Eigen::Quaterniond(-sqrt(0.5),0,0,sqrt(0.5))));
    // some parameters on how to do the online estimation:
    okvis::CameraParameters cameraParameters;
    cameraParameters.online_calibration.do_extrinsics = false;


    // set up camera with intrinsics
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0(okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry1(okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());

    // create an N-camera system:
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(new okvis::cameras::NCameraSystem);
    cameraSystem->addCamera(T_SC_0, cameraGeometry0, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    cameraSystem->addCamera(T_SC_1, cameraGeometry1,okvis::cameras::NCameraSystem::DistortionType::Equidistant);

    // create an Estimator
    okvis::ViGraphEstimator viGraph2;

    // create landmark grid
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > homogeneousPoints;
    std::vector<okvis::LandmarkId> lmIds;
    for (double x = - RADIUS - 5.0; x <= RADIUS + 5.0; x += 2.5) { // 2.5 for best test case
      for (double y = - RADIUS - 5.0; y <= RADIUS + 5.0 ; y += 2.5) {
          for (double z = -RADIUS -5.0; z <= RADIUS + 5.0; z+= 2.5){
              homogeneousPoints.push_back(Eigen::Vector4d(x, y, z, 1));
              lmIds.push_back(viGraph2.addLandmark(homogeneousPoints.back(), true));
          }
      }
    }

    std::cout << "Created " << lmIds.size() << " landmarks." << std::endl;

    // add sensors
    viGraph2.addCamera(cameraParameters);
    viGraph2.addCamera(cameraParameters);
    viGraph2.addImu(imuParameters);
    viGraph2.addGps(gpsParameters);


    // Generate Motion: circular trajectory with constant speed
    Eigen::Vector3d omegaVec(OMEGA,OMEGA,OMEGA); // in world frame
    Eigen::Vector3d p_c(-RADIUS,0.0,0.0);// center point of circle
    Eigen::Vector3d p0 = Eigen::Vector3d(0,0,0); // starting point at (0,0,0)
    // initialize states
    okvis::kinematics::Transformation T_WS(p0, Eigen::Quaterniond::Identity());
    Eigen::Quaterniond q = T_WS.q();
    Eigen::Vector3d r = T_WS.r();
    Eigen::Vector3d v_w = omegaVec.cross(r-p_c);
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    speedAndBias.head<3>() = v_w; // speed and biases are in world frame

    double time = 0;
    size_t k = 0; // counter for camera frames
    size_t g = 0; // counter for gps frames
    okvis::StateId id;
    okvis::kinematics::Transformation T_WS_est;
    okvis::SpeedAndBias speedAndBias_est;
    okvis::kinematics::Transformation T_GW_est;
    // DO THE PROPAGATION AND INTEGRATION

    std::vector<okvis::kinematics::Transformation> gtPoses;
    gtPoses.clear();
    std::vector<okvis::SpeedAndBias> gtSbs;
    gtSbs.clear();

    for(size_t i=0; i<size_t(DURATION*IMU_RATE); ++i){

        time = double(i)/IMU_RATE; // current time in interval

        // --- IMU Signal creation ---

        // Compute velocity vector
        v_w = omegaVec.cross(r-p_c); // v = omega x r
        // Compute acceleration vector
        Eigen::Vector3d a_W = omegaVec.cross(v_w);

        // Propagate position
        r += v_w*DT;

        // Propagate Orientation
        Eigen::Quaterniond dq;
        const double alpha_half = omegaVec.norm()*DT*0.5;
        const double sinc_alpha_half = okvis::kinematics::sinc(alpha_half);
        const double cos_alpha_half = cos(alpha_half);
        dq.vec() = sinc_alpha_half * 0.5 * omegaVec * DT;
        dq.w() = cos_alpha_half;
        q = q * dq;

        // update GT T_WS
        T_WS = okvis::kinematics::Transformation(r,q);

        // speedAndBias - v only, obviously, since this is the Ground Truth
        speedAndBias.head<3>() = v_w;// velocity in world frame

        // generate imu measurements
        Eigen::Vector3d gyr = omegaVec + imuParameters.sigma_g_c/sqrt(DT)*Eigen::Vector3d::Random();
        Eigen::Vector3d acc = T_WS.inverse().C()*(a_W+Eigen::Vector3d(0,0,imuParameters.g)) + imuParameters.sigma_a_c/sqrt(DT)*Eigen::Vector3d::Random();
        imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(time),okvis::ImuSensorReadings(gyr,acc)));

        // camera frame creation
        if(time >= k / FRAME_RATE/*time >= static_cast<double>(k) / static_cast<double>(K) * DURATION*/){

            // assemble a multi-frame
            std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
            mf->setTimestamp(okvis::Time(time));

            // add frames
            mf->resetCameraSystemAndFrames(*cameraSystem);

            // add it in the window to create a new time instance
            gtPoses.push_back(T_WS);
            gtSbs.push_back(speedAndBias);

            if(k==0) {
              id = viGraph2.addStatesInitialise(mf->timestamp(), imuMeasurements, *cameraSystem);
            } else {
              const bool isKeyframe = (k%3 == 0); // every third as keyframe
              id = viGraph2.addStatesPropagate(mf->timestamp(), imuMeasurements, isKeyframe);
            }
            //std::cout << "Frame " << k << " successfully added. ID="<< id.value() << std::endl;
            mf->setId(id.value());

            T_WS_est = viGraph2.pose(okvis::StateId(mf->id()));

            // now let's add also landmark observations
            std::vector<cv::KeyPoint> keypoints;
            for (size_t jj = 0; jj < homogeneousPoints.size(); ++jj) {
              for (size_t ii = 0; ii < mf->numFrames(); ++ii) {
                Eigen::Vector2d projection;
                Eigen::Vector4d point_C = mf->T_SC(ii)->inverse() * T_WS.inverse() * homogeneousPoints[jj];
                okvis::cameras::ProjectionStatus status =
                        mf->geometryAs<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>>(ii)->projectHomogeneous(point_C, &projection);
                if (status == okvis::cameras::ProjectionStatus::Successful) {
                  Eigen::Vector2d measurement(projection + Eigen::Vector2d::Random());
                  keypoints.push_back(cv::KeyPoint(static_cast<float>(measurement[0]),
                                      static_cast<float>(measurement[1]), 8.0));
                  mf->resetKeypoints(ii,keypoints);
                  okvis::KeypointIdentifier kid(mf->id(), ii, mf->numKeypoints(ii)-1);
                  viGraph2.addObservation<
                      okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>>(
                        *mf, lmIds[jj], kid);
                }
              }
            }
            ++k;

            // Optimize Graph
            viGraph2.addGpsMeasurements(gpsMeasurements, imuMeasurements,nullptr);
            gpsMeasurements.clear();
            viGraph2.optimise(10, 1, false);
            T_GW_est = viGraph2.T_GW();
        }

        if(time >= g / GPS_RATE/*time >= static_cast<double>(g) / static_cast<double>(G) * DURATION*/){
            double errGps = (double) rand() / RAND_MAX * GPS_STD;
            Eigen::Matrix3d covGps;
            covGps.setIdentity();
            covGps *= std::pow(GPS_STD,2);
            okvis::GpsMeasurement gpsMeasurement;
            gpsMeasurement.timeStamp = okvis::Time(time);
            gpsMeasurement.measurement.position = T_GW.C() * (T_WS.r() + T_WS.C()*r_SA) + T_GW.r() + errGps * Eigen::Vector3d::Random();
            gpsMeasurement.measurement.covariances = covGps;
            gpsMeasurements.push_back(gpsMeasurement);
            ++g;
        }
    }

    // Try to add GPS measurements to the graphs
    viGraph2.addGpsMeasurements(gpsMeasurements, imuMeasurements, nullptr);

    // Cleaning landmarks that have < 2 observations
    int nClean = viGraph2.cleanUnobservedLandmarks();
    std::cout << id.value() << " frames created." << std::endl;
    std::cout << "Removed " << nClean << " landmarks with no or only one observation." << std::endl;

    // run the optimization
    viGraph2.optimise(10, 1, false);

    std::cout << "Verifyint estimation results..." << std::endl;

    for(size_t kk = 0; kk < gtPoses.size(); ++kk){

        // Obtain estimated pose and speed and biases
        T_WS_est = viGraph2.pose(okvis::StateId(kk+1));
        speedAndBias_est = viGraph2.speedAndBias(okvis::StateId(kk+1));
        T_GW_est = viGraph2.T_GW();
        //std::cout << "Estimated T_GW \n" << T_GW_est.T() << std::endl;

        EXPECT_TRUE((gtPoses.at(kk).r() - T_WS_est.r()).norm()<1e-1) << "Pose trans error: " << (gtPoses.at(kk).r() - T_WS_est.r()).norm() << std::endl;
        EXPECT_TRUE(2*(gtPoses.at(kk).q() * T_WS_est.q().inverse()).vec().norm()<2e-2);
        EXPECT_TRUE((speedAndBias_est - gtSbs.at(kk)).norm() < 0.04);
        EXPECT_TRUE((T_GW.r() - T_GW_est.r()).norm()<1e-1);
        EXPECT_TRUE(2*(T_GW.q() * T_GW_est.q().inverse()).vec().norm()<2e-2);
    }

}
