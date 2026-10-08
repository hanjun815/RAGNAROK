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
 * @file GpsErrorAsynchronous.cpp
 * @brief Source File for the GpsErrorAsynchronous class
 * @author Simon Boche
 */

#include <eigen3/Eigen/Core>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/GpsErrorAsynchronous.hpp>
#include <okvis/ceres/ImuError.hpp> // required for propagation
#include <okvis/ceres/ode/ode.hpp>
//#include <gtest/gtest.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

std::atomic_bool GpsErrorAsynchronous::redoPropagationAlways(false);
std::atomic_bool GpsErrorAsynchronous::useImuCovariance(true);

// Constructor with provided information matrix
GpsErrorAsynchronous::GpsErrorAsynchronous(const measurement_t & measurement, const covariance_t & information,
                                           const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                                           const okvis::Time& tk, const okvis::Time& tg,
                                           const okvis::GpsParameters & gpsParameters){
    setMeasurement(measurement);
    setInformation(information);
    setImuMeasurements(imuMeasurements);
    setImuParameters(imuParameters);
    setTk(tk);
    setTg(tg);
    setGpsParameters(gpsParameters);

    // derivatives of covariance matrix w.r.t. sigmas
    dPdsigma_.resize(4);
    for(size_t j=0; j<4; ++j) {
      dPdsigma_.at(j).setZero();
    }
}

// Constructor using per-axis standard deviations
GpsErrorAsynchronous::GpsErrorAsynchronous(const measurement_t & measurement, const double sigma_x, const double sigma_y, const double sigma_z,
                                           const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                                           const okvis::Time& tk, const okvis::Time& tg,
                                           const GpsParameters& gpsParameters){

    // construct diagonal covariance matrix from inputs
    Eigen::Matrix3d informationMat = Eigen::Matrix3d::Identity();
    informationMat(0,0) = 1/(sigma_x*sigma_x);
    informationMat(1,1) = 1/(sigma_y*sigma_y);
    informationMat(2,2) = 1/(sigma_z*sigma_z);

    setMeasurement(measurement);
    setInformation(informationMat);
    setImuMeasurements(imuMeasurements);
    setImuParameters(imuParameters);
    setTk(tk);
    setTg(tg);
    setGpsParameters(gpsParameters);

    // derivatives of covariance matrix w.r.t. sigmas
    dPdsigma_.resize(4);
    for(size_t j=0; j<4; ++j) {
      dPdsigma_.at(j).setZero();
    }

}

// Set the information.
void GpsErrorAsynchronous::setInformation(const covariance_t& information) {
    gpsInformation_ = information;
    gpsCovariance_ = information.inverse();
}

// This evaluates the error term and additionally computes the Jacobians.
bool GpsErrorAsynchronous::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {

    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool GpsErrorAsynchronous::EvaluateWithMinimalJacobians(double const* const * parameters,
                                                        double* residuals, double** jacobians,
                                                        double** jacobiansMinimal) const {

    // Obtain Trafos from parameters

    // robot pose at t=k
    Eigen::Map<const Eigen::Vector3d> r_WS(&parameters[0][0]);
    const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3],parameters[0][4], parameters[0][5]);
    okvis::kinematics::Transformation T_WS_tk(r_WS, q_WS);

    // speed and biases at t=k
    okvis::SpeedAndBias speedAndBiases;
    for (size_t i = 0; i < 9; ++i) {
      speedAndBiases[i] = parameters[1][i];
    }

    // GPS <-> world trafo
    Eigen::Map<const Eigen::Vector3d> r_GW(&parameters[2][0]);
    const Eigen::Quaterniond q_GW(parameters[2][6], parameters[2][3],parameters[2][4], parameters[2][5]);
    Eigen::Matrix3d C_GW = q_GW.toRotationMatrix();

    // Obtain translation antenna to sensor frame
    Eigen::Vector3d r_SA = gpsParameters_.r_SA;

    // ----- PRE-INTEGRATION PROPAGATION - BEGIN -----

    // this will NOT be changed:
    const Eigen::Matrix3d C_WS_tk = T_WS_tk.C();

    // call the propagation
    const double Delta_t = (tg_ - tk_).toSec();
    Eigen::Matrix<double, 6, 1> Delta_b;
    // ensure unique access
    {
      std::lock_guard<std::mutex> lock(preintegrationMutex_);
      Delta_b = speedAndBiases.tail<6>()
            - speedAndBiases_ref_.tail<6>();
      redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
      if (redoPropagationAlways || (redo_ && ((imuMeasurements_.size() < 50) )) || redoCounter_==0) {
        redoPreintegration(T_WS_tk, speedAndBiases);
        redoCounter_++;
        Delta_b.setZero();
        redo_ = false;
      }
    }

    // actual propagation output:
    std::lock_guard<std::mutex> lock(preintegrationMutex_); // this is a bit stupid, but shared read-locks only come in C++14
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    // estimate robot pose at t=g from preintegration results
    okvis::kinematics::Transformation T_WS_tg;
    T_WS_tg.set(r_WS + speedAndBiases.head<3>()*Delta_t - 0.5*g_W*Delta_t*Delta_t
                + C_WS_tk * (acc_doubleintegral_ + dp_db_g_ * Delta_b.head<3>() - C_doubleintegral_ * Delta_b.tail<3>()),
                T_WS_tk.q()*Delta_q_*okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>()));

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double,15,15> Jprop =
      Eigen::Matrix<double,15,15>::Identity(); // holds for d/db_g, d/db_a

    Jprop.block<3,3>(0,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_doubleintegral_);
    Jprop.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*Delta_t;
    Jprop.block<3,3>(0,9) = C_WS_tk*dp_db_g_;
    Jprop.block<3,3>(0,12) = -C_WS_tk*C_doubleintegral_;
    Jprop.block<3,3>(3,9) = -C_WS_tk*dalpha_db_g_;
    Jprop.block<3,3>(6,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_integral_);
    Jprop.block<3,3>(6,9) = C_WS_tk*dv_db_g_;
    Jprop.block<3,3>(6,12) = -C_WS_tk*C_integral_;

    // ----- PRE-INTEGRATION PROPAGATION - END -----

    if(useImuCovariance){

        // We need to first transform the Covariance into world frame!
        Eigen::Matrix<double,15,15> T = Eigen::Matrix<double,15,15>::Identity();
        T.topLeftCorner<3,3>() = T_WS_tk.C();
        T.block<3,3>(3,3) = T_WS_tk.C();
        T.block<3,3>(6,6) = T_WS_tk.C();
        Eigen::Matrix<double,15,15> P;
        P = T * P_delta_ * T.transpose();

        // compute square root information matrix
        Eigen::Matrix<double,3,6> Jws;
        Jws.block<3,3>(0,0) = C_GW;
        Jws.block<3,3>(0,3) = -C_GW * okvis::kinematics::crossMx(T_WS_tk.C()*r_SA);

        Eigen::Matrix<double,3,3> covOverall = gpsCovariance_ + Jws * P.block<6,6>(0,0) * Jws.transpose(); // consider both sources of covariances
        Eigen::LLT<information_t> lltOfInformation(covOverall.inverse());
        squareRootInformation_ = lltOfInformation.matrixL().transpose();

    }

    else{
        Eigen::LLT<information_t> lltOfInformation(gpsCovariance_.inverse());
        squareRootInformation_ = lltOfInformation.matrixL().transpose();
    }




    // calculate the GPS error
    measurement_t error = measurement_ - (C_GW * (T_WS_tg.r() + T_WS_tg.C()*r_SA) + r_GW);
    error_ = error;
    // weight error by square root of information matrix:
    measurement_t weighted_error = squareRootInformation_ * error;

    // assign residuals
    residuals[0] = weighted_error[0];
    residuals[1] = weighted_error[1];
    residuals[2] = weighted_error[2];

    // Compute jacobians if required
    if(jacobians!=NULL){

        if(jacobians[0]!=NULL){ // w.r.t. robot pose at tk

            Eigen::Matrix<double,3,6> J0_minimal_tg;
            J0_minimal_tg.setZero();
            J0_minimal_tg.topLeftCorner<3,3>() = -C_GW;
            J0_minimal_tg.topRightCorner<3,3>() = C_GW * okvis::kinematics::crossMx(T_WS_tg.C()*r_SA);

            // Propagation to xk in minimal representation
            Eigen::Matrix<double,3,6> J0_minimal_tk;
            Eigen::Matrix<double,6,6> J_dxg_dxt;
            J_dxg_dxt = Jprop.topLeftCorner<6,6>();
            J0_minimal_tk = J0_minimal_tg * J_dxg_dxt;

            // Lift to non-minimal representation
            Eigen::Matrix<double,6,7, Eigen::RowMajor> J0_tk_lift;
            PoseManifold::minusJacobian(parameters[0], J0_tk_lift.data());

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J0(jacobians[0]);
            J0 = squareRootInformation_ * J0_minimal_tk * J0_tk_lift;

            // if requested, provide minimal Jacobians
            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
                J0_minimal_mapped = squareRootInformation_ * J0_minimal_tk;
              }
            }

        }

        if(jacobians[1]!=NULL){

            // de / d_x_tg
            Eigen::Matrix<double,3,6> J1_minimal_tg;
            J1_minimal_tg.setZero();
            J1_minimal_tg.topLeftCorner<3,3>() = -C_GW;
            J1_minimal_tg.topRightCorner<3,3>() = C_GW * okvis::kinematics::crossMx(T_WS_tg.C()*r_SA);

            // Propagation to sb_k in minimal representation
            Eigen::Matrix<double,3,9> J1_tk;
            Eigen::Matrix<double,6,9> J_dxg_dsbk;
            J_dxg_dsbk = Jprop.block<6,9>(0,6);
            J1_tk = J1_minimal_tg * J_dxg_dsbk;

            // No Lift needed in that case
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1(jacobians[1]);
            J1 = squareRootInformation_ * J1_tk;

            // if requested, provide minimal Jacobians
            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
                J1_minimal_mapped = J1;
              }
            }

        }

        if(jacobians[2]!=NULL){ // w.r.t. GPS/world trafo

            Eigen::Matrix<double,3,6> J2_minimal;
            J2_minimal.setZero();
            J2_minimal.topLeftCorner<3,3>() = -Eigen::Matrix3d::Identity();
            J2_minimal.topRightCorner<3,3>() = okvis::kinematics::crossMx(C_GW*(T_WS_tg.r()+T_WS_tg.C()*r_SA));

            // pseudo inverse of the local parametrization Jacobian:
            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J2_lift;
            PoseManifold::minusJacobian(parameters[2], J2_lift.data());

            // hallucinate Jacobian w.r.t. state
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J2(jacobians[2]);
            J2 = squareRootInformation_ * J2_minimal * J2_lift;

            // if requested, provide minimal Jacobians
            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J2_minimal_mapped(jacobiansMinimal[2]);
                J2_minimal_mapped = squareRootInformation_ * J2_minimal;
              }
            }

        }

    }

    return true;

}

// Propagates pose, speeds and biases with given IMU measurements.
int GpsErrorAsynchronous::redoPreintegration(const okvis::kinematics::Transformation& /*T_WS*/,
                                 const okvis::SpeedAndBias & speedAndBiases) const {

  // now the propagation
  okvis::Time time = tk_;
  okvis::Time end = tg_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp<=time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  // derivatives of covariance matrix w.r.t. sigmas
  dPdsigma_.resize(4);
  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j).setZero();
  }

  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = tg_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = tg_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double gyr_sat_mult = 1.0;
    double acc_sat_mult = 1.0;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      gyr_sat_mult *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      acc_sat_mult *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment < 3 > (3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment < 3 > (6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    if(useImuCovariance){

        // covariance propagation
        Eigen::Matrix<double, 15, 15> F_delta =
            Eigen::Matrix<double, 15, 15>::Identity();
        // transform
        F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
            acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
        F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
        F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
            + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
            + 0.25 * (C + C_1) * dt * dt;
        F_delta.block<3, 3>(3, 9) = -dt * C_1;
        F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
            0.5 * (C + C_1) * acc_S_true * dt);
        F_delta.block<3, 3>(6, 9) = 0.5 * dt
            * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;

        // Q = K * sigma_sq
        Eigen::Matrix<double,15,15> K0 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K1 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K2 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K3 = Eigen::Matrix<double,15,15>::Zero();
        K0.block<3,3>(3,3) = gyr_sat_mult*dt * Eigen::Matrix3d::Identity();
        K1.block<3,3>(0,0) = 0.5 * dt*dt*dt * acc_sat_mult*acc_sat_mult*acc_sat_mult * Eigen::Matrix3d::Identity();
        K1.block<3,3>(6,6) = acc_sat_mult*dt* Eigen::Matrix3d::Identity();
        K2.block<3,3>(9,9) = dt * Eigen::Matrix3d::Identity();
        K3.block<3,3>(12,12) = dt * Eigen::Matrix3d::Identity();
        dPdsigma_.at(0) = F_delta*dPdsigma_.at(0)*F_delta.transpose() + K0;
        dPdsigma_.at(1) = F_delta*dPdsigma_.at(1)*F_delta.transpose() + K1;
        dPdsigma_.at(2) = F_delta*dPdsigma_.at(2)*F_delta.transpose() + K2;
        dPdsigma_.at(3) = F_delta*dPdsigma_.at(3)*F_delta.transpose() + K3;

    }


    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == tg_)
      break;

  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  if(useImuCovariance){

      // get the weighting:
      // enforce symmetric
      for(int j=0; j<4; ++j) {
        dPdsigma_.at(j) = 0.5 * dPdsigma_.at(j) + 0.5 * dPdsigma_.at(j).transpose().eval();
      }
      P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
      P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
      P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
      P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

  }



  return i;
}
} /* namespace ceres */
} /* namespace okvis */
