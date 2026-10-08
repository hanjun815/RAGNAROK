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
 * @file ViInterface.cpp
 * @brief Source file for the ViInterface class.
 * @author Paul Furgale
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <fstream>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ViInterface.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

void expImageTransform(cv::Mat& frame) {
  cv::exp(-frame / 10.f * std::log(2), frame);  // equivalent to 2^(-frame/10)
}

void logImageTransform(cv::Mat& frame) {
  cv::log(frame, frame);
  frame = (-10.f / std::log(2)) * frame;
}

void Trajectory::update(const TrackingState & trackingState,
                        std::shared_ptr<const AlignedMap<StateId,State> > updatedStates,
                        std::set<StateId>& affectedStateIds) {
  if(updatedStates->empty()) {
    return; // nothing to do.
  }
  //std::lock_guard<std::mutex> lock(stateMutex_);
  allStateIds_.insert(trackingState.id);
  const State & currentState = updatedStates->at(trackingState.id);
  // important: need to update keyframes in the past *first* so relative poses will be correct.
  for(const auto & state : *updatedStates) {
    affectedStateIds.insert(state.second.id);
    if(keyframeStates_.count(state.second.id)) {
      keyframeStates_.at(state.second.id) = state.second; // update
      if(nonKeyframeStatesByKeyframeId_.count(state.second.id)) {
        for(const auto & nonKeyframeState : nonKeyframeStatesByKeyframeId_.at(state.second.id)) {
          affectedStateIds.insert(nonKeyframeState.first);
        }
      }
    }
  }
  // now current state

  if(trackingState.isKeyframe) {
    keyframeStates_[trackingState.id] = currentState; // insert
  } else {
    if(keyframeStates_.count(trackingState.currentKeyframeId)) {
      const okvis::kinematics::Transformation T_Sk_S = 
          keyframeStates_.at(trackingState.currentKeyframeId).T_WS.inverse() * currentState.T_WS;
      NonKeyframeState nonKeyframeState;
      nonKeyframeState.T_Sk_S = T_Sk_S;
      nonKeyframeState.v_Sk = T_Sk_S*currentState.T_WS.inverse()*currentState.v_W;
      nonKeyframeState.omega_S_raw = currentState.omega_S - currentState.b_g;
      nonKeyframeState.timestamp = currentState.timestamp;
      nonKeyframeState.keyframeId = trackingState.currentKeyframeId;
      nonKeyframeState.id = currentState.id;
      nonKeyframeState.previousImuMeasurements = currentState.previousImuMeasurements;
      nonKeyframeStatesByKeyframeId_[trackingState.currentKeyframeId][currentState.id] =
          nonKeyframeState;
      keyframeIdByStateId_[currentState.id] = trackingState.currentKeyframeId;
    }
  }
  statesByTimestampNs_[currentState.timestamp.toNSec()] = currentState.id;

  // prepare new propagator
  livePropagator_.reset(new Propagator(currentState.timestamp));
  while (!imuMeasurements_.empty()) {
    if (currentState.timestamp - imuMeasurements_.front().timeStamp < Duration(0.1)) {
      break;
    }
    imuMeasurements_.pop_front();
  }
  if (imuMeasurements_.empty()) {
    return; // happens, if addImuMeasurement is not called on this Trajectory object
  }
  SpeedAndBias sb;
  sb.head<3>() = currentState.v_W;
  sb.segment<3>(3) = currentState.b_g;
  sb.tail<3>() = currentState.b_a;
  livePropagator_->appendTo(imuMeasurements_,
                            currentState.T_WS,
                            sb,
                            imuMeasurements_.back().timeStamp);
}

bool Trajectory::addImuMeasurement(const Time &stamp,
                                   const Eigen::Vector3d &alpha,
                                   const Eigen::Vector3d &omega,
                                   State &state)
{
  // store IMU measurement
  ImuMeasurement imuMeasurement(stamp, ImuSensorReadings(omega, alpha));
  imuMeasurements_.push_back(imuMeasurement);
  if (allStateIds_.empty() || !livePropagator_) {
    return false;
  }

  // get state to propagate from
  State unpropagatedState;
  if (!getState(*allStateIds_.rbegin(), unpropagatedState)) {
    return false;
  }

  // somehow the IMU measurements were buffered and have not arrived here yet. Weird and bad!
  if(imuMeasurement.timeStamp < unpropagatedState.timestamp) {
    return false;
  }

  // propagate
  kinematics::Transformation T_WS_1;
  SpeedAndBias sb_0, sb_1;
  sb_0.head<3>() = unpropagatedState.v_W;
  sb_0.segment<3>(3) = unpropagatedState.b_g;
  sb_0.tail<3>() = unpropagatedState.b_a;
  livePropagator_->appendTo(imuMeasurements_,
                            unpropagatedState.T_WS,
                            sb_0,
                            imuMeasurement.timeStamp);
  if (!livePropagator_->getState(unpropagatedState.T_WS, sb_0, state.T_WS, sb_1, state.omega_S)) {
    return false;
  }
  state.v_W = sb_1.head<3>();
  state.b_g = sb_1.segment<3>(3);
  state.b_a = sb_1.tail<3>();
  state.isKeyframe = false;
  state.timestamp = imuMeasurement.timeStamp;
  return true;
}

bool Trajectory::getState(const Time & timestamp, State& state) {
  if(statesByTimestampNs_.count(timestamp.toNSec())) {
    const StateId stateId = statesByTimestampNs_.at(timestamp.toNSec());
    return getState(stateId, state);
  } else {
    // find closest timestamp
    auto iter = statesByTimestampNs_.lower_bound(timestamp.toNSec());
    if(iter == statesByTimestampNs_.end()) {
      return false;
    }
    if(iter == statesByTimestampNs_.begin()) {
      return false;
    }
    State state0, state1;
    if(!getState(iter->second, state1)) {
      return false;
    }
    iter--; // now pointing to the timestamp before...
    if(!getState(iter->second, state0)) {
      return false;
    }

    // for propagation:
    kinematics::Transformation T_WS_0 = state0.T_WS;
    SpeedAndBias speedAndBias_0;
    speedAndBias_0.head<3>() = state0.v_W;
    speedAndBias_0.segment<3>(3) = state0.b_g;
    speedAndBias_0.tail<3>() = state0.b_a;
    kinematics::Transformation T_WS_1;
    SpeedAndBias speedAndBias_1;

    // check if cached:
    bool found = false;
    const auto propagators_iter = propagatorsByKeyframeIdAndEndTimeNs_.find(iter->second);
    if(propagators_iter != propagatorsByKeyframeIdAndEndTimeNs_.end()) {
      const auto propagator_iter = propagators_iter->second.find(timestamp.toNSec());
      if(propagator_iter != propagators_iter->second.end()) {
        found = propagator_iter->second.getState(T_WS_0,
                                                 speedAndBias_0,
                                                 T_WS_1,
                                                 speedAndBias_1,
                                                 state.omega_S);
      }
    }
    if(!found) {
      // do the propagation...
      // (smartly keep propagating doesn't seem to help with speed)
      if(state1.previousImuMeasurements.empty()) { //No IMU data, constant velocity model
        ConstantVelocityPropagator propagator(state0, state1);
        propagator.getState(timestamp,
                            T_WS_1, speedAndBias_1, state.omega_S);


      } else {
        Propagator propagator(state0.timestamp);
        propagator.appendTo(state1.previousImuMeasurements,
                            T_WS_0, speedAndBias_0, timestamp);
        propagator.getState(T_WS_0, speedAndBias_0,
                            T_WS_1, speedAndBias_1, state.omega_S);

        // cache...!
        if(propagators_iter == propagatorsByKeyframeIdAndEndTimeNs_.end()) {
          propagatorsByKeyframeIdAndEndTimeNs_[iter->second]
              .emplace(timestamp.toNSec(), propagator);
        } else {
          propagatorsByKeyframeIdAndEndTimeNs_.at(iter->second)
              .emplace(timestamp.toNSec(), propagator);
        }
      }
    }

    state.T_WS = T_WS_1; // Transformation between World W and Sensor S.
    state.v_W = speedAndBias_1.head<3>(); // Velocity in frame W [m/s].
    state.b_g = speedAndBias_1.segment<3>(3); // Gyro bias [rad/s].
    state.b_a = speedAndBias_1.tail<3>(); // Accelerometer bias [m/s^2].
    state.timestamp = timestamp; // Timestamp corresponding to this state.
    state.id = StateId(); // set invalid.
    state.previousImuMeasurements = state0.previousImuMeasurements;
    state.isKeyframe = false;
  }
  return true;
}

bool Trajectory::getStates(const std::vector<Time> & timestamps, std::vector<State>& states) {
  // Use first timestamp to obtain closest timestamp
  if(timestamps.size() == 0){
    return false;
  }
  // find closest timestamp
  auto iter = statesByTimestampNs_.lower_bound(timestamps[0].toNSec());

  if(iter == statesByTimestampNs_.end()) {
    return false;
  }
  if(iter == statesByTimestampNs_.begin()) {
    return false;
  }
  State state0, state1;
  if(!getState(iter->second, state1)) {
    return false;
  }
  iter--; // now pointing to the timestamp before...
  if(!getState(iter->second, state0)) {
    return false;
  }

  if(timestamps.front().toNSec() < state0.timestamp.toNSec())
  {
    if(!getState(iter->second, state1)) {
      return false;
    }
    iter--;
    if(!getState(iter->second, state0)) {
      return false;
    }
  }

  if(timestamps.back().toNSec() > state1.timestamp.toNSec())
  {
    state0 = state1;
    iter++; iter++;
    if(!getState(iter->second, state1)) {
      return false;
    }
  }


  // do the propagation...
  kinematics::Transformation T_WS_0 = state0.T_WS;
  SpeedAndBias speedAndBias_0;
  speedAndBias_0.head<3>() = state0.v_W;
  speedAndBias_0.segment<3>(3) = state0.b_g;
  speedAndBias_0.tail<3>() = state0.b_a;
  kinematics::Transformation T_WS_1;
  SpeedAndBias speedAndBias_1;

  auto nextiter = iter;
  nextiter ++;

  State state;
  if(!state1.previousImuMeasurements.empty()) {
    BatchedLidarPropagator propagator(state0.timestamp, state1.previousImuMeasurements);
    // Iterate measurements
    //std::cout << "### Getting states is called for id = " << iter->first << std::endl;
    for(size_t i = 0; i < timestamps.size(); i++){
      if(statesByTimestampNs_.count(timestamps[i].toNSec())) {
        const StateId stateId = statesByTimestampNs_.at(timestamps[i].toNSec());
        State state;
        getState(stateId, state);
        states.emplace_back(state);
        continue;
      }
      propagator.appendTo(state1.previousImuMeasurements,
                          T_WS_0, speedAndBias_0, timestamps.at(i));
      propagator.getState(T_WS_0, speedAndBias_0,
                          T_WS_1, speedAndBias_1, state.omega_S);

      state.T_WS = T_WS_1; // Transformation between World W and Sensor S.
      state.v_W = speedAndBias_1.head<3>(); // Velocity in frame W [m/s].
      state.b_g = speedAndBias_1.segment<3>(3); // Gyro bias [rad/s].
      state.b_a = speedAndBias_1.tail<3>(); // Accelerometer bias [m/s^2].
      state.timestamp = timestamps[i]; // Timestamp corresponding to this state.
      state.id = StateId(); // set invalid.
      state.previousImuMeasurements = state0.previousImuMeasurements;
      state.isKeyframe = false;
      // Write into return vector
      states.emplace_back(state);
    }
  } else {
    ConstantVelocityPropagator propagator(state0, state1);
    //Assume that we are in the no IMU case and then we perform a constant velocity model
    for(size_t i = 0; i < timestamps.size(); i++){
      if(statesByTimestampNs_.count(timestamps[i].toNSec())) {
        const StateId stateId = statesByTimestampNs_.at(timestamps[i].toNSec());
        State state;
        getState(stateId, state);
        states.emplace_back(state);
        continue;
      }
      //We add here the constant linear velocity model
      propagator.getState(timestamps[i],
                          T_WS_1, speedAndBias_1, state.omega_S);

      state.T_WS = T_WS_1; // Transformation between World W and Sensor S.
      state.v_W = speedAndBias_1.head<3>(); // Velocity in frame W [m/s].
      state.b_g = speedAndBias_1.segment<3>(3); // Gyro bias [rad/s].
      state.b_a = speedAndBias_1.tail<3>(); // Accelerometer bias [m/s^2].
      state.timestamp = timestamps[i]; // Timestamp corresponding to this state.
      state.id = StateId(); // set invalid.
      state.previousImuMeasurements = state0.previousImuMeasurements;
      state.isKeyframe = false;
      // Write into return vector
      states.emplace_back(state);
    }

  }

  return true;
}


bool Trajectory::getState(const StateId & stateId, State& state) const {
  if((keyframeStates_.count(stateId))) {
    state = keyframeStates_.at(stateId);
    return true;
  } else if (keyframeIdByStateId_.count(stateId)){
    const StateId keyframeId = keyframeIdByStateId_.at(stateId);
    const NonKeyframeState & nonKeyframeState =
        nonKeyframeStatesByKeyframeId_.at(keyframeId).at(stateId);
    const State & keyframeState = keyframeStates_.at(keyframeId);
    state.T_WS = keyframeState.T_WS*nonKeyframeState.T_Sk_S;
    state.v_W = keyframeState.T_WS*nonKeyframeState.v_Sk;
    state.b_g = keyframeState.b_g;
    state.b_a = keyframeState.b_a;
    state.omega_S = nonKeyframeState.omega_S_raw - keyframeState.b_g;
    state.timestamp = nonKeyframeState.timestamp;
    state.previousImuMeasurements = nonKeyframeState.previousImuMeasurements;
    state.isKeyframe = false;
    return true;
  }
  return false;
}

const AlignedMap<StateId, State>& Trajectory::keyframeStates() const {
  return keyframeStates_;
}

bool Trajectory::clearCache(const StateId& keyframeStateId)
{
  auto iter = propagatorsByKeyframeIdAndEndTimeNs_.find(keyframeStateId);
  if(iter == propagatorsByKeyframeIdAndEndTimeNs_.end()) {
    return false; // not present, nothing to do
  } else {
    iter->second.clear(); // delete everything.
  }
  return true;
}

ViInterface::ViInterface() /*: realtimePropagation_(false)*/ {
}

ViInterface::~ViInterface() {
  if (csvImuFile_)
    csvImuFile_->close();
  // also close all registered tracks files
  for (FilePtrMap::iterator it = csvTracksFiles_.begin();
      it != csvTracksFiles_.end(); ++it) {
    if (it->second)
      it->second->close();
  }
}

// Set the callback to be called every time states change.
void ViInterface::setOptimisedGraphCallback(const OptimisedGraphCallback & optimisedGraphCallback) {
  optimisedGraphCallback_ = optimisedGraphCallback;
}

// Write first line of IMU CSV file to describe columns.
bool ViInterface::writeImuCsvDescription() {
  if (!csvImuFile_)
    return false;
  if (!csvImuFile_->good())
    return false;
  *csvImuFile_ << "timestamp" << ", " << "omega_tilde_WS_S_x" << ", "
      << "omega_tilde_WS_S_y" << ", " << "omega_tilde_WS_S_z" << ", "
      << "a_tilde_WS_S_x" << ", " << "a_tilde_WS_S_y" << ", "
      << "a_tilde_WS_S_z" << std::endl;
  return true;
}

// Write first line of tracks (data associations) CSV file to describe columns.
bool ViInterface::writeTracksCsvDescription(size_t cameraId) {
  if (!csvTracksFiles_[cameraId])
    return false;
  if (!csvTracksFiles_[cameraId]->good())
    return false;
  *csvTracksFiles_[cameraId] << "timestamp" << ", " << "landmark_id" << ", "
      << "z_tilde_x" << ", " << "z_tilde_y" << ", " << "z_tilde_stdev" << ", "
      << "descriptor" << std::endl;
  return false;
}

// Set a CVS file where the IMU data will be saved to.
bool ViInterface::setImuCsvFile(std::fstream& csvFile) {
  if (csvImuFile_) {
    csvImuFile_->close();
  }
  csvImuFile_.reset(&csvFile);
  writeImuCsvDescription();
  return csvImuFile_->good();
}

// Set a CVS file where the IMU data will be saved to.
bool ViInterface::setImuCsvFile(const std::string& csvFileName) {
  csvImuFile_.reset(
      new std::fstream(csvFileName.c_str(), std::ios_base::out));
  writeImuCsvDescription();
  return csvImuFile_->good();
}

// Set a CVS file where the tracks (data associations) will be saved to.
bool ViInterface::setTracksCsvFile(size_t cameraId, std::fstream& csvFile) {
  if (csvTracksFiles_[cameraId]) {
    csvTracksFiles_[cameraId]->close();
  }
  csvTracksFiles_[cameraId].reset(&csvFile);
  writeTracksCsvDescription(cameraId);
  return csvTracksFiles_[cameraId]->good();
}

// Set a CVS file where the tracks (data associations) will be saved to.
bool ViInterface::setTracksCsvFile(size_t cameraId,
                                    const std::string& csvFileName) {
  csvTracksFiles_[cameraId].reset(
      new std::fstream(csvFileName.c_str(), std::ios_base::out));
  writeTracksCsvDescription(cameraId);
  return csvTracksFiles_[cameraId]->good();
}

Propagator::Propagator(const Time & t_start) : t_start_(t_start), t_end_(t_start) {}

bool Propagator::appendTo(const ImuMeasurementDeque & imuMeasurements,
                          const kinematics::Transformation & ,
                          const SpeedAndBias & speedAndBiases,
                          const Time & t_end) {

  // now the propagation
  okvis::Time time = t_end_; // re-start from previous end; in the beginning: t_start_.
  okvis::Time end = t_end;

  // sanity check:
  OKVIS_ASSERT_TRUE_DBG(std::runtime_error,
                        imuMeasurements.front().timeStamp <= time,
                        "inconsistent timestamps" << imuMeasurements.front().timeStamp
                                                  << " !<= " << time)

  if (!(imuMeasurements.back().timeStamp >= end))
    return false;  // nothing to do...

  t_end_ = t_end; // remember
  sb_ref_ = speedAndBiases; // remember linearisation point

  bool hasStarted = false;
  for (ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else {
      nexttime = (it + 1)->timeStamp;
    }

    double dt = (nexttime - time).toSec();
    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
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

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;

    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5*(acc_S_0+acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5*(C + C_1)*dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_ + 0.5*(C + C_1)*acc_S_true*dt;

    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral_ += acc_integral_*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    // Jacobian parts
    dalpha_db_g_ += dt*C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross_ +
                                        kinematics::rightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_ + 0.5*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
    dp_db_g_ += dt*dv_db_g_ + 0.25*dt*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
    omega_S_ = omega_S_true;

    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    if (nexttime == t_end) {
      break;
    }
  }

  Delta_t_ = (t_end_-t_start_).toSec();
  return true;
}

bool Propagator::getState(
    const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
    kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
    Eigen::Vector3d & omega_S) {

  // actual propagation output: TODO should get the 9.81 from parameters..
  const Eigen::Vector3d g_W = 9.81 * Eigen::Vector3d(0, 0, 6371009).normalized();

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();

  // call the propagation
  const Eigen::Matrix<double, 6, 1> Delta_b
      = sb_0.tail<6>() - sb_ref_.tail<6>();

  // intermediate stuff
  const Eigen::Vector3d r_est_W =
      T_WS_0.r() + sb_0.head<3>()*Delta_t_ - 0.5*g_W*Delta_t_*Delta_t_;

  const Eigen::Vector3d v_est_W = sb_0.head<3>() - g_W*Delta_t_;
  const Eigen::Quaterniond Dq =
      okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>())*Delta_q_;

  // the overall error vector
  Eigen::Matrix<double, 15, 1> error;
  const Eigen::Vector3d r_W_1 = r_est_W
                                + C_WS_0
                                    * (acc_doubleintegral_ + dp_db_g_ * Delta_b.head<3>()
                                       - C_doubleintegral_ * Delta_b.tail<3>());
  const Eigen::Quaterniond q_WS_1 = T_WS_0.q()*Dq;
  const Eigen::Vector3d v_W_1 = v_est_W
                                + C_WS_0
                                    * (acc_integral_ + dv_db_g_ * Delta_b.head<3>()
                                       - C_integral_ * Delta_b.tail<3>());

  // assign output
  T_WS_1.set(r_W_1, q_WS_1);
  sb_1 = sb_0;
  sb_1.head<3>() = v_W_1;
  omega_S = omega_S_;

  return true;
}

double Propagator::sinc(double x) {
  if (fabs(x) > 1e-6) {
    return sin(x) / x;
  } else {
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}

BatchedLidarPropagator::BatchedLidarPropagator(const okvis::Time &t_start,
                                               const okvis::ImuMeasurementDeque &imuMeasurements) :
                                               Propagator(t_start), imu_iterator_(imuMeasurements.begin()) {}


bool BatchedLidarPropagator::appendTo(const ImuMeasurementDeque & imuMeasurements,
                          const kinematics::Transformation & ,
                          const SpeedAndBias & speedAndBiases,
                          const Time & t_end) {

  // now the propagation
  okvis::Time time = t_end_; // re-start from previous end; in the beginning: t_start_.
  okvis::Time end = t_end;

  // sanity check:
  OKVIS_ASSERT_TRUE_DBG(std::runtime_error, imuMeasurements.front().timeStamp<=time,
                        imuMeasurements.front().timeStamp << " !<= " << time);

  if (!(imuMeasurements.back().timeStamp >= end))
    return false;  // nothing to do...

  sb_ref_ = speedAndBiases; // remember linearisation point

  bool hasStarted = false;
  for (; imu_iterator_ != imuMeasurements.end(); ++imu_iterator_) {
    Eigen::Vector3d omega_S_0 = imu_iterator_->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = imu_iterator_->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (imu_iterator_ + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (imu_iterator_ + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((imu_iterator_ + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else {
      nexttime = (imu_iterator_ + 1)->timeStamp;
    }

    double dt = (nexttime - time).toSec(); /// checking delta between next imu measurement and current propagator end timestamp
    if (end < nexttime) { /// if (query time < next IMU measurement)
      double interval = (nexttime - imu_iterator_->timeStamp).toSec(); /// following from here we do an interpolation between IMU measurements
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) { /// if next IMU measurement is older than the current propagator end time skip IMU measurement and continue integration
      continue;
    }
    if (!hasStarted) { /// first iteration
      hasStarted = true;
      const double r = dt / (nexttime - imu_iterator_->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    /// now do the actual propagation for new IMU measurements that have not been propagated yet
    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;

    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5*(acc_S_0+acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5*(C + C_1)*dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_ + 0.5*(C + C_1)*acc_S_true*dt;

    if(nexttime == t_end){
      // Stop IMU integration and interpolate
      C_doubleintegral_interp_ = C_doubleintegral_ + C_integral_*dt + 0.25*(C + C_1)*dt*dt;
      acc_doubleintegral_interp_ = acc_doubleintegral_ + acc_integral_*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

      dalpha_db_g_interp_ = dalpha_db_g_ + dt*C_1;
      const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross_ +
                                      kinematics::rightJacobian(omega_S_true*dt)*dt;
      const Eigen::Matrix3d acc_S_x = kinematics::crossMx(acc_S_true);
      Eigen::Matrix3d dv_db_g_1 = dv_db_g_ + 0.5*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
      dp_db_g_interp_ = dp_db_g_ + dt*dv_db_g_ + 0.25*dt*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
      omega_S_interp_ = omega_S_true;

      // memory shift
      Delta_q_interp_ = Delta_q_1;
      C_integral_interp_ = C_integral_1;
      acc_integral_interp_ = acc_integral_1;
      cross_interp_ = cross_1;
      dv_db_g_interp_ = dv_db_g_1;
      break;
    }

    /// Overwrite preintegration results using interpolated values
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral_ += acc_integral_*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    // Jacobian parts
    dalpha_db_g_ += dt*C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross_ +
                                    kinematics::rightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_ + 0.5*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
    dp_db_g_ += dt*dv_db_g_ + 0.25*dt*dt*(C*acc_S_x*cross_ + C_1*acc_S_x*cross_1);
    omega_S_ = omega_S_true;

    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;
    t_end_ = nexttime;
  }

  Delta_t_ = (t_end-t_start_).toSec();
  return true;
}

bool BatchedLidarPropagator::getState(
        const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
        kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
        Eigen::Vector3d & omega_S) {

  // actual propagation output: TODO should get the 9.81 from parameters..
  const Eigen::Vector3d g_W = 9.81 * Eigen::Vector3d(0, 0, 6371009).normalized();

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();

  // call the propagation
  const Eigen::Matrix<double, 6, 1> Delta_b
          = sb_0.tail<6>() - sb_ref_.tail<6>();

  // intermediate stuff
  const Eigen::Vector3d r_est_W =
          T_WS_0.r() + sb_0.head<3>()*Delta_t_ - 0.5*g_W*Delta_t_*Delta_t_;

  const Eigen::Vector3d v_est_W = sb_0.head<3>() - g_W*Delta_t_;
  const Eigen::Quaterniond Dq =
          okvis::kinematics::deltaQ(-dalpha_db_g_interp_*Delta_b.head<3>())*Delta_q_interp_;

  // the overall error vector
  Eigen::Matrix<double, 15, 1> error;
  const Eigen::Vector3d r_W_1 =  r_est_W + C_WS_0*(acc_doubleintegral_interp_ + dp_db_g_interp_*Delta_b.head<3>() - C_doubleintegral_interp_*Delta_b.tail<3>());
  const Eigen::Quaterniond q_WS_1 = T_WS_0.q()*Dq;
  const Eigen::Vector3d v_W_1 = v_est_W + C_WS_0*(acc_integral_interp_ + dv_db_g_interp_*Delta_b.head<3>() - C_integral_interp_*Delta_b.tail<3>());

  // assign output
  T_WS_1.set(r_W_1, q_WS_1);
  sb_1 = sb_0;
  sb_1.head<3>() = v_W_1;
  omega_S = omega_S_interp_;

  return true;
}

ConstantVelocityPropagator::ConstantVelocityPropagator(const State & stateStart,
                                                     const State & stateEnd) : 
                                                     startState_(stateStart), endState_(stateEnd),
                                                     timeDelta_(endState_.timestamp - startState_.timestamp),
                                                     r_s0_s1_((startState_.T_WS.inverse().T() * endState_.T_WS.T()).topRightCorner(3, 1)) {}


Eigen::Quaterniond ConstantVelocityPropagator::extrapolateAngle(const Time& timestamp) {
  Eigen::AngleAxisd axis_s0s1(startState_.T_WS.C().transpose() * endState_.T_WS.C());
  double angle_rot = axis_s0s1.angle();
  std::cout << "Doing angle extrapolations in the constant velocity model, this is not how this should be used" << std::endl;
  if(timestamp > endState_.timestamp) {
    //We extrapolate from endState
    return Eigen::Quaterniond(endState_.T_WS.C() * Eigen::AngleAxisd(angle_rot * ((timestamp - startState_.timestamp).toSec() / timeDelta_.toSec()), 
                              axis_s0s1.axis()).toRotationMatrix());
  } else {
    //We extrapoalte from startState
    return Eigen::Quaterniond(startState_.T_WS.C() * Eigen::AngleAxisd(angle_rot * ((timestamp - startState_.timestamp).toSec() / timeDelta_.toSec()), 
                              axis_s0s1.axis()).toRotationMatrix());
  }
}


bool ConstantVelocityPropagator::getState(const Time& timestamp,
          kinematics::Transformation & T_WS_e, SpeedAndBias & sb_e,
          Eigen::Vector3d & omega_S) 
  {
    if(timestamp >= startState_.timestamp && timestamp <= endState_.timestamp) {
      Eigen::Matrix3d C_S0Se = startState_.T_WS.C().transpose() *  startState_.T_WS.q().slerp(((timestamp - startState_.timestamp).toSec() / timeDelta_.toSec()),
                                                                                                endState_.T_WS.q()).toRotationMatrix();      
      kinematics::Transformation T_S0Se(r_s0_s1_ * ((timestamp - startState_.timestamp).toSec() / timeDelta_.toSec()), 
                                        Eigen::Quaterniond(C_S0Se));
      T_WS_e = startState_.T_WS * T_S0Se;
      sb_e.head<3>() = startState_.v_W;
      sb_e.segment<3>(3) = startState_.b_g;
      sb_e.tail<3>() = startState_.b_a;
      omega_S = startState_.omega_S;
    } else if(timestamp > endState_.timestamp) {
      Eigen::Matrix3d C_S1Se = endState_.T_WS.C().transpose() * extrapolateAngle(timestamp).toRotationMatrix();
      kinematics::Transformation T_S1Se(r_s0_s1_ * (timestamp - endState_.timestamp).toSec() / timeDelta_.toSec(), 
                                        Eigen::Quaterniond(C_S1Se));
      T_WS_e = endState_.T_WS * T_S1Se;
      sb_e.head<3>() = endState_.v_W;
      sb_e.segment<3>(3) = endState_.b_g;
      sb_e.tail<3>() = endState_.b_a;
      omega_S = endState_.omega_S;
    } else {
      Eigen::Matrix3d C_S0Se = startState_.T_WS.C().transpose() * extrapolateAngle(timestamp).toRotationMatrix();
      kinematics::Transformation T_S0Se(r_s0_s1_ * (timestamp - startState_.timestamp).toSec() / timeDelta_.toSec(), 
                                        Eigen::Quaterniond(C_S0Se));
      T_WS_e = endState_.T_WS * T_S0Se;
      sb_e.head<3>() = startState_.v_W;
      sb_e.segment<3>(3) = startState_.b_g;
      sb_e.tail<3>() = startState_.b_a;
      omega_S = startState_.omega_S;
    }
    return true;
  }

void ConstantVelocityPropagator::newState(const State & newState) {
    if(newState.timestamp > endState_.timestamp) {
      startState_ = endState_;
      endState_ = newState;
    } else if(newState.timestamp > startState_.timestamp) {
      endState_ = newState;
    } else if(newState.timestamp < startState_.timestamp) {
      endState_ = startState_;
      startState_ = newState;
    }
    timeDelta_ = endState_.timestamp - startState_.timestamp;
    r_s0_s1_ = (startState_.T_WS.inverse().T() * endState_.T_WS.T()).topRightCorner(3, 1);
  }


}  // namespace okvis
