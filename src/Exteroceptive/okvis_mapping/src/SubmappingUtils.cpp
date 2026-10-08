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

#include <random>
#include <okvis/SubmappingUtils.hpp>

namespace okvis{

void save_bounding_box_vtk(const se::Submap<okvis::SupereightMapType>& submap, const std::string& filename){
  // Get Maximum and Minimum of active submap
  Eigen::Vector3f map_min = submap.map->aabb().min();
  Eigen::Vector3f map_max = submap.map->aabb().max(); 

  // Convert map aabb into world frame

  Eigen::Vector3f corner000(map_min.x(), map_min.y(), map_min.z());
  Eigen::Vector3f corner100(map_max.x(), map_min.y(), map_min.z());
  Eigen::Vector3f corner010(map_min.x(), map_max.y(), map_min.z());
  Eigen::Vector3f corner001(map_min.x(), map_min.y(), map_max.z());
  Eigen::Vector3f corner110(map_max.x(), map_max.y(), map_min.z());
  Eigen::Vector3f corner101(map_max.x(), map_min.y(), map_max.z());
  Eigen::Vector3f corner011(map_min.x(), map_max.y(), map_max.z());
  Eigen::Vector3f corner111(map_max.x(), map_max.y(), map_max.z());

  // Transform to Submap Frame of active ID
  corner000 = submap.T_WK * corner000.homogeneous();
  corner100 = submap.T_WK * corner100.homogeneous();
  corner010 = submap.T_WK * corner010.homogeneous();
  corner001 = submap.T_WK * corner001.homogeneous();
  corner110 = submap.T_WK * corner110.homogeneous();
  corner101 = submap.T_WK * corner101.homogeneous();
  corner011 = submap.T_WK * corner011.homogeneous();
  corner111 = submap.T_WK * corner111.homogeneous();

  // Compute new AABB
  float x_min = std::min({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});
  float x_max = std::max({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});
  float y_min = std::min({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});
  float y_max = std::max({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});
  float z_min = std::min({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});
  float z_max = std::max({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});

  Eigen::Vector3f min_world;
  Eigen::Vector3f max_world;
  min_world.x() = x_min;
  min_world.y() = y_min;
  min_world.z() = z_min;
  max_world.x() = x_max;
  max_world.y() = y_max;
  max_world.z() = z_max;

  // Open the file for writing.
  std::ofstream file(filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return;
  }

  // Write the header.
  file << "# vtk DataFile Version 1.0\n";
  file << "Bounding Box of Submap\n";
  file << "ASCII\n";
  file << "DATASET POLYDATA\n";

  file << "POINTS 8 float\n";
  file << min_world.x() << " " << min_world.y() << " " << min_world.z() << "\n"; // 0
  file << max_world.x() << " " << min_world.y() << " " << min_world.z() << "\n"; // 1
  file << max_world.x() << " " << max_world.y() << " " << min_world.z() << "\n"; // 2
  file << min_world.x() << " " << max_world.y() << " " << min_world.z() << "\n"; // 3
  file << min_world.x() << " " << min_world.y() << " " << max_world.z() << "\n"; // 4
  file << max_world.x() << " " << min_world.y() << " " << max_world.z() << "\n"; // 5
  file << max_world.x() << " " << max_world.y() << " " << max_world.z() << "\n"; // 6
  file << min_world.x() << " " << max_world.y() << " " << max_world.z() << "\n"; // 7

  file << "LINES 12 36\n";
  file << "2 0 1\n2 1 2\n2 2 3\n2 3 0\n2 4 5\n2 5 6\n2 6 7\n2 7 4\n2 0 4\n2 1 5\n2 2 6\n2 3 7\n";

}

    // ToDo: template!
void downsamplePointCloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& originalPointCloud,
                          std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downsampledPointCloud,
                          size_t num_of_points)
{
  // If not enough points provided, return all
  if(num_of_points > originalPointCloud.size()){
    downsampledPointCloud.insert(downsampledPointCloud.end(), originalPointCloud.begin(), originalPointCloud.end());
  }
  else{
    std::sample(originalPointCloud.begin(), originalPointCloud.end(), std::back_inserter(downsampledPointCloud), num_of_points,
                std::mt19937{std::random_device{}()});
  }

}

void downsamplePointsUncertainty(
  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points, const std::vector<float>& sigmas,
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downPoints, std::vector<float>& downSigma, size_t maxNum) {

  downPoints.clear();
  downSigma.clear();

  std::vector<size_t> idx(sigmas.size());
  std::iota(idx.begin(), idx.end(), 0);

  // Sorting indices in ascending order
  std::stable_sort(idx.begin(), idx.end(),
    [&sigmas](size_t i1, size_t i2) {return sigmas[i1] < sigmas[i2];});

  size_t numPoint = points.size();
  size_t numFactor = numPoint < maxNum ? numPoint : maxNum;
  for (size_t i = 0; i < numFactor; i ++) {
    downPoints.push_back(points[idx[i]]);
    downSigma.push_back(sigmas[idx[i]]);            
  }

}


bool LidarMotionCompensation::appendTo(const ImuMeasurementDeque & imuMeasurements,
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

  // t_end_ = t_end; // remember ToDo: do not remember this but latest IMU integrated
  sb_ref_ = speedAndBiases; // remember linearisation point

  bool hasStarted = false;
  for (/*imu_iterator_*/; imu_iterator_ != imuMeasurements.end(); ++imu_iterator_) {
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

bool LidarMotionCompensation::getState(const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
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

}