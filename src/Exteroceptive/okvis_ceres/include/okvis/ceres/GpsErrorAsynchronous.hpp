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
 * @file ceres/GpsErrorAsynchronous.hpp
 * @brief Header file for the asynchronous GpsError class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CERES_GPSERRORASYNCHRONOUS_HPP_
#define INCLUDE_OKVIS_CERES_GPSERRORASYNCHRONOUS_HPP_

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <ceres/sized_cost_function.h>
#include <ceres/covariance.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
namespace ceres {


/// \brief Implements a nonlinear IMU factor.
class GpsErrorAsynchronous :
    public ::ceres::SizedCostFunction<3 /* number of residuals */,
        7 /* size of first parameter (RobotPoseParameterBlock T_WS at t=k ) */,
        9 /* size of second parameter (SpeedAndBiasParameterBlock at t=k)*/,
        7 /* size of third parameter (PoseParameterBlock T_GW) */>,
    public ErrorInterface {

 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::atomic_bool redoPropagationAlways;
  static std::atomic_bool useImuCovariance;

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<3, 7, 9, 7> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 3;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 3, 3> covariance_t;

  /// \brief The measurement type.
  typedef Eigen::Vector3d measurement_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of Jacobian w.r.t. robot pose in world frame
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 3, 7> jacobian0_t;

  /// \brief The type of the Jacobian w.r.t. speed and biases
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 3, 9> jacobian1_t;

  /// \brief The type of the Jacobian w.r.t. poses GPS <-> world--
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 3, 7> jacobian2_t;

  /// \brief Default constructor.
  GpsErrorAsynchronous();

  /// \brief Construct with measurement and information matrix
  /// @param[in] measurement The measurement.
  /// @param[in] information The information matrix (INVERSE covariance matrix).
  /// @param[in] imuMeasurements Queue containing IMU measurements
  /// @param[in] imuParameters IMU sensor parameters
  /// @param[in] tk Timestamp of previous state / camera frame
  /// @param[in] tg Timestamp of the GPS measurement
  /// @param[in] gpsParameters GPS sensor parameters
  GpsErrorAsynchronous(const measurement_t & measurement, const covariance_t & information,
                       const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                       const okvis::Time& tk, const okvis::Time& tg,
                       const GpsParameters& gpsParameters);

  /// \brief Construct with measurement and standard deviations
  /// @param[in] measurement The measurement.
  /// @param[in] sigma_x / sigma_y / sigma_z Standard deviation / error of measurement per axis
  /// @param[in] imuMeasurements Queue containing IMU measurements
  /// @param[in] imuParameters IMU sensor parameters
  /// @param[in] tk Timestamp of previous state / camera frame
  /// @param[in] tg Timestamp of the GPS measurement
  /// @param[in] gpsParameters GPS sensor parameters
  GpsErrorAsynchronous(const measurement_t & measurement, const double sigma_x, const double sigma_y, const double sigma_z,
                       const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                       const okvis::Time& tk, const okvis::Time& tg,
                       const GpsParameters& gpsParameters);



  /// \brief Trivial destructor.
  virtual ~GpsErrorAsynchronous()
  {
  }

  // setters
  /// \brief Set gps parameters
  /// @param[in] gpsParameters the parameters of GPS sensor
  virtual void setGpsParameters(const GpsParameters & gpsParameters){
      gpsParameters_ = gpsParameters;
  }

  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  virtual void setInformation(const covariance_t& information);

  /// \brief Set the time.
  /// @param[in] tg The timestamp of the gps measurement.
  void setTg(const okvis::Time& tg)
  {
    tg_ = tg;
  }

  /// \brief Set the time.
  /// @param[in] tk The timestamp of the latest camera frame.
  void setTk(const okvis::Time& tk)
  {
    tk_ = tk;
  }

  /// \brief (Re)set the parameters.
  /// \@param[in] imuParameters The parameters to be used.
  void setImuParameters(const okvis::ImuParameters& imuParameters){
      imuParameters_ = imuParameters;
  }

  /// \brief (Re)set the measurements
  /// \@param[in] imuMeasurements All the IMU measurements.
  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  // getters

  /// \brief Get the measurement.
  /// \return The measurement vector.
  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  virtual const covariance_t& information() const
  {
    return gpsInformation_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  virtual const covariance_t& covariance() const
  {
    return gpsCovariance_;
  }

  /// \brief Get the time.
  /// \return The timestamp of the GPS measurement
  okvis::Time tg() const
  {
    return tg_;
  }

  /// \brief Get the time.
  /// \return The timestamp of the latest corresponding camera frame
  okvis::Time tk() const
  {
    return tk_;
  }

  /// \brief Get the IMU Parameters.
  /// \return the IMU parameters.
  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  /// \brief Get the IMU measurements.
  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  /// \brief Get the GPS parameters.
  const okvis::GpsParameters& gpsParameters() const {
    return gpsParameters_;
  }

  /// \brief Get unweighted error.
  const Eigen::Vector3d error() const {
      return error_;
  }

  /// \brief Get propagated pose using preintegration
  /// \@param[in] T_WS_in pose of state
  /// \@param[in] sb_in speed and bias of state
  /// \@param[out] T_WS_prop propagated pose
  bool applyPreInt(const okvis::kinematics::Transformation T_WS_in, const SpeedAndBias sb_in, okvis::kinematics::Transformation& T_WS_prop){

    // this will NOT be changed:
    const Eigen::Matrix3d C_WS_tk = T_WS_in.C();
    const Eigen::Vector3d r_WS_tk = T_WS_in.r();

    // call the propagation
    const double Delta_t = (tg_ - tk_).toSec();
    Eigen::Matrix<double, 6, 1> Delta_b;
    // ensure unique access
    {
      Delta_b = sb_in.tail<6>()
                - speedAndBiases_ref_.tail<6>();
      redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
      //std::cout << imuMeasurements_.size() << std::endl;
    }

    // actual propagation output:
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    // estimate robot pose at t=g from preintegration results
    T_WS_prop.set(r_WS_tk + sb_in.head<3>()*Delta_t - 0.5*g_W*Delta_t*Delta_t
                + C_WS_tk * (acc_doubleintegral_ + dp_db_g_ * Delta_b.head<3>() - C_doubleintegral_ * Delta_b.tail<3>()),
                T_WS_in.q()*Delta_q_*okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>()));

    return true;

  }
  ///
  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of the evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements using pre-integration scheme.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  // added convenient check
//  virtual bool VerifyJacobianNumDiff(double const* const * parameters, double** jacobian) const;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  int parameterBlockDim(int parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "GpsErrorAsynchronous";
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (3D) measurement.

  // the error
  mutable Eigen::Vector3d error_; /// < The unweighted error

  // GPS parameters
  GpsParameters gpsParameters_; ///< The gps parameters

  // GPS information / covariance
  covariance_t gpsInformation_; ///< The 3x3 information matrix.
  //covariance_t squareRootInformation_; ///< The 3x3 square root information matrix.
  covariance_t gpsCovariance_; ///< The 3x3 covariance matrix.

  // IMU parameters
  okvis::ImuParameters imuParameters_; ///< The IMU parameters.
  // IMU measurements
  okvis::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements used. Must be spanning tk_ - tg_.

  // times
  okvis::Time tk_; ///< The start time (i.e. time of the first set of states).
  okvis::Time tg_; ///< The end time (i.e. time of the GPS Signal).

  // ----- preintegration stuff -----
  // the mutable is a TERRIBLE HACK, but what can I do.

  mutable std::mutex preintegrationMutex_; //< Protect access of intermediate results.
  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

  // for gradient/hessian w.r.t. the sigmas:
  mutable AlignedVector<Eigen::Matrix<double,15,15>> dPdsigma_;

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; ///< Keeps track of whether or not this redoPreintegration() needs to be called.
  mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  //mutable Eigen::Matrix<double,15,15> imuInformation_; ///< The information matrix for integrated IMU measurements
  // ----- preintegration stuff end -----

  mutable information_t squareRootInformation_; ///< The overall square root information matrix for this error term (GPS + IMU covariances).

};

}  // namespace ceres
}  // namespace okvis

#endif // INCLUDE_OKVIS_CERES_GPSERRORASYNCHRONOUS_HPP_
