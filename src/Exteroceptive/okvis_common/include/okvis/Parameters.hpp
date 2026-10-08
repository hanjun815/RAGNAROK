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
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_PARAMETERS_HPP_
#define INCLUDE_OKVIS_PARAMETERS_HPP_

#include <set>
#include <string>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <okvis/Time.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <optional>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief Struct that contains all the camera calibration information.
struct CameraCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
  Eigen::Vector2i imageDimension;           ///< Image dimension. [pixels]
  Eigen::VectorXd distortionCoefficients;   ///< Distortion Coefficients.
  Eigen::Vector2d focalLength;              ///< Focal length.
  Eigen::Vector2d principalPoint;           ///< Principal point.
  std::string cameraModel;                  ///< camera model. ('pinhole' 'eucm')
  Eigen::Vector2d eucmParameters;           ///< alpha, beta eucm parameters

  /// \brief Distortion type. ('radialtangential' 'radialtangential8' 'equdistant')
  std::string distortionType;

  cameras::NCameraSystem::CameraType cameraType; ///< Some additional info about the camera.
};

/*!
 * \brief Camera parameters.
 *
 * A simple struct to specify properties of a Camera.
 *
 */
struct CameraParameters{
  double timestamp_tolerance; ///< Stereo frame out-of-sync tolerance. [s]
  std::set<size_t> sync_cameras; ///< The cameras that will be synchronised.
  std::vector<size_t> stereo_indices; ///< camera indices for the left and right camera for the stereo network

  /// \brief Image timestamp error. [s] timestamp_camera_correct = timestamp_camera - image_delay.
  double image_delay;

  /**
   * @brief Some parameters to set the online calibrator.
   */
  struct OnlineCalibrationParameters {
    bool do_extrinsics; ///< Do we online-calibrate extrinsics?
    bool do_extrinsics_final_ba; ///< Do we calibrate extrinsics in final BA?
    double sigma_r; ///< T_SCi position prior stdev [m]
    double sigma_alpha; ///< T_SCi orientation prior stdev [rad]
    double sigma_r_final_ba; ///< T_SCi position prior stdev in final BA [m]
    double sigma_alpha_final_ba; ///< T_SCi orientation prior stdev in final BA [rad]
  };

  OnlineCalibrationParameters online_calibration; ///< Online calibration parameters.

  /// If true, apply \c frontend_parameters image enhancement (vignette + legacy AGCWD/CLAHE or paper mode) to stereo DNN input before rectification.
  bool stereo_dnn_apply_image_enhancement = false;
};

/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool use; ///< Use the IMU at all?
  bool use_preintegration_residual = true; ///< Add nonlinear IMU preintegration residuals (ImuError) to the graph?
  /// \brief Estimate the gyroscope bias as part of the state?
  /// If false, b_g is pinned to g0 (see ViParametersReader), so the raw gyro readings are used.
  /// This matches GaRLILEO, which integrates the gyro with a constant (zero) bias.
  bool estimate_gyro_bias = false;
  /// \brief Estimate the accelerometer bias as part of the state?
  /// If false, b_a is pinned to a0 (see ViParametersReader), so the raw accelerometer readings
  /// are used. This matches GaRLILEO, whose b_a is never an active parameter in the
  /// leg-enabled configuration. Defaults to true: unlike b_g, b_a is genuinely observable in a
  /// visual-inertial graph through the gravity direction, so disabling it is opt-in.
  bool estimate_accel_bias = true;
  okvis::kinematics::Transformation T_BS; ///< Transform from Body frame to IMU (sensor frame S).
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_ba;  ///< Initial accelerometer bias
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  Eigen::Vector3d g0;  ///< Mean of the prior gyro bias.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  double g;  ///< Earth acceleration.
  Eigen::Vector3d s_a; ///< Scale factor for accelerometer measurements
};

/**
 * @brief Parameters for detection etc.
 */
struct FrontendParameters {
  double detection_threshold; ///< Detection threshold. By default the uniformity radius in pixels.
  double absolute_threshold; ///< Absolute Harris corner threshold (noise floor).
  double matching_threshold; ///< BRISK descriptor matching threshold.
  int octaves; ///< Number of octaves for detection. 0 means single-scale at highest resolution.
  int max_num_keypoints; ///< Restrict to a maximum of this many keypoints per img (strongest ones).
  double keyframe_overlap; ///< Minimum field-of-view overlap.
  bool use_cnn; ///< Use the CNN (if available) to filter out dynamic content / sky.
  bool parallelise_detection; ///< Run parallel detect & describe.
  int num_matching_threads; ///< Parallelise matching with this number of threads.

  // Optional: radial vignette correction (applied before enhancement).
  bool vignette_correction_enable = false;   ///< If true: apply radial gain correction to brighten dark corners.
  double vignette_correction_k1 = 0.0;       ///< Polynomial coefficient for r^2.
  double vignette_correction_k2 = 0.0;       ///< Polynomial coefficient for r^4.
  double vignette_correction_k3 = 0.0;       ///< Polynomial coefficient for r^6.

  // Optional: trigger-based Bayesian optimization for AGCWD + CLAHE parameters.
  bool agcwd_clahe_bo_enable = false;             ///< If true: dynamically optimize agcwd_weighting_param & clahe_clip_limit.
  /// Re-run BO when the raw-image relative utility discrepancy
  /// |U_real,t - U_real,t-1| / |U_real,t-1| exceeds this threshold.
  double agcwd_clahe_bo_utility_discrepancy_threshold = 0.2;
  int agcwd_clahe_bo_initial_samples = 4;         ///< Number of random warmup samples.
  int agcwd_clahe_bo_iterations = 6;              ///< Number of BO iterations after warmup.
  int agcwd_clahe_bo_candidate_grid = 9;          ///< Candidate grid size per dimension for EI search.
  double agcwd_clahe_bo_agcwd_min = 0.2;          ///< Lower bound for agcwd_weighting_param.
  double agcwd_clahe_bo_agcwd_max = 1.2;          ///< Upper bound for agcwd_weighting_param.
  double agcwd_clahe_bo_clahe_min = 1.0;          ///< Lower bound for clahe_clip_limit.
  double agcwd_clahe_bo_clahe_max = 8.0;          ///< Upper bound for clahe_clip_limit.
  /// Scalar multiplier on mean(SFA-EWG) over available downsampled enhanced views (sole BO objective term).
  double agcwd_clahe_bo_image_utility_weight = 1.0;

  // Optional: image preprocessing for robustness in low-light.
  // AGCWD = Adaptive Gamma Correction with Weighting Distribution (contrast enhancement).
  bool agcwd_enable = false;            ///< If true: apply AGCWD preprocessing before feature extraction.
  double agcwd_weighting_param = 0.5;   ///< Weighting parameter (typically in (0,1]). Use 0.5 as a reasonable default.

  // CLAHE (Contrast Limited Adaptive Histogram Equalization).
  bool clahe_enable = false;            ///< If true: apply CLAHE preprocessing before feature extraction.
  double clahe_clip_limit = 2.0;        ///< CLAHE clip limit (higher => stronger local contrast).
  int clahe_tile_grid_size = 8;         ///< CLAHE tile grid size (NxN).

  // ----- EWG (Entropy-Weighted Gradient) image utility -----------------------
  // Reference:
  //   J. Kim, Y. Cho, A. Kim, "Proactive Camera Attribute Control Using
  //   Bayesian Optimization for Illumination-Resilient Visual Navigation,"
  //   IEEE T-RO, 36(4), 2020.
  //
  // Per-camera SFA-EWG utility is computed in `Frontend::detectAndDescribe` and
  // cached for `Frontend::getImageUtility(...)`.
  // The fallback uses the utility of the preprocessed image, or of the raw image when
  // `image_utility_fallback_use_preprocessed` is false.
  // Used by low-feature fallback gating (GaRLILEOParameters::image_utility_threshold).
  // EWG hyperparameters (paper Eq. 1-5).
  double image_utility_ewg_Hthres = 0.125;        ///< Saturation threshold (paper default 0.05).
  double image_utility_ewg_alpha = 32.0;          ///< Activation function steepness.
  double image_utility_ewg_tau = 4.0;             ///< Activation function bias.
  int image_utility_ewg_local_patch = 5;          ///< Local entropy patch (paper default 5x5).
  int image_utility_ewg_num_bins = 8;             ///< Histogram quantization bins (8 = 256/32).
  double image_utility_ewg_lambda_f = 1.0;        ///< SFA-EWG: weight on Q_f * Q_s bonus term.
  double image_utility_ewg_N0 = 500.0;           ///< SFA-EWG: BRISK keypoint count scale N0 for Q_f.
  int image_utility_ewg_grid_rows = 3;           ///< SFA-EWG: spatial uniformity grid rows.
  int image_utility_ewg_grid_cols = 4;          ///< SFA-EWG: spatial uniformity grid cols.
  /// If true, resize the raw image (longest side) before per-frame SFA-EWG (faster; retune thresholds).
  bool image_utility_ewg_resize_enable = false;
  /// Max longer side in pixels when `image_utility_ewg_resize_enable` is true (e.g. 640).
  int image_utility_ewg_resize_max_side = 640;
  /// If true, cache fallback utility from preprocessed image; if false, from raw image.
  bool image_utility_fallback_use_preprocessed = true;
};

/**
 * @brief Parameters regarding the estimator.
 */
struct EstimatorParameters {
  int num_keyframes; ///< Number of keyframes in optimisation window.
  int num_loop_closure_frames; ///< Number of loop closure frames in optimisation window.
  int num_imu_frames; ///< Number of frames linked by most recent nonlinear IMU error terms.
  bool do_loop_closures; ///< Whether to do VI-SLAM or VIO.
  bool do_final_ba; ///< Whether to run a final full BA.
  int final_ba_max_iterations = 100; ///< Iteration cap for each final BA pass.
  double final_ba_function_tolerance = 1.0e-6; ///< Relative cost stopping criterion in final BA.
  bool enforce_realtime; ///< Whether to limit the time budget for optimisation.
  int realtime_min_iterations; ///< Minimum number of iterations always performed.
  int realtime_max_iterations; ///< Never do more than these, even if not converged.
  double realtime_time_limit; ///< Time budget for realtime optimisation. [s]
  int realtime_num_threads; ///< Number of threads for the realtime optimisation.
  int full_graph_iterations; ///< Don't do more than these for the full (background) optimisation.
  int full_graph_num_threads; ///< Number of threads for the full (background) optimisation.
  double p_dbow; ///< Match threshold for dBoW -- unfortunately this varies with setups.
  double drift_percentage_heuristic; ///< % allowed drift in loop closures rel. to dist. travelled.
};

/**
 * @brief Some options for how and what to output.
 */
struct OutputParameters {
    bool display_topview; ///< Displays top view (Non-causal part might be slow).
    bool display_matches; ///< Displays debug video and matches. May be slow.
    bool display_overhead; ///< Debug overhead image. Is slow.
    bool enable_submapping; //< Whether or not is submapping enabled
};

/**
 * @brief Parameters for GaRLILEO spline coupling (between-frame relative pose constraint).
 *
 * GaRLILEO publishes `/garlileo/spline_state` as `nav_msgs/msg/Odometry` where:
 * - pose.pose.position: world-frame position (p_WS) from dead-reckoning
 * - pose.pose.orientation: world-from-body rotation (R_WS)
 * - twist.twist.linear: body-frame linear velocity (v_body)
 *
 * OKVIS2-X uses this (when messages are available) to create a between-frame relative pose
 * constraint by directly using GaRLILEO's dead-reckoned poses at the two frame timestamps.
 *
 * Note: There is no explicit enable flag; if `/garlileo/spline_state` is not available, the constraint
 * will simply not be added.
 */
struct GaRLILEOParameters {
  bool use_rotation = true;          ///< If true: use GaRLILEO rotation in between factor (k-1,k).
  bool use_translation = true;       ///< Use translation part of between constraint.
  bool use_velocity = true;          ///< Use velocity constraint.
  double rotation_roll_pitch_std_deg = 5.0; ///< Roll/pitch 1-sigma rotation std for between-frame constraint [deg].
  double rotation_yaw_std_deg = 5.0; ///< Yaw 1-sigma rotation std for between-frame constraint [deg].
  double translation_std = 0.01;      ///< 1-sigma translation std for between-frame constraint [m].
  double velocity_std = 0.2;         ///< 1-sigma velocity std for ego velocity constraint [m/s].
  /// If true, multiply nominal stds by sqrt(max(trackingQuality, std_scale_min_tracking_quality)).
  /// If false, keep nominal stds unscaled (factor 1.0).
  bool std_scale_use_tracking_quality = true;
  /// Quality floor when scaling the nominal stds above (and velocity_std).
  /// The information gain is 1/max(trackingQuality, this), capped at 1/this for a positive floor.
  double std_scale_min_tracking_quality = 0.5; ///< [0..1] Typical default 0.5 for minimum quality.
  double max_time_diff = 0.05;       ///< Maximum allowed |dt| for endpoint association/interpolation [s].
  double history_sec = 2.0;          ///< How much GaRLILEO history to keep for association [s].
  bool apply_imu_extrinsic = true;   ///< If true: map GaRLILEO body frame to OKVIS IMU frame using imu.T_BS.

  // Frontend pose prediction (T_WS -> T_WC initial guess passed to Frontend::detectAndDescribe).
  // If enabled and GaRLILEO measurements are available, OKVIS will apply GaRLILEO relative
  // pose delta on top of the previous OKVIS pose guess for frontend initialisation.
  // This only affects the *frontend initial guess* and does not replace the estimator IMU model.
  bool frontend_use_garlileo = true;       ///< If true: use GaRLILEO to improve frontend pose prediction.
  bool frontend_use_rotation = true;        ///< If true: use GaRLILEO rotation for frontend prediction.
  bool frontend_use_translation = true;     ///< If true: use GaRLILEO translation for frontend prediction.
  double frontend_max_age_sec = 0.2;        ///< Max |dt| allowed when using last GaRLILEO meas for frontend [s].

  // Frontend BRISK extraction direction (gravity projection).
  bool frontend_use_garlileo_gravity = true; ///< If true: use GaRLILEO gravity (from /garlileo/gravity) for BRISK extraction direction.

  // Keyframe gating using GaRLILEO ego speed (performance optimisation).
  // If enabled (>0), the frontend will skip the overlap-based keyframe decision and force
  // asKeyframe=false when the GaRLILEO ego speed is below this threshold.
  double keyframe_stationary_speed_threshold = 0.0; ///< [m/s] 0 disables.

  // SFA-EWG image utility based fallback trigger (entropy-weighted gradient + AGAST modulation).
  // When >0, `low_features` uses the minimum per-camera utility across cameras
  // (worst camera) compared to this threshold.
  double image_utility_threshold = 0.0;
  /// Upper edge of the semi-band for utility-driven fallback blending [same units as the
  /// per-frame minimum camera utility]. When that value lies in [image_utility_threshold,
  /// image_utility_semi_threshold), fallback blend transitions toward full fallback as it decreases.
  double image_utility_semi_threshold = 80.0;
  /// Sigmoid steepness for utility fallback blending inside [threshold, semi_threshold].
  /// Larger => sharper transition, smaller => gentler transition.
  double image_utility_blend_sigmoid_steepness = 6.0;
  /// Hysteresis deadband width on utility value used for fallback blending.
  /// If |u_t - u_{t-1,hyst}| < this, keep previous utility for smoother switching.
  double image_utility_blend_hysteresis = 0.5;

  // Tracking lost threshold used by Frontend/ThreadedSlam. If trackingQuality < this value, we
  // treat visual tracking as lost (and may fall back to GaRLILEO constraints).
  double tracking_lost_quality_threshold = 0.01; ///< [0..1] Default matches previous hard-coded behaviour.
  // Relative-pose fallback (optional): when vision is unreliable (low features or tracking lost),
  // add a strong GaRLILEO relative pose constraint (between factor) to pull OKVIS motion towards GaRLILEO.
  bool pose_fallback_enable = false;
  double pose_fallback_translation_std = 0.05;   ///< [m] Smaller => stronger constraint.
  double pose_fallback_rotation_roll_pitch_std_deg = 2.0; ///< Fallback roll/pitch std [deg]. Smaller => stronger constraint.
  double pose_fallback_rotation_yaw_std_deg = 2.0; ///< Fallback yaw std [deg]. Smaller => stronger constraint.
  double pose_fallback_weight = 1.0;            ///< Unitless multiplier on the relative pose information.
  double pose_fallback_velocity_std = -1.0;      ///< [m/s] If >0: use this std for velocity/speed prior during fallback. If <=0: use velocity_std.

  // Visual fallback (optional): when (low_features || tracking lost), downweight camera factors.
  // Implementation detail: we scale the information matrices of reprojection errors belonging to the
  // current frame before the main optimisation/marginalisation step. This reduces the influence of
  // camera reprojection residuals, and also indirectly weakens visual relative-pose constraints
  // created during marginalisation (TwoPoseGraphError) since they are built from these reprojections.
  // For more aggressive vision-only-suppression in dark / textureless conditions, the
  // *_information_scale fields below can also independently scale the marginalised
  // visual relative-pose error and submap alignment factors.
  bool visual_fallback_downweight_enable = false;
  double visual_fallback_reprojection_information_scale = 1.0; ///< Multiplier on 2x2 reprojection information (<1 => weaker camera).
  // Optional camera-only utility ramp. The existing scale is the upper target;
  // at/below utility_low use the minimum, at/above utility_high use the upper.
  // This does not change GaRLILEO weighting, submap weighting or LC classification.
  bool visual_fallback_reprojection_utility_adaptive = false;
  double visual_fallback_reprojection_information_scale_min = 0.01;
  double visual_fallback_reprojection_utility_low = 2.0;
  double visual_fallback_reprojection_utility_high = 3.0;
  // Multiplier on already-marginalised visual relative-pose constraints (TwoPoseGraphError).
  // <1 => the link's residual and Jacobian are scaled by sqrt(scale), which is equivalent to
  // multiplying the information matrix by `scale` (weaker visual coupling).
  double visual_fallback_relpose_information_scale = 1.0;
  // Multiplier on submap alignment factor information.
  // Implementation: the per-point sigma vector passed to addSubmapAlignmentConstraints is
  // divided by sqrt(scale), which multiplies the resulting information by `scale`.
  // <1 => weaker submap alignment.
  double visual_fallback_submap_information_scale = 1.0;

  // Landmark motion consistency gating (optional).
  // Goal: when a new 3D landmark would be created from a (prev,curr) match in
  //       Frontend::matchMotionStereo, also verify that the *observed pixel motion*
  //       (curr_pixel - prev_pixel) is consistent with the *predicted pixel motion*
  //       implied by the current pose estimate (which already includes any GaRLILEO
  //       state-propagation blend / between factors applied earlier in the frame).
  // Implementation: evaluate the epipolar residual induced by the current relative
  //       pose T_C1_C0 = T_WC1.inverse() * T_WC0 using back-projected bearing vectors.
  //       If the residual exceeds the threshold (derived from max_pixel), the
  //       new-landmark creation for this match is skipped (and so is its observation),
  //       which suppresses landmarks created from noisy or moving keypoints.
  //       Existing-landmark observation paths are not affected.
  bool landmark_motion_consistency_enable = false; ///< Enable the motion-consistency gate.
  double landmark_motion_consistency_max_pixel = 5.0; ///< Pixel-domain tolerance used to derive epipolar residual threshold.

  // Visual fallback - GaRLILEO between-factor enforcement (optional).
  // GaRLILEO is expected to publish a state on every camera-synchronous timestamp once
  // initialised, so by waiting briefly we can guarantee that the GaRLILEO measurement
  // matching this frame's timestamp is available before optimisation. When enabled, fallback
  // mode (low_features || tracking_lost) will block the worker thread up to
  // garlileo_wait_max_sec (polling at garlileo_wait_poll_period_ms) for an exact-ns
  // GaRLILEO measurement at the current and previous frame timestamps so the GaRLILEO
  // between factor is always added in fallback regimes.
  bool visual_fallback_garlileo_force_between = false; ///< Wait for and force GaRLILEO between in fallback.
  double visual_fallback_garlileo_wait_max_sec = 0.20; ///< Max blocking wait per query [s].
  int    visual_fallback_garlileo_wait_poll_period_ms = 5; ///< Polling period during the wait [ms].

  // Between-factor backfill (optional): late GaRLILEO messages can arrive after we already added the
  // between constraint for a pair (k-1,k). If enabled, we will revisit a small number of recent
  // pairs that are still in the sliding window and overwrite their GaRLILEO between constraints
  // when a closer time-associated measurement becomes available.
  bool between_backfill_enable = false;
  int between_backfill_max_pairs = 5; ///< Number of recent consecutive pairs (k-1,k) to consider for backfill (0 disables).
  /// If true, backfill overwrites existing between constraints whenever a usable newer GaRLILEO
  /// pair is available, even when |dt| is not improved (assumes newer = higher quality).
  bool between_backfill_overwrite_on_quality_improvement = false;

  // Radar input for place-recognition fusion (loop closure).
  bool place_recognition_radar0_enable = false; ///< Enable radar0 input for place recognition.
  bool place_recognition_radar1_enable = false; ///< Enable radar1 input for place recognition.
  double place_recognition_radar_max_time_diff = 0.05; ///< Max |dt| for associating radar scans with camera frames [s].
  int place_recognition_radar_scans_per_radar = 5; ///< Number of recent scans to aggregate per enabled radar (total can be up to 2x this value).
  okvis::kinematics::Transformation place_recognition_radar0_T_BR; ///< Radar0-to-body transform T_BR (R0->B).
  okvis::kinematics::Transformation place_recognition_radar1_T_BR; ///< Radar1-to-body transform T_BR (R1->B).

  // Place recognition (loop closure) using radar context descriptor (optional).
  // If enabled, the loop-closure candidate score is fused as:
  //   s = alpha * s_img + (1-alpha) * s_rad
  // where alpha is reduced when the current visual tracking quality is low.
  bool place_recognition_use_radar = false; ///< Derived: true if radar0 or radar1 is enabled. Fuses the radar-context score into visual DBoW2 place recognition.
  double place_recognition_radar_alpha = 1.0; ///< Base alpha for score fusion (will be reduced when vision is poor).
  double place_recognition_radar_min_intensity = 0.0; ///< Ignore radar points with intensity below this when building the polar image.
  bool fallback_loop_closure_persistent_enable = false; ///< Keep fallback-mode loop relative-pose constraints in the final full graph.
  bool fallback_loop_closure_full_graph_optimisation_enable = true; ///< Run background full-graph optimisation after fallback loop closure.
  bool fallback_loop_closure_path_constraints_enable = false; ///< Add extra relative-pose constraints along the corrected fallback loop path.
  int fallback_loop_closure_path_constraints_mode = 1; ///< 0: anchor-to-start, 1: chain along path, 2: both.
  int fallback_loop_closure_chain_constraint_stride = 1; ///< Add one chain constraint every N corrected poses.
  bool fallback_loop_weaken_existing_constraints_enable = false; ///< Remove/suppress existing visual/GaRLILEO path constraints before adding fallback LC chain constraints.
  double fallback_loop_existing_constraint_scale = 0.01; ///< Legacy compatibility; fallback LC now removes/suppresses existing path constraints instead of scaling them.
  double fallback_loop_information_scale = 10000.0; ///< Initial full-graph relative-pose information scale for fallback-mode loop closures.
  double fallback_loop_persistent_information_scale = 10.0; ///< Persistent full-graph information scale for fallback-mode loop closures.
  double fallback_loop_chain_information_scale = 100.0; ///< Information scale for fallback loop path chain constraints.
  /// If true (default): after a fallback LC with a complete corrected path, do not add GaRLILEO between
  /// factors on pairs that touch the corrected [pose_i, pose_j] ID range (chain + IMU handle it);
  /// resume GaRLILEO between only on consecutive pairs with both IDs strictly greater than max(pose_i, pose_j).
  bool fallback_loop_garlileo_between_resume_after_path_only = true;
  /// Visual fallback LC only: descriptor-distinctiveness gate rejects if
  /// avg < threshold and RANSAC inliers < 20 (same logic as the fixed 182.0 for normal LC).
  /// Use a lower value than 182 to relax (e.g. 80). Default 0 effectively disables the avg part
  /// in fallback (avg is nonnegative). Use negative (e.g. -1) to skip the avg check in fallback.
  double fallback_loop_closure_descriptor_distinctiveness_avg_threshold = 0.0;

};
/**
  * @brief Struct to specify parameters of GPS sensor
  */
struct GpsParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::string type; ///< Format of GPS data: "cartesian" | "geodetic" | "geodetic-leica":
    Eigen::Vector3d r_SA; ///< Translation IMU sensor to GPS antenna; known from calibration
    double yawErrorThreshold; /// < Threshold on maximum estimated yaw error [degree] for initialization
    bool robustGpsInit; /// < Flag if robust initialization is needed (low-grade GPS sensor)

    /// Default Constructor (no GPS)
    GpsParameters() : type("none"), r_SA(Eigen::Vector3d(0., 0., 0.)),
                      yawErrorThreshold(0.), robustGpsInit(false)
                      {}
};


/// @brief  Struct to specify the parameters of a LiDAR sensor
struct LidarParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SL;   ///< Transformation from LiDAR to sensor (IMU) frame.
    float elevation_resolution_angle; ///< Resolution angle for the elevation of the LiDAR sensor
    float azimuth_resolution_angle; ///< Resolution angle for the azimuth of the LiDAR sensor
};


/// @brief Struct to combine all parameters and settings.
struct ViParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::cameras::NCameraSystem nCameraSystem;  ///< Camera extrinsics and intrinsics.
  CameraParameters camera; ///< Camera parameters.
  ImuParameters imu; ///< Imu parameters.
  std::optional<GpsParameters> gps; ///< Gps parameters.
  std::optional<LidarParameters> lidar; ///< LiDAR parameters
  FrontendParameters frontend; ///< Frontend parameters.
  EstimatorParameters estimator; ///< Estimator parameters.
  OutputParameters output; ///< Output parameters.
  GaRLILEOParameters garlileo; ///< GaRLILEO pseudo-measurement coupling parameters.
  CameraCalibration rgb;  ///< RGB parameters.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_PARAMETERS_HPP_
