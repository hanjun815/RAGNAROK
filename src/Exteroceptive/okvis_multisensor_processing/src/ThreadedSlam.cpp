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
 * @file ThreadedSlam.cpp
 * @brief Source file for the ThreadedSlam3 class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <numeric>
#include <mutex>
#include <chrono>
#include <thread>
#include <optional>
#include <fstream>
 #include <limits>
#include <iomanip>
#include <sstream>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <utility>
 #include <pthread.h>
 #include <sched.h>
 #include <sys/resource.h>
 
 #include <opencv2/imgproc/imgproc.hpp>
 
 #include <glog/logging.h>
 
 #include <okvis/ThreadedSlam.hpp>
 #include <okvis/assert_macros.hpp>
 #include <okvis/ceres/ImuError.hpp>
 #include <okvis/LidarMotionUndistortion.hpp>
 
 #include <pcl/io/ply_io.h>
 #include <pcl/point_types.h>
 
 /// \brief okvis Main namespace of this package.
 namespace okvis
 {


// A single reusable camera worker. run() returns only after both cameras finish,
// including exceptional exits; the calling frame and pose references stay alive.
class StereoDetectionWorker {
 public:
  StereoDetectionWorker() = default;
  StereoDetectionWorker(const StereoDetectionWorker&) = delete;
  StereoDetectionWorker& operator=(const StereoDetectionWorker&) = delete;
  ~StereoDetectionWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  void run(std::function<void()> mainCamera, std::function<void()> workerCamera) {
    // Frame processing has one caller; also serialize accidental concurrent submissions.
    std::lock_guard<std::mutex> runLock(runMutex_);
    std::packaged_task<void()> next(std::move(workerCamera));
    auto completed = next.get_future();
    if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = std::move(next);
      pending_ = true;
    }
    ready_.notify_one();
    std::exception_ptr error;
    try { mainCamera(); } catch (...) { error = std::current_exception(); }
    try { completed.get(); } catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
  }

 private:
  void loop() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || pending_; });
        if (stopping_ && !pending_) return;
        task = std::move(task_);
        pending_ = false;
      }
      // packaged_task carries exceptions back to the frame-processing caller.
      task();
    }
  }
  std::mutex runMutex_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::packaged_task<void()> task_;
  bool pending_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

 static const int cameraInputQueueSize = 24;

// Cumulative GaRLILEO between-constraint accounting, summarised in stopThreading().
// Distinguishes constraints added immediately from those recovered later by the
// backfill pass, which is what decides whether the blocking wait is needed at all.
namespace {
std::atomic<uint64_t> g_betweenImmediate{0};   ///< added at the frame itself
std::atomic<uint64_t> g_betweenBackfilled{0};  ///< added later by backfill
std::atomic<uint64_t> g_betweenRemoved{0};     ///< removed again by backfill (over dt)
std::atomic<uint64_t> g_betweenPairs{0};       ///< pairs seen at the frame itself
std::atomic<uint64_t> g_backfillLagFrames{0};  ///< sum of (curr - backfilled) frame gaps

}  // namespace

 
 // overlap of imu data before and after two consecutive frames [seconds]:
 static const double imuTemporalOverlap = 0.02;
 
namespace {
std::string formatSecondsOrNa(double value) {
  if (!std::isfinite(value)) {
    return "n/a";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << value;
  return oss.str();
}

double positiveOrFallback(double value, double fallback) {
  return value > 0.0 ? value : fallback;
}

/// Paper Section F scales nominal information by 1 / max(tracking_quality, floor).
/// Since information is inverse variance, multiply stds by sqrt(max(quality, floor)).
double garlileoNominalStdScale(double tracking_quality,
                               double min_tracking_quality_floor) {
  const double q =
      std::isfinite(tracking_quality) ? std::max(0.0, tracking_quality) : 0.0;
  const double floor_q = std::max(0.0, min_tracking_quality_floor);
  return std::sqrt(std::max(q, floor_q));
}

double garlileoNominalStdScaleWithConfig(bool use_tracking_quality_scale,
                                         double tracking_quality,
                                         double min_tracking_quality_floor) {
  if (!use_tracking_quality_scale) {
    return 1.0;
  }
  return garlileoNominalStdScale(tracking_quality, min_tracking_quality_floor);
}

/// Blend weight in [0,1] toward pose-fallback from per-frame camera-utility aggregate.
/// Uses a sigmoid transition in [thr_low, thr_high] and utility hysteresis for smoother switching.
double utilityFallbackBlendFromEwgMean(double utility_aggregate,
                                       double thr_low,
                                       double thr_high,
                                       bool ewg_gate_enabled,
                                       double sigmoid_steepness,
                                       double hysteresis_width) {
  static thread_local bool has_hyst_utility = false;
  static thread_local double hyst_utility = std::numeric_limits<double>::quiet_NaN();
  if (!ewg_gate_enabled || !std::isfinite(utility_aggregate) || thr_low <= 0.0) {
    has_hyst_utility = false;
    hyst_utility = std::numeric_limits<double>::quiet_NaN();
    return 0.0;
  }

  const double hyst = std::max(0.0, hysteresis_width);
  double utility_eff = utility_aggregate;
  if (!has_hyst_utility || !std::isfinite(hyst_utility)) {
    hyst_utility = utility_aggregate;
    has_hyst_utility = true;
  } else if (std::abs(utility_aggregate - hyst_utility) > hyst) {
    hyst_utility = utility_aggregate;
  }
  utility_eff = hyst_utility;

  if (thr_high <= thr_low) {
    return (utility_eff < thr_low) ? 1.0 : 0.0;
  }
  if (utility_eff <= thr_low) {
    return 1.0;
  }
  if (utility_eff >= thr_high) {
    return 0.0;
  }

  const double center = 0.5 * (thr_low + thr_high);
  const double half_width = std::max(1.0e-9, 0.5 * (thr_high - thr_low));
  const double k = std::max(1.0e-6, sigmoid_steepness);
  auto logistic = [&](double u) -> double {
    const double x = (u - center) / half_width;
    const double z = std::clamp(k * x, -60.0, 60.0);
    return 1.0 / (1.0 + std::exp(z));  // decreasing with utility.
  };
  const double s = logistic(utility_eff);
  const double s_lo = logistic(thr_low);
  const double s_hi = logistic(thr_high);
  const double denom = std::max(1.0e-12, s_lo - s_hi);
  const double w = (s - s_hi) / denom;
  return std::clamp(w, 0.0, 1.0);
}

inline double lerpPair(double a, double b, double t) {
  const double tt = std::clamp(t, 0.0, 1.0);
  return (1.0 - tt) * a + tt * b;
}

// Optional camera-only response within the existing fallback regime. Keep the
// GaRLILEO/LC utility gate independent; an image can still supply useful camera
// observations even while its utility is below that gate. No history or sequence
// identity is used. Smoothstep avoids slope discontinuities at the two bounds.
double visualFallbackReprojectionTarget(double base_scale, bool adaptive,
                                         double minimum_scale, double utility_low,
                                         double utility_high, double utility) {
  if (!adaptive) {
    return base_scale;
  }
  // ViParametersReader rejects invalid enabled configurations. Also guard callers
  // constructing parameters directly, without introducing invalid factor weights.
  if (!std::isfinite(base_scale) || !std::isfinite(minimum_scale)
      || minimum_scale <= 0.0 || minimum_scale > base_scale
      || !std::isfinite(utility_low) || !std::isfinite(utility_high)
      || utility_low < 0.0 || utility_high <= utility_low) {
    return base_scale;
  }
  if (!std::isfinite(utility) || utility <= utility_low) {
    return minimum_scale;
  }
  if (utility >= utility_high) {
    return base_scale;
  }
  const double t = (utility - utility_low) / (utility_high - utility_low);
  const double blend = t * t * (3.0 - 2.0 * t);
  return lerpPair(minimum_scale, base_scale, blend);
}

Eigen::Matrix<double, 6, 6> makeGarlileoBetweenInformation(
    const GaRLILEOParameters& params,
    bool use_rotation,
    bool use_translation,
    double fallback_blend,
    double nominal_std_scale) {
  constexpr double kHugeStd = 1e6;
  constexpr double kDegToRad = 0.017453292519943295;

  const double s = std::max(0.0, nominal_std_scale);
  const double w_fb = std::clamp(fallback_blend, 0.0, 1.0);

  const double trans_nom = params.translation_std * s;
  const double rot_rp_nom_deg =
      positiveOrFallback(params.rotation_roll_pitch_std_deg, 5.0) * s;
  const double rot_yaw_nom_deg =
      positiveOrFallback(params.rotation_yaw_std_deg, 5.0) * s;

  const double trans_fb = params.pose_fallback_translation_std;
  const double rot_rp_fb_deg =
      positiveOrFallback(params.pose_fallback_rotation_roll_pitch_std_deg, 2.0);
  const double rot_yaw_fb_deg =
      positiveOrFallback(params.pose_fallback_rotation_yaw_std_deg, 2.0);

  const bool merge = params.pose_fallback_enable;
  const double trans_std_used =
      merge ? lerpPair(trans_nom, trans_fb, w_fb) : trans_nom;
  const double rot_rp_std_deg_used =
      merge ? lerpPair(rot_rp_nom_deg, rot_rp_fb_deg, w_fb) : rot_rp_nom_deg;
  const double rot_yaw_std_deg_used =
      merge ? lerpPair(rot_yaw_nom_deg, rot_yaw_fb_deg, w_fb) : rot_yaw_nom_deg;

  const double w_pose =
      merge ? lerpPair(1.0, std::max(0.0, params.pose_fallback_weight), w_fb) : 1.0;

  const double trans_std =
      use_translation ? std::max(1.0e-12, trans_std_used) : kHugeStd;
  const double rot_rp_std =
      (use_rotation ? std::max(1.0e-12, rot_rp_std_deg_used) : kHugeStd) * kDegToRad;
  const double rot_yaw_std =
      (use_rotation ? std::max(1.0e-12, rot_yaw_std_deg_used) : kHugeStd) * kDegToRad;

  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
  information.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * (w_pose / (trans_std * trans_std));
  information(3, 3) = w_pose / (rot_rp_std * rot_rp_std);
  information(4, 4) = w_pose / (rot_rp_std * rot_rp_std);
  information(5, 5) = w_pose / (rot_yaw_std * rot_yaw_std);
  return information;
}
}  // namespace

 
 // Constructor.
 ThreadedSlam::ThreadedSlam(ViParameters &parameters, std::string dBowDir, se::SubMapConfig subMapConfig) :
   visualizer_(parameters),
   hasStarted_(false),
   frontend_(parameters.nCameraSystem.numCameras(), dBowDir),
   parameters_(parameters),
   submapConfig_(subMapConfig),
   useAlignmentFactors_(submapConfig_.useMap2LiveFactors)
 {
   setBlocking(false);
   init();
   
   ///// HACK: multi-session and multi-agent //////
   //frontend_.loadComponent(
   //  "/Users/leuteneg/Documents/datasets/euroc/V2_03_difficult/mav0/okvis2-slam-final_map.g2o",
   //  parameters_.imu, parameters_.nCameraSystem);
   ////////////////////////////////////////////////
 }
 
bool ThreadedSlam::applyGarlileoBetweenIfAllowed(
    StateId poseId0,
    StateId poseId1,
    const okvis::kinematics::Transformation& T_S0S1,
    const Eigen::Matrix<double, 6, 6>& information) {
  if (parameters_.garlileo.fallback_loop_garlileo_between_resume_after_path_only
      && estimator_.suppressGarlileoBetweenForFallbackLoopPathPair(poseId0, poseId1)) {
    (void)estimator_.removeRelativePoseConstraint(poseId0, poseId1);
    return false;
  }
  // A rejected insertion must remain retryable and must not mark GaRLILEO ready.
  return estimator_.addRelativePoseConstraint(poseId0, poseId1, T_S0S1, information);
}
 
bool ThreadedSlam::updateGarlileoVelocityPrior(
    StateId poseId, const Eigen::Vector3d& v_S, double stddev, double abs_dt, bool enabled) {
  const double information = enabled ? 1.0 / (stddev * stddev) : 0.0;
  const bool applied = estimator_.setVelocityPrior(
      poseId, v_S, information, enabled);
  auto& association = garlileoVelocityAssocById_[poseId.value()];
  if(applied) {
    association.abs_dt = abs_dt;
    association.has_constraint = enabled;
    association.retry_pending = false;
  } else {
    // Keep the association of the prior that is actually still in the graph.
    // Existing bounded backfill retries once full-graph access becomes available.
    association.retry_pending = true;
  }
  return applied;
}

 // Initialises settings and calls startThreads().
 void ThreadedSlam::init()
 {
   assert(parameters_.nCameraSystem.numCameras() > 0);
   const size_t numCameras = parameters_.nCameraSystem.numCameras();
   shutdown_ = false;
  frontendDeltaHasPrev_ = false;
  garlileoHistory_.clear();
  garlileoGravityHistory_.clear();
  hasLastGarlileo_ = false;
  hasLastGarlileoGravity_ = false;
  garlileoFrameQualityById_.clear();
  garlileoBetweenAssocByCurrId_.clear();
  garlileoVelocityAssocById_.clear();
  lastCameraQueuePopWallTimeSec_.store(-1.0, std::memory_order_relaxed);
  lastOptimisationDurationSec_.store(-1.0, std::memory_order_relaxed);
  lastOptimisationFinishedWallTimeSec_.store(-1.0, std::memory_order_relaxed);
  lastIncomingCameraStampSec_.store(-1.0, std::memory_order_relaxed);
 
   // setup frontend
   frontend_.setBriskDetectionOctaves(size_t(parameters_.frontend.octaves));
   frontend_.setBriskDetectionThreshold(parameters_.frontend.detection_threshold);
   frontend_.setBriskDetectionAbsoluteThreshold(parameters_.frontend.absolute_threshold);
   frontend_.setBriskMatchingThreshold(parameters_.frontend.matching_threshold);
   frontend_.setBriskDetectionMaximumKeypoints(size_t(parameters_.frontend.max_num_keypoints));
   frontend_.setKeyframeInsertionOverlapThreshold(float(parameters_.frontend.keyframe_overlap));
  frontend_.setVignetteCorrection(parameters_.frontend.vignette_correction_enable,
                                  parameters_.frontend.vignette_correction_k1,
                                  parameters_.frontend.vignette_correction_k2,
                                  parameters_.frontend.vignette_correction_k3);
  frontend_.setAgcwdPreprocessing(parameters_.frontend.agcwd_enable,
                                 parameters_.frontend.agcwd_weighting_param);
  frontend_.setClahePreprocessing(parameters_.frontend.clahe_enable,
                                 parameters_.frontend.clahe_clip_limit,
                                 parameters_.frontend.clahe_tile_grid_size);
  frontend_.setAgcwdClaheBayesOptConfig(
      parameters_.frontend.agcwd_clahe_bo_enable,
      parameters_.frontend.agcwd_clahe_bo_utility_discrepancy_threshold,
      parameters_.frontend.agcwd_clahe_bo_initial_samples,
      parameters_.frontend.agcwd_clahe_bo_iterations,
      parameters_.frontend.agcwd_clahe_bo_candidate_grid,
      parameters_.frontend.agcwd_clahe_bo_agcwd_min,
      parameters_.frontend.agcwd_clahe_bo_agcwd_max,
      parameters_.frontend.agcwd_clahe_bo_clahe_min,
      parameters_.frontend.agcwd_clahe_bo_clahe_max,
      parameters_.frontend.agcwd_clahe_bo_image_utility_weight);
  frontend_.setEwgImageUtilityConfig(parameters_.frontend.image_utility_ewg_Hthres,
                                     parameters_.frontend.image_utility_ewg_alpha,
                                     parameters_.frontend.image_utility_ewg_tau,
                                     parameters_.frontend.image_utility_ewg_local_patch,
                                     parameters_.frontend.image_utility_ewg_num_bins,
                                     parameters_.frontend.image_utility_ewg_lambda_f,
                                     parameters_.frontend.image_utility_ewg_N0,
                                     parameters_.frontend.image_utility_ewg_grid_rows,
                                     parameters_.frontend.image_utility_ewg_grid_cols,
                                     parameters_.frontend.image_utility_ewg_resize_enable,
                                     parameters_.frontend.image_utility_ewg_resize_max_side,
                                     parameters_.frontend.image_utility_fallback_use_preprocessed);
 
   // setup estimator
   estimator_.addImu(parameters_.imu);
   for (size_t im = 0; im < numCameras; ++im) {
     // parameters_.camera_extrinsics is never set (default 0's)...
     // do they ever change?
     estimator_.addCamera(parameters_.camera);
   }
   if(parameters_.gps){
     estimator_.addGps(*parameters_.gps);
   }
   estimator_.setDetectorUniformityRadius(parameters_.frontend.detection_threshold);
 
   // time limit if requested
   if(parameters_.estimator.enforce_realtime) {
     estimator_.setOptimisationTimeLimit(
           parameters_.estimator.realtime_time_limit,
           parameters_.estimator.realtime_min_iterations);
   }
 
   // Set calibration parameters for live-to-map factor
   // TODO: generalize to n > 1 mapping camera
   if(parameters_.output.enable_submapping) {
     if(parameters_.lidar){
       T_SD_ = parameters_.lidar.value().T_SL;
     }
     else{
       std::shared_ptr<const cameras::CameraBase> camera;
       kinematics::Transformation T_SD;
       bool mapping_camera_found = false;
       for(size_t i = 0; i < parameters_.nCameraSystem.numCameras(); i++){
         if(parameters_.nCameraSystem.cameraType(i).isUsedMapping) {
           mapping_camera_found = true;
           if (!parameters_.nCameraSystem.cameraType(i).depthType.needRectify) {
             camera = parameters_.nCameraSystem.cameraGeometry(i);
             T_SD = *parameters_.nCameraSystem.T_SC(i);
           } else {
             camera = parameters_.nCameraSystem.rectifyCameraGeometry(i);
             T_SD = *parameters_.nCameraSystem.rectifyT_SC(i);
             T_rect_ = parameters_.nCameraSystem.T_SC(i)->inverse() * T_SD;
           }
           break;
         }
       }
       if(!mapping_camera_found){
         LOG(WARNING) << "Neither LiDAR nor camera for mapping specified! Running in VI mode without depth." \
         "If you want to run without depth specify enabling_submapping: false in the okvis_config. If you want to use" \
         "the LiDAR, please specify a lidar section in the okvis_config. If you wanted to use depth from cameras " \
         ", please specify mapping: true for the camera from which depth will be integrated.";
         useAlignmentFactors_ = false;
       } else {
         se::PinholeCamera::Config depthCameraConfig;
         Eigen::VectorXd intrinsics;
         camera->getIntrinsics(intrinsics);
         depthCameraConfig.fx = intrinsics(0);
         depthCameraConfig.fy = intrinsics(1);
         depthCameraConfig.cx = intrinsics(2);
         depthCameraConfig.cy = intrinsics(3);
         depthCameraConfig.T_BS = T_SD.T().cast<float>();
         depthCameraConfig.width = camera->imageWidth();
         depthCameraConfig.height = camera->imageHeight();
         depthCameraConfig.near_plane = submapConfig_.near_plane;
         depthCameraConfig.far_plane = submapConfig_.far_plane;
         depthCamera_.reset(new se::PinholeCamera(depthCameraConfig, submapConfig_.depthImageResDownsampling));
         T_SD_ = T_SD;
       }
     }
   } else if(useAlignmentFactors_){
     useAlignmentFactors_ = false;
     LOG(WARNING) << "Alignment factors have been set in the se2_config but the enable_submapping in the okvis_config is set to false." \
     " Due to this inconsistency you will be running VI mode no submap constraints.";
   }
 
   if(useAlignmentFactors_ && parameters_.output.enable_submapping){
     LOG(INFO) << "[Info SLAM] SLAM will be using Submap Alignment Constraints.";
   }
 
   startThreads();
 }
 
 // Start all threads.
 void ThreadedSlam::startThreads()
 {
 
   // visualisation
   if(parameters_.output.display_matches) {
     visualisationThread_ = std::thread(&ThreadedSlam::visualisationLoop, this);
   }
 
   //setMainThreadPriority(SCHED_RR, 99);
 
   // publishing
   publishingThread_ = std::thread(&ThreadedSlam::publishingLoop, this);
 
 }
 
 // Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
 ThreadedSlam::~ThreadedSlam()
 {
 
   // shutdown and join threads
   stopThreading();
   stereoDetectionWorker_.reset();
 }
 
 // Add a new image.
 bool ThreadedSlam::addImages(const okvis::Time & stamp,
                              const std::map<size_t, cv::Mat> & images,
                              const std::map<size_t, cv::Mat> & depthImages)
 {
   // remove image delay:
   // timestamp_camera_correct = timestamp_camera - image_delay
   const Time stampCorrected = stamp - Duration(parameters_.camera.image_delay);
 
   // assemble frame
   const size_t numCameras = parameters_.nCameraSystem.numCameras();
   std::vector<okvis::CameraMeasurement> frames(numCameras);
   frames.at(0).timeStamp = stampCorrected; // slight hack -- always have the timestamp here.
 
   bool useFrame = false;
   for(const auto & image : images) {
     if(parameters_.nCameraSystem.isCameraConfigured(image.first) && 
        parameters_.nCameraSystem.cameraType(image.first).isUsed) {
       if(image.second.channels()==1) {
         frames.at(image.first).measurement.image = image.second;
       } else {
         cv::cvtColor(image.second, frames.at(image.first).measurement.image, cv::COLOR_BGR2GRAY);
       }
       frames.at(image.first).timeStamp = stampCorrected;
       frames.at(image.first).sensorId = image.first;
       frames.at(image.first).measurement.deliversKeypoints = false;
       useFrame = true;
     }
   }
 
   for(const auto & depthImage : depthImages) {
     if(parameters_.nCameraSystem.cameraType(depthImage.first).isUsed) {
       frames.at(depthImage.first).measurement.depthImage = depthImage.second;
       frames.at(depthImage.first).timeStamp = stampCorrected;
       frames.at(depthImage.first).sensorId = depthImage.first;
       frames.at(depthImage.first).measurement.deliversKeypoints = false;
     }
   }
 
   if(!useFrame) {
     return true; // the frame is not used, so quitely ignore.
   }

  const double stamp_sec = stampCorrected.toSec();
  const double prev_stamp_sec =
      lastIncomingCameraStampSec_.exchange(stamp_sec, std::memory_order_relaxed);
  const double incoming_dt_sec = (prev_stamp_sec > 0.0)
                                     ? (stamp_sec - prev_stamp_sec)
                                     : std::numeric_limits<double>::quiet_NaN();
 
   if (blocking_)
   {
     return cameraMeasurementsReceived_.PushBlockingIfFull(frames,1);
   }
   else
   {
     if(cameraMeasurementsReceived_.PushNonBlockingDroppingIfFull(frames, cameraInputQueueSize)) {
      const size_t queue_size = cameraMeasurementsReceived_.Size();
      std::vector<okvis::CameraMeasurement> oldest_frames;
      std::vector<okvis::CameraMeasurement> newest_frames;
      double queue_span_sec = std::numeric_limits<double>::quiet_NaN();
      if (cameraMeasurementsReceived_.getCopyOfFront(&oldest_frames)
          && cameraMeasurementsReceived_.getCopyOfBack(&newest_frames)
          && !oldest_frames.empty() && !newest_frames.empty()) {
        queue_span_sec =
            std::abs((newest_frames.front().timeStamp - oldest_frames.front().timeStamp).toSec());
      }
      const double now_wall_sec = okvis::Time::now().toSec();
      const double last_pop_wall_sec =
          lastCameraQueuePopWallTimeSec_.load(std::memory_order_relaxed);
      const double since_last_pop_sec = (last_pop_wall_sec > 0.0)
                                            ? (now_wall_sec - last_pop_wall_sec)
                                            : std::numeric_limits<double>::quiet_NaN();
      const double last_optimisation_dur_sec =
          lastOptimisationDurationSec_.load(std::memory_order_relaxed);
      const double last_optimisation_end_wall_sec =
          lastOptimisationFinishedWallTimeSec_.load(std::memory_order_relaxed);
      const double since_last_optimisation_sec = (last_optimisation_end_wall_sec > 0.0)
                                                     ? (now_wall_sec - last_optimisation_end_wall_sec)
                                                     : std::numeric_limits<double>::quiet_NaN();
      const bool likely_processing_backlog =
          (!std::isfinite(incoming_dt_sec) || incoming_dt_sec <= 0.2)
          || (std::isfinite(queue_span_sec) && queue_span_sec > 0.2);
      const char* cause = likely_processing_backlog
                              ? "camera_queue_full_processing_backlog"
                              : "camera_queue_full_input_timestamp_jump_or_burst";
      LOG(WARNING) << "[FrameDrop][QueueFull] t=" << stampCorrected
                   << " cause=" << cause
                   << " queue_size=" << queue_size << "/" << cameraInputQueueSize
                   << " queue_span_sec=" << formatSecondsOrNa(queue_span_sec)
                   << " incoming_dt_sec=" << formatSecondsOrNa(incoming_dt_sec)
                   << " since_last_dequeue_sec=" << formatSecondsOrNa(since_last_pop_sec)
                   << " last_optimise_dur_sec=" << formatSecondsOrNa(last_optimisation_dur_sec)
                   << " since_last_optimise_done_sec=" << formatSecondsOrNa(since_last_optimisation_sec)
                   << " hint='consumer slower than producer; large optimise_dur implies optimisation bottleneck'";
       return false;
     }
     return true;
   }
 }
 
 // Add an IMU measurement.
 bool ThreadedSlam::addImuMeasurement(const okvis::Time& stamp,
                                      const Eigen::Vector3d& alpha,
                                      const Eigen::Vector3d& omega)
 {
   static bool warnOnce = true;
   if (!parameters_.imu.use) {
     if (warnOnce) {
       LOG(WARNING) << "imu measurement added, but IMU disabled";
       warnOnce = false;
     }
     return false;
   }
   okvis::ImuMeasurement imu_measurement;
   imu_measurement.measurement.accelerometers.x() = parameters_.imu.s_a.x() * alpha.x();
   imu_measurement.measurement.accelerometers.y() = parameters_.imu.s_a.y() * alpha.y();
   imu_measurement.measurement.accelerometers.z() = parameters_.imu.s_a.z() * alpha.z();
   imu_measurement.measurement.gyroscopes = omega;
   imu_measurement.timeStamp = stamp;
 
   const int imuQueueSize = 5000;
 
   if (blocking_)
   {
     return imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, size_t(imuQueueSize));
   }
   else
   {
     if(imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
          imu_measurement, size_t(imuQueueSize))) {
       LOG(WARNING) << "imu measurement drop ";
       return false;
     }
     return true;
   }
 
 }
 
 // Add a LiDAR measurement.
 bool ThreadedSlam::addLidarMeasurement(const okvis::Time &stamp,
                                        const Eigen::Vector3d &rayMeasurement)
 {
   if(!useAlignmentFactors_){
     return false;
   }
 
   okvis::LidarMeasurement lidarMeasurement;
   lidarMeasurement.measurement.rayMeasurement = rayMeasurement;
   lidarMeasurement.measurement.intensity = 0;
   lidarMeasurement.timeStamp = stamp;
 
   const int lidarQueueSize = 500000;
   sensorMeasurementDownsamplingCounter_ ++;
   bool drop = false;
   if(sensorMeasurementDownsamplingCounter_ % submapConfig_.sensorMeasurementDownsampling == 0){
     if (blocking_)
     {
       return lidarMeasurementsReceived_.PushBlockingIfFull(lidarMeasurement, size_t(lidarQueueSize));
     }
     else
     {
       drop = lidarMeasurementsReceived_.PushNonBlockingDroppingIfFull(
               lidarMeasurement, size_t(lidarQueueSize));
       if(drop)
         LOG(WARNING) << "lidar measurement drop ";
     }
     return drop;
   }
   return true;
 }
 
 bool ThreadedSlam::addDepthMeasurement(const okvis::Time &stamp,
                                        const cv::Mat &depthImage,
                                        const std::optional<cv::Mat> &sigmaImage) {
   CameraMeasurement depthMeasurement;
   
   // This assumes the image_delay is also applied to the depth camera.
   // That is valid for most of RGB-D cameras and network depths.
   // But, caution is needed if a depth camera has a different timestamp delay than gray (or RGB) camera.
   depthMeasurement.timeStamp = stamp - Duration(parameters_.camera.image_delay);
   depthMeasurement.measurement.depthImage = submapConfig_.depthScalingFactor*depthImage;
   if (sigmaImage) {
     depthMeasurement.measurement.sigmaImage = sigmaImage.value();
   }
 
   // We want to buffer 4 depth images to give OKVIS to have time to compute the states
   const size_t depthQueueSize = 4;
   sensorMeasurementDownsamplingCounter_ ++;
   bool drop = false;
   if(sensorMeasurementDownsamplingCounter_ % submapConfig_.sensorMeasurementDownsampling == 0){
     sensorMeasurementDownsamplingCounter_ = 0;
     if (blocking_) {
       return depthMeasurementsReceived_.PushBlockingIfFull(depthMeasurement, depthQueueSize);
     }
     else {
       drop = depthMeasurementsReceived_.PushNonBlockingDroppingIfFull(depthMeasurement, depthQueueSize);
       if (drop) 
         LOG(WARNING) << "Depth measurement drop ";
     }
     return drop;
   }
   return true;
 }

void ThreadedSlam::addGaRLILEOPseudoMeasurement(const okvis::Time& stamp,
                                                const Eigen::Quaterniond& R_WS,
                                                const Eigen::Vector3d& v_body,
                                                const Eigen::Vector3d& p_W) {
  const bool q_finite = std::isfinite(R_WS.w()) && std::isfinite(R_WS.x())
                        && std::isfinite(R_WS.y()) && std::isfinite(R_WS.z());
  const double q_norm = R_WS.norm();
  if (!q_finite || !std::isfinite(q_norm) || q_norm < 1.0e-12
      || !v_body.allFinite() || !p_W.allFinite()) {
    static thread_local double last_invalid_log = 0.0;
    const double now = okvis::Time::now().toSec();
    if (now - last_invalid_log > 1.0) {
      LOG(WARNING) << "[GaRLILEO] dropped invalid spline_state measurement: "
                   << "q_norm=" << q_norm
                   << " q_finite=" << q_finite
                   << " v_finite=" << v_body.allFinite()
                   << " p_finite=" << p_W.allFinite();
      last_invalid_log = now;
    }
    return;
  }
  GaRLILEOMeasurement meas;
  meas.timeStamp = stamp;
  meas.R_WS = R_WS.normalized();
  meas.v_body = v_body;
  meas.p_W = p_W;
  const size_t queueSize = 5000;
  garlileoMeasurementsReceived_.PushNonBlockingDroppingIfFull(meas, queueSize);
  {
    std::lock_guard<std::mutex> lock(garlileoMutex_);
    lastGarlileoMeas_ = meas;
    hasLastGarlileo_ = true;
  }
  static thread_local double last_log = 0.0;
  double now = stamp.toSec();
  if (now - last_log > 1.0) {
    LOG(INFO) << "[GaRLILEO] queued measurement t=" << stamp.toSec();
    last_log = now;
  }
}

void ThreadedSlam::addGaRLILEOGravityMeasurement(const okvis::Time& stamp,
                                                 const Eigen::Vector3d& g_W) {
  if (!g_W.allFinite() || g_W.squaredNorm() < 1.0e-12) {
    static thread_local double last_invalid_log = 0.0;
    const double now = okvis::Time::now().toSec();
    if (now - last_invalid_log > 1.0) {
      LOG(WARNING) << "[GaRLILEO] dropped invalid gravity measurement: "
                   << "g_finite=" << g_W.allFinite()
                   << " g_norm=" << g_W.norm();
      last_invalid_log = now;
    }
    return;
  }
  GaRLILEOGravityMeasurement meas;
  meas.timeStamp = stamp;
  meas.g_W = g_W;
  const size_t queueSize = 5000;
  const bool dropped_oldest =
      garlileoGravityMeasurementsReceived_.PushNonBlockingDroppingIfFull(meas, queueSize);
  if (dropped_oldest) {
    static thread_local double last_drop_log = 0.0;
    const double now = okvis::Time::now().toSec();
    if (now - last_drop_log > 1.0) {
      LOG(WARNING) << "[GaRLILEO] gravity queue full in OKVIS. "
                   << "Dropped oldest sample, accepted newest.";
      last_drop_log = now;
    }
  }
  {
    std::lock_guard<std::mutex> lock(garlileoGravityMutex_);
    lastGarlileoGravityMeas_ = meas;
    hasLastGarlileoGravity_ = true;
  }
  static thread_local double last_log = 0.0;
  const double now = stamp.toSec();
  if (now - last_log > 1.0) {
    LOG(INFO) << "[GaRLILEO] queued gravity t=" << stamp.toSec()
              << " g_W=[" << g_W.transpose() << "]";
    last_log = now;
  }
}

void ThreadedSlam::addRadarTargetsMeasurement(const okvis::Time& stamp,
                                              const okvis::RadarTargetReadings& targets,
                                              int radar_index) {
  RadarTargetsMeasurement meas;
  meas.timeStamp = stamp;
  meas.targets = targets;
  meas.radarIndex = radar_index;
  const size_t queueSize = 200; // radar is usually ~10-20Hz; keep moderate buffer
  radarTargetsMeasurementsReceived_.PushNonBlockingDroppingIfFull(meas, queueSize);
  {
    std::lock_guard<std::mutex> lock(radarTargetsMutex_);
    lastRadarTargetsMeas_ = meas;
    hasLastRadarTargets_ = true;
  }
  static double last_log = 0.0;
  const double now = stamp.toSec();
  if (now - last_log > 1.0) {
    LOG(INFO) << "[RadarPR] queued radar scan t=" << stamp.toSec()
              << " radar_index=" << radar_index
              << " num_targets=" << targets.size();
    last_log = now;
  }
 }
 
 // Add a GPS measurement.
 bool ThreadedSlam::addGpsMeasurement(const okvis::Time& stamp,
                                      const Eigen::Vector3d& pos,
                                      const Eigen::Vector3d& err)
 {
   okvis::GpsMeasurement  gps_measurement;
   okvis::GpsSensorReadings gpsReading(pos, err(0), err(1), err(2));
   gps_measurement.timeStamp = stamp;
   gps_measurement.measurement = gpsReading;
   const int gpsQueueSize = 5000;
 
   if (blocking_)
   {
     return gpsMeasurementsReceived_.PushBlockingIfFull(gps_measurement, size_t(gpsQueueSize));
   }
   else
   {
     if(gpsMeasurementsReceived_.PushNonBlockingDroppingIfFull(gps_measurement, size_t(gpsQueueSize))) {
       LOG(WARNING) << "gps measurement drop ";
       return false;
     }
 
     return true;
   }
 
 }
 
 // Add a GPS measurement (geodetic input).
 bool ThreadedSlam::addGeodeticGpsMeasurement(const okvis::Time& stamp,
                                              double lat, double lon, double height,
                                              double hAcc, double vAcc)
 {
 
   okvis::GpsMeasurement  gps_measurement;
   okvis::GpsSensorReadings gpsReading(lat, lon, height, hAcc, vAcc);
   gps_measurement.timeStamp = stamp;
   gps_measurement.measurement = gpsReading;
   const int gpsQueueSize = 5000;
 
   if (blocking_)
   {
     return gpsMeasurementsReceived_.PushBlockingIfFull(gps_measurement, size_t(gpsQueueSize));
   }
   else
   {
     if(gpsMeasurementsReceived_.PushNonBlockingDroppingIfFull(gps_measurement, size_t(gpsQueueSize))) {
       LOG(WARNING) << "gps measurement drop ";
       return false;
     }
 
     return true;
   }
 
 }
 
 // Add Submap alignment constraints to estimator
 bool ThreadedSlam::addSubmapAlignmentConstraints(const SupereightMapType* submap_A_ptr,
                                                  const SupereightMapType* submap_B_ptr,
                                                  const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                                                  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                                  std::vector<float> sensorError) {
 
   submapAlignmentFactorsReceived_.PushNonBlocking(
           AlignmentTerm(submap_A_ptr, submap_B_ptr,frame_A_id,frame_B_id, pointCloud, sensorError));
 
   return true;
 }
 
 // Set the blocking variable that indicates whether the addMeasurement() functions
 // should return immediately (blocking=false), or only when the processing is complete.
 void ThreadedSlam::setBlocking(bool blocking)
 {
   blocking_ = blocking;
   // disable time limit for optimization
   if(blocking_)
   {
     /// \todo Lock estimator
     //estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
   }
 }
 
 bool ThreadedSlam::getNextFrame(MultiFramePtr &multiFrame) {
   std::vector<okvis::CameraMeasurement> frames;
   if(!cameraMeasurementsReceived_.getCopyOfFront(&frames)) {
     return false;
   }
   const size_t numCameras = frames.size();
   multiFrame.reset(new okvis::MultiFrame(parameters_.nCameraSystem, frames.at(0).timeStamp, 0));
   frontend_.prepareImageUtilityFrameId(multiFrame->id());
   bool success = false;
   for(size_t im = 0; im < numCameras; ++im)
   {
     cv::Mat filtered = frames.at(im).measurement.image;
     if(!filtered.empty()) {
       success = true; // require at least one image, depth-only is not considered a success.
     }
     multiFrame->setImage(im, filtered);
     multiFrame->setDepthImage(im, frames.at(im).measurement.depthImage);
   }
   return success;
 }
 
 bool ThreadedSlam::processFrame() {
   MultiFramePtr multiFrame;
   ImuMeasurement imuMeasurement;
   LidarMeasurement lidarMeasurement;
   GpsMeasurement gpsMeasurement;
   CameraMeasurement depthMeasurement;
   const size_t numCameras = parameters_.nCameraSystem.numCameras();
 
   kinematics::Transformation T_WS;
   SpeedAndBias speedAndBias;

  auto isValidGarlileoMeasurement = [&](const GaRLILEOMeasurement& m) -> bool {
    const Eigen::Quaterniond q = m.R_WS;
    const bool q_finite = std::isfinite(q.w()) && std::isfinite(q.x())
                          && std::isfinite(q.y()) && std::isfinite(q.z());
    const double q_norm = q.norm();
    return q_finite && std::isfinite(q_norm) && q_norm > 1.0e-12
           && m.v_body.allFinite() && m.p_W.allFinite();
  };
  auto isValidGarlileoGravityMeasurement = [&](const GaRLILEOGravityMeasurement& m) -> bool {
    return m.g_W.allFinite() && (m.g_W.squaredNorm() > 1.0e-12);
  };
  auto garlileoKeepSec = [&]() -> double {
    return parameters_.garlileo.history_sec
           + parameters_.garlileo.max_time_diff
           + 1.0;  // guard for delayed publish
  };
  auto drainGarlileoHistory = [&](const okvis::Time& t_ref) {
    GaRLILEOMeasurement m;
    static double last_invalid_log = 0.0;
    while (!shutdown_ && garlileoMeasurementsReceived_.PopNonBlocking(&m)) {
      if (!isValidGarlileoMeasurement(m)) {
        const double now = okvis::Time::now().toSec();
        if (now - last_invalid_log > 1.0) {
          LOG(WARNING) << "[GaRLILEO] ignoring invalid measurement from queue.";
          last_invalid_log = now;
        }
        continue;
      }
      garlileoHistory_.push_back(m);
    }
    if (garlileoHistory_.empty() && hasLastGarlileo_) {
      std::lock_guard<std::mutex> lock(garlileoMutex_);
      if (isValidGarlileoMeasurement(lastGarlileoMeas_)) {
        garlileoHistory_.push_back(lastGarlileoMeas_);
      }
    }
    const double keep_sec = garlileoKeepSec();
    while (!garlileoHistory_.empty()
           && (t_ref - garlileoHistory_.front().timeStamp).toSec() > keep_sec) {
      garlileoHistory_.pop_front();
    }
  };
  auto drainGarlileoGravityHistory = [&](const okvis::Time& t_ref) {
    GaRLILEOGravityMeasurement m;
    static double last_invalid_log = 0.0;
    while (!shutdown_ && garlileoGravityMeasurementsReceived_.PopNonBlocking(&m)) {
      if (!isValidGarlileoGravityMeasurement(m)) {
        const double now = okvis::Time::now().toSec();
        if (now - last_invalid_log > 1.0) {
          LOG(WARNING) << "[GaRLILEO] ignoring invalid gravity measurement from queue.";
          last_invalid_log = now;
        }
        continue;
      }
      garlileoGravityHistory_.push_back(m);
    }
    if (garlileoGravityHistory_.empty() && hasLastGarlileoGravity_) {
      std::lock_guard<std::mutex> lock(garlileoGravityMutex_);
      if (isValidGarlileoGravityMeasurement(lastGarlileoGravityMeas_)) {
        garlileoGravityHistory_.push_back(lastGarlileoGravityMeas_);
      }
    }
    const double keep_sec = garlileoKeepSec();
    while (!garlileoGravityHistory_.empty()
           && (t_ref - garlileoGravityHistory_.front().timeStamp).toSec() > keep_sec) {
      garlileoGravityHistory_.pop_front();
    }
  };

  // If image preprocessing is enabled (vignette/enhancement), publish the preprocessed images for RViz debugging.
  // We convert mono images to BGR since Publisher::publishImages uses "bgr8" encoding.
  auto pushAgcwdImagesForRviz = [&]() {
    if (!parameters_.frontend.vignette_correction_enable
        && !parameters_.frontend.agcwd_enable
        && !parameters_.frontend.clahe_enable) {
      return;
    }
    std::vector<cv::Mat> imgs(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
      cv::Mat m;
      if (!frontend_.getLastAgcwdImage(i, m) || m.empty()) {
        continue;
      }
      if (m.channels() == 1) {
        cv::cvtColor(m, imgs[i], cv::COLOR_GRAY2BGR);
      } else if (m.channels() == 3) {
        imgs[i] = m;
      } else if (m.channels() == 4) {
        cv::cvtColor(m, imgs[i], cv::COLOR_BGRA2BGR);
      }
    }
    agcwdImages_.PushNonBlockingDroppingIfFull(imgs, 1);
  };

  // Check whether an exact (nanosecond-level) GaRLILEO timestamp exists for the query.
  // If polling_wait_max_sec > 0, this will block (polling at polling_period_ms) until either
  //   - an exact-ns GaRLILEO measurement for `t_query` is available in the history / last cache,
  //   - the wait deadline expires,
  //   - or shutdown_ is requested.
  // The time-tolerance argument is informational: we always require an exact-ns match, since
  // GaRLILEO is expected to publish a state at every camera-synchronous timestamp once
  // initialised. Callers that don't want to block pass polling_wait_max_sec=0.0.
  auto waitForGarlileoMeasurement =
      [&](const okvis::Time& t_query,
          double dt_tolerance,
          const char* tag,
          double polling_wait_max_sec,
          int polling_period_ms) -> bool {
        (void)dt_tolerance;
        const uint64_t query_ns = t_query.toNSec();
        auto hasRequired = [&]() -> bool {
          for (const auto& m : garlileoHistory_) {
            if (!isValidGarlileoMeasurement(m)) {
              continue;
            }
            if (m.timeStamp.toNSec() == query_ns) {
              return true;
            }
          }
          if (hasLastGarlileo_) {
            std::lock_guard<std::mutex> lock(garlileoMutex_);
            if (isValidGarlileoMeasurement(lastGarlileoMeas_)) {
              if (lastGarlileoMeas_.timeStamp.toNSec() == query_ns) {
                return true;
              }
            }
          }
          return false;
        };
        drainGarlileoHistory(t_query);
        if (hasRequired()) {
          return true;
        }
        if (polling_wait_max_sec <= 0.0 || shutdown_) {
          return false;
        }
        const auto t_start = std::chrono::steady_clock::now();
        const auto deadline = t_start
                              + std::chrono::microseconds(
                                    static_cast<int64_t>(polling_wait_max_sec * 1.0e6));
        const int sleep_ms = std::max(1, polling_period_ms);
        static double last_wait_log = 0.0;
        const double now_log = okvis::Time::now().toSec();
        bool got = false;
        size_t poll_iters = 0;
        while (!shutdown_ && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
          drainGarlileoHistory(t_query);
          ++poll_iters;
          if (hasRequired()) {
            got = true;
            break;
          }
        }
        const double elapsed_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        if (got) {
          if (now_log - last_wait_log > 1.0) {
            LOG(INFO) << "[GaRLILEO][Wait] " << (tag ? tag : "")
                      << " satisfied after polling: iters=" << poll_iters
                      << " elapsed=" << elapsed_sec << "s"
                      << " budget=" << polling_wait_max_sec << "s";
            last_wait_log = now_log;
          }
        } else {
          if (now_log - last_wait_log > 1.0) {
            LOG(WARNING) << "[GaRLILEO][Wait] " << (tag ? tag : "")
                         << " timed out: iters=" << poll_iters
                         << " elapsed=" << elapsed_sec << "s"
                         << " budget=" << polling_wait_max_sec << "s";
            last_wait_log = now_log;
          }
        }
        return got;
      };

  // Optional: use GaRLILEO to improve the frontend pose prediction (T_WS -> T_WC) used for
  // feature extraction direction / camera-aware descriptor extraction.
  // This does *not* replace IMU usage in the estimator; it only affects the pose guess passed to
  // Frontend::detectAndDescribe.
  auto applyGarlileoFrontendPrediction =
      [&](const okvis::Time& frameStamp, okvis::kinematics::Transformation& T_WS_pred) {
        if (!parameters_.garlileo.frontend_use_garlileo) {
          frontendDeltaHasPrev_ = false;
          return;
        }
        static double last_log = 0.0;
        // Use wall-time for throttling so logs show up "in real time" even if dataset timestamps
        // are not advancing as expected.
        const double now = okvis::Time::now().toSec();

        auto logFallbackImu = [&](const char* why) {
          if (now - last_log > 1.0) {
            if (parameters_.imu.use) {
              LOG(INFO) << "[GaRLILEO][Frontend] GaRLILEO pose prediction not used (" << why
                        << ") -> IMU used (frontend pose guess)";
            } else {
              LOG(INFO) << "[GaRLILEO][Frontend] GaRLILEO pose prediction not used (" << why
                        << ") -> IMU not used (default frontend pose guess)";
            }
            last_log = now;
          }
        };

        auto logApplied = [&](double dt) {
          if (now - last_log > 1.0) {
            LOG(INFO) << "[GaRLILEO][Frontend] GaRLILEO pose prediction applied: "
                      << "dt=" << dt << "s"
                      << " use_rot=" << parameters_.garlileo.frontend_use_rotation
                      << " use_trans=" << parameters_.garlileo.frontend_use_translation;
            last_log = now;
          }
        };

        if (!parameters_.garlileo.frontend_use_rotation
            && !parameters_.garlileo.frontend_use_translation) {
          frontendDeltaHasPrev_ = false;
          logFallbackImu("frontend_use_rotation=false & frontend_use_translation=false");
          return;
        }

        const okvis::Time t_query = frameStamp;
        drainGarlileoHistory(t_query);

        // Find the closest GaRLILEO measurement in history (same association strategy as backend).
        std::optional<GaRLILEOMeasurement> best_meas;
        double best_abs_dt = std::numeric_limits<double>::infinity();
        if (!garlileoHistory_.empty()) {
          for (const auto& m : garlileoHistory_) {
            if (!isValidGarlileoMeasurement(m)) {
              continue;
            }
            const double adt = std::abs((m.timeStamp - t_query).toSec());
            if (adt < best_abs_dt) {
              best_abs_dt = adt;
              best_meas = m;
            }
          }
        } else if (hasLastGarlileo_) {
          GaRLILEOMeasurement m;
          {
            std::lock_guard<std::mutex> lock(garlileoMutex_);
            m = lastGarlileoMeas_;
          }
          if (isValidGarlileoMeasurement(m)) {
            best_abs_dt = std::abs((m.timeStamp - t_query).toSec());
            best_meas = m;
          }
        }

        if (!best_meas) {
          frontendDeltaHasPrev_ = false;
          logFallbackImu("no_garlileo_measurement_yet");
          return;
        }

        const GaRLILEOMeasurement meas = *best_meas;
        const double dt = (t_query - meas.timeStamp).toSec();
        if (std::abs(dt) > parameters_.garlileo.frontend_max_age_sec) {
          frontendDeltaHasPrev_ = false;
          if (now - last_log > 1.0) {
            LOG(INFO) << "[GaRLILEO][Frontend] GaRLILEO pose prediction not used (garlileo_measurement_too_old)"
                      << " dt=" << dt << "s"
                      << " max_age=" << parameters_.garlileo.frontend_max_age_sec << "s"
                      << " -> " << (parameters_.imu.use ? "IMU used" : "IMU not used")
                      << " (frontend pose guess)";
            last_log = now;
          }
          return;
        }

        // Build pose measurement from GaRLILEO (assumed pose of its base/body frame in world).
        const Eigen::Quaterniond q_WB = meas.R_WS.normalized();
        okvis::kinematics::Transformation T_WB(meas.p_W, q_WB);
        okvis::kinematics::Transformation T_WS_meas = T_WB;
        if (parameters_.garlileo.apply_imu_extrinsic) {
          // parameters_.imu.T_BS is S->B, so T_WB * T_BS gives T_WS (S->W).
          T_WS_meas = T_WB * parameters_.imu.T_BS;
        }

        // Always use relative delta on top of previous OKVIS pose.
        // For the first valid GaRLILEO sample, seed delta history and keep IMU prediction.
        if (!frontendDeltaHasPrev_) {
          frontendDeltaPrevMeas_ = meas;
          frontendDeltaPrevOkvisPose_ = T_WS_pred;
          frontendDeltaHasPrev_ = true;
          logFallbackImu("waiting_prev_measurement_for_delta");
          return;
        }

        // Delta mode: compute GaRLILEO delta between previous and current GaRLILEO measurements,
        // then apply to previous OKVIS pose.
        okvis::kinematics::Transformation T_WB_prev(frontendDeltaPrevMeas_.p_W, frontendDeltaPrevMeas_.R_WS.normalized());
        okvis::kinematics::Transformation T_WS_prev = T_WB_prev;
        if (parameters_.garlileo.apply_imu_extrinsic) {
          T_WS_prev = T_WB_prev * parameters_.imu.T_BS;
        }
        const okvis::kinematics::Transformation T_prev_to_curr = T_WS_prev.inverse() * T_WS_meas;

        // base pose = previous OKVIS pose; apply delta
        okvis::kinematics::Transformation T_WS_delta_applied = frontendDeltaPrevOkvisPose_ * T_prev_to_curr;

        // Optionally keep only rotation/translation parts
        const Eigen::Vector3d r_pred = T_WS_pred.r(); // current IMU-propagated pose (fallback)
        const Eigen::Quaterniond q_pred(T_WS_pred.q());
        const Eigen::Vector3d r_final =
            parameters_.garlileo.frontend_use_translation ? T_WS_delta_applied.r() : r_pred;
        const Eigen::Quaterniond q_final =
            parameters_.garlileo.frontend_use_rotation ? T_WS_delta_applied.q() : q_pred;
        T_WS_pred.set(r_final, q_final.normalized());

        // update prev for next frame
        frontendDeltaPrevMeas_ = meas;
        frontendDeltaPrevOkvisPose_ = T_WS_delta_applied;
        frontendDeltaHasPrev_ = true;

        logApplied(dt);
      };

  auto anchorGarlileoFrontendPredictionToOkvis =
      [&](const okvis::Time& frameStamp,
          const okvis::kinematics::Transformation& T_WS_corrected) {
        if (!parameters_.garlileo.frontend_use_garlileo
            || (!parameters_.garlileo.frontend_use_rotation
                && !parameters_.garlileo.frontend_use_translation)) {
          frontendDeltaHasPrev_ = false;
          return;
        }

        const okvis::Time t_query = frameStamp;
        drainGarlileoHistory(t_query);

        std::optional<GaRLILEOMeasurement> best_meas;
        double best_abs_dt = std::numeric_limits<double>::infinity();
        if (!garlileoHistory_.empty()) {
          for (const auto& m : garlileoHistory_) {
            if (!isValidGarlileoMeasurement(m)) {
              continue;
            }
            const double adt = std::abs((m.timeStamp - t_query).toSec());
            if (adt < best_abs_dt) {
              best_abs_dt = adt;
              best_meas = m;
            }
          }
        } else if (hasLastGarlileo_) {
          GaRLILEOMeasurement m;
          {
            std::lock_guard<std::mutex> lock(garlileoMutex_);
            m = lastGarlileoMeas_;
          }
          if (isValidGarlileoMeasurement(m)) {
            best_abs_dt = std::abs((m.timeStamp - t_query).toSec());
            best_meas = m;
          }
        }

        if (!best_meas || best_abs_dt > parameters_.garlileo.frontend_max_age_sec) {
          return;
        }

        frontendDeltaPrevMeas_ = *best_meas;
        frontendDeltaPrevOkvisPose_ = T_WS_corrected;
        frontendDeltaHasPrev_ = true;

        static double last_anchor_log = 0.0;
        const double now = okvis::Time::now().toSec();
        if (now - last_anchor_log > 1.0) {
          LOG(INFO) << "[GaRLILEO][Frontend] Re-anchored GaRLILEO delta prediction to corrected OKVIS pose: "
                    << "frame_time=" << t_query.toSec()
                    << " |dt|=" << best_abs_dt << "s"
                    << " max_age=" << parameters_.garlileo.frontend_max_age_sec << "s";
          last_anchor_log = now;
        }
      };

  // Optional: use GaRLILEO gravity spline to improve BRISK extraction direction.
  auto applyGarlileoFrontendGravity =
      [&](const okvis::Time& frameStamp) {
        if (!parameters_.garlileo.frontend_use_garlileo_gravity) {
          frontend_.clearExternalGravity();
          return;
        }
        const okvis::Time t_query = frameStamp;
        drainGarlileoHistory(t_query);
        drainGarlileoGravityHistory(t_query);

        std::optional<GaRLILEOMeasurement> poseMeas;
        double dt_pose = std::numeric_limits<double>::infinity();
        for (const auto& m : garlileoHistory_) {
          if (!isValidGarlileoMeasurement(m)) {
            continue;
          }
          const double adt = std::abs((m.timeStamp - t_query).toSec());
          if (adt < dt_pose) {
            dt_pose = adt;
            poseMeas = m;
          }
        }

        std::optional<GaRLILEOGravityMeasurement> gravMeas;
        double dt_grav = std::numeric_limits<double>::infinity();
        for (const auto& m : garlileoGravityHistory_) {
          if (!isValidGarlileoGravityMeasurement(m)) {
            continue;
          }
          const double adt = std::abs((m.timeStamp - t_query).toSec());
          if (adt < dt_grav) {
            dt_grav = adt;
            gravMeas = m;
          }
        }

        if (!poseMeas || !gravMeas) {
          frontend_.clearExternalGravity();
          return;
        }

        if (dt_pose > parameters_.garlileo.frontend_max_age_sec
            || dt_grav > parameters_.garlileo.frontend_max_age_sec) {
          frontend_.clearExternalGravity();
          return;
        }

        const Eigen::Vector3d g_W = gravMeas->g_W;
        if (g_W.squaredNorm() < 1e-12) {
          frontend_.clearExternalGravity();
          return;
        }

        // GaRLILEO provides R_WB (world-from-base/body). Convert gravity into OKVIS IMU frame S.
        const Eigen::Quaterniond q_WB = poseMeas->R_WS.normalized();
        const Eigen::Vector3d g_B = q_WB.inverse() * g_W;
        Eigen::Vector3d g_S = g_B;
        if (parameters_.garlileo.apply_imu_extrinsic) {
          // imu.T_BS is S->B, so B->S rotation is transpose.
          g_S = parameters_.imu.T_BS.C().transpose() * g_B;
        }
        if (g_S.squaredNorm() < 1e-12) {
          frontend_.clearExternalGravity();
          return;
        }

        frontend_.setExternalGravityInSensorFrame(g_S);

        static double last_log = 0.0;
        const double now = okvis::Time::now().toSec();
        if (now - last_log > 1.0) {
          LOG(INFO) << "[GaRLILEO][Frontend] gravity applied (BRISK): "
                    << "dt_pose=" << dt_pose << "s dt_grav=" << dt_grav << "s"
                    << " g_S=[" << g_S.normalized().transpose() << "]";
          last_log = now;
        }
      };
 
  // Always print the runtime config once (even if GaRLILEO is disabled), so it's obvious whether
  // the running binary actually has the feature and what it is set to.
  static bool garlileo_frontend_cfg_logged = false;
  if (!garlileo_frontend_cfg_logged) {
    LOG(INFO) << "[GaRLILEO][Frontend] runtime (ThreadedSlam): frontend_use_garlileo="
              << std::boolalpha << parameters_.garlileo.frontend_use_garlileo
              << " frontend_use_rotation=" << parameters_.garlileo.frontend_use_rotation
              << " frontend_use_translation=" << parameters_.garlileo.frontend_use_translation
              << " frontend_use_garlileo_gravity=" << parameters_.garlileo.frontend_use_garlileo_gravity
              << " spline_state_topic=/garlileo/spline_state"
              << " frontend_max_age_sec=" << parameters_.garlileo.frontend_max_age_sec
              << " use_rotation=" << parameters_.garlileo.use_rotation
              << " use_translation=" << parameters_.garlileo.use_translation
              << " use_velocity=" << parameters_.garlileo.use_velocity
              << " keyframe_stationary_speed_threshold=" << parameters_.garlileo.keyframe_stationary_speed_threshold
              << " place_recognition_use_radar=" << parameters_.garlileo.place_recognition_use_radar
              << " place_recognition_radar0_enable=" << parameters_.garlileo.place_recognition_radar0_enable
              << " place_recognition_radar1_enable=" << parameters_.garlileo.place_recognition_radar1_enable
              << " place_recognition_radar_max_time_diff=" << parameters_.garlileo.place_recognition_radar_max_time_diff
              << " place_recognition_radar_scans_per_radar="
              << parameters_.garlileo.place_recognition_radar_scans_per_radar
              << " hasLastGarlileo_=" << hasLastGarlileo_;
    garlileo_frontend_cfg_logged = true;
  }
 
   bool ranDetection = false;
   if(firstFrame_) {
     // in the very beginning, we need to wait for the IMU
     if(parameters_.imu.use && !imuMeasurementsReceived_.getCopyOfFront(&imuMeasurement)) {
       return false;
     }
 
     // now get the frames (synchronised timestamp)
     if(!getNextFrame(multiFrame)) {
       return false;
     }
 
     if(parameters_.imu.use) {
       if(multiFrame->timestamp()-Duration(imuTemporalOverlap) <= imuMeasurement.timeStamp) {
         // that's bad, we have frames without IMU measurements. Discard too old frames
         LOG(WARNING) << "startup: dropping frame because IMU measurements are newer, t="
                      << imuMeasurement.timeStamp;
         std::vector<okvis::CameraMeasurement> frames;
         cameraMeasurementsReceived_.PopBlocking(&frames);
         return false;
       }
 
       // also make sure we have enough IMU measuerements
       if(!imuMeasurementsReceived_.getCopyOfBack(&imuMeasurement)) {
         return false;
       }
       if(imuMeasurement.timeStamp < multiFrame->timestamp() + Duration(imuTemporalOverlap)) {
         return false; // wait for more IMU measurements
       }
 
       // now get all relevant IMU measurements we have received thus far
       do {
         if(imuMeasurementsReceived_.PopNonBlocking(&imuMeasurement))
         {
           imuMeasurementDeque_.push_back(imuMeasurement);
         } else {
           return false;
         }
       } while(imuMeasurementDeque_.back().timeStamp
               < multiFrame->timestamp() + Duration(imuTemporalOverlap));
     }
 
     // Drop LiDAR MEasurements before first frame ToDo: is this necessary?
     while(!lidarMeasurementsReceived_.Empty()
       && lidarMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp()){
       lidarMeasurementsReceived_.PopBlocking(&lidarMeasurement);
     }
 
     // Drop GPS Measurements that are older than the first frame (= first state)
     while(!gpsMeasurementsReceived_.Empty() && gpsMeasurementsReceived_.queue_.front().timeStamp < multiFrame ->timestamp()){
         gpsMeasurementsReceived_.PopBlocking(&gpsMeasurement);
     } // nothing else to do here for GPS
 
     firstFrame_ = false;
   } else {
     // wait for next frame
     if(!getNextFrame(multiFrame)) {
       if(optimisationThread_.joinable()) {
         // in the very beginning, we can't join because it was not started
         optimisationThread_.join();
       }
 
       return false;
     }
     // now get all relevant IMU measurements we have received thus far
     if(parameters_.imu.use) {
       while(!shutdown_ && imuMeasurementDeque_.back().timeStamp <
             multiFrame->timestamp() + Duration(imuTemporalOverlap))
       {
         if(imuMeasurementsReceived_.PopNonBlocking(&imuMeasurement))
         {
           imuMeasurementDeque_.push_back(imuMeasurement);
         } else {
           return false;
         }
       }
     }
   }
   if (!lastOptimisedState_.id.isInitialised()) {
     // initial state
     if (parameters_.imu.use) {
       bool success = ceres::ImuError::initPose(imuMeasurementDeque_, T_WS);
       OKVIS_ASSERT_TRUE_DBG(Exception,
                             success,
                             "pose could not be initialized from imu measurements.")
       (void) (success); // avoid warning on unused variable
     } else {
       // otherwise we assume the camera is vertical & upright
       kinematics::Transformation T_WC;
       T_WC.set(T_WC.r(), T_WC.q() * Eigen::Quaterniond(-sqrt(2), sqrt(2), 0, 0));
       T_WS = T_WC * parameters_.nCameraSystem.T_SC(0)->inverse();
     }

    // Optionally wait for a camera-time GaRLILEO measurement before using it in the frontend.
    if (parameters_.garlileo.frontend_use_garlileo) {
      const okvis::Time t_query = multiFrame->timestamp();
      (void)waitForGarlileoMeasurement(t_query, parameters_.garlileo.frontend_max_age_sec,
                                       "frontend", 0.0, 5);
    }

    // Optionally override/augment the initial pose guess for the frontend using GaRLILEO.
    applyGarlileoFrontendPrediction(multiFrame->timestamp(), T_WS);
    applyGarlileoFrontendGravity(multiFrame->timestamp());
 
     // detection -- needed to check if we can start up
     TimerSwitchable detectTimer("1 DetectAndDescribe");
     // With BO disabled, stereo cameras can run independently. Keep the worker
     // alive across frames so its image utility caches are retained.
     if(parameters_.frontend.parallelise_detection && numCameras == 2
        && !parameters_.frontend.agcwd_clahe_bo_enable
        && !multiFrame->image(0).empty() && !multiFrame->image(1).empty()) {
       if (!stereoDetectionWorker_) {
         stereoDetectionWorker_ = std::make_unique<StereoDetectionWorker>();
       }
       const kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
       const kinematics::Transformation T_WC1 = T_WS * (*parameters_.nCameraSystem.T_SC(1));
       stereoDetectionWorker_->run(
           [this, multiFrame, &T_WC0] { frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr); },
           [this, multiFrame, &T_WC1] { frontend_.detectAndDescribe(1, multiFrame, T_WC1, nullptr); });
     } else if(parameters_.frontend.parallelise_detection && numCameras > 2) {
      // Run cam0 first so trigger-based BO parameter updates are decided
      // before other camera detection threads start.
      kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
      if(!multiFrame->image(0).empty()) {
        frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr);
      }
       std::vector<std::shared_ptr<std::thread>> detectionThreads;
       for(size_t im = 1; im < numCameras; ++im) {
         kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
         if(!multiFrame->image(im).empty()) {
           detectionThreads.emplace_back(
               new std::thread(&Frontend::detectAndDescribe, &frontend_,
                               im,  multiFrame, T_WC, nullptr));
         }
       }
       for(size_t i = 0; i < detectionThreads.size(); ++i) {
         detectionThreads[i]->join();
       }
     } else {
       for(size_t im = 0; im < numCameras; ++im) {
         kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
         if(!multiFrame->image(im).empty()) {
           frontend_.detectAndDescribe(im,  multiFrame, T_WC, nullptr);
         }
       }
     }
     if(!frontend_.isInitialized() && multiFrame->numKeypoints() < 15) {
       LOG(WARNING) << "Not enough keypoints (" << multiFrame->numKeypoints()
                    << ") -- cannot initialise yet.";
       std::vector<okvis::CameraMeasurement> frames;
       cameraMeasurementsReceived_.PopBlocking(&frames);
       return false;
     }
     ranDetection = true;
     detectTimer.stop();
    pushAgcwdImagesForRviz();
 
   } else {
     // propagate to have a sensible pose estimate (as needed for detection)
     if (parameters_.imu.use) {
       // NOTE: we are not allowed to access the Estimator object here, as optimisation might run.
       T_WS = lastOptimisedState_.T_WS;
       kinematics::Transformation T_WS_prev = T_WS;
       speedAndBias.head<3>() = lastOptimisedState_.v_W;
       speedAndBias.segment<3>(3) = lastOptimisedState_.b_g;
       speedAndBias.tail<3>() = lastOptimisedState_.b_a;
       ceres::ImuError::propagation(imuMeasurementDeque_,
                                            parameters_.imu,
                                            T_WS,
                                            speedAndBias,
                                            lastOptimisedState_.timestamp,
                                            multiFrame->timestamp());
     } else {
       T_WS = lastOptimisedState_.T_WS;
       if (preLastOptimisedState_.id.isInitialised()) {
         const double r = (multiFrame->timestamp() - lastOptimisedState_.timestamp).toSec()
         / (lastOptimisedState_.timestamp - preLastOptimisedState_.timestamp).toSec();
         kinematics::Transformation T_WS_m1 = preLastOptimisedState_.T_WS;
         Eigen::Vector3d dr = r * (T_WS.r() - T_WS_m1.r());
         Eigen::AngleAxisd daa(T_WS.q() * T_WS_m1.q().inverse());
         daa.angle() *= r;
         T_WS.set(T_WS.r() + dr, T_WS.q() * Eigen::Quaterniond(daa));
       }
     }

    // Optionally wait for a camera-time GaRLILEO measurement before using it in the frontend.
    if (parameters_.garlileo.frontend_use_garlileo) {
      const okvis::Time t_query = multiFrame->timestamp();
      (void)waitForGarlileoMeasurement(t_query, parameters_.garlileo.frontend_max_age_sec,
                                       "frontend", 0.0, 5);
    }

    // Optionally override/augment the predicted pose guess for the frontend using GaRLILEO.
    applyGarlileoFrontendPrediction(multiFrame->timestamp(), T_WS);
    applyGarlileoFrontendGravity(multiFrame->timestamp());
     // now also get all relevant GPS measurements received thus far
     while(!shutdown_ && !gpsMeasurementsReceived_.Empty() && gpsMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp())
     {
         if(gpsMeasurementsReceived_.PopBlocking(&gpsMeasurement))
         {
           gpsMeasurementDeque_.push_back(gpsMeasurement);
         }
     }
 
     // now also get all relevant LiDAR measurements received thus far
     while(!shutdown_ && !lidarMeasurementsReceived_.Empty() && lidarMeasurementsReceived_.queue_.front().timeStamp < multiFrame->timestamp())
     {
       if(lidarMeasurementsReceived_.PopBlocking(&lidarMeasurement))
       {
         lidarMeasurementDeque_.push_back(lidarMeasurement);
       }
     }
 
     // Get depth measurements: in stereo network, depth and multiFrame are perfectly synced.
     while(!shutdown_ && !depthMeasurementsReceived_.Empty() && depthMeasurementsReceived_.queue_.front().timeStamp <= multiFrame->timestamp())
     {
       if (depthMeasurementsReceived_.PopBlocking(&depthMeasurement))
       {
         depthMeasurementDeque_.push_back(depthMeasurement);
       }
     }

   // Keep GaRLILEO pose/gravity histories up-to-date for downstream associations.
   drainGarlileoHistory(multiFrame->timestamp());
   drainGarlileoGravityHistory(multiFrame->timestamp());

    // Drain all available radar scans into history (non-blocking)
    RadarTargetsMeasurement radarMeas;
    while(!shutdown_ && radarTargetsMeasurementsReceived_.PopNonBlocking(&radarMeas)) {
      radarTargetsHistory_.push_back(radarMeas);
    }
    if(radarTargetsHistory_.empty() && hasLastRadarTargets_) {
      std::lock_guard<std::mutex> lock(radarTargetsMutex_);
      radarTargetsHistory_.push_back(lastRadarTargetsMeas_);
    }
   const double radar_keep_sec = garlileoKeepSec();
    while(!radarTargetsHistory_.empty()
         && (multiFrame->timestamp() - radarTargetsHistory_.front().timeStamp).toSec() > radar_keep_sec) {
      radarTargetsHistory_.pop_front();
     }
   }
   // success, frame processed. Need to remove from the queue, as getNextFrame only obtains a copy.
   std::vector<okvis::CameraMeasurement> frames;
   cameraMeasurementsReceived_.PopBlocking(&frames);
  lastCameraQueuePopWallTimeSec_.store(okvis::Time::now().toSec(),
                                       std::memory_order_relaxed);
 
   // detection
   if(!ranDetection) {
     TimerSwitchable detectTimer("1 DetectAndDescribe");
     // With BO disabled, stereo cameras can run independently. Keep the worker
     // alive across frames so its image utility caches are retained.
     if(parameters_.frontend.parallelise_detection && numCameras == 2
        && !parameters_.frontend.agcwd_clahe_bo_enable
        && !multiFrame->image(0).empty() && !multiFrame->image(1).empty()) {
       if (!stereoDetectionWorker_) {
         stereoDetectionWorker_ = std::make_unique<StereoDetectionWorker>();
       }
       const kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
       const kinematics::Transformation T_WC1 = T_WS * (*parameters_.nCameraSystem.T_SC(1));
       stereoDetectionWorker_->run(
           [this, multiFrame, &T_WC0] { frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr); },
           [this, multiFrame, &T_WC1] { frontend_.detectAndDescribe(1, multiFrame, T_WC1, nullptr); });
     } else if(parameters_.frontend.parallelise_detection && numCameras > 2) {
      // Run cam0 first so trigger-based BO parameter updates are decided
      // before other camera detection threads start.
      kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
      if(!multiFrame->image(0).empty()) {
        frontend_.detectAndDescribe(0, multiFrame, T_WC0, nullptr);
      }
       std::vector<std::shared_ptr<std::thread>> detectionThreads;
       for(size_t im = 1; im < numCameras; ++im) {
         kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
         if(!multiFrame->image(im).empty()) {
           detectionThreads.emplace_back(
               new std::thread(&Frontend::detectAndDescribe, &frontend_,
                               im,  multiFrame, T_WC, nullptr));
         }
       }
       for(size_t i = 0; i < detectionThreads.size(); ++i) {
         detectionThreads[i]->join();
       }
     } else {
       for(size_t im = 0; im < numCameras; ++im) {
         kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(im));
         if(!multiFrame->image(im).empty()) {
           frontend_.detectAndDescribe(im,  multiFrame, T_WC, nullptr);
         }
       }
     }
     if(!frontend_.isInitialized() && multiFrame->numKeypoints() < 15) {
       LOG(WARNING) << "Not enough keypoints (" << multiFrame->numKeypoints()
                    << ") -- cannot initialise yet.";
       return true;
     }
     detectTimer.stop();
    pushAgcwdImagesForRviz();
   }
 
   // IMPORTANT: the matcher needs the optimiser to be finished:
   if(optimisationThread_.joinable()) {
     // in the very beginning, we can't join because it was not started
     optimisationThread_.join();
   }
 
   // now store last optimised state for later use
   if (estimator_.numFrames() > 0) {
     const StateId currentId = estimator_.currentStateId();
     lastOptimisedState_.T_WS = estimator_.pose(currentId);
     SpeedAndBias speedAndBias = estimator_.speedAndBias(currentId);
     lastOptimisedState_.v_W = speedAndBias.head<3>();
     lastOptimisedState_.b_g = speedAndBias.segment<3>(3);
     lastOptimisedState_.b_a = speedAndBias.tail<3>();
     lastOptimisedState_.id = currentId;
     lastOptimisedState_.timestamp = estimator_.timestamp(currentId);
     anchorGarlileoFrontendPredictionToOkvis(lastOptimisedState_.timestamp,
                                             lastOptimisedState_.T_WS);
     if (estimator_.numFrames() > 1) {
       const StateId previousId(currentId.value() - 1);
       preLastOptimisedState_.T_WS = estimator_.pose(previousId);
       preLastOptimisedState_.id = previousId;
       preLastOptimisedState_.timestamp = estimator_.timestamp(previousId);
     }
   }
 
   // break here since all local threads joined
   if(shutdown_) {
     return false;
   }
 
   // remove imuMeasurements from deque
   if (estimator_.numFrames() > 0) {
     if(parameters_.imu.use) {
       while (!shutdown_
              && (imuMeasurementDeque_.front().timeStamp
                  < lastOptimisedState_.timestamp - Duration(imuTemporalOverlap))) {
         auto imuMeasurement = imuMeasurementDeque_.front();
         imuMeasurementDeque_.pop_front();
         if (imuMeasurementDeque_.empty()
             || (imuMeasurementDeque_.front().timeStamp
                 > lastOptimisedState_.timestamp - Duration(imuTemporalOverlap))) {
           imuMeasurementDeque_.push_front(imuMeasurement); // re-add
           break; // finished popping
         }
       }
     }
   }
 
   // start the matching
   Time matchingStart = Time::now();
   TimerSwitchable matchTimer("2 Match");
   bool asKeyframe = false;
 
  // (low_features) uses the minimum per-camera SFA-EWG utility (worst camera).
  // Per-camera values are computed in Frontend::detectAndDescribe and cached.
  const double image_utility_thr = parameters_.garlileo.image_utility_threshold;
  const bool ewg_gate_enabled = (image_utility_thr > 0.0);
  bool low_features = false;
  double image_utility_sum = std::numeric_limits<double>::quiet_NaN();
  double image_utility_min = std::numeric_limits<double>::quiet_NaN();
  int    image_utility_cams = 0;
  if (ewg_gate_enabled) {
    std::vector<double> image_utils;
    if (frontend_.getImageUtility(multiFrame->id(), image_utils)) {
      double sum_u = 0.0;
      double min_u = std::numeric_limits<double>::infinity();
      int n_fin = 0;
      for (double u : image_utils) {
        if (std::isfinite(u)) {
          sum_u += u;
          min_u = std::min(min_u, u);
          ++n_fin;
        }
      }
      if (n_fin > 0) {
        image_utility_sum = sum_u;
        image_utility_min = min_u;
        image_utility_cams = n_fin;
        low_features = (image_utility_min < image_utility_thr);
      }
    }
  }

  const double image_utility_semi_thr = parameters_.garlileo.image_utility_semi_threshold;
  const double w_util_fallback_blend = utilityFallbackBlendFromEwgMean(
      image_utility_min, image_utility_thr, image_utility_semi_thr, ewg_gate_enabled,
      parameters_.garlileo.image_utility_blend_sigmoid_steepness,
      parameters_.garlileo.image_utility_blend_hysteresis);
  const double image_utility_mean_log =
      (image_utility_cams > 0 && std::isfinite(image_utility_sum))
          ? (image_utility_sum / double(image_utility_cams))
          : std::numeric_limits<double>::quiet_NaN();

  double w_vision_fallback_blend = w_util_fallback_blend;
  if (frontend_.isInitialized()) {
    const StateId fid(multiFrame->id());
    if (estimator_.multiFrame(fid)) {
      const double tq_vis = estimator_.trackingQuality(fid);
      const double lost_thr_vis =
          std::max(0.0, parameters_.garlileo.tracking_lost_quality_threshold);
      if (std::isfinite(tq_vis) && tq_vis < lost_thr_vis) {
        w_vision_fallback_blend = std::max(w_vision_fallback_blend, 1.0);
      }
    }
  }

  auto findClosestGarlileo =
      [&](const okvis::Time& t,
          const std::optional<uint64_t>& excluded_stamp_nsec = std::nullopt)
      ->std::pair<std::optional<GaRLILEOMeasurement>, double> {
        drainGarlileoHistory(t);
        auto isUsableForQuery = [&](const GaRLILEOMeasurement& m) -> bool {
          if (!isValidGarlileoMeasurement(m)) {
            return false;
          }
          const double abs_dt = std::abs((t - m.timeStamp).toSec());
          return std::isfinite(abs_dt) && abs_dt <= garlileoKeepSec();
        };
        const uint64_t query_stamp_nsec = t.toNSec();
        if (garlileoHistory_.empty()) {
          if (hasLastGarlileo_) {
            std::lock_guard<std::mutex> lock(garlileoMutex_);
            if (isUsableForQuery(lastGarlileoMeas_)) {
              const uint64_t ts = lastGarlileoMeas_.timeStamp.toNSec();
              if (!excluded_stamp_nsec || ts != *excluded_stamp_nsec) {
                if (ts == query_stamp_nsec) {
                  return {lastGarlileoMeas_, 0.0};
                }
                const double dt = std::abs((lastGarlileoMeas_.timeStamp - t).toSec());
                return {lastGarlileoMeas_, dt};
              }
            }
          }
          return {std::nullopt, std::numeric_limits<double>::infinity()};
        }
        double best = std::numeric_limits<double>::infinity();
        std::optional<GaRLILEOMeasurement> bestMeas;
        double bestPast = std::numeric_limits<double>::infinity();
        std::optional<GaRLILEOMeasurement> bestPastMeas;
        for (const auto& m : garlileoHistory_) {
          if (!isUsableForQuery(m)) {
            continue;
          }
          if (excluded_stamp_nsec && m.timeStamp.toNSec() == *excluded_stamp_nsec) {
            continue;
          }
          if (m.timeStamp.toNSec() == query_stamp_nsec) {
            return {m, 0.0};
          }
          const double signed_dt = (t - m.timeStamp).toSec();
          const double dt = std::abs(signed_dt);
          if (signed_dt >= 0.0 && dt < bestPast) {
            bestPast = dt;
            bestPastMeas = m;
          }
          if (dt < best) {
            best = dt;
            bestMeas = m;
          }
        }
        if (hasLastGarlileo_) {
          std::lock_guard<std::mutex> lock(garlileoMutex_);
          if (isUsableForQuery(lastGarlileoMeas_)) {
            const uint64_t ts = lastGarlileoMeas_.timeStamp.toNSec();
            if (!excluded_stamp_nsec || ts != *excluded_stamp_nsec) {
              if (ts == query_stamp_nsec) {
                return {lastGarlileoMeas_, 0.0};
              }
              const double signed_dt = (t - lastGarlileoMeas_.timeStamp).toSec();
              const double dt = std::abs(signed_dt);
              if (signed_dt >= 0.0 && (!bestPastMeas || dt < bestPast)) {
                bestPast = dt;
                bestPastMeas = lastGarlileoMeas_;
              }
              if (!bestMeas || dt < best) {
                best = dt;
                bestMeas = lastGarlileoMeas_;
              }
            }
          }
        }
        if (bestPastMeas) {
          return {bestPastMeas, bestPast};
        }
        return {bestMeas, best};
      };
  auto enforceDistinctPair =
      [&](const okvis::Time& t_prev,
          const okvis::Time& t_curr,
          std::optional<GaRLILEOMeasurement>& meas_prev,
          double& min_dt_prev,
          std::optional<GaRLILEOMeasurement>& meas_curr,
          double& min_dt_curr) {
        if (!meas_prev || !meas_curr) {
          return;
        }
        const uint64_t prev_ns = meas_prev->timeStamp.toNSec();
        const uint64_t curr_ns = meas_curr->timeStamp.toNSec();
        if (prev_ns != curr_ns) {
          return;
        }

        auto [alt_prev, alt_prev_dt] = findClosestGarlileo(t_prev, curr_ns);
        auto [alt_curr, alt_curr_dt] = findClosestGarlileo(t_curr, prev_ns);

        if (alt_prev && (!alt_curr || alt_prev_dt <= alt_curr_dt)) {
          meas_prev = alt_prev;
          min_dt_prev = alt_prev_dt;
          return;
        }
        if (alt_curr) {
          meas_curr = alt_curr;
          min_dt_curr = alt_curr_dt;
        }
      };
  // The new frame has no matched landmarks before data association. Use the
  // latest completed frame for causal nominal weighting; full fallback does not
  // need this image-coverage calculation when its own stds are in use.
  double garlileo_tracking_quality_before_matching = 0.0;
  uint64_t garlileo_tracking_quality_frame_id = 0;
  const bool garlileo_needs_nominal_quality =
      !parameters_.garlileo.pose_fallback_enable
      || w_vision_fallback_blend < 1.0
      || (parameters_.garlileo.use_velocity
          && !(parameters_.garlileo.pose_fallback_velocity_std > 0.0));
  if (parameters_.garlileo.std_scale_use_tracking_quality
      && garlileo_needs_nominal_quality && frontend_.isInitialized()
      && estimator_.numFrames() > 0) {
    const StateId matched_id = estimator_.currentStateId();
    if (estimator_.multiFrame(matched_id)) {
      garlileo_tracking_quality_before_matching = estimator_.trackingQuality(matched_id);
      garlileo_tracking_quality_frame_id = matched_id.value();
    }
  }
   const uint64_t image_utility_frame_id = multiFrame->id();
   if (!estimator_.addStates(multiFrame, imuMeasurementDeque_, asKeyframe)) {
     LOG(ERROR)<< "Failed to add state! will drop multiframe.";
     matchTimer.stop();
     return true;
   }
   else{
    // Detection used the provisional frame id; loop closure queries the graph id.
    frontend_.reassignImageUtilityFrameId(image_utility_frame_id, multiFrame->id());
    {
      std::lock_guard<std::mutex> lock(ewgUtilityMinRecordsMutex_);
      ewgUtilityMinRecords_.push_back(
          EwgUtilityMinRecord{multiFrame->id(), multiFrame->timestamp().toSec(),
                              image_utility_min, image_utility_cams});
    }
    // --- GaRLILEO backend constraints (position/rotation between + velocity) ---
    // NOTE: Must be added before frontend matching/initialisation, since the frontend calls
    // optimiseRealtimeGraph() internally and should see these constraints.
    if (estimator_.numFrames() >= 2) {
      const StateId currentId = estimator_.currentStateId();
      const StateId prevId(currentId.value() - 1);
      const double tracking_quality_used = garlileo_tracking_quality_before_matching;
      const double garlileo_nominal_std_scale = garlileoNominalStdScaleWithConfig(
          parameters_.garlileo.std_scale_use_tracking_quality, tracking_quality_used,
          parameters_.garlileo.std_scale_min_tracking_quality);
      const okvis::Time t_curr = estimator_.timestamp(currentId);
      const okvis::Time t_prev = estimator_.timestamp(prevId);
      // Force-between (visual fallback): if vision is currently weak (low_features) and the
      // user has opted in, block briefly so that the GaRLILEO measurements at t_prev and t_curr
      // (which GaRLILEO is expected to publish on every camera-synchronous timestamp once
      // initialised) are guaranteed to be in our history before we try to add the between
      // factor. Once they are present, the standard ns-exact lookup yields |dt|=0 and the
      // between constraint is added unconditionally.
      const bool force_between_now =
          parameters_.garlileo.visual_fallback_garlileo_force_between
          && frontend_.isInitialized()
          && (w_util_fallback_blend > 0.0);
      const double wait_sec = force_between_now
          ? std::max(0.0, parameters_.garlileo.visual_fallback_garlileo_wait_max_sec)
          : 0.0;
      const int wait_poll_ms = std::max(1,
          parameters_.garlileo.visual_fallback_garlileo_wait_poll_period_ms);
      (void)waitForGarlileoMeasurement(t_prev, parameters_.garlileo.max_time_diff,
                                       "backend_prev", wait_sec, wait_poll_ms);
      (void)waitForGarlileoMeasurement(t_curr, parameters_.garlileo.max_time_diff,
                                       "backend_curr", wait_sec, wait_poll_ms);

      auto [meas_curr, min_dt_curr] = findClosestGarlileo(t_curr);
      auto [meas_prev, min_dt_prev] = findClosestGarlileo(t_prev);
      enforceDistinctPair(t_prev, t_curr, meas_prev, min_dt_prev, meas_curr, min_dt_curr);

      if (meas_curr && meas_prev) {
        const double max_dt = parameters_.garlileo.max_time_diff;
        // signed dt convention:
        //   dt_signed = (t_query - t_meas)
        //   dt_signed > 0  => GaRLILEO measurement is earlier/older than the queried OKVIS time (GaRLILEO lags).
        //   dt_signed < 0  => GaRLILEO measurement is later/newer than the queried OKVIS time (GaRLILEO leads).
        const double dt_curr_signed = (t_curr - meas_curr->timeStamp).toSec();
        const double dt_prev_signed = (t_prev - meas_prev->timeStamp).toSec();
        const double dt_curr_abs = std::abs(dt_curr_signed);
        const double dt_prev_abs = std::abs(dt_prev_signed);
        // Build measured poses. GaRLILEO provides pose of its base frame in world (T_WB).
        // If requested, map to OKVIS IMU/sensor frame using imu.T_BS (S->B).
        const Eigen::Quaterniond q_WB0 = meas_prev->R_WS.normalized();
        const Eigen::Quaterniond q_WB1 = meas_curr->R_WS.normalized();
        okvis::kinematics::Transformation T_WB0(meas_prev->p_W, q_WB0);
        okvis::kinematics::Transformation T_WB1(meas_curr->p_W, q_WB1);

        okvis::kinematics::Transformation T_WS0 = T_WB0;
        okvis::kinematics::Transformation T_WS1 = T_WB1;
        if (parameters_.garlileo.apply_imu_extrinsic) {
          T_WS0 = T_WB0 * parameters_.imu.T_BS;
          T_WS1 = T_WB1 * parameters_.imu.T_BS;
        }
        const okvis::kinematics::Transformation T_S0S1 = T_WS0.inverse() * T_WS1;

        const bool use_rot_between = parameters_.garlileo.use_rotation;
        const bool use_trans_between = parameters_.garlileo.use_translation;
        const bool can_use_pair = (dt_curr_abs <= max_dt) && (dt_prev_abs <= max_dt);
        const bool over_max_dt = !can_use_pair;
        ++g_betweenPairs;

        bool added_between = false;
        if (can_use_pair && (use_rot_between || use_trans_between)) {
          const Eigen::Matrix<double, 6, 6> information = makeGarlileoBetweenInformation(
              parameters_.garlileo, use_rot_between, use_trans_between,
              w_vision_fallback_blend, garlileo_nominal_std_scale);
          added_between =
              applyGarlileoBetweenIfAllowed(prevId, currentId, T_S0S1, information);
          if (added_between) {
            ++g_betweenImmediate;
          }

          // Remember association quality for potential backfill.
          GaRLILEOBetweenAssoc assoc;
          assoc.max_abs_dt = std::max(dt_prev_abs, dt_curr_abs);
          assoc.has_constraint = added_between;
          garlileoBetweenAssocByCurrId_[currentId.value()] = assoc;
        } else {
          // Keep max_time_diff as a hard gate: if association is too far, disable this pair.
          (void)estimator_.removeRelativePoseConstraint(prevId, currentId);
          GaRLILEOBetweenAssoc assoc;
          assoc.max_abs_dt = std::max(dt_prev_abs, dt_curr_abs);
          assoc.has_constraint = false;
          garlileoBetweenAssocByCurrId_[currentId.value()] = assoc;
        }

        if (parameters_.garlileo.use_velocity) {
          // NOTE: velocity constraint is handled separately based on dt_curr only (see below),
          // so we don't add it here to avoid tying it to the prev endpoint association.
        }

        const double trans_nom_log =
            parameters_.garlileo.translation_std * garlileo_nominal_std_scale;
        const double rot_rp_nom_log =
            positiveOrFallback(parameters_.garlileo.rotation_roll_pitch_std_deg, 5.0) *
            garlileo_nominal_std_scale;
        const double rot_yaw_nom_log =
            positiveOrFallback(parameters_.garlileo.rotation_yaw_std_deg, 5.0) *
            garlileo_nominal_std_scale;
        const double wvb_log = w_vision_fallback_blend;
        const bool pf_log = parameters_.garlileo.pose_fallback_enable;
        const double trans_std_log =
            pf_log ? lerpPair(trans_nom_log, parameters_.garlileo.pose_fallback_translation_std,
                              wvb_log)
                   : trans_nom_log;
        const double rot_rp_std_deg_log =
            pf_log ? lerpPair(rot_rp_nom_log,
                              positiveOrFallback(
                                  parameters_.garlileo.pose_fallback_rotation_roll_pitch_std_deg,
                                  2.0),
                              wvb_log)
                   : rot_rp_nom_log;
        const double rot_yaw_std_deg_log =
            pf_log ? lerpPair(rot_yaw_nom_log,
                              positiveOrFallback(
                                  parameters_.garlileo.pose_fallback_rotation_yaw_std_deg,
                                  2.0),
                              wvb_log)
                   : rot_yaw_nom_log;
        const double vel_nom_log =
            parameters_.garlileo.velocity_std * garlileo_nominal_std_scale;
        const double vel_fb_log =
            parameters_.garlileo.pose_fallback_velocity_std > 0.0
                ? parameters_.garlileo.pose_fallback_velocity_std
                : vel_nom_log;
        const double vel_std_log_nominal =
            pf_log ? lerpPair(vel_nom_log, vel_fb_log, wvb_log) : vel_nom_log;

        LOG(INFO) << "[GaRLILEO] Applied factors: prev=" << prevId.value()
                  << " curr=" << currentId.value()
                  << " dt_prev_s=" << dt_prev_signed << "s"
                  << " dt_curr_s=" << dt_curr_signed << "s"
                  << " |dt_prev|=" << dt_prev_abs << "s"
                  << " |dt_curr|=" << dt_curr_abs << "s"
                  << " used_between=" << added_between
                  << " over_max_dt=" << over_max_dt
                  << " max_dt=" << max_dt
                  << " use_rot_between=" << parameters_.garlileo.use_rotation
                  << " use_trans_between=" << parameters_.garlileo.use_translation
                  << " use_vel=" << parameters_.garlileo.use_velocity
                  << " tracking_quality=" << tracking_quality_used
                  << " tracking_quality_frame=" << garlileo_tracking_quality_frame_id
                  << " nominal_std_scale=" << garlileo_nominal_std_scale
                  << " w_util_blend=" << w_util_fallback_blend
                  << " w_vision_blend=" << w_vision_fallback_blend
                  << " trans_std=" << trans_std_log
                  << " rot_rp_std_deg=" << rot_rp_std_deg_log
                  << " rot_yaw_std_deg=" << rot_yaw_std_deg_log
                  << " vel_std_blended=" << vel_std_log_nominal
                  << " low_features=" << low_features
                  << " ewg_gate=" << ewg_gate_enabled
                  << " ewg_utility_min=" << image_utility_min
                  << " ewg_utility_mean=" << image_utility_mean_log
                  << " ewg_utility_sum=" << image_utility_sum
                  << " ewg_utility_thr=" << image_utility_thr
                  << " ewg_utility_semi_thr=" << image_utility_semi_thr
                  << " ewg_utility_cams=" << image_utility_cams
                  << " (dt_s = (t_query - t_meas); + => meas older/lag; - => meas newer/lead)";
      } else {
        GaRLILEOBetweenAssoc assoc;
        assoc.max_abs_dt = std::numeric_limits<double>::infinity();
        assoc.has_constraint = false;
        garlileoBetweenAssocByCurrId_[currentId.value()] = assoc;

        LOG(INFO) << "[GaRLILEO] Skip factors (missing GaRLILEO msg): "
                  << "prev_found=" << static_cast<bool>(meas_prev)
                  << " curr_found=" << static_cast<bool>(meas_curr)
                  << " history_size=" << garlileoHistory_.size()
                  << " min_dt_prev=" << min_dt_prev << " min_dt_curr=" << min_dt_curr
                  << " t_prev=" << t_prev.toSec() << " t_curr=" << t_curr.toSec();
      }

      // Backfill/update GaRLILEO between constraints for recent pairs still in the sliding window.
      // Motivation: GaRLILEO messages may arrive slightly delayed; by the time we process frame (k+1),
      // the perfectly time-aligned measurement for frame k may have arrived. Backfilling updates
      // (k-1,k) to use that improved association (dt≈0).
      if (parameters_.garlileo.between_backfill_enable) {
        const int max_pairs = std::max(0, parameters_.garlileo.between_backfill_max_pairs);
        if (max_pairs > 0) {
          static double last_log = 0.0;
          const double now_wall = okvis::Time::now().toSec();

          const int64_t newest_curr = static_cast<int64_t>(currentId.value()) - 1; // exclude the newest pair (prev,current)
          const int64_t oldest_curr = std::max<int64_t>(2, static_cast<int64_t>(currentId.value()) - max_pairs);
          size_t updated = 0;
          size_t considered = 0;
          size_t updated_vel = 0;
          size_t removed_vel = 0;

          for (int64_t cid_val = newest_curr; cid_val >= oldest_curr; --cid_val) {
            const StateId cid(static_cast<uint64_t>(cid_val));
            if (!estimator_.multiFrame(cid)) {
              continue;
            }

            const uint64_t cid_key = static_cast<uint64_t>(cid_val);
            const okvis::Time t_c = estimator_.timestamp(cid);
            const auto [m_c, min_dt_c] = findClosestGarlileo(t_c);
            (void)min_dt_c;

            const double dt_c_s = m_c ? (t_c - m_c->timeStamp).toSec()
                                      : std::numeric_limits<double>::infinity();
            const double dt_c_abs = std::abs(dt_c_s);

            // Both backfill factors consume the same frame's nominal quality.
            // Velocity-prior replacement does not change feature observations;
            // the realtime optimiser was joined before entering this frame.
            // Keep this lazy cache inside ONE cid iteration, never across frames.
            bool have_nominal_scale_cid = false;
            double nominal_scale_cid = 1.0;
            const auto nominalScaleForCid = [&]() {
              if (!have_nominal_scale_cid) {
                nominal_scale_cid = garlileoNominalStdScale(
                    estimator_.trackingQuality(cid),
                    parameters_.garlileo.std_scale_min_tracking_quality);
                have_nominal_scale_cid = true;
              }
              return nominal_scale_cid;
            };

            // --- Velocity/speed backfill (unary) ---
            if (parameters_.garlileo.use_velocity) {
              const bool has_meas = static_cast<bool>(m_c) && std::isfinite(dt_c_abs);
              const bool can_use =
                  has_meas && (dt_c_abs <= parameters_.garlileo.max_time_diff);
              auto itv = garlileoVelocityAssocById_.find(cid_key);
              const bool had = (itv != garlileoVelocityAssocById_.end()) && itv->second.has_constraint;

              if (can_use) {
                bool should_update_v = false;
                if (itv == garlileoVelocityAssocById_.end()) {
                  should_update_v = true;
                } else if (!itv->second.has_constraint) {
                  should_update_v = true;
                } else if (itv->second.retry_pending
                           && dt_c_abs <= itv->second.abs_dt + 1.0e-9) {
                  should_update_v = true;
                } else if (dt_c_abs + 1.0e-9 < itv->second.abs_dt) {
                  should_update_v = true;
                }
                if (should_update_v) {
                  Eigen::Vector3d v_S = m_c->v_body;
                  if (parameters_.garlileo.apply_imu_extrinsic) {
                    const okvis::kinematics::Transformation T_SB = parameters_.imu.T_BS.inverse();
                    v_S = T_SB.C() * m_c->v_body;
                  }

                  // If this older frame had poor vision, preserve the blended fallback strength.
                  double w_vel_bf = 0.0;
                  auto it_qv = garlileoFrameQualityById_.find(cid_key);
                  if (it_qv != garlileoFrameQualityById_.end()) {
                    w_vel_bf = it_qv->second.vision_fallback_blend;
                    if (it_qv->second.tracking_lost) {
                      w_vel_bf = std::max(w_vel_bf, 1.0);
                    }
                  }
                  // Full fallback with a dedicated velocity std does not consume
                  // nominal quality. trackingQuality rasterises matched features.
                  const bool need_nominal_velocity_quality_bf =
                      parameters_.garlileo.std_scale_use_tracking_quality
                      && (!parameters_.garlileo.pose_fallback_enable
                          || w_vel_bf < 1.0
                          || !(parameters_.garlileo.pose_fallback_velocity_std > 0.0));
                  const double vel_scale_cid = need_nominal_velocity_quality_bf
                      ? nominalScaleForCid()
                      : 1.0;
                  const double vel_nom_bf = parameters_.garlileo.velocity_std * vel_scale_cid;
                  const double vel_fb_bf =
                      parameters_.garlileo.pose_fallback_velocity_std > 0.0
                          ? parameters_.garlileo.pose_fallback_velocity_std
                          : vel_nom_bf;
                  const double vel_std_used =
                      parameters_.garlileo.pose_fallback_enable
                          ? lerpPair(vel_nom_bf, vel_fb_bf, w_vel_bf)
                          : vel_nom_bf;
                  const double vel_std_eff = std::max(1.0e-12, vel_std_used);
                  const bool applied_velocity = updateGarlileoVelocityPrior(
                      cid, v_S, vel_std_eff, dt_c_abs, true);
                  if (applied_velocity) ++updated_vel;
                  if (applied_velocity && now_wall - last_log > 1.0) {
                    LOG(INFO) << "[GaRLILEO][Backfill] updated velocity: k=" << cid_key
                              << " dt_s=" << dt_c_s
                              << " |dt|=" << dt_c_abs
                              << " over_max_dt=" << (dt_c_abs > parameters_.garlileo.max_time_diff)
                              << " vel_std=" << vel_std_eff
                              << " w_vel_blend=" << w_vel_bf;
                    last_log = now_wall;
                  }
                }
              } else {
                const bool removed = updateGarlileoVelocityPrior(
                    cid, Eigen::Vector3d::Zero(), 1.0, dt_c_abs, false);
                if (had && removed) {
                  ++removed_vel;
                  if (now_wall - last_log > 1.0) {
                    LOG(INFO) << "[GaRLILEO][Backfill] removed velocity (missing_or_over_dt): k=" << cid_key
                              << " dt_s=" << dt_c_s
                              << " |dt|=" << dt_c_abs
                              << " max=" << parameters_.garlileo.max_time_diff;
                    last_log = now_wall;
                  }
                }
              }
            }

            // --- Between backfill (binary) ---
            const StateId pid(static_cast<uint64_t>(cid_val - 1));
            if (!estimator_.multiFrame(pid)) {
              continue;
            }
            const okvis::Time t_p = estimator_.timestamp(pid);
            auto m_c_pair = m_c;
            double min_dt_c_pair = min_dt_c;
            auto [m_p, min_dt_p] = findClosestGarlileo(t_p);
            if (!m_c_pair || !m_p) {
              continue;
            }
            enforceDistinctPair(t_p, t_c, m_p, min_dt_p, m_c_pair, min_dt_c_pair);
            if (!m_c_pair || !m_p) {
              continue;
            }
            const double dt_c_pair_s = (t_c - m_c_pair->timeStamp).toSec();
            const double dt_c_pair_abs = std::abs(dt_c_pair_s);
            const double dt_p_s = (t_p - m_p->timeStamp).toSec();
            const double dt_p_abs = std::abs(dt_p_s);
            const double pair_max_abs_dt = std::max(dt_c_pair_abs, dt_p_abs);
            if (!std::isfinite(pair_max_abs_dt)) {
              continue;
            }
            const bool can_use_pair = pair_max_abs_dt <= parameters_.garlileo.max_time_diff;
            auto it_assoc = garlileoBetweenAssocByCurrId_.find(cid_key);
            const bool had_between =
                (it_assoc != garlileoBetweenAssocByCurrId_.end()) && it_assoc->second.has_constraint;
            if (!can_use_pair) {
              if (had_between) {
                (void)estimator_.removeRelativePoseConstraint(pid, cid);
                ++g_betweenRemoved;
              }
              GaRLILEOBetweenAssoc assoc;
              assoc.max_abs_dt = pair_max_abs_dt;
              assoc.has_constraint = false;
              garlileoBetweenAssocByCurrId_[cid_key] = assoc;
              if (had_between && now_wall - last_log > 1.0) {
                LOG(INFO) << "[GaRLILEO][Backfill] removed between (over_dt): k=" << cid_key
                          << " dt_p_s=" << dt_p_s
                          << " dt_c_s=" << dt_c_pair_s
                          << " |dt|max=" << pair_max_abs_dt
                          << " max=" << parameters_.garlileo.max_time_diff;
                last_log = now_wall;
              }
              continue;
            }

            considered++;
            bool should_update = false;
            if (parameters_.garlileo.between_backfill_overwrite_on_quality_improvement) {
              // User guarantees newer backfill values are better -> always overwrite when usable.
              should_update = true;
            } else if (it_assoc == garlileoBetweenAssocByCurrId_.end()) {
              should_update = true;
            } else if (!it_assoc->second.has_constraint) {
              should_update = true;
            } else if (pair_max_abs_dt + 1.0e-9 < it_assoc->second.max_abs_dt) {
              should_update = true;
            }
            if (!should_update) {
              continue;
            }

            double w_blend_bf = 0.0;
            auto it_q = garlileoFrameQualityById_.find(cid_key);
            if (it_q != garlileoFrameQualityById_.end()) {
              w_blend_bf = it_q->second.vision_fallback_blend;
              if (it_q->second.tracking_lost) {
                w_blend_bf = std::max(w_blend_bf, 1.0);
              }
            }

            const Eigen::Quaterniond q_WB0 = m_p->R_WS.normalized();
            const Eigen::Quaterniond q_WB1 = m_c_pair->R_WS.normalized();
            okvis::kinematics::Transformation T_WB0(m_p->p_W, q_WB0);
            okvis::kinematics::Transformation T_WB1(m_c_pair->p_W, q_WB1);

            okvis::kinematics::Transformation T_WS0 = T_WB0;
            okvis::kinematics::Transformation T_WS1 = T_WB1;
            if (parameters_.garlileo.apply_imu_extrinsic) {
              T_WS0 = T_WB0 * parameters_.imu.T_BS;
              T_WS1 = T_WB1 * parameters_.imu.T_BS;
            }
            const okvis::kinematics::Transformation T_S0S1 = T_WS0.inverse() * T_WS1;

            const bool use_rot_between = parameters_.garlileo.use_rotation;
            const bool use_trans_between = parameters_.garlileo.use_translation;
            const bool need_nominal_pose_quality_bf =
                parameters_.garlileo.std_scale_use_tracking_quality
                && (!parameters_.garlileo.pose_fallback_enable || w_blend_bf < 1.0);
            const double nominal_scale_backfill = need_nominal_pose_quality_bf
                ? nominalScaleForCid()
                : 1.0;

            if (use_rot_between || use_trans_between) {
              const Eigen::Matrix<double, 6, 6> information = makeGarlileoBetweenInformation(
                  parameters_.garlileo, use_rot_between, use_trans_between, w_blend_bf,
                  nominal_scale_backfill);

              const bool added_between =
                  applyGarlileoBetweenIfAllowed(pid, cid, T_S0S1, information);

              GaRLILEOBetweenAssoc assoc;
              assoc.max_abs_dt = pair_max_abs_dt;
              assoc.has_constraint = added_between;
              garlileoBetweenAssocByCurrId_[cid_key] = assoc;

              if (added_between) {
                ++updated;
                if (!had_between) {
                  ++g_betweenBackfilled;
                  g_backfillLagFrames +=
                      static_cast<uint64_t>(std::max<int64_t>(0,
                          static_cast<int64_t>(currentId.value()) - cid_key));
                }
              }
              if (now_wall - last_log > 1.0) {
                LOG(INFO) << "[GaRLILEO][Backfill] updated between (k-1,k): k=" << cid_key
                          << " used_between=" << added_between
                          << " dt_p_s=" << dt_p_s << " dt_c_s=" << dt_c_pair_s
                          << " |dt|max=" << pair_max_abs_dt
                          << " w_vision_blend=" << w_blend_bf;
                last_log = now_wall;
              }
            }
          }

          (void)considered;
          (void)updated;
          (void)updated_vel;
          (void)removed_vel;
        }
      }
    }

    // Velocity / speed prior (unary): associate only by current timestamp (t_curr).
    // This is independent of whether the (prev,curr) between constraint could be added.
    if (parameters_.garlileo.use_velocity) {
      const StateId currentId = estimator_.currentStateId();
      const double garlileo_nominal_std_scale = garlileoNominalStdScaleWithConfig(
          parameters_.garlileo.std_scale_use_tracking_quality,
          garlileo_tracking_quality_before_matching,
          parameters_.garlileo.std_scale_min_tracking_quality);
      const okvis::Time t_curr = estimator_.timestamp(currentId);
      const auto [meas_curr, min_dt_curr] = findClosestGarlileo(t_curr);
      (void)min_dt_curr;
      if (meas_curr) {
        const double max_dt = parameters_.garlileo.max_time_diff;
        const double dt_curr_signed = (t_curr - meas_curr->timeStamp).toSec();
        const double dt_curr_abs = std::abs(dt_curr_signed);
        auto itv = garlileoVelocityAssocById_.find(currentId.value());
        const bool can_use = dt_curr_abs <= max_dt;
        if (can_use) {
          bool should_update_v = false;
          if (itv == garlileoVelocityAssocById_.end()) {
            should_update_v = true;
          } else if (!itv->second.has_constraint) {
            should_update_v = true;
          } else if (itv->second.retry_pending
                     && dt_curr_abs <= itv->second.abs_dt + 1.0e-9) {
            should_update_v = true;
          } else if (dt_curr_abs + 1.0e-9 < itv->second.abs_dt) {
            should_update_v = true;
          }
          if (should_update_v) {
            // Overwrite any previous velocity/speed priors on this state (GaRLILEO-only usage).
            Eigen::Vector3d v_S = meas_curr->v_body;
            if (parameters_.garlileo.apply_imu_extrinsic) {
              // imu.T_BS is S->B, so inverse is B->S
              const okvis::kinematics::Transformation T_SB = parameters_.imu.T_BS.inverse();
              v_S = T_SB.C() * meas_curr->v_body;
            }
            const double vel_nom_m =
                parameters_.garlileo.velocity_std * garlileo_nominal_std_scale;
            const double vel_fb_m =
                parameters_.garlileo.pose_fallback_velocity_std > 0.0
                    ? parameters_.garlileo.pose_fallback_velocity_std
                    : vel_nom_m;
            const double vel_std_used =
                parameters_.garlileo.pose_fallback_enable
                    ? lerpPair(vel_nom_m, vel_fb_m, w_vision_fallback_blend)
                    : vel_nom_m;
            const double vel_std_eff = std::max(1.0e-12, vel_std_used);
            const bool applied_velocity = updateGarlileoVelocityPrior(
                currentId, v_S, vel_std_eff, dt_curr_abs, true);

            static double last_vel_log = 0.0;
            const double now = okvis::Time::now().toSec();
            if (applied_velocity && now - last_vel_log > 1.0) {
              LOG(INFO) << "[GaRLILEO] Applied velocity prior: k=" << currentId.value()
                        << " dt_s=" << dt_curr_signed
                        << " |dt|=" << dt_curr_abs
                        << " over_max_dt=" << (dt_curr_abs > max_dt)
                        << " max_dt=" << max_dt
                        << " w_vel_blend=" << w_vision_fallback_blend
                        << " vel_std=" << vel_std_eff;
              last_vel_log = now;
            }
          }
        } else {
          (void)updateGarlileoVelocityPrior(
              currentId, Eigen::Vector3d::Zero(), 1.0, dt_curr_abs, false);
        }
      } else {
        // Missing measurement: ensure unused.
        (void)updateGarlileoVelocityPrior(currentId, Eigen::Vector3d::Zero(), 1.0,
                                         std::numeric_limits<double>::infinity(), false);
      }
    }
 
     // Also add Lidar Measurements as Live Factors; doing it here before the dataAssociationAndInitialization
     // ensures that we can add the factors before the first optimization
 
     if(useAlignmentFactors_)
     {
       TimerSwitchable tProcessLiveDepthLidar("8 Processing data for map-to-frame factors");
 
       // Motion Compensation of LiDAR Point Cloud
       kinematics::Transformation T_WS_live = estimator_.pose(StateId(multiFrame->id()));
       if(parameters_.lidar) {
         if(previousSubmap_){
           TimerSwitchable tLiveUndistortion("8.1 Live undistortion");
           LidarMotionUndistortion motionUndistortion(lastOptimisedState_, T_WS_live, T_SD_,
                                                     lidarMeasurementDeque_, imuMeasurementDeque_);
           motionUndistortion.deskew();
           tLiveUndistortion.stop();
 
           //ToDo: Two-State downsampling!! (or think about it more carefully) => otherwise determining observed points quite slow
           TimerSwitchable tFilterObserved("8.2 Filter observed points");
           size_t observed_points = motionUndistortion.filterObserved(previousSubmap_, estimator_.pose(StateId(previousSubmapId_)));
           if(observed_points < submapConfig_.numSubmapFactors){
               noOverlapCounter_++;
           }
           else{
               noOverlapCounter_ = 0;
           }
           tFilterObserved.stop();
           TimerSwitchable tDownsampling("8.3 Downsampling");
           motionUndistortion.downsample(submapConfig_.numSubmapFactors, submapConfig_.voxelGridResolution);
           tDownsampling.stop();
 
           if(alignmentPublishCallback_){
             alignmentPublishCallback_(multiFrame->timestamp(), T_WS_live, motionUndistortion.deskewedDownsampledPointCloud(), false);
           } 
 
          // Also add LiDAR Factors (live factors only)
           std::vector<float> sensorErrors(motionUndistortion.deskewedDownsampledPointCloud().size());
           std::fill(sensorErrors.begin(), sensorErrors.end(), submapConfig_.sensorError);
           // Visual fallback (optional): blend submap information scale using the same EWG/tracking
           // weight as other GaRLILEO fallbacks. Data association is not run yet here, so we do not
           // include the per-frame (!da_ok) term (see w_visual_total after matching).
           {
             const double submap_tgt = parameters_.garlileo.visual_fallback_submap_information_scale;
             const double submap_eff =
                 lerpPair(1.0, submap_tgt, w_vision_fallback_blend);
             if (parameters_.garlileo.visual_fallback_downweight_enable && w_vision_fallback_blend > 0.0
                 && submap_eff > 0.0 && submap_eff != 1.0) {
               const float inv_sqrt_s = static_cast<float>(1.0 / std::sqrt(submap_eff));
               for (auto& s : sensorErrors) {
                 s *= inv_sqrt_s;
               }
               LOG_FIRST_N(WARNING, 5)
                   << "[VisualFallback] downweighted submap (LiDAR live): scale=" << submap_eff
                   << " npoints=" << sensorErrors.size();
             }
           }
           estimator_.addSubmapAlignmentConstraints(
                   previousSubmap_, previousSubmapId_, multiFrame->id(),
                   motionUndistortion.deskewedDownsampledPointCloud(), sensorErrors, true, "Tukey");
         }
       }
       else {
         std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> livePoints;
         std::vector<float> liveSigmas;
         computeLiveDepthMeasurements(T_WS_live, multiFrame->timestamp(), livePoints, liveSigmas);
         if (livePoints.size() > 0.05*submapConfig_.numSubmapFactors) {
           if(alignmentPublishCallback_){
             alignmentPublishCallback_(multiFrame->timestamp(), T_WS_live, livePoints, false);
           }

           // Visual fallback (optional): inflate per-point sigma when vision is weak so that
           // the submap alignment information is reduced by the configured scale.
           {
             const double submap_tgt_d = parameters_.garlileo.visual_fallback_submap_information_scale;
             const double submap_eff_d =
                 lerpPair(1.0, submap_tgt_d, w_vision_fallback_blend);
             if (parameters_.garlileo.visual_fallback_downweight_enable && w_vision_fallback_blend > 0.0
                 && submap_eff_d > 0.0 && submap_eff_d != 1.0) {
               const float inv_sqrt_s = static_cast<float>(1.0 / std::sqrt(submap_eff_d));
               for (auto& s : liveSigmas) {
                 s *= inv_sqrt_s;
               }
               LOG_FIRST_N(WARNING, 5)
                   << "[VisualFallback] downweighted submap (depth live): scale=" << submap_eff_d
                   << " npoints=" << liveSigmas.size();
             }
           }
           estimator_.addSubmapAlignmentConstraints(previousSubmap_, previousSubmapId_, multiFrame->id(),
                                                     livePoints, liveSigmas, false, "Tukey");
           float sigmaMean = 0;
           for (size_t ii = 0; ii < liveSigmas.size(); ii++) {
             sigmaMean += liveSigmas[ii];
           }
           DLOG(INFO) << "Map-to-frame factor added "<< previousSubmapId_ << "-" 
             << multiFrame->id() << " with " << livePoints.size()
             << " with uncertainty " << sigmaMean / liveSigmas.size() << std::endl;
           // saveAlignedPoints(multiFrame->id(), livePoints,
           //                   "frame2map_"+std::to_string(previousSubmapId_));
         }  
       }
       tProcessLiveDepthLidar.stop();
     }
   }
   imuMeasurementsByFrame_[StateId(multiFrame->id())] = imuMeasurementDeque_;
 
   // Check if we need new kf due to lidar overlap
   bool kfPrior = false;
   if(submapConfig_.useMap2LiveFactors && parameters_.lidar){
     kfPrior = needsNewLidarKeyframe();
     if(kfPrior) {
       lidarKeyframes_.insert(StateId(multiFrame->id()));
     }
   }
 
  // Provide GaRLILEO ego speed to the frontend so it can skip keyframe overlap checks when stationary.
  if (parameters_.garlileo.keyframe_stationary_speed_threshold > 0.0) {
   const okvis::Time t_query = multiFrame->timestamp();
    double best_dt = std::numeric_limits<double>::infinity();
    std::optional<GaRLILEOMeasurement> best_meas;
    if (!garlileoHistory_.empty()) {
      for (const auto& m : garlileoHistory_) {
        if (!isValidGarlileoMeasurement(m)) {
          continue;
        }
        const double dt = std::abs((m.timeStamp - t_query).toSec());
        if (dt < best_dt) {
          best_dt = dt;
          best_meas = m;
        }
      }
    } else if (hasLastGarlileo_) {
      std::lock_guard<std::mutex> lock(garlileoMutex_);
      if (isValidGarlileoMeasurement(lastGarlileoMeas_)) {
        best_meas = lastGarlileoMeas_;
        best_dt = std::abs((lastGarlileoMeas_.timeStamp - t_query).toSec());
      }
    }

    if (best_meas && best_dt <= parameters_.garlileo.max_time_diff) {
      const double speed = best_meas->v_body.norm();
      frontend_.setExternalEgoSpeed(speed);
    } else {
      frontend_.clearExternalEgoSpeed();
    }
  } else {
    frontend_.clearExternalEgoSpeed();
  }

  // Radar place recognition (optional): provide radar targets for loop-closure ranking.
  frontend_.clearExternalRadarTargets();
  const bool need_radar = parameters_.garlileo.place_recognition_use_radar;
  if (need_radar) {
    const okvis::Time t_query = multiFrame->timestamp();
    const double max_dt = parameters_.garlileo.place_recognition_radar_max_time_diff;

    auto radarEnabled = [&](int radar_index) -> bool {
      if (radar_index == 1) {
        return parameters_.garlileo.place_recognition_radar1_enable;
      }
      return parameters_.garlileo.place_recognition_radar0_enable;
    };
    auto radarTBr = [&](int radar_index) -> const okvis::kinematics::Transformation& {
      if (radar_index == 1) {
        return parameters_.garlileo.place_recognition_radar1_T_BR;
      }
      return parameters_.garlileo.place_recognition_radar0_T_BR;
    };

    const int scans_per_radar =
        std::max(1, parameters_.garlileo.place_recognition_radar_scans_per_radar);

    auto findClosestForRadar =
        [&](int radar_index, RadarTargetsMeasurement& best_meas, double& best_abs_dt) -> bool {
          best_abs_dt = std::numeric_limits<double>::infinity();
          bool found = false;
          if (!radarTargetsHistory_.empty()) {
            for (const auto& m : radarTargetsHistory_) {
              if (m.radarIndex != radar_index) {
                continue;
              }
              const double dt = std::abs((m.timeStamp - t_query).toSec());
              if (dt < best_abs_dt) {
                best_abs_dt = dt;
                best_meas = m;
                found = true;
              }
            }
          }
          if (!found && hasLastRadarTargets_) {
            std::lock_guard<std::mutex> lock(radarTargetsMutex_);
            if (lastRadarTargetsMeas_.radarIndex == radar_index) {
              best_meas = lastRadarTargetsMeas_;
              best_abs_dt = std::abs((lastRadarTargetsMeas_.timeStamp - t_query).toSec());
              found = true;
            }
          }
          return found;
        };

    auto collectRecentForRadar =
        [&](int radar_index, const RadarTargetsMeasurement& anchor,
            std::vector<RadarTargetsMeasurement>& out_scans) {
          out_scans.clear();
          if (!radarTargetsHistory_.empty()) {
            size_t best_idx = size_t(-1);
            double best_abs_dt = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < radarTargetsHistory_.size(); ++i) {
              const auto& m = radarTargetsHistory_[i];
              if (m.radarIndex != radar_index) {
                continue;
              }
              const double adt = std::abs((m.timeStamp - anchor.timeStamp).toSec());
              if (adt < best_abs_dt) {
                best_abs_dt = adt;
                best_idx = i;
              }
            }

            if (best_idx != size_t(-1)) {
              std::vector<size_t> radar_indices;
              radar_indices.reserve(best_idx + 1);
              for (size_t i = 0; i <= best_idx; ++i) {
                if (radarTargetsHistory_[i].radarIndex == radar_index) {
                  radar_indices.push_back(i);
                }
              }

              const size_t take = std::min<size_t>(size_t(scans_per_radar), radar_indices.size());
              if (take > 0) {
                const size_t begin = radar_indices.size() - take;
                for (size_t k = begin; k < radar_indices.size(); ++k) {
                  out_scans.push_back(radarTargetsHistory_[radar_indices[k]]);
                }
              }
            }
          }

          if (out_scans.empty()) {
            out_scans.push_back(anchor);
          }
        };

    std::vector<RadarTargetsMeasurement> scans;
    scans.reserve((radarEnabled(0) ? scans_per_radar : 0)
                  + (radarEnabled(1) ? scans_per_radar : 0));
    double best_dt_radar = std::numeric_limits<double>::infinity();
    for (int radar_index = 0; radar_index < 2; ++radar_index) {
      if (!radarEnabled(radar_index)) {
        continue;
      }
      RadarTargetsMeasurement radar_best;
      double best_dt_this = std::numeric_limits<double>::infinity();
      if (!findClosestForRadar(radar_index, radar_best, best_dt_this)) {
        continue;
      }
      if (!(best_dt_this <= max_dt)) {
        continue;
      }
      best_dt_radar = std::min(best_dt_radar, best_dt_this);
      std::vector<RadarTargetsMeasurement> per_radar_scans;
      collectRecentForRadar(radar_index, radar_best, per_radar_scans);
      scans.insert(scans.end(), per_radar_scans.begin(), per_radar_scans.end());
    }

    if (!scans.empty()) {
      std::sort(scans.begin(), scans.end(),
                [](const RadarTargetsMeasurement& a, const RadarTargetsMeasurement& b) {
                  return a.timeStamp.toSec() < b.timeStamp.toSec();
                });

      const okvis::RadarTargetReadings* targets_for_polar = &scans.back().targets;
      okvis::RadarTargetReadings pr_targets_motion_comp;

      // Place recognition: build a denser radar context by aggregating recent scans
      // from each enabled radar with motion compensation using GaRLILEO spline poses.
      if (parameters_.garlileo.place_recognition_use_radar) {
        const int n = int(scans.size());
        if (n > 0) {
          const double pose_max_dt = std::max(0.0, parameters_.garlileo.max_time_diff);
          const auto [ref_pose_meas, ref_pose_dt] = findClosestGarlileo(scans.back().timeStamp);
          const bool has_ref_pose = ref_pose_meas && ref_pose_dt <= pose_max_dt;
          okvis::kinematics::Transformation T_WB_ref;
          if (has_ref_pose) {
            T_WB_ref = okvis::kinematics::Transformation(
                ref_pose_meas->p_W, ref_pose_meas->R_WS.normalized());
          }

          const size_t scan_count = static_cast<size_t>(n);
          AlignedVector<okvis::kinematics::Transformation> T_WB_scan(scan_count);
          std::vector<bool> has_scan_pose(scan_count, false);
          int num_pose_found = 0;
          double max_pose_abs_dt = 0.0;
          for (int j = 0; j < n; ++j) {
            const auto [scan_pose_meas, scan_pose_dt] = findClosestGarlileo(scans[size_t(j)].timeStamp);
            if (scan_pose_meas && scan_pose_dt <= pose_max_dt) {
              T_WB_scan[size_t(j)] = okvis::kinematics::Transformation(
                  scan_pose_meas->p_W, scan_pose_meas->R_WS.normalized());
              has_scan_pose[size_t(j)] = true;
              ++num_pose_found;
              max_pose_abs_dt = std::max(max_pose_abs_dt, scan_pose_dt);
            }
          }

          // Transform each scan into body frame, then motion-compensate and aggregate.
          size_t reserve_n = 0;
          for (const auto& s : scans) {
            reserve_n += s.targets.size();
          }
          pr_targets_motion_comp.reserve(reserve_n);
          size_t radar0_targets = 0;
          size_t radar1_targets = 0;
          for (int j = 0; j < n; ++j) {
            const int radar_index = scans[size_t(j)].radarIndex;
            const okvis::kinematics::Transformation& T_BR = radarTBr(radar_index);
            const Eigen::Matrix3d C_BR = T_BR.C();
            const Eigen::Vector3d r_BR = T_BR.r();
            for (const auto& tar : scans[size_t(j)].targets) {
              okvis::RadarTargetReading t = tar;
              const Eigen::Vector3d p_B = C_BR * tar.point_R + r_BR;
              if (has_ref_pose && has_scan_pose[size_t(j)]) {
                const Eigen::Vector3d p_W = T_WB_scan[size_t(j)].C() * p_B
                                            + T_WB_scan[size_t(j)].r();
                t.point_R = T_WB_ref.C().transpose() * (p_W - T_WB_ref.r());
              } else {
                t.point_R = p_B;
              }
              pr_targets_motion_comp.push_back(t);
              if (radar_index == 1) {
                radar1_targets++;
              } else {
                radar0_targets++;
              }
            }
          }

          targets_for_polar = &pr_targets_motion_comp;
          frontend_.setExternalRadarTargets(scans.back().timeStamp, pr_targets_motion_comp);

          static double last_log = 0.0;
          const double now = okvis::Time::now().toSec();
          if (now - last_log > 1.0) {
            int used_radar0_scans = 0;
            int used_radar1_scans = 0;
            for (const auto& s : scans) {
              if (s.radarIndex == 1) {
                ++used_radar1_scans;
              } else {
                ++used_radar0_scans;
              }
            }
            LOG(INFO) << "[RadarPR] agg_scans_total=" << n
                      << " scans_per_radar=" << scans_per_radar
                      << " used_radar0_scans=" << used_radar0_scans
                      << " used_radar1_scans=" << used_radar1_scans
                      << " best_dt_radar=" << best_dt_radar << "s"
                      << " agg_targets=" << pr_targets_motion_comp.size()
                      << " agg_targets_radar0=" << radar0_targets
                      << " agg_targets_radar1=" << radar1_targets
                      << " pose_comp_ref_ok=" << has_ref_pose
                      << " pose_found=" << num_pose_found << "/" << n
                      << " max_pose_abs_dt=" << max_pose_abs_dt << "s"
                      << " pose_max_dt=" << pose_max_dt << "s";
            last_log = now;
          }
        }
      }

      // Publish radar polar (40x40) debug image for RViz.
      {
        constexpr int kNumRangeBins = 40;
        constexpr int kNumAzBins = 40;
        constexpr double kRangeBinM = 0.5;
        constexpr double kAzBinDeg = 3.0;
        constexpr double kAzMinDeg = -60.0;
        constexpr double kAzMaxDeg = kAzMinDeg + kNumAzBins * kAzBinDeg;
        constexpr double kPi = 3.14159265358979323846;

        const double intensity_min = parameters_.garlileo.place_recognition_radar_min_intensity;
        cv::Mat polar_f = cv::Mat::zeros(kNumAzBins, kNumRangeBins, CV_32FC1); // rows: az, cols: range
        for (const auto& tar : *targets_for_polar) {
          const double intensity = tar.intensity;
          if (!std::isfinite(intensity) || intensity <= intensity_min) {
            continue;
          }
          const double x = tar.point_R.x();
          const double y = tar.point_R.y();
          if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
          }
          const double range = std::sqrt(x * x + y * y);
          if (!std::isfinite(range) || range <= 0.0) {
            continue;
          }
          const int r_bin = int(std::floor(range / kRangeBinM));
          if (r_bin < 0 || r_bin >= kNumRangeBins) {
            continue;
          }
          const double az_deg = std::atan2(y, x) * (180.0 / kPi);
          if (az_deg < kAzMinDeg || az_deg >= kAzMaxDeg) {
            continue;
          }
          const int a_bin = int(std::floor((az_deg - kAzMinDeg) / kAzBinDeg));
          if (a_bin < 0 || a_bin >= kNumAzBins) {
            continue;
          }
          float& cell = polar_f.at<float>(a_bin, r_bin);
          cell = std::max(cell, float(intensity));
        }

        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(polar_f, &minv, &maxv);
        cv::Mat polar_u8;
        if (maxv > 1e-9) {
          polar_f.convertTo(polar_u8, CV_8UC1, 255.0 / maxv);
        } else {
          polar_u8 = cv::Mat::zeros(kNumAzBins, kNumRangeBins, CV_8UC1);
        }
        cv::Mat polar_color;
        cv::applyColorMap(polar_u8, polar_color, cv::COLORMAP_JET);
        cv::Mat polar_vis;
        cv::resize(polar_color, polar_vis, cv::Size(400, 400), 0.0, 0.0, cv::INTER_NEAREST);

        radarPolarImages_.PushNonBlockingDroppingIfFull(polar_vis, 1);
      }

     }
   }
 
   // call the matcher
  const bool da_ok = frontend_.dataAssociationAndInitialization(
        estimator_, parameters_, multiFrame, kfPrior, &asKeyframe);

  if(!da_ok && !frontend_.isInitialized()) {
     LOG(WARNING) << "Not enough matches, cannot initialise";
     frontend_.clear();
     estimator_.clear();
     frontendDeltaHasPrev_ = false;
     return false;
   }

  // Blend in [0,1] toward visual/pose-fallback scales: EWG semi-band + tracking-Q loss,
  // plus full fallback when data association fails for this frame.
  const double w_visual_total =
      std::max(w_vision_fallback_blend, !da_ok ? 1.0 : 0.0);

// Relative-pose fallback (optional): when tracking is lost (frontend failed),
// strengthen the GaRLILEO between constraint for the last pair.
  if (frontend_.isInitialized()
      && parameters_.garlileo.pose_fallback_enable
      && !da_ok
      && estimator_.numFrames() >= 2) {
    const StateId currentId = estimator_.currentStateId();
    const StateId prevId(currentId.value() - 1);
    const okvis::Time t_curr = estimator_.timestamp(currentId);
    const okvis::Time t_prev = estimator_.timestamp(prevId);
    // Tracking-lost is itself a visual fallback regime; if the user has opted in to
    // forcing the GaRLILEO between factor, block briefly here so that the GaRLILEO
    // measurements at t_prev/t_curr are guaranteed to be available before we add
    // the strengthened between constraint below.
    const double tl_wait_sec =
        parameters_.garlileo.visual_fallback_garlileo_force_between
            ? std::max(0.0, parameters_.garlileo.visual_fallback_garlileo_wait_max_sec)
            : 0.0;
    const int tl_wait_poll_ms = std::max(1,
        parameters_.garlileo.visual_fallback_garlileo_wait_poll_period_ms);
    (void)waitForGarlileoMeasurement(t_prev, parameters_.garlileo.max_time_diff,
                                     "relpose_fallback_prev", tl_wait_sec, tl_wait_poll_ms);
    (void)waitForGarlileoMeasurement(t_curr, parameters_.garlileo.max_time_diff,
                                     "relpose_fallback_curr", tl_wait_sec, tl_wait_poll_ms);

    auto [meas_curr, min_dt_curr] = findClosestGarlileo(t_curr);
    auto [meas_prev, min_dt_prev] = findClosestGarlileo(t_prev);
    (void)min_dt_curr;
    (void)min_dt_prev;
    enforceDistinctPair(t_prev, t_curr, meas_prev, min_dt_prev, meas_curr, min_dt_curr);
    if (meas_curr && meas_prev) {
      const double max_dt = parameters_.garlileo.max_time_diff;
      const double dt_curr_signed = (t_curr - meas_curr->timeStamp).toSec();
      const double dt_prev_signed = (t_prev - meas_prev->timeStamp).toSec();
      const double dt_curr_abs = std::abs(dt_curr_signed);
      const double dt_prev_abs = std::abs(dt_prev_signed);
      const Eigen::Quaterniond q_WB0 = meas_prev->R_WS.normalized();
      const Eigen::Quaterniond q_WB1 = meas_curr->R_WS.normalized();
      okvis::kinematics::Transformation T_WB0(meas_prev->p_W, q_WB0);
      okvis::kinematics::Transformation T_WB1(meas_curr->p_W, q_WB1);

      okvis::kinematics::Transformation T_WS0 = T_WB0;
      okvis::kinematics::Transformation T_WS1 = T_WB1;
      if (parameters_.garlileo.apply_imu_extrinsic) {
        T_WS0 = T_WB0 * parameters_.imu.T_BS;
        T_WS1 = T_WB1 * parameters_.imu.T_BS;
      }
      const okvis::kinematics::Transformation T_S0S1 = T_WS0.inverse() * T_WS1;

      const bool use_rot_between = parameters_.garlileo.use_rotation;
      const bool use_trans_between = parameters_.garlileo.use_translation;
      const double nominal_scale_rpf = garlileoNominalStdScaleWithConfig(
          parameters_.garlileo.std_scale_use_tracking_quality,
          estimator_.trackingQuality(currentId),
          parameters_.garlileo.std_scale_min_tracking_quality);

      if ((use_rot_between || use_trans_between) && (dt_curr_abs <= max_dt) && (dt_prev_abs <= max_dt)) {
        const Eigen::Matrix<double, 6, 6> information = makeGarlileoBetweenInformation(
            parameters_.garlileo, use_rot_between, use_trans_between, 1.0, nominal_scale_rpf);
        const bool added_between =
            applyGarlileoBetweenIfAllowed(prevId, currentId, T_S0S1, information);

        const double rot_rp_std_deg =
            positiveOrFallback(parameters_.garlileo.pose_fallback_rotation_roll_pitch_std_deg,
                               2.0);
        const double rot_yaw_std_deg =
            positiveOrFallback(parameters_.garlileo.pose_fallback_rotation_yaw_std_deg,
                               2.0);

        // Remember association quality for potential later backfill.
        GaRLILEOBetweenAssoc assoc;
        assoc.max_abs_dt = std::max(dt_prev_abs, dt_curr_abs);
        assoc.has_constraint = added_between;
        garlileoBetweenAssocByCurrId_[currentId.value()] = assoc;

        LOG(WARNING) << "[GaRLILEO][RelPoseFallback] strengthened between constraint (tracking lost): "
                     << "prev=" << prevId.value()
                     << " curr=" << currentId.value()
                     << " used_between=" << added_between
                     << " dt_prev_s=" << dt_prev_signed << "s"
                     << " dt_curr_s=" << dt_curr_signed << "s"
                     << " |dt_prev|=" << dt_prev_abs << "s"
                     << " |dt_curr|=" << dt_curr_abs << "s"
                     << " over_max_dt=" << ((dt_prev_abs > max_dt) || (dt_curr_abs > max_dt))
                     << " max_dt=" << max_dt
                     << " trans_between_std=" << parameters_.garlileo.pose_fallback_translation_std
                     << " rot_rp_between_std_deg=" << rot_rp_std_deg
                     << " rot_yaw_between_std_deg=" << rot_yaw_std_deg
                     << " pose_fallback_weight=" << parameters_.garlileo.pose_fallback_weight
                     << " (dt_s = (t_query - t_meas); + => meas older/lag; - => meas newer/lead)";
      }
    }
  }

  // Velocity fallback (optional): when tracking is lost (frontend failed), optionally tighten the
  // GaRLILEO velocity/speed prior for the current state. This uses a dedicated fallback std so the
  // user can strengthen velocity priors during fallback without affecting normal operation.
  if (frontend_.isInitialized()
      && parameters_.garlileo.pose_fallback_enable
      && parameters_.garlileo.pose_fallback_velocity_std > 0.0
      && parameters_.garlileo.use_velocity
      && !da_ok) {
    const StateId currentId = estimator_.currentStateId();
    const okvis::Time t_curr = estimator_.timestamp(currentId);
    const auto [meas_curr, min_dt_curr] = findClosestGarlileo(t_curr);
    (void)min_dt_curr;
    if (meas_curr) {
      const double max_dt = parameters_.garlileo.max_time_diff;
      const double dt_curr_signed = (t_curr - meas_curr->timeStamp).toSec();
      const double dt_curr_abs = std::abs(dt_curr_signed);
      auto itv = garlileoVelocityAssocById_.find(currentId.value());
      const bool can_use = dt_curr_abs <= max_dt;
      if (can_use) {
        bool should_update_v = false;
        if (itv == garlileoVelocityAssocById_.end()) {
          should_update_v = true;
        } else if (!itv->second.has_constraint) {
          should_update_v = true;
        } else if (itv->second.retry_pending
                   && dt_curr_abs <= itv->second.abs_dt + 1.0e-9) {
          should_update_v = true;
        } else if (dt_curr_abs + 1.0e-9 < itv->second.abs_dt) {
          should_update_v = true;
        } else if (w_vision_fallback_blend < 1.0
                   && dt_curr_abs <= itv->second.abs_dt + 1.0e-9) {
          // The same valid measurement still needs the full fallback covariance
          // when matching has just failed after a nominal or blended prior.
          // Preserve an existing association if the new one is worse.
          should_update_v = true;
        }
        if (should_update_v) {
          Eigen::Vector3d v_S = meas_curr->v_body;
          if (parameters_.garlileo.apply_imu_extrinsic) {
            const okvis::kinematics::Transformation T_SB = parameters_.imu.T_BS.inverse();
            v_S = T_SB.C() * meas_curr->v_body;
          }
          const double vel_std_eff =
              std::max(1.0e-12, parameters_.garlileo.pose_fallback_velocity_std);
          const bool applied_velocity = updateGarlileoVelocityPrior(
              currentId, v_S, vel_std_eff, dt_curr_abs, true);

          if (applied_velocity) LOG(WARNING) << "[GaRLILEO][VelFallback] applied velocity fallback prior (tracking lost): "
                       << "k=" << currentId.value()
                       << " dt_s=" << dt_curr_signed << "s"
                       << " |dt|=" << dt_curr_abs << "s"
                       << " over_max_dt=" << (dt_curr_abs > max_dt)
                       << " max_dt=" << max_dt
                       << " vel_std=" << vel_std_eff;
        }
      } else {
        (void)updateGarlileoVelocityPrior(
            currentId, Eigen::Vector3d::Zero(), 1.0, dt_curr_abs, false);
      }
    }
  }

  // Cache per-frame vision quality so backfill can preserve whether fallback weights should apply
  // for older (k-1,k) pairs.
  if (frontend_.isInitialized()) {
    GaRLILEOFrameQuality q;
    q.low_features = low_features;
    q.tracking_lost = !da_ok;
    q.vision_fallback_blend = w_vision_fallback_blend;
    garlileoFrameQualityById_[multiFrame->id()] = q;

    // Ordered maps: erase only the expired prefix, without scanning retained frames.
    const uint64_t id_now = multiFrame->id();
    constexpr uint64_t kKeepFrames = 5000;
    const uint64_t prune_before = (id_now > kKeepFrames) ? (id_now - kKeepFrames) : 0;
    garlileoFrameQualityById_.erase(
        garlileoFrameQualityById_.begin(), garlileoFrameQualityById_.lower_bound(prune_before));
    garlileoBetweenAssocByCurrId_.erase(
        garlileoBetweenAssocByCurrId_.begin(), garlileoBetweenAssocByCurrId_.lower_bound(prune_before));
    garlileoVelocityAssocById_.erase(
        garlileoVelocityAssocById_.begin(), garlileoVelocityAssocById_.lower_bound(prune_before));
  }

  // Visual fallback (optional): when vision is unreliable (low features or tracking lost),
  // downweight camera factors for the *current* frame. This reduces the influence of camera
  // factors in the subsequent optimisation and marginalisation step.
  //
  // Three independent multipliers are supported, each applied only while GaRLILEO is
  // actually anchoring this frame (i.e. between or velocity constraint is active):
  //   (a) reprojection_information_scale  -> scales current-frame reprojection 2x2 info
  //   (b) relpose_information_scale       -> scales every TwoPoseGraphError(*)Const
  //                                          factor (already-marginalised visual
  //                                          relative-pose constraints) in the window
  //   (c) submap_information_scale        -> scales information of subsequent submap
  //                                          alignment factors (handled at the submap
  //                                          add site, not here)
  //
  // (b) is applied through the backend so the change is visible in both the realtime
  // and full graph immediately. When fallback is *not* active for this frame we reset
  // the relpose scale back to 1.0 so the previous-frame downweighting does not leak
  // into a healthy frame.
  const bool poor_visual_now = w_visual_total > 0.0;
  bool relpose_scaled_active = false;
  if (frontend_.isInitialized()
      && parameters_.garlileo.visual_fallback_downweight_enable
      && poor_visual_now) {
    // IMPORTANT: Only downweight visual factors once GaRLILEO is actually available for this frame.
    // Otherwise, at startup (or when GaRLILEO is delayed), we would weaken vision without having
    // a reliable GaRLILEO constraint yet, which can make the system jittery/unstable.
    const uint64_t sid_key = multiFrame->id();
    bool garlileo_between_ok = false;
    bool garlileo_vel_ok = false;
    double garlileo_between_dt = std::numeric_limits<double>::infinity();
    double garlileo_vel_dt = std::numeric_limits<double>::infinity();
    {
      const auto itb = garlileoBetweenAssocByCurrId_.find(sid_key);
      if (itb != garlileoBetweenAssocByCurrId_.end() && itb->second.has_constraint) {
        garlileo_between_ok = true;
        garlileo_between_dt = itb->second.max_abs_dt;
      }
      const auto itv = garlileoVelocityAssocById_.find(sid_key);
      if (itv != garlileoVelocityAssocById_.end() && itv->second.has_constraint) {
        garlileo_vel_ok = true;
        garlileo_vel_dt = itv->second.abs_dt;
      }
    }
    const bool garlileo_ready_now = garlileo_between_ok || garlileo_vel_ok;
    if (!garlileo_ready_now) {
      static double last_skip_log = 0.0;
      const double now = okvis::Time::now().toSec();
      if (now - last_skip_log > 1.0) {
        LOG(WARNING) << "[VisualFallback] skip downweight (GaRLILEO not ready yet): "
                     << "between_ok=" << garlileo_between_ok
                     << " vel_ok=" << garlileo_vel_ok
                     << " dt_between=" << garlileo_between_dt
                     << " dt_vel=" << garlileo_vel_dt
                     << " history_size=" << garlileoHistory_.size()
                     << " low_features=" << low_features
                     << " tracking_lost=" << (!da_ok)
                     << " ewg_gate=" << ewg_gate_enabled
                     << " ewg_utility_min=" << image_utility_min
                     << " ewg_utility_mean=" << image_utility_mean_log
                     << " ewg_utility_sum=" << image_utility_sum
                     << " ewg_utility_cams=" << image_utility_cams
                     << " ewg_utility_thr=" << image_utility_thr;
        last_skip_log = now;
      }
      // Do not apply downweighting yet.
    } else {
      // (a) Reprojection downweight (current frame): interpolate 1 -> s_rp with w_visual_total.
      const double s_rp = visualFallbackReprojectionTarget(
          parameters_.garlileo.visual_fallback_reprojection_information_scale,
          parameters_.garlileo.visual_fallback_reprojection_utility_adaptive,
          parameters_.garlileo.visual_fallback_reprojection_information_scale_min,
          parameters_.garlileo.visual_fallback_reprojection_utility_low,
          parameters_.garlileo.visual_fallback_reprojection_utility_high,
          image_utility_min);
      const double s_rp_eff = lerpPair(1.0, s_rp, w_visual_total);
      if (w_visual_total > 0.0 && s_rp > 0.0 && s_rp_eff > 0.0 && s_rp_eff != 1.0) {
        const StateId sid(multiFrame->id());
        const size_t scaled_obs =
            estimator_.setFrameReprojectionInformationScale(sid, s_rp_eff);
        LOG(WARNING) << "[VisualFallback] downweighted reprojection (poor vision): "
                     << "scale=" << s_rp_eff << " (target=" << s_rp << ")"
                     << " utility_adaptive=" << parameters_.garlileo.visual_fallback_reprojection_utility_adaptive
                     << " w_visual_total=" << w_visual_total
                     << " obs=" << scaled_obs
                     << " low_features=" << low_features
                     << " tracking_lost=" << (!da_ok)
                     << " ewg_gate=" << ewg_gate_enabled
                     << " ewg_utility_min=" << image_utility_min
                     << " ewg_utility_mean=" << image_utility_mean_log
                     << " ewg_utility_sum=" << image_utility_sum
                     << " ewg_utility_cams=" << image_utility_cams
                     << " ewg_utility_thr=" << image_utility_thr;
      }
      // (b) TwoPoseGraphError (visual relative-pose) downweight: interpolate 1 -> s_rel.
      const double s_rel = parameters_.garlileo.visual_fallback_relpose_information_scale;
      const double s_rel_eff = lerpPair(1.0, s_rel, w_visual_total);
      if (w_visual_total > 0.0 && std::isfinite(s_rel) && s_rel >= 0.0
          && s_rel_eff >= 0.0 && s_rel_eff != 1.0) {
        visualRelposeScaleMayNeedReset_ = true;
        const int touched = estimator_.scaleAllTwoPoseGraphErrors(s_rel_eff);
        relpose_scaled_active = true;
        LOG(WARNING) << "[VisualFallback] downweighted TwoPoseGraphError (poor vision): "
                     << "scale=" << s_rel_eff << " (target=" << s_rel << ")"
                     << " w_visual_total=" << w_visual_total << " touched=" << touched;
      }
    }
  }
  // If fallback is not active (or downweight disabled) this frame, ensure any previous
  // downweighting on the marginalised visual relative-pose factors is reset so the
  // optimiser sees their full information.
  // New factors start at unit scale and clones preserve their source scale. Until
  // any non-unit request or possible loop-path suppression, walking both graphs
  // to write 1.0 again is unnecessary. Keep the history once set: a later merge
  // can insert old-weight factors even after the previous request was already 1.
  visualRelposeScaleMayNeedReset_ = visualRelposeScaleMayNeedReset_
      || parameters_.garlileo.fallback_loop_weaken_existing_constraints_enable;
  if (!relpose_scaled_active && visualRelposeScaleMayNeedReset_) {
    (void)estimator_.scaleAllTwoPoseGraphErrors(1.0);
  }

   estimator_.setKeyframe(StateId(multiFrame->id()), asKeyframe);
   matchTimer.stop();
 
   // Add GPS Measurements
   estimator_.addGpsMeasurementsOnAllGraphs(gpsMeasurementDeque_, imuMeasurementDeque_);
 
   // remove gpsMeasurements from deque
   while(!shutdown_ && !gpsMeasurementDeque_.empty() && gpsMeasurementDeque_.front().timeStamp < multiFrame->timestamp() )
   {
     gpsMeasurementDeque_.pop_front();
   }
 
   // remove lidarMeasurements from deque
   while(!shutdown_ && !lidarMeasurementDeque_.empty() && lidarMeasurementDeque_.front().timeStamp < multiFrame->timestamp() )
   {
     lidarMeasurementDeque_.pop_front();
   }
 
   // remove depthMeasurements from deque
   while(!shutdown_ && !depthMeasurementDeque_.empty() && depthMeasurementDeque_.front().timeStamp <= multiFrame->timestamp() ) {
     depthMeasurementDeque_.pop_front();
   }
 
   // Map to Map Alignment Factors are added here...
   AlignmentTerm alignmentTerm;
   while(!shutdown_ && !submapAlignmentFactorsReceived_.Empty())
   {
     submapAlignmentFactorsReceived_.PopNonBlocking(&alignmentTerm);

    // (low-light handling removed)
 
     // Set reference submap for live constraints
     // Prioritize the older submap
     if(parameters_.lidar) {
       // add submap alignment constraints between map keyframes A and B
       estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
                                                alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
                                                alignmentTerm.sensorError, true, "Tukey");
       
       previousSubmap_ = alignmentTerm.submap_B_ptr;
       previousSubmapId_ = alignmentTerm.frame_B_id;
     }
     else {
       // add submap alignment constraints between map keyframes A and B
       if (alignmentTerm.submap_A_ptr) {
         estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
                                                 alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
                                                 alignmentTerm.sensorError, false, "Tukey");
         // saveAlignedPoints(alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
         //                 "map2map_"+std::to_string(alignmentTerm.frame_A_id));
       }
       if (alignmentTerm.frame_A_id != 0) {
         previousSubmap_ = alignmentTerm.submap_A_ptr;
         previousSubmapId_ = alignmentTerm.frame_A_id;
       }
       else {
         previousSubmap_ = alignmentTerm.submap_B_ptr;
         previousSubmapId_ = alignmentTerm.frame_B_id;
       }
     }
     LOG(INFO) << "[ThreadedSlam] Registered new reference submap with ID " << previousSubmapId_;
   }
 
   // start optimise, publish&visualise, marginalise:
   Eigen::Vector3d gyr(0,0,0);
   auto riter = imuMeasurementDeque_.rbegin();
   while(riter!=imuMeasurementDeque_.rend() && riter->timeStamp > multiFrame->timestamp()) {
     gyr = riter->measurement.gyroscopes;
     ++riter;
   }
   Time now = Time::now();
   double dt = parameters_.estimator.realtime_time_limit-(now-matchingStart).toSec();
   if(dt < 0.0) {
     dt = 0.01;
   }
   if(parameters_.estimator.enforce_realtime) {
     estimator_.setOptimisationTimeLimit(dt,
           parameters_.estimator.realtime_min_iterations);
   }
   optimisationThread_ = std::thread(&ThreadedSlam::optimisePublishMarginalise,
                                     this, multiFrame, gyr);
 
   // kick off posegraph optimisation, if needed, too...
   if(estimator_.needsFullGraphOptimisation()) {
     if(fullGraphOptimisationThread_.joinable()) {
       fullGraphOptimisationThread_.join();
     }
     //std::cout << "launching full graph optimisation" << std::endl;
     // hack: call full graph optimisation
     fullGraphOptimisationThread_ = std::thread(
           &ViSlamBackend::optimiseFullGraph, &estimator_,
           parameters_.estimator.full_graph_iterations,
           std::ref(posegraphOptimisationSummary_),
           parameters_.estimator.full_graph_num_threads, false, 1.0e-6);
   }
 
   return true;
 }
 
 void ThreadedSlam::optimisePublishMarginalise(MultiFramePtr multiFrame,
                                               const Eigen::Vector3d& gyroReading) {
   kinematics::Transformation T_WS;
   SpeedAndBias speedAndBiases;
   const size_t numCameras = parameters_.nCameraSystem.numCameras();
 
   // optimise (if initialised)
   TimerSwitchable optimiseTimer("3 Optimise");
  const double optimise_start_wall_sec = okvis::Time::now().toSec();
   std::vector<StateId> updatedStatesRealtime;
   estimator_.optimiseRealtimeGraph(
       parameters_.estimator.realtime_max_iterations, updatedStatesRealtime,
       parameters_.estimator.realtime_num_threads,
       false, false, frontend_.isInitialized());
   optimiseTimer.stop();
  const double optimise_end_wall_sec = okvis::Time::now().toSec();
  lastOptimisationDurationSec_.store(optimise_end_wall_sec - optimise_start_wall_sec,
                                     std::memory_order_relaxed);
  lastOptimisationFinishedWallTimeSec_.store(optimise_end_wall_sec,
                                             std::memory_order_relaxed);
 
   // import pose graph optimisation
   std::vector<StateId> updatedStatesSync;
   if(estimator_.isLoopClosureAvailable()) {
     OKVIS_ASSERT_TRUE(Exception,
                       !estimator_.isLoopClosing(),
                       "loop closure available, but still loop closing -- bug")
     TimerSwitchable synchronisationTimer("5 Import full optimisation");
     estimator_.synchroniseRealtimeAndFullGraph(updatedStatesSync);
     synchronisationTimer.stop();
   }
 
   // prepare for publishing
   TimerSwitchable publishTimer("4 Prepare publishing");
   T_WS = estimator_.pose(StateId(multiFrame->id()));
   speedAndBiases = estimator_.speedAndBias(StateId(multiFrame->id()));
   StateId id(multiFrame->id());
   State state;
   state.id = id;
   state.T_WS = T_WS;
   state.v_W = speedAndBiases.head<3>();
   state.b_g = speedAndBiases.segment<3>(3);
   state.b_a = speedAndBiases.tail<3>();
   state.omega_S = gyroReading - speedAndBiases.segment<3>(3);
   state.timestamp = multiFrame->timestamp();
   state.previousImuMeasurements = imuMeasurementsByFrame_.at(id);
   state.isKeyframe = estimator_.isKeyframe(id);
   {
     // Log the optimised biases so they can be compared against the ones GaRLILEO estimates.
     std::lock_guard<std::mutex> lock(imuBiasRecordsMutex_);
     imuBiasRecords_.push_back(ImuBiasRecord{multiFrame->id(),
                                             state.timestamp.toSec(),
                                             state.b_g,
                                             state.b_a,
                                             state.isKeyframe});
   }
   state.isOnlineExtrinsics = parameters_.camera.online_calibration.do_extrinsics;
   if (state.isOnlineExtrinsics && depthCamera_) {
     for (size_t camIndex = 0; camIndex < numCameras; ++camIndex) {
       if (parameters_.nCameraSystem.cameraType(camIndex).isUsedMapping) {
         if (parameters_.nCameraSystem.cameraType(camIndex).depthType.needRectify) {
           state.extrinsics[camIndex] = estimator_.extrinsics(id, camIndex) * T_rect_;
         }
         else {
           state.extrinsics[camIndex] = estimator_.extrinsics(id, camIndex);
         }
         T_SD_ = state.extrinsics[camIndex]; // update T_SD (D: corresponding camera)
       }
     }
   }
 
   AlignedMap<uint64_t, kinematics::Transformation> T_AiW;
   if (estimator_.T_AiS_.count(id)) {
     AlignedMap<uint64_t, kinematics::Transformation> T_AiS = estimator_.T_AiS_.at(id);
     for (const auto &T_AS : T_AiS) {
       T_AiW[T_AS.first] = T_AS.second * T_WS.inverse();
     }
   }
   state.T_AiW = T_AiW;
   estimator_.getObservedIds(state.id, state.covisibleFrameIds);
 
   TrackingState trackingState;
   if(lidarKeyframes_.count(id) == 0){
     trackingState.isLidarKeyframe = false;
   }
   else{
     trackingState.isLidarKeyframe = true;
   }
   trackingState.id = id;
   trackingState.isKeyframe = estimator_.isKeyframe(id);
   trackingState.recognisedPlace = estimator_.closedLoop(id);
   const double trackingQuality = estimator_.trackingQuality(id);
   const double lostThr = std::max(0.0, parameters_.garlileo.tracking_lost_quality_threshold);
   const double marginalThr = std::max(lostThr, 0.3);
   if(trackingQuality < lostThr) {
     trackingState.trackingQuality = TrackingQuality::Lost;
   } else if (trackingQuality < marginalThr){
     trackingState.trackingQuality = TrackingQuality::Marginal;
   } else {
     trackingState.trackingQuality = TrackingQuality::Good;
   }
   trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(id, false);
 
   // re-propagate
   hasStarted_.store(true);
 
   // now publish
   if(optimisedGraphCallback_) {
     // current state & tracking info via State and Tracking State.
     // the graph:
     std::vector<StateId> updatedStateIds;
     PublicationData publicationData;
     publicationData.state = state;
     publicationData.trackingState = trackingState;
     publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
     if(updatedStatesSync.size()>0) {
       updatedStateIds = updatedStatesSync;
     } else {
       updatedStateIds = updatedStatesRealtime;
     }
     for(const auto & id : updatedStateIds) {
       kinematics::Transformation T_WS = estimator_.pose(id);
       SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
       Time timestamp = estimator_.timestamp(id);
       const bool isKeyframe = estimator_.isKeyframe(id);
       ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
       Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
       for(auto riter = imuMeasurements.rbegin(); riter!=imuMeasurements.rend(); ++riter) {
         if(riter->timeStamp < timestamp) {
           omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
           break;
         }
       }
 
       std::set<StateId> observedIds;
       estimator_.getObservedIds(id, observedIds);
       AlignedVector<Eigen::Vector3d> gpsPoints;
       estimator_.gpsMeasurements(id, gpsPoints);
       (*publicationData.updatedStates)[id] = State{T_WS, speedAndBias.head<3>(),
                                     speedAndBias.segment<3>(3), speedAndBias.tail<3>(),
                                     omega_S, timestamp, id, imuMeasurements, isKeyframe,
                                     AlignedMap<uint64_t, kinematics::Transformation>(),
                                     observedIds, true,
                                     false, AlignedMap<uint64_t, kinematics::Transformation>(),
                                     estimator_.T_GW(), gpsPoints};
     }
     for (const auto &id : affectedStates_) {
       kinematics::Transformation T_WS = estimator_.pose(id);
       SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
       Time timestamp = estimator_.timestamp(id);
       ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
       Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
       for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
         if (riter->timeStamp < timestamp) {
           omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
           break;
         }
       }
       std::set<StateId> observedIds;
       estimator_.getObservedIds(id, observedIds);
       AlignedVector<Eigen::Vector3d> gpsPoints;
       estimator_.gpsMeasurements(id, gpsPoints);
       (*publicationData.updatedStates)[id]
         = State{T_WS,
                 speedAndBias.head<3>(),
                 speedAndBias.segment<3>(3),
                 speedAndBias.tail<3>(),
                 omega_S,
                 timestamp,
                 id,
                 imuMeasurements,
                 estimator_.isKeyframe(id),
                 AlignedMap<uint64_t, kinematics::Transformation>(),
                 observedIds, false,
                 false, AlignedMap<uint64_t, kinematics::Transformation>(),
                 estimator_.T_GW(), gpsPoints};
     }
     affectedStates_.clear();
 
     // landmarks:
     publicationData.landmarksPublish.reset(new MapPointVector());
     MapPoints landmarks;
     estimator_.getLandmarks(landmarks);
     publicationData.landmarksPublish->reserve(landmarks.size());
     for(const auto & lm : landmarks) {
       auto latestObservedFrameId = *lm.second.observations.rbegin();
       publicationData.landmarksPublish->push_back(
             MapPoint(lm.first.value(), lm.second.point, lm.second.quality,
                      latestObservedFrameId.getFrameId()));
     }
 
     // now publish in separate thread. queue size 3 to ensure nothing ever lost.
     if(blocking_){
       // in blocked processing, we also want to be able to block the okvis-estimator from outside
       publicationQueue_.PushBlockingIfFull(publicationData,1);
     }
     else{
       const bool overrun = publicationQueue_.PushNonBlockingDroppingIfFull(publicationData,3);
       if(overrun) {
         LOG(ERROR) << "publication (full update) overrun: dropping";
       }
     }
 
     // if(realtimePropagation_) {
     //   // pass on into IMU processing loop for realtime propagation later.
     //   // queue size 1 because we can afford to lose them; re-prop. just needs newest stuff.
     //   const bool overrun2 = lastOptimisedQueue_.PushNonBlockingDroppingIfFull(publicationData,3);
     //   if(overrun2) {
     //     LOG(WARNING) << "publication (full update) overrun 2: dropping";
     //   }
     // }
 
     // Update Realtime Trajectory object
     std::set<StateId> affectedStateIds;
     trajectory_.update(publicationData.trackingState, publicationData.updatedStates, affectedStateIds);
   }
   publishTimer.stop();
 
   // visualise
   TimerSwitchable visualisationTimer("6 Visualising");
   if(parameters_.output.display_overhead) {
     TimerSwitchable visualisation1Timer("6.1 Visualising overhead");
     // draw debug overhead image
     cv::Mat image(580, 580, CV_8UC3);
     image.setTo(cv::Scalar(10, 10, 10));
     estimator_.drawOverheadImage(image);
     overheadImages_.PushNonBlockingDroppingIfFull(image,1);
     visualisation1Timer.stop();
   }
   if(parameters_.output.display_matches) {
     TimerSwitchable visualisation2Timer("6.2 Prepare visualising matches");
     // draw matches
     // fill in information that requires access to estimator.
     ViVisualizer::VisualizationData::Ptr visualizationDataPtr(
           new ViVisualizer::VisualizationData());
     visualizationDataPtr->observations.resize(multiFrame->numKeypoints());
     okvis::MapPoint2 landmark;
     okvis::ObservationVector::iterator it = visualizationDataPtr
         ->observations.begin();
     for (size_t camIndex = 0; camIndex < numCameras; ++camIndex) {
       visualizationDataPtr->T_SCi.push_back(
         estimator_.extrinsics(estimator_.currentStateId(), camIndex));
       for (size_t k = 0; k < multiFrame->numKeypoints(camIndex); ++k) {
         OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),
                               "Observation-vector not big enough")
         it->keypointIdx = k;
         multiFrame->getKeypoint(camIndex, k, it->keypointMeasurement);
         multiFrame->getKeypointSize(camIndex, k, it->keypointSize);
         it->cameraIdx = camIndex;
         it->frameId = multiFrame->id();
         it->landmarkId = multiFrame->landmarkId(camIndex, k);
         if (estimator_.isLandmarkAdded(LandmarkId(it->landmarkId)) &&
             estimator_.isObserved(KeypointIdentifier(it->frameId, camIndex, k))) {
           estimator_.getLandmark(LandmarkId(it->landmarkId), landmark);
           it->landmark_W = landmark.point;
           it->classification = landmark.classification;
           if (estimator_.isLandmarkInitialised(LandmarkId(it->landmarkId)))
             it->isInitialized = true;
           else
             it->isInitialized = false;
         } else {
           it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);
           // set to infinity to tell visualizer that landmark is not added...
         }
         ++it;
       }
     }
     visualizationDataPtr->T_WS = estimator_.pose(estimator_.currentStateId());
     visualizationDataPtr->currentFrames = multiFrame;
     visualizationDataPtr->isKeyframe = trackingState.isKeyframe;
     visualizationDataPtr->recognisedPlace = trackingState.recognisedPlace;
     if(trackingState.trackingQuality == TrackingQuality::Lost) {
       visualizationDataPtr->trackingQuality =
           ViVisualizer::VisualizationData::TrackingQuality::Lost;
     } else if(trackingState.trackingQuality == TrackingQuality::Marginal) {
       visualizationDataPtr->trackingQuality =
           ViVisualizer::VisualizationData::TrackingQuality::Marginal;
     }
     visualisation2Timer.stop();
     visualisationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr,1);
   }
   visualisationTimer.stop();
 
   // apply marginalisation strategy
   bool expand = true;
   TimerSwitchable marginaliseTimer("7 Marginalise");
   estimator_.applyStrategy(
         size_t(parameters_.estimator.num_keyframes),
         size_t(parameters_.estimator.num_loop_closure_frames),
     size_t(parameters_.estimator.num_imu_frames), affectedStates_, expand);
   marginaliseTimer.stop();
 }
 
 bool ThreadedSlam::needsNewLidarKeyframe()
 {
   if(previousSubmap_){
     if(noOverlapCounter_ > 1){
       DLOG(INFO) << "Manually triggering new KF for lidar tracking!!! For lastOptimisedState.id = " << lastOptimisedState_.id.value();
       noOverlapCounter_ = 0;
       return true;
     }
   }
   return false;
 }
 
 
 // Loop to process visualisations.
 void ThreadedSlam::visualisationLoop()
 {
   while (!shutdown_) {
     ViVisualizer::VisualizationData::Ptr visualisationData;
     if(visualisationData_.PopBlocking(&visualisationData)) {
       std::vector<cv::Mat> outImages(parameters_.nCameraSystem.numCameras());
       for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
         if(parameters_.nCameraSystem.cameraType(i).isUsed
            && !visualisationData->currentFrames->image(i).empty()) {
           outImages[i] = visualizer_.drawMatches(visualisationData, i);
         }
       }
       visualisationImages_.PushNonBlockingDroppingIfFull(outImages,1);
     } else {
       return;
     }
   }
 }
 
 // Loop to process publishing.
 void ThreadedSlam::publishingLoop()
 {
   while (!shutdown_) {
     PublicationData publicationData;
     if(publicationQueue_.PopBlocking(&publicationData)) {
       if(optimisedGraphCallback_) {
         optimisedGraphCallback_(publicationData.state, publicationData.trackingState,
                                 publicationData.updatedStates, publicationData.landmarksPublish);
       }
     } else {
       return;
     }
   }
 }
 
 // trigger display (needed because OSX won't allow threaded display)
 void ThreadedSlam::display(std::map<std::string, cv::Mat> &images)
 {
 
   std::vector<cv::Mat> outImages(parameters_.nCameraSystem.numCameras());
   if(visualisationImages_.PopNonBlocking(&outImages)) {
     // draw
     for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); i++) {
       std::string name;
       if(parameters_.nCameraSystem.cameraType(i).isColour) {
         name = "rgb"+std::to_string(i);
       } else {
         name = "cam"+std::to_string(i);
       }
       if(!outImages[i].empty()) {
         images[name] = outImages[i];
       }
     }
   }
 
   // top view
   cv::Mat topDebugImg;
   if(overheadImages_.PopNonBlocking(&topDebugImg)) {
     if(!topDebugImg.empty()) {
       images["Top Debug View"] = topDebugImg;
     }
   }

  // radar polar image
  cv::Mat radarPolarImg;
  if (radarPolarImages_.PopNonBlocking(&radarPolarImg)) {
    if (!radarPolarImg.empty()) {
      images["Radar Polar Image"] = radarPolarImg;
    }
  }

  // AGCWD-preprocessed images (per camera)
  std::vector<cv::Mat> agcwdImgs(parameters_.nCameraSystem.numCameras());
  if (agcwdImages_.PopNonBlocking(&agcwdImgs)) {
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      std::string name;
      if (parameters_.nCameraSystem.cameraType(i).isColour) {
        name = "rgb" + std::to_string(i);
      } else {
        name = "cam" + std::to_string(i);
      }
      if (!agcwdImgs[i].empty()) {
        images[name + "_agcwd"] = agcwdImgs[i];
      }
    }
  }

 }
 
 void ThreadedSlam::stopThreading() {
   if (shutdown_) {
     return;
   }

   {
     const uint64_t pairs = g_betweenPairs.load();
     const uint64_t imm = g_betweenImmediate.load();
     const uint64_t bf = g_betweenBackfilled.load();
     const uint64_t rm = g_betweenRemoved.load();
     const double pct = pairs ? 100.0 * static_cast<double>(imm + bf) / static_cast<double>(pairs) : 0.0;
     LOG(INFO) << "[GaRLILEO][BetweenSummary] pairs=" << pairs
               << " immediate=" << imm
               << " backfilled=" << bf
               << " removed_by_backfill=" << rm
               << " effective=" << (imm + bf) << " (" << pct << "%)"
               << " mean_backfill_lag_frames="
               << (bf ? static_cast<double>(g_backfillLagFrames.load()) / static_cast<double>(bf) : 0.0);
   }
 
   // process all received
   while (processFrame()) {
     std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
 
   // stop CNN stuff
   frontend_.endCnnThreads();
 
   if(optimisationThread_.joinable()) {
     optimisationThread_.join(); // this should not be necessary after having called processFrame.
   }
 
   // shutdown queues
   imuMeasurementsReceived_.Shutdown();
   cameraMeasurementsReceived_.Shutdown();
   gpsMeasurementsReceived_.Shutdown();
   lidarMeasurementsReceived_.Shutdown();
  garlileoMeasurementsReceived_.Shutdown();
  garlileoGravityMeasurementsReceived_.Shutdown();
  radarTargetsMeasurementsReceived_.Shutdown();
   submapAlignmentFactorsReceived_.Shutdown();
   visualisationImages_.Shutdown();
  radarPolarImages_.Shutdown();
  agcwdImages_.Shutdown();
   visualisationData_.Shutdown();
   publicationQueue_.Shutdown();
   lastOptimisedQueue_.Shutdown();

   // Finish any in-flight publication before importing the final loop closure.
   // A callback mutex prevents races but cannot order old and final revisions.
   if(publishingThread_.joinable()) {
     publishingThread_.join();
   }
 
   // force background optimisation to finish
   if(fullGraphOptimisationThread_.joinable()) {
     LOG(INFO) << "wait for background optimisation to finish...";
     fullGraphOptimisationThread_.join();
     // import pose graph optimisation
     if (estimator_.isLoopClosureAvailable()) {
       OKVIS_ASSERT_TRUE(Exception,
                         !estimator_.isLoopClosing(),
                         "loop closure available, but still loop closing -- bug")
       LOG(INFO) << "import final background optimisation and publish...";
       TimerSwitchable synchronisationTimer("5 Import full optimisation");
       std::vector<StateId> updatedStates;
       estimator_.synchroniseRealtimeAndFullGraph(updatedStates);
       synchronisationTimer.stop();
 
       // prepare publishing
       StateId currentId = estimator_.currentStateId();
       kinematics::Transformation T_WS = estimator_.pose(currentId);
       SpeedAndBias speedAndBiases = estimator_.speedAndBias(currentId);
       State state;
       state.id = currentId;
       state.T_WS = T_WS;
       state.v_W = speedAndBiases.head<3>();
       state.b_g = speedAndBiases.segment<3>(3);
       state.b_a = speedAndBiases.tail<3>();
       state.omega_S.setZero(); // FIXME: use actual value
       state.timestamp = estimator_.timestamp(currentId);
       state.previousImuMeasurements = imuMeasurementsByFrame_.at(currentId);
       state.isKeyframe = estimator_.isKeyframe(currentId);
       TrackingState trackingState;
       trackingState.id = currentId;
       trackingState.isKeyframe = estimator_.isKeyframe(currentId);
       trackingState.recognisedPlace = estimator_.closedLoop(currentId);
       const double trackingQuality = estimator_.trackingQuality(currentId);
       const double lostThr = std::max(0.0, parameters_.garlileo.tracking_lost_quality_threshold);
       const double marginalThr = std::max(lostThr, 0.3);
       if (trackingQuality < lostThr) {
         trackingState.trackingQuality = TrackingQuality::Lost;
       } else if (trackingQuality < marginalThr) {
         trackingState.trackingQuality = TrackingQuality::Marginal;
       } else {
         trackingState.trackingQuality = TrackingQuality::Good;
       }
       trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(currentId, false);
       hasStarted_.store(true);
 
       // now publish
       if (optimisedGraphCallback_) {
         // current state & tracking info via State and Tracking State.
         PublicationData publicationData;
         publicationData.state = state;
         publicationData.trackingState = trackingState;
         publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
         for (const auto &id : updatedStates) {
           kinematics::Transformation T_WS = estimator_.pose(id);
           SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
           Time timestamp = estimator_.timestamp(id);
           ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
           Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
           for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
             if (riter->timeStamp < timestamp) {
               omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
               break;
             }
           }
           std::set<StateId> observedIds;
           estimator_.getObservedIds(id, observedIds);
           (*publicationData.updatedStates)[id] =
             State{T_WS, speedAndBias.head<3>(), speedAndBias.segment<3>(3), speedAndBias.tail<3>(),
                   omega_S, timestamp, id, imuMeasurements, estimator_.isKeyframe(id),
                   AlignedMap<uint64_t, kinematics::Transformation>(), observedIds, true,
                   false, AlignedMap<uint64_t, kinematics::Transformation>(),
                   kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
           affectedStates_.erase(id); // do not separately publish as affected state...
         }
         for (const auto &id : affectedStates_) {
           kinematics::Transformation T_WS = estimator_.pose(id);
           SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
           Time timestamp = estimator_.timestamp(id);
           ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
           Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
           for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
             if (riter->timeStamp < timestamp) {
               omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
               break;
             }
           }
           std::set<StateId> observedIds;
           estimator_.getObservedIds(id, observedIds);
           (*publicationData.updatedStates)[id]
             = State{T_WS,
                     speedAndBias.head<3>(),
                     speedAndBias.segment<3>(3),
                     speedAndBias.tail<3>(),
                     omega_S,
                     timestamp,
                     id,
                     imuMeasurements,
                     estimator_.isKeyframe(id),
                     AlignedMap<uint64_t, kinematics::Transformation>(),
                     observedIds, false,
                     false, AlignedMap<uint64_t, kinematics::Transformation>(),
                     kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
         }
         affectedStates_.clear();
 
         // landmarks:
         publicationData.landmarksPublish.reset(new MapPointVector());
         MapPoints landmarks;
         estimator_.getLandmarks(landmarks);
         publicationData.landmarksPublish->reserve(landmarks.size());
         for (const auto &lm : landmarks) {
           publicationData.landmarksPublish->push_back(
             MapPoint(lm.first.value(), lm.second.point, lm.second.quality));
         }
 
         // and publish
         optimisedGraphCallback_(publicationData.state,
                                 publicationData.trackingState,
                                 publicationData.updatedStates,
                                 publicationData.landmarksPublish);
       }
     }
   }
 
   // end remaining threads
   shutdown_ = true;
   if(visualisationThread_.joinable()) {
     visualisationThread_.join();
   }
 }
 
 void ThreadedSlam::writeFinalTrajectoryCsv()
 {
   // thread safety -- join running stuff
   stopThreading();
 
   // trajectory writing
   if(!finalTrajectoryCsvFileName_.empty()) {
     estimator_.writeFinalCsvTrajectory(finalTrajectoryCsvFileName_, rpg_);
   }
 }

void ThreadedSlam::writeEwgUtilityMinCsv(const std::string& csvFileName) {
  std::vector<EwgUtilityMinRecord> records;
  {
    std::lock_guard<std::mutex> lock(ewgUtilityMinRecordsMutex_);
    records = ewgUtilityMinRecords_;
  }
  std::ofstream os(csvFileName);
  if (!os.is_open()) {
    LOG(WARNING) << "Could not open EWG utility CSV file for writing: " << csvFileName;
    return;
  }
  os << "frame_id,timestamp_sec,ewg_utility_min,ewg_utility_cams\n";
  os << std::setprecision(17);
  for (const auto& rec : records) {
    os << rec.frame_id << "," << rec.timestamp_sec << "," << rec.utility_min << ","
       << rec.utility_cams << "\n";
  }
  LOG(INFO) << "Wrote per-frame EWG utility CSV: " << csvFileName
            << " rows=" << records.size();
}

void ThreadedSlam::writeImuBiasCsv(const std::string& csvFileName) {
  std::vector<ImuBiasRecord> records;
  {
    std::lock_guard<std::mutex> lock(imuBiasRecordsMutex_);
    records = imuBiasRecords_;
  }
  std::ofstream os(csvFileName);
  if (!os.is_open()) {
    LOG(WARNING) << "Could not open IMU bias CSV file for writing: " << csvFileName;
    return;
  }
  // Sign convention: omega_true = omega_meas - b_g, acc_true = acc_meas - b_a (see ImuError),
  // which is the same convention GaRLILEO uses, so b_a is directly comparable across the two.
  os << "frame_id,timestamp_sec,is_keyframe,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n";
  os << std::setprecision(17);
  for (const auto& rec : records) {
    os << rec.frame_id << "," << rec.timestamp_sec << "," << (rec.is_keyframe ? 1 : 0) << ","
       << rec.b_g.x() << "," << rec.b_g.y() << "," << rec.b_g.z() << ","
       << rec.b_a.x() << "," << rec.b_a.y() << "," << rec.b_a.z() << "\n";
  }
  LOG(INFO) << "Wrote per-frame IMU bias CSV: " << csvFileName << " rows=" << records.size();
}
 
 void ThreadedSlam::writeGlobalTrajectoryCsv(const std::string& csvFileName)
 {
 
   estimator_.writeGlobalCsvTrajectory(csvFileName);
 }
 
 void ThreadedSlam::doFinalBa()
 {
   // Check if there are still alignment factors to be added.
   // Map to Map Alignment Factors are added here...
   AlignmentTerm alignmentTerm;
   while(!submapAlignmentFactorsReceived_.Empty())
   {
 
     submapAlignmentFactorsReceived_.PopNonBlocking(&alignmentTerm);
     LOG(WARNING) << "Remaining Map-to-Map Contstraints are added between " << alignmentTerm.frame_A_id << " and " << alignmentTerm.frame_B_id;
     // add submap alignment constraints between map keyframes A and B
     if (parameters_.lidar) {
       estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
         alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
         alignmentTerm.sensorError, true, "Tukey");
     }
     else {
       estimator_.addSubmapAlignmentConstraints(alignmentTerm.submap_A_ptr, alignmentTerm.frame_A_id,
         alignmentTerm.frame_B_id, alignmentTerm.pointCloud_B,
         alignmentTerm.sensorError, false, "Tukey");
     }
 
     // Set reference submap for live constraints to recently finished submap
     previousSubmap_ = alignmentTerm.submap_B_ptr;
     previousSubmapId_ = alignmentTerm.frame_B_id;
   }
 
   // thread safety -- join running stuff
   stopThreading();
 
   // now call it
   // A launch may be stopped after warmup but before its first input frame.
   // Inspect the graph only after joining workers; final BA and publication
   // both require an existing state.
   if (estimator_.numFrames() == 0) {
     LOG(INFO) << "Skipping final BA: no input frames were processed.";
     return;
   }

   const int numThreads = parameters_.estimator.realtime_num_threads
       +parameters_.estimator.full_graph_num_threads;
   const bool do_extrinsics_final_ba =
       parameters_.camera.online_calibration.do_extrinsics_final_ba;
   const double sigma_r = do_extrinsics_final_ba ?
                          parameters_.camera.online_calibration.sigma_r_final_ba :
                          0.0;
   const double sigma_alpha = do_extrinsics_final_ba ?
                              parameters_.camera.online_calibration.sigma_alpha_final_ba :
                              0.0;
   std::set<StateId> updatedStatesBa;
   LOG(INFO) << "[FinalBA] max_iterations=" << parameters_.estimator.final_ba_max_iterations
             << " function_tolerance=" << parameters_.estimator.final_ba_function_tolerance;
   estimator_.doFinalBa(parameters_.estimator.final_ba_max_iterations,
       posegraphOptimisationSummary_, updatedStatesBa,
       sigma_r, sigma_alpha, numThreads, true,
       parameters_.estimator.final_ba_function_tolerance);
 
   // prepare publishing
   StateId currentId = estimator_.currentStateId();
   kinematics::Transformation T_WS = estimator_.pose(currentId);
   SpeedAndBias speedAndBiases = estimator_.speedAndBias(currentId);
   State state;
   state.id = currentId;
   state.T_WS = T_WS;
   state.v_W = speedAndBiases.head<3>();
   state.b_g = speedAndBiases.segment<3>(3);
   state.b_a = speedAndBiases.tail<3>();
   state.omega_S.setZero(); // FIXME: use actual value
   state.timestamp = estimator_.timestamp(currentId);
   state.previousImuMeasurements = imuMeasurementsByFrame_.at(currentId);
   state.isKeyframe = estimator_.isKeyframe(currentId);
   TrackingState trackingState;
   trackingState.id = currentId;
   trackingState.isKeyframe = estimator_.isKeyframe(currentId);
   trackingState.recognisedPlace = estimator_.closedLoop(currentId);
   const double trackingQuality = estimator_.trackingQuality(currentId);
   const double lostThr = std::max(0.0, parameters_.garlileo.tracking_lost_quality_threshold);
   const double marginalThr = std::max(lostThr, 0.3);
   if(trackingQuality < lostThr) {
     trackingState.trackingQuality = TrackingQuality::Lost;
   } else if (trackingQuality < marginalThr){
     trackingState.trackingQuality = TrackingQuality::Marginal;
   } else {
     trackingState.trackingQuality = TrackingQuality::Good;
   }
   trackingState.currentKeyframeId = estimator_.mostOverlappedStateId(currentId, false);
   hasStarted_.store(true);
 
   // now publish
   if(optimisedGraphCallback_) {
     // current state & tracking info via State and Tracking State.
     std::vector<StateId> updatedStateIds;
     PublicationData publicationData;
     publicationData.state = state;
     publicationData.trackingState = trackingState;
     publicationData.updatedStates.reset(new AlignedMap<StateId, State>());
     for(const auto & id : updatedStatesBa) {
       kinematics::Transformation T_WS = estimator_.pose(id);
       SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
       Time timestamp = estimator_.timestamp(id);
       ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
       Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
       for(auto riter = imuMeasurements.rbegin(); riter!=imuMeasurements.rend(); ++riter) {
         if(riter->timeStamp < timestamp) {
           omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
           break;
         }
       }
       std::set<StateId> observedIds;
       estimator_.getObservedIds(id, observedIds);
       (*publicationData.updatedStates)[id] =
           State{T_WS, speedAndBias.head<3>(),speedAndBias.segment<3>(3),
           speedAndBias.tail<3>(),omega_S, timestamp, id, imuMeasurements,
           estimator_.isKeyframe(id),
           AlignedMap<uint64_t, kinematics::Transformation>(), observedIds, true,
           false, AlignedMap<uint64_t, kinematics::Transformation>(),
           kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
     }
     for (const auto &id : affectedStates_) {
       kinematics::Transformation T_WS = estimator_.pose(id);
       SpeedAndBias speedAndBias = estimator_.speedAndBias(id);
       Time timestamp = estimator_.timestamp(id);
       ImuMeasurementDeque imuMeasurements = imuMeasurementsByFrame_.at(id);
       Eigen::Vector3d omega_S(0.0, 0.0, 0.0); // get this for real now:
       for (auto riter = imuMeasurements.rbegin(); riter != imuMeasurements.rend(); ++riter) {
         if (riter->timeStamp < timestamp) {
           omega_S = riter->measurement.gyroscopes - speedAndBias.segment<3>(3);
           break;
         }
       }
       std::set<StateId> observedIds;
       estimator_.getObservedIds(id, observedIds);
       (*publicationData.updatedStates)[id]
         = State{T_WS,
                 speedAndBias.head<3>(),
                 speedAndBias.segment<3>(3),
                 speedAndBias.tail<3>(),
                 omega_S,
                 timestamp,
                 id,
                 imuMeasurements,
                 estimator_.isKeyframe(id),
                 AlignedMap<uint64_t, kinematics::Transformation>(),
                 observedIds, false,
                 false, AlignedMap<uint64_t, kinematics::Transformation>(),
                 kinematics::Transformation::Identity(), AlignedVector<Eigen::Vector3d>()};
     }
     affectedStates_.clear();
 
     // landmarks:
     publicationData.landmarksPublish.reset(new MapPointVector());
     MapPoints landmarks;
     estimator_.getLandmarks(landmarks);
     publicationData.landmarksPublish->reserve(landmarks.size());
     for(const auto & lm : landmarks) {
       publicationData.landmarksPublish->push_back(
             MapPoint(lm.first.value(), lm.second.point, lm.second.quality));
     }
 
     // and publish
     optimisedGraphCallback_(publicationData.state, publicationData.trackingState,
                             publicationData.updatedStates, publicationData.landmarksPublish);
   }
 }
 
 bool ThreadedSlam::saveMap() {
   // thread safety -- join running stuff
   stopThreading();
 
   // now call it
   if(!finalTrajectoryCsvFileName_.empty()) {
     return estimator_.saveMap(mapCsvFileName_);
   }
   return false;
 }
 
 void ThreadedSlam::dumpGpsResiduals(const std::string &gpsResCsvFileName)
 {
   // thread safety -- join running stuff
   stopThreading();
 
   // call residual writer on graph
   estimator_.dumpGpsResiduals(gpsResCsvFileName);
 
 
 }
 
 void ThreadedSlam::computeLiveDepthMeasurements(
   const kinematics::Transformation& T_WS_live,
   const okvis::Time& curTime,
   std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& livePoints,
   std::vector<float>& liveSigmas) {
 
   livePoints.clear();
   liveSigmas.clear();
 
   if (previousSubmap_ && !depthMeasurementDeque_.empty()) {
     // {M}: previous submap pose
     // {D}: depth camera pose in multiFrame
     const kinematics::Transformation T_WD = T_WS_live * T_SD_;
     const kinematics::Transformation T_WM = estimator_.pose(StateId(previousSubmapId_));
     const kinematics::Transformation T_MD = T_WM.inverse() * T_WD;
 
     for (size_t i = 0; i < depthMeasurementDeque_.size(); i ++) {
       CameraMeasurement liveDepth = depthMeasurementDeque_[i];
 
       // Precompute point in depth camera frame
       cv::Mat lowDepth, lowSigma;
       cv::resize(liveDepth.measurement.depthImage, lowDepth, cv::Size(),
         1.0 / submapConfig_.depthImageResDownsampling, 
         1.0 / submapConfig_.depthImageResDownsampling, cv::INTER_NEAREST);
 
       if (submapConfig_.useUncertainty && !liveDepth.measurement.sigmaImage.empty()) {
         cv::resize(liveDepth.measurement.sigmaImage, lowSigma, cv::Size(), 
           1.0 / submapConfig_.depthImageResDownsampling,
           1.0 / submapConfig_.depthImageResDownsampling, cv::INTER_NEAREST);
       }
 
       int depthWidth = lowDepth.cols;
       int depthHeight = lowDepth.rows;
 
       // T_delta = T_{D}{D_meas} where the latter is depth frame when measurement actually happens.
       kinematics::Transformation T_delta(Eigen::Matrix4d::Identity());
       if (depthMeasurementDeque_[i].timeStamp != curTime) {
         // Pose interpolation 
         okvis::State stateMeas;
         if (!trajectory_.getState(depthMeasurementDeque_[i].timeStamp, stateMeas)) {
           okvis::State latestTrajState;
           const std::set<StateId> allIds = trajectory_.stateIds();
           if (!allIds.empty()) {
             StateId latestTrajId = *allIds.rbegin(); // get the most recently added id
             trajectory_.getState(latestTrajId, latestTrajState);
             double deltaTime = (curTime-latestTrajState.timestamp).toSec();
             double deltaTime0 = (depthMeasurementDeque_[i].timeStamp-latestTrajState.timestamp).toSec();
             float interpWeight = deltaTime0/deltaTime;
             if (interpWeight > 0 && interpWeight < 1) {
               Eigen::Vector3d interpPos = (1-interpWeight)*latestTrajState.T_WS.r() + interpWeight*T_WS_live.r();
               Eigen::Quaterniond interpQuat = latestTrajState.T_WS.q().slerp(interpWeight, T_WS_live.q());
               okvis::kinematics::Transformation T_interp(interpPos, interpQuat);
               stateMeas.T_WS = T_interp;
             }
           }
         }
         T_delta = T_WD.inverse() * (stateMeas.T_WS * T_SD_);
       }
 
       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPoints;
       std::vector<float> observedSigmas;
 
       // Uniformly sample depths and sigmas
       for (int x = 0; x < depthWidth; x+=10) {
         for (int y = 0; y < depthHeight; y+=10) {
           Eigen::Vector2f imagePoint_xy(static_cast<float>(x), static_cast<float>(y));
           Eigen::Vector3f cameraPoint;
           depthCamera_->model.backProject(imagePoint_xy, &cameraPoint);
 
           float pointZ = lowDepth.at<float>(y,x);
           float sigmaZ = submapConfig_.sensorError;
           if (submapConfig_.useUncertainty && !lowSigma.empty()) {
             sigmaZ = lowSigma.at<float>(y,x);
           }
 
           if (pointZ < submapConfig_.near_plane || pointZ > submapConfig_.far_plane || sigmaZ > 3.0) continue;
           
           cameraPoint = cameraPoint * pointZ;
 
           Eigen::Vector3f p_M = (T_MD*T_delta).T3x4().cast<float>() * cameraPoint.homogeneous();
           std::optional<float> occ = okvis::interpFieldMeanOccup<se::Safe::On>(*previousSubmap_, p_M);
           std::optional<Eigen::Vector3f> gradf = okvis::gradFieldMeanOccup<se::Safe::On>(*previousSubmap_, p_M);
           const float occ_margin = 2.0f;
           if (occ && gradf) {
             if (occ.value() > -occ_margin && occ.value() < occ_margin && gradf.value().norm() >= 1e-03) {
               Eigen::Vector3f p_S = (T_SD_*T_delta).T3x4().cast<float>() * cameraPoint.homogeneous();
               observedPoints.push_back(p_S);
               observedSigmas.push_back(sigmaZ);
             }
           } 
         }
       }
       // Downample points based on uncertainty
       if (observedPoints.size() > submapConfig_.numSubmapFactors) {
         okvis::downsamplePointsUncertainty(observedPoints, observedSigmas, 
                                     livePoints, liveSigmas,
                                     submapConfig_.numSubmapFactors);
       } 
       else {
         livePoints = observedPoints;
         liveSigmas = observedSigmas;
       }     
     }
   }
 }
 
 void ThreadedSlam::saveAlignedPoints(size_t id,
                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points,
                                      std::string type) {
   kinematics::Transformation T_WS = estimator_.pose(StateId(id));
   size_t Npoints = points.size();
   pcl::PointCloud<pcl::PointXYZ> cloud;
   cloud.width = Npoints;
   cloud.height = 1;
   cloud.is_dense = false;
   cloud.resize (cloud.width * cloud.height);
   size_t tmpCnt = 0;
   for (auto& point: cloud) {
     Eigen::Vector3d point_S = points[tmpCnt].cast<double>();
     Eigen::Vector3d point_W = (T_WS.T() * point_S.homogeneous()).head<3>();
     point.x = point_W(0);
     point.y = point_W(1);
     point.z = point_W(2);
     tmpCnt ++;
   }
   
   std::string mesh_numbering;
   if (id < 10) {
     mesh_numbering = "00000" + std::to_string(id);
   }
   else if (id < 100) {
     mesh_numbering = "0000" + std::to_string(id);
   }
   else if (id < 1000) {
     mesh_numbering = "000" + std::to_string(id);
   }
   else if (id < 10000) {
     mesh_numbering = "00" + std::to_string(id);
   }
   else if (id < 100000) {
     mesh_numbering = "0" + std::to_string(id);
   }
   else if (id < 1000000) {
     mesh_numbering = std::to_string(id);
   }
   
   std::string saveName = finalTrajectoryCsvFileName_.substr(0, finalTrajectoryCsvFileName_.find("okvis2"))
     + type + '_' + mesh_numbering + ".ply";
   pcl::io::savePLYFileASCII (saveName, cloud);
 }
 
 void ThreadedSlam::writeDebugStatisticsCsv(const std::string& csvFilePrefix)
 {
 
   estimator_.writeLidarDebugStatisticsCsv(csvFilePrefix);
  
 }
 
 }  // namespace okvis
