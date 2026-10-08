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
 * @file Frontend.hpp
 * @brief Header file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_FRONTEND_HPP_
#define INCLUDE_OKVIS_FRONTEND_HPP_

#include <mutex>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>

#include <okvis/Component.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Time.hpp>
#include <okvis/ViFrontendInterface.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include <thread>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief A frontend using BRISK features
 */
class Frontend : public ViFrontendInterface {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   * @param dBowVocDir The directory containing the DBoW vocabulary.
   */
  Frontend(size_t numCameras, std::string dBowVocDir);
  virtual ~Frontend() override;

  /**
   * @brief Load another, previously saved VI-SLAM component.
   * @param filename Filename.
   * @param imuParameters Imu parameters of the loaded component.
   * @param nCameraSystem Multi-camera configuration of the loaded component.
   * @param componentFixed Should the component be treated as fixed?
   * @return True on success.
   */
  bool loadComponent(std::string filename,
                     const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     bool componentFixed = true);

  ///@{
  /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @remark This method is threadsafe.
   * @param cameraIndex Index of camera to do detection and description.
   * @param frameOut    Multiframe containing the frames.
   *                    Resulting keypoints and descriptors are saved in here.
   * @param T_WC        Pose of camera with index cameraIndex at image capture time.
   * @param[in] keypoints If the keypoints are already available from a different source, provide
   *                      them here in order to skip detection.
   * @warning Using keypoints from a different source is not yet implemented.
   * @return True if successful.
   */
  virtual bool detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) override final;

  /**
   * @brief Set an external gravity direction expressed in the IMU/sensor frame S (unit vector).
   *
   * If set, BRISK extraction direction will use this gravity direction (mapped into the camera
   * frame via T_SC) instead of assuming gravity == world -Z.
   *
   * Thread-safe.
   */
  void setExternalGravityInSensorFrame(const Eigen::Vector3d& g_S_unit);

  /// @brief Disable the external gravity override (fall back to world -Z gravity).
  void clearExternalGravity();

  /// @brief Set external ego speed magnitude [m/s] (e.g. from GaRLILEO) for keyframe gating.
  void setExternalEgoSpeed(double speed_mps);

  /// @brief Clear external ego speed (disables speed-based keyframe gating for this frame).
  void clearExternalEgoSpeed();

  /// @brief Provide external radar targets (in radar frame R) for the current frame (place recognition).
  void setExternalRadarTargets(const okvis::Time& stamp,
                               const okvis::RadarTargetReadings& targets);

  /// @brief Clear the external radar targets.
  void clearExternalRadarTargets();

  /// @brief Enable/disable AGCWD image preprocessing (contrast enhancement) before feature extraction.
  /// @param enable If true: apply AGCWD before detect/describe.
  /// @param weighting_param AGCWD weighting parameter (typically in (0,1]).
  void setAgcwdPreprocessing(bool enable, double weighting_param);

  /// @brief Enable/disable CLAHE preprocessing before feature extraction.
  /// @param enable If true: apply CLAHE before detect/describe.
  /// @param clip_limit CLAHE clip limit.
  /// @param tile_grid_size CLAHE tile grid size (NxN).
  void setClahePreprocessing(bool enable, double clip_limit, int tile_grid_size);

  /// @brief Configure the EWG (Entropy-Weighted Gradient) image utility metric.
  /// @param Hthres Saturation threshold (paper Eq. 4 default 0.05).
  /// @param alpha Activation function steepness (paper Eq. 3).
  /// @param tau Activation function bias (paper Eq. 3).
  /// @param local_patch Local entropy patch size in pixels (paper default 5).
  /// @param num_bins Histogram quantization bins (default 8 = 256/32).
  /// @param lambda_f SFA-EWG feature bonus weight `lambda_f` (default 1.0).
  /// @param N0 Feature count scale N0 for Q_f (BRISK keypoints; default 500).
  /// @param grid_rows Grid rows for spatial uniformity Q_s (default 3).
  /// @param grid_cols Grid cols for spatial uniformity Q_s (default 4).
  /// SFA-EWG uses the same BRISK scale-space detector settings as
  /// initialiseBriskFeatureDetectors() (not configured here).
  /// @param ewg_resize_enable If true, downsample raw image before SFA-EWG (see max_side).
  /// @param ewg_resize_max_side Longest side limit [px] when resize is enabled.
  /// @param fallback_use_preprocessed If true, cache fallback utility from preprocessed image.
  void setEwgImageUtilityConfig(double Hthres,
                                double alpha,
                                double tau,
                                int local_patch,
                                int num_bins,
                                double lambda_f = 1.0,
                                double N0 = 500.0,
                                int grid_rows = 3,
                                int grid_cols = 4,
                                bool ewg_resize_enable = false,
                                int ewg_resize_max_side = 640,
                                bool fallback_use_preprocessed = true);

  /// @brief Get per-camera SFA-EWG utilities for a multiframe id.
  bool getImageUtility(uint64_t frame_id, std::vector<double>& utilities) const;

  /// @brief Discard a previous provisional entry before detecting a new multiframe.
  /// Call once before any camera detection threads start, never between cameras.
  void prepareImageUtilityFrameId(uint64_t frame_id);

  /// @brief Preserve detected utilities when the estimator assigns the multiframe id.
  /// Call after camera detection threads have joined and addStates() has succeeded.
  void reassignImageUtilityFrameId(uint64_t previous_frame_id, uint64_t frame_id);

  /// @brief Get the sum of per-camera SFA-EWG utilities for a frame (NaNs skipped).
  /// @param frame_id        MultiFrame id.
  /// @param[out] total      Sum of per-camera utilities (NaN entries skipped).
  /// @param[out] num_recorded Number of cameras with a recorded utility.
  /// @return True if at least one camera has a recorded utility for this frame.
  bool getImageUtilitySum(uint64_t frame_id, double& total, int& num_recorded) const;

  /// @brief Enable/disable radial vignette correction before enhancement.
  /// @param enable If true: apply vignette correction.
  /// @param k1 Polynomial coefficient for r^2.
  /// @param k2 Polynomial coefficient for r^4.
  /// @param k3 Polynomial coefficient for r^6.
  void setVignetteCorrection(bool enable, double k1, double k2, double k3);

  /// @brief Configure trigger-based Bayesian optimization for AGCWD/CLAHE parameters.
  /// @param enable If true: use BO instead of fixed agcwd_weighting_param/clahe_clip_limit.
  /// @param utility_discrepancy_threshold Re-optimize when
  ///        |U_real,t - U_real,t-1| / |U_real,t-1| exceeds this.
  /// @param initial_samples Number of random warmup samples.
  /// @param iterations Number of BO iterations.
  /// @param candidate_grid Grid size per dimension for EI candidate search.
  /// @param agcwd_min Lower bound for AGCWD weighting.
  /// @param agcwd_max Upper bound for AGCWD weighting.
  /// @param clahe_min Lower bound for CLAHE clip limit.
  /// @param clahe_max Upper bound for CLAHE clip limit.
  void setAgcwdClaheBayesOptConfig(bool enable,
                                   double utility_discrepancy_threshold,
                                   int initial_samples,
                                   int iterations,
                                   int candidate_grid,
                                   double agcwd_min,
                                   double agcwd_max,
                                   double clahe_min,
                                   double clahe_max,
                                   double image_utility_weight);

  /// @brief Get the latest preprocessed image for a camera (for RViz/debug display).
  /// Thread-safe. Returns false if no image is available.
  bool getLastAgcwdImage(size_t cameraIndex, cv::Mat& outImage) const;

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in another thread.
   * @param estimator       Estimator.
   * @param params          Configuration parameters.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param kfPrior         A prior to trigger new keyframe from other criteria (e.g. LiDAR overlap)
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const okvis::ViParameters & params,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) override final;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see okvis::ceres::ImuError::propagation()
   * @remark This method is threadsafe.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const override final;

  ///@}
  /// @name Getters related to the BRISK detector
  /// @{

  /// @brief Get the number of octaves of the BRISK detector.
  size_t getBriskDetectionOctaves() const {
    return briskDetectionOctaves_;
  }

  /// @brief Get the detection threshold of the BRISK detector.
  double getBriskDetectionThreshold() const {
    return briskDetectionThreshold_;
  }

  /// @brief Get the absolute threshold of the BRISK detector.
  double getBriskDetectionAbsoluteThreshold() const {
    return briskDetectionAbsoluteThreshold_;
  }

  /// @brief Get the maximum amount of keypoints of the BRISK detector.
  size_t getBriskDetectionMaximumKeypoints() const {
    return briskDetectionMaximumKeypoints_;
  }

  ///@}
  /// @name Getters related to the BRISK descriptor
  /// @{

  /// @brief Get the rotation invariance setting of the BRISK descriptor.
  bool getBriskDescriptionRotationInvariance() const {
    return briskDescriptionRotationInvariance_;
  }

  /// @brief Get the scale invariance setting of the BRISK descriptor.
  bool getBriskDescriptionScaleInvariance() const {
    return briskDescriptionScaleInvariance_;
  }

  ///@}
  /// @name Other getters
  /// @{

  /// @brief Get the matching threshold.
  double getBriskMatchingThreshold() const {
    return briskMatchingThreshold_;
  }

  /// @brief Get the area overlap threshold under which a new keyframe is inserted.
  float getKeyframeInsertionOverlapThershold() const {
    return keyframeInsertionOverlapThreshold_;
  }

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() {
    return isInitialized_;
  }

  /// @}
  /// @name Setters related to the BRISK detector
  /// @{

  /// @brief Set the number of octaves of the BRISK detector.
  void setBriskDetectionOctaves(size_t octaves) {
    briskDetectionOctaves_ = octaves;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the detection threshold of the BRISK detector.
  void setBriskDetectionThreshold(double threshold) {
    briskDetectionThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the absolute threshold of the BRISK detector.
  void setBriskDetectionAbsoluteThreshold(double threshold) {
    briskDetectionAbsoluteThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the maximum number of keypoints of the BRISK detector.
  void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
    briskDetectionMaximumKeypoints_ = maxKeypoints;
    initialiseBriskFeatureDetectors();
  }

  /// @}
  /// @name Setters related to the BRISK descriptor
  /// @{

  /// @brief Set the rotation invariance setting of the BRISK descriptor.
  void setBriskDescriptionRotationInvariance(bool invariance) {
    briskDescriptionRotationInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the scale invariance setting of the BRISK descriptor.
  void setBriskDescriptionScaleInvariance(bool invariance) {
    briskDescriptionScaleInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  ///@}
  /// @name Other setters
  /// @{

  /// @brief Set the matching threshold.
  void setBriskMatchingThreshold(double threshold) {
    briskMatchingThreshold_ = threshold;
  }

  /// @brief Set the area overlap threshold under which a new keyframe is inserted.
  void setKeyframeInsertionOverlapThreshold(float threshold) {
    keyframeInsertionOverlapThreshold_ = threshold;
  }

  /// @}

  /// \brief Stop all CNN background threads.
  void endCnnThreads();

  /// \brief Clears and resets everything (so you can re-start).
  void clear();

private:

  /**
   * @brief   feature detectors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the detector.
   */
  std::vector<std::shared_ptr<cv::FeatureDetector> > featureDetectors_;
  /**
   * @brief   feature descriptors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the descriptor.
   */
  std::vector<std::shared_ptr<cv::DescriptorExtractor> > descriptorExtractors_;
  /// Mutexes for feature detectors and descriptors.
  std::vector<std::unique_ptr<std::mutex> > featureDetectorMutexes_;

  // External gravity override (expressed in IMU/sensor frame S, unit vector). Used for BRISK
  // extraction direction. Guarded for thread-safety (detectAndDescribe can run in parallel).
  mutable std::mutex externalGravityMutex_;
  bool hasExternalGravity_{false};
  Eigen::Vector3d externalGravity_S_{0.0, 0.0, -1.0};

  // External ego speed (magnitude) for speed-based keyframe gating.
  mutable std::mutex externalEgoSpeedMutex_;
  bool hasExternalEgoSpeed_{false};
  double externalEgoSpeed_{0.0};

  // External radar targets (radar frame) for the current frame (place recognition).
  mutable std::mutex externalRadarTargetsMutex_;
  bool hasExternalRadarTargets_{false};
  okvis::Time externalRadarTargetsStamp_{0.0};
  okvis::RadarTargetReadings externalRadarTargets_;

  // Optional: preprocessing parameters (vignette + AGCWD/CLAHE/paper mode).
  mutable std::mutex agcwdMutex_;
  bool agcwdEnable_{false};
  double agcwdWeightingParam_{0.5};
  bool claheEnable_{false};
  double claheClipLimit_{2.0};
  int claheTileGridSize_{8};
  bool vignetteCorrectionEnable_{false};
  double vignetteCorrectionK1_{0.0};
  double vignetteCorrectionK2_{0.0};
  double vignetteCorrectionK3_{0.0};
  bool agcwdClaheBoEnable_{false};
  double agcwdClaheBoUtilityDiscrepancyThreshold_{0.2};
  int agcwdClaheBoInitialSamples_{4};
  int agcwdClaheBoIterations_{6};
  int agcwdClaheBoCandidateGrid_{9};
  double agcwdClaheBoAgcwdMin_{0.2};
  double agcwdClaheBoAgcwdMax_{1.2};
  double agcwdClaheBoClaheMin_{1.0};
  double agcwdClaheBoClaheMax_{8.0};
  double agcwdClaheBoImageUtilityWeight_{1.0};
  bool agcwdClaheBoHasPrevRealUtility_{false};
  double agcwdClaheBoPrevRealUtility_{0.0};
  bool agcwdClaheBoCurrentInitialized_{false};
  double agcwdClaheBoCurrentAgcwd_{0.5};
  double agcwdClaheBoCurrentClahe_{2.0};
  std::thread agcwdClaheBoThread_;
  bool agcwdClaheBoRunning_{false};
  bool agcwdClaheBoStop_{false};
  uint64_t agcwdClaheBoJobSeq_{0};

  // EWG (Entropy-Weighted Gradient) image utility configuration.
  // Guarded by agcwdMutex_ for consistency with the other enhancement params.
  double ewgHthres_{0.125};
  double ewgAlpha_{32.0};
  double ewgTau_{4.0};
  int    ewgLocalPatch_{5};
  int    ewgNumBins_{8};
  double ewgLambdaF_{1.0};
  double ewgN0_{500.0};
  int    ewgGridRows_{3};
  int    ewgGridCols_{4};
  bool   ewgResizeEnable_{false};
  int    ewgResizeMaxSide_{640};
  bool   ewgFallbackUsePreprocessed_{true};

  // Per-frame, per-camera SFA-EWG utility cache (frame_id -> utility per camera).
  // Used by ThreadedSlam to compute the (low_features) signal from the image
  // utility instead of the keypoint count. Pruned periodically.
  void pruneImageUtilityCacheLocked(uint64_t frame_id); ///< Caller holds imageUtilityMutex_.
  mutable std::mutex imageUtilityMutex_;
  std::unordered_map<uint64_t, std::vector<double>> imageUtilityByFrameId_;

  // Latest preprocessed images for debug/visualisation (one per camera).
  // Note: detectAndDescribe may run in parallel for different cameras.
  mutable std::mutex agcwdImagesMutex_;
  std::vector<cv::Mat> agcwdImages_;

  // Stored radar place-recognition descriptors per keyframe (fixed 40x40 polar grid => 1600 floats).
  mutable std::mutex radarPrDescriptorsMutex_;
  std::unordered_map<uint64_t, std::array<float, 1600>> radarPrDescriptors_;
  // Stored aggregated radar targets per keyframe (body frame) for fallback loop verification.
  mutable std::mutex radarPrTargetsMutex_;
  std::unordered_map<uint64_t, okvis::RadarTargetReadings> radarPrTargetsByKeyframe_;

  bool isInitialized_;        ///< Is the pose initialised?
  const size_t numCameras_;   ///< Number of cameras in the configuration.

  /// @name BRISK detection parameters
  /// @{

  size_t briskDetectionOctaves_;            ///< The set number of brisk octaves.
  double briskDetectionThreshold_;          ///< The set BRISK detection threshold.
  double briskDetectionAbsoluteThreshold_;  ///< The set BRISK absolute detection threshold.
  size_t briskDetectionMaximumKeypoints_;   ///< The set maximum number of keypoints.

  /// @}
  /// @name BRISK descriptor extractor parameters
  /// @{

  bool briskDescriptionRotationInvariance_; ///< The set rotation invariance setting.
  bool briskDescriptionScaleInvariance_;    ///< The set scale invariance setting.

  ///@}
  /// @name BRISK matching parameters
  ///@{

  double briskMatchingThreshold_; ///< The set BRISK matching threshold.

  ///@}

  /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing
   *        landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionOverlapThreshold_;  //0.55

  /**
   * @brief Decision whether a new frame should be keyframe or not, based on overlap heuristic.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(const Estimator& estimator,
                            std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Match a new multiframe to existing keyframes
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe
   * @param      estimator              Estimator.
   * @param[in]  params                 Parameter struct.
   * @param[in]  currentFrameId         ID of the current frame that should be matched against
   *                                    keyframes.
   * @param[in]  loopClosureLandmarksToUseExclusively Use these landmarks exclusively, if supplied.
   * @return The number of matches in total.
   */
  template<class CAMERA_GEOMETRY>
  int matchToMap(Estimator& estimator,
                 const okvis::ViParameters& params,
                 const uint64_t currentFrameId,
                 const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively = nullptr);

  /**
   * @brief Match the frames inside the multiframe to each other to initialise new landmarks.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param multiFrame Multiframe containing the frames to match.
   * @param params The VI parameters.
   * @param asKeyframe Whether this is a keyframe.
   */
  template<class CAMERA_GEOMETRY>
  void matchStereo(Estimator& estimator,
                   std::shared_ptr<okvis::MultiFrame> multiFrame,
                   const okvis::ViParameters& params,
                   bool asKeyframe);

  /// \brief DBoW for loop closure
  /// https://en.cppreference.com/w/cpp/language/pimpl
  class DBoW;
  std::unique_ptr<DBoW> dBow_; ///< DBoW object (PIMPL).

  /**
   * @brief Get filtered DBoW query.
   * @param[in] dBow DBoW database to use.
   * @param[in] features Features to match against DBoW.
   * @param[out] stateIds Resulting matching (keyframe) pose IDs with corresponding scores.
   * @return Number of matching keyframes.
   */
  int getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                            const std::vector<std::vector<uchar>> &features,
                            std::vector<std::pair<StateId, double>> &stateIds) const;

  /**
   * @brief Verifies a recognised place with 3D2D matching, ransac, and nonlinear pose refinement.
   * @param[in] estimator Estimator.
   * @param[in] params The VI parameters.
   * @param[in] framesInOut Current multiframe.
   * @param[in] oldFrame Old multiframe to match against.
   * @param[out] T_Sold_Snew The relative pose found.
   * @param[out] H Information matrix corresponding to T_Sold_Snew.
   * @param[in] descriptorDistinctivenessAvgThreshold Reject if avg < threshold and RANSAC inliers < 20;
   *            use <0 to skip this avg check. Default 182 matches legacy loop-closure strictness.
   * @param[in] minInliers Minimum number of inliers required.
   */
  bool verifyRecognisedPlace(const Estimator &estimator,
                             const okvis::ViParameters &params,
                             const std::shared_ptr<const MultiFrame> framesInOut,
                             const std::shared_ptr<const MultiFrame> oldFrame,
                             kinematics::Transformation &T_Sold_Snew,
                             Eigen::Matrix<double, 6, 6>& H,
                             double descriptorDistinctivenessAvgThreshold = 182.0,
                             int minInliers = 10);

  /**
   * @brief Perform 3D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @param initializePose  Initialize the pose from RANSAC?
   * @param removeOutliers  Remove observation of outliers in estimator.
   * @return True on success.
   */
  bool runRansac3d2d(Estimator &estimator,
                    const okvis::cameras::NCameraSystem &nCameraSystem,
                    std::shared_ptr<okvis::MultiFrame> currentFrame,
                    bool initializePose,
                    bool removeOutliers);
  /**
   * @brief Remove outliers on current frame.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @return Number of inliers.
   */
  template <class CAMERA_GEOMETRY>
  int removeOutliers(Estimator &estimator,
                     const okvis::cameras::NCameraSystem &nCameraSystem,
                     std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Perform 2D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator         Estimator.
   * @param params            Parameter struct.
   * @param currentFrameId    ID of the new multiframe containing matches with the frame with ID
   *                          olderFrameId.
   * @param olderFrameId      ID of the multiframe to which the current frame has been matched
   *                          against.
   * @param initializePose    If the pose has not yet been initialised should the function try to
   *                          initialise it.
   * @param removeOutliers    Remove observation of outliers in estimator.
   * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
   * @return Number of inliers.
   */
  int runRansac2d2d(Estimator& estimator,
                    const okvis::ViParameters& params, uint64_t currentFrameId,
                    uint64_t olderFrameId, bool initializePose,
                    bool removeOutliers, bool &rotationOnly);

  /// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or
  /// at startup.
  void initialiseBriskFeatureDetectors();

  /// \brief Classification network for keypoints (if enabled).
  std::vector<std::shared_ptr<Network>> networks_;

  /**
   * @brief Match the frames to older, co-visible frames and triangulate.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param[out] rotationOnly Result obtained matches with rotation-only RANSAC.
   */
  template<class CAMERA_GEOMETRY>
  int matchMotionStereo(Estimator &estimator, const okvis::ViParameters &params,
                        const uint64_t currentFrameId, bool &rotationOnly);

  /// \brief Helper struct (internally) for landmarks to match.
  struct LandmarkToMatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p_W; ///< 3D point in World coordinates.
    cv::Mat descriptors; ///< All its descriptors.
    std::vector<KeypointIdentifier> kids; ///< All its observations.
    Eigen::Matrix3Xd e_W; ///< All directions in W-coords of the observations.
    Eigen::Matrix3Xd r_W; ///< Image centres of all the observations (W-coords).
    bool is3d = false; ///< Determine whether treated as 3D initialised.
    Eigen::Vector2d projection; ///< 2D projection location in pixels.
    bool ignore = false; ///< Ignore if classified as sky / person.
  };

  /**
   * @brief Parallelisable sub-part of matchToMap -- proper 3D points..
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   * @param[out] reprErrs Reprojection errors (by im).
   */
  template<class CAMERA_GEOMETRY>
  void matchToMapByThread(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs,
      std::vector<double>& reprErrs) const;

  /**
   * @brief Parallelisable sub-part of matchToMap -- unitialised points.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   */
  template<class CAMERA_GEOMETRY>
  void matchToMapByThreadUnitialised(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs) const;

  std::atomic_bool trackingLost_; ///< Is the tracking currently lost?

  /// \brief Hacky: remember which CNNs are running on what frame.
  std::map<StateId,std::vector<std::thread*>> cnnThreads_;

  AlignedVector<Component> components_; ///< Loaded other components.
  std::vector<std::unique_ptr<DBoW>> componentDBows_; ///< Corresponding DBoWs for place recogn.
  std::vector<bool> componentsFixed_; ///< Which ones of the other comp's are to be treated fixed.
};

}  // namespace okvis

#endif // INCLUDE_OKVIS_FRONTEND_HPP_
