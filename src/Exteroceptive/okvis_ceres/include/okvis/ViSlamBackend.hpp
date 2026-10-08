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
 * @file okvis/ViSlamBackend.hpp
 * @brief Header file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_VISLAMBACKEND_HPP_
#define INCLUDE_OKVIS_VISLAMBACKEND_HPP_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <okvis/ViGraphEstimator.hpp>
#include <okvis/ceres/RealtimeTrustRegionRadius.hpp>
#include "se/map/map.hpp"

#include <Eigen/StdVector>

/// \brief okvis Main namespace of this package.
namespace okvis {

//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body
 C: Camera
 S: Sensor (IMU)
 */
class ViSlamBackend //: public VioBackendInterface
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  ViSlamBackend() {
    needsFullGraphOptimisation_ = false;
    isLoopClosing_ = false;
    isLoopClosureAvailable_ = false;
    components_.resize(1);
    gpsObservability_ = false;
  }

  /**
   * @brief Destructor, does nothing.
   */
  virtual ~ViSlamBackend() {}

  /// \brief Helper function to check all observed landmarks are in front of respective cameras.
  /// \return True If all observations are correctly of landmarks in front of respective cameras.
  bool areLandmarksInFrontOfCameras() const {
    return realtimeGraph_.areLandmarksInFrontOfCameras();
  }

  /// @name Sensor configuration related
  ///@{
  /**
   * @brief Add a camera to the configuration. Sensors can only be added and never removed.
   * @param cameraParameters The parameters that tell how to estimate extrinsics.
   * @return Index of new camera.
   */
  int addCamera(
      const okvis::CameraParameters & cameraParameters);

  /**
   * @brief Add an IMU to the configuration.
   * @warning Currently there is only one IMU supported.
   * @param imuParameters The IMU parameters.
   * @return index of IMU.
   */
  int addImu(const okvis::ImuParameters & imuParameters);

  /**
   * @brief Add a GPS sensor to the configuration.
   * @warning Currently there is only one GPS supported.
   * @param gpsParameters The GPS parameters.
   * @return index of GPS.
   */
  int addGps(const okvis::GpsParameters & gpsParameters);


  /**
   * @brief Add a pose to the state.
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   */
  bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque & imuMeasurements,
                 bool asKeyframe);

  /**
   * @brief Prints state information to buffer.
   * @param stateId The pose Id for which to print.
   * @param buffer The puffer to print into.
   */
  void printStates(StateId stateId, std::ostream & buffer) const;

  /**
   * @brief Add a landmark.
   * @param landmarkId ID of the new landmark.
   * @param landmark Homogeneous coordinates of landmark in W-frame.
   * @param isInitialised Signal if 3D position initialised.
   * @return True if successful.
   */
  bool addLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark, bool isInitialised);

  /**
   * @brief Add a new landmark and get a new ID.
   * @param homogeneousPoint The point position in World frame.
   * @param initialised Defines the landmark initialisation status (depth known or not).
   * @return The ID of the newly created landmark.
   */
  LandmarkId addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised);

  /**
   * @brief Add a landmark.
   * @param landmarkId ID of the new landmark.
   * @param landmark Homogeneous coordinates of landmark in W-frame.
   * @param isInitialised Signal if 3D position initialised.
   * @return True if successful.
   */
  bool setLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark, bool isInitialised);

  /**
   * @brief Set landmark classification.
   * @param landmarkId ID of the new landmark.
   * @param classification The classification (e.g. by CNN).
   * @return True if successful.
   */
  bool setLandmarkClassification(LandmarkId landmarkId, int classification);

  /**
   * @brief Is this keypoint observed?
   * @param kid The keypoint identifier.
   * @return True if it is.
   */
  bool isObserved(KeypointIdentifier kid) const {
    return realtimeGraph_.observations_.count(kid) != 0;
  }

  /**
   * @brief Add an observation to a landmark.
   * \tparam GEOMETRY_TYPE The camera geometry type for this observation.
   * @param landmarkId ID of landmark.
   * @param stateId ID of state where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @param useCauchy Indicate whether to use a Cauchy robustifier.
   * @return True if scceeded.
   */
  template<class GEOMETRY_TYPE>
  bool addObservation(LandmarkId landmarkId, StateId stateId,
                      size_t camIdx, size_t keypointIdx, bool useCauchy = true) {
    KeypointIdentifier kid{stateId.value(), camIdx, keypointIdx};
    OKVIS_ASSERT_TRUE_DBG(Exception, multiFrames_.count(stateId), "frame does noe exist")

    MultiFramePtr multiFrame = multiFrames_.at(stateId);
    const bool success = realtimeGraph_.addObservation<GEOMETRY_TYPE>(
          *multiFrame, landmarkId, kid, useCauchy);
    OKVIS_ASSERT_TRUE_DBG(Exception, !auxiliaryStates_.at(stateId).isPoseGraphFrame,
                      "not allowed to have observations in pose graph frames!")
    if(isLoopClosing_ || isLoopClosureAvailable_) {
      touchedStates_.insert(stateId);
      touchedLandmarks_.insert(landmarkId);
    } else {
      fullGraph_.addObservation<GEOMETRY_TYPE>(*multiFrame, landmarkId, kid, useCauchy);
    }
    // A landmark may be triangulated after this frame was processed. Keep
    // such delayed observations at the frame's already selected visual weight.
    const double frameScale = auxiliaryStates_.at(stateId).reprojectionInformationScale;
    if(success && frameScale != 1.0) {
      scaleObservationInformation(stateId, camIdx, keypointIdx, frameScale);
    }
    return success;
  }
  /**
   * @brief Remove an observation from a landmark, if available.
   * @param stateId ID of state where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @return True if observation was present and successfully removed.
   */
  bool removeObservation(StateId stateId,  size_t camIdx,
                         size_t keypointIdx);

  /// \brief Set the information of an observation.
  /// @param stateId ID of state where the landmark was observed.
  /// @param camIdx ID of camera frame where the landmark was observed.
  /// @param keypointIdx ID of keypoint corresponding to the landmark.
  /// \param information Information of the observation.
  /// \return True on success.
  bool setObservationInformation(StateId stateId, size_t camIdx,
                                 size_t keypointIdx, const Eigen::Matrix2d & information);

  /// \brief Scale (multiply) the information matrix of an observation by a factor.
  /// This preserves any prior per-observation adjustments (e.g., radar support boosting).
  /// @param stateId ID of state where the landmark was observed.
  /// @param camIdx ID of camera frame where the landmark was observed.
  /// @param keypointIdx ID of keypoint corresponding to the landmark.
  /// @param information_scale Multiplicative scale (>0). Values <1 downweight the observation.
  /// @return True if the observation existed and was updated.
  bool scaleObservationInformation(StateId stateId, size_t camIdx,
                                  size_t keypointIdx, double information_scale);

  /// \brief Set a frame's absolute visual information scale, including later observations.
  /// Existing per-observation adjustments are preserved; repeating a scale is a no-op.
  /// @return Number of existing observations updated; invalid or absent frames return zero.
  size_t setFrameReprojectionInformationScale(StateId stateId, double informationScale);

  /// \brief Apply a multiplicative weight scale to every TwoPoseGraphError /
  ///        TwoPoseGraphErrorConst factor in the realtime + full graphs.
  ///        Used during visual fallback to globally weaken the marginalised
  ///        visual relative-pose constraints. Reset by passing 1.0.
  /// @param weightScale Multiplier on the information matrix (>=0). 1.0 = no change.
  /// @return Number of TwoPoseGraphError factors whose weightScale was updated
  ///         across both graphs.
  int scaleAllTwoPoseGraphErrors(double weightScale);

  /// \brief After fallback loop closure with a complete corrected path, suppress GaRLILEO between
  ///        on pairs that touch the corrected segment (see GaRLILEOParameters::
  ///        fallback_loop_garlileo_between_resume_after_path_only in ThreadedSlam).
  /// \return True if this pair should not receive a GaRLILEO between factor (caller may remove stale).
  bool suppressGarlileoBetweenForFallbackLoopPathPair(StateId poseId0, StateId poseId1) const;

  /// \brief Set the detector uniformity radius (for overlap computations).
  /// \param uniformityRadius Detector uniformity radius.
  void setDetectorUniformityRadius(double uniformityRadius) {
    kptradius_ = 0.09 * uniformityRadius / 36.0;
  }

  /**
   * @brief Applies the graph edge creation / state fixation / observation dropping & IMU-merging.
   *        The new number of frames in the window will be numKeyframes+numImuFrames.
   * @param numKeyframes Number of keyframes.
   * @param numLoopClosureFrames Number of loopclosure frames to keep around.
   * @param numImuFrames Number of frames in IMU window.
   * @param affectedFrames Frames with added or removed pose graph edges.
   * @param expand Whether or not to allow the strategy to branch out to include new keyframes.
   * @return True if successful.
   */
  bool applyStrategy(size_t numKeyframes, size_t numLoopClosureFrames, size_t numImuFrames,
                     std::set<StateId> & affectedFrames,
                     bool expand = true);

  /**
   * @brief Start ceres optimisation of the realtime problem.
   * @param[in] numIter Maximum number of iterations.
   * @param[out] updatedStates The state IDs of all the updated states.
   * @param[in] numThreads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   * @param[in] onlyNewestState Whether to only optimise the newest states (landmarks also fixed).
   * @param[in] isInitialised If false, will soft-constrain the position to the on of State ID 1.
   */
  void optimiseRealtimeGraph(
      int numIter, std::vector<StateId>& updatedStates,
      int numThreads = 1, bool verbose = false, bool onlyNewestState = false, bool isInitialised = true);

  /**
   * @brief Set a time limit for the realtime problem optimisation process.
   * @param[in] timeLimit Time limit in seconds. If timeLimit < 0 the time limit is removed.
   * @param[in] minIterations minimum iterations the optimisation process should do
   *            disregarding the time limit.
   * @return True if successful.
   */
  bool setOptimisationTimeLimit(double timeLimit, int minIterations);

  /**
   * @brief Checks whether the landmark is added to the estimator.
   * @param landmarkId The ID.
   * @return True if added.
   */
  bool isLandmarkAdded(LandmarkId landmarkId) const;

  /**
   * @brief Checks whether the landmark is initialised.
   * @param landmarkId The ID.
   * @return True if initialised.
   */
  bool isLandmarkInitialised(LandmarkId landmarkId) const;

  /// @name Getters
  ///\{
  /**
   * @brief Get a specific landmark.
   * @param[in]  landmarkId ID of desired landmark.
   * @param[out] mapPoint Landmark information, such as quality, coordinates etc.
   * @return True if successful.
   */
  bool getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const;

  /**
   * @brief Get a copy of all the landmarks as a PointMap.
   * @param[out] landmarks The landmarks.
   * @return number of landmarks.
   */
  size_t getLandmarks(MapPoints &landmarks) const;

  /**
   * @brief Get a multiframe.
   * @param stateId ID of desired multiframe.
   * @return Shared pointer to multiframe.
   */
  okvis::MultiFramePtr multiFrame(StateId stateId) const {
    if(!multiFrames_.count(stateId)) {
      return nullptr;
    }
    return multiFrames_.at(stateId);
  }

  // get/set states
  /**
   * @brief Get the pose (T_WS).
   * @param id The state ID from which go get the pose.
   * @return The pose (T_WS).
   */
  const kinematics::TransformationCacheless & pose(StateId id) const {
    return realtimeGraph_.pose(id);
  }
  /**
   * @brief Get the speed/biases [v_W, b_g, b_a].
   * @param id The state ID from which go get the speed/biases.
   * @return The pose [v_W, b_g, b_a].
   */
  const SpeedAndBias & speedAndBias(StateId id) const {
    return realtimeGraph_.speedAndBias(id);
  }
  /**
   * @brief Get the extrinsics pose (T_SC).
   * @param id The state ID from which go get the extrinsics.
   * @param camIdx The camera index of the extrinsics.
   * @return The extrinsics pose (T_SC).
   */
  const kinematics::TransformationCacheless & extrinsics(StateId id, uchar camIdx) const {
    return realtimeGraph_.extrinsics(id, camIdx);
  }

  /// \brief Get current estimate of T_GW
  kinematics::Transformation T_GW() const {
    return realtimeGraph_.T_GW();
  }

  /// \brief Get gps Measurements
  /// \param states set of states for which GPS measurements should be returned
  /// \param gpsMeasurements[output] Fill with measurements
  void gpsMeasurements(StateId stateId, AlignedVector<Eigen::Vector3d>& gpsMeasurements) {
    return realtimeGraph_.gpsMeasurements(stateId, gpsMeasurements);
  }

  /**
   * @brief Is a state a keyframe?
   * @param id The state ID in question.
   * @return True if it is.
   */
  bool isKeyframe(StateId id) const {
    return realtimeGraph_.isKeyframe(id);
  }

  /**
   * @brief Is a state a pose graph frame (i.e. without any observations)?
   * @param id The state ID in question.
   * @return True if it is.
   */
  bool isPoseGraphFrame(StateId id) const {
    //OKVIS_CHECK_MAP(auxiliaryStates_,id);
    return auxiliaryStates_.at(id).isPoseGraphFrame;
  }

  /**
   * @brief Is a state considered for place recognition?
   * @param id The state ID in question.
   * @return True if it is.
   */
  bool isPlaceRecognitionFrame(StateId id) const {
    //OKVIS_CHECK_MAP(auxiliaryStates_,id);
    return auxiliaryStates_.at(id).isPlaceRecognitionFrame;
  }

  /**
   * @brief The tracking quality w.r.t. the map:
   * fraction of the image pixels covered with matches.
   * @param id The state ID in question.
   * @return The quality.
   */
  double trackingQuality(StateId id) const;

  /**
   * @brief Set if keyframe or not.
   * @warning Only applicable to IMU frames.
   * @param id The state ID in question.
   * @param isKeyframe Whether or not to set it to be a keyframe.
   * @return True on success.
   */
  bool setKeyframe(StateId id, bool isKeyframe);

  /**
   * @brief Has this state closed a loop / recognised a place?
   * @param id The state ID in question.
   * @return True if it has.
   */
  bool closedLoop(StateId id) const {
    //OKVIS_CHECK_MAP(auxiliaryStates_,id);
    return auxiliaryStates_.at(id).closedLoop;
  }

  /// \brief Get the recent loop closure frames.
  /// \return The recent loop closure frames.
  const std::set<StateId> & recentLoopClosureFrames(StateId id) const {
    //OKVIS_CHECK_MAP(auxiliaryStates_,id);
    return auxiliaryStates_.at(id).recentLoopClosureFrames;
  }

  /// \brief The ID of the most recent (current) frame.
  /// \return The ID.
  StateId currentStateId() const {
    return realtimeGraph_.currentStateId();
  }

  /// \brief The ID of the current keyframe, i.e. the one with most overlapping matches.
  /// \return The ID.
  StateId currentKeyframeStateId(bool considerLoopClosureFrames = true) const;

  /// \brief The ID of the most overlapping frame.
  /// \param frame Frame in question.
  /// \param considerLoopClosureFrames Should loop closure frames also be considered?
  /// \return The ID.
  StateId mostOverlappedStateId(StateId frame, bool considerLoopClosureFrames = true) const;

  /// \brief The ID of the loopclosure frame with currently most overlapping matches.
  /// \return The ID.
  StateId currentLoopclosureStateId() const;

  /// \brief The ID of a state by age (from current/newest frame).
  /// \param age The age.
  /// \return The ID.
  StateId stateIdByAge(size_t age) const {
    return realtimeGraph_.stateIdByAge(age);
  }

  /**
   * @brief Set the pose (T_WS).
   * @param id The state ID for which go set the pose.
   * @param pose The pose (T_WS).
   * @return True on success.
   */
  bool setPose(StateId id, const kinematics::TransformationCacheless & pose);
  /**
   * @brief Set the speed/biases [v_W, b_g, b_a].
   * @param id The state ID for which go set the speed/biases.
   * @param speedAndBias The the speed/biases [v_W, b_g, b_a].
   * @return True on success.
   */
  bool setSpeedAndBias(StateId id, const SpeedAndBias & speedAndBias);
  /**
   * @brief Set the extrinsics pose (T_SC).
   * @param id The state ID for which go set the extrinsics.
   * @param camIdx The camera index of the extrinsics.
   * @param extrinsics The extrinsics pose (T_SC).
   * @return True on success.
   */
  bool setExtrinsics(
      StateId id, uchar camIdx, const kinematics::TransformationCacheless & extrinsics);

  /// @brief Get the number of states/frames in the estimator.
  /// \return The number of frames.
  size_t numFrames() const {
    return multiFrames_.size();
  }

  ///@}

  /**
   * @brief Checks if a particular frame is still in the IMU window.
   * @param[in] id ID of frame to check.
   * @return True if the frame is in IMU window.
   */
  bool isInImuWindow(StateId id) const;

  /// @name Getters
  /// @{
  /**
   * @brief Get the timestamp for a particular frame.
   * @param[in] id ID of frame.
   * @return Timestamp of frame.
   */
  okvis::Time timestamp(StateId id) const;

  /// \brief Draws a debug overhead image/
  /// \param image The image to be drawn into.
  /// \param idx 0 for realtime, 1 for full, 2 for observation-less.
  void drawOverheadImage(cv::Mat & image, int idx=0) const;

  /**
   * @brief Get the co-observed frames for a particular frame.
   * @param[in] id ID of frame.
   * @param[out] observedIds IDs co-observed with frame id.
   * @return true on success.
   */
  bool getObservedIds(StateId id, std::set<StateId>& observedIds) const;

  /**
   * @brief Is this state an active loop-closure frame?
   * @param frameId The state ID in question.
   * @return True if it is.
   */
  bool isLoopClosureFrame(StateId frameId) const;

  /**
   * @brief Was this state recently used as a loop-closure frame?
   * @param frameId The state ID in question.
   * @return True if it is.
   */
  bool isRecentLoopClosureFrame(StateId frameId) const;

  /// \brief Removes landmarks that are not observed.
  /// \return The number of landmarks removed.
  int cleanUnobservedLandmarks();

  /// \brief Merge landmark with two IDs into one, taking care of all the observations.
  /// \param fromId Landmark from ID (will be deleted).
  /// \param intoId Landmark to ID (will be kept).
  /// \return True on success.
  bool mergeLandmark(const LandmarkId &fromId, const LandmarkId &intoId);

  /// \brief Merge landmarks with two IDs into one, taking care of all the observations.
  /// \param fromIds Landmarks from ID (will be deleted).
  /// \param intoIds Landmarks to ID (will be kept).
  /// \return Number of successful merges.
  int mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds);

  /// \brief Write the full optimised trajectory into a file.
  /// \param csvFileName Path to file to write.
  /// \param rpg Whether to use the RPG format (instead of EuRoC).
  /// \return True on success
  bool writeFinalCsvTrajectory(const std::string& csvFileName, bool rpg = false) const;

  /// \brief Write the full optimised trajectory in the global reference frame into a file.
  /// \param csvFileName Path to file to write.
  /// \return True on success
  bool writeGlobalCsvTrajectory(const std::string& csvFileName) const;

  /// \brief Attempt loop closure.
  /// \param pose_i The old frame that was recognised.
  /// \param pose_j The new pose.
  /// \param T_Si_Sj The relative pose between the IMU frames S.
  /// \param information The relative pose information.
  /// \param skipFullGraphOptimisation Whether to not subsequently optimise the full graph.
  /// \param driftPercentageHeuristic Allowed drift in % of distance travelled.
  /// \return True on success.
  bool attemptLoopClosure(StateId pose_i, StateId pose_j,
                          const kinematics::Transformation& T_Si_Sj,
                          const Eigen::Matrix<double, 6, 6>& information,
                          bool & skipFullGraphOptimisation,
                          double driftPercentageHeuristic,
                          bool keepFullGraphRelativePoseConstraint = false,
                          double fullGraphRelativePoseInformationScale = 100.0,
                          double persistentFullGraphRelativePoseInformationScale = 100.0,
                          bool addLoopPathRelativePoseConstraints = false,
                          int loopPathRelativePoseConstraintMode = 1,
                          int loopPathRelativePoseConstraintStride = 1,
                          bool weakenExistingLoopPathConstraints = false,
                          double existingLoopPathConstraintScale = 0.01,
                          double loopPathChainInformationScale = 100.0);

  /// \brief Add a loopclosure frame (after successful attempt).
  /// \brief loopClosureFrameId The ID of the frame to be added.
  /// \brief loopClosureLandmarks The landmarks observed by the loopclosure frame.
  void addLoopClosureFrame(StateId loopClosureFrameId, std::set<LandmarkId> &loopClosureLandmarks,
                           bool skipFullGraphOptimisation);

  /// \brief Check if full graph needs to be optimised after addLoopClosureFrame
  /// (or during interrupted full graph optimisation).
  /// \return True if it needs optimisation.
  bool needsFullGraphOptimisation() const {return needsFullGraphOptimisation_; }

  /**
   * @brief Start ceres optimisation of the full graph problem.
   * @param[in] numIter Maximum number of iterations.
   * @param[out] summary Get optimisation summary back.
   * @param[in] numThreads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   * @param[in] functionTolerance Relative cost stopping criterion of the main solve.
   */
  void optimiseFullGraph(int numIter, ::ceres::Solver::Summary &summary,
                         int numThreads = 1, bool verbose = false,
                         double functionTolerance = 1.0e-6);

  /// \brief Performs a final Bundle Adjustment.
  /// \param numIter Number of iterations.
  /// \param summary Ceres summary.
  /// \param updatedStatesBa States that were changed -- should be all of them, reall.
  /// \param extrinsicsPositionUncertainty Std dev. of extrinsics position. 0.0 for fixed.
  /// \param extrinsicsOrientationUncertainty Std dev. of extrinsics orientation. 0.0 for fixed.
  /// \param numThreads Number of solver threads to use.
  /// \param verbose Console outputs or not?
  /// \param functionTolerance Relative cost stopping criterion for both final BA passes.
  void doFinalBa(int numIter, ::ceres::Solver::Summary &summary,
                 std::set<StateId> & updatedStatesBa,
                 double extrinsicsPositionUncertainty = 0.0,
                 double extrinsicsOrientationUncertainty = 0.0,
                 int numThreads = 1, bool verbose = false,
                 double functionTolerance = 1.0e-6);

  /// \brief Save the map to CSV.
  /// \param path CSV file path.
  bool saveMap(std::string path);

  /// \brief Check if currently closing loop.
  /// \return True if it is.
  bool isLoopClosing() const { return isLoopClosing_; }

  /// \brief Check if a background loop closure has finished.
  /// \return True if it is.
  bool isLoopClosureAvailable() const { return isLoopClosureAvailable_ && !isLoopClosing_; }

  /// \brief After full graph optimisation, this synchronises to
  /// the realtime / observation-less graphs.
  /// \return True on success.
  bool synchroniseRealtimeAndFullGraph(std::vector<StateId>& updatedStates);

  /// \brief Is the state of fullGraph_ vs. realtimeGraph_ synchronised?
  /// \return True if it is.
  bool isSynched() const {
    return fullGraph_.isSynched(realtimeGraph_);
  }

  /// \brief Get the overlap fraction (of pixels) between two frames.
  /// \param frameA frame A.
  /// \param frameB frame B.
  /// \return The overlap fraction.
  double overlapFraction(const MultiFramePtr frameA,
                                const MultiFramePtr frameB) const;

  /// \brief Get the set of key frame IDs.
  /// \return The set of key frame IDs.
  const std::set<StateId>& keyFrames() const {return keyFrames_;}

  /// \brief Get the set of IMU frame IDs.
  /// \return The set of IMU frame IDs.
  const std::set<StateId>& imuFrames() const {return imuFrames_;}

  /// \brief Get the set of loop closure frame IDs.
  /// \return The set of loop closure frame IDs.
  const std::set<StateId>& loopClosureFrames() const {return loopClosureFrames_;}

  /// \brief Eliminate IMU frames that are too much (that are not keyframes).
  /// \param numImuFrames Number of IMU frames to keep.
  void eliminateImuFrames(size_t numImuFrames, std::set<StateId> & affectedFrames);

  /// \brief Add a relative pose constraint between two states.
  ///
  /// This adds the constraint to the realtime graph. If the full graph is accessible it is
  /// also added there. If loop-closing is active, the constraint is stored in the realtime
  /// graph and marked for later synchronisation.
  ///
  /// \param poseId0     ID of one pose (reference).
  /// \param poseId1     ID of the other pose.
  /// \param T_S0S1      Relative transform measurement from pose0 to pose1.
  /// \param information 6x6 information matrix (weight).
  /// \param overwriteProtected If true, allow replacing fallback loop-chain protected edges.
  /// \return True on success.
  bool addRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                const kinematics::Transformation &T_S0S1,
                                const Eigen::Matrix<double, 6, 6>& information,
                                bool overwriteProtected = false);

  /// \brief Remove a relative pose constraint between two states, if present.
  /// \param poseId0 ID of one pose.
  /// \param poseId1 ID of the other pose.
  /// \param removeProtected If true, allow removing fallback loop-chain protected edges.
  /// \return True if a constraint was removed from at least one graph.
  bool removeRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                    bool removeProtected = false);

  /// \brief Add an absolute pose constraint (unary pose prior) on one state.
  ///
  /// This is intended as a fallback when vision is unreliable: it pulls the current OKVIS pose
  /// towards an external pose measurement (e.g. GaRLILEO).
  ///
  /// \param poseId     ID of the pose to constrain.
  /// \param T_WS       Pose measurement (sensor frame in world).
  /// \param information 6x6 information matrix (weight).
  /// \return True on success.
  bool addPoseConstraint(StateId poseId,
                         const kinematics::Transformation &T_WS,
                         const Eigen::Matrix<double, 6, 6>& information);

  /// \brief Add an ego-velocity prior for a single state.
  bool addVelocityConstraint(StateId poseId,
                             const Eigen::Vector3d& v_S,
                             const Eigen::Matrix3d& information);

  /// \brief Remove all velocity priors on one state (if any).
  bool removeVelocityConstraints(StateId poseId);

  /// Replace the velocity prior with one isotropic body-velocity prior, or clear it
  /// when enabled=false. False means neither graph was modified; retry later.
  /// Never blocks or mutates the full graph during a solve/awaiting synchronization.
  bool setVelocityPrior(StateId poseId, const Eigen::Vector3d& v_S,
                        double information, bool enabled);

  /// GPS STUFF COMES HERE...

  /// \brief Add GPS constraints on all Graph members
  /// \param gpsMeasurementDeque Queue containing a sequence of GPS measurements
  /// \param imuMeasurementDeque Queue containing a sequence of IMU measurements
  /// \return True on success
  bool addGpsMeasurementsOnAllGraphs(GpsMeasurementDeque& gpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque);

  /// \brief Check for (and if needed apply) available alignments due to GPS signals
  /// \return True if alignment has been applied, false if not
  bool tryGpsAlignment();


  /// \brief Debugging function to dump residual values of a graph
  void dumpGpsResiduals(const std::string &gpsResCsvFileName)
  {
    // call residual writer on graph
    realtimeGraph_.dumpGpsResiduals(gpsResCsvFileName);
  }

  /// \brief Attempt global GPS alignment
  /// \param gpsLossId Id of the fixed state where last GPS signal is received
  /// \param gpsReturnId Id of the staet where GPS measurements are available again
  /// \return True on success
  bool attemptFullGpsAlignment(StateId gpsLossId , StateId gpsReturnId, const okvis::kinematics::Transformation& T_GW_new);

  /// \brief Attempt global GPS alignment (position only)
  /// \param gpsLossId Id of the fixed state where last GPS signal is received
  /// \param gpsReturnId Id of the staet where GPS measurements are available again
  /// \return True on success
  bool attemptPosGpsAlignment(StateId gpsLossId , StateId gpsReturnId, const Eigen::Vector3d& posAlignVec);

  /// \brief Add a GPS alignment ("GPS loop closure") frame (after successful attempt).
  void addGpsAlignmentFrame(StateId gpsLossFrameId);

  /// \brief             Add Alignment constraints from submapping interface
  /// @param frame_A_id  ID of frame {A}
  /// @param frame_B_id  ID of frame {B}
  /// @param pointCloud  Point Cloud in {B} that adds constraints w.r.t. {B}
  /// @param sensorError Depth uncertainty (1-sigma) of pointCloud
  /// @param isLidar Flag if Factors are derived from a LiDAR (Depth otherwise)
  /// @param robustFunction Which robust function is used in the graph "Cauchy" or "Tukey"
  /// \return Returns true normally
  bool addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                     const uint64_t& frame_A_id, const uint64_t frame_B_id,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                     std::vector<float> sensorError, bool isLidar=false, const std::string robustFunction = "Tukey");

  /// \brief Clears the underlying graphs and resets everything (so you can re-start).
  void clear();

  /// \brief Write some debug information to csv file
  /// \param csvFilePrefix File Prefix vor csv files
  void writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix);

private:

  /// \brief Convert to pose graph edges via MST creation.
  /// \param framesToConvert The state IDs of those to be converted to pose graph states.
  /// \param framesToConsider The frames to consider for the operation (i.e. to do the MST over).
  /// \param affectedFrames Frames that with added pose graph edges.
  /// \return True on success.
  bool convertToPoseGraphMst(const std::set<StateId> &framesToConvert,
                             const std::set<StateId> &framesToConsider,
                             std::set<StateId> &affectedFrames);

  /// \brief Re-activate the neighbouring states (i.e. convert back to reprojection errors).
  /// \param keyframe The keyframe to consider.
  /// \return The number of frames re-activated.
  int expandKeyframe(StateId keyframe);

  /// \brief Delete place recognition frames with much overlap from database.
  /// \return Number of pruned frames.
  int prunePlaceRecognitionFrames();

  std::map<StateId, MultiFramePtr> multiFrames_; ///< All the multiframes added so far.

  /// \brief Helper struct to store information about states.
  struct AuxiliaryState
  {
    double reprojectionInformationScale = 1.0; ///< Also applies to observations added later.
    bool isKeyframe = false;             ///< Is it a keyframe?
    bool isImuFrame = false;             ///< Is it an IMU frame?
    bool isPoseGraphFrame = false;       ///< Is it a pose graph frame?
    bool closedLoop = false;             ///< Has it closed a loop?
    bool isLost = false;                 ///< Is it lost?
    bool isPlaceRecognitionFrame = true; ///< Is it in the DBoW database as place recognition frame?
    StateId loopId; ///< The ID of the loop (needed for unfreezing before full graph optimisation)/
    std::set<StateId> recentLoopClosureFrames; ///< These were recent loop closure frames.
  };
  std::map<StateId, AuxiliaryState> auxiliaryStates_; ///< Store information about states.

  /// \brief Store disconnected SLAM components (currently unused).
  struct Component0 {
    std::set<StateId> poseIds; ///< Pose IDs of the component.
    StateId referenceId; ///< Reference pose ID.
    StateId connectedReferenceId; ///< Connection to other component.
  };
  std::vector<Component0> components_; ///< All the created components.
  size_t currentComponentIdx_ = 0; ///< The index of the current component.

  std::set<StateId> loopClosureFrames_; ///< All the current loop closure frames.

  ///
  std::set<StateId> imuFrames_; ///< All the current IMU frames.
  std::set<StateId> keyFrames_; ///< All the current keyframes.
  ///

  // underlying graphs;
  ViGraphEstimator realtimeGraph_; ///< The realtime estimator.
  ceres::RealtimeTrustRegionRadius realtimeTrustRegionRadius_;
  ViGraphEstimator fullGraph_; ///< The full grapf for asynchronous optimisation.

  // A request cache, not a solver weight: the worker snapshots it before solving.
  // Actual error weights remain fixed throughout the full-graph solve below.
  std::atomic<double> requestedVisualTwoPoseWeightScale_{1.0};
  // Solve owns this mutex; per-frame updates only try_lock, so LC never blocks them.
  std::mutex fullGraphTwoPoseWeightMutex_;

  std::atomic_bool needsFullGraphOptimisation_; ///< Do we need a full graph optimisation now?
  std::atomic_bool isLoopClosing_; ///< Is there currently a full graph optimisation running?
  std::atomic_bool isLoopClosureAvailable_; ///< New result from full graph optimisation available?
  std::set<StateId> currentLoopClosureFrames_; ///< The set of current loop closure frames.

  /// \brief Helper struct to store graph inconsistencies.
  struct AddStatesBacklog {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Time timestamp; ///< Timestamp.
    StateId id; ///< State ID.
    ImuMeasurementDeque imuMeasurements; ///< IMU measurements leading up to it.
  };

  AlignedVector<AddStatesBacklog> addStatesBacklog_; ///< Backlog of states to add to fullGraph_.
  std::map<StateId, StateId> eliminateStates_; ///< States eliminated in realtimeGraph_.
  std::set<StateId> touchedStates_; ///< States modified in realtimeGraph_.
  std::set<LandmarkId> touchedLandmarks_; ///< Landmarks modified in realtimeGraph_.

  std::set<StateId> updatedStatesLoopClosureAttempt_; ///< States updated in loop-closure attempt.

  /// \brief Helper struct for relative poses.
  struct RelPoseInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    kinematics::Transformation T_Si_Sj; ///< Relative Transform.
    Eigen::Matrix<double, 6, 6> information; ///< Relative uncertainty.
    StateId pose_i; ///< Reference pose ID.
    StateId pose_j; ///< Other pose ID.
    bool keepInFullGraph = false; ///< Keep this constraint after initial loop optimisation.
    double informationScale = 100.0; ///< Initial full-graph multiplier for the relative pose information.
    double persistentInformationScale = 100.0; ///< Persistent full-graph multiplier for kept relative pose information.
  };

  /// \brief Persistent corrected loop path snapshot.
  struct LoopPathConstraintSnapshot {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector<StateId> stateIds; ///< Ordered corrected path state IDs.
    AlignedVector<kinematics::Transformation> T_WS; ///< Corrected poses at loop closure time.
    Eigen::Matrix<double, 6, 6> information; ///< Base loop information.
    double informationScale = 100.0; ///< Multiplier used for corrected-path chain constraints.
    int stride = 1; ///< Recreate one chain edge every N surviving states.
  };

  /// \brief Corrected fallback loop path whose pre-existing constraints are suppressed.
  struct LoopPathConstraintSuppressionSnapshot {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector<StateId> stateIds; ///< Ordered corrected path state IDs.
  };

  AlignedVector<RelPoseInfo> fullGraphRelativePoseConstraints_; ///< Relative pose constraints.
  AlignedVector<RelPoseInfo> persistentFullGraphRelativePoseConstraints_; ///< Full-graph-only loop relative-pose constraints kept across sync.
  AlignedVector<LoopPathConstraintSnapshot> persistentLoopPathConstraintSnapshots_; ///< Persistent corrected loop path chain constraints.
  AlignedVector<LoopPathConstraintSuppressionSnapshot>
      suppressedLoopPathConstraintSnapshots_; ///< Corrected fallback loop path constraints removed/suppressed across sync.

  /// Last fallback LC path [lo,hi] by StateId::value(); GaRLILEO between is resumed only for pairs
  /// strictly after hi (both endpoints > hi) when ThreadedSlam policy is enabled.
  bool fallbackLoopGarlileoPathBoundsActive_ = false;
  uint64_t fallbackLoopGarlileoPathIdLo_ = 0;
  uint64_t fallbackLoopGarlileoPathIdHi_ = 0;

  void setFallbackLoopGarlileoBetweenPathBounds(StateId pose_i, StateId pose_j);
  void clearFallbackLoopGarlileoBetweenPathBounds();

  bool isPersistentFullGraphRelativePoseConstraint(StateId poseId0, StateId poseId1) const;
  void addOrUpdatePersistentFullGraphRelativePoseConstraint(const RelPoseInfo& constraint);
  void erasePersistentFullGraphRelativePoseConstraint(StateId poseId0, StateId poseId1);
  void reapplyPersistentFullGraphRelativePoseConstraints();
  std::pair<uint64_t, uint64_t> orderedStatePairKey(StateId poseId0, StateId poseId1) const;
  bool isProtectedLoopPathRelativePoseConstraint(StateId poseId0, StateId poseId1) const;
  bool isLoopPathRelativePoseConstraintInstalled(
      StateId poseId0, StateId poseId1, bool fullGraph) const;
  void protectLoopPathRelativePoseConstraint(
      StateId poseId0, StateId poseId1,
      bool installedRealtime = true, bool installedFullGraph = true);
  void unprotectLoopPathRelativePoseConstraint(StateId poseId0, StateId poseId1);
  bool addOrUpdateProtectedLoopPathRelativePoseConstraint(
      StateId poseId0, StateId poseId1,
      const kinematics::Transformation& T_S0S1,
      const Eigen::Matrix<double, 6, 6>& information,
      bool updateRealtimeGraph, bool updateFullGraph);
  void addPersistentLoopPathConstraintSnapshot(
      const std::vector<StateId>& stateIds,
      const AlignedVector<kinematics::Transformation>& T_WS,
      const Eigen::Matrix<double, 6, 6>& information,
      double informationScale, int stride,
      bool weakenExistingConstraints, double existingConstraintScale);
  void reapplyPersistentLoopPathRelativePoseConstraints(
      bool updateRealtimeGraph, bool updateFullGraph);
  bool isStatePairInSuppressedLoopPath(StateId poseId0, StateId poseId1) const;
  bool isSuppressedLoopPathRelativePoseConstraint(StateId poseId0, StateId poseId1) const;
  bool isSuppressedLoopPathTwoPoseConstraint(StateId poseId0, StateId poseId1) const;
  void addSuppressedLoopPathConstraintSnapshot(const std::vector<StateId>& stateIds);
  std::pair<size_t, size_t> suppressLoopPathConstraintsInGraph(
      ViGraphEstimator& graph, const std::vector<StateId>& stateIds);
  void applySuppressedLoopPathConstraintRemovals(
      bool updateRealtimeGraph, bool updateFullGraph);
  bool removeTwoPoseLinkIfPresent(ViGraphEstimator& graph, StateId poseId0, StateId poseId1);
  bool removeTwoPoseConstLinkIfPresent(ViGraphEstimator& graph, StateId poseId0, StateId poseId1);
  int scaleVisualTwoPoseGraphErrors(ViGraphEstimator& graph, double weightScale);
  std::set<std::pair<uint64_t, uint64_t>> protectedLoopPathRelativePoseConstraints_;
  // Overwrite protection is shared, but installation can be deferred per graph.
  std::set<std::pair<uint64_t, uint64_t>> protectedLoopPathRelativePoseConstraintsRealtime_;
  std::set<std::pair<uint64_t, uint64_t>> protectedLoopPathRelativePoseConstraintsFullGraph_;
  mutable std::mutex loopPathProtectionMutex_; ///< Short metadata operations only; never held during Solve.
  std::set<std::pair<uint64_t, uint64_t>> suppressedLoopPathRelativePoseConstraints_;
  std::set<std::pair<uint64_t, uint64_t>> suppressedLoopPathTwoPoseConstraints_;

  StateId lastFreeze_; ///< Store up to where the realtimeGraph_ states were fixed.

  double kptradius_ = 0.09; ///< Constant of how large keypoints should appear for overlap comp.

  // gps stuff

  //std::atomic_bool isPositionAligning_ = false; /// < Needed to wait for a potential position alignment while > is it needed??
  std::atomic_bool gpsObservability_; ///< Flag if extrinsic gps transformation (gps <-> world) is observable
  struct AddGpsBacklog{
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      StateId id;
      GpsMeasurement gpsMeasurement;
      ImuMeasurementDeque imuMeasurements;
      bool reInitFlag;
  };
  AlignedVector<AddGpsBacklog> addGpsBacklog_;

  // Backlog for Submap Alignment Constraints
  struct AddSubmapAlignmentBacklog{
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      const SupereightMapType* submap_ptr;
      uint64_t frame_A_id;
      uint64_t frame_B_id;
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud;
      std::vector<float> sensorError; ///< Uncertainty value for the LiDAR / Depth Sensor [in meters]
      bool isLidar;
      std::string robustFunction;
  };
  AlignedVector<AddSubmapAlignmentBacklog> addSubmapAlignmentBacklog_;

  public: /// \todo remove this horrible hack once these are properly being estimated!
  AlignedMap<StateId, AlignedMap<uint64_t, kinematics::Transformation>> T_AiS_; ///< Agent i to World pose (if available).
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_VISLAMBACKEND_HPP_ */
