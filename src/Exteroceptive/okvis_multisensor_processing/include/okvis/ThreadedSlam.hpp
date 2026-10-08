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
 * @file ThreadedSlam.hpp
 * @brief Header file for the ThreadedSlam3 class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

 #ifndef INCLUDE_OKVIS_THREADEDSLAM3_HPP_
 #define INCLUDE_OKVIS_THREADEDSLAM3_HPP_
 
 #include <thread>
 #include <atomic>
 #include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <deque>
#include <limits>
#include <vector>
 
 #include <opencv2/core/core.hpp>
 
 #include <okvis/cameras/NCameraSystem.hpp>
 #include <okvis/Measurements.hpp>
 #include <okvis/Frontend.hpp>
 #include <okvis/MultiFrame.hpp>
 #include <okvis/Parameters.hpp>
 #include <okvis/assert_macros.hpp>
 
 #include <okvis/ViVisualizer.hpp>
 #include <okvis/timing/Timer.hpp>
 #include <okvis/threadsafe/ThreadsafeQueue.hpp>
 #include <okvis/ViSlamBackend.hpp>
 #include <okvis/ViInterface.hpp>
 
 #include <se/supereight.hpp>
 #include <okvis/config_mapping.hpp>
 #include <okvis/mapTypedefs.hpp>
 #include <okvis/SubmappingInterface.hpp>
 #include <Eigen/StdVector>
 
 /// \brief okvis Main namespace of this package.
 namespace okvis {

 class StereoDetectionWorker;
 
 /**
  *  \brief
  *  This class manages the complete data flow in and out of the algorithm, as well as
  *  between the processing threads.
  *
  *  To ensure fast return from user callbacks, new data are collected in thread save
  *  input queues and are processed in data consumer threads, one for each data source.
  *  The algorithm tasks are running in individual threads with thread save queue for
  *  message passing.
  *  For sending back data to the user, publisher threads are created for each output
  *  which ensure that the algorithm is not being blocked by slow users.
  *
  *  All the queues can be limited to size 1 to back propagate processing congestions
  *  to the user callback.
  */
 class ThreadedSlam : public ViInterface {
  public:
 
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
 
   /**
    * \brief Constructor.
    * \param parameters Parameters and settings.
    * @param dBowDir The directory to the DBoW vocabulary.
    */
   ThreadedSlam(okvis::ViParameters& parameters, std::string dBowDir, se::SubMapConfig submapConfig = se::SubMapConfig());
 
   /// \brief Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
   virtual ~ThreadedSlam() override;
 
   /// \name Add measurements to the algorithm
   /// \{
   /**
    * \brief             Add a set of new image.
    * \param stamp       The image timestamp.
    * \warning           This is the original timestamp -- this function will compensate the
    *                    image delay as timestamp_camera_correct = stamp - image_delay!
    * \param images      The images. Can be gray (8UC1) or RGB (8UC3).
    * \param depthImages The depth images as float32, in [m]
    * \return            Returns true normally. False, if previous one not processed yet.
    */
   virtual bool addImages(const okvis::Time & stamp,
                          const std::map<size_t, cv::Mat> & images,
                          const std::map<size_t, cv::Mat> & depthImages
                          = std::map<size_t, cv::Mat>()) override final;
 
   /**
    * \brief          Add an IMU measurement.
    * \param stamp    The measurement timestamp.
    * \param alpha    The acceleration measured at this time.
    * \param omega    The angular velocity measured at this time.
    * \return Returns true normally. False if the previous one has not been processed yet.
    */
   virtual bool addImuMeasurement(const okvis::Time & stamp,
                                  const Eigen::Vector3d & alpha,
                                  const Eigen::Vector3d & omega) override final;
 
   /**
    * \brief          Add a LiDAR measurement.
    * \param stamp    The measurement timestamp.
    * \param rayMeasurement The ray measurement (cartesian coordinates).
    * \return Returns true normally. False if the previous one has not been processed yet.
    */
   virtual bool addLidarMeasurement(const okvis::Time & stamp,
                                    const Eigen::Vector3d & rayMeasurement) override final;
 
   /**
    * @brief Adds a depth image image to the measurement Queue
    * @param[in] stamp The timestamp
    * @param[in] depthImage The depth image
    * @param[in] sigmaImage The 1-sigma image
    * 
    * @return True if successful
   */
   bool addDepthMeasurement(const okvis::Time &stamp,
                            const cv::Mat &depthImage,
                            const std::optional<cv::Mat> &sigmaImage = std::nullopt) override final;

  /// \brief          Add GaRLILEO pseudo measurement (pose/velocity).
  void addGaRLILEOPseudoMeasurement(const okvis::Time& stamp,
                                    const Eigen::Quaterniond& R_WS,
                                    const Eigen::Vector3d& v_body,
                                    const Eigen::Vector3d& p_W) override final;

  /// \brief          Add GaRLILEO gravity measurement (world-frame gravity vector).
  void addGaRLILEOGravityMeasurement(const okvis::Time& stamp,
                                     const Eigen::Vector3d& g_W) override final;

  /// \brief          Add radar targets measurement.
  void addRadarTargetsMeasurement(const okvis::Time& stamp,
                                  const okvis::RadarTargetReadings& targets,
                                  int radar_index = 0) override final;
 
   /**
    * \brief          Add a GPS measurement.
    * \param stamp    The measurement timestamp.
    * \param posGps   The position measured at this time.
    * \param errGps   Standard deviation (error) of measurement.
    * \return Returns true normally. False if the previous one has not been processed yet.
    */
   virtual bool addGpsMeasurement(const okvis::Time & stamp,
                                  const Eigen::Vector3d & posGps,
                                  const Eigen::Vector3d & errGps);
 
   /**
    * \brief          Add a GPS measurement with geodetic coordinates.
    * \param stamp    The measurement timestamp.
    * \param lat      Latitude Measurement.
    * \param lon      Longitude Measurement.
    * \param height   Height Measurement.
    * \param hAcc     Horizontal Measurement accuracy.
    * \param vAcc     Vertical Measurement Accuracy.
    * \return Returns true normally. False if the previous one has not been processed yet.
    */
   virtual bool addGeodeticGpsMeasurement(const okvis::Time & stamp,
                                          double lat, double lon, double height,
                                          double hAcc, double vAcc);
 
     /**
    * \brief             Add alignment constraint
    * @param submap_A_ptr  pointer to map {A}
    * @param submap_B_ptr  pointer to map {B}
    * @param frame_A_id  ID of frame {A}
    * @param frame_B_id  ID of frame {B}
    * @param pointCloud  Point Cloud in {B} that adds constraints w.r.t. {B}
    * @param sensorError 1-sigma of incoming measurements
    * @return Returns true normally.
    */
     bool addSubmapAlignmentConstraints(const SupereightMapType* submap_A_ptr,
                                        const SupereightMapType* submap_B_ptr,
                                        const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                                        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                        std::vector<float> sensorError);
 
   /// \}
   /// \name Setters
   /// \{
   /**
    * \brief Set the blocking variable that indicates whether the addMeasurement() functions
    *        should return immediately (blocking=false), or only when the processing is complete.
    */
   virtual void setBlocking(bool blocking) override final;
 
   /// \brief Runs main processing iteration, call in your main loop.
   bool processFrame();
 
   /// \brief Trigger display (needed because OSX won't allow threaded display).
   void display(std::map<std::string, cv::Mat> & images) override final;
 
   /// \brief Set the final trajectory CSV file.
   /// \param finalTrajectoryCsvFileName File name.
   /// \param rpg Use RPG format?
   void setFinalTrajectoryCsvFile(const std::string & finalTrajectoryCsvFileName, bool rpg = false) {
     finalTrajectoryCsvFileName_ = finalTrajectoryCsvFileName;
     rpg_ = rpg;
   }
 
   /// \brief Set the map CSV file.
   /// \param mapCsvFileName File name.
   void setMapCsvFile(const std::string & mapCsvFileName) {
     mapCsvFileName_ = mapCsvFileName;
   }
   
   /// \brief Stops all threading.
   void stopThreading();
 
   /// \brief Writes the final trajectory CSV file.
   void writeFinalTrajectoryCsv();

  /// \brief Writes per-frame minimum EWG utility to CSV.
  /// \param csvFileName output file path.
  void writeEwgUtilityMinCsv(const std::string& csvFileName);

  /// \brief Write the per-frame optimised IMU biases (b_g, b_a) to a CSV file.
  /// \param csvFileName The file to write to.
  void writeImuBiasCsv(const std::string& csvFileName);
 
   /// \brief  Writes final trajectory in the global reference frame to a csv File
   /// \param csvFileName name of the output file
   void writeGlobalTrajectoryCsv(const std::string& csvFileName);
 
   /// \brief Write some debug information to csv file
   /// \param csvFilePrefix File Prefix vor csv files
   void writeDebugStatisticsCsv(const std::string& csvFilePrefix);
 
   /// \brief Performs a final Bundle Adjustment.
   void doFinalBa();
 
   /// \brief Save the map to CSV.
   bool saveMap();
 
   /// \brief Export SE2 maps.
   bool exportSe2Map();
   /// \brief Create SE2 maps at very end with bundle-adjusted best-possible trajectory.
   bool createFinalSe2Map();
 
   /// \}
   /// \brief Write Value of GPS residuals to a file
   /// \param gpsResCsvFileName Name of the file
   void dumpGpsResiduals(const std::string & gpsResCsvFileName);
 
   /// \brief Check if a new estimator keyframe is needed based on LiDAR overlap
   /// \return True if not enough LiDAR points can be tracked in the current submap
   bool needsNewLidarKeyframe();
 
   /// \brief Callback to publish Submap alignment Points
   typedef std::function<void(const okvis::Time&, const okvis::kinematics::Transformation&,
                               const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&, bool)> AlignmentPublishCallback;
 
   /// \brief Set Submap Alignment Points Publishing Callback
   void setAlignmentPublishCallback(const AlignmentPublishCallback& alignPublishCallback){
       alignmentPublishCallback_ = alignPublishCallback;
   }
  /// \}
 
 private:
   /// \brief Start all threads.
   void startThreads();
   /// \brief Initialises settings and calls startThreads().
   void init();
 
   /// \brief Wait (blocking) for next synchronised multiframe.
   bool getNextFrame(MultiFramePtr &multiFrame);
 
   /// \brief Performs the optimisation, publishing&visualisation preps, and marginalisation.
   void optimisePublishMarginalise(MultiFramePtr multiFrame, const Eigen::Vector3d& gyroReading);
 
   /// \brief Performs visualisation in the background.
   void visualisationLoop();
 
   /// \brief Performs publishing in the background.
   void publishingLoop();
 
   /// \brief Process depth measurement deque to obtain proper input for frame-to-map factors
   void computeLiveDepthMeasurements(const kinematics::Transformation& T_WS_live, const okvis::Time& curTime,
                                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints, std::vector<float>& observedSigmas);
 
   std::atomic_bool blocking_; ///< Blocking or non-blocking operation mode?
 
   /// @name Measurement input queues
   /// @{
 
   /// Camera measurement input queues. For each camera in the configuration one.
   threadsafe::Queue<std::vector<okvis::CameraMeasurement> > cameraMeasurementsReceived_;
 
   /// IMU measurement input queue.
   threadsafe::Queue<okvis::ImuMeasurement, Eigen::aligned_allocator<okvis::ImuMeasurement>>
       imuMeasurementsReceived_;
 
   /// Lidar measurement input queue.
   threadsafe::Queue<okvis::LidarMeasurement , Eigen::aligned_allocator<okvis::LidarMeasurement>>
       lidarMeasurementsReceived_;
 
   /// Depth-sigma measurement input queue.
   threadsafe::Queue<okvis::CameraMeasurement, Eigen::aligned_allocator<okvis::CameraMeasurement>>
       depthMeasurementsReceived_;
 
   /// Input Queue of Submap alignment measurements
   struct AlignmentTerm{
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
       const SupereightMapType* submap_A_ptr; ///< Reference Submap
       const SupereightMapType* submap_B_ptr; ///< Recently finished submap
       uint64_t frame_A_id; ///< ID of keyframe A (submap to be referred to (previous submap))
       uint64_t frame_B_id; ///< ID of keyframe B  (current new submap)
       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud_B; ///< Point cloud with points in keyframe B that are aligned to submap in frame A
       std::vector<float> sensorError; ///< Uncertainty value for the LiDAR / Depth Sensor [in meters]
 
       AlignmentTerm()
       {};
 
       AlignmentTerm(const SupereightMapType* submap_A_ptr,
                     const SupereightMapType* submap_B_ptr,
                     const uint64_t& frame_A_id, const uint64_t& frame_B_id,
                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud_B,
                     std::vector<float> sensorError)
               :  submap_A_ptr(submap_A_ptr), submap_B_ptr(submap_B_ptr),
               frame_A_id(frame_A_id), frame_B_id(frame_B_id), pointCloud_B(pointCloud_B), sensorError(sensorError)
       {
       };
   };
   threadsafe::Queue<AlignmentTerm, Eigen::aligned_allocator<AlignmentTerm>>
       submapAlignmentFactorsReceived_;
 
   /// GPS measurement input queue
   threadsafe::Queue<okvis::GpsMeasurement> gpsMeasurementsReceived_;

  struct GaRLILEOMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::Time timeStamp;
    Eigen::Quaterniond R_WS;
    Eigen::Vector3d v_body;
    Eigen::Vector3d p_W;
  };
  threadsafe::Queue<GaRLILEOMeasurement> garlileoMeasurementsReceived_;
  std::deque<GaRLILEOMeasurement> garlileoHistory_;
  GaRLILEOMeasurement lastGarlileoMeas_;
  std::atomic_bool hasLastGarlileo_{false}; // Flag is read before taking the payload mutex.
  std::mutex garlileoMutex_;

  // Book-keeping to enable backfilling GaRLILEO between constraints when late messages arrive.
  struct GaRLILEOFrameQuality {
    bool low_features = false;
    bool tracking_lost = false;
    /// max(utility_blend, early tracking-lost) in [0,1] for reproducing semi-band / GaRLILEO
    /// blend when backfilling between or velocity factors.
    double vision_fallback_blend = 0.0;
  };
  struct GaRLILEOBetweenAssoc {
    double max_abs_dt = std::numeric_limits<double>::infinity(); ///< max(|dt_prev|,|dt_curr|) used for this pair.
    bool has_constraint = false;
  };
  struct GaRLILEOVelocityAssoc {
    double abs_dt = std::numeric_limits<double>::infinity(); ///< |dt| used for this state.
    bool has_constraint = false;
    bool retry_pending = false; ///< Backend rejected a replacement/removal; preserve the applied association.
  };
  // Keyed by current frame id k (represents the pair (k-1,k)).
  std::map<uint64_t, GaRLILEOFrameQuality> garlileoFrameQualityById_;
  std::map<uint64_t, GaRLILEOBetweenAssoc> garlileoBetweenAssocByCurrId_;
  // Keyed by frame id k (unary velocity/speed prior on state k).
  std::map<uint64_t, GaRLILEOVelocityAssoc> garlileoVelocityAssocById_;

  // SLAM-thread history: once a non-unit visual scale or path suppression was
  // possible, future reset calls must also cover late inserted/cloned edges.
  // Keep this true thereafter; the previous request alone is not sufficient.
  bool visualRelposeScaleMayNeedReset_ = false;

  struct GaRLILEOGravityMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::Time timeStamp;
    Eigen::Vector3d g_W;
  };
  threadsafe::Queue<GaRLILEOGravityMeasurement> garlileoGravityMeasurementsReceived_;
  std::deque<GaRLILEOGravityMeasurement> garlileoGravityHistory_;
  GaRLILEOGravityMeasurement lastGarlileoGravityMeas_;
  std::atomic_bool hasLastGarlileoGravity_{false}; // Payload remains mutex-protected.
  std::mutex garlileoGravityMutex_;

  // Frontend GaRLILEO relative-delta state. Kept as class state (instead of function-local statics)
  // so we can safely reset it on re-initialisation gaps / stale measurements.
  bool frontendDeltaHasPrev_{false};
  GaRLILEOMeasurement frontendDeltaPrevMeas_;
  okvis::kinematics::Transformation frontendDeltaPrevOkvisPose_;

  struct RadarTargetsMeasurement {
    okvis::Time timeStamp;
    okvis::RadarTargetReadings targets;
    int radarIndex = 0;
  };
  threadsafe::Queue<RadarTargetsMeasurement> radarTargetsMeasurementsReceived_;
  std::deque<RadarTargetsMeasurement> radarTargetsHistory_;
  RadarTargetsMeasurement lastRadarTargetsMeas_;
  bool hasLastRadarTargets_{false};
  std::mutex radarTargetsMutex_;
 
   /// The queue containing the matching data
   threadsafe::Queue<ViVisualizer::VisualizationData::Ptr> visualisationData_;
 
   /// The queue containing the actual overhead display images
   threadsafe::Queue<cv::Mat> overheadImages_;
 
   /// The queue containing the actual matching display images
   threadsafe::Queue<std::vector<cv::Mat>> visualisationImages_;

  /// The queue containing the radar polar (40x40) debug image for RViz display.
  threadsafe::Queue<cv::Mat> radarPolarImages_;
 
  /// The queue containing the AGCWD-preprocessed images (one per camera) for RViz display.
  threadsafe::Queue<std::vector<cv::Mat>> agcwdImages_;

   ViVisualizer visualizer_; ///< The matching visualiser.
 
   /// @}
 
   /// @name Consumer threads
   /// @{
 
   std::thread visualisationThread_; ///< Thread running visualisation.
   std::thread publishingThread_; ///< Thread running publishing.
 
   /// @}
   /// @name Algorithm threads
   /// @{
 
   std::thread optimisationThread_; ///< Thread running optimisation.
   std::thread fullGraphOptimisationThread_; ///< Thread running loopclosure full graph optimisation.
 
   /// @}
 
   // publishing:
   /// \brief Realtime publishing data struct.
   struct PublicationData {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     State state; ///< The current state.
     TrackingState trackingState; ///< The tracking state.
     std::shared_ptr<AlignedMap<StateId, State>> updatedStates; ///< Updated states (may be empty).
     std::shared_ptr<MapPointVector> landmarksPublish; ///< The landmarks.
   };
   threadsafe::Queue<PublicationData> lastOptimisedQueue_; ///< Queue of last optimised results.
   threadsafe::Queue<PublicationData> publicationQueue_; ///< Publication queue.
   std::set<StateId> affectedStates_; ///< Affected states (changed pose graph edges) since pub.
 
   // realtime publishing:
   std::atomic_bool hasStarted_; ///< Has it started yet?
 
  // Frame-drop diagnostics (camera pipeline).
  std::atomic<double> lastCameraQueuePopWallTimeSec_{-1.0};
  std::atomic<double> lastOptimisationDurationSec_{-1.0};
  std::atomic<double> lastOptimisationFinishedWallTimeSec_{-1.0};
  std::atomic<double> lastIncomingCameraStampSec_{-1.0};

  struct EwgUtilityMinRecord {
    uint64_t frame_id = 0;
    double timestamp_sec = std::numeric_limits<double>::quiet_NaN();
    double utility_min = std::numeric_limits<double>::quiet_NaN();
    int utility_cams = 0;
  };
  std::vector<EwgUtilityMinRecord> ewgUtilityMinRecords_;
  mutable std::mutex ewgUtilityMinRecordsMutex_;

  /// \brief One optimised IMU bias estimate, recorded for the bias-consistency comparison
  /// against GaRLILEO (which estimates b_a only, with b_g held at zero).
  struct ImuBiasRecord {
    uint64_t frame_id = 0;
    double timestamp_sec = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d b_g = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d b_a = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    bool is_keyframe = false;
  };
  std::vector<ImuBiasRecord> imuBiasRecords_;
  mutable std::mutex imuBiasRecordsMutex_;

   // processing loop variables:
   bool firstFrame_ = true; ///< Is it the first frame?
   ImuMeasurementDeque imuMeasurementDeque_;  ///< Stored IMU measurements to be used next.
   GpsMeasurementDeque gpsMeasurementDeque_;  ///< Stored GPS Measurements to be used next.
   LidarMeasurementDeque lidarMeasurementDeque_;  ///< Stored lidar Measurements to be used next.
   DepthMeasurementDeque depthMeasurementDeque_; ///< Stored depth Measurements to be used next.
 
   /// \brief Stored IMU measurements by frame.
   AlignedMap<StateId, ImuMeasurementDeque> imuMeasurementsByFrame_;

   /// \brief Add GaRLILEO between unless fallback-LC policy suppresses this state pair; may remove stale between.
   /// \return True if a between constraint was added.
   bool applyGarlileoBetweenIfAllowed(StateId poseId0, StateId poseId1,
                                     const okvis::kinematics::Transformation& T_S0S1,
                                     const Eigen::Matrix<double, 6, 6>& information);
 
   /// Update association bookkeeping only after the backend accepts the whole operation.
   bool updateGarlileoVelocityPrior(StateId poseId, const Eigen::Vector3d& v_S,
                                    double stddev, double abs_dt, bool enabled);

   /// @name Algorithm objects.
   /// @{
 
   okvis::ViSlamBackend estimator_;    ///< The backend estimator.
   okvis::Frontend frontend_;      ///< The frontend.
   std::unique_ptr<StereoDetectionWorker> stereoDetectionWorker_; ///< Lazy BO-off stereo worker.
   okvis::Trajectory trajectory_;
   std::ofstream trajectory_debug_output_;
   std::ofstream base_frame_debug_output_;
   /// @}
 
   okvis::ViParameters parameters_; ///< The parameters and settings.
 
   std::string finalTrajectoryCsvFileName_; ///< File name of final trajectory log.
   bool rpg_ = false; ///< Use the RPG format for logs rather than EuRoC.
   std::string mapCsvFileName_; ///< The map file name.
   
   ::ceres::Solver::Summary posegraphOptimisationSummary_; ///< Posegraph optimisation summary.
 
   State lastOptimisedState_; ///< Store at least pose/id/timestamp here to avoid race on estimator_.
   State preLastOptimisedState_; ///< Store at previous to last state here to avoid race.
 
   std::atomic_bool shutdown_; ///< Has shutdown been called?
 
   se::SubMapConfig submapConfig_; ///< Submapping Config (e.g. alignment settings, kf thresholds,...)
   okvis::kinematics::Transformation T_SD_; /// < Transformation {S} IMU <-> {D} Depth source (Lidar or Camera)
   okvis::kinematics::Transformation T_rect_; ///< T_{D}{D_rectification}
   bool useAlignmentFactors_;
   const SupereightMapType* previousSubmap_ = nullptr; ///< Always have a reference to the most recent completed submap
   uint64_t previousSubmapId_ = -1;
   size_t previouslyObservedLidarPoints_ = 0;
   size_t noOverlapCounter_ = 0;
   std::set<StateId> lidarKeyframes_;
 
   AlignmentPublishCallback alignmentPublishCallback_;
 
   int sensorMeasurementDownsamplingCounter_ = 0;
   std::shared_ptr<se::PinholeCamera> depthCamera_;
 
   void saveAlignedPoints(size_t id,
     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points,
     std::string type);
 };
 
 }  // namespace okvis
 
 #endif /* INCLUDE_OKVIS_THREADEDSLAM3_HPP_ */
