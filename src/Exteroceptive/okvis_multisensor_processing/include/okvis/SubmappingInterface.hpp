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

#ifndef INCLUDE_OKVIS_SUBMAPPINGINTERFACE_HPP
#define INCLUDE_OKVIS_SUBMAPPINGINTERFACE_HPP

#include <okvis/ThreadedSlam.hpp>
#include <okvis/VoxelGridFilter.hpp>
#include <okvis/SubmappingUtils.hpp>
#include <okvis/mapTypedefs.hpp>
#include <Eigen/StdVector>
#include <mutex>


namespace okvis {

  // Some convenient typedefs
  typedef se::Image<float> DepthFrame;
  typedef okvis::threadsafe::Queue<std::map<size_t, std::vector<okvis::CameraMeasurement>>> DepthFrameQueue;
  typedef okvis::threadsafe::Queue<LidarMeasurement, Eigen::aligned_allocator<LidarMeasurement>> LidarMeasurementQueue;
  typedef okvis::kinematics::Transformation Transformation;
  /**
  * @brief Extended okvis update: contains all the things we need for depth integration.
  *
  */
    struct OkvisUpdate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        State latestState;
        okvis::Time timestamp;
        TrackingState trackingState;
        std::set<StateId> affectedStates;
        std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates;

        OkvisUpdate(const State &latestState = State(),
                    const okvis::Time &timestamp = okvis::Time(),
                    const TrackingState &trackingState = TrackingState(),
                    const std::set<StateId> affectedStates = std::set<StateId>(),
                    std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates = std::make_shared<const okvis::AlignedMap<StateId, State>>(okvis::AlignedMap<StateId, State>()))
                : latestState(latestState), timestamp(timestamp),
                  trackingState(trackingState), affectedStates(affectedStates), updatedStates(updatedStates) {};
        
        OkvisUpdate(const OkvisUpdate& newElem) = default;

        OkvisUpdate& operator=(const OkvisUpdate& newElem){
            latestState = newElem.latestState;
            timestamp = newElem.timestamp;
            trackingState = newElem.trackingState;
            affectedStates = newElem.affectedStates;
            updatedStates = newElem.updatedStates;
            return *this;
        }
    };

    typedef okvis::threadsafe::Queue<OkvisUpdate, Eigen::aligned_allocator<OkvisUpdate>> StateUpdatesQueue;

/**
 * @brief Contains essential data about a keyframe: its Id and transformation.
 *
 */
    struct SiFrameData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t id;
        Transformation T_WS;
        std::set<StateId> covisibles_;

        SiFrameData(const uint64_t &id = 0,
                    const Transformation &T_WS = Transformation::Identity(),
                    const std::set<StateId> &covisibles = std::set<StateId>())
                : id(id), T_WS(T_WS), covisibles_(covisibles) {};
    };

    typedef std::map<uint64_t, SiFrameData, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, SiFrameData>>> siFrameDataMap;


/**
 * @brief Contains the data required for a single supereight map integration
 * step. each seframe contains the entire list of keyframes with updated poses
 *
 */
    typedef std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>, 
              Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>> RayVector;

    struct SupereightFrames {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t keyFrameId; // id of current Supereight Map
        siFrameDataMap frameData_;
        bool loop_closure;
        unsigned int stateCounter_;
        RayVector vecRayMeasurements;
        std::vector<okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>>> vecDepthFrames;

        SupereightFrames(const uint64_t &keyframeId = 0,
                        const siFrameDataMap &FrameDataMap = siFrameDataMap{},
                        const bool loop_closure = false,
                        std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector3f>, Eigen::aligned_allocator<std::pair<Eigen::Matrix4f, Eigen::Vector3f>>>* vecRayMeasurementsVal = nullptr,
                        std::vector<okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Matrix4f, CameraMeasurement>>>>* vecDepthFramesVal = nullptr)
                          : keyFrameId(keyframeId), frameData_(FrameDataMap), 
                            loop_closure(loop_closure), stateCounter_(0) {

                        if(vecRayMeasurementsVal){
                          for(size_t i = 0; i < vecRayMeasurementsVal->size(); i++){
                            vecRayMeasurements.push_back({Eigen::Isometry3f(vecRayMeasurementsVal->at(i).first),
                                                          vecRayMeasurementsVal->at(i).second});
                          }
                        }

                        if(vecDepthFramesVal){
                          for(size_t i = 0; i < vecDepthFramesVal->size(); i++){
                            okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>> measurement;
                            for(auto& elements : vecDepthFramesVal->at(i)){
                              auto& posed_idx_measurements = measurement[elements.first];
                              for(auto& element : elements.second) {
                                posed_idx_measurements.emplace_back(std::make_pair(Eigen::Isometry3f(element.first), element.second));
                              }
                            }
                            vecDepthFrames.push_back(measurement);
                          }
                        }

                  };
    };

    struct SubmapAlignmentTerm{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t frame_A_id; ///< ID of keyframe A (submap to be referred to (previous submap))
        uint64_t frame_B_id; ///< ID of keyframe B  (current new submap)
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pointCloud_B; ///< Point cloud with points in keyframe B that are aligned to submap in frame A
        std::vector<float> sigma_B; /// 1-sigma of depth measurement of pointCloud_B
        VoxelHashMap voxelHashMap;

        SubmapAlignmentTerm() : frame_A_id(0), frame_B_id(0), voxelHashMap(0.06, 0.0, 1)
        {
        };

        SubmapAlignmentTerm(const uint64_t& frame_A_id, const uint64_t& frame_B_id)
          : frame_A_id(frame_A_id), frame_B_id(frame_B_id), voxelHashMap(0.06, 0.0, 1)
        {
        };

        SubmapAlignmentTerm(const uint64_t& frame_A_id, const uint64_t& frame_B_id, float voxel_grid_resolution)
          : frame_A_id(frame_A_id), frame_B_id(frame_B_id), voxelHashMap(voxel_grid_resolution, 0.0, 1)
        {
        };

    };


    typedef okvis::threadsafe::Queue<SupereightFrames, Eigen::aligned_allocator<SupereightFrames>> SupereightFrameQueue;
    typedef std::function<void(AlignedUnorderedMap<uint64_t, Transformation>,
                               std::unordered_map<uint64_t, std::shared_ptr<SupereightMapType>>)> submapCallback;
    typedef std::function<void(const State&,
                               const AlignedUnorderedMap<uint64_t, se::Submap<SupereightMapType>>&)> fieldCallback;
    typedef std::function<void(const okvis::State&, const AlignedVector<Eigen::Matrix4f>&, const std::vector<uint64_t>&, const size_t)> trajectoryAnchoringCallback;
    typedef std::function<void(const okvis::State&, std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>>)> anchoredTrajectoryUpdateCallback;
    typedef std::function<void(const okvis::Time&, const okvis::kinematics::Transformation&,
                               const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&, bool)> alignmentPublishCallback;
    typedef std::function<void(const okvis::Time&, const Eigen::Vector3d&, const Eigen::Quaterniond&,
                               const Eigen::Vector3d&, const Eigen::Vector3d&)> realTimePublishingCallback;
    typedef std::function<void(const std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                                Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>>& rayPoseBatch)> IntegrationPublishCallback;

    class SubmappingInterface {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static constexpr size_t UNINITIALIZED_ID = std::numeric_limits<size_t>::max();

        SubmappingInterface() = delete;

        /**
         * @brief      Constructs a new instance.
         *
         * @param[in]  cameraConfig Pinhole camera configuration.
         * @param[in]  mapConfig Map configuration.
         * @param[in]  dataConfig Occupancy mapping configuration.
         * @param[in]  T_SC Homogenous transformation from sensor to depth camera.
         * @param[in]  mapConfig Map configuration.
         * @param[in]  meshesPath  Directory where we store map meshes.
         */
        SubmappingInterface(
                const SupereightMapType::Config &mapConfig,
                const SupereightMapType::DataConfigType &dataConfig,
                const se::SubMapConfig &submapConfig,
                const okvis::ViParameters &parameters)
                : mapConfig_(mapConfig), dataConfig_(dataConfig), submapConfig_(submapConfig),
                  meshesPath_(submapConfig.resultsDirectory),
                  kfThreshold_(submapConfig.submapKfThreshold),
                  viParameters_(parameters),
                  sensorMeasurementDownsampling_(submapConfig.sensorMeasurementDownsampling),
                  depthImageResDownsamplingRatio_(submapConfig.depthImageResDownsampling),
                  depthScaleFactor_(submapConfig.depthScalingFactor){

          blocking_ = false;
          
          //Quite a hacky way of doing this. Problem is that as the variable the pointers point at run afterwards 
          //in a different thread, they are getting deleted once we get out of scope. If anyone has got a better approach/proposal
          // I am quite open as to improve this initialization

          bool camInMapping = false;

          for(size_t i = 0; i < parameters.nCameraSystem.numCameras(); i++){
            if(parameters.nCameraSystem.cameraType(i).isUsedMapping) {
              LOG(INFO) << "Camera with index " << i << " will be used for mapping";
              camInMapping = true;
              std::shared_ptr<const cameras::CameraBase> camera;
              if (!parameters.nCameraSystem.cameraType(i).depthType.needRectify) {
                camera = parameters.nCameraSystem.cameraGeometry(i);
              } else {
                camera = parameters.nCameraSystem.rectifyCameraGeometry(i);
              }
              se::PinholeCamera::Config camConfig;
              Eigen::VectorXd intrinsics;
              camera->getIntrinsics(intrinsics);

              camConfig.fx = intrinsics(0);
              camConfig.fy = intrinsics(1);
              camConfig.cx = intrinsics(2);
              camConfig.cy = intrinsics(3);
              camConfig.T_BS = parameters.nCameraSystem.T_SC(i)->T().cast<float>();
              camConfig.width = camera->imageWidth();
              camConfig.height = camera->imageHeight();
              camConfig.near_plane = submapConfig.near_plane;
              camConfig.far_plane = submapConfig.far_plane;

              if(!cameraSensors_) {
                cameraSensors_ = okvis::AlignedMap<uint64_t,  std::pair<std::shared_ptr<const okvis::kinematics::Transformation>, se::PinholeCamera>>();
              }
              if(!parameters.nCameraSystem.cameraType(i).depthType.needRectify) {
                (*cameraSensors_).emplace(std::make_pair(i,
                                       std::make_pair(parameters.nCameraSystem.T_SC(i), 
                                                      se::PinholeCamera(camConfig, 
                                                                        submapConfig.depthImageResDownsampling))));
              } else {
                (*cameraSensors_).emplace(std::make_pair(i,
                                       std::make_pair(parameters.nCameraSystem.rectifyT_SC(i), 
                                                      se::PinholeCamera(camConfig, 
                                                                        submapConfig.depthImageResDownsampling))));
              }
              
            }

          }

          if(camInMapping && parameters.lidar) {
            throw std::runtime_error("There are LiDAR and cameras with mapping enabled in your config file, currently there is only support for either depth cameras or LiDAR at the same time");
          } else if(parameters.lidar) {
            submapAlignBlock_ = SubmapAlignmentTerm(0, 0, submapConfig.voxelGridResolution);
            //Generate the LiDAR sensor
            se::Lidar::Config lidarConfig;
            lidarConfig.elevation_resolution_angle_ = (*parameters.lidar).elevation_resolution_angle;
            lidarConfig.azimuth_resolution_angle_ = (*parameters.lidar).azimuth_resolution_angle;
            lidarConfig.far_plane = submapConfig.far_plane;
            lidarConfig.near_plane = submapConfig.near_plane;
            lidarSensors_ = std::pair<okvis::kinematics::Transformation, se::Lidar>(std::make_pair((*parameters.lidar).T_SL,
                                                                                                         se::Lidar(lidarConfig)));
          }
        };

        /**
         * @brief      Destroys the object.
         */
        ~SubmappingInterface() {
          stopIntegrationThreads();
        };

        /**
         * @brief     Adds a lidar measurement to the measurement Queue.
         * @param[in] stamp     The timestamp.
         * @param[in] ray       The lidar measurement
         *
         * @return    True if successful.
         */
        bool addLidarMeasurement(const okvis::Time &stamp, const Eigen::Vector3d &ray);

        /**
         * @brief Adds a depth image to the measurement Queue
         * @param[in] frames All the information from all the cameras needed for the submapping interface
         * 
         * @return True if successful
        */
        bool addDepthMeasurement(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames);

        /**
         * \brief          Add an IMU measurement for propagation and publishing.
         * Calls must be serialized by the IMU producer. Trajectory access is
         * synchronized with the mapping thread; queued IMU samples are retained
         * when that thread holds the trajectory mutex.
         * \param stamp    The measurement timestamp.
         * \param alpha    The acceleration measured at this time.
         * \param omega    The angular velocity measured at this time.
         * \return True on success.
         */
        bool realtimePredict(const okvis::Time& stamp,
                             const Eigen::Vector3d& alpha,
                             const Eigen::Vector3d& omega);


        /**
         * @brief      Starts the processing and data prepatation threads.
         *
         * @return     True when successful.
         */
        bool start();

        /**
         * @brief      Stores the state and keyframe updates provided by OKVIS
         *
         * @param[in]  latestState          The current OKVIS state
         * @param[in]  latestTrackingState  The current tracking state
         * @param[in]  keyframeStates       The state of the updated Keyframes
         *
         * @return     True when successful
         */
        bool stateUpdateCallback(const State &latestState,
                                 const TrackingState &latestTrackingState,
                                 std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates);
                                 

        /**
         * @brief      Gets the size of the to-be-processed supereight frames.
         *
         * @return     The supereight queue size.
         */
        size_t getSupereightQueueSize() { return supereightFrames_.Size(); };

        /**
         * @brief      Set blocking/not blocking mode.
         *
         */
        void setBlocking(bool blocking) {
          blocking_ = blocking;
        }

        /**
         * @brief      Main function of the processing thread. It integrates the
         * assembled supereigh frames (i.e. depth/lidar ray + pose) and creates a new submap
         * when required
         */
        void processSupereightFrames();

        /**
         * @brief       Checks if required data (LiDAR measurements + OKVIS estimates) is available and prepares for
         * processing / integration into the map
         *
         * @return      True, if data available. False otherwise.
         */
        bool checkForAvailableData();

        /**
         * @brief Function which updates the supereight frame with the new poses of the keyframes and metadata regarding loop closures and keyframeID
         * @param[in] propagatedStateUpdate okvis update with all the new information of the current state which has been processed
        */
        void frameUpdater(okvis::OkvisUpdate& propagatedStateUpdate);

        void collectInfo(){
            std::cout << "There are " + std::to_string(stateUpdates_.Size()) + " messages in the state queue" << std::endl;
            std::cout << "There are " + std::to_string(supereightFrames_.Size()) + " messages in the supereight queue" << std::endl;
            //std::cout << "There are " + std::to_string(lidarMeasurements_.Size()) + " messages in the lidar measurements queue" << std::endl;
            std::cout << "There are " + std::to_string(depthMeasurements_.Size()) + " messages in the depth camera measurements queue" << std::endl;
            std::map<size_t, std::vector<CameraMeasurement>> oldestDepthImage;
            OkvisUpdate toPopState;
            OkvisUpdate newestState;
            if(stateUpdates_.getCopyOfFront(&toPopState)){
              stateUpdates_.getCopyOfBack(&newestState);
              std::cout << "Earliest timestamp from the states is " << toPopState.timestamp << std::endl;
              std::cout << "Most recent timestamp from the states is " << newestState.timestamp << std::endl;
            }

            if(depthMeasurements_.getCopyOfFront(&oldestDepthImage)) {
              std::cout << "Earliest timestamp from the depth image is " << oldestDepthImage.at(0).at(0).timeStamp << std::endl;
            }

        }

        /**
         * @brief      Set function that handles submaps visualization (blocks version)
         *
        */
        void setSubmapCallback(const submapCallback &submapCallback) { submapCallback_ = submapCallback; }

        void setFieldSliceCallback(const fieldCallback &fieldCallback) { fieldCallback_ = fieldCallback; }

        /**
         * @brief Function that sets the trajectory anchoring callback. When a path is planned, the trajectory is then anchored with the callback function
         * @param[in] trajCallback the trajectory anchoring callback function
        */
        void setTrajectoryAnchoringCallback(const trajectoryAnchoringCallback &trajCallback) {trajectoryAnchoringCallback_ = trajCallback;}

        /**
         * @brief Function to set the update callback for trajectory anchoring
         * @param[in] anchoredUpdateCallback the callback to be used for updating the anchors in trajectory anchoring
        */
        void setAnchoredTrajectoryUpdateCallback(const anchoredTrajectoryUpdateCallback &anchoredUpdateCallback) {anchoredTrajectoryUpdateCallback_ = anchoredUpdateCallback;};

        /**
         * @brief      Launch visualization thread.
         *
        */
        void publishSubmaps(AlignedUnorderedMap<uint64_t, Transformation> submapPoses,
                            std::unordered_map<uint64_t, std::shared_ptr<SupereightMapType>> submaps);

        /**
         * @brief Function which does the saving of submaps into meshes
         * @param[in] kfId the Id of the submaps keyframe we are going to store
        */
        void saveSubmap(uint64_t kfId);

        void updateSEFrameData(siFrameDataMap& seFData, siFrameDataMap& newFData){
            for(auto it : newFData) seFData[it.first] = it.second;
            return;
        }

        void setT_BS(const okvis::kinematics::Transformation& T_BS);

        // To access maps
        AlignedUnorderedMap<uint64_t, se::Submap<SupereightMapType>> seSubmapLookup_; // Use this to access submaps and their poses
        AlignedUnorderedMap<uint64_t, Eigen::Matrix<float, 6, 1>> submapDimensionLookup_; // use this when reindexing maps on loop closures (index,dims)

        AlignedUnorderedMap<uint64_t, se::TriangleMesh<SupereightMapType::DataType::col_,SupereightMapType::DataType::id_>> seMeshLookup_; ///< Lookup for meshes, need to be stored here to use them for Feature Thing v2

        /**
         * @brief Function to know if there are still messages to be processed or not
         * @return true if finished false if there is still data to be processed
        */
        bool finishedIntegrating();
        
        /**
         * @brief Function to check if the integration of the interface has finished
        */
        bool isFinished(){
            std::lock_guard<std::mutex> l(finishMutex_);
            return isFinished_;
        }

        /**
         * @brief Function that sets the finished boolean to true
        */
        void setFinished();

        /// @brief Set the callback for tightly coupling alignment terms to OKVIS
        /// @param imuCallback The IMU callback to register.
        typedef std::function<
                bool(const SupereightMapType* submap_A_ptr,
                     const SupereightMapType* submap_B_ptr,
                     const uint64_t& frame_A_id,
                     const uint64_t& frame_B_id,
                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                     std::vector<float> sensorError)> AlignCallback;

        void setAlignCallback(const AlignCallback& alignCallback) {
          alignCallback_ = alignCallback;
        }

        /**
         * @brief Publish points that are used in frame-to-map factors
        */
        void setAlignmentPublishCallback(const alignmentPublishCallback& alignPublishCallback){
          alignmentPublishCallback_ = alignPublishCallback;
        }

        void setIntegrationPublishCallback(const IntegrationPublishCallback& integrationPublishCallback){
          integrationPublishCallback_ = integrationPublishCallback;
        }

        /**
         * @brief Publish Real-Time Poses (IMU Propagation): position, orientation, linear and angular velocity
        */
        void setRealtimePublishCallback(const realTimePublishingCallback& realTimePublishCallback){
          realTimePublishCallback_ = realTimePublishCallback;
        }

        void setOdometryPublishingRate(float odometry_publishing_rate) {
          odometryPublishingRate_ = odometry_publishing_rate;
        }


        /**
         * @brief Function to add groundtuth poses
         * @param[in] t timestamp that is associated to the current pose
         * @param[in] transform the transform T_WS associated to the robot at time t
        */
        void addGtPoses(okvis::Time t, okvis::kinematics::Transformation transform){
            std::lock_guard<std::mutex> l(gtDebug_);
            gtPoses_[t.toNSec()] = transform;
        }
        
        /**
         * @brief Function that provided a timestamp returns the groundtruth pose associated with that timestamp
         * if no pose is available, false will be returned
         * @param[in] t timestamp for which we want to query a specific pose
         * @param[out] bool boolean which states whether a pose for the required timestamp was available. No interpolation is allowed
        */
        bool getGtPoses(okvis::Time t, okvis::kinematics::Transformation &T_WS){
            std::lock_guard<std::mutex> l(gtDebug_);
            
            if(gtPoses_.find(t.toNSec()) != gtPoses_.end()){
                T_WS = gtPoses_[t.toNSec()];
                return true;
            }
            return false;
        }
        
        /**
         * @brief function to set that the submapping interface use ground truth poses instead of OKVIS poses
         * This could be simulated odometry or VICON/OptiTrack
         * @param[in] flag boolean flag that allows to use groundtruth poses
        */
        void setUseGtPoses(bool flag){
            useGtPoses_ = flag;
            return;
        }

        void generatePointCloud(const Eigen::Matrix4f& T_WS, const DepthFrame& dImage, const se::PinholeCamera& sensor, int counter);

        /**
         * @brief Function that stores raw pointclouds from lidar measurements
         * 
         * @param[in] lidarMeas Vector of pairs of (T_WL, point) which contain the transform of lidar to world and the lidar measurement
         * @param[in] counter integer to give a name to the generated mesh
        */
        void generatePointCloudFromLidar(const std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector3f>,
                Eigen::aligned_allocator<std::pair<Eigen::Matrix4f, Eigen::Vector3f>>>& lidarMeas, int counter);
        
        /**
         * @brief Function that saves all submaps meshes
        */
        void saveAllSubmapMeshes();

        /**
         *
         * @param downsampledPointCloud
         * @param activeId Active submap Id, will be excluded from the check
         * @return The ID of the most overlapping submap (based on IoU of convex hull and aabb of submap)
         */
        uint64_t findMostOverlappingSubmap(std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& downsampledPointCloud,
                                           uint64_t activeId, uint64_t previousId);

        /**
         * @brief Find point_B that is observed in map_A
        */
        size_t determineObservedPoints(uint64_t& mapIdA, uint64_t& mapIdB,
                                       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints);

        /**
         * @brief Find point_B and associated uncertainty that is observed in map_A
        */
        size_t determineObservedPointsAndSigma(uint64_t& mapIdA, uint64_t& mapIdB,
                                               std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                               std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints,
                                               std::vector<float>& sigma, std::vector<float>& observedSigma);

        /**
         * @brief Computing volume overlap between map and seFrame by frustrum sampling.
         * @param map Previously built submap
         * @param mapId (visual keyframe) ID associated to the map
         * @param seFrame SupereightFrame containing live depth frame
        */
        float evaluateDepthOverlap(const SupereightMapType& map, uint64_t& mapId, SupereightFrames& seFrame);

        /**
         * @brief Function that computes overlap of a Lidar Point Cloud with current active submap to decide new submap creation
         * @param map Current active submap
         * @param mapId (visual keyframe) ID associated to the map
         * @param seFrame SupereightFrame containing batch of ray measurements
         * @param overlapThreshold Threshold for minimum overlap (if overlap < threshold new submap will be created)
         * @return Ratio of observed (considered overlapping) points to total number of measurements
         */
        double evaluateLidarOverlap(const SupereightMapType& map, uint64_t& mapId,
                                    SupereightFrames& seFrame, double overlapThreshold = 0.5);

        /**
         * @brief Function that warps depth image to color camera
         * @param depthData Pair containing Pose of depth camera and the depth measurement itself
         * @param rgbData Pair containing Pose of RGB camera and the RGB measurement itself
         * @param depthCamIdx Index of depth camera
         * @param colorCamIdx Index of RGB camera
         * @param[out] warped_depth_image The depth image warped into RGB frame
         */
        bool warpDepthImageToColor(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
                                   const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& rgbData,
                                   const int depthCamIdx, const int colorCamIdx, 
                                   cv::Mat& warped_depth_image);
        
        /**
         * @brief function that returns a reference to the okvis::Trajectory object of the interface.
         * Only use while mapping and IMU processing are stopped. The returned
         * reference does not retain the internal trajectory mutex.
        */
        const okvis::Trajectory& getOkvisTrajectory() const {return propagatedStates;};

        void printAssociations(){
            DLOG(INFO) << "--- Map-to-Map associations ---\n";
            for(auto association : map_to_map_association_){
                DLOG(INFO) << "Associating map " << association.first << " to " << association.second << "\n";
            }
        }

        void drawSubmaps();
        bool publishSubmapTopView(cv::Mat& submapPlot);

    private:

        /**
         * @brief Loop that performs the supereight frame data collection and integrates it
        */
        void integrationLoop();

        // Called only while holding dataIntegrationMutex_. Counts queued and
        // already-popped frames until the consumer completes all map work.
        bool enqueueSupereightFrame(size_t maxQueueSize);
        void stopIntegrationThreads();

        
        /**
         * @brief      Given time stamp of depth frame, propagates latest okvis update to get pose estimate. It also gets from the states updates queue:
         * vector of updated keyframes, latest keyframe id, loop closure status.
         *
         * @param[in]  finalTimestamp  Timestamp of the depth frame.
         * @param[in]   T_SC  Extrinsics between IMU and depth image
         * @param[out]  T_WC  Predicted depth frame pose w.r.t. world.
         */
        bool predict(const okvis::Time &finalTimestamp,
                     const Transformation &T_SC,
                     Transformation &T_WC);
        
        /**
         * @brief Convert cv::Mat images into se::Image which is the input for Supereight
         * @param[in] inputDepth cv::Mat depth image data which is going to be converted to se::Image<float> for Supereight
         *         
         * @return DepthFrame which is a typedef from se::Image<float>
         */ 
        DepthFrame depthMat2Image(const cv::Mat &inputDepth);

        /**
         * @brief Convert cv::Mat images into se::Image which is the input for Supereight
         * @param[in] inputRGB cv::Mat rgb image data which is going to be converted to se::Image<se::colour_t> for Supereight
         * @return se::Image<se::colour_t>
         */ 
        se::Image<se::colour_t> rgbMat2Image(const cv::Mat &inputRGB);

        /**
         * @brief   Transform the submap's AABB from submap frame to world frame and get the new bounds.
         *
         * @param[in]  min_coords_W  Output: vector for storage of AABB's minimal bounds in world frame.
         * @param[in]  max_coords_W  Output: vector for storage of AABB's maximal bounds in world frame.
         * @param[in]  vert_coords_K  Input: coordinates of AABB's 8 vertices in submap frame. Use K as it is anchored to a keyframe
         * @param[in]  T_WK  Transformation from submap frame to world frame. Use K as it is anchored to a keyframe
         */
        void aabbTransform(Eigen::Vector3f& min_coords_W, Eigen::Vector3f& max_coords_W,
                           const Eigen::MatrixXf& vert_coords_K, const Eigen::Matrix4f& T_WK);

        void obtainCovisibles(std::vector<uint64_t>& orderedIdx, const okvis::StateId& latestCovisible, int recursionDepth);

        /**
         * @brief Decide whether a new submap is required.
        */
        bool decideNewSubmap(SupereightFrames& supereightFrame);

        /**
         * @brief Determine and add submap alignment factors (from depth or LiDAR) to the estimator calling alignCallback_.
        */
        void addSubmapAlignmentFactors(const SupereightFrames& supereightFrame);

        /**
         * @brief Determine occupancy-to-point measurements for depth-based alignment 
        */
        void depthMap2MapFactors();

        /**
         * @brief Determine occupancy-to-point measurements for lidar-based alignment
        */
        void lidarMap2MapFactors();

        /**
         * @brief Update SubmapAlignBlock_ from incoming depth measurements.
        */
        void updateDepthAlignBlock(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
          const kinematics::Transformation& T_WK, const int& depthImage_idx);

        /**
         * @brief Update SubmapAlignBlock_ from incoming lidar measurements.
        */
        void updateLidarAlignBlock(const RayVector& vecRayMeasurements,
          const kinematics::Transformation& T_WK, RayVector& vecRayMeasurementsToIntegrate);
        
        SupereightMapType::Config mapConfig_;       ///< Supereight Map config
        SupereightMapType::DataConfigType dataConfig_; ///< Supereight Data config
        se::SubMapConfig submapConfig_; ///< Submapping Config (e.g. alignment settings, kf thresholds,...)
        std::string meshesPath_;  ///< Path to save the meshes
        
        DepthFrameQueue
                depthMeasurements_; ///< Queue with the buffered Depth measurements
        LidarMeasurementQueue
                lidarMeasurements_; ///< Queue with buffered LiDAR Measurements
        StateUpdatesQueue
                stateUpdates_; ///< Queue containing all the state updates from okvis
        SupereightFrameQueue
                supereightFrames_; ///< Queue with the s8 frames (i.e. poses and depth

        std::mutex finishMutex_; // mutex for checking if the integration of data has finished
        std::mutex gtDebug_;
        std::mutex trajMutex_;
        AlignedUnorderedMap<uint64_t, Transformation> gtPoses_;
        bool useGtPoses_ = false;

        // callbacks
        submapCallback submapCallback_; // to visualize in Publisher
        fieldCallback fieldCallback_; // to visualize in Publisher
        trajectoryAnchoringCallback trajectoryAnchoringCallback_;
        anchoredTrajectoryUpdateCallback anchoredTrajectoryUpdateCallback_;
        alignmentPublishCallback alignmentPublishCallback_;

        bool blocking_;

        // We use this to store the active keyframe in predict(), to prepare the supereightframe
        // warning: do not use anywhere else
        struct LatestKeyframe {
            uint64_t Id;
            bool started;

            LatestKeyframe() : Id(1), started(false) {};
        } latestKeyframe;

        
        
        //Keyframe threshold to generate a new map
        unsigned int kfThreshold_;
        std::set<uint64_t> updated_maps_; ////Stores kf indexes of modified/updated submaps

        // Need this to extract okvis estimates at abitrary timestamps
        okvis::Trajectory propagatedStates;
        std::vector<ImuMeasurement> imuMeasurements_; ///< Buffered IMU measurements (realtime pub.).
        std::mutex trajectoryMutex_; ///< Serializes all access to the mutable trajectory.
        realTimePublishingCallback realTimePublishCallback_;
        double odometryPublishingRate_ = 40.0; ///< Publishing rate for realtime propagation.
        okvis::Time lastTime_ = okvis::Time(0); ///< Keep track of last publishing (to maintain rate).

        AlignedMap<StateId, std::set<StateId>> covisibleFrames_;

        uint32_t integration_counter_ = 0;
        uint32_t image_counter_ = 0;

        /// Thread which is in charge of doing the submap integration when data is available
        std::thread submapIntegration_;

        std::thread dataIntegration_;

        /// Attribute with the value of the latest KeyFrame which has been processed. Initial value is -1 to ensure in the first iteration it is updated
        uint64_t prevKeyframeId_ = UNINITIALIZED_ID;

        // The latest finished submap index that is the previous index of prevKeyframeId_
        uint64_t previousSubmapId_ = UNINITIALIZED_ID;

        // The keyframe ID of incoming supereight frames.
        uint64_t curKeyframeId_ = 0;

        // Keyframe counter to decide a new submap.
        unsigned int keyframeCounter_ = 0;

        // Submap counter
        unsigned int submapCounter_ = 0;

        // Own the complete producer operation, not only the final frame reset:
        // an empty state queue can still have a popped update in flight.
        std::mutex dataIntegrationMutex_;
        std::atomic_size_t pendingSupereightFrames_{0};
        std::unique_ptr<SupereightFrames> frame_;

        bool isFinished_ = false;
        bool finalMeshSaved_ = false;  // guarded by finishMutex_, after worker joins
        std::atomic_bool isProcessingSeFrame_ = false;

        std::optional<okvis::AlignedMap<uint64_t,  std::pair<std::shared_ptr<const okvis::kinematics::Transformation>, se::PinholeCamera>>> cameraSensors_; ///< Multi-camera sensor setup
        std::optional<std::pair<Transformation, se::Lidar>> lidarSensors_; ///< Lidar sensor used in supereight ray extensionimuConsume

        const okvis::ViParameters viParameters_; ///< ViParameters with all the extrinsics associated to the sensors used

        const int sensorMeasurementDownsampling_;

        int sensorMeasurementDownsamplingCounter_ = 0;

        const int depthImageResDownsamplingRatio_;

        float depthScaleFactor_ = 1.0f;

        // Lidar related parameters
        std::atomic_bool lidarKfDetected_ = false;
        std::atomic_bool forceSubmap_ = false;
        std::atomic_int ratio_streak_ = 0;
        unsigned int numIntegratedLidarFrames_ = 0;
        unsigned int numIntegratedDepthFrames_ = 0;
        SupereightMapType* previousSubmap_ = nullptr; ///< Store reference / pointer to previous submap to be able to add submap alignment constraints

        // Callback for submap alignment
        AlignCallback alignCallback_;
        IntegrationPublishCallback integrationPublishCallback_;

        kinematics::Transformation T_BS_;
        kinematics::Transformation T_SB_;

        State latestState_;
        TrackingState latestTrackingState_;
        std::mutex state_mtx_;

        size_t last_integrated_state_;

        Eigen::Matrix4f currentPose_ = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f currentPose_WB_ = Eigen::Matrix4f::Identity();
        std::mutex currentPose_mtx_;

        // Book-Keeping which maps are associated for submap alignment
        SubmapAlignmentTerm submapAlignBlock_;
        std::map<uint64_t,uint64_t> map_to_map_association_;

        // Submap Visualizer
        std::atomic_bool visualizeSubmaps_ = false; ///< Flag if submaps should be visualized
        int submapTopViewImageSize_ = 1000; ///< size of the top view (in m).
        threadsafe::Queue<cv::Mat> submapVisQueue_;
    };
} // namespace okvis
#endif //INCLUDE_OKVIS_SUBMAPPINGINTERFACE_HPP
