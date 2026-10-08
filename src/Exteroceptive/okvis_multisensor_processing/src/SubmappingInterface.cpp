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

#include <okvis/SubmappingInterface.hpp>
#include <okvis/DepthUtils.hpp>
#include <filesystem>
#include "se/common/point_cloud_io.hpp"
#include <stdexcept>

#include <opencv2/highgui.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>


void saveAlignedPoints(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points,
                                     okvis::kinematics::Transformation T_WS,
                                     std::string saveName) {
  size_t Npoints = points.size();
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = Npoints;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.resize (cloud.width * cloud.height);
  size_t tmpCnt = 0;
  for (auto& point: cloud) {
    Eigen::Vector3d point_S = points[tmpCnt];
    Eigen::Vector3d point_W = (T_WS.T() * point_S.homogeneous()).head<3>();
    point.x = point_W(0);
    point.y = point_W(1);
    point.z = point_W(2);
    tmpCnt ++;
  }
  
  pcl::io::savePLYFileASCII (saveName, cloud);
}

namespace okvis {

  void get_bbox(const se::Submap<okvis::SupereightMapType>& submap,
                const Eigen::Matrix4f& T_AB,
                Eigen::Vector3f& bbox_world_max,
                Eigen::Vector3f& bbox_world_min)
  {
    // Get Maximum and Minimum of active submap (frame {B})
    Eigen::Vector3f map_min = submap.map->aabb().min();
    Eigen::Vector3f map_max = submap.map->aabb().max();

    // Convert map aabb into {A} frame
    Eigen::Vector3f corner000(map_min.x(), map_min.y(), map_min.z());
    Eigen::Vector3f corner100(map_max.x(), map_min.y(), map_min.z());
    Eigen::Vector3f corner010(map_min.x(), map_max.y(), map_min.z());
    Eigen::Vector3f corner001(map_min.x(), map_min.y(), map_max.z());
    Eigen::Vector3f corner110(map_max.x(), map_max.y(), map_min.z());
    Eigen::Vector3f corner101(map_max.x(), map_min.y(), map_max.z());
    Eigen::Vector3f corner011(map_min.x(), map_max.y(), map_max.z());
    Eigen::Vector3f corner111(map_max.x(), map_max.y(), map_max.z());

    // Transform to Submap Frame of active ID
    corner000 = (T_AB * corner000.homogeneous()).head<3>();
    corner100 = (T_AB * corner100.homogeneous()).head<3>();
    corner010 = (T_AB * corner010.homogeneous()).head<3>();
    corner001 = (T_AB * corner001.homogeneous()).head<3>();
    corner110 = (T_AB * corner110.homogeneous()).head<3>();
    corner101 = (T_AB * corner101.homogeneous()).head<3>();
    corner011 = (T_AB * corner011.homogeneous()).head<3>();
    corner111 = (T_AB * corner111.homogeneous()).head<3>();

    // Compute new AABB
    float x_min = std::min({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});
    float x_max = std::max({corner000.x(),corner100.x(),corner010.x(),corner001.x(),corner110.x(),corner101.x(),corner011.x(),corner111.x()});

    float y_min = std::min({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});
    float y_max = std::max({corner000.y(),corner100.y(),corner010.y(),corner001.y(),corner110.y(),corner101.y(),corner011.y(),corner111.y()});

    float z_min = std::min({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});
    float z_max = std::max({corner000.z(),corner100.z(),corner010.z(),corner001.z(),corner110.z(),corner101.z(),corner011.z(),corner111.z()});

    bbox_world_min.x() = x_min;
    bbox_world_min.y() = y_min;
    bbox_world_min.z() = z_min;
    bbox_world_max.x() = x_max;
    bbox_world_max.y() = y_max;
    bbox_world_max.z() = z_max;

  }

  void get_bbox_world(const se::Submap<okvis::SupereightMapType>& submap,
                      Eigen::Vector3f& bbox_world_max,
                      Eigen::Vector3f& bbox_world_min)
  {
    get_bbox(submap, submap.T_WK.matrix(), bbox_world_max, bbox_world_min);
  }

  void saveDepthImages(DepthFrame& dImage, std::string filename){

    Eigen::Vector2i size(dImage.width(), dImage.height());
    se::save_depth_png(dImage.data(), size, filename);

  }

  bool depthImgResDownsampling(DepthFrame& originalImage, DepthFrame& downsampledImage, int downsamplingFactor){
    if(originalImage.width() % downsamplingFactor == 0 && originalImage.height() % downsamplingFactor == 0){
      se::preprocessor::downsample_depth(originalImage, downsampledImage);
      return true;
    }
    else{
      LOG(WARNING) << "Invalid downsampling factor, providing original image";
      downsampledImage = originalImage.clone();
      return false;
    }
  }

  void SubmappingInterface::drawSubmaps() {
    if(seSubmapLookup_.size() == 0)
      return;

    cv::Mat submapTopView;
    submapTopView.create(submapTopViewImageSize_, submapTopViewImageSize_, CV_8UC3);
    submapTopView.setTo(cv::Scalar(0,0,0));

    Eigen::MatrixXf aabbMins(3, seSubmapLookup_.size());
        Eigen::MatrixXf aabbMaxs(3, seSubmapLookup_.size());
    Eigen::MatrixXf submapPositions(3, seSubmapLookup_.size());
    std::map<uint64_t, int> submapIds;


    int j = 0;
    for (auto& it : seSubmapLookup_){
      // Get AABB
      const Eigen::AlignedBox3f& aabb_box = it.second.map->aabb();
      Eigen::Matrix<float, 3, 2> aabb_bounds_M;
      aabb_bounds_M.col(0) = aabb_box.min();
      aabb_bounds_M.col(1) = aabb_box.max();
      Eigen::Matrix<float, 3, 8> vert_coords_M;
      for (int i = 0; i < 3; i++) {
          for (int j = 0; j < pow(2, i); j++) {
              vert_coords_M.block(i, pow(2, 3 - i) * j, 1, pow(2, 2 - i))
                      << Eigen::MatrixXf::Constant(1, pow(2, 2 - i), aabb_bounds_M(i, 0));
              vert_coords_M.block(i, pow(2, 3 - i) * j + pow(2, 2 - i), 1, pow(2, 2 - i))
                      << Eigen::MatrixXf::Constant(1, pow(2, 2 - i), aabb_bounds_M(i, 1));
          }
      }

      // Transform from submap frame into world frame.
      Eigen::Vector3f min_coords_W, max_coords_W;
      aabbTransform(min_coords_W, max_coords_W, vert_coords_M, it.second.T_WK.matrix());

      aabbMins.col(j) = min_coords_W;
      aabbMaxs.col(j) = max_coords_W;
      submapPositions.col(j) = it.second.T_WK.translation();
      submapIds[it.first] = j;
      j++;
    }

    // Emphasize latest constraint
    auto last_association = map_to_map_association_.rbegin();

    // Determine Visualization scale
    double min_x = aabbMins.row(0).minCoeff() - 0.5;
    double min_y = aabbMins.row(1).minCoeff() - 0.5;
    double max_x = aabbMaxs.row(0).maxCoeff() + 0.5;
    double max_y = aabbMaxs.row(1).maxCoeff() + 0.5;
    double scale = std::min(submapTopViewImageSize_ / (max_x - min_x), submapTopViewImageSize_ / (max_y - min_y));

    // Now do the actual visualization
    std::vector<cv::Point2f> path;
    std::map<uint64_t, cv::Point2f> submap_positions;
    for(auto id : submapIds){
      cv::Point2f c1_m(aabbMins(0,id.second), aabbMins(1,id.second));
      cv::Point2f c1 = (c1_m - cv::Point2f(min_x, min_y)) * scale;
      c1.y = submapTopViewImageSize_ - c1.y;

      cv::Point2f c2_m(aabbMaxs(0,id.second), aabbMaxs(1,id.second));
      cv::Point2f c2 = (c2_m - cv::Point2f(min_x, min_y)) * scale;
      c2.y = submapTopViewImageSize_ - c2.y;

      cv::Scalar bbox_color(0,255,0);
      int line_thickness = 1;
      if(map_to_map_association_.size() > 0){
          if(id.first == last_association->first || id.first == last_association->second){
              line_thickness = 3;
              bbox_color = cv::Scalar(0, 0, 255);
          }
      }
      cv::rectangle(submapTopView, c1, c2, bbox_color, line_thickness);

      // Submap Pose
      cv::Point2f submap_xy_m(submapPositions(0,id.second), submapPositions(1,id.second));
      cv::Point2f submap_xy = (submap_xy_m - cv::Point2f(min_x, min_y)) * scale;
      submap_xy.y = submapTopViewImageSize_ - submap_xy.y;
      cv::circle(submapTopView, submap_xy, 2, cv::Scalar(0, 0, 255));
      submap_positions[id.first] = submap_xy;
      cv::putText(submapTopView, std::to_string(id.first), submap_xy + cv::Point2f(3.0,3.0),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      path.push_back(submap_xy);
    }

    // Draw Path
    for(size_t i = 1; i < path.size(); i++){
        cv::line(submapTopView, path[i], path[i-1] , cv::Scalar(255,0,0), 1, cv::LINE_AA);
    }

    // Draw Associations
    for (const auto& map_to_map_factor : map_to_map_association_){
        cv::line(submapTopView, submap_positions[map_to_map_factor.first], submap_positions[map_to_map_factor.second] , cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }




    // Some text
    cv::putText(submapTopView, "Number of submaps: " + std::to_string(seSubmapLookup_.size()),
                cv::Point(15.0,15.0), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    submapVisQueue_.PushNonBlockingDroppingIfFull(submapTopView,1);
  }

  bool SubmappingInterface::publishSubmapTopView(cv::Mat& submapPlot)
  {
    if(submapVisQueue_.Empty()){
        return false;
    }

    submapVisQueue_.PopNonBlocking(&submapPlot);
    return true;
  }

  void SubmappingInterface::generatePointCloudFromLidar(const std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector3f>,
          Eigen::aligned_allocator<std::pair<Eigen::Matrix4f, Eigen::Vector3f>>>& lidarMeas, int counter){
        
        se::Image<Eigen::Vector3f> pc(lidarMeas.size(), 1);
        Eigen::Isometry3f T_WS = Eigen::Isometry3f::Identity();

        for(size_t i = 0; i < lidarMeas.size(); i++){
          pc[i] = (lidarMeas[i].first * lidarMeas[i].second.homogeneous()).head<3>();
        }

        std::string filename = meshesPath_ + "/../lidarPc/" + std::to_string(counter) + ".vtk";
        
        int test = save_point_cloud_vtk(pc, filename, T_WS);

        if(test == 0) LOG(INFO) << "Correctly saved pointcloud in file " + filename;
        else LOG(ERROR) << "Incorrectly saved pointcloud";

    }

    // Calculate the volume of a 3D bounding box
  float calculateVolume(const Eigen::Vector3f& mins, const Eigen::Vector3f& maxs) {
    return std::max(0.0f, maxs.x() - mins.x()) *
           std::max(0.0f, maxs.y() - mins.y()) *
           std::max(0.0f, maxs.z() - mins.z());
  }


  // Calculate the intersection volume of two 3D bounding boxes
  float calculateIntersectionVolume(const Eigen::Vector3f& mins1, const Eigen::Vector3f& maxs1,
                                    const Eigen::Vector3f& mins2, const Eigen::Vector3f& maxs2) {
    float x_overlap = std::max(0.0f, std::min(maxs1.x(), maxs2.x()) - std::max(mins1.x(), mins2.x()));
    float y_overlap = std::max(0.0f, std::min(maxs1.y(), maxs2.y()) - std::max(mins1.y(), mins2.y()));
    float z_overlap = std::max(0.0f, std::min(maxs1.z(), maxs2.z()) - std::max(mins1.z(), mins2.z()));
    return x_overlap * y_overlap * z_overlap;
  }


  // Calculate the 3D IoU of two bounding boxes
  float calculateIoU3D(const Eigen::Vector3f& mins1, const Eigen::Vector3f& maxs1,
                       const Eigen::Vector3f& mins2, const Eigen::Vector3f& maxs2) {
    float intersection_volume = calculateIntersectionVolume(mins1, maxs1, mins2, maxs2);
    float union_volume = calculateVolume(mins1, maxs1) + calculateVolume(mins2, maxs2) - intersection_volume;

    // Handle the case where union_volume is close to zero to avoid division by zero
    if (union_volume < std::numeric_limits<float>::epsilon()) {
      return 0.0f;
    }

    return intersection_volume / union_volume;
  }

  bool SubmappingInterface::warpDepthImageToColor(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
                                                  const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& rgbData,
                                                  const int depthCamIdx, const int colorCamIdx, 
                                                  cv::Mat& warped_depth_image){

    if(!cameraSensors_) {
      LOG(ERROR) << "No cameras specified.";
      return false;
    }
    if((*cameraSensors_).count(depthCamIdx) == 0){
      LOG(ERROR) << "No depth camera specified for image warping.";
      return false;
    }
    if((*cameraSensors_).count(colorCamIdx) == 0){
      LOG(ERROR) << "No color camera specified for image warping.";
      return false;
    }
    if(!(warped_depth_image.type() == CV_32FC1 || warped_depth_image.type() == CV_64F)) {
      LOG(ERROR) << "Depth Image target type is not CV_32FC1 or CV_64F.";
      return false;
    }
    

    // Get Depth Intrinsics
    Eigen::VectorXf depthIntrinsics;
    (*cameraSensors_).at(depthCamIdx).second.model.getIntrinsics(depthIntrinsics);
    cv::Matx33f intrinsics_depth = cv::Matx33f::eye();
    intrinsics_depth(0,0) = depthIntrinsics(0);
    intrinsics_depth(1,1) = depthIntrinsics(1);
    intrinsics_depth(0,2) = depthIntrinsics(2);
    intrinsics_depth(1,2) = depthIntrinsics(3);

    // Get Color intrinsics
    Eigen::VectorXf rgbIntrinsics;
    (*cameraSensors_).at(colorCamIdx).second.model.getIntrinsics(rgbIntrinsics);
    cv::Matx33f intrinsics_color = cv::Matx33f::eye();
    intrinsics_color(0,0) = rgbIntrinsics(0);
    intrinsics_color(1,1) = rgbIntrinsics(1);
    intrinsics_color(0,2) = rgbIntrinsics(2);
    intrinsics_color(1,2) = rgbIntrinsics(3);
    
    // Warp the depth image to the RGB sensor
    cv::Mat dist_coef_color; // Empty distortion coefficients
    Eigen::Matrix<float,4,4,Eigen::RowMajor> T_CsDs = (rgbData.first.inverse() * depthData.first).matrix();
    cv::Matx44f T_CrgbCdepth(T_CsDs.data());
    
    // cv::rgbd::registerDepth(intrinsics_depth, intrinsics_color, dist_coef_color,
                            // T_CrgbCdepth,
                            // depthData.second.measurement.depthImage,
                            // rgbData.second.measurement.image.size(),
                            // warped_depth_image,
                            // true // Depth dilation
                            // );
    okvis::registerDepth(intrinsics_depth, intrinsics_color, dist_coef_color,
                         T_CrgbCdepth,
                         depthData.second.measurement.depthImage,
                         rgbData.second.measurement.image.size(),
                         warped_depth_image,
                         true // Depth dilation
                        );
    return true;
  }

  bool SubmappingInterface::addLidarMeasurement(const okvis::Time &stamp,
                                                  const Eigen::Vector3d &ray) { // ToDo: adjust logic for lidar (consider large number of measurements)
    // Create an OKVIS LiDAR measurement.
    LidarMeasurement lidarMeasurement;
    lidarMeasurement.timeStamp = stamp;
    lidarMeasurement.measurement.rayMeasurement = ray;

    // Push data to the Queue.
    const size_t lidarQueueSize =100000000; ///< Arbitrary number. ToDo -> fine-tune
    sensorMeasurementDownsamplingCounter_++;
        
    if(sensorMeasurementDownsamplingCounter_ % sensorMeasurementDownsampling_ == 0){
      sensorMeasurementDownsamplingCounter_ = 0;
      if (blocking_) { // ToDo: Remove blocking behavior?
        const bool result =
                lidarMeasurements_.PushBlockingIfFull(lidarMeasurement, lidarQueueSize);
        return result;
      } else {
        // Push measurement and pop the oldest entry.
        const bool result = lidarMeasurements_.PushNonBlockingDroppingIfFull(
                lidarMeasurement, lidarQueueSize);
        if (result)
          DLOG(INFO) << "Oldest LiDAR measurement dropped";
        }
    }

    return true;

  }

    bool SubmappingInterface::addDepthMeasurement(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        const size_t depthQueueSize = 4; //We want to buffer 4 depth images to give OKVIS to have time to compute the states
        sensorMeasurementDownsamplingCounter_++;

        if(sensorMeasurementDownsampling_ == 0){
          throw std::runtime_error("The sensorMeasurementDownsampling_ is set to 0. This seems to be an error in your config file");
        }
        if(sensorMeasurementDownsamplingCounter_ % sensorMeasurementDownsampling_ == 0){
          sensorMeasurementDownsamplingCounter_ = 0;

          // This assumes the image_delay is also applied to the depth camera.
          // That is valid for most of RGB-D cameras and network depths.
          // But, caution is needed if a depth camera has a different timestamp delay than gray (or RGB) camera.
          for (auto& frame : frames) {
            for (auto& depthFrame : frame.second) {
              depthFrame.timeStamp = depthFrame.timeStamp - Duration(viParameters_.camera.image_delay);
            }
          }
          if(blocking_){
            bool result = depthMeasurements_.PushBlockingIfFull(frames, depthQueueSize);
            return result;
          }else{
            bool result = depthMeasurements_.PushNonBlockingDroppingIfFull(frames, depthQueueSize);

            if(result) {

              DLOG(INFO) << "Oldest depth image dropped";
            }
              
            return true;
          }
        }
        
        return true;

    }

    bool SubmappingInterface::realtimePredict(const okvis::Time& stamp,
                                              const Eigen::Vector3d& alpha,
                                              const Eigen::Vector3d& omega) {
      // store in any case
      imuMeasurements_.push_back(ImuMeasurement(stamp, ImuSensorReadings(omega,alpha)));

      // add to Trajectory if possible
      bool success = false;
      State state;
      {
        std::unique_lock<std::mutex> lock(trajectoryMutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          return false;  // Keep buffered IMU samples for the next callback.
        }
        for(auto & imuMeasurement : imuMeasurements_) {
          State propagatedState;
          if(propagatedStates.addImuMeasurement(imuMeasurement.timeStamp,
                                                imuMeasurement.measurement.accelerometers,
                                                imuMeasurement.measurement.gyroscopes,
                                                propagatedState)) {
            state = propagatedState;
            success = true;
          }
        }
        imuMeasurements_.clear();

      }
      if(!success) {
        return false;
      }

      // Callbacks may query the trajectory; publish the copied state without
      // holding its mutex, as in the mapping update callbacks below.
      if(anchoredTrajectoryUpdateCallback_ && (state.timestamp - lastTime_).toSec()
          >= (1.0/double(odometryPublishingRate_))){
        state.id = StateId(0);
        TimerSwitchable anchored_traj_udate("8.7 Anchored trajectory update");
        anchoredTrajectoryUpdateCallback_(state, std::shared_ptr<const okvis::AlignedMap<StateId, State>> (new okvis::AlignedMap<StateId, State>()));
        anchored_traj_udate.stop();
      }

      // ToDo: Publishing for real-time odometry!!
      if(realTimePublishCallback_) {
        // only publish according to rate
        if((state.timestamp - lastTime_).toSec()
            < (1.0/double(odometryPublishingRate_))) {
          return false;
        }
        lastTime_ = state.timestamp;
        const okvis::kinematics::Transformation T_WS = state.T_WS;
        const kinematics::Transformation T_SB = T_BS_.inverse();
        const okvis::kinematics::Transformation T_WB = T_WS * T_SB;

        // fill orientation
        const Eigen::Quaterniond q = T_WB.q();

        // fill position
        const Eigen::Vector3d r = T_WB.r();

        // fill velocity
        const kinematics::Transformation T_BW = T_WB.inverse();
        const Eigen::Vector3d v = T_BW.C() * state.v_W + T_BS_.C() * T_SB.r().cross(state.omega_S);

        // fill angular velocity
        const Eigen::Matrix3d C_BS = T_BS_.C();
        const Eigen::Vector3d omega_B = C_BS * state.omega_S; // of body represented in body

        realTimePublishCallback_(state.timestamp, r, q, v, omega_B);

      }
      return true;
    
    }

    void SubmappingInterface::integrationLoop(){
      
      LOG(INFO) << "In the integration loop";
      while(!isFinished()){
        checkForAvailableData();
      }
    }


    bool SubmappingInterface::enqueueSupereightFrame(size_t maxQueueSize) {
      if (!frame_) {
        return false;
      }
      // Increment before the queue can expose the frame to its consumer.
      ++pendingSupereightFrames_;
      bool dropped = false;
      if (blocking_) {
        if (!supereightFrames_.PushBlockingIfFull(*frame_, 1)) {
          --pendingSupereightFrames_;  // shutdown rejected the new frame
        }
      } else {
        dropped = supereightFrames_.PushNonBlockingDroppingIfFull(*frame_, maxQueueSize);
        if (dropped) {
          --pendingSupereightFrames_;  // the discarded queued frame will not complete
        }
      }
      frame_.reset();
      return dropped;
    }

    bool SubmappingInterface::finishedIntegrating() {
      // Queue emptiness does not mean the producer has finished using frame_.
      // Do not block a caller while that producer is waiting for input/output.
      std::unique_lock<std::mutex> producerLock(dataIntegrationMutex_, std::try_to_lock);
      if (!producerLock.owns_lock() || !stateUpdates_.Empty()
          || pendingSupereightFrames_.load() != 0) {
        return false;
      }

      // Flush only after all producer work is complete. The pending count also
      // covers the consumer's queue-pop/copy interval before map processing.
      // Preserve the existing LiDAR finish policy: it finalizes alignment
      // directly rather than enqueueing an extra residual frame.
      if (!lidarSensors_ && frame_ && (!seSubmapLookup_.empty() || !frame_->vecDepthFrames.empty()
                     || !frame_->vecRayMeasurements.empty())) {
        enqueueSupereightFrame(2);
        return false;
      }
      frame_.reset();  // unused LiDAR remainder or metadata-only short input

      if (!seSubmapLookup_.empty()) {
        if (lidarSensors_) {
          lidarMap2MapFactors();
          if (visualizeSubmaps_) {
            drawSubmaps();
          }
        } else {
          depthMap2MapFactors();
        }
      }

      // Remaining measurements have no state to propagate them with. This
      // lock excludes the only other consumer; PopBlocking also wakes any
      // bounded-queue producer, unlike PopNonBlocking in the existing queue.
      LidarMeasurement lidarMeasurement;
      while (!lidarMeasurements_.Empty()
             && lidarMeasurements_.PopBlocking(&lidarMeasurement)) {}
      std::map<size_t, std::vector<okvis::CameraMeasurement>> depthMeasurement;
      while (!depthMeasurements_.Empty()
             && depthMeasurements_.PopBlocking(&depthMeasurement)) {}
      return true;
    }

    void SubmappingInterface::stopIntegrationThreads() {
      {
        std::lock_guard<std::mutex> lock(finishMutex_);
        isFinished_ = true;
      }
      // Wake blocking producers/consumers before joining. No worker may retain
      // references when setFinished meshes, or the destructor frees members.
      depthMeasurements_.Shutdown();
      lidarMeasurements_.Shutdown();
      stateUpdates_.Shutdown();
      supereightFrames_.Shutdown();
      submapVisQueue_.Shutdown();
      if (dataIntegration_.joinable()) {
        dataIntegration_.join();
      }
      if (submapIntegration_.joinable()) {
        submapIntegration_.join();
      }
    }

    void SubmappingInterface::setFinished() {
      stopIntegrationThreads();
      std::lock_guard<std::mutex> lock(finishMutex_);
      if (finalMeshSaved_) {
        return;
      }
      const auto last = seSubmapLookup_.find(prevKeyframeId_);
      if (last != seSubmapLookup_.end() && last->second.map) {
        seMeshLookup_[prevKeyframeId_] = last->second.map->mesh();
      }
      finalMeshSaved_ = true;
    }

    void SubmappingInterface::frameUpdater(okvis::OkvisUpdate& propagatedStateUpdate) {

      if (propagatedStateUpdate.trackingState.isKeyframe || !latestKeyframe.started) { // first update should always be keyframe, but lets be safe
        latestKeyframe.started = true;

        if(propagatedStateUpdate.latestState.id.value() != latestKeyframe.Id) {
          latestKeyframe.Id = propagatedStateUpdate.latestState.id.value();
        }
      }
      siFrameDataMap frameDataMap;

      for (const auto& updatedState : *propagatedStateUpdate.updatedStates) {
        frameDataMap[updatedState.first.value()] = SiFrameData(updatedState.first.value(), updatedState.second.T_WS, updatedState.second.covisibleFrameIds);
      }

      // Is current state a loop closure state?
      bool loop_closure = propagatedStateUpdate.trackingState.recognisedPlace;

      if(!frame_) {
        frame_.reset(new SupereightFrames(latestKeyframe.Id, frameDataMap, loop_closure));
      }
      else {
        frame_->loop_closure = loop_closure;
        updateSEFrameData(frame_->frameData_, frameDataMap);
      }
    }

    bool SubmappingInterface::predict(const okvis::Time &finalTimestamp,
                                      const Transformation &T_SC,
                                      Transformation &T_WC)
    {

      // Get the okvis update closest to the finalTimestamp (the stamp of the depth frame)
      // We use Okvis::trajectory to propagate state. Also need to use the updates queue bc we also need
      // Updated kfs and all that other stuff that we need to build supereight frames

      okvis::State measurementState;

      if(useGtPoses_ && gtPoses_.find(finalTimestamp.toNSec()) != gtPoses_.end()){
          measurementState.T_WS = gtPoses_[finalTimestamp.toNSec()];
      }else{
        std::lock_guard<std::mutex> lock(trajectoryMutex_);
        if(!propagatedStates.getState(finalTimestamp, measurementState)){
          return false;
        }
      }

      T_WC = measurementState.T_WS * T_SC;
      return true;

    }


    void SubmappingInterface::saveSubmap(uint64_t kfId){
      std::string mesh_numbering;
      if (kfId < 10) {
        mesh_numbering = "00000" + std::to_string(kfId);
      }
      else if (kfId < 100) {
        mesh_numbering = "0000" + std::to_string(kfId);
      }
      else if (kfId < 1000) {
        mesh_numbering = "000" + std::to_string(kfId);
      }
      else if (kfId < 10000) {
        mesh_numbering = "00" + std::to_string(kfId);
      }
      else if (kfId < 100000) {
        mesh_numbering = "0" + std::to_string(kfId);
      }
      else if (kfId < 1000000) {
        mesh_numbering = std::to_string(kfId);
      }

      std::filesystem::path save_dir{meshesPath_};
      if(!std::filesystem::exists(save_dir)){
        std::filesystem::create_directory(save_dir);
      }
      const std::string meshFilename = meshesPath_ + "/mesh_kf" + mesh_numbering + ".ply";
      LOG(INFO) << "Saving to path " + meshFilename << std::endl;

      auto start = std::chrono::high_resolution_clock::now();
      okvis::kinematics::Transformation submapPose(seSubmapLookup_[kfId].T_WK.matrix().cast<double>());

      SupereightMapType::SurfaceMesh mesh;
      if (seMeshLookup_.find(kfId) != seMeshLookup_.end()) {
        mesh = seMeshLookup_[kfId];
      }
      else {
        mesh = seSubmapLookup_[kfId].map->mesh();
      }

      #if OKVIS_COLIDMAP
      // Filter Mesh for Faces with valid color only
      SupereightMapType::SurfaceMesh filtered_mesh;
      filtered_mesh.reserve(mesh.size());
      std::copy_if(mesh.begin(), mesh.end(), std::back_inserter(filtered_mesh),
                   [](const auto& face){
                     return face.colour.vertexes.has_value();
                    });
      if(filtered_mesh.size() > 0) {
        se::io::save_mesh(filtered_mesh, meshFilename, Eigen::Isometry3f(submapPose.T().cast<float>()));
      }
      else {
        se::io::save_mesh(mesh, meshFilename, Eigen::Isometry3f(submapPose.T().cast<float>()));
      }
      #else
      se::io::save_mesh(mesh, meshFilename, Eigen::Isometry3f(submapPose.T().cast<float>()));   
      #endif

      auto end_first = std::chrono::high_resolution_clock::now();
      
      DLOG(INFO) << "It took " << std::to_string(std::chrono::duration_cast<std::chrono::seconds>(end_first - start).count()) << " seconds to process the mesh";

    }

    void SubmappingInterface::setT_BS(const okvis::kinematics::Transformation& T_BS) {
      T_BS_ = T_BS;
      T_SB_ = T_BS_.inverse();
    }

    void SubmappingInterface::obtainCovisibles(std::vector<uint64_t>& orderedIdx, const okvis::StateId& latestCovisible, int recursionDepth){
      std::set<uint64_t> covisibleMaps;
      std::queue<std::pair<int, okvis::StateId>> covisible_queue;
      std::set<StateId> checkedStates;

      if(seSubmapLookup_.find(latestCovisible.value()) != seSubmapLookup_.end()){
          covisibleMaps.insert(latestCovisible.value());
        }

      for(const auto& state: covisibleFrames_[latestCovisible]) {
        covisible_queue.push({0, state});
        if(seSubmapLookup_.find(state.value()) != seSubmapLookup_.end()){
          covisibleMaps.insert(state.value());
        }
      }

      checkedStates.insert(latestCovisible);

      while(!covisible_queue.empty()) {
        const auto& state = covisible_queue.front();
        if(checkedStates.find(state.second)== checkedStates.end() &&
           covisibleFrames_.find(state.second) != covisibleFrames_.end() &&
           state.first <= recursionDepth){
          checkedStates.insert(state.second);
          for(const auto& covisible: covisibleFrames_[state.second]) {
            covisible_queue.push({state.first + 1, covisible});
            if(seSubmapLookup_.find(covisible.value()) != seSubmapLookup_.end()){
              covisibleMaps.insert(covisible.value());
            }
          }
        }
        covisible_queue.pop();
      }

      for(const auto& covMap: covisibleMaps) {
        orderedIdx.push_back(covMap);
      }
    }

    void SubmappingInterface::processSupereightFrames() {

      // Get Supereight Frames --> supposed to be integrated into submaps
      SupereightFrames supereightFrame;
      while(!isFinished()){
        while (supereightFrames_.PopBlocking(&supereightFrame)){
          if (visualizeSubmaps_) {
            drawSubmaps();
          }
          TimerSwitchable timingProcessSeFrame("9 Process supereight frames");
          isProcessingSeFrame_ = true;

          TimerSwitchable timingNewSubmap("9.1 Decide a new submap");
          bool create_new_submap = decideNewSubmap(supereightFrame);
          timingNewSubmap.stop();

          updated_maps_.insert(supereightFrame.keyFrameId);

          for (auto &frameData: supereightFrame.frameData_) {

            // We only update the poses of the keyframes which have a submap associated to it
            const Transformation T_WS = frameData.second.T_WS;
            if(covisibleFrames_.find(StateId(frameData.first)) == covisibleFrames_.end()){
              covisibleFrames_.emplace(StateId(frameData.first), frameData.second.covisibles_);
            } else {
              covisibleFrames_[StateId(frameData.first)] = frameData.second.covisibles_;
            }

            // Check If Id exists.
            if (seSubmapLookup_.count(frameData.first)) {
              updated_maps_.insert(frameData.first);
              seSubmapLookup_[frameData.first].T_WK = Eigen::Isometry3f(T_WS.T().cast<float>());
            } else if(frameData.first == supereightFrame.keyFrameId){
              // Insert
              seSubmapLookup_[supereightFrame.keyFrameId] = {nullptr, Eigen::Isometry3f(T_WS.T().cast<float>())};
            }
          }

          // Create a new submap and reset frame counter
          if (create_new_submap) {

            submapCounter_++;
            if(submapCounter_ > 1){
              previousSubmap_ = seSubmapLookup_[prevKeyframeId_].map.get();
              previousSubmapId_ = prevKeyframeId_;
            }

            addSubmapAlignmentFactors(supereightFrame);

            LOG(INFO) << "New submap generation with counter " << submapCounter_ << " (kf Id: " << supereightFrame.keyFrameId << ")";

            // now we integrate in this keyframe, until we find a new one that is distant enough
            seSubmapLookup_[supereightFrame.keyFrameId].map = std::shared_ptr<SupereightMapType>(new SupereightMapType(mapConfig_, dataConfig_));
            prevKeyframeId_ = supereightFrame.keyFrameId;

            lidarKfDetected_ = false;
            numIntegratedLidarFrames_ = 0;
            numIntegratedDepthFrames_ = 0; 
          }
          auto& activeMap = *(seSubmapLookup_[supereightFrame.keyFrameId].map);

          se::MapIntegrator integrator(activeMap);

          TimerSwitchable batchIntegration("9.2 Lidar batch integration");
          if(lidarSensors_ && supereightFrame.vecRayMeasurements.size() > 0 && supereightFrames_.Size() <= 1){
            // Here prevKeyframeId_ will always be the current active submap
            okvis::kinematics::Transformation T_WK(seSubmapLookup_[prevKeyframeId_].T_WK.matrix().cast<double>());
            RayVector vecRayMeasurementsToIntegrate;

            updateLidarAlignBlock(supereightFrame.vecRayMeasurements, T_WK, vecRayMeasurementsToIntegrate);

            // Now we transform the measurements into the map frame for integration
            if(integrationPublishCallback_) integrationPublishCallback_(supereightFrame.vecRayMeasurements);

            TimerSwitchable actualIntegrationTimer("9.2.1 Actual lidar integration");
            integrator.integrateRayBatch(integration_counter_, vecRayMeasurementsToIntegrate, (*lidarSensors_).second);
            actualIntegrationTimer.stop();
            numIntegratedLidarFrames_++;
            integration_counter_++;

          }

          batchIntegration.stop();

          if(cameraSensors_ && supereightFrames_.Size() <= 1){
            okvis::kinematics::Transformation T_WK(seSubmapLookup_[prevKeyframeId_].T_WK.matrix().cast<double>());

            for(size_t i = 0; i < supereightFrame.vecDepthFrames.size(); i++){
              TimerSwitchable diIntegration("9.3 Depth integration");
              try{
                int depthImage_idx = -1;
                int colourImage_idx = -1;
                for(const auto& idx_cam_measurements: supereightFrame.vecDepthFrames[i]) {
                  if(viParameters_.nCameraSystem.isDepthCamera(idx_cam_measurements.first) &&
                    (*cameraSensors_).find(idx_cam_measurements.first) != (*cameraSensors_).end()){
                      for(const auto& measurement : idx_cam_measurements.second) {
                        if(!measurement.second.measurement.depthImage.empty()) {
                          depthImage_idx = idx_cam_measurements.first;
                        }
                      }
                  }

                  if(viParameters_.nCameraSystem.cameraType(idx_cam_measurements.first).isColour
                            && (*cameraSensors_).find(idx_cam_measurements.first) != (*cameraSensors_).end()) {
                      for(const auto& measurement : idx_cam_measurements.second) {
                        if(!measurement.second.measurement.image.empty()) {
                          colourImage_idx = idx_cam_measurements.first;
                        }
                      }
                  }
                }

                std::pair<Eigen::Isometry3f, okvis::CameraMeasurement> depthData;
                std::pair<Eigen::Isometry3f, okvis::CameraMeasurement> rgbData;

                if(depthImage_idx == -1) {
                  // LOG(WARNING) << "No integration is going to happen, either there is no depth camera with the mapping flag set to true or there are no depth images";
                  continue;
                } else if(depthImage_idx != -1 && colourImage_idx == -1) {
                  DLOG(INFO) << "Only depth";

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(depthImage_idx)) {
                    if(!measurements.second.measurement.depthImage.empty()) {
                      depthData = measurements;
                      break;
                    }
                  }

                  se::Measurements seMeasurements = {se::Measurement{
                    depthMat2Image(depthData.second.measurement.depthImage),
                    (*cameraSensors_).at(depthImage_idx).second,
                    Eigen::Isometry3f(T_WK.inverse().T().cast<float>() * depthData.first.matrix())}};
                  se::Image<float> seSigmaImage(depthData.second.measurement.depthImage.cols,
                                                depthData.second.measurement.depthImage.rows);
                  if (submapConfig_.useUncertainty && !depthData.second.measurement.sigmaImage.empty()) {
                    seSigmaImage = depthMat2Image(depthData.second.measurement.sigmaImage);
                    seMeasurements.depth_sigma = &seSigmaImage;
                  }
                  integrator.integrateDepth(integration_counter_, seMeasurements);
                  integration_counter_++;
                  numIntegratedDepthFrames_++;
                } else if(depthImage_idx != -1 && colourImage_idx != -1) {

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(depthImage_idx)) {
                    if(!measurements.second.measurement.depthImage.empty()) {
                      depthData = measurements;
                      break;
                    }
                  }

                  for(const auto& measurements : supereightFrame.vecDepthFrames[i].at(colourImage_idx)) {
                    if(!measurements.second.measurement.image.empty()) {
                      rgbData = measurements;
                      break;
                    }
                  }

                  TimerSwitchable diWarp("8.5.2 -- warping");
                  // Register depth
                  cv::Mat warped_depth(depthData.second.measurement.depthImage.size(),
                                       CV_64F);

                  if(!warpDepthImageToColor(depthData, rgbData, depthImage_idx, colourImage_idx, warped_depth)) {
                    LOG(WARNING) << "Warping Depth failed. Use original depth.";
                    warped_depth = depthData.second.measurement.depthImage;
                  }
                  diWarp.stop();

                  integrator.integrateDepth(
                      integration_counter_,
                      se::Measurements{se::Measurement{
                                          depthMat2Image(depthData.second.measurement.depthImage),
                                          (*cameraSensors_).at(depthImage_idx).second,
                                          Eigen::Isometry3f(T_WK.inverse().T().cast<float>() * depthData.first.matrix())},
                                        se::Measurement<se::PinholeCamera, se::colour_t>{
                                          rgbMat2Image(rgbData.second.measurement.image),
                                          (*cameraSensors_).at(colourImage_idx).second,
                                          Eigen::Isometry3f(T_WK.inverse().T().cast<float>() * rgbData.first.matrix())}
                                        });
                  integration_counter_++;
                  numIntegratedDepthFrames_++;
                }
                last_integrated_state_ = supereightFrame.frameData_.rbegin()->first;
                diIntegration.stop();

                if (submapConfig_.useMap2MapFactors) {
                  updateDepthAlignBlock(depthData, T_WK, depthImage_idx);
                }
              } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
              }   
            } 
          }

          //Do it with depth images

          // Get the latest state estimate if one exists.
          okvis::State current_state; // Use only T_WS
          Eigen::Matrix4f currentPose_WS;
          {
            std::lock_guard lockCurrentPose(currentPose_mtx_);
            currentPose_WS = currentPose_;
          }
          current_state.T_WS = okvis::kinematics::Transformation(currentPose_WS.cast<double>());

          State lastKnownState;
          TrackingState lastTrackingState;
          {
            std::lock_guard _(state_mtx_);
            lastKnownState = latestState_;
            lastTrackingState = latestTrackingState_;
          }

          if (fieldCallback_) {
            fieldCallback_(current_state, seSubmapLookup_);
          }
        
          if(create_new_submap && previousSubmapId_ != UNINITIALIZED_ID){

            LOG(INFO) << "Completed integrating submap " << previousSubmapId_ << " which is submap number " << seSubmapLookup_.size();
            LOG(INFO) << okvis::timing::Timing::print();

            DLOG(INFO) << "Trying to mesh for frame " << previousSubmapId_;
            if(previousSubmapId_ != UNINITIALIZED_ID){
              // ToDo: mesh in asynchronous thread, not to block processing
              seMeshLookup_[previousSubmapId_] = seSubmapLookup_[previousSubmapId_].map->mesh();
            }


            // save mesh and publish submap
            if (submapCallback_) {
              AlignedUnorderedMap<uint64_t, Transformation> submapPoses;
              std::unordered_map<uint64_t, std::shared_ptr<SupereightMapType>> submaps;

              for(auto it: updated_maps_){
                if(it == prevKeyframeId_) continue;
                okvis::kinematics::Transformation submapPose(seSubmapLookup_[it].T_WK.matrix().cast<double>());
                submapPoses[it] = submapPose;
                submaps[it] = seSubmapLookup_[it].map;
              }
              updated_maps_.clear();
              
              TimerSwitchable submap_viz("9.4 Submap visualization");

              publishSubmaps(submapPoses, submaps);
              submap_viz.stop();
            }
          }
          isProcessingSeFrame_ = false;
          timingProcessSeFrame.stop();
          --pendingSupereightFrames_;  // release only after every map/callback access
        }
      }
      return;
    }

    void SubmappingInterface::generatePointCloud(const Eigen::Matrix4f& T_WS, const DepthFrame& dImage, const se::PinholeCamera& sensor, int counter){

        se::Image<Eigen::Vector3f> pc(sensor.model.imageWidth(), sensor.model.imageHeight());
        std::string filename = meshesPath_ + "/../pc" + std::to_string(counter) + ".vtk";
        std::string diFilename = meshesPath_ + "/../depth_images/dI" + std::to_string(counter) + ".png";
        Eigen::Vector2i size(dImage.width(), dImage.height());
        se::save_depth_png(dImage.data(), size, diFilename);
        se::preprocessor::depth_to_point_cloud(pc, dImage, sensor);
        int test = save_point_cloud_vtk(pc, filename, Eigen::Isometry3f(T_WS));

        if(test == 0) LOG(INFO) << "Correctly saved pointcloud in file " + filename;
        else LOG(INFO) << "Incorrectly saved pointcloud";

    }

    void SubmappingInterface::saveAllSubmapMeshes(){
      LOG(INFO) << "There are " << seSubmapLookup_.size() << " submaps to save";
      for(auto it = seSubmapLookup_.begin(); it != seSubmapLookup_.end(); ++it){
        saveSubmap(it->first);        
      }
    }

    DepthFrame SubmappingInterface::depthMat2Image(const cv::Mat &inputDepth) {

      // Initialise and copy
      if(inputDepth.type() != CV_32FC1) {
        throw std::runtime_error("Only implemented for CV_32FC1 cv::Mat");
      }
      DepthFrame output(inputDepth.cols, inputDepth.rows);

      // cv::MAT and DepthFrame keep data stored in row major format.
      if(!inputDepth.isContinuous()) {
        //TODO write down row by row, first iterate rows the columns and add them to a vector
        throw std::runtime_error("Only implemented for continuous cv::Mat");
      }

      memcpy(output.data(), inputDepth.data,
            inputDepth.cols * inputDepth.rows * sizeof(float));
      
      return output;

    }

    se::Image<se::colour_t> SubmappingInterface::rgbMat2Image(const cv::Mat &inputRGB) {
      // Initialise and copy
      if(inputRGB.type() != CV_8UC3) {
        throw std::runtime_error("Only implemented for CV_8UC3 cv::Mat");
      }
      se::Image<se::colour_t> output(inputRGB.cols, inputRGB.rows);

      // cv::MAT and DepthFrame keep data stored in row major format.
      if(!inputRGB.isContinuous()) {
        //TODO write down row by row, first iterate rows the columns and add them to a vector
        throw std::runtime_error("Only implemented for continuous cv::Mat");
      }

      memcpy(output.data(), inputRGB.data,
            inputRGB.cols * inputRGB.rows * inputRGB.channels() * sizeof(uchar));
      
      return output;

    }

    bool SubmappingInterface::checkForAvailableData() {
      std::lock_guard<std::mutex> producerLock(dataIntegrationMutex_);
      std::map<size_t, std::vector<okvis::CameraMeasurement>> depthMeasurement;
      LidarMeasurement oldestLidarMeasurement;
      bool integrateLidar = true;
      bool integrateDepthImage = true;

      // Check if any measurements in the queue to be integrated
      if (!lidarMeasurements_.getCopyOfFront(&oldestLidarMeasurement) && !depthMeasurements_.getCopyOfFront(&depthMeasurement)){
        integrateLidar = false;
        integrateDepthImage = false;
      }

      // Check if State Updates available (if so, get neweset State).
      OkvisUpdate newestState;
      if(!stateUpdates_.getCopyOfBack(&newestState)){
        DLOG(INFO) << "not having any state updates";
        return false;
      }

      // Check if oldest Measurement to be integrated is older than the newest State.
      if(!(oldestLidarMeasurement.timeStamp <= newestState.timestamp) || !lidarSensors_){
        integrateLidar = false;
      }
      
      okvis::Time newestDataTimestamp;
      if(!cameraSensors_){
        newestDataTimestamp = oldestLidarMeasurement.timeStamp;
        integrateDepthImage = false;
      } else if(!depthMeasurement.empty()){
        for(const auto& idx_cam_data : depthMeasurement) {
          for(const auto& measurement : idx_cam_data.second) {
            if(measurement.timeStamp > newestDataTimestamp) {
              newestDataTimestamp = measurement.timeStamp;
            }
          }
        }
        
        if(newestDataTimestamp > newestState.timestamp) {
            integrateDepthImage = false;
        }
      }

      if(!(integrateDepthImage || integrateLidar)){
        OkvisUpdate popState;
        bool continuePopping = true;
        while(continuePopping){
          

          if(blocking_) {
            continuePopping = stateUpdates_.PopBlocking(&popState);
          } else {
            continuePopping = stateUpdates_.PopNonBlocking(&popState);
          }
          if (!continuePopping) {
            break;  // shutdown/empty queue did not supply an update
          }

          // update Trajectory object
          {
            std::lock_guard<std::mutex> lock(trajectoryMutex_);
            propagatedStates.update(popState.trackingState, popState.updatedStates, popState.affectedStates);
          }

          if(anchoredTrajectoryUpdateCallback_) {
            TimerSwitchable anchored_traj_udate("8.7 Anchored trajectory update");
            anchoredTrajectoryUpdateCallback_(popState.latestState, popState.updatedStates);
            anchored_traj_udate.stop();
          }

          frameUpdater(popState);

          if(continuePopping) {
            continuePopping = popState.timestamp < newestState.timestamp;
          }

        }

        return false;
      } 

      // Get oldest state update to iterate
      OkvisUpdate currentStateUpdate;
      bool continue_popping = true;

      while(continue_popping) {
        const bool popped = blocking_
            ? stateUpdates_.PopBlocking(&currentStateUpdate)
            : stateUpdates_.PopNonBlocking(&currentStateUpdate);
        if (!popped) {
          return false;  // shutdown/empty queue
        }
        // Update internal representation of the trajectory
        // update Trajectory object
        {
          std::lock_guard<std::mutex> lock(trajectoryMutex_);
          propagatedStates.update(currentStateUpdate.trackingState, currentStateUpdate.updatedStates, currentStateUpdate.affectedStates);
        }
        
  
        if(anchoredTrajectoryUpdateCallback_) {
          TimerSwitchable anchored_traj_udate("8.7 Anchored trajectory update");
          anchoredTrajectoryUpdateCallback_(currentStateUpdate.latestState, currentStateUpdate.updatedStates);
          anchored_traj_udate.stop();
        }

        if(!frame_) {
          frameUpdater(currentStateUpdate);
        }

        if(continue_popping) {
            continue_popping = currentStateUpdate.timestamp < newestDataTimestamp;
        }
      }

      // Setting some variables required for looping over state updates and measurements
      bool keepLooping = true;
      const size_t supereightQueueSize = 5000; /// Large Queue needed for LiDAR measurements if we do not want to drop anything.
      Transformation T_WD;

      // Iterate available measurements
      LidarMeasurement lidarMeasurement;
      lidarMeasurements_.getCopyOfFront(&lidarMeasurement);
      depthMeasurements_.getCopyOfFront(&depthMeasurement);

      // Motion de-skewed LiDAR Measurements
      std::vector<okvis::State> states;
      std::vector<okvis::Time> times;
      std::vector<okvis::LidarMeasurement, Eigen::aligned_allocator<okvis::LidarMeasurement>> measurements;

      while(keepLooping){

        if(integrateLidar && lidarMeasurement.timeStamp <= currentStateUpdate.timestamp){

          const bool popped = blocking_
              ? lidarMeasurements_.PopBlocking(&lidarMeasurement)
              : lidarMeasurements_.PopNonBlocking(&lidarMeasurement);
          if (!popped) {
            return false;  // shutdown/empty queue
          }

          measurements.push_back(lidarMeasurement);
          times.push_back(lidarMeasurement.timeStamp);
        }
        bool depthImageExisting = integrateDepthImage;
        size_t mappingCameraId = 0;
        if(cameraSensors_) {
          mappingCameraId = (*cameraSensors_).begin()->first;
        }
        if(depthImageExisting && !depthMeasurement.empty() && depthMeasurement.at(mappingCameraId).size() > 0){
          depthImageExisting = depthImageExisting && (depthMeasurement.at(mappingCameraId).at(0).timeStamp <= currentStateUpdate.timestamp);
        }
        if(depthImageExisting){

          const bool popped = blocking_
              ? depthMeasurements_.PopBlocking(&depthMeasurement)
              : depthMeasurements_.PopNonBlocking(&depthMeasurement);
          if (!popped) {
            return false;  // shutdown/empty queue
          }
          
          okvis::AlignedMap<size_t, std::vector<std::pair<Eigen::Isometry3f, CameraMeasurement>>> posedCameraMeasurements;
          bool all_assigned = true;
          for(auto& cam_idx_measurements : depthMeasurement) {
            std::vector<std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>> posed_cam_idx_measurements;
            okvis::kinematics::Transformation T_WD;
            for(auto& measurement : cam_idx_measurements.second){
              kinematics::Transformation T_SCi = *viParameters_.nCameraSystem.T_SC(cam_idx_measurements.first);
              if (viParameters_.nCameraSystem.cameraType(cam_idx_measurements.first).depthType.needRectify) {
                T_SCi = *viParameters_.nCameraSystem.rectifyT_SC(cam_idx_measurements.first);
              }
              if (currentStateUpdate.latestState.isOnlineExtrinsics) {
                T_SCi = currentStateUpdate.latestState.extrinsics[cam_idx_measurements.first];
              }

              if(!predict(measurement.timeStamp, T_SCi, T_WD)){
                LOG(ERROR) << "Depth image dropped with timestamp " << measurement.timeStamp << " and index is " << cam_idx_measurements.first;
                all_assigned = false;
                break;
              }

              if(viParameters_.nCameraSystem.isDepthCamera(cam_idx_measurements.first) && !measurement.measurement.depthImage.empty()){
                //Apply all configs: downsampling, rescaling, etc...
                measurement.measurement.depthImage = depthScaleFactor_ * measurement.measurement.depthImage;
                if(depthImageResDownsamplingRatio_ > 1 ) {
                  // TODO: change this to using SE2 function for downsampling (se::preprocessor::downsample_depth)
                  cv::Mat depthImageSubsampled;
                  cv::resize(measurement.measurement.depthImage, depthImageSubsampled, cv::Size(), 1.0 / depthImageResDownsamplingRatio_, 1.0 / depthImageResDownsamplingRatio_, cv::INTER_NEAREST);
                  measurement.measurement.depthImage = depthImageSubsampled;
                  if(!measurement.measurement.sigmaImage.empty()){
                    cv::Mat sigmaImageSubsampled;
                    cv::resize(measurement.measurement.sigmaImage, sigmaImageSubsampled, cv::Size(), 1.0 / depthImageResDownsamplingRatio_, 1.0 / depthImageResDownsamplingRatio_, cv::INTER_NEAREST);
                    measurement.measurement.sigmaImage = sigmaImageSubsampled;
                  }
                }
              }

              posed_cam_idx_measurements.emplace_back(std::make_pair(Eigen::Isometry3f(T_WD.T().cast<float>()), measurement));
            }

            //Change camera transformation to Matrix4f
            posedCameraMeasurements.emplace(std::make_pair(cam_idx_measurements.first, posed_cam_idx_measurements));
          }

          if(!all_assigned) {
            break;
          }

          if(frame_){
              frame_->vecDepthFrames.push_back(posedCameraMeasurements);
          } else {
            throw std::runtime_error("When adding image data we have found a point where there is no frame initialized where there should have be");
          }
        }

        bool keepLoopingDepth = true;
        if(!integrateDepthImage || !depthMeasurements_.getCopyOfFront(&depthMeasurement)){
            keepLoopingDepth = false;
        }
        else if(depthMeasurement.at(mappingCameraId).at(0).timeStamp >= currentStateUpdate.timestamp) {
            keepLoopingDepth = false;
        }
        bool keepLoopingLidar = true;
        if((!lidarMeasurements_.getCopyOfFront(&lidarMeasurement) || lidarMeasurement.timeStamp >= currentStateUpdate.timestamp || !integrateLidar)){
            keepLoopingLidar = false;
        }

        if(!keepLoopingDepth && !keepLoopingLidar ) {
          keepLooping = false;
          if(integrateLidar){
            states.reserve(times.size());
            bool getStatesSuccessfully;
            {
              std::lock_guard<std::mutex> lock(trajectoryMutex_);
              getStatesSuccessfully = propagatedStates.getStates(times, states);
            }
            if(getStatesSuccessfully){
              for(size_t i = 0; i < times.size(); i++){
                Eigen::Matrix4f T_WL = states[i].T_WS.T().cast<float>() * (*viParameters_.lidar).T_SL.T().cast<float>();
                if(!frame_){
                  throw std::runtime_error("About to add lidar measurements and we have found a point where there is no frame initialized where there should have be");
                }else{
                  frame_->vecRayMeasurements.push_back({Eigen::Isometry3f(T_WL), 
                                                        measurements[i].measurement.rayMeasurement.template cast<float>()});
                }
              }
            }
          }

          // Push SupereightFrame for every state
          frameUpdater(currentStateUpdate);

          if (enqueueSupereightFrame(supereightQueueSize)) {
            LOG(WARNING) << "Oldest Supereight frame dropped";
          }
        }
      }
      return true;
    }

    bool SubmappingInterface::start() {

      LOG(INFO) << "\n\nStarting supereight processing... \n\n";

      dataIntegration_ = std::thread(&SubmappingInterface::integrationLoop, this);
      submapIntegration_ = std::thread(&SubmappingInterface::processSupereightFrames, this);

      return true;
    }
    
    bool SubmappingInterface::stateUpdateCallback(
            const State &latestState, const TrackingState &latestTrackingState,
            std::shared_ptr<const okvis::AlignedMap<StateId, State>> updatedStates) {

      {
        std::lock_guard lockCurrentPose(currentPose_mtx_);
        currentPose_ = latestState.T_WS.T().cast<float>();
        currentPose_WB_ = currentPose_ * T_SB_.T().cast<float>();
      }

      if(latestTrackingState.isLidarKeyframe && numIntegratedLidarFrames_ > submapConfig_.submapMinFrames){
        DLOG(INFO) << " LIDAR KF!!! FORCE NEW SUBMAP!" << std::endl;
        lidarKfDetected_ = true;
        forceSubmap_ = true;
        ratio_streak_ = 0;
      }

      std::set<StateId> affectedStateIds;
      auto it = updatedStates->begin();
      while(it != updatedStates->end()){
        affectedStateIds.insert(it->first);
        ++it;
      }
      
      {
        std::lock_guard _(state_mtx_);
        latestState_ = latestState;
        latestTrackingState_ = latestTrackingState;
      }

      OkvisUpdate latestStateData(latestState,
                                  latestState.timestamp,
                                  latestTrackingState,
                                  affectedStateIds,
                                  updatedStates);
      const size_t stateUpdateQueue = 100000;

      if (blocking_) {
        bool success = stateUpdates_.PushBlockingIfFull(latestStateData, 1);
        return success;
      } else {
        if (stateUpdates_.PushNonBlockingDroppingIfFull(latestStateData,
                                                        stateUpdateQueue)) {
          LOG(WARNING) << "Oldest state  measurement dropped";
          return true;
        }
      }

      return false;
    }

    void SubmappingInterface::publishSubmaps(AlignedUnorderedMap<uint64_t, Transformation> submapPoses,
                                             std::unordered_map<uint64_t, std::shared_ptr<SupereightMapType>> submaps){

      DLOG(INFO) << "publishing submaps";

      if (submapCallback_) {
        submapCallback_(submapPoses, submaps);
      }
    }
    

    void SubmappingInterface::aabbTransform(Eigen::Vector3f& min_coords_W, Eigen::Vector3f& max_coords_W, 
                                            const Eigen::MatrixXf& vert_coords_K, const Eigen::Matrix4f& T_WK) {
        // Transform coordinates of vertices from submap frame to world frame
        Eigen::MatrixXf vert_coords_W = T_WK * vert_coords_K.colwise().homogeneous();
        // Get the maximal and minimal coordinates of the AABB in world frame
        min_coords_W = vert_coords_W.rowwise().minCoeff().head<3>();
        max_coords_W = vert_coords_W.rowwise().maxCoeff().head<3>();
    }


  float SubmappingInterface::evaluateDepthOverlap(const SupereightMapType& map, uint64_t& mapId, SupereightFrames& seFrame) {

    if (seFrame.vecDepthFrames.empty()) {
      LOG(WARNING) << "Depth measurements are empty in supereight frame.";
      return 1.0;
    }

    if (prevKeyframeId_ == UNINITIALIZED_ID) {
      return 1.0;
    }
    
    Eigen::Isometry3f T_KW = seSubmapLookup_[mapId].T_WK.inverse();
    unsigned int numObserved = 0;
    unsigned int numAll = 0;
    constexpr int skipInterval = 8;

    for(size_t i = 0; i < seFrame.vecDepthFrames.size(); i++) {
      for(const auto& measurements : seFrame.vecDepthFrames[i]){
        if(viParameters_.nCameraSystem.isDepthCamera(measurements.first)){
          for(const auto& cam_idx_measurements : measurements.second) {
            if(!cam_idx_measurements.second.measurement.depthImage.empty()) {
              Eigen::Matrix4f T_MD = T_KW * cam_idx_measurements.first.matrix();
              int depthWidth = cam_idx_measurements.second.measurement.depthImage.cols;
              int depthHeight = cam_idx_measurements.second.measurement.depthImage.rows;

              unsigned int num2D = (static_cast<unsigned int>(
                std::ceil(static_cast<float>(depthWidth)/skipInterval)))*(static_cast<unsigned int>(
                std::ceil(static_cast<float>(depthHeight)/skipInterval)));
              Eigen::Matrix2Xf samplePoints(2, num2D);
              unsigned int tmpCnt = 0;

              for (int x = 0; x < depthWidth; x+=skipInterval) {
                for (int y = 0; y < depthHeight; y+=skipInterval) {
                  samplePoints(0,tmpCnt) = x;
                  samplePoints(1,tmpCnt) = y;
                  tmpCnt ++;
                }
              }
              
              Eigen::Matrix3Xf sampleDirections(3,num2D);
              std::vector<bool> sampleSuccess;
              auto& camera = (*cameraSensors_).at(measurements.first).second;
              camera.model.backProjectBatch(samplePoints, &sampleDirections, &sampleSuccess);
              const float near = camera.near_plane;
              const float far = camera.far_plane;
              const float voxResolution = 4.0 * mapConfig_.res;
              unsigned int numSample = std::floor((far-near)/voxResolution);

              for (int j = 0; j < sampleDirections.cols(); j++) {
                const int ui = samplePoints(0,j);
                const int vi = samplePoints(1,j);
                const float measDepth_i = cam_idx_measurements.second.measurement.depthImage.at<float>(vi,ui);

                for (unsigned int zi = 0; zi < numSample; zi++) {
                  float depth_i = near + voxResolution*zi;
                  // Only consider voxels within measured depths
                  if (depth_i > measDepth_i) {
                    break;
                  }
                  Eigen::Vector3f cameraPoint;
                  cameraPoint = sampleDirections.col(j) * depth_i;

                  Eigen::Vector3f p_M = T_MD.topLeftCorner<3,3>() * cameraPoint + T_MD.topRightCorner<3,1>();

                  std::optional<float> occ = map.interpField<se::Safe::On>(p_M);
                  numAll++;
                  if (occ){
                    numObserved ++;
                  }
                }
              }
            }
          }
        }
      }
    }

    if(numAll == 0) {
      return 1.0;
    }

    return float(numObserved) / numAll;
  }


  double SubmappingInterface::evaluateLidarOverlap(const SupereightMapType& map, uint64_t& mapId,
                                                 SupereightFrames& seFrame, double overlapThreshold){

    size_t observed_counter = 0;
    size_t number_of_measurements = seFrame.vecRayMeasurements.size();
    size_t number_of_actual_measurements = number_of_measurements;
    if(number_of_measurements==0)
    {
        return 1.0;
    }

    okvis::kinematics::Transformation T_WM(seSubmapLookup_[mapId].T_WK.matrix().cast<double>());

    for(size_t i = 0; i < number_of_measurements; i++) {
        if((seFrame.vecRayMeasurements[i].second.norm() > lidarSensors_->second.far_plane) || (seFrame.vecRayMeasurements[i].second.norm() < lidarSensors_->second.near_plane)){
            number_of_actual_measurements--;
            continue;
        }
        // Transform se frame measurement into submap: T_WM.inverse() * T_WD * ray == world_to_submap*depth_to_submap * measurement_in_depth_sensor_frame
        Eigen::Vector3f pt = T_WM.inverse().T3x4().cast<float>() * seFrame.vecRayMeasurements[i].first * seFrame.vecRayMeasurements[i].second.homogeneous();
        std::optional<float> occ = map.interpField<se::Safe::On>(pt);
        if (occ){
            observed_counter++;
        }
    }

    double ratio = double(observed_counter)/double(number_of_actual_measurements);

    if (ratio <= overlapThreshold && ratio_streak_ > 0){
        ratio_streak_++;
        lidarKfDetected_ = true;
        DLOG(INFO) << "Ratio = " << std::to_string(ratio) << " < " << std::to_string(overlapThreshold)
                     << ", will create new submap! Has been lower now for " << ratio_streak_ << " frames.";
        ratio_streak_ = 0;
    }
    else if (ratio <= overlapThreshold){
      ratio_streak_++;
      DLOG(INFO) << "Ratio = " << std::to_string(ratio) << " < " << std::to_string(overlapThreshold)
                   << ", has been lower now for " << ratio_streak_ << " frames.";
    }
    else{
      ratio_streak_ = 0;
    }


    return ratio;
  }

  uint64_t SubmappingInterface::findMostOverlappingSubmap(
          std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &downsampledPointCloud,
          uint64_t activeId, uint64_t previousId) {

    // Bounding box for active map
    // Get Maximum and Minimum of active submap (frame {B})
    Eigen::Vector3f bbox_active_min = seSubmapLookup_[activeId].map->aabb().min();
    Eigen::Vector3f bbox_active_max = seSubmapLookup_[activeId].map->aabb().max();
    // Go through all submaps to find the most overlapping submap based on IoU
    // Take 5 most overlapping frames and choose the oldest one
    std::map <float, uint64_t> iou_id_map; // map: iou -> frame id
    uint64 max_id = std::numeric_limits<uint64_t>::max();

    // Find the covisible submaps
    std::vector<uint64_t> covisibleIds;
    obtainCovisibles(covisibleIds, StateId(activeId), 15);
    for(auto& it : seSubmapLookup_){
      // Skip active, previous submap and non-covisible submaps (for depth); only active and previous for lidar
      bool skip_submap = lidarSensors_ ? (it.first >= previousId) : (it.first >= previousId || std::find(covisibleIds.begin(), covisibleIds.end(), it.first) == covisibleIds.end());
      if(skip_submap){
        continue;
      }

      // Get Bounding Box of Submap with testId in aciveIds frame
      Eigen::Vector3f bbox_test_max;
      Eigen::Vector3f bbox_test_min;
      get_bbox(seSubmapLookup_[it.first], seSubmapLookup_[activeId].T_WK.inverse().matrix() * seSubmapLookup_[it.first].T_WK.matrix(), bbox_test_max, bbox_test_min);
      float iou = calculateIoU3D(bbox_active_min, bbox_active_max, bbox_test_min, bbox_test_max);

      iou_id_map[iou] = it.first;
    }

    // Now get N most overlapping submaps and take oldest ones
    size_t num_candidates = 5;
    size_t num_tested_maps = (iou_id_map.size() > num_candidates ) ? num_candidates : iou_id_map.size();
    std::set<uint64_t> topNoverlapIds;
    auto reverseIter = iou_id_map.rbegin();
    for (size_t count = 0; count < num_tested_maps; count ++){
      if(reverseIter->first > 0.2)
        topNoverlapIds.insert(reverseIter->second);
      reverseIter++;
    }
    auto setIter = topNoverlapIds.begin();
    while(setIter!=topNoverlapIds.end()){
      // Check for observed points
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> tmp;
      uint64_t testId = *setIter;
      size_t observed_points = determineObservedPoints(testId, activeId, downsampledPointCloud, tmp);
      if(observed_points > 0.5*submapConfig_.numSubmapFactors){
        max_id = *setIter;
        break;
      }
      setIter++;
    }

    return max_id;
  }

  size_t SubmappingInterface::determineObservedPoints(uint64_t& mapIdA, uint64_t& mapIdB,
                                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints){
    size_t observed_counter = 0;
    size_t number_of_measurements = points.size();
    okvis::kinematics::Transformation T_WM_A(seSubmapLookup_[mapIdA].T_WK.matrix().cast<double>());
    okvis::kinematics::Transformation T_WM_B(seSubmapLookup_[mapIdB].T_WK.matrix().cast<double>());

    for(size_t i = 0; i < number_of_measurements; i++) {

      // Transform from submap B to submap A
      Eigen::Vector3f pt_A = T_WM_A.inverse().T3x4().cast<float>() * T_WM_B.T().cast<float>() * points.at(i).homogeneous();
      std::optional<float> occ = seSubmapLookup_[mapIdA].map->interpField<se::Safe::On>(pt_A);
      if (occ){
        observedPoints.push_back(points.at(i));
        observed_counter++;
      }
    }
    return observed_counter;
  }


  size_t SubmappingInterface::determineObservedPointsAndSigma(uint64_t& mapIdA, uint64_t& mapIdB,
                                                              std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points,
                                                              std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& observedPoints,
                                                              std::vector<float>& sigma, std::vector<float>& observedSigma){
    size_t observed_counter = 0;
    size_t number_of_measurements = points.size();
    const float occ_margin = 0.8*dataConfig_.field.log_odd_max;
    okvis::kinematics::Transformation T_WM_A(seSubmapLookup_[mapIdA].T_WK.matrix().cast<double>());
    okvis::kinematics::Transformation T_WM_B(seSubmapLookup_[mapIdB].T_WK.matrix().cast<double>());

    for(size_t i = 0; i < number_of_measurements; i++) {

      // Transform from submap B to submap A
      Eigen::Vector3f pt_A = T_WM_A.inverse().T3x4().cast<float>() * T_WM_B.T().cast<float>() * points.at(i).homogeneous();
      std::optional<float> occ = okvis::interpFieldMeanOccup<se::Safe::On>(*(seSubmapLookup_[mapIdA].map), pt_A);
      std::optional<Eigen::Vector3f> gradf = okvis::gradFieldMeanOccup<se::Safe::On>(*(seSubmapLookup_[mapIdA].map), pt_A);
      if (occ && gradf){
        if (occ.value() > -occ_margin && occ.value() < occ_margin && gradf.value().norm() >= 1e-03) {
          observedPoints.push_back(points.at(i));
          observedSigma.push_back(sigma.at(i));
          observed_counter++;
        }
      }
    }
    return observed_counter;
  }


  bool SubmappingInterface::decideNewSubmap(SupereightFrames& supereightFrame) {
    bool create_new_submap = false;
    if(cameraSensors_){
      if(prevKeyframeId_ == UNINITIALIZED_ID){
        create_new_submap = true;
      }
      else if(supereightFrame.keyFrameId != prevKeyframeId_){
        if (supereightFrame.keyFrameId != curKeyframeId_) {
          keyframeCounter_ ++; // added per each new keyframe
          curKeyframeId_ = supereightFrame.keyFrameId;
        }
        float overlap = 1.0;
        if(submapCounter_ > 1){
          overlap = evaluateDepthOverlap(*(seSubmapLookup_[previousSubmapId_].map), previousSubmapId_, supereightFrame);
        }
        DLOG(INFO) << "overlap = " << overlap << ", " << submapConfig_.submapOverlapRatio
          << ", keyframeCounter_ = " << keyframeCounter_<< ", " << submapConfig_.submapKfThreshold
          << ", numIntegratedDepthFrames_ = " << numIntegratedDepthFrames_ << ", " << submapConfig_.submapMinFrames;
        if ((overlap < submapConfig_.submapOverlapRatio || keyframeCounter_ > submapConfig_.submapKfThreshold)
            && numIntegratedDepthFrames_ > submapConfig_.submapMinFrames) {
          keyframeCounter_ = 0;
          create_new_submap = true;
        }
        else {
          supereightFrame.keyFrameId = prevKeyframeId_;
        }
      }
    }
    else if(lidarSensors_){
      // Only Lidar available, use only lidar submapping strategy
      if(prevKeyframeId_ == UNINITIALIZED_ID){
        DLOG(INFO) << "Creating first map, no overlap or any other checks needed!" << std::endl;
        create_new_submap = true;
      }
      else if(supereightFrame.keyFrameId != prevKeyframeId_){ // new (visual) keyframe registered
        if (supereightFrame.keyFrameId != curKeyframeId_) {
          keyframeCounter_ ++; // added per each new keyframe
          curKeyframeId_ = supereightFrame.keyFrameId;
        }
        if((lidarKfDetected_) || (seSubmapLookup_.size() == 1 && numIntegratedLidarFrames_ > submapConfig_.submapMinFrames)){ // we also want a new Lidar Keyframe based on previous observations
          if(keyframeCounter_ >= submapConfig_.submapKfThreshold || forceSubmap_){
            keyframeCounter_ = 0;
            forceSubmap_ = false;
            create_new_submap = true;
          }
          else{
            supereightFrame.keyFrameId = prevKeyframeId_;
          }
        }
        else{ // we check lidar overlap
          if(numIntegratedLidarFrames_ > submapConfig_.submapMinFrames){
            evaluateLidarOverlap(*seSubmapLookup_[prevKeyframeId_].map, prevKeyframeId_,
                                  supereightFrame, submapConfig_.submapOverlapRatio);
          }
          if(!lidarKfDetected_){
            supereightFrame.keyFrameId = prevKeyframeId_;
          }
          else{
            if(keyframeCounter_ >= submapConfig_.submapKfThreshold){
              create_new_submap = true;
              keyframeCounter_ = 0;
            }
            else{
              supereightFrame.keyFrameId = prevKeyframeId_;
            }
          }
        }
      }
      else{ // no new visual keyframe
        if(lidarKfDetected_){
          // Do nothing
        }
        else{
          // Evaluate Overlap
          if(numIntegratedLidarFrames_ > submapConfig_.submapMinFrames)
            evaluateLidarOverlap(*seSubmapLookup_[prevKeyframeId_].map, prevKeyframeId_,
                                  supereightFrame, submapConfig_.submapOverlapRatio);
        }
      }
    }

    // For debugging purpose.
    DLOG(INFO) << "curKeyframeId_ = " << curKeyframeId_
      << ", supereightFrame.keyFrameId = " << supereightFrame.keyFrameId
      << ", prevKeyframeId_ = " << prevKeyframeId_
      << ", submapCounter_ = " << submapCounter_ 
      << ", previousSubmapId_ = " << previousSubmapId_;

    return create_new_submap;
  }


  void SubmappingInterface::addSubmapAlignmentFactors(const SupereightFrames& supereightFrame) {
    if (submapConfig_.useMap2MapFactors) {
      if (submapCounter_ > 2) {
        if (lidarSensors_) {
          lidarMap2MapFactors();
        }
        else if (cameraSensors_) {
          depthMap2MapFactors();
        }
      }
      else{
        if(alignCallback_ && previousSubmap_) {
          DLOG(INFO) << "Calling callback with previousSubmapId_ = " << previousSubmapId_
                    << "and " << submapAlignBlock_.pointCloud_B.size() << " points";
          std::vector<float> zeros(submapAlignBlock_.pointCloud_B.size());
          std::fill(zeros.begin(), zeros.end(), 0.0f);
          alignCallback_(nullptr, seSubmapLookup_[previousSubmapId_].map.get(), 0, previousSubmapId_,
                         submapAlignBlock_.pointCloud_B, zeros);
        }
      }

      // Create new submap alignment term
      submapAlignBlock_.frame_A_id = prevKeyframeId_;
      submapAlignBlock_.frame_B_id = supereightFrame.keyFrameId;
      submapAlignBlock_.pointCloud_B.clear();
      submapAlignBlock_.sigma_B.clear();
      submapAlignBlock_.voxelHashMap.Clear();
      DLOG(INFO) << "Created new SubmapAlignmentTerm with frame A id: " << prevKeyframeId_ 
                << " and frame B id: " << supereightFrame.keyFrameId;
    }
    else if(submapConfig_.useMap2LiveFactors){
        if(previousSubmap_){
            std::vector<float> zeros;
            alignCallback_(nullptr, seSubmapLookup_[previousSubmapId_].map.get(), 0, previousSubmapId_,
                           submapAlignBlock_.pointCloud_B, zeros);
        }
    }
  }


  void SubmappingInterface::depthMap2MapFactors() {
    if(submapAlignBlock_.pointCloud_B.size() > 0){
      // Determine most overlapping point cloud
      uint64_t most_overlapping_id = findMostOverlappingSubmap(submapAlignBlock_.pointCloud_B, submapAlignBlock_.frame_B_id, submapAlignBlock_.frame_A_id);
      if(most_overlapping_id == std::numeric_limits<uint64_t>::max()) {
        most_overlapping_id = submapAlignBlock_.frame_A_id;
      }

      // Determine observed points
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPoints;
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> downsampledPointCloud;
      std::vector<float> observedSigmas, downsampledSigmas;

      determineObservedPointsAndSigma(most_overlapping_id, submapAlignBlock_.frame_B_id,
                  submapAlignBlock_.pointCloud_B, observedPoints,
                  submapAlignBlock_.sigma_B, observedSigmas);
      if (observedPoints.size() > 5*submapConfig_.numSubmapFactors) {
        okvis::downsamplePointsUncertainty(observedPoints, observedSigmas, downsampledPointCloud, downsampledSigmas, 5*submapConfig_.numSubmapFactors);
      }
      else if (observedPoints.size() > 0) {
        downsampledPointCloud = observedPoints;
        downsampledSigmas = observedSigmas;
      }
      if (downsampledPointCloud.size() > 0) {
        alignCallback_(seSubmapLookup_[most_overlapping_id].map.get(), seSubmapLookup_[submapAlignBlock_.frame_B_id].map.get(),
                        most_overlapping_id,
                        submapAlignBlock_.frame_B_id,
                        downsampledPointCloud, downsampledSigmas);

        if (alignmentPublishCallback_) {
          okvis::kinematics::Transformation T_tmp(seSubmapLookup_[submapAlignBlock_.frame_B_id].T_WK.matrix().cast<double>());
          okvis::Time t_cb;
          t_cb.fromNSec(submapAlignBlock_.frame_B_id);
          alignmentPublishCallback_(t_cb, T_tmp, downsampledPointCloud, true);
        }

        float sigmaMean = 0;
        for (size_t ii = 0; ii < downsampledSigmas.size(); ii++) {
          sigmaMean += downsampledSigmas[ii];
        }

        DLOG(INFO) << "[Depth: Map-to-map] Most overlap, ids: "
          << most_overlapping_id << "-" << submapAlignBlock_.frame_B_id
          << " with " << observedPoints.size()
          << " overlap points --> " << downsampledPointCloud.size()
          << " with uncertainty " << sigmaMean / downsampledSigmas.size();
      }
    }
  }

  
  void SubmappingInterface::lidarMap2MapFactors() {
    okvis::Time t_cb;
    t_cb.fromNSec(submapAlignBlock_.frame_B_id);

    // Determine observable points from which to downsample
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> voxelHashPcl = submapAlignBlock_.voxelHashMap.Pointcloud();
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> voxelHashPclFloat(voxelHashPcl.size());
    for(size_t i = 0; i < voxelHashPcl.size(); i++){
        voxelHashPclFloat[i] = voxelHashPcl[i].cast<float>();
    }
    if(voxelHashPclFloat.size() > 0){
        // Some variables for an overlap with the previous submap.
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPointsPrevious;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> downsampledPointCloudPrevious;

        // Determine most overlapping point cloud
        uint64_t most_overlapping_id = findMostOverlappingSubmap(voxelHashPclFloat, submapAlignBlock_.frame_B_id, submapAlignBlock_.frame_A_id);
        if(most_overlapping_id == std::numeric_limits<uint64_t>::max()){
            most_overlapping_id = submapAlignBlock_.frame_A_id;

            // Determine observed points
            determineObservedPoints(most_overlapping_id, submapAlignBlock_.frame_B_id,
                                    voxelHashPclFloat, observedPointsPrevious);
            okvis::downsamplePointCloud(observedPointsPrevious, downsampledPointCloudPrevious, 10 * submapConfig_.numSubmapFactors);

            if(alignmentPublishCallback_){
                alignmentPublishCallback_(t_cb, Transformation(seSubmapLookup_[submapAlignBlock_.frame_B_id].T_WK.matrix().cast<double>()),
                                          downsampledPointCloudPrevious, true);
            }
            if(alignCallback_){
                std::vector<float> sensorErrors(downsampledPointCloudPrevious.size());
                std::fill(sensorErrors.begin(), sensorErrors.end(), submapConfig_.sensorError);
                alignCallback_(seSubmapLookup_[most_overlapping_id].map.get(),
                               seSubmapLookup_[submapAlignBlock_.frame_B_id].map.get(),
                               most_overlapping_id, submapAlignBlock_.frame_B_id,
                               downsampledPointCloudPrevious, sensorErrors);
            }
        }
        else{
            // Determine observed points
            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> observedPointsOverlap;
            determineObservedPoints(most_overlapping_id, submapAlignBlock_.frame_B_id,
                                    voxelHashPclFloat, observedPointsOverlap);
            determineObservedPoints(submapAlignBlock_.frame_A_id, submapAlignBlock_.frame_B_id,
                                    voxelHashPclFloat, observedPointsPrevious);

            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> downsampledPointCloudOverlap;
            okvis::downsamplePointCloud(observedPointsOverlap, downsampledPointCloudOverlap, 5 * submapConfig_.numSubmapFactors);
            okvis::downsamplePointCloud(observedPointsPrevious, downsampledPointCloudPrevious, 5 * submapConfig_.numSubmapFactors);

            if(alignmentPublishCallback_){
                alignmentPublishCallback_(t_cb, Transformation(seSubmapLookup_[submapAlignBlock_.frame_B_id].T_WK.matrix().cast<double>()), downsampledPointCloudOverlap, true);
            }
            if(alignCallback_){
                std::vector<float> sensorErrorsOverlap(downsampledPointCloudOverlap.size());
                std::fill(sensorErrorsOverlap.begin(), sensorErrorsOverlap.end(), submapConfig_.sensorError);
                std::vector<float> sensorErrorsPrevious(downsampledPointCloudPrevious.size());
                std::fill(sensorErrorsPrevious.begin(), sensorErrorsPrevious.end(), submapConfig_.sensorError);

                alignCallback_(seSubmapLookup_[most_overlapping_id].map.get(), seSubmapLookup_[submapAlignBlock_.frame_B_id].map.get(), most_overlapping_id, submapAlignBlock_.frame_B_id, downsampledPointCloudOverlap, sensorErrorsOverlap);
                alignCallback_(seSubmapLookup_[submapAlignBlock_.frame_A_id].map.get(), seSubmapLookup_[submapAlignBlock_.frame_B_id].map.get(), submapAlignBlock_.frame_A_id, submapAlignBlock_.frame_B_id, downsampledPointCloudPrevious, sensorErrorsPrevious);
            }
            DLOG(INFO) << "[Lidar: Map-to-map] Most overlap, ids: "
                      << most_overlapping_id << "-" << submapAlignBlock_.frame_B_id
                      << " with " << voxelHashPclFloat.size()
                      << " overlap points --> " << downsampledPointCloudOverlap.size();
        }
        map_to_map_association_[submapAlignBlock_.frame_B_id] = most_overlapping_id;
        DLOG(INFO) << "[Lidar: Map-to-map] With the previous submap, ids: "
                  << submapAlignBlock_.frame_A_id << "-" << submapAlignBlock_.frame_B_id
                  << " with " << voxelHashPclFloat.size()
                  << " overlap points --> " << downsampledPointCloudPrevious.size();
    }
  }


  void SubmappingInterface::updateDepthAlignBlock(const std::pair<Eigen::Isometry3f, okvis::CameraMeasurement>& depthData,
    const kinematics::Transformation& T_WK, const int& depthImage_idx) {

    // Pixel to 3D position backprojection
    int depthWidth = depthData.second.measurement.depthImage.cols;
    int depthHeight = depthData.second.measurement.depthImage.rows;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> depthRays(depthWidth * depthHeight);
    std::vector<float> depthSigmas(depthWidth * depthHeight);
    Eigen::Matrix4f T_MD = T_WK.inverse().T().cast<float>() * depthData.first.matrix();

    // Select low-uncertainty depths if available
    size_t counter_xy = 0;
    for (int x = 0; x < depthWidth; x+=20) {
      for (int y = 0; y < depthHeight; y+=20) {

        float pointZ = depthData.second.measurement.depthImage.at<float>(y,x);
        float sigmaZ = submapConfig_.sensorError;
        if (submapConfig_.useUncertainty && !depthData.second.measurement.sigmaImage.empty()) {
          sigmaZ = depthData.second.measurement.sigmaImage.at<float>(y,x);
          if (pointZ < submapConfig_.near_plane || pointZ > submapConfig_.far_plane || sigmaZ > 3.0) continue;
        }

        Eigen::Vector2f imagePoint_xy(static_cast<float>(x), static_cast<float>(y));
        Eigen::Vector3f cameraPoint;
        (*cameraSensors_).at(depthImage_idx).second.model.backProject(imagePoint_xy, &cameraPoint);
        cameraPoint = cameraPoint * pointZ;
        depthRays[counter_xy] = T_MD.block<3,3>(0,0) * cameraPoint + T_MD.block<3,1>(0,3);
        depthSigmas[counter_xy] = sigmaZ;
        counter_xy++;
      }
    }
    depthRays.resize(counter_xy);
    depthSigmas.resize(counter_xy);
    submapAlignBlock_.pointCloud_B.insert(submapAlignBlock_.pointCloud_B.end(), depthRays.begin(), depthRays.end());
    submapAlignBlock_.sigma_B.insert(submapAlignBlock_.sigma_B.end(), depthSigmas.begin(), depthSigmas.end());
  }


  void SubmappingInterface::updateLidarAlignBlock(const RayVector& vecRayMeasurements,
    const kinematics::Transformation& T_WK, RayVector& vecRayMeasurementsToIntegrate) {

    size_t numRays = vecRayMeasurements.size();

    // Measurements in SE Frame are pairs of T_WL (lidar in world frame), r_L (ray in lidar frame)
    // for integration as well as for map-to-map factors, points have to be relative to the submap frame
    vecRayMeasurementsToIntegrate.resize(numRays);

    // Only Rays have to be kept for submap alignment
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> onlyRays;
    if(submapConfig_.useMap2MapFactors){
      onlyRays.reserve(numRays);
    }

    // Now go through measurements and transform from LiDAR frame into submap frame
    for(size_t i = 0; i < numRays; i++){
      // LiDAR-to-submap transformation 
      Eigen::Isometry3f T_KL(Eigen::Isometry3f(T_WK.inverse().T().cast<float>()) * vecRayMeasurements[i].first.cast<float>());
      vecRayMeasurementsToIntegrate[i] = 
              std::pair<Eigen::Isometry3f, Eigen::Vector3f> (T_KL, vecRayMeasurements[i].second.cast<float>());
      if(submapConfig_.useMap2MapFactors){
        onlyRays.push_back(T_KL.cast<double>() * vecRayMeasurements[i].second.cast<double>());
      }
    }

    // If points used for submap alignment add them to the hash map
    size_t n_points = onlyRays.size();
    if(n_points > 0){
      submapAlignBlock_.voxelHashMap.AddPoints(onlyRays);
      }
  }

}
