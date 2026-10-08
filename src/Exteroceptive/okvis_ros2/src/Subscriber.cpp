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
 * @file Subscriber.cpp
 * @brief Source file for the Subscriber class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */
 
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <okvis/ros2/Subscriber.hpp>
#include <okvis/ros2/PointCloudUtilities.hpp>
#include <pcl_conversions/pcl_conversions.h>

#ifdef OKVIS_USE_NN
#include <okvis/Processor.hpp>
#endif

#define OKVIS_THRESHOLD_SYNC 0.01 ///< Sync threshold in seconds.

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {
std::string formatSecondsOrNa(double value) {
  if (!std::isfinite(value)) {
    return "n/a";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << value;
  return oss.str();
}
}  // namespace

Subscriber::~Subscriber()
{
  shutdown();
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
}

Subscriber::Subscriber(std::shared_ptr<rclcpp::Node> node,
                       okvis::ViInterface* viInterfacePtr,
                       okvis::Publisher* publisher, 
                       const okvis::ViParameters& parameters)
{
  viInterface_ = viInterfacePtr;
  publisher_ = publisher;
  parameters_ = parameters;
  setNodeHandle(node);
}

Subscriber::Subscriber(std::shared_ptr<rclcpp::Node> node, 
                       okvis::ViInterface* viInterfacePtr,
                       okvis::Publisher* publisher, 
                       const okvis::ViParameters& parameters,
                       okvis::SubmappingInterface* seInterface,
                       bool isDepthCamera, bool isLiDAR)
{
  viInterface_ = viInterfacePtr;
  seInterface_ = seInterface;
  publisher_ = publisher;
  parameters_ = parameters;
  setNodeHandle(node, isDepthCamera, isLiDAR);
}

void Subscriber::setNodeHandle(std::shared_ptr<rclcpp::Node> node,
                               bool isDepthCamera, bool isLiDAR)
{
  shutdown(); // Also joins a previous worker before replacing its node/callback state.
  node_ = node;
  std::string okvis_sync_stamp_topic = "/okvis/synchronized_stamp";
  node_->declare_parameter("okvis_sync_stamp_topic", okvis_sync_stamp_topic);
  (void)node_->get_parameter("okvis_sync_stamp_topic", okvis_sync_stamp_topic);
  pubOkvisSyncStamp_ =
      node_->create_publisher<builtin_interfaces::msg::Time>(
          okvis_sync_stamp_topic, rclcpp::QoS(1000).reliable());
  LOG(INFO) << "[GaRLILEO][Sync] publishing synchronized stamp topic: "
            << okvis_sync_stamp_topic;

  imageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  depthImageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  imagesReceived_.resize(parameters_.nCameraSystem.numCameras());
  depthImagesReceived_.resize(parameters_.nCameraSystem.numCameras());

  // set up image reception
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
  imgTransport_ = std::make_shared<image_transport::ImageTransport>(node_);

  // set up callbacks
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
    imageSubscribers_[i] = imgTransport_->subscribe(
      "/okvis/cam" + std::to_string(i) +"/image_raw",
      30 * parameters_.nCameraSystem.numCameras(),
      std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i,
        parameters_.nCameraSystem.cameraType(i).isColour));
  }

  subImu_ = node_->create_subscription<sensor_msgs::msg::Imu>(
      "/okvis/imu0", 100000,
      std::bind(&Subscriber::imuCallback, this, std::placeholders::_1));

  if(isDepthCamera){
    syncDepthImages_ = true;

    // set up callbacks
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      depthImageSubscribers_[i] = imgTransport_->subscribe(
        "/okvis/depth" + std::to_string(i) + "/image_raw",
        30 * parameters_.nCameraSystem.numCameras(),
        std::bind(&Subscriber::depthCallback, this, std::placeholders::_1, i));
    }
  }

  if(isLiDAR){
    subLiDAR_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/okvis/lidar", 100000,
      std::bind(&Subscriber::lidarCallback, this, std::placeholders::_1));
  }

  // Radar (place recognition)
  if (parameters_.garlileo.place_recognition_use_radar) {
    const std::array<bool, 2> radar_enabled = {
        parameters_.garlileo.place_recognition_radar0_enable,
        parameters_.garlileo.place_recognition_radar1_enable};
    // Remapped to the radar topics in the launch files, like the camera and IMU topics.
    const std::array<std::string, 2> radar_topics = {"/okvis/radar0", "/okvis/radar1"};

    for (int radar_index = 0; radar_index < 2; ++radar_index) {
      if (!radar_enabled[size_t(radar_index)]) {
        continue;
      }
      const std::string& topic = radar_topics[size_t(radar_index)];
      subRadars_[size_t(radar_index)] =
          node_->create_subscription<sensor_msgs::msg::PointCloud2>(
              topic, rclcpp::QoS(200),
              [this, radar_index](
                  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
                this->radarCallback(msg, radar_index);
              });
      LOG(INFO) << "[RadarPR] Subscribing radar" << radar_index
                << " topic: " << subRadars_[size_t(radar_index)]->get_topic_name();
    }
  }

  garlileoCallbackGroup_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive, false);
  rclcpp::SubscriptionOptions garlileoOptions;
  garlileoOptions.callback_group = garlileoCallbackGroup_;
  garlileo_spline_state_topic_ = "/garlileo/spline_state";
  garlileo_gravity_topic_ = "/garlileo/gravity";
  LOG(INFO) << "[GaRLILEO] Subscribing spline_state topic: " << garlileo_spline_state_topic_;
  subGarlileo_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      garlileo_spline_state_topic_, rclcpp::QoS(200),
      std::bind(&Subscriber::garlileoCallback, this, std::placeholders::_1), garlileoOptions);

  LOG(INFO) << "[GaRLILEO] Subscribing gravity topic: " << garlileo_gravity_topic_;
  subGarlileoGravity_ = node_->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      garlileo_gravity_topic_, rclcpp::QoS(200),
      std::bind(&Subscriber::garlileoGravityCallback, this, std::placeholders::_1), garlileoOptions);

  garlileo_watchdog_ = node_->create_wall_timer(std::chrono::seconds(2), [this]() {
    if(!garlileo_received_) {
      RCLCPP_WARN(node_->get_logger(), "[GaRLILEO] No %s received yet.",
                  garlileo_spline_state_topic_.c_str());
    } else {
      const rclcpp::Time now = node_->get_clock()->now();
      const double dt = (now - last_garlileo_stamp_).seconds();
      if (dt > 3.0) {
        RCLCPP_WARN(node_->get_logger(),
                    "[GaRLILEO] Last %s older than %.2fs (now=%.3f, last=%.3f)",
                    garlileo_spline_state_topic_.c_str(), dt, now.seconds(),
                    last_garlileo_stamp_.seconds());
      }
    }

    if(!garlileo_gravity_received_) {
      RCLCPP_WARN(node_->get_logger(), "[GaRLILEO] No %s received yet.",
                  garlileo_gravity_topic_.c_str());
    } else {
      const rclcpp::Time now = node_->get_clock()->now();
      const double dt = (now - last_garlileo_gravity_stamp_).seconds();
      if (dt > 3.0) {
        RCLCPP_WARN(node_->get_logger(),
                    "[GaRLILEO] Last %s older than %.2fs (now=%.3f, last=%.3f)",
                    garlileo_gravity_topic_.c_str(), dt, now.seconds(),
                    last_garlileo_gravity_stamp_.seconds());
      }
    }
  }, garlileoCallbackGroup_);

  rclcpp::ExecutorOptions executorOptions;
  executorOptions.context = node_->get_node_base_interface()->get_context();
  garlileoExecutor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executorOptions);
  garlileoExecutor_->add_callback_group(garlileoCallbackGroup_, node_->get_node_base_interface());
  startGarlileoCallbacks(); // Last setup step: all callback-owned state is initialized.
}

void Subscriber::startGarlileoCallbacks() {
  stopGarlileoCallbacks_.store(false);
  garlileoCallbackThread_ = std::thread([this]() {
    while (!stopGarlileoCallbacks_.load()
           && rclcpp::ok(node_->get_node_base_interface()->get_context())) {
      // cancel() may happen just before spin_once starts and be reset internally.
      // A bounded wait makes that race harmless: shutdown still joins promptly.
      garlileoExecutor_->spin_once(std::chrono::milliseconds(5));
    }
  });
}

void Subscriber::stopGarlileoCallbacks() {
  stopGarlileoCallbacks_.store(true);
  if (garlileoExecutor_) {
    garlileoExecutor_->cancel();
  }
  if (garlileoCallbackThread_.joinable()) {
    garlileoCallbackThread_.join();
  }
}

void Subscriber::shutdown() {
  // Join before releasing callback targets or changing cached callback state.
  // Owner/main thread only; none of this group's callbacks invokes shutdown().
  stopGarlileoCallbacks();
  garlileo_watchdog_.reset();
  subGarlileo_.reset();
  subGarlileoGravity_.reset();
  garlileoExecutor_.reset();
  garlileoCallbackGroup_.reset();
  for (auto& subscriber : imageSubscribers_) {
    subscriber.shutdown();
  }
  for (auto& subscriber : depthImageSubscribers_) {
    subscriber.shutdown();
  }
  subImu_.reset();
  subLiDAR_.reset();
  for (auto& subscriber : subRadars_) {
    subscriber.reset();
  }
}

void Subscriber::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                               unsigned int cameraIndex, bool isColour)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat raw;
  try
  {
    if(!isColour){
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    } else {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    raw = cv_ptr->image;
  }
  catch (cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat filtered;
  filtered = raw.clone();

  // adapt timestamp
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  // insert
  std::lock_guard<std::mutex> lock(time_mutex_);
  imagesReceived_.at(cameraIndex)[t.toNSec()] = filtered;
  
  // try sync
  synchronizeData();
}

void Subscriber::imuCallback(const sensor_msgs::msg::Imu& msg)
{
  // construct measurement
  okvis::Time timestamp(msg.header.stamp.sec, msg.header.stamp.nanosec);
  Eigen::Vector3d acc(msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z);
  Eigen::Vector3d gyr(msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z);                    
  
  // forward to estimator
  viInterface_->addImuMeasurement(timestamp, acc, gyr);
  
   // also forward for realtime prediction
  if(seInterface_) {
    seInterface_->realtimePredict(timestamp, acc, gyr);
  }
  else if(publisher_) {
    publisher_->realtimePredictAndPublish(timestamp, acc, gyr);
  }
}

void Subscriber::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg, 
                               unsigned int cameraIndex){
  // TODO now we work only with one camera of depth images,
  // as it is what the submapping interface accepts
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat raw;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    raw = cv_ptr->image;
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  if(!viInterface_->addDepthMeasurement(t, raw)){
    //LOG(WARNING) << "Dropped last depth image frame for okvis interface";
  }

  // try sync
  std::lock_guard<std::mutex> lock(time_mutex_);
  depthImagesReceived_.at(cameraIndex)[t.toNSec()] = raw;
  //add here the depth images to the receiver
  synchronizeData();
}

void Subscriber::lidarCallback(const sensor_msgs::msg::PointCloud2& msg){

  // Check which type of LiDAR Point Cloud it is
  bool is_blk = (okvis::pointcloud_ros::has_field(msg, "stamp_high") && okvis::pointcloud_ros::has_field(msg, "stamp_low"));
  bool is_hesai = okvis::pointcloud_ros::has_field(msg, "timestamp");

  if(!(is_blk || is_hesai)){ // Default; use header timestamp
    Eigen::Vector3d ray;
    okvis::Time t(msg.header.stamp.sec, msg.header.stamp.nanosec);

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    pcl::PointCloud<pcl::PointXYZ>::iterator it = temp_cloud->begin();

    for (/*it*/; it != temp_cloud->end(); it++){
      ray << it->x, it->y, it->z;
      if(!seInterface_->addLidarMeasurement(t, ray)) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(t, ray);
    }
  }
  else{
    std::vector<okvis::Time, Eigen::aligned_allocator<okvis::Time>> timestamps;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> rays;

    if(is_hesai){
      okvis::pointcloud_ros::hesai_lidar2points(msg, timestamps, rays);
    }
    else if(is_blk){
      okvis::pointcloud_ros::blk_lidar2points(msg, timestamps, rays);
    }

    for(size_t i = 0; i < timestamps.size(); i++){
      if(!seInterface_->addLidarMeasurement(timestamps[i], rays[i])) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(timestamps[i], rays[i]);
    }
  }
}

void Subscriber::radarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                               int radar_index) {
  if (!viInterface_) {
    return;
  }
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  okvis::RadarTargetReadings targets;
  targets.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height));

  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<float> iter_velocity(*msg, "velocity");
    for (; iter_x != iter_x.end();
         ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_velocity) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      const float intensity = *iter_intensity;
      const float vel = *iter_velocity;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(vel)) {
        continue;
      }
      okvis::RadarTargetReading r;
      r.point_R << static_cast<double>(x), static_cast<double>(y), static_cast<double>(z);
      r.intensity = static_cast<double>(intensity);
      r.radialVelocity = static_cast<double>(vel);
      targets.push_back(r);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "[RadarPR] Failed to parse radar PointCloud2 fields: " << e.what();
    return;
  }

  viInterface_->addRadarTargetsMeasurement(t, targets, radar_index);

  static okvis::Time last_log_time = okvis::Time(0.0);
  if ((t - last_log_time).toSec() > 1.0) {
    LOG(INFO) << "[RadarPR] received radar scan t=" << t.toSec()
              << " radar_index=" << radar_index
              << " num_targets=" << targets.size();
    last_log_time = t;
  }
}

void Subscriber::garlileoCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg){
  if(!viInterface_) { return; }
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);
  Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                       msg->pose.pose.orientation.x,
                       msg->pose.pose.orientation.y,
                       msg->pose.pose.orientation.z);
  Eigen::Vector3d v_body(msg->twist.twist.linear.x,
                         msg->twist.twist.linear.y,
                         msg->twist.twist.linear.z);
  Eigen::Vector3d p_W(msg->pose.pose.position.x,
                      msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
  const double q_norm = q.norm();
  if (!std::isfinite(q_norm) || q_norm < 1.0e-12
      || !v_body.allFinite() || !p_W.allFinite()) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                         "[GaRLILEO] Dropping invalid %s message.",
                         garlileo_spline_state_topic_.c_str());
    return;
  }
  q.normalize();
  viInterface_->addGaRLILEOPseudoMeasurement(t, q, v_body, p_W);

  static thread_local rclcpp::Time last_log(0, 0, RCL_ROS_TIME);
  rclcpp::Time now = msg->header.stamp;
  garlileo_received_ = true;
  last_garlileo_stamp_ = now;
  if ((now - last_log).seconds() > 1.0) {
    LOG(INFO) << "[GaRLILEO] received " << garlileo_spline_state_topic_ << " at t="
              << t.toSec();
    last_log = now;
  }
}

void Subscriber::garlileoGravityCallback(const geometry_msgs::msg::Vector3Stamped::ConstSharedPtr& msg){
  if(!viInterface_) { return; }
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);
  Eigen::Vector3d g_W(msg->vector.x, msg->vector.y, msg->vector.z);
  if (!g_W.allFinite() || g_W.squaredNorm() < 1.0e-12) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                         "[GaRLILEO] Dropping invalid %s message.",
                         garlileo_gravity_topic_.c_str());
    return;
  }
  viInterface_->addGaRLILEOGravityMeasurement(t, g_W);

  static thread_local rclcpp::Time last_log(0, 0, RCL_ROS_TIME);
  rclcpp::Time now = msg->header.stamp;
  garlileo_gravity_received_ = true;
  last_garlileo_gravity_stamp_ = now;
  if ((now - last_log).seconds() > 1.0) {
    LOG(INFO) << "[GaRLILEO] received " << garlileo_gravity_topic_ << " at t="
              << t.toSec() << " g_W=[" << g_W.transpose() << "]";
    last_log = now;
  }
}

void Subscriber::synchronizeData() {
  std::set<uint64_t> allTimes;
  const int numCameras = imagesReceived_.size();
  for(int i=0; i < numCameras; ++i) {
    if(!parameters_.nCameraSystem.cameraType(i).isUsed) continue; // Only consider SLAM cameras here
    for(const auto & entry : imagesReceived_.at(i)) {
      allTimes.insert(entry.first);
    }
  } //We are synchronizing the depth cameras against the ir cameras
  static double last_sync_issue_log_wall = -1.0;

  for(const auto & time : allTimes) {
    // note: ordered old to new
    std::vector<uint64_t> syncedTimes(numCameras, 0);
    uint64_t depthSyncedTime = 0; // Assume that there is only one depth image (currently supported by the submapping interface)
    std::map<uint64_t, cv::Mat> images;
    std::map<uint64_t, cv::Mat> depthImages;

    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedImages;
    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedDepthImages;
    std::vector<double> bestImageAbsDt(numCameras, std::numeric_limits<double>::infinity());
    std::vector<double> bestImageSignedDt(numCameras, std::numeric_limits<double>::quiet_NaN());
    std::vector<uint64_t> bestImageStamp(numCameras, 0);
    std::vector<bool> hasImageCandidate(numCameras, false);
    int unsyncedCamera = -1;

    okvis::Time tcheck;
    tcheck.fromNSec(time);
    bool synced = true;
    for(int i=0; i < numCameras; ++i) {
      bool syncedi = false;
      for(const auto & entry : imagesReceived_.at(i)) {
        okvis::Time ti;
        ti.fromNSec(entry.first);
        const double signed_dt = (tcheck-ti).toSec();
        const double abs_dt = std::abs(signed_dt);
        if(abs_dt < bestImageAbsDt.at(i)) {
          bestImageAbsDt.at(i) = abs_dt;
          bestImageSignedDt.at(i) = signed_dt;
          bestImageStamp.at(i) = entry.first;
          hasImageCandidate.at(i) = true;
        }
        bool slam_use = !parameters_.nCameraSystem.cameraType(i).isUsed;
        if(abs_dt < OKVIS_THRESHOLD_SYNC && (tcheck >= ti || slam_use)) {
          syncedTimes.at(i) = entry.first;
          images[i] = imagesReceived_.at(i).at(entry.first);
          timestampedImages[i] = std::make_pair(ti, imagesReceived_.at(i).at(entry.first));
          syncedi = true;
          break;
        } 
      }

      if(!syncedi) {
        synced = false;
        unsyncedCamera = i;
        break;
      }
    }

    bool syncedDepth = true;
    double bestDepthAbsDt = std::numeric_limits<double>::infinity();
    double bestDepthSignedDt = std::numeric_limits<double>::quiet_NaN();
    int bestDepthCamera = -1;
    uint64_t bestDepthStamp = 0;
    bool hasDepthCandidate = false;
    if(syncDepthImages_){
      syncedDepth = false;
      for(int i = 0; i < numCameras; ++i) {
        for(const auto & entry : depthImagesReceived_.at(i)){
          okvis::Time tdepth;
          tdepth.fromNSec(entry.first);
          const double signed_dt = (tcheck - tdepth).toSec();
          const double abs_dt = std::abs(signed_dt);
          if(abs_dt < bestDepthAbsDt) {
            bestDepthAbsDt = abs_dt;
            bestDepthSignedDt = signed_dt;
            bestDepthCamera = i;
            bestDepthStamp = entry.first;
            hasDepthCandidate = true;
          }
          //There is a higher unsynchronization between depth images and ir images from the realsense
          if(abs_dt < OKVIS_THRESHOLD_SYNC && tcheck>=tdepth) {
            depthSyncedTime = entry.first;
            depthImages[i] = depthImagesReceived_.at(i).at(entry.first);
            timestampedDepthImages[i] = std::make_pair(tdepth, depthImagesReceived_.at(i).at(entry.first));
            syncedDepth = true;
            break;
          }
        }

        if(syncedDepth){
          //For now we assume there is only one depth image per synchronization process, needs discussion
          break;
        }
      }
    }

    if(!(synced && syncedDepth)) {
      const double now_wall = okvis::Time::now().toSec();
      if(now_wall - last_sync_issue_log_wall > 1.0) {
        std::ostringstream camBuffer;
        camBuffer << "{";
        for(int i = 0; i < numCameras; ++i) {
          if(i > 0) {
            camBuffer << ",";
          }
          camBuffer << "cam" << i << ":" << imagesReceived_.at(i).size();
        }
        camBuffer << "}";

        if(!synced && unsyncedCamera >= 0) {
          const bool cam_used = parameters_.nCameraSystem.cameraType(unsyncedCamera).isUsed;
          const bool has_candidate = hasImageCandidate.at(unsyncedCamera);
          const bool gap_too_large =
              has_candidate && bestImageAbsDt.at(unsyncedCamera) >= OKVIS_THRESHOLD_SYNC;
          const char* cause = !has_candidate
                                  ? "missing_camera_frame"
                                  : (gap_too_large
                                         ? "stereo_time_gap_too_large"
                                         : "stereo_timestamp_ordering_or_jitter");
          LOG(WARNING) << "[FrameDrop][Sync] t_ref=" << tcheck
                       << " cause=" << cause
                       << " cam=" << unsyncedCamera
                       << " cam_used=" << cam_used
                       << " nearest_dt_sec=" << formatSecondsOrNa(bestImageAbsDt.at(unsyncedCamera))
                       << " nearest_signed_dt_sec=" << formatSecondsOrNa(bestImageSignedDt.at(unsyncedCamera))
                       << " threshold_sec=" << OKVIS_THRESHOLD_SYNC
                       << " nearest_stamp_nsec=" << bestImageStamp.at(unsyncedCamera)
                       << " cam_buffer_sizes=" << camBuffer.str();
        } else if(syncDepthImages_ && !syncedDepth) {
          std::ostringstream depthBuffer;
          depthBuffer << "{";
          for(int i = 0; i < numCameras; ++i) {
            if(i > 0) {
              depthBuffer << ",";
            }
            depthBuffer << "depth" << i << ":" << depthImagesReceived_.at(i).size();
          }
          depthBuffer << "}";
          const bool gap_too_large = hasDepthCandidate && bestDepthAbsDt >= OKVIS_THRESHOLD_SYNC;
          const char* cause = !hasDepthCandidate
                                  ? "missing_depth_frame"
                                  : (gap_too_large
                                         ? "depth_time_gap_too_large"
                                         : "depth_timestamp_ordering_or_jitter");
          LOG(WARNING) << "[FrameDrop][SyncDepth] t_ref=" << tcheck
                       << " cause=" << cause
                       << " depth_cam=" << bestDepthCamera
                       << " nearest_dt_sec=" << formatSecondsOrNa(bestDepthAbsDt)
                       << " nearest_signed_dt_sec=" << formatSecondsOrNa(bestDepthSignedDt)
                       << " threshold_sec=" << OKVIS_THRESHOLD_SYNC
                       << " nearest_stamp_nsec=" << bestDepthStamp
                       << " depth_buffer_sizes=" << depthBuffer.str();
        }
        last_sync_issue_log_wall = now_wall;
      }
      continue;
    }

    if(synced && syncedDepth) {
      bool isProcessor = false;
      bool frame_added = false;
      okvis::Time okvis_sync_time = tcheck;
      #ifdef OKVIS_USE_NN
      okvis::Processor* casted_processor = dynamic_cast<okvis::Processor*>(viInterface_);
      if (casted_processor) {
        isProcessor = true;
        // Keep the same synchronized timestamp basis as the Processor path:
        // choose the designated sync camera if available, otherwise first synced image.
        size_t sync_cam = 0;
        if (!parameters_.camera.sync_cameras.empty()) {
          sync_cam = *parameters_.camera.sync_cameras.begin();
        }
        auto it_sync = timestampedImages.find(sync_cam);
        if (it_sync != timestampedImages.end()) {
          okvis_sync_time = it_sync->second.first;
        } else if (!timestampedImages.empty()) {
          okvis_sync_time = timestampedImages.begin()->second.first;
        }
      }
      #endif
      // OKVIS stamps its states at the corrected camera time
      // (timestamp_camera_correct = timestamp_camera - image_delay), so GaRLILEO must be
      // driven with the same time; its exact-stamp waits never match otherwise.
      okvis_sync_time = okvis_sync_time - okvis::Duration(parameters_.camera.image_delay);

      // Publish the synchronized stamp *before* addImages().
      // Exact-timestamp wait mode in ThreadedSlam runs inside addImages(),
      // so publishing afterwards can cause a circular wait.
      if (pubOkvisSyncStamp_) {
        builtin_interfaces::msg::Time stamp_msg;
        stamp_msg.sec = static_cast<decltype(stamp_msg.sec)>(okvis_sync_time.sec);
        stamp_msg.nanosec = static_cast<decltype(stamp_msg.nanosec)>(okvis_sync_time.nsec);
        pubOkvisSyncStamp_->publish(stamp_msg);
      }

      #ifdef OKVIS_USE_NN
      if (casted_processor) {
        frame_added = casted_processor->addImages(timestampedImages, timestampedDepthImages);
        if(!frame_added) {
          LOG(WARNING) << "Frame not added to Processor at t=" << okvis_sync_time
                       << " (processor rejected synchronized frame; check [FrameDrop] diagnostics)";
        }
      }
      else {
        frame_added = viInterface_->addImages(tcheck, images, depthImages);
        if(!frame_added) {
          LOG(WARNING) << "Frame not added at t=" << tcheck
                       << " (viInterface rejected synchronized frame; check [FrameDrop][QueueFull] diagnostics)";
        }
      }
      #else
      frame_added = viInterface_->addImages(tcheck, images, depthImages);
      if(!frame_added) {
        LOG(WARNING) << "Frame not added at t=" << tcheck
                     << " (viInterface rejected synchronized frame; check [FrameDrop][QueueFull] diagnostics)";
      }
      #endif

      if(!isProcessor && syncDepthImages_) {
        //If we have depth images, then we send the data to SI from here, if not the Processor handles it. As for LiDAR, we need to decide
        //how to continue with it. Do we want colour fusion at all?
        std::map<size_t, std::vector<okvis::CameraMeasurement>> cameraMeasurements;
        
        for(const auto& it : timestampedImages) {
          okvis::CameraMeasurement cameraMeasure;
          cameraMeasure.timeStamp = tcheck;
          cameraMeasure.measurement.image = it.second.second.clone();
          cameraMeasure.sensorId = it.first;
          cameraMeasurements[it.first] = {cameraMeasure};
        }

        for(const auto& it : timestampedDepthImages) {
          if(cameraMeasurements.find(it.first) != cameraMeasurements.end()) {
            //Check if they are time synchronized or not
            bool imageAdded = false;
            for(auto& image : cameraMeasurements.at(it.first)) {
              if(image.timeStamp == it.second.first){
                image.measurement.depthImage = it.second.second.clone();
                imageAdded = true;
              }
            }

            if(!imageAdded) {
              okvis::CameraMeasurement cameraMeasure;
              cameraMeasure.timeStamp = it.second.first;
              cameraMeasure.measurement.depthImage = it.second.second.clone();
              cameraMeasure.sensorId = it.first;
              cameraMeasurements[it.first] = {cameraMeasure};
            }
          } else {
            okvis::CameraMeasurement cameraMeasure;
            cameraMeasure.timeStamp = it.second.first;
            cameraMeasure.measurement.depthImage = it.second.second.clone();
            cameraMeasure.sensorId = it.first;
            cameraMeasurements[it.first] = {cameraMeasure};
          }
        }
        
        if(!seInterface_->addDepthMeasurement(cameraMeasurements)){
          LOG(WARNING) << "Frame not added to SI at t="<< cameraMeasurements.at(0)[0].timeStamp;
        }
      }

      // remove all the older stuff from buffer
      for(int i=0; i < numCameras; ++i) {
        auto end = imagesReceived_.at(i).find(syncedTimes.at(i));
        if(end!=imagesReceived_.at(i).end()) {
          ++end;
        }
        imagesReceived_.at(i).erase(imagesReceived_.at(i).begin(), end);
      }

      if(syncDepthImages_) {
        for(int i=0; i < numCameras; ++i) {
          auto end = std::find_if(depthImagesReceived_.at(i).begin(), depthImagesReceived_.at(i).end(),
                                 [depthSyncedTime](const auto& x) { return x.first > depthSyncedTime; });
          depthImagesReceived_.at(i).erase(depthImagesReceived_.at(i).begin(), end);
        }
      }
    }
  }
}

} // namespace okvis
