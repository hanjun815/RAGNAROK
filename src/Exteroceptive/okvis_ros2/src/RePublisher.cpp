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
 * @file RePublisher.cpp
 * @brief Source file for the RePublisher class.
 * @author Stefan Leutenegger
 */

#include <okvis/ros2/RePublisher.hpp>
#ifdef OKVIS_USE_NN
#include <okvis/Processor.hpp>
#endif
#include <sensor_msgs/image_encodings.hpp>
#define OKVIS_THRESHOLD_SYNC 0.05 // in seconds

/// \brief okvis Main namespace of this package.
namespace okvis {

RePublisher::RePublisher(
  std::shared_ptr<rclcpp::Node> node, 
  const okvis::ViParameters& viParameters,
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
  std::shared_ptr<ThreadedPublisher> threadedPublisher)
  : node_(node),
    numCams_(viParameters.nCameraSystem.numCameras()),
    threadedImagePublisher_(threadedImagePublisher),
    threadedPublisher_(threadedPublisher) {}

RePublisher::RePublisher(
  std::shared_ptr<rclcpp::Node> node, 
  const okvis::ViParameters& viParameters,
  ViInterface* estimator,
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
  std::shared_ptr<ThreadedPublisher> threadedPublisher)
  : node_(node),
    estimator_(estimator),
    numCams_(viParameters.nCameraSystem.numCameras()),
    threadedImagePublisher_(threadedImagePublisher),
    threadedPublisher_(threadedPublisher) {

  for(size_t i = 0; i < numCams_; i++){
    if(viParameters.nCameraSystem.cameraType(i).isUsed){
      slamCamIdx_.insert(i);
    }
  }

  imagesReceived_.resize(numCams_);
  depthImagesReceived_.resize(1); //If recording depth data, only use one camera
}

void RePublisher::setTopics(const std::string& imuTopic, 
                            const std::string& camTopic,
                            const std::string& rgbTopic,
                            const std::string& depthTopic) {
  imuTopic_ = imuTopic;
  camTopic_ = camTopic;
  rgbTopic_ = rgbTopic;
  depthTopic_ = depthTopic;

  if(depthTopic_.size() != 0){
    syncDepthImages_ = true;
  } else {
    syncDepthImages_ = false;
  }

  // set up the publishers
  pubImu_ = threadedPublisher_->registerPublisher<sensor_msgs::msg::Imu>(imuTopic_);
  for(size_t i=0; i<numCams_; ++i) {
    camPublisherVector_.push_back(
      threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>(
        camTopic_ + std::to_string(i) + "/image_raw"
      ));

    if(!rgbTopic.empty()) {
      rgbPublisherVector_.push_back(
        threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>(
          rgbTopic_ + std::to_string(i) + "/image_raw"
        ));
    }

    if(!depthTopic.empty()) {
      auto preprocessDepthImage =
        [](std::shared_ptr<std::tuple<okvis::Time, cv::Mat>> data) {
          sensor_msgs::msg::Image::SharedPtr msg;
          msg = cv_bridge::CvImage(std_msgs::msg::Header(),
                                   sensor_msgs::image_encodings::TYPE_32FC1,
                                   std::get<1>(*data))
                    .toImageMsg();
          msg->header.stamp =
              rclcpp::Time(std::get<0>(*data).sec, std::get<0>(*data).nsec);
          return msg;
        };

      depthPublisherVector_.push_back(
        threadedImagePublisher_->registerPublisher<
          sensor_msgs::msg::Image, std::tuple<okvis::Time, cv::Mat>>(
          depthTopic_ + std::to_string(i) + "/image_raw",
          preprocessDepthImage));

      // quantized and transformed debug depth image
      auto preprocessDebugDepthImage =
        [](std::shared_ptr<std::tuple<okvis::Time, cv::Mat>> data) {
          sensor_msgs::msg::Image::SharedPtr msg;
          cv::Mat depthImg = std::get<1>(*data).clone();
          expImageTransform(depthImg);
          depthImg.convertTo(depthImg, CV_8UC1, 255.0);
          msg = cv_bridge::CvImage(std_msgs::msg::Header(),
                                   sensor_msgs::image_encodings::MONO8,
                                   depthImg)
                    .toImageMsg();
          return msg;
        };

      debugDepthPublisherVector_.push_back(
        threadedImagePublisher_->registerPublisher<
          sensor_msgs::msg::Image, std::tuple<okvis::Time, cv::Mat>>(
          depthTopic_ + std::to_string(i) + "/quantized_view",
          preprocessDebugDepthImage));
    }
  }
}
    
bool RePublisher::publishImages(const okvis::Time & stamp,
                   const std::map<size_t, cv::Mat> & images,
                   const std::map<size_t, cv::Mat> & depthImages) {
  // re-publish images
  for(size_t i=0; i<numCams_; ++i) {
    sensor_msgs::msg::Image::SharedPtr msg;
    if(images.count(i)) {
      if(images.at(i).channels() == 1) {
        // grayscale
        msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", images.at(i)).toImageMsg();
        msg->header.stamp = rclcpp::Time(stamp.sec, stamp.nsec);
        camPublisherVector_[i].publish(msg);
      } else {
        // colour
        msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", images.at(i)).toImageMsg();
        msg->header.stamp = rclcpp::Time(stamp.sec, stamp.nsec);
        rgbPublisherVector_[i].publish(msg);
      }
    }
    if(depthImages.count(i)) {
      cv::Mat depthImg = depthImages.at(i);

      auto depthPayload = std::make_shared<std::tuple<okvis::Time, cv::Mat>>(stamp, depthImg);
      depthPublisherVector_[i].publish(depthPayload);

      auto debugDepthPayload = std::make_shared<std::tuple<okvis::Time, cv::Mat>>(stamp, depthImg.clone());
      debugDepthPublisherVector_[i].publish(debugDepthPayload);
    }
  }
  return true;
}

bool RePublisher::publishImuMeasurement(const okvis::Time & stamp,
                           const Eigen::Vector3d & alpha,
                           const Eigen::Vector3d & omega) {
  // re-publish IMU
  auto header = std_msgs::msg::Header();
  header.stamp = rclcpp::Time(stamp.sec, stamp.nsec);
  auto msg = std::make_shared<sensor_msgs::msg::Imu>();
  msg->header = header;
  msg->linear_acceleration.x = alpha[0];
  msg->linear_acceleration.y = alpha[1];
  msg->linear_acceleration.z = alpha[2];
  msg->angular_velocity.x = omega[0];
  msg->angular_velocity.y = omega[1];
  msg->angular_velocity.z = omega[2];
  pubImu_.publish(msg);
  return true;
}

void RePublisher::synchronizeData(const okvis::Time& stamp,
                                  const std::map<size_t, cv::Mat> & images,
                                  const std::map<size_t, cv::Mat> & depthImages,
			                            std::map<size_t, std::pair<okvis::Time, cv::Mat>>& syncedImages,
                                  std::map<size_t, std::pair<okvis::Time, cv::Mat>>& syncedDepthImages){
  //First add the new images
  for(auto& element : images) {
    if(element.first < imagesReceived_.size()) {
      if(element.second.channels() == 1) {
        // Grayscale
        imagesReceived_.at(element.first)[stamp.toNSec()] = element.second.clone();
      }
      else {
        // Color
        cv::cvtColor(element.second, element.second, cv::COLOR_BGR2RGB);
        imagesReceived_.at(element.first)[stamp.toNSec()] = element.second.clone();
      }
    }
  }

  for(auto& element : depthImages) {
    depthImagesReceived_.at(0)[stamp.toNSec()] = element.second.clone();
  }

  //Now we start the synchronization process
  std::set<uint64_t> allTimes;
  for(size_t i=0; i < numCams_; ++i) {
    if(slamCamIdx_.find(i) == slamCamIdx_.end()) continue; // Only consider SLAM cameras here
    for(const auto & entry : imagesReceived_.at(i)) {
      allTimes.insert(entry.first);
    }
  } //We are synchronizing the depth cameras against the ir cameras

  for(const auto & time : allTimes) {
    // note: ordered old to new
    std::vector<uint64_t> syncedTimes(numCams_, 0);
    uint64_t depthSyncedTime; //Assume that there is only one depth image (currently supported by the submapping interface)
    syncedImages.clear();
    syncedDepthImages.clear();

    okvis::Time tcheck;
    tcheck.fromNSec(time);
    bool synced = true;
    for(size_t i=0; i < numCams_; ++i) {
      bool syncedi = false;
      for(const auto & entry : imagesReceived_.at(i)) {
        okvis::Time ti;
        ti.fromNSec(entry.first);
        bool slam_use = slamCamIdx_.find(i) != slamCamIdx_.end();
        // For Non-SLAM cameras, we want the timestamp to be strictly smaller or equal than the timestamp
        if(fabs((tcheck - ti).toSec()) < OKVIS_THRESHOLD_SYNC && (tcheck >= ti || slam_use)) {
          syncedTimes.at(i) = entry.first;
          syncedImages[i] = std::make_pair(ti, imagesReceived_.at(i).at(entry.first));
          syncedi = true;
          break;
        } 
      }

      if(!syncedi) {
        synced = false;
        break;
      }
    }
    
    bool syncedDepth = true;
    if(syncDepthImages_){
      syncedDepth = false;
      for(const auto & entry : depthImagesReceived_.at(0)){
        okvis::Time tdepth;
        tdepth.fromNSec(entry.first);
        //There is a higher unsynchronization between depth images and ir images from the realsense
        if(std::abs((tcheck - tdepth).toSec()) < OKVIS_THRESHOLD_SYNC && tcheck >= tdepth) {
          depthSyncedTime = entry.first;
          syncedDepthImages[0] = std::make_pair(tdepth, depthImagesReceived_.at(0).at(entry.first));
          syncedDepth = true;
          break;
        }
      }
    }

    if(synced && syncedDepth) {
      for(size_t i=0; i < numCams_; ++i) {
        const int size0 = imagesReceived_.at(i).size();
        auto end = std::find_if(imagesReceived_.at(i).begin(), imagesReceived_.at(i).end(),
                                [&syncedTimes, i](const auto& x) { return x.first > syncedTimes.at(i); });
        imagesReceived_.at(i).erase(imagesReceived_.at(i).begin(), end);
        const int size1 = imagesReceived_.at(i).size();
        if (size0-size1>1) {
          LOG(WARNING) << "dropped " << size0-size1-1 << " unsyncable frame(s) of camera " << i << " before t=" << tcheck;
        }
      } 

      if(syncDepthImages_) {
        const int size0 = depthImagesReceived_.at(0).size();
        auto end = std::find_if(depthImagesReceived_.at(0).begin(), depthImagesReceived_.at(0).end(),
                                [depthSyncedTime](const auto& x) { return x.first > depthSyncedTime; });
        depthImagesReceived_.at(0).erase(depthImagesReceived_.at(0).begin(), end);
        const int size1 = depthImagesReceived_.at(0).size();
        if (size0-size1>1) {
          LOG(WARNING) << "dropped " << size0-size1-1 << " unsyncable frame(s) of depth camera before t=" << tcheck;
        }
      }
      break;
    } else {
      syncedImages.clear();
      syncedDepthImages.clear();
    }
  }

}
    
bool RePublisher::addImages(const okvis::Time & stamp,
                            const std::map<size_t, cv::Mat> & images,
                            const std::map<size_t, cv::Mat> & depthImages) {
  bool success = true;
  std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorImages;
  std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorDepthImages;
  if(!rgbPublisherVector_.empty() || !depthPublisherVector_.empty()){
    synchronizeData(stamp, images, depthImages, estimatorImages, estimatorDepthImages);
  } else {
    for(const auto& image: images) {
      estimatorImages.at(image.first) = std::make_pair(stamp, image.second.clone());
    }

    for(const auto& depthImage : depthImages) {
      estimatorDepthImages.at(depthImage.first) = std::make_pair(stamp, depthImage.second.clone());
    }
  }

  if(estimator_ && !estimatorImages.empty()) {
    #ifdef OKVIS_USE_NN
    okvis::Processor* casted_processor = dynamic_cast<okvis::Processor*>(estimator_);
    if (casted_processor) {
      if(!casted_processor->addImages(estimatorImages, estimatorDepthImages)){
        success = false;
      }
    }
    else {
      success = addSyncedImages(estimatorImages, estimatorDepthImages);
    }
    #else
      success = addSyncedImages(estimatorImages, estimatorDepthImages);
    #endif
  }

  addDepthMeasurement(estimatorImages, estimatorDepthImages);
  return success;
}

bool RePublisher::addDepthMeasurement(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorImages,
                                      const std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorDepthImages) {

  if(depthImageCallback_ && estimatorDepthImages.size() > 0) {
    std::map<size_t, std::vector<okvis::CameraMeasurement>> camera_readings;

    for(const auto& image : estimatorImages) {
      okvis::CameraMeasurement cam_reading;
      cam_reading.timeStamp = image.second.first;
      cam_reading.measurement.image = image.second.second.clone();
      camera_readings[image.first] = {cam_reading};
    }

    for(const auto& depthImage : estimatorDepthImages){
      if(camera_readings.find(depthImage.first) != camera_readings.end()){
        bool added = false;
        for(auto& measurement : camera_readings.at(depthImage.first)) {
          if(measurement.timeStamp == depthImage.second.first) {
            added = true;
            measurement.measurement.depthImage = depthImage.second.second.clone();
          }
        }

        if(!added) {
          okvis::CameraMeasurement cam_reading;
          cam_reading.timeStamp = depthImage.second.first;
          cam_reading.measurement.depthImage = depthImage.second.second.clone();
          camera_readings[depthImage.first].push_back(cam_reading);
        }

      } else {
        okvis::CameraMeasurement cam_reading;
        cam_reading.timeStamp = depthImage.second.first;
        cam_reading.measurement.depthImage = depthImage.second.second.clone();
        camera_readings[depthImage.first] = {cam_reading};
      }
    }
    
    depthImageCallback_(camera_readings);

  }
  return true;
}

bool RePublisher::addSyncedImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>>& estimatorImages,
                                  const std::map<size_t, std::pair<okvis::Time, cv::Mat>>& estimatorDepthImages) {
  std::map <size_t, cv::Mat> syncedImages, syncedDepthImages;
  
  // Set synced images
  for(const auto& image : estimatorImages){
    syncedImages[image.first] = image.second.second.clone();
  }
  // Set synced depth images
  for(const auto& image : estimatorDepthImages){
    syncedDepthImages[image.first] = image.second.second.clone();
  }

  okvis::Time t = estimatorImages.begin()->second.first;
  if(!estimator_->addImages(t, syncedImages, syncedDepthImages)) {
    return false;
  }
  return true;
}

}  // namespace okvis
