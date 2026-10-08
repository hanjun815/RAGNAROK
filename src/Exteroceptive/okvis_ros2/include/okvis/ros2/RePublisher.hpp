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
 * @file RePublisher.hpp
 * @brief Header file for the RePublisher class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_REPUBLISHER_HPP_
#define INCLUDE_OKVIS_REPUBLISHER_HPP_

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#if __has_include(<cv_bridge/cv_bridge.hpp>) // requires GCC >= 5
  #include <cv_bridge/cv_bridge.hpp>
#else
  #include <cv_bridge/cv_bridge.h> // ros2 changed to .hpp some point...
#endif
#pragma GCC diagnostic pop
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <okvis/ThreadedSlam.hpp>
#include <okvis/ThreadedPublisher.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief Publishes images and IMU as ROS2 messages.
class RePublisher {
public:
  typedef std::function<
          bool(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames)> DepthImageCallback;

  /**
   * \brief              Constructor with node and number of cameras.
   * \param node         ROS2 node.
   * \param numCams      number of cameras.
   */
  RePublisher(std::shared_ptr<rclcpp::Node> node, 
              const okvis::ViParameters& viParameters,
              std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
              std::shared_ptr<ThreadedPublisher> threadedPublisher);

  RePublisher(std::shared_ptr<rclcpp::Node> node, 
              const okvis::ViParameters& viParameters,
              ViInterface* estimator,
              std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
              std::shared_ptr<ThreadedPublisher> threadedPublisher);
  /**
   * \brief              Name the topics.
   * \param imuTopic     IMU topic name.
   * \param camTopic     Image topic name.
   * \param rgbTopic     RGB image topic name.
   * \param depthTopic   Depth image topic name.
   */
  void setTopics(const std::string& imuTopic, 
                 const std::string& camTopic,
                 const std::string& rgbTopic = std::string(),
                 const std::string& depthTopic = std::string());
    
  /// \name Add measurements to the algorithm
  /// \{
  /**
   * \brief              Add a new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \param depthImages  The depth images.
   * \warning The frame consumer loop does not support using existing keypoints yet.
   * \warning Already specifying whether this frame should be a keyframe is not implemented yet.
   * \return             Returns true normally. False, if the previous one has not been processed
   *                     yet.
   */
  bool publishImages(const okvis::Time &stamp,
                     const std::map<size_t, cv::Mat> &images,
                     const std::map<size_t, cv::Mat> &depthImages);

  /**
   * \brief          Add an IMU measurement.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return Returns true normally. False if the previous one has not been processed yet.
   */
  bool publishImuMeasurement(const okvis::Time & stamp,
                         const Eigen::Vector3d & alpha,
                         const Eigen::Vector3d & omega);

  /// \name Add measurements to the algorithm
  /// \{
  /**
   * \brief              Add a new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \warning The frame consumer loop does not support using existing keypoints yet.
   * \warning Already specifying whether this frame should be a keyframe is not implemented yet.
   * \return             Returns true normally. False, if the previous one has not been processed
   *                     yet.
   */
  bool addImages(const okvis::Time & stamp,
                 const std::map<size_t, cv::Mat> & images,
                 const std::map<size_t, cv::Mat> & depthImages);

  /**
   * @brief synchronization process for images. Important for synchronizing rgb images and infrared, 
   * which becomes crucial for further steps if this is needed. 
   */
  void synchronizeData(const okvis::Time& stamp,
                       const std::map<size_t, cv::Mat> & images,
                       const std::map<size_t, cv::Mat> & depthImages,
                       std::map<size_t, std::pair<okvis::Time, cv::Mat>>& syncedImages,
                       std::map<size_t, std::pair<okvis::Time, cv::Mat>>& syncedDepthImages);
  
  /**
   * @brief Add images to the estimator_'s callback.
   */
  bool addSyncedImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>>& estimatorImages,
                       const std::map<size_t, std::pair<okvis::Time, cv::Mat>>& estimatorDepthImages);

  /**
    * @brief Adds a depth image to the measurement Queue
    * @param[in] stamp The timestamp
    * @param[in] depthImage The depth image
    * 
    * @return True if successful
    */
  bool addDepthMeasurement(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorImages,
                           const std::map<size_t, std::pair<okvis::Time, cv::Mat>> estimatorDepthImages);

  void setMappingCallback(const RePublisher::DepthImageCallback & depthImageCallback) {
    depthImageCallback_ = depthImageCallback;
  }
                              
private:

  std::shared_ptr<rclcpp::Node> node_; ///< The node.
  ViInterface* estimator_;

  size_t numCams_ = 0; ///< Number of cameras.
  std::set<size_t> slamCamIdx_; ///< Set of cam indices used for estimator.
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher_;
  std::shared_ptr<ThreadedPublisher> threadedPublisher_;
  /// \brief The image publishers.
  std::vector<okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::Image>> camPublisherVector_;
  /// \brief The rgb image publishers.
  std::vector<okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::Image>> rgbPublisherVector_;
  /// \brief The depth image publishers.
  std::vector<okvis::ThreadedPublisher::PublisherHandle<std::tuple<okvis::Time, cv::Mat>>> depthPublisherVector_;
  /// \brief The quantized and transformed debug depth image publishers.
  std::vector<okvis::ThreadedPublisher::PublisherHandle<std::tuple<okvis::Time, cv::Mat>>> debugDepthPublisherVector_;

  okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::Imu> pubImu_; ///< IMU publisher

  std::string imuTopic_; ///< IMU topic name.
  std::string camTopic_; ///< Camera topic name.
  std::string rgbTopic_; ///< RGB image topic name.
  std::string depthTopic_; ///< Depth image topic name.
  RePublisher::DepthImageCallback depthImageCallback_;

  //Synchronization in the RePublisher
  bool syncDepthImages_ = false;
  std::vector<std::map<uint64_t, cv::Mat>> imagesReceived_; ///< The images obtained and buffered (to sync).
  std::vector<std::map<uint64_t, cv::Mat>> depthImagesReceived_; ///> The depth images obtained and buffered (to sync)
};

}

#endif /* INCLUDE_OKVIS_REPUBLISHER_HPP_ */
