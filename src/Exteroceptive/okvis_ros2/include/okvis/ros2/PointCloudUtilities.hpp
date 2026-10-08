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
 * @file PointCloudUtilities.hpp
 * @brief Some functions to work with sensor_msgs::msg::PointCloud2 messages.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_PCL_UTILITIES_HPP_
#define INCLUDE_OKVIS_PCL_UTILITIES_HPP_

#include <string>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <okvis/Time.hpp>

namespace okvis {
namespace pointcloud_ros{

/**
 * @ brief check if a certain data field exists in the message
 */
bool has_field(const sensor_msgs::msg::PointCloud2& pointcloud, const std::string & field_name)
{
  for (const auto & field : pointcloud.fields) {
    if (field.name == field_name) {
      return true;
    }
  }
  return false;
}

/**
 * @ brief Get point-wise timestamps and measurements for BLK2FLY type PointCloud msgs
 */
inline void blk_lidar2points(const sensor_msgs::msg::PointCloud2& msg,
                             std::vector<okvis::Time, Eigen::aligned_allocator<okvis::Time>>& timestamps,
                             std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& rays) 
{

  timestamps.reserve(msg.width * msg.height);
  rays.reserve(msg.width * msg.height);

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x"), iter_y(msg, "y"), iter_z(msg,"z");
  sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_stamp_high(msg, "stamp_high"), iter_stamp_low(msg, "stamp_low");
  size_t counter=0;
  while (iter_x != iter_x.end()){
    rays.push_back(Eigen::Vector3d(*iter_x, *iter_y, *iter_z));
    okvis::Time ts;
    ts.fromNSec((static_cast<uint64_t>(*iter_stamp_high) << 32) | *iter_stamp_low);
    timestamps.push_back(ts);
    ++iter_x;
    ++iter_y;
    ++iter_z;
    if(iter_x != iter_x.end()){
      ++iter_stamp_low;
      ++iter_stamp_high;
    }
    counter++;
  }
}

/**
 * @ brief Get point-wise timestamps and measurements for Hesai type PointCloud msgs (Hilti22 SLAM Challenge)
 */
inline void hesai_lidar2points(const sensor_msgs::msg::PointCloud2& msg,
                               std::vector<okvis::Time, Eigen::aligned_allocator<okvis::Time>>& timestamps,
                               std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& rays)
{
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x"), iter_y(msg, "y"), iter_z(msg,"z");
  sensor_msgs::PointCloud2ConstIterator<double> iter_stamp(msg, "timestamp");

  while (iter_x != iter_x.end()){
    rays.push_back(Eigen::Vector3d(*iter_x, *iter_y, *iter_z));
    okvis::Time ts;
    ts.fromSec(*iter_stamp);
    timestamps.push_back(ts);
    ++iter_x;
    ++iter_y;
    ++iter_z;
    if(iter_x != iter_x.end()){
      ++iter_stamp;
    }
  }
}

} // namespace pointcloud_ros
} // namespace okvis

#endif // INCLUDE_OKVIS_PCL_UTILITIES_HPP_