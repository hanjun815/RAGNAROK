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

#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/vector3.hpp>

/**
 * @brief Replacement functions for eigen_conversions package which is no longer available in ros2
 * 
 * This namespace provides direct conversion functions between Eigen types and ROS message types
 * to replace the functionality from the eigen_conversions package.
 */
namespace tf {
    inline void quaternionMsgToEigen(const geometry_msgs::msg::Quaternion &q_msg, Eigen::Quaterniond &q) {
        q = Eigen::Quaterniond(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    }

    inline void quaternionEigenToMsg(const Eigen::Quaterniond &q, geometry_msgs::msg::Quaternion &q_msg) {
        q_msg.x = q.x();
        q_msg.y = q.y();
        q_msg.z = q.z();
        q_msg.w = q.w();
    }

    inline void pointMsgToEigen(const geometry_msgs::msg::Point &p_msg, Eigen::Vector3d &p) {
        p.x() = p_msg.x;
        p.y() = p_msg.y;
        p.z() = p_msg.z;
    }

    inline void pointEigenToMsg(const Eigen::Vector3d &p, geometry_msgs::msg::Point &p_msg) {
        p_msg.x = p.x();
        p_msg.y = p.y();
        p_msg.z = p.z();
    }

    inline void poseMsgToEigen(const geometry_msgs::msg::Pose &pose_msg, Eigen::Isometry3d &pose) {
        Eigen::Quaterniond q;
        quaternionMsgToEigen(pose_msg.orientation, q);
        pose = Eigen::Isometry3d::Identity();
        pose.linear() = q.toRotationMatrix();
        
        Eigen::Vector3d p;
        pointMsgToEigen(pose_msg.position, p);
        pose.translation() = p;
    }

    inline void poseEigenToMsg(const Eigen::Isometry3d &pose, geometry_msgs::msg::Pose &pose_msg) {
        Eigen::Quaterniond q(pose.linear());
        quaternionEigenToMsg(q, pose_msg.orientation);
        pointEigenToMsg(pose.translation(), pose_msg.position);
    }

    inline void vectorMsgToEigen(const geometry_msgs::msg::Vector3 &v_msg, Eigen::Vector3d &v) {
        v.x() = v_msg.x;
        v.y() = v_msg.y;
        v.z() = v_msg.z;
    }

    inline void vectorEigenToMsg(const Eigen::Vector3d &v, geometry_msgs::msg::Vector3 &v_msg) {
        v_msg.x = v.x();
        v_msg.y = v.y();
        v_msg.z = v.z();
    }

    inline void transformMsgToEigen(const geometry_msgs::msg::Transform &t_msg, Eigen::Affine3d &t) {
        Eigen::Quaterniond q;
        q = Eigen::Quaterniond(t_msg.rotation.w, t_msg.rotation.x, t_msg.rotation.y, t_msg.rotation.z);
        t = Eigen::Affine3d::Identity();
        t.linear() = q.toRotationMatrix();
        t.translation() << t_msg.translation.x, t_msg.translation.y, t_msg.translation.z;
    }

    inline void transformEigenToMsg(const Eigen::Affine3d &t, geometry_msgs::msg::Transform &t_msg) {
        Eigen::Quaterniond q(t.linear());
        t_msg.rotation.x = q.x();
        t_msg.rotation.y = q.y();
        t_msg.rotation.z = q.z();
        t_msg.rotation.w = q.w();
        t_msg.translation.x = t.translation().x();
        t_msg.translation.y = t.translation().y();
        t_msg.translation.z = t.translation().z();
    }
}