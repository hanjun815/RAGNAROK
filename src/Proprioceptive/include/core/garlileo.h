// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
//
// Copyright (c) 2024
//   School of Geodesy and Geomatics (SGG), Wuhan University, China
//   Based on: "River: A Tightly-Coupled Radar-Inertial Velocity Estimator
//              Based on Continuous-Time Optimization"
//   Upstream: https://github.com/Unsigned-Long/River
//   Author:   Shuolong Chen
//
// Copyright (c) 2025
//   Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
//
// See LICENSE for the full MIT License text.

#ifndef GARLILEO_H
#define GARLILEO_H
#include "config/configor.h"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include "core/state_manager.h"
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <deque>
#include <mutex>
#include <chrono>
#include <optional>

namespace garlileo {
    class GaRLILEO {
    public:
        using Ptr = std::shared_ptr<GaRLILEO>;

    private:
        rclcpp::Node::SharedPtr node_;
        rclcpp::Node::SharedPtr handler;
        Configor::Ptr           configor;

        DataManager::Ptr                 dataMagr;
        StateManager::Ptr                stateMagr;
        std::shared_ptr<std::thread>     stateMagrThread;

        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPublisher;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr splinePublisher;
        rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr gravityPublisher;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr     posePublisher;
        rclcpp::Subscription<builtin_interfaces::msg::Time>::SharedPtr okvis_sync_stamp_sub_;

        nav_msgs::msg::Path pose_path;
        nav_msgs::msg::Path pose_path_G;

        bool first_publish_flag    = true;
        bool first_publish_flag_G  = true;

        double prev_time                 = 0.0;
        double last_timestamp_pose_pub_  = 0.0;
        double prev_time_G               = 0.0;
        double last_timestamp_pose_pub_G = 0.0;
        double trajectory_length         = 0.0;
        double last_publish_time_        = 0.0;  // Wall-clock time of last spline state publication
        // The status display is independent of the high-rate sensor publishers.
        std::optional<std::chrono::steady_clock::time_point> last_dashboard_log_time_;
        bool cam_state_initialized_      = false;
        double last_cam_state_time_      = 0.0;
        Eigen::Vector3d cam_position_W_  = Eigen::Vector3d::Zero();
        Eigen::Vector3d last_cam_velocity_W_ = Eigen::Vector3d::Zero();
        
        rclcpp::TimerBase::SharedPtr publish_timer_;  // Timer for accurate publishing rate
        // Sync timestamp queue for OKVIS camera-time aligned publishing of spline/gravity.
        // We enqueue OKVIS synchronized stamps and publish once spline states are available at that exact time.
        std::mutex cam_stamp_mutex_;
        std::deque<rclcpp::Time> cam_stamp_queue_;
        size_t cam_stamp_queue_max_ = 10000;
        size_t cam_stamp_max_publishes_per_tick_ = 200;
        double last_queue_warn_wall_sec_ = 0.0;

        // std::string rpm = R"(
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⢤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡍⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣷⡙⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⡿⠛⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣺⡟⠃⠀⠀⠀⠙⢿⣧⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣽⣧⠀⢿⠂⠀⢿⠇⢀⣽⢿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⠟⣥⣮⡹⢿⣦⡀⠀⢀⣴⡿⢋⣵⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣔⣮⣧⡿⠏⠛⣿⣦⡙⢿⣶⡿⢋⣴⡿⠋⢻⣷⡷⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠈⣹⣿⠄⠀⠀⠙⢿⣦⣩⣶⠿⠋⠀⠀⠺⣿⣄⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
        // ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠁⠀⠀⠀⠀⠀⠙⠿⠉⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                                                                
        //               ___  ___  __  ___
        //              / _ \/ _ \/  |/  /
        //             / , _/ ___/ /|_/ / 
        //            /_/|_/_/  /_/  /_/  
        //     ___  ____  ___  ____  ___________________
        //    / _ \/ __ \/ _ )/ __ \/_  __/  _/ ___/ __/
        //   / , _/ /_/ / _  / /_/ / / / _/ // /___\ \  
        //  /_/|_|\____/____/\____/ /_/ /___/\___/___/  
        // )"; 
        std::string rpm = R"(                               
         ______      ____  __    ______    __________ 
        / ____/___ _/ __ \/ /   /  _/ /   / ____/ __ \
       / / __/ __ `/ /_/ / /    / // /   / __/ / / / /
      / /_/ / /_/ / _, _/ /____/ // /___/ /___/ /_/ / 
      \____/\__,_/_/ |_/_____/___/_____/_____/\____/   
        )"; 

    public:
        explicit GaRLILEO(const Configor::Ptr &configor);

        static Ptr Create(const Configor::Ptr &configor);

        void Run();

        void Save();

        void save_pose_tum(const std::string& filename, 
                   const std::vector<std::pair<double, Eigen::Vector3d>>& velocity,
                   const std::vector<std::pair<double, Sophus::SO3d>>& quatVec);
        void log_fancy(double current_time_s, geometry_msgs::msg::PoseStamped& pose_stamped, std::optional<StateManager::StatePack> &status);
        void HandleOkvisSyncStamp(const builtin_interfaces::msg::Time::ConstSharedPtr &msg);
        void EnqueueSyncStamp(const rclcpp::Time &stamp_ros);
        inline std::string string_from_double(double d, int precision = 3) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(precision) << d;
            return ss.str();
        }   

    protected:
        void PrepareOkvisInitializationTime();
        void PublishGaRLILEOState(const GaRLILEOStatus::StatusPack &status);
    };
}

#endif
