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

#include <vector>
#include "core/garlileo.h"

#include <fstream>
#include <iomanip>
#include "cereal/types/utility.hpp"
using std::placeholders::_1;

namespace garlileo {

    GaRLILEO::GaRLILEO(const Configor::Ptr &configor)
            : handler(rclcpp::Node::make_shared("garlileo")), configor(configor),
              dataMagr(DataManager::Create(handler, configor)),
              stateMagr(StateManager::Create(dataMagr, configor)),
              stateMagrThread(std::make_shared<std::thread>(&StateManager::Run, stateMagr)) {

        posePublisher     = handler->create_publisher<nav_msgs::msg::Path>(
                                "/path",
                                rclcpp::QoS(100'000));

        odomPublisher     = handler->create_publisher<nav_msgs::msg::Odometry>(
                                "/Odometry",
                                rclcpp::QoS(100'000));
        splinePublisher   = handler->create_publisher<nav_msgs::msg::Odometry>(
                                "/garlileo/spline_state",
                                rclcpp::QoS(200));
        gravityPublisher  = handler->create_publisher<geometry_msgs::msg::Vector3Stamped>(
                                "/garlileo/gravity",
                                rclcpp::QoS(200));
        const std::string okvis_sync_stamp_topic =
            handler->declare_parameter<std::string>(
                "okvis_sync_stamp_topic", "/okvis/synchronized_stamp");
        spdlog::info("[GaRLILEO][Sync] fixed to OKVIS synchronized stamps, topic={}",
                     okvis_sync_stamp_topic);
        okvis_sync_stamp_sub_ =
            handler->create_subscription<builtin_interfaces::msg::Time>(
                okvis_sync_stamp_topic,
                rclcpp::QoS(4000).reliable(),
                std::bind(&GaRLILEO::HandleOkvisSyncStamp, this, std::placeholders::_1));

        const int queue_max_param =
            handler->declare_parameter<int>("cam_stamp_queue_max", 10000);
        cam_stamp_queue_max_ = static_cast<size_t>(queue_max_param > 1 ? queue_max_param : 1);

        const int max_pub_per_tick_param =
            handler->declare_parameter<int>("cam_stamp_max_publishes_per_tick", 200);
        cam_stamp_max_publishes_per_tick_ =
            static_cast<size_t>(max_pub_per_tick_param > 1 ? max_pub_per_tick_param : 1);

        spdlog::info(
            "[GaRLILEO][SyncQueue] queue_max={}, max_publishes_per_tick={}, "
            "mode=exact_stamp_with_safety_lag, safety_lag={:.3f}s",
            cam_stamp_queue_max_, cam_stamp_max_publishes_per_tick_,
            configor->preference.SplineStateSafetyLagSec);

        // Create a timer for accurate publishing rate
        const double publish_period = 1.0 / configor->preference.PublishRate;
        publish_timer_ = handler->create_wall_timer(
            std::chrono::duration<double>(publish_period),
            [this]() {
                auto status = GaRLILEOStatus::GetStatusPackSafely();
                this->PublishGaRLILEOState(status);
            }
        );

    }

    GaRLILEO::Ptr GaRLILEO::Create(const Configor::Ptr &configor) {
        return std::make_shared<GaRLILEO>(configor);
    }

    void GaRLILEO::Run()
    {
        auto exec = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
        exec->add_node(handler);
        std::thread spin_thr([&](){ exec->spin(); });

        rclcpp::Rate rate(configor->preference.IncrementalOptRate);

        while (rclcpp::ok()) {
            auto status = GaRLILEOStatus::GetStatusPackSafely();
            if (GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::ShouldQuit, status.StateMagr)) {
                spdlog::warn("GaRLILEO StateManager thread quits.");
                break;
            }

            // Publishing is now handled by the timer, so we don't call PublishGaRLILEOState here

            if (GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::HasInitialized, status.StateMagr)) {
                dataMagr->EraseOldDataPieceSafely(status.ValidStateEndTime - 0.2);
            }
            rate.sleep();
        }
        exec->cancel();
        spin_thr.join();
        stateMagrThread->join();
    }

    void GaRLILEO::save_pose_tum(const std::string& filename,
        const std::vector<std::pair<double, Eigen::Vector3d>>& velocity,
        const std::vector<std::pair<double, Sophus::SO3d>>& quatVec)
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        assert(velocity.size() == quatVec.size());
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(9);
        double t_prev = velocity[0].first;
        file << t_prev << " " << position.x() << " " << position.y() << " " << position.z() << " ";
        // Every row is expressed in the first-IMU reference frame.
        const Eigen::Matrix3d R_RefW = stateMagr->GetSO3_RefToW().matrix().transpose();
        auto q0 = Sophus::SO3d(R_RefW * quatVec[0].second.matrix()).unit_quaternion();
        file << q0.x() << " " << q0.y() << " " << q0.z() << " " << q0.w() << "\n";

        for (size_t i = 1; i < velocity.size(); ++i) {
            double t = velocity[i].first;
            double dt = t - t_prev;
            t_prev = t;

            const Eigen::Vector3d& vel_body = velocity[i].second;
            const Sophus::SO3d& so3 = quatVec[i].second; //R_wb
            Eigen::Matrix3d R_wb = R_RefW * so3.matrix();  // body to First IMU(R_I0w * R_wb)

            Eigen::Vector3d vel_world = R_wb * vel_body;
            position += vel_world * dt;

            Sophus::SO3d so3_wb(R_wb);
            Eigen::Quaterniond q_wb = so3_wb.unit_quaternion();
            file << t << " "
                << position.x() << " " << position.y() << " " << position.z() << " "
                << q_wb.x() << " " << q_wb.y() << " " << q_wb.z() << " " << q_wb.w() << "\n";
        }
        file.close();

    }



    void GaRLILEO::Save() {
        // Run() joins the executor and state-manager threads before Save(). A
        // shutdown while waiting for sensor data can leave the spline bundle
        // absent or only partially initialized. Check before dereferencing it
        // or replacing output from a previous successful run.
        const auto status = GaRLILEOStatus::GetStatusPackSafely();
        const auto &splines = stateMagr->GetSplines();
        const auto &epoch_opt = dataMagr->GetGaRLILEOTimeEpoch();
        if (!GaRLILEOStatus::IsWith(
                    GaRLILEOStatus::StateManager::Status::HasInitialized, status.StateMagr)
                || !splines
                || (!configor->preference.OutputResultsWithTimeAligned && !epoch_opt)) {
            spdlog::info("No initialized trajectory to save; existing output is preserved.");
            return;
        }

        // The directory also holds this run's online calibration logs, so keep it.
        const std::string tarDir = configor->dataStream.OutputPath + "/proprioceptive";
        if (!std::filesystem::exists(tarDir) && !std::filesystem::create_directories(tarDir)) {
            throw Status(Status::Flag::WARNING, fmt::format(
                    "the output path for data, i.e., '{}', does not exist and create failed!", tarDir)
            );
        }

        auto &velSpline = splines->GetRdSpline(Configor::Preference::VelSpline);
        auto &so3Spline = splines->GetSo3Spline(Configor::Preference::SO3Spline);
        auto &gravSpline = splines->GetRdSpline(Configor::Preference::GravitySpline);
        const double epoch = configor->preference.OutputResultsWithTimeAligned ? 0.0 : *epoch_opt;
        const double velST = velSpline.MinTime(), velET = velSpline.MaxTime();
        const double so3ST = so3Spline.MinTime(), so3ET = so3Spline.MaxTime();

        velSpline.SetStartTime(velST + epoch);
        so3Spline.SetStartTime(so3ST + epoch);

        velSpline.SetStartTime(velST);
        so3Spline.SetStartTime(so3ST);

        // velocity
        const double st = std::max(velST, so3ST), et = std::min(velET, so3ET);

        std::vector<std::pair<double, Eigen::Vector3d>> gravityInB, velocityInB;
        std::vector<std::pair<double, Sophus::SO3d>> quatBtoW;
        for (double t = st; t < et;) {
            Eigen::Vector3d LIN_VEL_BtoWinB = velSpline.Evaluate(t);
            velocityInB.emplace_back(t + epoch, LIN_VEL_BtoWinB);

            auto SO3_CurToW = so3Spline.Evaluate(t);
            quatBtoW.emplace_back(t + epoch, SO3_CurToW);

            Eigen::Vector3d GRAV = gravSpline.Evaluate(t);
            gravityInB.emplace_back(t + epoch, GRAV);

            t += 0.01;
        }
        save_pose_tum(tarDir +"/poses.txt", velocityInB, quatBtoW);

        spdlog::info("pose of 'GaRLILEO' has been saved to '{}'.", tarDir + "/poses.txt");
    }

    void GaRLILEO::log_fancy(double current_time_s, geometry_msgs::msg::PoseStamped& pose_stamped, std::optional<StateManager::StatePack> &status) {

        // Formatting and flushing the full dashboard at every publisher tick
        // floods launch logs and delays the camera-time publications below.
        // Only the display is rate limited; publishing still runs every tick.
        const auto now = std::chrono::steady_clock::now();
        if (last_dashboard_log_time_
            && now - *last_dashboard_log_time_ < std::chrono::seconds(1)) {
            return;
        }
        last_dashboard_log_time_ = now;

        std::cout<<"\033[2J\033[1;1H"; //clear screen
        std::cout<<"\033[0m" << rpm <<std::endl; 
        std::cout<<"\033[0m"; 

        std::time_t current_time = std::time(nullptr);
        double elapsed_time = current_time_s;

        std::string asc_time = std::asctime(std::localtime(&current_time)); asc_time.pop_back();
        std::cout << "| " << std::left << asc_time;
        std::cout << std::right << std::setfill(' ') << std::setw(35)
        << "Elapsed Time: " + string_from_double(elapsed_time) + " seconds "
        << "|" << std::endl;

        std::cout << "|------------------------------------------------------------|" << std::endl;

        const auto& p = pose_stamped.pose.position;
        std::cout << "| " << std::left << std::setfill(' ') << std::setw(59)
                << "Position (x,y,z)    [m] : " 
                + string_from_double(p.x) + " " 
                + string_from_double(p.y) + " " 
                + string_from_double(p.z)
                << "|" << std::endl;

        const auto& o = pose_stamped.pose.orientation;
        const Eigen::Quaterniond q(o.w, o.x, o.y, o.z);

        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2); 
        euler *= 180.0 / M_PI;

        std::cout << "| " << std::left << std::setfill(' ') << std::setw(60)
                << "Orientation (r,p,y) [°] : "
                + string_from_double(euler(0)) + " "
                + string_from_double(euler(1)) + " "
                + string_from_double(euler(2))
                << "|" << std::endl;

        std::cout << "| " << std::left << std::setfill(' ') << std::setw(59)
        << "Lin Velocity {B}  [xyz] : " + string_from_double(status->LIN_VEL_CurToRefInCur(0)) + " "
                                    + string_from_double(status->LIN_VEL_CurToRefInCur(1)) + " "
                                    + string_from_double(status->LIN_VEL_CurToRefInCur(2)) << "|" << std::endl;

        std::cout << "| " << std::left << std::setfill(' ') << std::setw(59)
        << "Trajectory Length   [m] : " + string_from_double(trajectory_length) << "|" << std::endl;

        std::cout << "|------------------------------------------------------------|" << std::endl;


        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(59)
        // << "Acc. Bias (x,y,z) [m/s2]  : " + string_from_double(state_point.ba(0)) + " " 
        // + string_from_double(state_point.ba(1)) + " " + string_from_double(state_point.ba(2)) << "|" << std::endl;

        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(59)
        // << "Gyro Bias (x,y,z) [rad/s] : " + string_from_double(state_point.bg(0)) + " " 
        // + string_from_double(state_point.bg(1)) + " " + string_from_double(state_point.bg(2)) << "|" << std::endl;
    
        // std::cout << "|------------------------------------------------------------|" << std::endl;

        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(26)
        // << "Effective Points    [#] : " <<  std::left << std::setw(33) << n_effective_points << "|" << std::endl;

        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(26)
        // << "Intensity Features  [#] : " << std::left << std::setw(6) << n_features << std::left << std::setw(13) 
        // << " Added: " + std::to_string(n_added) << std::left << std::setw(14)<< " Removed: " + std::to_string(n_removed) 
        // << "|" << std::endl;

        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(26)
        // << "Uninformative Dir.  [#] : " << std::left << std::setw(33) << n_uninformative << "|" << std::endl;

        // std::cout << "|------------------------------------------------------------|" << std::endl;

        // double mean_s = timing::Timing::GetMeanSeconds("all");
        // double min_s = timing::Timing::GetMinSeconds("all");
        // double max_s = timing::Timing::GetMaxSeconds("all");
        // std::cout << "| " << std::left << std::setfill(' ') << std::setw(26)
        // << "Computation Time    [s] : " << std::left << "Avg: " << string_from_double(mean_s) << std::left 
        // << " Max: " << string_from_double(max_s)  << std::left << " Min: " << string_from_double(min_s) 
        // << " |" << std::endl;
    }

    /// \brief Convert GaRLILEO internal time (seconds, starting at 0) to ROS time (seconds).
    /// If OutputResultsWithTimeAligned is true, we keep the aligned (relative) time.
    /// Otherwise we add the epoch (first sensor stamp) so that external consumers using ROS header stamps
    /// (e.g. OKVIS2-X) can time-associate correctly.
    static inline double toRosTimeSeconds(const garlileo::Configor::Ptr& configor,
                                         const garlileo::DataManager::Ptr& dataMagr,
                                         double t_internal_sec) {
        // Always publish ROS epoch time if epoch is known, so external consumers align timestamps.
        if (dataMagr) {
            const auto& epoch_opt = dataMagr->GetGaRLILEOTimeEpoch();
            if (epoch_opt) {
                return t_internal_sec + *epoch_opt;
            }
        }
        return t_internal_sec;
    }

    void GaRLILEO::PrepareOkvisInitializationTime() {
        if (dataMagr->GetOkvisInitTimeSafely()) {
            return;
        }
        const auto epoch = dataMagr->GetGaRLILEOTimeEpoch();
        if (!epoch) {
            return;
        }

        // Initialization needs the camera stamp before HasInitialized becomes
        // true. Keep the publication queue intact, and retry on later timer
        // ticks when IMU coverage has not yet reached the requested overlap.
        std::vector<rclcpp::Time> candidates;
        {
            std::lock_guard<std::mutex> lock(cam_stamp_mutex_);
            candidates.assign(cam_stamp_queue_.begin(), cam_stamp_queue_.end());
        }
        for (const auto &stamp : candidates) {
            if (dataMagr->TrySetOkvisInitTimeFromCameraStamp(stamp.seconds() - *epoch, 0.02)) {
                return;
            }
        }
    }

    void GaRLILEO::PublishGaRLILEOState(const GaRLILEOStatus::StatusPack &status) {
        PrepareOkvisInitializationTime();
        // Only publish if the system has initialized
        if (!GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::HasInitialized, status.StateMagr)) {
            return;
        }

        // Timer-based publishing: no need to check interval here as timer handles it
        // But we still track last_publish_time_ for potential future use
        const double current_time = rclcpp::Clock().now().seconds();

        // Try to get state at ValidStateEndTime - 0.3, but if that fails, use the latest valid time from spline
        double t_query_state = status.ValidStateEndTime - 0.3;
        std::optional<StateManager::StatePack> state = this->stateMagr->GetStatePackSafely(t_query_state);

        // If failed, try to get the latest valid time from spline
        if (state == std::nullopt) {
            const auto range = this->stateMagr->GetSplineTimeRangeSafely();
            if (range) {
                const auto [min_valid_time, max_valid_time] = *range;
                // Clamp to valid range
                t_query_state = std::max(min_valid_time, std::min(max_valid_time - 0.01, status.ValidStateEndTime));
                state = this->stateMagr->GetStatePackSafely(t_query_state);
            }
            // If still failed, skip this publication but update last_publish_time_ to avoid blocking
            if (state == std::nullopt) {
                last_publish_time_ = current_time;
                return;
            }
        }

        geometry_msgs::msg::PoseStamped pose_stamped;
        const double t_pose_pub = toRosTimeSeconds(configor, dataMagr, state->timestamp);
        pose_stamped.header.stamp    = rclcpp::Time(static_cast<int64_t>(t_pose_pub * 1e9));
        pose_stamped.header.frame_id = "world";
               
        
        double dt = this->first_publish_flag ? 0.0 : state->timestamp - this->prev_time;
        geometry_msgs::msg::Pose last_pose;
        last_pose = this->first_publish_flag ? geometry_msgs::msg::Pose() : this->pose_path.poses.back().pose;
        this->first_publish_flag = false;
        this->prev_time = state->timestamp;

        pose_stamped.pose.position.x = last_pose.position.x + (state->SO3_CurToRef * state->LIN_VEL_CurToRefInCur * dt)(0);
        pose_stamped.pose.position.y = last_pose.position.y + (state->SO3_CurToRef * state->LIN_VEL_CurToRefInCur * dt)(1);
        pose_stamped.pose.position.z = last_pose.position.z + (state->SO3_CurToRef * state->LIN_VEL_CurToRefInCur * dt)(2);
        pose_stamped.pose.orientation.x = state->SO3_CurToRef.unit_quaternion().x();
        pose_stamped.pose.orientation.y = state->SO3_CurToRef.unit_quaternion().y();
        pose_stamped.pose.orientation.z = state->SO3_CurToRef.unit_quaternion().z();
        pose_stamped.pose.orientation.w = state->SO3_CurToRef.unit_quaternion().w();

        this->pose_path.poses.push_back(pose_stamped);

        // Dashboard logging
        Eigen::Vector3d d_ref = state->SO3_CurToRef * state->LIN_VEL_CurToRefInCur * dt;
        double segment_length = d_ref.norm(); 
        this->trajectory_length += segment_length;

        log_fancy(status.ValidStateEndTime - 0.3, pose_stamped, state);

        if ((state->timestamp - this->last_timestamp_pose_pub_) > 1.0 / 5)
        {
            this->pose_path.header.stamp = rclcpp::Time(static_cast<int64_t>(t_pose_pub * 1e9));
            this->pose_path.header.frame_id = "world";
            posePublisher->publish(this->pose_path);
            this->last_timestamp_pose_pub_ = state->timestamp;
        }

        nav_msgs::msg::Odometry axis_path;
        axis_path.header.stamp = rclcpp::Time(static_cast<int64_t>(t_pose_pub * 1e9));
        axis_path.header.frame_id = "world";
        axis_path.pose.pose.position.x = pose_stamped.pose.position.x;
        axis_path.pose.pose.position.y = pose_stamped.pose.position.y;
        axis_path.pose.pose.position.z = pose_stamped.pose.position.z;
        axis_path.pose.pose.orientation.x = pose_stamped.pose.orientation.x;
        axis_path.pose.pose.orientation.y = pose_stamped.pose.orientation.y;
        axis_path.pose.pose.orientation.z = pose_stamped.pose.orientation.z;
        axis_path.pose.pose.orientation.w = pose_stamped.pose.orientation.w;

        odomPublisher->publish(axis_path);

        // Publish /garlileo/spline_state and /garlileo/gravity at *OKVIS synchronized timestamps*.
        // Stamps are buffered from /okvis/synchronized_stamp. The header stamp remains exact, but
        // publication is delayed until the requested spline state is behind the valid tail by the
        // configured safety lag, matching the stable-state policy used by /path.
        if (splinePublisher || gravityPublisher) {
            const auto epoch_opt = dataMagr->GetGaRLILEOTimeEpoch();
            if (epoch_opt) {
                const double safety_lag =
                    configor->preference.SplineStateSafetyLagSec > 0.0
                        ? configor->preference.SplineStateSafetyLagSec
                        : 0.0;
                int published = 0;
                const int max_publishes_per_tick =
                    static_cast<int>(cam_stamp_max_publishes_per_tick_);
                while (published < max_publishes_per_tick) {
                    rclcpp::Time stamp_ros;
                    {
                        std::lock_guard<std::mutex> lock(cam_stamp_mutex_);
                        if (cam_stamp_queue_.empty()) {
                            break;
                        }
                        stamp_ros = cam_stamp_queue_.front();
                    }

                    const double t_ros = stamp_ros.seconds();
                    const double t_internal = t_ros - *epoch_opt;

                    // Record OKVIS2-X compatible init time t0 (first camera stamp with IMU coverage).
                    (void)dataMagr->TrySetOkvisInitTimeFromCameraStamp(t_internal, 0.02);

                    const auto range = this->stateMagr->GetSplineTimeRangeSafely();
                    if (!range) {
                        break; // not ready yet
                    }
                    const auto [min_valid_time, max_valid_time] = *range;

                    // Fixed strict mode: evaluate state exactly at camera-sync timestamp.
                    if (t_internal < min_valid_time) {
                        const double now = rclcpp::Clock().now().seconds();
                        if (now - last_queue_warn_wall_sec_ > 1.0) {
                            spdlog::warn(
                                "[GaRLILEO][SyncQueue] dropping too-old stamp (exact mode): "
                                "t_internal={:.6f}, min_valid={:.6f}",
                                t_internal, min_valid_time);
                            last_queue_warn_wall_sec_ = now;
                        }
                        std::lock_guard<std::mutex> lock(cam_stamp_mutex_);
                        if (!cam_stamp_queue_.empty()) {
                            cam_stamp_queue_.pop_front();
                        }
                        continue;
                    }
                    const double stable_max_time =
                        std::min(max_valid_time, status.ValidStateEndTime) - safety_lag;
                    if (t_internal > stable_max_time) {
                        // Not mature enough yet. Keep the exact stamp queued until the estimator
                        // advances by safety_lag, then publish with the original header stamp.
                        break;
                    }

                    const double t_eval = t_internal;
                    Eigen::Vector3d omega_body;
                    auto st_opt = this->stateMagr->GetStatePackSafely(t_eval, &omega_body);
                    if (st_opt == std::nullopt) {
                        // Rare: on boundary. Wait and retry later.
                        break;
                    }

                    const auto &st = *st_opt;
                    const Eigen::Vector3d v_body = st.LIN_VEL_CurToRefInCur;
                    const Eigen::Vector3d v_path = st.SO3_CurToRef * v_body;

                    if (cam_state_initialized_) {
                        const double dt_cam = t_eval - last_cam_state_time_;
                        if (dt_cam > 0.0) {
                            // Integrate the world-frame endpoint velocities. Camera stamps
                            // are sparser than the saved 100 Hz trajectory; a right-endpoint
                            // rectangle adds first-order position error during acceleration.
                            cam_position_W_ += 0.5 * (last_cam_velocity_W_ + v_path) * dt_cam;
                        }
                    } else {
                        cam_position_W_.setZero();
                        cam_state_initialized_ = true;
                    }
                    last_cam_state_time_ = t_eval;
                    last_cam_velocity_W_ = v_path;

                    nav_msgs::msg::Odometry spline_msg;
                    spline_msg.header.stamp = stamp_ros;
                    spline_msg.header.frame_id = "world";
                    spline_msg.child_frame_id = "garlileo_base";
                    auto q = st.SO3_CurToRef.unit_quaternion();
                    spline_msg.pose.pose.orientation.x = q.x();
                    spline_msg.pose.pose.orientation.y = q.y();
                    spline_msg.pose.pose.orientation.z = q.z();
                    spline_msg.pose.pose.orientation.w = q.w();
                    spline_msg.pose.pose.position.x = cam_position_W_.x();
                    spline_msg.pose.pose.position.y = cam_position_W_.y();
                    spline_msg.pose.pose.position.z = cam_position_W_.z();
                    spline_msg.twist.twist.linear.x = v_body.x();
                    spline_msg.twist.twist.linear.y = v_body.y();
                    spline_msg.twist.twist.linear.z = v_body.z();
                    spline_msg.twist.twist.angular.x = omega_body.x();
                    spline_msg.twist.twist.angular.y = omega_body.y();
                    spline_msg.twist.twist.angular.z = omega_body.z();

                    if (splinePublisher) {
                        splinePublisher->publish(spline_msg);
                    }
                    if (gravityPublisher) {
                        geometry_msgs::msg::Vector3Stamped gmsg;
                        gmsg.header = spline_msg.header;
                        const Eigen::Vector3d g_W = st.SO3_CurToRef * st.gravity;
                        gmsg.vector.x = g_W.x();
                        gmsg.vector.y = g_W.y();
                        gmsg.vector.z = g_W.z();
                        gravityPublisher->publish(gmsg);
                    }

                    {
                        std::lock_guard<std::mutex> lock(cam_stamp_mutex_);
                        if (!cam_stamp_queue_.empty()) {
                            cam_stamp_queue_.pop_front();
                        }
                    }
                    published++;
                }
            }
        }

        last_publish_time_ = current_time; // Update last publish time

        {
            LOCK_GARLILEO_STATUS
            GaRLILEOStatus::StateManager::CurStatus ^= GaRLILEOStatus::StateManager::Status::NewStateNeedToPublish;
        }
    }

    void GaRLILEO::HandleOkvisSyncStamp(const builtin_interfaces::msg::Time::ConstSharedPtr &msg) {
        // Use the exact synchronized timestamp produced by OKVIS::Subscriber::synchronizeData().
        const rclcpp::Time stamp_ros(msg->sec, msg->nanosec, RCL_ROS_TIME);
        EnqueueSyncStamp(stamp_ros);
    }

    void GaRLILEO::EnqueueSyncStamp(const rclcpp::Time &stamp_ros) {
        {
            std::lock_guard<std::mutex> lock(cam_stamp_mutex_);
            if (!cam_stamp_queue_.empty() && cam_stamp_queue_.back() == stamp_ros) {
                return;  // avoid duplicate stamps from mirrored topics
            }

            if (cam_stamp_queue_.size() >= cam_stamp_queue_max_) {
                const double now = rclcpp::Clock().now().seconds();
                if (now - last_queue_warn_wall_sec_ > 1.0) {
                    spdlog::warn(
                        "[GaRLILEO][SyncQueue] full(size={}), drop oldest stamp to keep latest sync. "
                        "Increase cam_stamp_queue_max / cam_stamp_max_publishes_per_tick / PublishRate.",
                        cam_stamp_queue_.size());
                    last_queue_warn_wall_sec_ = now;
                }
                cam_stamp_queue_.pop_front();
                cam_stamp_queue_.push_back(stamp_ros);
            } else {
                cam_stamp_queue_.push_back(stamp_ros);
            }
        }
    }
}
