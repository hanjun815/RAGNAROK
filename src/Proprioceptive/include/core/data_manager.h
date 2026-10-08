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

#ifndef GARLILEO_DATA_MANAGER_H
#define GARLILEO_DATA_MANAGER_H

#include "config/configor.h"
#include "sensor/imu_data_loader.h"
#include "sensor/radar_data_loader.h"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include "sensor_msgs/msg/imu.hpp"
#include "core/status.h"

#include "sensor/leg_data_loader.h"
#include "sensor/foot_data_loader.h"
#include "sensor/leg_odometry.h"

#include <spot_msgs/msg/foot_state.hpp>
#include <spot_msgs/msg/foot_state_array.hpp>
#include <spot_msgs/msg/ego_velocity_foot_state_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <functional>
#include <optional>
#include <mutex>
#include <deque>
#include <cstdint>

using namespace std::placeholders;

namespace garlileo {
    // unique locks for mutexes
#define LOCK_IMU_DATA_SEQ std::unique_lock<std::mutex> imuDataLock(DataManager::IMUDataSeqMutex);
#define LOCK_RADAR_DATA_SEQ std::unique_lock<std::mutex> radarDataLock(DataManager::RadarDataSeqMutex);
#define LOCK_RADAR1_DATA_SEQ std::unique_lock<std::mutex> radar1DataLock(DataManager::Radar1DataSeqMutex);
#define LOCK_LEG_DATA_SEQ std::unique_lock<std::mutex> legDataLock(DataManager::LegDataSeqMutex);
#define LOCK_FOOT_DATA_SEQ std::unique_lock<std::mutex> footDataLock(DataManager::FootDataSeqMutex);
#define LOCK_RADAR_INIT_FRAMES std::unique_lock<std::mutex> radarInitFramesLock(DataManager::RadarInitFramesMutex);

    class DataManager {
    public:
        using Ptr = std::shared_ptr<DataManager>;

    private:
        rclcpp::Node::SharedPtr handler;
        Configor::Ptr configor;

        // ros subscribers
        rclcpp::SubscriptionBase::SharedPtr imuSuber;
        rclcpp::SubscriptionBase::SharedPtr radarSuber;
        rclcpp::SubscriptionBase::SharedPtr radar1Suber;     // second radar
        rclcpp::SubscriptionBase::SharedPtr foot_Suber;
        rclcpp::SubscriptionBase::SharedPtr jointStateSuber; // /joint_states direct processing

        // containers to store sensor data
        std::list<IMUFrame::Ptr> imuDataSeq;
        std::list<RadarTarget::Ptr> radarDataSeq;
        std::list<RadarTarget::Ptr> radar1DataSeq;  // second radar data
        std::list<LegFrame::Ptr> legDataSeq;
        std::list<FootFrame::Ptr> footDataSeq;

        // the radar target array for initialization, size: 10
        std::list<RadarTargetArray::Ptr> radarTarAryForInit;

        std::optional<double> GARLILEO_TIME_EPOCH;
        mutable std::mutex timeEpochMutex_;

        double GetOrInitializeTimeEpoch(double timestamp, const char* source);
        int GetContactDurationSafely() const;

        // OKVIS2-X compatible initialisation time (t0): first camera timestamp for which
        // IMU data covers [t0-imuTemporalOverlap, t0+imuTemporalOverlap].
        mutable std::mutex okvisInitMutex_;
        std::optional<double> okvisInitTimeT0_;

        bool contact=false;
        int contact_duration = 0;

        // Internal leg odometry processor (replaces spot_msgs external node)
        LegOdometryProcessor::Ptr legOdomProcessor_;

        // Constant gyroscope bias removed from every sample when UseGyroBias is set.
        Eigen::Vector3d gyroBias_ = Eigen::Vector3d::Zero();

        // Decodes SPOT_ego_Velocity foot states (foot_msg_format: spot_ego_velocity).
        rclcpp::Serialization<spot_msgs::msg::EgoVelocityFootStateArray> egoVelocityFootSerialization_;
        std::uint64_t droppedEgoVelocityFootMessages_ = 0;

        struct PendingJointState {
            double timestamp;  // epoch-relative, before leg time calibration
            sensor_msgs::msg::JointState::ConstSharedPtr message;
        };
        // Both joint and foot callbacks can drain this queue. The processor's
        // kinematic/contact history must advance serially in joint input order.
        std::mutex jointStateProcessingMutex_;
        std::deque<PendingJointState> pendingJointStates_;
        std::uint64_t discardedPendingJointStates_ = 0;
        void ProcessPendingJointStates();
        void ProcessJointStateWithContact(
                const sensor_msgs::msg::JointState::ConstSharedPtr& msg,
                const FootFrame::Ptr& foot, double timestamp);

        bool imu_is_come = false;
        // Only the mutually-exclusive IMU callback accesses this timestamp.
        // It remains valid when the shared measurement buffer is pruned.
        double previousImuInternalTime_ = 0.0;
        Eigen::Quaternion<double> init_orient = Eigen::Quaternion<double>::Identity();
        Eigen::Quaternion<double> prev_orient = Eigen::Quaternion<double>::Identity();
        Eigen::Quaternion<double> SO3_W2Ref = Eigen::Quaternion<double>::Identity();
        

    public:
        // mutexes employed in multi-thread framework
        static std::mutex IMUDataSeqMutex;
        static std::mutex RadarDataSeqMutex;
        static std::mutex Radar1DataSeqMutex;
        static std::mutex LegDataSeqMutex;
        static std::mutex FootDataSeqMutex;
        static std::mutex RadarInitFramesMutex;

    public:
        explicit DataManager(const rclcpp::Node::SharedPtr &handler, const Configor::Ptr &configor);

        static Ptr Create(const rclcpp::Node::SharedPtr &handler, const Configor::Ptr &configor);

        [[nodiscard]] std::optional<double> GetGaRLILEOTimeEpoch() const;

        [[nodiscard]] double GetEndTimeSafely() const;

        [[nodiscard]] double GetIMUEndTimeSafely() const;

        [[nodiscard]] double GetRadarEndTimeSafely() const;

        const std::list<IMUFrame::Ptr> &GetIMUDataSeq() const;

        const std::list<RadarTarget::Ptr> &GetRadarDataSeq() const;

        void ShowDataStatus() const;

        // OKVIS2-X style initialisation support.
        // Store the first camera timestamp (internal time) that satisfies IMU coverage around it.
        // Returns true if the time was set by this call.
        bool TrySetOkvisInitTimeFromCameraStamp(double t0_internal_sec, double imuTemporalOverlap);
        std::optional<double> GetOkvisInitTimeSafely() const;

        [[nodiscard]] std::list<RadarTargetArray::Ptr> GetRadarTarAryForInitSafely() const;

        std::list<IMUFrame::Ptr> ExtractIMUDataPieceSafely(double start, double end);

        std::list<IMUFrame::Ptr> ExtractIMUDataPieceSafely(double start);

        std::list<RadarTarget::Ptr> ExtractRadarDataPieceSafely(double start, double end);

        std::list<RadarTarget::Ptr> ExtractRadarDataPieceSafely(double start);

        std::list<RadarTarget::Ptr> ExtractRadar1DataPieceSafely(double start, double end);

        std::list<RadarTarget::Ptr> ExtractRadar1DataPieceSafely(double start);

        std::list<LegFrame::Ptr> ExtractLegDataPieceSafely(double start, double end);

        static std::pair<Eigen::Vector3d, Eigen::Matrix3d> AcceMeanVar(const std::list<IMUFrame::Ptr> &data);

        static std::pair<Eigen::Vector3d, Eigen::Matrix3d> GyroMeanVar(const std::list<IMUFrame::Ptr> &data);

        void EraseOldDataPieceSafely(double time);

    protected:
        IMUFrame::Ptr ApplySMA(const IMUFrame::Ptr& frame) {

            Eigen::Vector3d sumLinearAccel;
            sumLinearAccel.x() = 0.0;
            sumLinearAccel.y() = 0.0;
            sumLinearAccel.z() = 0.0;
            size_t count = 0;

            auto it = imuDataSeq.rbegin();
            for (; it != imuDataSeq.rend() && count < 9; ++it) {
                const auto& prevFrame = *it;
                Eigen::Vector3d temp = prevFrame->GetAcce();
                sumLinearAccel.x() += temp.x();
                sumLinearAccel.y() += temp.y();
                sumLinearAccel.z() += temp.z();
                count++;
            }

            Eigen::Vector3d temp = frame->GetAcce();
            sumLinearAccel.x() += temp.x();
            sumLinearAccel.y() += temp.y();
            sumLinearAccel.z() += temp.z();
            
            count++;

            sumLinearAccel.x() = sumLinearAccel.x() / count;
            sumLinearAccel.y() = sumLinearAccel.y() / count;
            sumLinearAccel.z() = sumLinearAccel.z() / count;
            
            return IMUFrame::Create(frame->GetTimestamp(), frame->GetGyro(), sumLinearAccel, frame->GetOrientation());
        }

        template<class IMUMsgType>
        void HandleIMUMessage(const typename IMUMsgType::ConstSharedPtr &msg) {
            auto frame = IMUDataUnpacker::Unpack(msg);
            if (configor->prior.UseGyroBias) {
                Eigen::Vector3d gyro = frame->GetGyro() - gyroBias_;
                frame->SetGyro(gyro);
            }
            const double epoch = GetOrInitializeTimeEpoch(frame->GetTimestamp(), "imu frame");
            if(!imu_is_come){
                imu_is_come = true;
                prev_orient = frame->GetOrientation(); 
                SO3_W2Ref = frame->GetOrientation(); //R_RefW
            }
            else{
                Eigen::Matrix3d R_wo = init_orient.toRotationMatrix();
                
                Eigen::Matrix3d R_wc = frame->GetOrientation().toRotationMatrix();
                Eigen::Matrix3d delta_R = prev_orient.toRotationMatrix().inverse() * R_wc;
                Eigen::Vector3d gyro = frame->GetGyro();
                double dt = frame->GetTimestamp() - previousImuInternalTime_ - epoch;

                Eigen::AngleAxisd rollAngle(gyro(0)*dt, Eigen::Vector3d::UnitX());   
                Eigen::AngleAxisd pitchAngle(gyro(1)*dt, Eigen::Vector3d::UnitY());
                Eigen::AngleAxisd yawAngle(gyro(2)*dt, Eigen::Vector3d::UnitZ()); 

                Eigen::Matrix3d R_no_yaw = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();

                Eigen::Matrix3d R_oc = R_wo * R_no_yaw;
                double pitch = atan2(-delta_R(2, 0), sqrt(delta_R(2, 1) * delta_R(2, 1) + delta_R(2, 2) * delta_R(2, 2))); // theta
                double roll = atan2(delta_R(2, 1), delta_R(2, 2));          
                double yaw = atan2(R_oc(1, 0), R_oc(0, 0));           

                Eigen::AngleAxisd rollAngle_angular(roll, Eigen::Vector3d::UnitX());   
                Eigen::AngleAxisd pitchAngle_angular(pitch, Eigen::Vector3d::UnitY()); 
                Eigen::AngleAxisd yawAngle_angular(yaw, Eigen::Vector3d::UnitZ());    

                Eigen::Matrix3d R_refine = (yawAngle_angular * pitchAngle_angular * rollAngle_angular).toRotationMatrix();

                init_orient = Eigen::Quaternion<double>(R_refine);                 
            }

            Eigen::Quaternion<double> relative_orientation = Eigen::Quaternion<double>(init_orient);

            frame->SetOrientation(relative_orientation);

            frame->SetTimestamp(frame->GetTimestamp() - epoch);
            previousImuInternalTime_ = frame->GetTimestamp();
            // Read under the same mutex used by the foot callback, before
            // taking the IMU buffer mutex; no nested foot/IMU locking.
            const int contactDuration = GetContactDurationSafely();
            double t_front = 0.0, t_back = 0.0;
            std::size_t imu_size = 0;
            {
                LOCK_IMU_DATA_SEQ
                auto smoothedFrame = ApplySMA(frame);
                if(!imuDataSeq.empty()){
                    if(contactDuration>2){
                        imuDataSeq.push_back(smoothedFrame); 
                    }
                    else{
                        auto frame_ = IMUFrame::Create(frame->GetTimestamp(), frame->GetGyro(), imuDataSeq.back()->GetAcce(), frame->GetOrientation());
                        imuDataSeq.push_back(frame_); 
                    }
                }
                else{
                    imuDataSeq.push_back(frame); 
                }
                if (!imuDataSeq.empty()) {
                    t_front = imuDataSeq.front()->GetTimestamp();
                    t_back  = imuDataSeq.back()->GetTimestamp();
                    imu_size = imuDataSeq.size();
                }
            }

            // Fallback init trigger when radar0 is disabled.
            // OrganizeRadarTarAryForInit normally raises DataReadyForInit, but it only fires
            // along the radar0 callback. With radar0 off, signal readiness once enough IMU has
            // been buffered so StateManager::TryPerformInitialization can run on IMU(+leg) only.
            if (!configor->dataStream.UseRadar0 && imu_size >= 2 && (t_back - t_front) >= 1.0) {
                LOCK_GARLILEO_STATUS
                GaRLILEOStatus::DataManager::CurStatus |= GaRLILEOStatus::DataManager::Status::DataReadyForInit;
            }
        }

        template<class FootMsgType>
        void HandleFootMessage(const typename FootMsgType::ConstSharedPtr &msg) {
            auto frame = FootDataUnpacker::Unpack(msg);
            const double epoch = GetOrInitializeTimeEpoch(frame->GetTimestamp(), "foot frame");
            frame->SetTimestamp(frame->GetTimestamp() - epoch);
            {
                LOCK_FOOT_DATA_SEQ
                if(!footDataSeq.empty()){
                    Eigen::Vector4d contact_prev = footDataSeq.back()->GetContactLegs();
                    Eigen::Vector4d contact_curr = frame->GetContactLegs();
                    if(!contact_prev.isApprox(contact_curr)){
                        contact_duration = 0;
                    }
                    if(contact_prev.isApprox(contact_curr)){
                        contact = false;
                        contact_duration ++;
                    }
                }
                footDataSeq.push_back(frame);
            }
            // Release the foot mutex before taking the joint-processing mutex.
            // A matching foot packet can arrive after its joint packet.
            ProcessPendingJointStates();
        }

        // GaRLILEO and Co-RaL bags publish SPOT_ego_Velocity foot states under the
        // spot_msgs/msg/FootStateArray type name, so they arrive serialized and are
        // decoded with the matching layout.
        void HandleEgoVelocityFootMessage(const std::shared_ptr<rclcpp::SerializedMessage> &serialized) {
            auto msg = std::make_shared<spot_msgs::msg::EgoVelocityFootStateArray>();
            bool decoded = true;
            try {
                egoVelocityFootSerialization_.deserialize_message(serialized.get(), msg.get());
            } catch (const std::exception &) {
                decoded = false;
            }
            if (!decoded || msg->states.size() < 4) {
                if (droppedEgoVelocityFootMessages_++ % 1000 == 0) {
                    spdlog::warn("[LegOdom] dropped {} foot message(s) that do not match the "
                                 "SPOT_ego_Velocity layout with 4 feet.", droppedEgoVelocityFootMessages_);
                }
                return;
            }
            HandleFootMessage<spot_msgs::msg::EgoVelocityFootStateArray>(msg);
        }

        void HandleJointStateMessage(const sensor_msgs::msg::JointState::ConstSharedPtr &msg) {
            if (!legOdomProcessor_ || msg->position.size() < 12) return;
            const double raw_t = rclcpp::Time(msg->header.stamp).seconds();
            if (!std::isfinite(raw_t)) return;
            const double epoch = GetOrInitializeTimeEpoch(raw_t, "JointState");
            {
                std::lock_guard<std::mutex> lock(jointStateProcessingMutex_);
                pendingJointStates_.push_back({raw_t - epoch, msg});
            }
            ProcessPendingJointStates();
        }

        template<class RadarMsgType>
        void HandleRadarMessage(const typename RadarMsgType::ConstSharedPtr &msg) {
            auto targetAry = RadarDataUnpacker::Unpack(msg);
            const double epoch = GetOrInitializeTimeEpoch(targetAry->GetTimestamp(), "radar targetAry");
            // aligned the time of targetAry (i.e., radar time) to IMU time
            targetAry->SetTimestamp(
                    targetAry->GetTimestamp() - epoch + configor->dataStream.CalibParam.TIME_OFFSET_RtoB
            );
            for (auto &tar: targetAry->GetTargets()) {
                tar->SetTimestamp(
                        tar->GetTimestamp() - epoch + configor->dataStream.CalibParam.TIME_OFFSET_RtoB
                );
            }
            {
                LOCK_RADAR_DATA_SEQ
                radarDataSeq.insert(
                        radarDataSeq.end(), targetAry->GetTargets().cbegin(), targetAry->GetTargets().cend()
                );
            }
            auto status = GaRLILEOStatus::GetStatusPackSafely();
            if (!GaRLILEOStatus::IsWith(GaRLILEOStatus::StateManager::Status::HasInitialized, status.StateMagr)) {
                OrganizeRadarTarAryForInit(targetAry);
            } else {
                {
                    LOCK_RADAR_INIT_FRAMES
                    radarTarAryForInit.clear();
                }
                if (GaRLILEOStatus::IsWith(GaRLILEOStatus::DataManager::Status::DataReadyForInit, status.DataMagr)) {
                    LOCK_GARLILEO_STATUS
                    GaRLILEOStatus::DataManager::CurStatus ^= GaRLILEOStatus::DataManager::Status::DataReadyForInit;
                }
            }
        }

        template<class RadarMsgType>
        void HandleRadar1Message(const typename RadarMsgType::ConstSharedPtr &msg) {
            auto targetAry = RadarDataUnpacker::Unpack(msg);
            const double epoch = GetOrInitializeTimeEpoch(targetAry->GetTimestamp(), "radar1 targetAry");
            targetAry->SetTimestamp(
                    targetAry->GetTimestamp() - epoch + configor->dataStream.CalibParam.TIME_OFFSET_R1toB
            );
            for (auto &tar: targetAry->GetTargets()) {
                tar->SetTimestamp(
                        tar->GetTimestamp() - epoch + configor->dataStream.CalibParam.TIME_OFFSET_R1toB
                );
            }
            {
                LOCK_RADAR1_DATA_SEQ
                radar1DataSeq.insert(
                        radar1DataSeq.end(), targetAry->GetTargets().cbegin(), targetAry->GetTargets().cend()
                );
            }
        }

        inline void OrganizeRadarTarAryForInit(const RadarTargetArray::Ptr &rawTarAry);
    };

}


#endif 
