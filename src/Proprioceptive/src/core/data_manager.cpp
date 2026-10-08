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

#include "core/data_manager.h"
#include <algorithm>
#include <cctype>
#include <limits>

namespace garlileo {
    // -------------------
    // static member field
    // -------------------
    std::mutex DataManager::IMUDataSeqMutex = {};
    std::mutex DataManager::RadarDataSeqMutex = {};
    std::mutex DataManager::Radar1DataSeqMutex = {};
    std::mutex DataManager::LegDataSeqMutex = {};
    std::mutex DataManager::FootDataSeqMutex ={};
    std::mutex DataManager::RadarInitFramesMutex = {};

    DataManager::DataManager(const rclcpp::Node::SharedPtr &handler, const garlileo::Configor::Ptr &configor) 
            : handler(handler), configor(configor), GARLILEO_TIME_EPOCH(std::optional<double>()) {
        spdlog::info("'DataManager' has been booted, thread id: {}.", GARLILEO_TO_STR(std::this_thread::get_id()));
        if (configor->prior.UseGyroBias) {
            gyroBias_ = ToVector3d(configor->prior.GyroBiasInit, "GyroBiasInit");
        }
        // -------------------------------------------------------
        // create imu message subscriber based on the message type
        // -------------------------------------------------------

        auto cg_imu   = handler->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        auto cg_radar = handler->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        // Leg callback groups are only created when UseLeg=true (see below).

        auto make_opt = [](rclcpp::CallbackGroup::SharedPtr cg){
            rclcpp::SubscriptionOptions opt;
            opt.callback_group = cg;
            return opt;
        };



        IMUMsgType imuMsgType;
        try {
            imuMsgType = EnumCast::stringToEnum<IMUMsgType>(configor->dataStream.IMUMsgType); 
        } catch (...) {
            throw Status(
                    Status::Flag::WARNING,
                    fmt::format(
                            "Unsupported IMU Type: '{}'. ",
                            configor->dataStream.IMUMsgType
                    )
            );
        }
        /**
         * create subscriber of imu data
         * @topic configor->dataStream.IMUTopic
         * @queue_size configor->preference.IMUMsgQueueSize
         * @callback HandleIMUMessage(...)
         */
           
        switch (imuMsgType) {
            case IMUMsgType::SENSOR_IMU:
                imuSuber = handler->create_subscription<sensor_msgs::msg::Imu>(
                configor->dataStream.IMUTopic,
                rclcpp::QoS(configor->preference.IMUMsgQueueSize),
                std::bind(&DataManager::HandleIMUMessage<sensor_msgs::msg::Imu>, this, _1),
                make_opt(cg_imu));    
                break;
        }
        // ---------------------------------------------------------
        // create radar message subscriber based on the message type
        // ---------------------------------------------------------
        RadarMsgType radarMsgType;
        try {
            radarMsgType = EnumCast::stringToEnum<RadarMsgType>(configor->dataStream.RadarMsgType);
        } catch (...) {
            throw Status(
                    Status::Flag::WARNING,
                    fmt::format(
                            configor->dataStream.RadarMsgType
                    )
            );
        }

        // radar0 subscriber (only when UseRadar0 is true)
        if (configor->dataStream.UseRadar0) {
            switch (radarMsgType) {
                case RadarMsgType::GARLILEO:
                    radarSuber = handler->create_subscription<sensor_msgs::msg::PointCloud2>(
                    configor->dataStream.RadarTopic,
                    rclcpp::QoS(configor->preference.RadarMsgQueueSize),
                    std::bind(&DataManager::HandleRadarMessage<sensor_msgs::msg::PointCloud2>, this, _1),
                    make_opt(cg_radar));
                    break;
            }
            spdlog::info("[Radar] Subscribing to radar0: '{}'", configor->dataStream.RadarTopic);
        } else {
            spdlog::warn("[Radar] radar0 disabled by yaml (UseRadar0=false)");
        }

        // radar1 subscriber (only when UseRadar1 is true)
        if (configor->dataStream.UseRadar1) {
            auto cg_radar1 = handler->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            radar1Suber = handler->create_subscription<sensor_msgs::msg::PointCloud2>(
                configor->dataStream.RadarTopic1,
                rclcpp::QoS(configor->preference.RadarMsgQueueSize),
                std::bind(&DataManager::HandleRadar1Message<sensor_msgs::msg::PointCloud2>, this, _1),
                make_opt(cg_radar));
            spdlog::info("[Radar] Subscribing to radar1: '{}'", configor->dataStream.RadarTopic1);
        }

        // Compose a runtime mode string for diagnostics so the log clearly shows
        // which sensor combination is active under the current yaml.
        const bool useRadar0Cfg = configor->dataStream.UseRadar0;
        const bool useRadar1Cfg = configor->dataStream.UseRadar1;
        const bool useLegCfg    = configor->dataStream.UseLeg;
        if (!useRadar0Cfg && !useRadar1Cfg && useLegCfg) {
            spdlog::warn(
                "[Mode] both radars disabled - running in IMU+Leg only mode. "
                "Initialization will be triggered by IMU/Leg accumulation.");
        } else if (useRadar0Cfg || useRadar1Cfg) {
            if (!useLegCfg) {
                spdlog::warn(
                    "[Mode] leg disabled by yaml (UseLeg=false) - running in IMU+Radar only mode.");
            }
        } else {
            // No radar AND no leg.
            spdlog::warn(
                "[Mode] no radar / no leg enabled (UseRadar0=false, UseRadar1=false, UseLeg=false). "
                "Running in IMU-only mode (mostly useful for sanity checks).");
        }

        // Leg subscribers (only when UseLeg is true).
        if (configor->dataStream.UseLeg) {
            // Foot message layout, set by the dataset launch file:
            //   ragnarok          - spot_msgs FootStateArray with terrain data (RAGNAROK)
            //   spot_ego_velocity - SPOT_ego_Velocity layout (GaRLILEO, Co-RaL)
            const std::string foot_msg_format =
                handler->declare_parameter<std::string>("foot_msg_format", "ragnarok");
            auto cg_foot = handler->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            if (foot_msg_format == "ragnarok") {
                foot_Suber = handler->create_subscription<spot_msgs::msg::FootStateArray>(
                    configor->dataStream.FootTopic,
                    rclcpp::QoS(configor->preference.FootMsgQueueSize),
                    std::bind(&DataManager::HandleFootMessage<spot_msgs::msg::FootStateArray>, this, _1),
                    make_opt(cg_foot));
            } else if (foot_msg_format == "spot_ego_velocity") {
                // Same type name as the recording; take the payload serialized and
                // decode it with the SPOT_ego_Velocity layout.
                foot_Suber = handler->create_subscription<spot_msgs::msg::FootStateArray>(
                    configor->dataStream.FootTopic,
                    rclcpp::QoS(configor->preference.FootMsgQueueSize),
                    [this](std::shared_ptr<rclcpp::SerializedMessage> serialized) {
                        HandleEgoVelocityFootMessage(serialized);
                    },
                    make_opt(cg_foot));
            } else {
                throw Status(
                        Status::Flag::ERROR,
                        fmt::format(
                                "Unsupported foot_msg_format: '{}'. Use 'ragnarok' or 'spot_ego_velocity'.",
                                foot_msg_format
                        )
                );
            }

            // Subscribe to /joint_states and compute leg odometry internally
            LegOdometryProcessor::Config odom_cfg;
            odom_cfg.encoder_pos_sigma    = configor->prior.LegEncoderPosSigma;
            odom_cfg.encoder_vel_sigma    = configor->prior.LegEncoderVelSigma;
            odom_cfg.inter_leg_cov_scale  = configor->prior.LegInterLegCovarianceScale;
            odom_cfg.force_disc_scale     = configor->prior.LegForceDiscontinuityScale;
            odom_cfg.stance_age_scale     = configor->prior.LegStanceAgeCovarianceScale;
            odom_cfg.stance_age_floor_sec = configor->prior.LegStanceAgeFloorSec;
            odom_cfg.cov_floor            = configor->prior.LegVelocityCovarianceFloor;
            odom_cfg.rolling_contact_enable = configor->prior.LegRollingContactEnable;
            odom_cfg.rolling_contact_foot_radius = configor->prior.LegRollingContactFootRadius;
            odom_cfg.rolling_contact_body_normal = (foot_msg_format == "spot_ego_velocity");
            odom_cfg.SO3_BtoL = configor->dataStream.CalibParam.SO3_BtoL.matrix();
            legOdomProcessor_ = LegOdometryProcessor::Create(odom_cfg);

            auto cg_joint = handler->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            jointStateSuber = handler->create_subscription<sensor_msgs::msg::JointState>(
                configor->dataStream.JointStateTopic,
                rclcpp::QoS(configor->preference.JointStateMsgQueueSize),
                std::bind(&DataManager::HandleJointStateMessage, this, _1),
                make_opt(cg_joint));
            spdlog::info("[LegOdom] Subscribing to '{}' + '{}' (foot_msg_format: {})",
                         configor->dataStream.JointStateTopic, configor->dataStream.FootTopic,
                         foot_msg_format);
        } else {
            spdlog::warn("[LegOdom] leg disabled by yaml (UseLeg=false): not subscribing to "
                         "foot/joint_state topics.");
        }
    }

    DataManager::Ptr DataManager::Create(const rclcpp::Node::SharedPtr &handler, const Configor::Ptr &configor) {
        return std::make_shared<DataManager>(handler, configor);
    }

    void DataManager::ProcessPendingJointStates() {
        if (!legOdomProcessor_) return;
        std::lock_guard<std::mutex> processingLock(jointStateProcessingMutex_);
        while (!pendingJointStates_.empty()) {
            const double timestamp = pendingJointStates_.front().timestamp;
            FootFrame::Ptr foot;
            {
                LOCK_FOOT_DATA_SEQ
                // Topic-local order is the existing measurement-buffer contract.
                // Wait asynchronously until foot input covers this joint stamp.
                if (footDataSeq.empty() || footDataSeq.back()->GetTimestamp() < timestamp) break;
                const auto match = std::find_if(footDataSeq.rbegin(), footDataSeq.rend(),
                        [timestamp](const FootFrame::Ptr& candidate) {
                            return candidate->GetTimestamp() <= timestamp;
                        });
                if (match != footDataSeq.rend()) foot = *match;
            }
            // If joint data predates all available foot data, preserve the
            // kinematic update with zero known contacts instead of future ones.
            const auto pending = std::move(pendingJointStates_.front());
            pendingJointStates_.pop_front();
            ProcessJointStateWithContact(pending.message, foot, timestamp);
        }
        const std::size_t capacity = std::max<std::size_t>(
                1, configor->preference.JointStateMsgQueueSize);
        while (pendingJointStates_.size() > capacity) {
            pendingJointStates_.pop_front();
            ++discardedPendingJointStates_;
            if (discardedPendingJointStates_ == 1 || discardedPendingJointStates_ % 1000 == 0) {
                spdlog::warn("[LegOdom] Foot input has not covered queued joint timestamps; "
                             "discarded {} oldest pending joint messages (capacity {}).",
                             discardedPendingJointStates_, capacity);
            }
        }
    }

    void DataManager::ProcessJointStateWithContact(
            const sensor_msgs::msg::JointState::ConstSharedPtr& msg,
            const FootFrame::Ptr& foot, double timestamp) {
        // Called only while jointStateProcessingMutex_ is held. Match source
        // headers first, then apply the existing leg-to-body time calibration.
        const double t = timestamp + configor->dataStream.CalibParam.TIME_OFFSET_BtoL;
        Eigen::Vector4d contact_vec = Eigen::Vector4d::Zero();
        std::array<Eigen::Vector3d, 4> contact_normals{{
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
        std::array<bool, 4> normal_valid{{false, false, false, false}};
        if (foot) {
            const Eigen::Vector4d& raw = foot->GetContactLegs();
            for (int k = 0; k < 4; ++k) {
                contact_vec(k) = (static_cast<int>(raw(k)) == 1) ? 1.0 : 0.0;
            }
            contact_normals = foot->GetContactNormals();
            normal_valid = foot->GetContactNormalValid();
        }
        std::vector<double> pos(msg->position.begin(), msg->position.end());
        std::vector<double> vel(msg->velocity.begin(), msg->velocity.end());
        std::vector<double> eff(msg->effort.begin(), msg->effort.end());
        auto result = legOdomProcessor_->process(pos, vel, eff, contact_vec, t,
                                                 contact_normals, normal_valid);
        if (result.has_value()) {
            LOCK_LEG_DATA_SEQ
            legDataSeq.push_back(result.value());
        }
    }

    void DataManager::ShowDataStatus() const {
        std::size_t s = 0;
        double st = 0.0, et = 0.0;
        {
            // lock imu data and obtain its size, start time, and end time quickly
            LOCK_IMU_DATA_SEQ
            if (!imuDataSeq.empty()) {
                s = imuDataSeq.size();
                st = imuDataSeq.front()->GetTimestamp();
                et = imuDataSeq.back()->GetTimestamp();
            }
        }
        spdlog::info("imu data size: {:02}, time span from '{:.6f}' to '{:.6f}'", s, st, et);

        s = 0, st = 0.0, et = 0.0;
        {
            // lock radar data and obtain its size, start time, and end time quickly
            LOCK_RADAR_DATA_SEQ
            if (!radarDataSeq.empty()) {
                s = radarDataSeq.size();
                st = radarDataSeq.front()->GetTimestamp();
                et = radarDataSeq.back()->GetTimestamp();
            }
        }
        spdlog::info("radar data size: {:02}, time span from '{:.6f}' to '{:.6f}'", s, st, et);
    }

    void DataManager::OrganizeRadarTarAryForInit(const RadarTargetArray::Ptr &rawTarAry) {
        // if this system has not been initialization, construct radar rawTarAry arrays
        static std::vector<RadarTarget::Ptr> tarAry = {};

        tarAry.insert(tarAry.end(), rawTarAry->GetTargets().cbegin(), rawTarAry->GetTargets().cend());

        static double scanHeadTime = tarAry.front()->GetTimestamp();

        if (tarAry.back()->GetTimestamp() - scanHeadTime < 0.1) { return; }

        // if time span is larger than 0.1 (s), try to organize these targets as a radar frame
        if (tarAry.size() < 3) {
            // this radar frame is invalid, clear current status
            tarAry.clear();

            LOCK_RADAR_INIT_FRAMES
            radarTarAryForInit.clear();
            return;
        }

        // this radar frame is valid, compute average time
        double avgTime = 0.0;
        for (const auto &item: tarAry) { avgTime += item->GetTimestamp(); }
        avgTime /= static_cast<double>(tarAry.size());

        {
            LOCK_RADAR_INIT_FRAMES
            radarTarAryForInit.push_back(RadarTargetArray::Create(avgTime, tarAry));

            if (radarTarAryForInit.size() > 10) {
                // keep data that lasting for 1.0 (s), i.e., ten frames, each lasting 0.1 (s)
                radarTarAryForInit.pop_front();

                LOCK_GARLILEO_STATUS
                GaRLILEOStatus::DataManager::CurStatus |= GaRLILEOStatus::DataManager::Status::DataReadyForInit;
            }
        }

        // clear current status
        scanHeadTime = tarAry.back()->GetTimestamp();
        tarAry.clear();
    }

    std::list<RadarTargetArray::Ptr> DataManager::GetRadarTarAryForInitSafely() const {
        LOCK_RADAR_INIT_FRAMES
        return radarTarAryForInit;
    }

    std::list<IMUFrame::Ptr> DataManager::ExtractIMUDataPieceSafely(double start, double end) {
        LOCK_IMU_DATA_SEQ
        std::list<IMUFrame::Ptr> dataSeq;
        auto sIter = std::find_if(imuDataSeq.rbegin(), imuDataSeq.rend(), [start](const IMUFrame::Ptr &frame) {
            return frame->GetTimestamp() < start;
        }).base();
        auto eIter = std::find_if(imuDataSeq.rbegin(), imuDataSeq.rend(), [end](const IMUFrame::Ptr &frame) {
            return frame->GetTimestamp() < end;
        }).base();
        std::copy(sIter, eIter, std::back_inserter(dataSeq));
        return dataSeq;
    }

    std::list<IMUFrame::Ptr> DataManager::ExtractIMUDataPieceSafely(double start) {
        LOCK_IMU_DATA_SEQ
        std::list<IMUFrame::Ptr> dataSeq;
        auto sIter = std::find_if(imuDataSeq.rbegin(), imuDataSeq.rend(), [start](const IMUFrame::Ptr &frame) {
            return frame->GetTimestamp() < start;
        }).base();
        std::copy(sIter, imuDataSeq.end(), std::back_inserter(dataSeq));
        return dataSeq;
    }

    std::list<RadarTarget::Ptr> DataManager::ExtractRadarDataPieceSafely(double start, double end) {
        LOCK_RADAR_DATA_SEQ
        std::list<RadarTarget::Ptr> tarSeq;
        auto sIter = std::find_if(radarDataSeq.rbegin(), radarDataSeq.rend(), [start](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < start;
        }).base();
        auto eIter = std::find_if(radarDataSeq.rbegin(), radarDataSeq.rend(), [end](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < end;
        }).base();
        std::copy(sIter, eIter, std::back_inserter(tarSeq));
        return tarSeq;
    }

    std::list<RadarTarget::Ptr> DataManager::ExtractRadarDataPieceSafely(double start) {
        LOCK_RADAR_DATA_SEQ
        std::list<RadarTarget::Ptr> tarSeq;
        
        auto sIter = std::find_if(radarDataSeq.rbegin(), radarDataSeq.rend(), [start](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < start;
        }).base();
        std::copy(sIter, radarDataSeq.end(), std::back_inserter(tarSeq));
        return tarSeq;
    }

    std::list<RadarTarget::Ptr> DataManager::ExtractRadar1DataPieceSafely(double start, double end) {
        LOCK_RADAR1_DATA_SEQ
        std::list<RadarTarget::Ptr> tarSeq;
        auto sIter = std::find_if(radar1DataSeq.rbegin(), radar1DataSeq.rend(), [start](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < start;
        }).base();
        auto eIter = std::find_if(radar1DataSeq.rbegin(), radar1DataSeq.rend(), [end](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < end;
        }).base();
        std::copy(sIter, eIter, std::back_inserter(tarSeq));
        return tarSeq;
    }

    std::list<RadarTarget::Ptr> DataManager::ExtractRadar1DataPieceSafely(double start) {
        LOCK_RADAR1_DATA_SEQ
        std::list<RadarTarget::Ptr> tarSeq;
        auto sIter = std::find_if(radar1DataSeq.rbegin(), radar1DataSeq.rend(), [start](const RadarTarget::Ptr &tar) {
            return tar->GetTimestamp() < start;
        }).base();
        std::copy(sIter, radar1DataSeq.end(), std::back_inserter(tarSeq));
        return tarSeq;
    }

    std::list<LegFrame::Ptr> DataManager::ExtractLegDataPieceSafely(double start, double end) {
        LOCK_LEG_DATA_SEQ
        std::list<LegFrame::Ptr> dataSeq; 
        auto sIter = std::find_if(legDataSeq.rbegin(), legDataSeq.rend(), [start](const LegFrame::Ptr &frame) {
            return frame->GetTimestamp() < start;
        }).base();
        auto eIter = std::find_if(legDataSeq.rbegin(), legDataSeq.rend(), [end](const LegFrame::Ptr &frame) {
            return frame->GetTimestamp() < end;
        }).base();
        std::copy(sIter, eIter, std::back_inserter(dataSeq));
        return dataSeq;
    }

    double DataManager::GetEndTimeSafely() const {
        double end_time = std::numeric_limits<double>::infinity();
        {
            LOCK_IMU_DATA_SEQ
            if (!imuDataSeq.empty()) {
                end_time = std::min(end_time, imuDataSeq.back()->GetTimestamp());
            }
        }
        if (configor->dataStream.UseRadar0) {
            LOCK_RADAR_DATA_SEQ
            if (!radarDataSeq.empty()) {
                end_time = std::min(end_time, radarDataSeq.back()->GetTimestamp());
            }
        }
        if (configor->dataStream.UseRadar1) {
            LOCK_RADAR1_DATA_SEQ
            if (!radar1DataSeq.empty()) {
                end_time = std::min(end_time, radar1DataSeq.back()->GetTimestamp());
            }
        }
        return end_time;
    }

    double DataManager::GetOrInitializeTimeEpoch(double timestamp, const char* source) {
        std::lock_guard<std::mutex> lock(timeEpochMutex_);
        if (!GARLILEO_TIME_EPOCH.has_value()) {
            GARLILEO_TIME_EPOCH = timestamp;
            spdlog::info("time epoch of GaRLILEO initialized by {}: {:.6f}",
                         source, *GARLILEO_TIME_EPOCH);
        }
        return *GARLILEO_TIME_EPOCH;
    }

    std::optional<double> DataManager::GetGaRLILEOTimeEpoch() const {
        std::lock_guard<std::mutex> lock(timeEpochMutex_);
        return GARLILEO_TIME_EPOCH;
    }

    int DataManager::GetContactDurationSafely() const {
        LOCK_FOOT_DATA_SEQ
        return contact_duration;
    }

    bool DataManager::TrySetOkvisInitTimeFromCameraStamp(double t0_internal_sec, double imuTemporalOverlap) {
        // Lock both the stored t0 and the IMU buffer.
        std::scoped_lock lock(okvisInitMutex_, IMUDataSeqMutex);
        if (okvisInitTimeT0_.has_value()) {
            return false;
        }
        if (imuDataSeq.empty()) {
            return false;
        }
        const double t_front = imuDataSeq.front()->GetTimestamp();
        const double t_back = imuDataSeq.back()->GetTimestamp();

        // OKVIS2-X requires IMU coverage around the first frame:
        //   imu_front < t0 - overlap  (otherwise drop frame)
        //   imu_back  >= t0 + overlap (otherwise wait)
        if (t_front >= t0_internal_sec - imuTemporalOverlap) {
            return false;
        }
        if (t_back < t0_internal_sec + imuTemporalOverlap) {
            return false;
        }

        okvisInitTimeT0_ = t0_internal_sec;
        spdlog::info(
                "[OKVIS-like] init t0 set from camera: t0={:.6f}s (internal), imu_front={:.6f}, imu_back={:.6f}, overlap={:.3f}s",
                *okvisInitTimeT0_, t_front, t_back, imuTemporalOverlap
        );
        return true;
    }

    std::optional<double> DataManager::GetOkvisInitTimeSafely() const {
        std::lock_guard<std::mutex> lock(okvisInitMutex_);
        return okvisInitTimeT0_;
    }

    double DataManager::GetIMUEndTimeSafely() const {
        LOCK_IMU_DATA_SEQ
        return imuDataSeq.back()->GetTimestamp();
    }

    double DataManager::GetRadarEndTimeSafely() const {
        LOCK_RADAR_DATA_SEQ
        if (radarDataSeq.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return radarDataSeq.back()->GetTimestamp();
    }

    std::pair<Eigen::Vector3d, Eigen::Matrix3d> DataManager::AcceMeanVar(const std::list<IMUFrame::Ptr> &data) {
        if (data.empty()) { return {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Zero()}; }

        Eigen::MatrixXd matrix(data.size(), 3);
        int i = 0;
        for (const auto &v: data) { matrix.row(i++) = v->GetAcce(); }

        Eigen::Vector3d mean = matrix.colwise().mean();
        Eigen::Matrix3d var = ((matrix.rowwise() - matrix.colwise().mean()).transpose() *
                               (matrix.rowwise() - matrix.colwise().mean())) / static_cast<double>(data.size() - 1);

        return {mean, var};
    }

    std::pair<Eigen::Vector3d, Eigen::Matrix3d> DataManager::GyroMeanVar(const std::list<IMUFrame::Ptr> &data) {
        if (data.empty()) { return {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Zero()}; }

        Eigen::MatrixXd matrix(data.size(), 3);
        int i = 0;
        for (const auto &v: data) { matrix.row(i++) = v->GetGyro(); }

        Eigen::Vector3d mean = matrix.colwise().mean();
        Eigen::Matrix3d var = ((matrix.rowwise() - matrix.colwise().mean()).transpose() *
                               (matrix.rowwise() - matrix.colwise().mean())) / static_cast<double>(data.size() - 1);

        return {mean, var};
    }

    void DataManager::EraseOldDataPieceSafely(double time) {
        {
            LOCK_IMU_DATA_SEQ
            auto iter = std::find_if(imuDataSeq.rbegin(), imuDataSeq.rend(), [time](const IMUFrame::Ptr &frame) {
                return frame->GetTimestamp() < time;
            }).base();
            imuDataSeq.erase(imuDataSeq.begin(), iter);
        }
        {
            LOCK_RADAR_DATA_SEQ
            auto iter = std::find_if(radarDataSeq.rbegin(), radarDataSeq.rend(), [time](const RadarTarget::Ptr &tar) {
                return tar->GetTimestamp() < time;
            }).base();
            radarDataSeq.erase(radarDataSeq.begin(), iter);
        }
        {
            LOCK_RADAR1_DATA_SEQ
            auto iter = std::find_if(radar1DataSeq.rbegin(), radar1DataSeq.rend(), [time](const RadarTarget::Ptr &tar) {
                return tar->GetTimestamp() < time;
            }).base();
            radar1DataSeq.erase(radar1DataSeq.begin(), iter);
        }
        {
            LOCK_LEG_DATA_SEQ
            auto iter = std::find_if(legDataSeq.rbegin(), legDataSeq.rend(), [time](const LegFrame::Ptr &frame) {
                return frame->GetTimestamp() < time;
            }).base();
            legDataSeq.erase(legDataSeq.begin(), iter);
        }
    }

    const std::list<IMUFrame::Ptr> &DataManager::GetIMUDataSeq() const {
        return imuDataSeq;
    }

    const std::list<RadarTarget::Ptr> &DataManager::GetRadarDataSeq() const {
        return radarDataSeq;
    }

}

