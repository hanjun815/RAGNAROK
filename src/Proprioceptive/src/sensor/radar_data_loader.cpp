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

#include "sensor/radar_data_loader.h"
#include "util/enum_cast.hpp"
#include "core/status.h"
#include "pcl_conversions/pcl_conversions.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace garlileo {
    namespace {
        const sensor_msgs::msg::PointField *FindField(const sensor_msgs::msg::PointCloud2 &msg,
                                                      const std::string &name) {
            for (const auto &field: msg.fields) {
                if (field.name == name) { return &field; }
            }
            return nullptr;
        }

        // Reads one point field as double according to its declared datatype.
        double ReadField(const std::uint8_t *point, const sensor_msgs::msg::PointField &field) {
            const std::uint8_t *src = point + field.offset;
            auto read = [src](auto value) {
                std::memcpy(&value, src, sizeof(value));
                return static_cast<double>(value);
            };
            using PF = sensor_msgs::msg::PointField;
            switch (field.datatype) {
                case PF::INT8: return read(std::int8_t{});
                case PF::UINT8: return read(std::uint8_t{});
                case PF::INT16: return read(std::int16_t{});
                case PF::UINT16: return read(std::uint16_t{});
                case PF::INT32: return read(std::int32_t{});
                case PF::UINT32: return read(std::uint32_t{});
                case PF::FLOAT32: return read(float{});
                case PF::FLOAT64: return read(double{});
                default: return std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    RadarDataUnpacker::Ptr RadarDataUnpacker::Create() {
        return std::make_shared<RadarDataUnpacker>();
    }

    RadarTargetArray::Ptr RadarDataUnpacker::Unpack(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
        // Co-RaL radar clouds carry the Doppler velocity as 'vr' instead of 'velocity'.
        const auto *vrField = FindField(*msg, "vr");
        if (FindField(*msg, "velocity") == nullptr && vrField != nullptr) {
            const auto *xField = FindField(*msg, "x");
            const auto *yField = FindField(*msg, "y");
            const auto *zField = FindField(*msg, "z");
            const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
            std::vector<RadarTarget::Ptr> targets;
            if (xField && yField && zField &&
                msg->data.size() >= static_cast<std::size_t>(msg->row_step) * msg->height) {
                targets.reserve(static_cast<std::size_t>(msg->width) * msg->height);
                for (std::uint32_t row = 0; row < msg->height; ++row) {
                    for (std::uint32_t col = 0; col < msg->width; ++col) {
                        const std::uint8_t *point =
                                msg->data.data() + row * msg->row_step + col * msg->point_step;
                        const Eigen::Vector3d target(ReadField(point, *xField), ReadField(point, *yField),
                                                     ReadField(point, *zField));
                        const double radialVel = ReadField(point, *vrField);
                        if (!target.allFinite() || std::isnan(radialVel)) { continue; }
                        targets.push_back(RadarTarget::Create(timestamp, target, radialVel));
                    }
                }
            }
            return RadarTargetArray::Create(timestamp, targets);
        }

        RadarPointCloud radarTargets;
        pcl::fromROSMsg(*msg, radarTargets);

        std::vector<RadarTarget::Ptr> targets;
        targets.reserve(radarTargets.size());

        for (const auto &tar: radarTargets) {
            if (std::isnan(tar.x) || std::isnan(tar.y) || std::isnan(tar.z) ||
                std::isnan(tar.velocity)) { continue; }

            targets.push_back(
                    RadarTarget::Create(rclcpp::Time(msg->header.stamp).seconds(), {tar.x, tar.y, tar.z}, tar.velocity)
            );
        }
        return RadarTargetArray::Create(rclcpp::Time(msg->header.stamp).seconds(), targets);
    }
}