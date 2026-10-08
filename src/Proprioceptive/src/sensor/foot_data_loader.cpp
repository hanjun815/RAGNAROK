// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#include "sensor/foot_data_loader.h"

#include <array>
#include <cmath>

namespace garlileo {
    FootDataUnpacker::Ptr FootDataUnpacker::Create() {
        return std::make_shared<FootDataUnpacker>();
    }

    FootFrame::Ptr FootDataUnpacker::Unpack(const spot_msgs::msg::FootStateArray::ConstSharedPtr& foot_msg) {

        auto contact = Eigen::Vector4d(
                foot_msg->states[0].contact,
                foot_msg->states[1].contact,
                foot_msg->states[2].contact,
                foot_msg->states[3].contact
        );     

        auto frame = FootFrame::Create(rclcpp::Time(foot_msg->header.stamp).seconds(), contact, false, 0);

        // Copy per-leg terrain contact normals when the Spot API populates
        // them (state.has_terrain && terrain.has_ground_contact_normal_rt_frame).
        // If unavailable, degenerate, or non-finite, mark as invalid so the
        // downstream leg-velocity factor can fall back to the previous
        // horizontal [bv_x, bv_y, 0]^T bias embedding.
        std::array<Eigen::Vector3d, 4> normals{{
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
        std::array<bool, 4> valid{{false, false, false, false}};
        for (int i = 0; i < 4; ++i) {
            const auto &st = foot_msg->states[i];
            if (!st.has_terrain) continue;
            if (!st.terrain.has_ground_contact_normal_rt_frame) continue;
            const auto &v = st.terrain.ground_contact_normal_rt_frame;
            Eigen::Vector3d n(v.x, v.y, v.z);
            if (!n.allFinite()) continue;
            const double nn = n.norm();
            if (!std::isfinite(nn) || nn < 1.0e-6) continue;
            normals[i] = n / nn;  // store as a unit vector
            valid[i] = true;
        }
        frame->SetContactNormals(normals, valid);

        return frame;
    }

    FootFrame::Ptr FootDataUnpacker::Unpack(const spot_msgs::msg::EgoVelocityFootStateArray::ConstSharedPtr& foot_msg) {
        auto contact = Eigen::Vector4d(
                foot_msg->states[0].contact,
                foot_msg->states[1].contact,
                foot_msg->states[2].contact,
                foot_msg->states[3].contact
        );
        return FootFrame::Create(rclcpp::Time(foot_msg->header.stamp).seconds(), contact, false, 0);
    }
}