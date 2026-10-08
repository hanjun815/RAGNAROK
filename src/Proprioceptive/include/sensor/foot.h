// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#ifndef FOOT_H
#define FOOT_H

#include "filesystem"
#include "fstream"
#include "memory"
#include "ostream"
#include <array>
#include "ctraj/utils/macros.hpp"
#include "ctraj/utils/eigen_utils.hpp"
#include "utility"
#include "ctraj/utils/sophus_utils.hpp"
#include "cereal/types/array.hpp"

namespace ns_foot {

struct FootFrame {
public:
    using Ptr = std::shared_ptr<FootFrame>;

private:
    double _timestamp;
    Eigen::Vector4d _contact;
    bool _contact_event; // whether a foot contact has just occurred
    int _contact_duration;
    // Per-leg ground contact normals copied from
    // spot_msgs::FootTerrainState::ground_contact_normal_rt_frame (see
    // FootDataUnpacker). Vectors are stored in whatever frame the message
    // declares — for Spot this is normally the body (B) frame, which matches
    // the leg-velocity factor residual frame.
    std::array<Eigen::Vector3d, 4> _contact_normals{{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
    std::array<bool, 4> _contact_normal_valid{{false, false, false, false}};


public:
    explicit FootFrame(double timestamp = INVALID_TIME_STAMP,
                      Eigen::Vector4d contact = Eigen::Vector4d::Ones(),
                      bool contact_event = false,
                      int contact_duration = 0);

    static FootFrame::Ptr Create(double timestamp = INVALID_TIME_STAMP,
                      Eigen::Vector4d contact = Eigen::Vector4d::Ones(),
                      bool contact_event = false,
                      int contact_duration = 0);

    [[nodiscard]] double GetTimestamp() const;

    [[nodiscard]] const Eigen::Vector4d &GetContactLegs() const;

    [[nodiscard]] const bool &GetContactEvent() const;

    [[nodiscard]] const int &GetContactDuration() const;

    // Per-leg contact normals (frame as declared by the spot_msgs producer).
    [[nodiscard]] const std::array<Eigen::Vector3d, 4> &GetContactNormals() const;
    [[nodiscard]] const std::array<bool, 4> &GetContactNormalValid() const;
    void SetContactNormals(const std::array<Eigen::Vector3d, 4> &normals,
                           const std::array<bool, 4> &valid);

    void SetTimestamp(double timestamp);

    friend std::ostream &operator<<(std::ostream &os, const FootFrame &frame);

    static bool SaveFramesToDisk(const std::string &filename,
                                 const std::vector<FootFrame::Ptr> &frames,
                                 int precision = 10);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    template <class Archive>
    void serialize(Archive &ar) {
        ar(cereal::make_nvp("timestamp", _timestamp),
           cereal::make_nvp("ContactLeg", _contact),
           cereal::make_nvp("ContactEvent", _contact_event),
           cereal::make_nvp("ContactDuration", _contact_duration),
           cereal::make_nvp("ContactNormals", _contact_normals),
           cereal::make_nvp("ContactNormalValid", _contact_normal_valid));
    }
};

} 

#endif
