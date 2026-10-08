// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#include "sensor/foot.h"

namespace ns_foot {

FootFrame::FootFrame(double timestamp, Eigen::Vector4d contact, bool contact_event, int contact_duration)
    : _timestamp(timestamp),
      _contact(std::move(contact)),
      _contact_event(contact_event),
      _contact_duration(contact_duration)
       {}

FootFrame::Ptr FootFrame::Create(double timestamp, Eigen::Vector4d contact, bool contact_event, int contact_duration) {
    return std::make_shared<FootFrame>(timestamp, contact, contact_event, contact_duration);
}

double FootFrame::GetTimestamp() const { return _timestamp; }

const Eigen::Vector4d &FootFrame::GetContactLegs() const { return _contact; }

const bool &FootFrame::GetContactEvent() const {return _contact_event;}

const int &FootFrame::GetContactDuration() const {return _contact_duration;}

const std::array<Eigen::Vector3d, 4> &FootFrame::GetContactNormals() const { return _contact_normals; }

const std::array<bool, 4> &FootFrame::GetContactNormalValid() const { return _contact_normal_valid; }

void FootFrame::SetContactNormals(const std::array<Eigen::Vector3d, 4> &normals,
                                  const std::array<bool, 4> &valid) {
    _contact_normals = normals;
    _contact_normal_valid = valid;
}

std::ostream &operator<<(std::ostream &os, const FootFrame &frame) {
    os << "timestamp: " << frame._timestamp << ", contact: " << frame._contact.transpose();
    return os;
}

void FootFrame::SetTimestamp(double timestamp) { _timestamp = timestamp; }

bool FootFrame::SaveFramesToDisk(const std::string &filename,
                                const std::vector<FootFrame::Ptr> &frames,
                                int precision) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(precision);
    cereal::JSONOutputArchive ar(file);
    ar(cereal::make_nvp("leg_frames", frames));
    return true;
}

}
