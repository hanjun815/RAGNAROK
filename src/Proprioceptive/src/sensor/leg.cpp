// GaRLILEO — Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
// © 2025 Chiyun Noh, Sangwoo Jung, Hanjun Kim, Yafei Hu, Laura Herlant, Ayoung Kim
// See LICENSE for the full MIT License text.

#include "sensor/leg.h"

namespace ns_leg {

LegFrame::LegFrame(double timestamp, Eigen::Vector3d linearLegVel,
                   Eigen::Matrix3d linearLegVelCov, Eigen::Matrix3d omegaCoupling)
    : _timestamp(timestamp),
      _linearLegVel(std::move(linearLegVel)),
      _linearLegVelCov(std::move(linearLegVelCov)),
      _omegaCoupling(std::move(omegaCoupling)) {}

LegFrame::Ptr LegFrame::Create(double timestamp,
                               const Eigen::Vector3d &linearLegVel,
                               const Eigen::Matrix3d &linearLegVelCov,
                               const Eigen::Matrix3d &omegaCoupling) {
    return std::make_shared<LegFrame>(timestamp, linearLegVel, linearLegVelCov, omegaCoupling);
}

double LegFrame::GetTimestamp() const { return _timestamp; }

const Eigen::Vector3d &LegFrame::GetVel() const { return _linearLegVel; }

const Eigen::Matrix3d &LegFrame::GetVelCov() const { return _linearLegVelCov; }

void LegFrame::SetVel(Eigen::Vector3d &linearLegVel){this->_linearLegVel = linearLegVel;}

void LegFrame::SetVelCov(const Eigen::Matrix3d &linearLegVelCov) { this->_linearLegVelCov = linearLegVelCov; }

const Eigen::Matrix3d &LegFrame::GetOmegaCoupling() const { return _omegaCoupling; }

void LegFrame::SetOmegaCoupling(const Eigen::Matrix3d &omegaCoupling) { this->_omegaCoupling = omegaCoupling; }

const LegFrame::BiasTangent &LegFrame::GetBiasTangent() const { return _biasTangent; }

bool LegFrame::IsBiasTangentValid() const { return _biasTangentValid; }

void LegFrame::SetBiasTangent(const BiasTangent &biasTangent, bool valid) {
    _biasTangent = biasTangent;
    _biasTangentValid = valid;
}

std::ostream &operator<<(std::ostream &os, const LegFrame &frame) {
    os << "timestamp: " << frame._timestamp
       << ", linear vel: " << frame._linearLegVel.transpose()
       << ", cov trace: " << frame._linearLegVelCov.trace();
    return os;
}

void LegFrame::SetTimestamp(double timestamp) { _timestamp = timestamp; }

bool LegFrame::SaveFramesToDisk(const std::string &filename,
                                const std::vector<LegFrame::Ptr> &frames,
                                int precision) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(precision);
    cereal::JSONOutputArchive ar(file);
    ar(cereal::make_nvp("leg_frames", frames));
    return true;
}

}
