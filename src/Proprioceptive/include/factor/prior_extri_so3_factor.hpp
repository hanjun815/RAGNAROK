// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
//
// Prior factor for an extrinsic rotation (SO3). This is a soft constraint to keep the online
// calibrated extrinsic close to the initial guess, similar to iKalibr's prior factors.

#ifndef GARLILEO_PRIOR_EXTRI_SO3_FACTOR_HPP
#define GARLILEO_PRIOR_EXTRI_SO3_FACTOR_HPP

#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Geometry>
#include <utility>

#include "ctraj/utils/sophus_utils.hpp"

namespace garlileo {

    struct PriorExtriSO3Factor {
    private:
        Eigen::Quaterniond q_prior_;
        double weight_;

    public:
        PriorExtriSO3Factor(const Sophus::SO3d &SO3_prior, double weight)
                : q_prior_(SO3_prior.unit_quaternion()), weight_(weight) {}

        static auto Create(const Sophus::SO3d &SO3_prior, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<PriorExtriSO3Factor>(
                    new PriorExtriSO3Factor(SO3_prior, weight)
            );
        }

        template<class T>
        bool operator()(T const *const *parBlocks, T *sResiduals) const {
            // parameter block: SO3 (Eigen quaternion, coeff order x,y,z,w)
            const Eigen::Map<const Eigen::Quaternion<T>> q_est(parBlocks[0]);
            const Sophus::SO3<T> SO3_est(q_est);

            const Sophus::SO3<T> SO3_prior(q_prior_.cast<T>());
            const Sophus::SO3<T> d = SO3_prior.inverse() * SO3_est;

            Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);
            residuals = T(weight_) * d.log();
            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}  // namespace garlileo

#endif  // GARLILEO_PRIOR_EXTRI_SO3_FACTOR_HPP

