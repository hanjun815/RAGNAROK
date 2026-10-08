// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
//
// Prior factor for an extrinsic position (translation). This is a soft constraint to keep the
// online calibrated extrinsic close to the initial guess, similar to iKalibr's prior factors.

#ifndef GARLILEO_PRIOR_EXTRI_POS_FACTOR_HPP
#define GARLILEO_PRIOR_EXTRI_POS_FACTOR_HPP

#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Core>
#include <utility>

namespace garlileo {

    struct PriorExtriPOSFactor {
    private:
        Eigen::Vector3d p_prior_;
        double weight_;

    public:
        PriorExtriPOSFactor(const Eigen::Vector3d &p_prior, double weight)
                : p_prior_(p_prior), weight_(weight) {}

        static auto Create(const Eigen::Vector3d &p_prior, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<PriorExtriPOSFactor>(
                    new PriorExtriPOSFactor(p_prior, weight)
            );
        }

        template<class T>
        bool operator()(T const *const *parBlocks, T *sResiduals) const {
            // parameter block: position (3)
            const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_est(parBlocks[0]);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);
            residuals = T(weight_) * (p_est - p_prior_.cast<T>());
            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}  // namespace garlileo

#endif  // GARLILEO_PRIOR_EXTRI_POS_FACTOR_HPP

