// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
//
// Prior factor for an extrinsic time offset (1-D). Used to anchor the online
// temporal calibration parameter (e.g. TIME_OFFSET_RtoB) near a user-provided
// initial value, mirroring iKalibr's PriorTimeOffsetFactor.

#ifndef GARLILEO_PRIOR_TIME_OFFSET_FACTOR_HPP
#define GARLILEO_PRIOR_TIME_OFFSET_FACTOR_HPP

#include <ceres/dynamic_autodiff_cost_function.h>

namespace garlileo {

    struct PriorTimeOffsetFactor {
    private:
        double TO_prior_;
        double weight_;

    public:
        PriorTimeOffsetFactor(double TO_prior, double weight)
                : TO_prior_(TO_prior), weight_(weight) {}

        static auto Create(double TO_prior, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<PriorTimeOffsetFactor>(
                    new PriorTimeOffsetFactor(TO_prior, weight)
            );
        }

        template<class T>
        bool operator()(T const *const *parBlocks, T *sResiduals) const {
            const T TO_est = parBlocks[0][0];
            sResiduals[0] = T(weight_) * (TO_est - T(TO_prior_));
            return true;
        }
    };

}  // namespace garlileo

#endif  // GARLILEO_PRIOR_TIME_OFFSET_FACTOR_HPP
