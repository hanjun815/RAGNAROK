/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file SubmapIcpError.hpp
 * @brief Header file for the Submap ICP error class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CERES_SUBMAPICPERROR_HPP_
#define INCLUDE_OKVIS_CERES_SUBMAPICPERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>

#include <okvis/mapTypedefs.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <se/supereight.hpp>
#include <Eigen/StdVector>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Relative error between two poses.
class SubmapIcpError : public ::ceres::SizedCostFunction<
        1 /* number of residuals */,
        7, /* size of first parameter T_WS_a*/
        7 /* size of second parameter T_WS_b*/>, public ErrorInterface {

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

    /// \brief The base class type.
    typedef ::ceres::SizedCostFunction<1, 7, 7> base_t;

    /// \brief Number of residuals (1, scalar product).
    static const int kNumResiduals = 1;

    /// \brief The information type (only a double).
    typedef double information_t;

    /// \brief The covariance type (same as information).
    typedef double covariance_t;

    /// \brief Default constructor.
    SubmapIcpError() = delete;

    /// \brief Construct with measurement and information matrix
    /// @param[in] submap reference submap.
    /// @param[in] point_b point in frame {B} that will be aligned
    /// @param pointCloud The point cloud that is aligned to the reference submap.
    SubmapIcpError(
            /* (1) Submap reference
             * (2) Point cloud to be aligned to reference submap
             * */
            const SupereightMapType& submap,
            const Eigen::Vector3d& point_b,
            const double sigma_measurement);

    /// \brief Trivial destructor.
    virtual ~SubmapIcpError() override = default;

    // setters
    /// \brief Set the submap.
    /// @param[in] submap The submap.
    //void setSubmap(const SupereightMapType& submap);

    /// \brief Set the point.
    /// @param[in] point_b The point.
    void setPb(const Eigen::Vector3d& point_b);

    // getters
    /// \brief Get the Submap.
    /// \return The map.
    const SupereightMapType& submap() const {
      //return *submapPtr_;
      return submapPtr_;
    }

    // getters
    /// \brief Get the point.
    /// \return The point
    const Eigen::Vector3d& p_b() const {
      return p_b_;
    }

    // error term and Jacobian implementation
    /**
      * @brief This evaluates the error term and additionally computes the Jacobians.
      * @param parameters Pointer to the parameters (see ceres)
      * @param residuals Pointer to the residual vector (see ceres)
      * @param jacobians Pointer to the Jacobians (see ceres)
      * @return success of th evaluation.
      */
    virtual bool Evaluate(double const* const * parameters, double* residuals,
                          double** jacobians) const override final;

    /**
     * @brief This evaluates the error term and additionally computes
     *        the Jacobians in the minimal internal representation.
     * @param parameters Pointer to the parameters (see ceres)
     * @param residuals Pointer to the residual vector (see ceres)
     * @param jacobians Pointer to the Jacobians (see ceres)
     * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
     * @return Success of the evaluation.
     */
    bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                      double* residuals, double** jacobians,
                                      double** jacobiansMinimal) const override final;

    // sizes
    /// \brief Residual dimension.
    int residualDim() const override final {
      return kNumResiduals;
    }

    /// \brief Number of parameter blocks.
    int parameterBlocks() const override final {
      return int(parameter_block_sizes().size());
    }

    /// \brief Dimension of an individual parameter block.
    int parameterBlockDim(int parameterBlockId) const override final {
      return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
    }

    /// @brief Residual block type as string
    virtual std::string typeInfo() const override final {
      return "SubmapIcpError";
    }

    std::pair<bool, bool> nonZeroFieldAndGradient(const double *const *parameters, float& fieldVal, float& gradNorm);

protected:


    const SupereightMapType& submapPtr_;
    //std::shared_ptr<SupereightMapType> submapPtr_; ///< Reference Submap
    Eigen::Vector3d p_b_; ///< Point in frame {B} to be aligned to submap in frame {A}

    covariance_t sigma_measurement_; ///< The Lidar Measurement Covariance
    mutable covariance_t sigma_map_; ///< The map covariance, evaluated every time, will be overwitten
    mutable information_t squareRootInformation_; ///< total variance changes with sigma_map_ => has to be modified

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_SUBMAPICPERROR_HPP_ */
