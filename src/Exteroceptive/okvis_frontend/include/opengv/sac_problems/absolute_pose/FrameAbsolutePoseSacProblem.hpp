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
 * @file FrameAbsolutePoseSacProblem.hpp
 * @brief Header file for the FrameAbsolutePoseSacProblem class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMEABSOLUTEPOSESACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMEABSOLUTEPOSESACPROBLEM_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <opengv/types.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#pragma GCC diagnostic pop
#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/LoopclosureNoncentralAbsoluteAdapter.hpp>
#include <okvis/assert_macros.hpp>

/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the sample consensus problems.
 */
namespace sac_problems {
/**
 * \brief The namespace for the absolute pose methods.
 */
namespace absolute_pose {

/**
 * \brief Provides functions for fitting an absolute-pose model to a set of
 *        bearing-vector to point correspondences, using different algorithms (central
 *        and non-central ones). Used in a sample-consenus paradigm for rejecting
 *        outlier correspondences.
 */
template<class DERIVED_ADAPTER_T>
class FrameAbsolutePoseSacProblem : public AbsolutePoseSacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The AbsolutePoseSacProblem base type.
  typedef AbsolutePoseSacProblem base_t;

  /** The type of adapter that is expected by the methods */
  using base_t::adapter_t;
  /** The possible algorithms for solving this problem */
  using base_t::algorithm_t;
  /** The model we are trying to fit (transformation) */
  using base_t::model_t;

  /**
   * @brief Constructor.
   * @param[in] adapter Visitor holding bearing vectors, world points, etc.
   * @param[in] algorithm The algorithm we want to use.
   * @warning Only okvis::absolute_pose::FrameNoncentralAbsoluteAdapter supported.
   */
  FrameAbsolutePoseSacProblem(adapter_t & adapter, algorithm_t algorithm)
      : base_t(adapter, algorithm),
        adapterDerived_(
            *static_cast<DERIVED_ADAPTER_T*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(&_adapter)
        || dynamic_cast<opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter*>(&_adapter),
        "unsupported adapter type")
  }

  /**
   * @brief Constructor.
   * @param[in] adapter Visitor holding bearing vectors, world points, etc.
   * @param[in] algorithm The algorithm we want to use.
   * @param[in] indices A vector of indices to be used from all available
   *                    correspondences.
   * @warning Only okvis::absolute_pose::FrameNoncentralAbsoluteAdapter supported.
   */
  FrameAbsolutePoseSacProblem(adapter_t & adapter, algorithm_t algorithm,
                              const std::vector<int> & indices)
      : base_t(adapter, algorithm, indices),
        adapterDerived_(
            *static_cast<DERIVED_ADAPTER_T*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::absolute_pose::FrameNoncentralAbsoluteAdapter*>(&_adapter)
        || dynamic_cast<opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter*>(&_adapter),
        "unsupported adapter type")
  }

  virtual ~FrameAbsolutePoseSacProblem() {
  }

  /**
   * \brief Compute the distances of all samples whith respect to given model
   *        coefficients.
   * \param[in] model The coefficients of the model hypothesis.
   * \param[in] indices The indices of the samples of which we compute distances.
   * \param[out] scores The resulting distances of the selected samples. Low
   *                    distances mean a good fit.
   */
  virtual void getSelectedDistancesToModel(const model_t & model,
                                           const std::vector<int> & indices,
                                           std::vector<double> & scores) const {
    //compute the reprojection error of all points

    //compute inverse transformation
    model_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = model.block<3, 3>(0, 0).transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0) * model.col(3);

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;
    scores.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
      //get point in homogeneous form
      p_hom.block<3, 1>(0, 0) = adapterDerived_.getPoint(indices[i]);

      //compute the reprojection (this is working for both central and
      //non-central case)
      const point_t bodyReprojection = inverseSolution * p_hom;
      point_t reprojection = adapterDerived_.getCamRotation(indices[i])
          .transpose()
          * (bodyReprojection - adapterDerived_.getCamOffset(indices[i]));
      reprojection = reprojection / reprojection.norm();

      //compute the score
      const point_t error = (reprojection
          - adapterDerived_.getBearingVector(indices[i]));
      const double error_squared = error.transpose() * error;
      scores.push_back(error_squared / adapterDerived_.getSigmaAngle(indices[i]));
    }
  }

 protected:
  /// The adapter holding the bearing, correspondences etc.
  DERIVED_ADAPTER_T & adapterDerived_;

};

}
}
}

#endif /* INCLUDE_OKVIS_OPENGV_FRAMEABSOLUTEPOSESACPROBLEM_HPP_ */
