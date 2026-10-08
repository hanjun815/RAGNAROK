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
 * @file FrameRelativePoseSacProblem.hpp
 * @brief Header file for the FrameRelativePoseSacProblem class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMERELATIVEPOSESACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMERELATIVEPOSESACPROBLEM_HPP_

#include <okvis/assert_macros.hpp>
#include <opengv/types.hpp>
#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the sample consensus problems.
 */
namespace sac_problems {
/**
 * \brief The namespace for the relative pose methods.
 */
namespace relative_pose {

/**
 * \brief Provides functions for fitting a relative-pose model to a set of
 *        bearing-vector to point correspondences, using different algorithms (only
 *        central case). Used in a sample-consenus paradigm for rejecting
 *        outlier correspondences.
 */
class FrameRelativePoseSacProblem : public CentralRelativePoseSacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The CentralRelativePoseSacProblem base type.
  typedef CentralRelativePoseSacProblem base_t;

  /** The type of adapter that is expected by the methods */
  using base_t::adapter_t;
  /** The possible algorithms for solving this problem */
  using base_t::algorithm_t;
  /** The model we are trying to fit (transformation) */
  using base_t::model_t;

  /**
   * \brief FrameRelativePoseSacProblem
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * \param[in] algorithm The algorithm we want to use.
   * @warning Only okvis::relative_pose::FrameRelativeAdapter supported.
   */
  FrameRelativePoseSacProblem(adapter_t & adapter, algorithm_t algorithm)
      : base_t(adapter, algorithm),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported")
  }

  /**
   * \brief FrameRelativePoseSacProblem
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * \param[in] algorithm The algorithm we want to use.
   * \param[in] indices A vector of indices to be used from all available
   *                    correspondences.
   * @warning Only okvis::relative_pose::FrameRelativeAdapter supported.
   */
  FrameRelativePoseSacProblem(adapter_t & adapter, algorithm_t algorithm,
                              const std::vector<int> & indices)
      : base_t(adapter, algorithm, indices),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported")
  }

  /// \brief Default destructor.
  virtual ~FrameRelativePoseSacProblem() override;

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
                                           std::vector<double> & scores) const override final {
    translation_t translation = model.col(3);
    rotation_t rotation = model.block<3, 3>(0, 0);
    adapterDerived_.sett12(translation);
    adapterDerived_.setR12(rotation);

    model_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = rotation.transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0) * translation;

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;

    for (size_t i = 0; i < indices.size(); i++) {
      p_hom.block<3, 1>(0, 0) = opengv::triangulation::triangulate2(
          adapterDerived_, size_t(indices[i]));
      bearingVector_t reprojection1 = p_hom.block<3, 1>(0, 0);
      bearingVector_t reprojection2 = inverseSolution * p_hom;
      reprojection1 = reprojection1 / reprojection1.norm();
      reprojection2 = reprojection2 / reprojection2.norm();
      bearingVector_t f1 = adapterDerived_.getBearingVector1(size_t(indices[i]));
      bearingVector_t f2 = adapterDerived_.getBearingVector2(size_t(indices[i]));

      //compute the score
      point_t error1 = (reprojection1 - f1);
      point_t error2 = (reprojection2 - f2);
      double error_squared1 = error1.transpose() * error1;
      double error_squared2 = error2.transpose() * error2;
      scores.push_back(
          error_squared1 * 0.5 / adapterDerived_.getSigmaAngle1(size_t(indices[i]))
              + error_squared2 * 0.5
                  / adapterDerived_.getSigmaAngle2(size_t(indices[i])));
    }
  }

 protected:
  /// The adapter holding the bearing, correspondences etc.
  opengv::relative_pose::FrameRelativeAdapter & adapterDerived_;

};

// Slight hack to avoid Clang warning
// "no out-of-line virtual method definitions; its vtable will beemitted in every translation unit"
FrameRelativePoseSacProblem::~FrameRelativePoseSacProblem() {}

}
}
}

#endif /* INCLUDE_OKVIS_OPENGV_FRAMERELATIVEPOSESACPROBLEM_HPP_ */
