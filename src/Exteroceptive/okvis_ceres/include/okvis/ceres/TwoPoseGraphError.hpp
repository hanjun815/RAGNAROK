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
 * @file TwoPoseGraphError.hpp
 * @brief Header file for the TwoPoseGraphError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_TWOPOSEGRAPHERROR_HPP_
#define INCLUDE_OKVIS_CERES_TWOPOSEGRAPHERROR_HPP_

#include <cmath>
#include <vector>

#include <ceres/sized_cost_function.h>
#include <ceres/loss_function.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>
#include <okvis/FrameTypedefs.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

class TwoPoseGraphErrorConst;
class TwoPoseStandardGraphError;

/// \brief Base class for Two-pose graph factors.
class TwoPoseGraphError : public ErrorInterface {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Trivial destructor.
  ~TwoPoseGraphError() = default;

  /// \brief Set the two pose IDs.
  /// \param referencePoseId The reference pose ID.
  /// \param otherPoseId The ID of the other pose.
  void setPoseIds(StateId referencePoseId, StateId otherPoseId) {
    referencePoseId_ = referencePoseId;
    otherPoseId_ = otherPoseId;
  }

  /// \brief Get the reference pose ID.
  StateId referencePoseId() const {return referencePoseId_; }

  /// \brief Get the other pose ID.
  StateId otherPoseId() const {return otherPoseId_; }

  /// \brief A helper struct holding information about an observation.
  struct Observation {
    KeypointIdentifier keypointIdentifier; ///< Keypoint ID.
    std::shared_ptr<ceres::PoseParameterBlock> pose; ///< Pose parameter block.
    std::shared_ptr<ceres::HomogeneousPointParameterBlock> hPoint; ///< Landmark in homog. coord.
    std::shared_ptr<ceres::PoseParameterBlock> extrinsics; ///< The extrinsics pose.
    std::shared_ptr<ReprojectionError2dBase> reprojectionError; ///< The reprojection error.
    ::ceres::LossFunction* lossFunction; ///< Do we use a robust loss?
    bool isMarginalised; ///< Has this been processed into the TwoPoseError?
    bool isDuplication; ///< Is this a duplication of the same observ. in a different TwoPoseError?
  };

  /// \brief Add one observation.
  /// @param[in] keypointIdentifier Keypoint ID.
  /// @param[in] reprojectionError The reprojection error.
  /// @param[in] lossFunction Do we use a robust loss?
  /// @param[in] pose Pose parameter block.
  /// @param[in] hPoint The landmark in homogeneous coordinates.
  /// @param[in] extrinsics The reprojection error.
  /// @param[in] isDuplication Is this a duplication of the same obs. in a different TwoPoseError?
  /// @param[in] weight Relative weight of this error term.
  bool addObservation(const KeypointIdentifier & keypointIdentifier,
                      const std::shared_ptr<const ReprojectionError2dBase> &reprojectionError,
                      ::ceres::LossFunction* lossFunction,
                      const std::shared_ptr<ceres::PoseParameterBlock> & pose,
                      const std::shared_ptr<ceres::HomogeneousPointParameterBlock> & hPoint,
                      const std::shared_ptr<ceres::PoseParameterBlock> & extrinsics,
                      bool isDuplication = false, double weight = 1.0);

  /// @brief Obtain all the landmarks held in here.
  /// @param[out] landmarks The landmarks.
  void getLandmarks(std::set<uint64_t> & landmarks) const;

  /// @brief Convert this relative pose error back into observations.
  /// @param[out] observations The observations originally passed.
  /// @param[out] duplicates Indicates which ones had been added originally as duplications.
  /// @return True on success.
  virtual bool convertToReprojectionErrors(std::vector<Observation> & observations,
                                   std::vector<KeypointIdentifier> & duplicates) = 0;

  /// @brief Access passed observations.
  /// @param[out] observations The observations originally passed.
  /// @return The number of observations.
  int getObservations(std::vector<Observation> & observations) const {
    for (const auto &obss : observations_) {
      for (const auto &obs : obss.second) {
        observations.push_back(obs);
      }
    }
    return int(observations.size());
  }

  /// @brief This computes a relative pose error from the observations.
  /// @return True on success.
  virtual bool compute() = 0;

  /// @brief The strength of this link (regarding position uncertainty only) -- use for
  /// visualisations.
  /// @return The strength as inverse position standard deviation [1/m].
  virtual double strength() const = 0;

  /// \brief Get a cloned relative pose error, but as a constant one (i.e. not back-convertalble)
  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const = 0;

  /// \brief Get the multiplicative weight applied to the residual / Jacobian when evaluating.
  ///        A value of 1.0 means no scaling. Smaller values weaken this constraint
  ///        (equivalent to multiplying the information matrix by `weightScale`).
  double weightScale() const { return weightScale_; }

  /// \brief Set the multiplicative weight applied to the residual / Jacobian when evaluating.
  ///        Negative or zero values are clamped to 0.0 (constraint disabled).
  void setWeightScale(double weightScale) {
    weightScale_ = (std::isfinite(weightScale) && weightScale > 0.0) ? weightScale : 0.0;
  }

  protected:

  /// \brief Adding residual blocks will invalidate this.
  /// Before optimizing, call updateErrorComputation()
  bool errorComputationValid_;

  /// \brief Multiplier applied to the unweighted residual r = J_ * error inside Evaluate().
  ///        Effectively rescales the (already factored) information matrix by this factor:
  ///        residual <- sqrt(weightScale_) * J_ * error,
  ///        jacobian <- sqrt(weightScale_) * J_ * Jerr.
  ///        weightScale_ = 1.0 reproduces the original behaviour.
  double weightScale_ = 1.0;

  /// \brief Helper struct for connected parameter blocks.
  template<int SIZE>
  struct ParameterBlockInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Eigen::Matrix<double, SIZE, 1> parameters; ///< Parameters: original values when added.

    /// \brief Default constructor.
    ParameterBlockInfo() {}

    /// \brief Constructor from parameter block and index.
    /// parameterBlock The parameter block.
    /// idx The ordering index.
    ParameterBlockInfo(const std::shared_ptr<ParameterBlock> & parameterBlock, size_t idx) :
        parameterBlock(parameterBlock), idx(idx)
    {
      std::memcpy(parameters.data(), parameterBlock->parameters(), SIZE*sizeof(double));
    }
    std::shared_ptr<ParameterBlock> parameterBlock; ///< The parameter block.
    size_t idx; ///< The ordering index.
  };

  /// \todo Redo -- needed for DeltaX computation...
  ParameterBlockInfo<7> poseParameterBlockInfos_[2]; ///< Parameter block infos (poses).
  AlignedVector<ParameterBlockInfo<7>> extrinsicsParameterBlockInfos_; ///< Extrinsics infos.
  AlignedVector<ParameterBlockInfo<4>> landmarkParameterBlockInfos_; ///< Landmark Parameter infos.

  /// \brief Maps parameter block Ids to index
  std::map<uint64_t, size_t> landmarkParameterBlockId2idx_;  ///< Maps parameter block Ids to index
  size_t extrinsicsSize_ = 0; ///< Size of extrinsics block.
  size_t sparseSize_ = 0; ///< Size of sparse part (landmarks).

  StateId referencePoseId_; ///< The reference pose ID.
  StateId otherPoseId_; ///< The ID of the other pose.

  bool isComputed_ = false; ///< Has the error term been computed?

  size_t numCams_ = 0;  ///< The number of cameras on the rig.
  bool stayConst_ = true;  ///< If true, the reference and other state extrinsics are the same.

  std::map<uint64_t, std::vector<Observation>> observations_; ///< Stores the added observations.
  AlignedMap<uint64_t, Eigen::Vector4d> landmarks_; ///< Landmarks in S0 coordinates.

  kinematics::Transformation linearisationPoint_T_S0S1_; ///< Relative poses
  std::vector<std::shared_ptr<kinematics::Transformation>> linearisationPoints_T_SC_; ///< Extins.
};

/// \brief Relative error between two poses. \todo also use non-const extrinsics...
class TwoPoseStandardGraphError : public ::ceres::SizedCostFunction<
    6 /* number of residuals */,
    7, /* size of first parameter */
    7 /* size of second parameter */>, public TwoPoseGraphError {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  /// \brief Number of residuals (6).
  static const int kNumResiduals = 6;

  /// \brief Constructor.
  TwoPoseStandardGraphError() = delete;

  /// \brief Simple constructor with the two pose IDs.
  /// \param referencePoseId The reference pose ID.
  /// \param otherPoseId The ID of the other pose.
  /// \param numCams The number of cameras on the rig.
  /// \param stayConst If true (default), the reference and other state extrinsics are the same.
  TwoPoseStandardGraphError(StateId referencePoseId, StateId otherPoseId, size_t numCams,
                            bool stayConst = true);

  /// @brief Convert this relative pose error back into observations.
  /// @param[out] observations The observations originally passed.
  /// @param[out] duplicates Indicates which ones had been added originally as duplications.
  /// @return True on success.
  virtual bool convertToReprojectionErrors(
    std::vector<Observation> &observations,
    std::vector<KeypointIdentifier> &duplicates) override final;

  /// @brief This computes a relative pose error from the observations.
  /// @return True on success.
  virtual bool compute() override final;

  // error term and Jacobian implementation (inherited pure virtuals from ::ceres::CostFunction)
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

  /// \brief Get a cloned relative pose error, but as a constant one (i.e. not back-convertalble)
  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const override final;

  /// @brief Residual block type as string
  virtual std::string typeInfo() const override final{
    return "TwoPoseStandardGraphError";
  }

  /// @brief The strength of this link (regarding position uncertainty only) -- use for
  /// visualisations.
  /// @return The strength as inverse position standard deviation [1/m].
  virtual double strength() const override final ;

protected:

  /// @name The internal storage of the linearised system.
  /// @{
  Eigen::Matrix<double, 6, 6> H00_;  ///< lhs - Hessian dense (sparse/sparse-dense comp. on the fly)
  Eigen::Matrix<double, 6, 1> b0_;  ///<  rhs constant part dense

  /// We use the formulation e = DelatX_i + T_WS_bar * (T_WS0^(-1)*T_WS)^(-1).
  /// With information matrix H = J_^T*J_.
  Eigen::Matrix<double,6,1> DeltaX_; ///< nominal deviation from linearisation point
  Eigen::Matrix<double,6,6> J_; ///< the decomposed inf. matrix (of the relative system at lin pt)
};

class TwoPoseGraphErrorConst : public ErrorInterface {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Get the multiplicative weight applied to the residual / Jacobian when evaluating.
  double weightScale() const { return weightScale_; }
  /// \brief Set the multiplicative weight applied to the residual / Jacobian when evaluating.
  void setWeightScale(double weightScale) {
    weightScale_ = (std::isfinite(weightScale) && weightScale > 0.0) ? weightScale : 0.0;
  }

  protected:
  /// \brief See TwoPoseGraphError::weightScale_.
  double weightScale_ = 1.0;
};

/// @brief Constant (non-convertible) version of the TwoPoseGraphError (see above).
class TwoPoseStandardGraphErrorConst : public ::ceres::SizedCostFunction<
    6 /* number of residuals */,
    7, /* size of first parameter */
    7 /* size of second parameter */>, public TwoPoseGraphErrorConst {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  /// \brief Number of residuals (6).
  static const int kNumResiduals = 6;

  /// \brief Default constructor.
  TwoPoseStandardGraphErrorConst() = delete;
  /// \brief To construct from TwoPoseGraphError.
  TwoPoseStandardGraphErrorConst(
      const Eigen::Matrix<double,6,1> & DeltaX, const Eigen::Matrix<double,6,6> & J,
      const kinematics::Transformation & linearisationPoint_T_S0S1) :
    DeltaX_(DeltaX), J_(J), linearisationPoint_T_S0S1_(linearisationPoint_T_S0S1)
  {}
  /// \brief Trivial destructor.
  virtual ~TwoPoseStandardGraphErrorConst() override = default;
  // error term and Jacobian implementation (inherited pure virtuals from ::ceres::CostFunction)
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
    return "TwoPoseStandardGraphErrorConst";
  }

protected:
  /// @name The internal storage of the linearised system.
  /// @{

  /// We use the formulation e = DelatX_i + T_WS_bar * (T_WS0^(-1)*T_WS)^(-1).
  /// With information matrix H = J_^T*J_.
  Eigen::Matrix<double,6,1> DeltaX_; ///< nominal deviation from linearisation point
  Eigen::Matrix<double,6,6> J_; ///< the decomposed info matrix (of the relative system at lin pt)
  kinematics::Transformation linearisationPoint_T_S0S1_; ///< Relative poses
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_TWOPOSEGRAPH_HPP_ */
