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
 * @file ErrorInterface.cpp
 * @brief Source file for the ErrorInterface class.
 * @author Stefan Leutenegger
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/cost_function.h>
#include <ceres/crs_matrix.h>
#include <ceres/evaluation_callback.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/ordered_groups.h>
#include <ceres/problem.h>
#include <ceres/product_manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <ceres/version.h>
#pragma GCC diagnostic pop

#include <okvis/ceres/ErrorInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

bool jacobiansCorrect(const ::ceres::Problem* problem, const ::ceres::ResidualBlockId id,
                      double tolerance) {
  const ::ceres::CostFunction* costFunction = problem->GetCostFunctionForResidualBlock(id);
  const okvis::ceres::ErrorInterface* error
      = dynamic_cast<const okvis::ceres::ErrorInterface*>(costFunction);

  // the parameter blocks
  std::vector<double*> parameterBlocks;
  problem->GetParameterBlocksForResidualBlock(id, &parameterBlocks);

  // check error term found and converted
  if(!error){
    return false;
  }

  // residual
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> residual(error->residualDim(),1);

  // set up jacobians and minimal jacobians
  std::vector<double*> jacobians(size_t(error->parameterBlocks()));
  std::vector<double*> jacobiansMinimal(size_t(error->parameterBlocks()));
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
      jacobiansEigen(size_t(error->parameterBlocks()));
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
      jacobiansMinimalEigen(size_t(error->parameterBlocks()));
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>
      jacobiansNumDiff(size_t(error->parameterBlocks()));
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>
      jacobiansMinimalNumDiff(size_t(error->parameterBlocks()));

  // now prepare for evaluation per parameter block
  std::vector<double*> parameterBlocksCopy(parameterBlocks.size());
  for(size_t i=0; i<parameterBlocks.size(); ++i) {
    parameterBlocksCopy.at(i) = parameterBlocks.at(i);
    const int dim = error->parameterBlockDim(int(i));
    int minDim = dim;
    const ::ceres::Manifold* manifold = problem->GetManifold(parameterBlocks[i]);
    if(manifold) {
      minDim = manifold->TangentSize();
    }
    jacobiansEigen.at(i) =
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(error->residualDim(), dim);
    jacobians[i] = jacobiansEigen[i].data();
    jacobiansMinimalEigen.at(i) =
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(error->residualDim(), minDim);
    jacobiansMinimal[i] = jacobiansMinimalEigen[i].data();
  }

  // evaluate to get the analytical jacobians
  error->EvaluateWithMinimalJacobians(parameterBlocks.data(),residual.data(),
                                      jacobians.data(),jacobiansMinimal.data());

  // now get the numDiff ones
  static const double delta = 1.0e-7;
  for(size_t i=0; i<parameterBlocks.size(); ++i) {
    const int dim = error->parameterBlockDim(int(i));
    int minDim = dim;
    Eigen::Matrix<double,Eigen::Dynamic,1> parameter(dim,1);
    for(int j=0; j<dim; ++j) {
      parameter[j] = parameterBlocks.at(i)[j];
    }
    const ::ceres::Manifold* manifold = problem->GetManifold(parameterBlocks[i]);
    if(manifold) {
      minDim = manifold->TangentSize();
    }

    // set up num diff Jacobians
    jacobiansNumDiff.at(i)
        = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(error->residualDim(), dim);
    jacobiansMinimalNumDiff.at(i)
        = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(error->residualDim(), minDim);

    // perturb along elements of the parameter dimension
    for(int j=0; j<minDim; ++j) {
      Eigen::Matrix<double,Eigen::Dynamic,1> parameter_p(dim,1);
      parameter_p = parameter;
      Eigen::Matrix<double,Eigen::Dynamic,1> parameter_m(dim,1);
      parameter_m = parameter;

      // perturbation
      Eigen::Matrix<double,Eigen::Dynamic,1> delta_p(minDim,1);
      Eigen::Matrix<double,Eigen::Dynamic,1> delta_m(minDim,1);
      delta_p.setZero();
      delta_p[j] = delta;
      delta_m.setZero();
      delta_m[j] = -delta;
      if(manifold) {
        manifold->Plus(parameter.data(), delta_p.data(), parameter_p.data());
        manifold->Plus(parameter.data(), delta_m.data(), parameter_m.data());
      } else {
        parameter_p = parameter + delta_p;
        parameter_m = parameter + delta_m;
      }

      // evaluate perturbed versions
      parameterBlocksCopy.at(i) = parameter_p.data();
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> residual_p(error->residualDim(),1);
      error->EvaluateWithMinimalJacobians(parameterBlocksCopy.data(),residual_p.data(),
                                          nullptr, nullptr);
      parameterBlocksCopy.at(i) = parameter_m.data();
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> residual_m(error->residualDim(),1);
      error->EvaluateWithMinimalJacobians(parameterBlocksCopy.data(),residual_m.data(),
                                          nullptr, nullptr);
      parameterBlocksCopy.at(i) = parameterBlocks.at(i);

      // fill Jacobian column
      jacobiansMinimalNumDiff.at(i).col(j) = (1.0/(2.0*delta))*(residual_p-residual_m);
    }

    if(!jacobiansMinimalEigen.at(i).isApprox(jacobiansMinimalNumDiff.at(i), tolerance)) {
      std::cout << "analytical Jacobian " << i << std::endl
                << jacobiansMinimalEigen.at(i) << std::endl;
      std::cout << "vs num. diff. Jacobian " << i << std::endl
                << jacobiansMinimalNumDiff.at(i) << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis

