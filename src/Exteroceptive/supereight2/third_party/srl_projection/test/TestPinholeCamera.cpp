/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Feb 3, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <vector>
#include "srl/projection/PinholeCamera.hpp"
#include "srl/projection/NoDistortion.hpp"
#include "srl/projection/RadialTangentialDistortion.hpp"
#include "srl/projection/RadialTangentialDistortion8.hpp"
#include "srl/projection/EquidistantDistortion.hpp"



TEST(PinholeCamera, functions)
{
  const size_t NUM_POINTS = 100;

  // instantiate all possible versions of test cameras
  std::vector<std::shared_ptr<srl::projection::ProjectionBase> > cameras;
  cameras.push_back(
      srl::projection::PinholeCamera<srl::projection::NoDistortion>::createTestObject());
  cameras.push_back(
      srl::projection::PinholeCamera<
          srl::projection::RadialTangentialDistortion>::createTestObject());
  cameras.push_back(
      srl::projection::PinholeCamera<srl::projection::EquidistantDistortion>::createTestObject());
  cameras.push_back(
      srl::projection::PinholeCamera<srl::projection::RadialTangentialDistortion8>::createTestObject());

  for (size_t c = 0; c < cameras.size(); ++c) {
    // try quite a lot of points:
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      // create a random point in the field of view:
      srl::Vector2f imagePoint = cameras.at(c)->createRandomImagePoint();

      // backProject
      srl::Vector3f ray;
      ASSERT_TRUE(cameras.at(c)->backProject(imagePoint, &ray)) <<
                        "unsuccessful back projection";

      // randomise distance
      ray.normalize();
      ray *= (0.2 + 8 * (srl::Vector2f::Random()[0] + 1.0));

      // project
      srl::Vector2f imagePoint2;
      srl::Matrixf<2, 3> J;
      srl::Matrix2Xf J_intrinsics;
      ASSERT_TRUE(cameras.at(c)->project(ray, &imagePoint2, &J, &J_intrinsics)
              == srl::projection::ProjectionStatus::Successful) <<
                        "unsuccessful projection";

      // check they are the same
      ASSERT_LT((imagePoint2 - imagePoint).norm(), 0.01) <<
                        "project/unproject failure";

      // check point Jacobian vs. NumDiff
      const srl::float_t dp = std::is_same<srl::float_t, double>::value ? 1.0e-7 : 1.0e-4;
      srl::Matrixf<2, 3> J_numDiff;
      for (size_t d = 0; d < 3; ++d) {
        srl::Vector3f point_p = ray
            + srl::Vector3f(d == 0 ? dp : 0,
                            d == 1 ? dp : 0,
                            d == 2 ? dp : 0);
        srl::Vector3f point_m = ray
            - srl::Vector3f(d == 0 ? dp : 0,
                            d == 1 ? dp : 0,
                            d == 2 ? dp : 0);
        srl::Vector2f imagePoint_p;
        srl::Vector2f imagePoint_m;
        cameras.at(c)->project(point_p, &imagePoint_p);
        cameras.at(c)->project(point_m, &imagePoint_m);
        J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
      }
      const srl::float_t threshold = std::is_same<srl::float_t, double>::value ? 1e-4 : 0.9;
      ASSERT_LT((J_numDiff - J).norm(), threshold) <<
                        "Jacobian Verification failed";

      // check intrinsics Jacobian
      const int numIntrinsics = cameras.at(c)->numIntrinsicsParameters();
      srl::VectorXf intrinsics;
      cameras.at(c)->getIntrinsics(intrinsics);
      srl::Matrix2Xf J_numDiff_intrinsics;
      J_numDiff_intrinsics.resize(2,numIntrinsics);
      for (int d = 0; d < numIntrinsics; ++d) {
        srl::VectorXf di;
        di.resize(numIntrinsics);
        di.setZero();
        di[d] = dp;
        srl::Vector2f imagePoint_p;
        srl::Vector2f imagePoint_m;
        srl::VectorXf intrinsics_p = intrinsics+di;
        srl::VectorXf intrinsics_m = intrinsics-di;
        cameras.at(c)->projectWithExternalParameters(ray, intrinsics_p, &imagePoint_p);
        cameras.at(c)->projectWithExternalParameters(ray, intrinsics_m, &imagePoint_m);
        J_numDiff_intrinsics.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
      }

      ASSERT_LT((J_numDiff_intrinsics - J_intrinsics).norm(), threshold) <<
          "Jacobian verification failed";
    }
  }
}

