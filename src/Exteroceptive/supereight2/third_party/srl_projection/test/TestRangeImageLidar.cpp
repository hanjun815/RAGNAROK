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
 *  Created on: March 24, 2020
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <vector>
#include "srl/projection/RangeImageLidar.hpp"



TEST(RangeImageLidar, functions)
{
  const size_t NUM_POINTS = 100;

  srl::projection::RangeImageLidar rangeImageLidar = srl::projection::RangeImageLidar::testObject();

  // try quite a lot of points:
  for (size_t i = 0; i < NUM_POINTS; ++i) {
    // create a random point in the field of view:
    srl::Vector2f imagePoint = rangeImageLidar.createRandomImagePoint();

    // backProject
    srl::Vector3f ray;
    ASSERT_TRUE(rangeImageLidar.backProject(imagePoint, &ray)) <<
                      "unsuccessful back projection";

    // randomise distance
    ray.normalize();
    ray *= (0.2 + 8 * (srl::Vector2f::Random()[0] + 1.0));

    // project
    srl::Vector2f imagePoint2;
    srl::Matrixf<2, 3> J;
    ASSERT_TRUE(rangeImageLidar.project(ray, &imagePoint2, &J, nullptr)
            == srl::projection::ProjectionStatus::Successful) <<
                      "unsuccessful projection";

    // check they are the same
    ASSERT_LT((imagePoint2 - imagePoint).norm(), 0.01) <<
                      "project/unproject failure";

    // check point Jacobian vs. NumDiff
    const srl::float_t dp = std::is_same<srl::float_t, double>::value ? 1.0e-7 : 1.0e-5;
    srl::Matrixf<2, 3> J_numDiff;
    for (size_t d = 0; d < 3; ++d) {
      srl::Vector3f point_p = ray
          + srl::Vector3f(d == 0 ? dp : 0, d == 1 ? dp : 0,
                            d == 2 ? dp : 0);
      srl::Vector3f point_m = ray
          - srl::Vector3f(d == 0 ? dp : 0, d == 1 ? dp : 0,
                            d == 2 ? dp : 0);
      srl::Vector2f imagePoint_p (-1.f, -1.f);
      srl::Vector2f imagePoint_m (-1.f, -1.f);
      ASSERT_EQ(rangeImageLidar.project(point_p, &imagePoint_p), srl::projection::ProjectionStatus::Successful);
      ASSERT_EQ(rangeImageLidar.project(point_m, &imagePoint_m), srl::projection::ProjectionStatus::Successful);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }
    const srl::float_t threshold = std::is_same<srl::float_t, double>::value ? 1e-4 : 26;
    ASSERT_LT((J_numDiff - J).norm(), threshold) << "Jacobian Verification failed";
  }
}
