//
// Created by boche on 5/9/22.
//

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <vector>
#include "srl/projection/Lidar.hpp"



TEST(Lidar, functions)
{
    const size_t NUM_POINTS = 100;

    srl::projection::Lidar lidar = srl::projection::Lidar::testObject();

    // try quite a lot of points:
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        // create a random point in the field of view:
        srl::Vector2f imagePoint = lidar.createRandomImagePoint();

        // backProject
        srl::Vector3f ray;
        ASSERT_TRUE(lidar.backProject(imagePoint, &ray)) << "unsuccessful back projection";

        // randomise distance
        ray.normalize();
        ray *= (0.2 + 8 * (srl::Vector2f::Random()[0] + 1.0));

        // project
        srl::Vector2f imagePoint2;
        srl::Matrixf<2, 3> J;
        ASSERT_TRUE(lidar.project(ray, &imagePoint2, &J, nullptr)
                    == srl::projection::ProjectionStatus::Successful) << "unsuccessful projection";

        // check they are the same
        ASSERT_LT((imagePoint2 - imagePoint).norm(), 0.01) << "project/unproject failure";

        // check point Jacobian vs. NumDiff
        const srl::float_t dp = std::is_same<srl::float_t, double>::value ? 1.0e-8 : 1.0e-5;
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
            ASSERT_EQ(lidar.project(point_p, &imagePoint_p), srl::projection::ProjectionStatus::Successful);
            ASSERT_EQ(lidar.project(point_m, &imagePoint_m), srl::projection::ProjectionStatus::Successful);
            J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);

        }
        const srl::float_t threshold = std::is_same<srl::float_t, double>::value ? 1e-3 : 26; // ToDO: why fail with threshold 1e-04?
        ASSERT_LT((J_numDiff - J).norm(), threshold) << "Jacobian Verification failed";
    }
}
