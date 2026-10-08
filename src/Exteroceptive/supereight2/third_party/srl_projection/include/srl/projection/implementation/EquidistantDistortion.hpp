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

/**
 * @file implementation/EquidistantDistortion.hpp
 * @brief Header implementation file for the EquidistantDistortion class.
 * @author Stefan Leutenegger
 */


#include <Eigen/LU>
#include <iostream>

/// \brief Main namespace of this package.
namespace srl {
/// \brief Namespace for camera-related functionality.
namespace projection {

// The default constructor with all zero ki
EquidistantDistortion::EquidistantDistortion()
    : k1_(0.0),
      k2_(0.0),
      k3_(0.0),
      k4_(0.0)
{
  parameters_.setZero();
}

// Constructor initialising ki
EquidistantDistortion::EquidistantDistortion(float_t k1, float_t k2, float_t k3,
                                             float_t k4)
{
  parameters_[0] = k1;
  parameters_[1] = k2;
  parameters_[2] = k3;
  parameters_[3] = k4;
  k1_ = k1;
  k2_ = k2;
  k3_ = k3;
  k4_ = k4;
}

bool EquidistantDistortion::setParameters(const VectorXf & parameters)
{
  if (parameters.cols() != NumDistortionIntrinsics) {
    return false;
  }
  parameters_ = parameters;
  k1_ = parameters[0];
  k2_ = parameters[1];
  k3_ = parameters[2];
  k4_ = parameters[3];
  return true;
}

bool EquidistantDistortion::distort(const Vector2f & pointUndistorted,
                                    Vector2f * pointDistorted) const
{
  // distortion only:
  const float_t u0 = pointUndistorted[0];
  const float_t u1 = pointUndistorted[1];
  const float_t r = sqrt(u0 * u0 + u1 * u1);
  const float_t theta = atan(r);
  const float_t theta2 = theta * theta;
  const float_t theta4 = theta2 * theta2;
  const float_t theta6 = theta4 * theta2;
  const float_t theta8 = theta4 * theta4;
  const float_t thetad = theta
      * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

  const float_t scaling = (r > 1e-8) ? thetad / r : 1.0;
  (*pointDistorted)[0] = scaling * u0;
  (*pointDistorted)[1] = scaling * u1;
  return true;
}
bool EquidistantDistortion::distort(const Vector2f & pointUndistorted,
                                    Vector2f * pointDistorted,
                                    Matrix2f * pointJacobian,
                                    Matrix2Xf * parameterJacobian) const
{
  // distortion first:
  const float_t u0 = pointUndistorted[0];
  const float_t u1 = pointUndistorted[1];
  const float_t r = sqrt(u0 * u0 + u1 * u1);
  const float_t theta = atan(r);
  const float_t theta2 = theta * theta;
  const float_t theta4 = theta2 * theta2;
  const float_t theta6 = theta4 * theta2;
  const float_t theta8 = theta4 * theta4;
  const float_t thetad = theta
      * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

  const float_t scaling = (r > 1e-8) ? thetad / r : 1.0;
  (*pointDistorted)[0] = scaling * u0;
  (*pointDistorted)[1] = scaling * u1;

  Matrix2f & J = *pointJacobian;
  if (r > 1e-8) {
    // mostly matlab generated...
    float_t t2;
    float_t t3;
    float_t t4;
    float_t t6;
    float_t t7;
    float_t t8;
    float_t t9;
    float_t t11;
    float_t t17;
    float_t t18;
    float_t t19;
    float_t t20;
    float_t t25;

    t2 = u0 * u0;
    t3 = u1 * u1;
    t4 = t2 + t3;
    t6 = atan(sqrt(t4));
    t7 = t6 * t6;
    t8 = 1.0 / sqrt(t4);
    t9 = t7 * t7;
    t11 = 1.0 / ((t2 + t3) + 1.0);
    t17 = (((k1_ * t7 + k2_ * t9) + k3_ * t7 * t9) + k4_ * (t9 * t9)) + 1.0;
    t18 = 1.0 / t4;
    t19 = 1.0 / sqrt(t4 * t4 * t4);
    t20 = t6 * t8 * t17;
    t25 = ((k2_ * t6 * t7 * t8 * t11 * u1 * 4.0
        + k3_ * t6 * t8 * t9 * t11 * u1 * 6.0)
        + k4_ * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
        + k1_ * t6 * t8 * t11 * u1 * 2.0;
    t4 = ((k2_ * t6 * t7 * t8 * t11 * u0 * 4.0
        + k3_ * t6 * t8 * t9 * t11 * u0 * 6.0)
        + k4_ * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
        + k1_ * t6 * t8 * t11 * u0 * 2.0;
    t7 = t11 * t17 * t18 * u0 * u1;
    J(0, 1) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
    J(1, 1) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
        + t6 * t8 * t25 * u1;
    J(0, 0) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
        + t6 * t8 * t4 * u0;
    J(1, 0) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;

    if (parameterJacobian) {
      Matrix2Xf & Ji = *parameterJacobian;
      Ji.resize(2,NumDistortionIntrinsics);
      // mostly matlab generated...
      float_t t6;
      float_t t2;
      float_t t3;
      float_t t8;
      float_t t10;

      t6 = u0 * u0 + u1 * u1;
      t2 = atan(sqrt(t6));
      t3 = t2 * t2;
      t8 = t3 * t3;
      t6 = 1.0 / sqrt(t6);
      t10 = t8 * t8;
      Ji(0, 0) = t2 * t3 * t6 * u0;
      Ji(1, 0) = t2 * t3 * t6 * u1;
      Ji(0, 1) = t2 * t8 * t6 * u0;
      Ji(1, 1) = t2 * t8 * t6 * u1;
      Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
      Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
      Ji(0, 3) = t2 * t6 * t10 * u0;
      Ji(1, 3) = t2 * t6 * t10 * u1;

    }
  } else {
    // handle limit case for [u0,u1]->0
    if (parameterJacobian) {
      parameterJacobian->resize(2,NumDistortionIntrinsics);
      parameterJacobian->setZero();
    }
    J.setIdentity();
  }

  return true;
}
bool EquidistantDistortion::distortWithExternalParameters(
    const Vector2f & pointUndistorted,
    const VectorXf & parameters, Vector2f * pointDistorted,
    Matrix2f * pointJacobian, Matrix2Xf * parameterJacobian) const
{
  // decompose parameters

  const float_t k1 = parameters[0];
  const float_t k2 = parameters[1];
  const float_t k3 = parameters[2];
  const float_t k4 = parameters[3];
  // distortion first:
  const float_t u0 = pointUndistorted[0];
  const float_t u1 = pointUndistorted[1];
  const float_t r = sqrt(u0 * u0 + u1 * u1);
  const float_t theta = atan(r);
  const float_t theta2 = theta * theta;
  const float_t theta4 = theta2 * theta2;
  const float_t theta6 = theta4 * theta2;
  const float_t theta8 = theta4 * theta4;
  const float_t thetad = theta
      * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

  const float_t scaling = (r > 1e-8) ? thetad / r : 1.0;
  (*pointDistorted)[0] = scaling * u0;
  (*pointDistorted)[1] = scaling * u1;

  Matrix2f & J = *pointJacobian;
  if (r > 1e-8) {
    // mostly matlab generated...
    float_t t2;
    float_t t3;
    float_t t4;
    float_t t6;
    float_t t7;
    float_t t8;
    float_t t9;
    float_t t11;
    float_t t17;
    float_t t18;
    float_t t19;
    float_t t20;
    float_t t25;

    t2 = u0 * u0;
    t3 = u1 * u1;
    t4 = t2 + t3;
    t6 = atan(sqrt(t4));
    t7 = t6 * t6;
    t8 = 1.0 / sqrt(t4);
    t9 = t7 * t7;
    t11 = 1.0 / ((t2 + t3) + 1.0);
    t17 = (((k1 * t7 + k2 * t9) + k3 * t7 * t9) + k4 * (t9 * t9)) + 1.0;
    t18 = 1.0 / t4;
    t19 = 1.0 / sqrt(t4 * t4 * t4);
    t20 = t6 * t8 * t17;
    t25 = ((k2 * t6 * t7 * t8 * t11 * u1 * 4.0
        + k3 * t6 * t8 * t9 * t11 * u1 * 6.0)
        + k4 * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
        + k1 * t6 * t8 * t11 * u1 * 2.0;
    t4 = ((k2 * t6 * t7 * t8 * t11 * u0 * 4.0
        + k3 * t6 * t8 * t9 * t11 * u0 * 6.0)
        + k4 * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
        + k1 * t6 * t8 * t11 * u0 * 2.0;
    t7 = t11 * t17 * t18 * u0 * u1;
    J(0, 0) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
    J(1, 0) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
        + t6 * t8 * t25 * u1;
    J(0, 1) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
        + t6 * t8 * t4 * u0;
    J(1, 1) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;
    if (parameterJacobian) {
      Matrix2Xf & Ji = *parameterJacobian;
      Ji.resize(2,NumDistortionIntrinsics);
      // mostly matlab generated...
      float_t t6;
      float_t t2;
      float_t t3;
      float_t t8;
      float_t t10;

      t6 = u0 * u0 + u1 * u1;
      t2 = atan(sqrt(t6));
      t3 = t2 * t2;
      t8 = t3 * t3;
      t6 = 1.0 / sqrt(t6);
      t10 = t8 * t8;
      Ji(0, 0) = t2 * t3 * t6 * u0;
      Ji(1, 0) = t2 * t3 * t6 * u1;
      Ji(0, 1) = t2 * t8 * t6 * u0;
      Ji(1, 1) = t2 * t8 * t6 * u1;
      Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
      Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
      Ji(0, 3) = t2 * t6 * t10 * u0;
      Ji(1, 3) = t2 * t6 * t10 * u1;

    }
  } else {
    // handle limit case for [u0,u1]->0
    if (parameterJacobian) {
      parameterJacobian->resize(2,NumDistortionIntrinsics);
      parameterJacobian->setZero();
    }
    J.setIdentity();
  }

  return true;
}
bool EquidistantDistortion::undistort(const Vector2f & pointDistorted,
                                      Vector2f * pointUndistorted) const
{
  // this is expensive: we solve with Gauss-Newton...
  Vector2f x_bar = pointDistorted;  // initialise at distorted point
  const int n = 5;  // just 5 iterations max.
  Matrix2f E;  // error Jacobian

  bool success = false;
  for (int i = 0; i < n; i++) {

    Vector2f x_tmp;

    distort(x_bar, &x_tmp, &E);

    Vector2f e(pointDistorted - x_tmp);
    Vector2f du = (E.transpose() * E).inverse() * E.transpose() * e;

    x_bar += du;

    const float_t chi2 = e.dot(e);
    if (chi2 < 1e-2) {
      success = true;
    }
    //std::cout<<"chi2"<<chi2<<std::endl;
    if (chi2 < 1e-15) {
      success = true;
      break;
    }

  }
  *pointUndistorted = x_bar;

  return success;
}
bool EquidistantDistortion::undistort(const Vector2f & pointDistorted,
                                      Vector2f * pointUndistorted,
                                      Matrix2f * pointJacobian) const
{
  // this is expensive: we solve with Gauss-Newton...
  Vector2f x_bar = pointDistorted;  // initialise at distorted point
  const int n = 5;  // just 5 iterations max.
  Matrix2f E;  // error Jacobian

  bool success = false;
  for (int i = 0; i < n; i++) {

    Vector2f x_tmp;

    distort(x_bar, &x_tmp, &E);

    Vector2f e(pointDistorted - x_tmp);
    Vector2f dx = (E.transpose() * E).inverse() * E.transpose() * e;

    x_bar += dx;

    const float_t chi2 = e.dot(e);
    if (chi2 < 1e-2) {
      success = true;
    }
    if (chi2 < 1e-15) {
      success = true;
      break;
    }

  }
  *pointUndistorted = x_bar;

  // the Jacobian of the inverse map is simply the inverse Jacobian.
  *pointJacobian = E.inverse();

  return success;
}

}  // namespace projection
}  // namespace srl
