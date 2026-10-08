/*********************************************************************************
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
 *  Created on: April 22, 2020
 *      Author: Sotiris Papatheodorou
 *********************************************************************************/

/**
 * @file typedefs.hpp
 * @brief Typedefs for ease of use.
 * @author Sotiris Papatheodorou
 */

#ifndef INCLUDE_SRL_TYPEDEFS_HPP_
#define INCLUDE_SRL_TYPEDEFS_HPP_

#include <limits>

#include <Eigen/Dense>

// Default to double precision
#ifndef SRL_FLOAT_T
#define SRL_FLOAT_T double
#endif


namespace srl {
  /// Set the precision of all operations (float or double).
  typedef SRL_FLOAT_T float_t;

  template <int _Rows, int _Cols>
  using Matrixf = Eigen::Matrix<float_t, _Rows, _Cols>;

  typedef Eigen::Matrix<float_t, 2, 2>                           Matrix2f;
  typedef Eigen::Matrix<float_t, 2, Eigen::Dynamic>              Matrix2Xf;
  typedef Eigen::Matrix<float_t, 3, 3>                           Matrix3f;
  typedef Eigen::Matrix<float_t, 3, Eigen::Dynamic>              Matrix3Xf;
  typedef Eigen::Matrix<float_t, 4, 4>                           Matrix4f;
  typedef Eigen::Matrix<float_t, 4, Eigen::Dynamic>              Matrix4Xf;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, 2>              MatrixX2f;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, 3>              MatrixX3f;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, 4>              MatrixX4f;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXf;

  typedef Eigen::Matrix<float_t, 1,2>               RowVector2f;
  typedef Eigen::Matrix<float_t, 1,3>               RowVector3f;
  typedef Eigen::Matrix<float_t, 1,4>               RowVector4f;
  typedef Eigen::Matrix<float_t, 1, Eigen::Dynamic> RowVectorXf;

  typedef Eigen::Matrix<float_t, 2, 1>              Vector2f;
  typedef Eigen::Matrix<float_t, 3, 1>              Vector3f;
  typedef Eigen::Matrix<float_t, 4, 1>              Vector4f;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, 1> VectorXf;

} // namespace srl

#endif

