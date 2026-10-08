/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 */

/**
 * @file EwgImageUtility.hpp
 * @brief Entropy-Weighted Gradient (EWG) based image utility
 *
 * Implements the per-pixel EWG metric and the image-level utility U_t
 * (sum of per-pixel utilities) defined in:
 *
 *   J. Kim, Y. Cho, A. Kim, "Proactive Camera Attribute Control Using
 *   Bayesian Optimization for Illumination-Resilient Visual Navigation,"
 *   IEEE Transactions on Robotics, vol. 36, no. 4, Aug 2020.
 *
 * The metric combines image gradient strength with a local-entropy weight
 * and a saturation-penalty mask:
 *
 *   H_local,i  = -Sum_k P(i_k) log_2 P(i_k)        (5x5 local patch)
 *   H_i        = H_local,i / Sum_j H_local,j
 *   w_i        = (1/sigma) exp(-(H_i - mean(H))^2 / (2 sigma^2))
 *   W_i        = w_i / Sum_j w_j
 *   pi(H_i)    = 2 / (1 + exp(-alpha H_i + tau)) - 1
 *   M_i(H_i)   = 1 if H_i < H_thres else 0
 * Spatial Feature-Aware EWG (SFA-EWG) extends positive EWG (U_EWG_pos) with
 * a BRISK keypoint-driven modulation (same detector family as OKVIS Frontend):
 *   U_SFA_EWG = U_EWG_pos * (1 + lambda_f * Q_f * Q_s)
 */

#ifndef INCLUDE_OKVIS_EWG_IMAGE_UTILITY_HPP_
#define INCLUDE_OKVIS_EWG_IMAGE_UTILITY_HPP_

#include <cstddef>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Compute SFA-EWG (positive EWG × BRISK feature / spatial-uniformity bonus).
 *
 * @param image_in           Input image (any number of channels; converted to 8-bit grayscale).
 * @param H_thres            Saturation gate (paper default: 0.05).
 * @param alpha              Activation function steepness (paper-style default: 5.0).
 * @param tau                Activation function bias (paper-style default: 1.0).
 * @param local_patch_size   Local entropy patch size in pixels (paper default: 5).
 * @param num_intensity_bins Quantization for local entropy histogram (default: 8 = 256/32).
 * @param lambda_f           Weight for (Q_f * Q_s) bonus term (default 1.0).
 * @param N0                 Feature-count scale for Q_f (default 400).
 * @param grid_rows          Rows in spatial uniformity grid (default 3).
 * @param grid_cols          Cols in spatial uniformity grid (default 4).
 * @param brisk_uniformity_radius   Same as Frontend BRISK (ScaleSpaceFeatureDetector) arg 1.
 * @param brisk_octaves             Same as Frontend BRISK octaves.
 * @param brisk_absolute_threshold Same as Frontend BRISK absolute threshold.
 * @param brisk_max_keypoints       Same as Frontend BRISK max keypoints.
 * @return U_SFA_EWG (finite, >= 0). Returns 0 on empty/degenerate input.
 */
double computeImageUtilityEWG(const cv::Mat& image_in,
                              double H_thres = 0.05,
                              double alpha = 5.0,
                              double tau = 1.0,
                              int local_patch_size = 5,
                              int num_intensity_bins = 8,
                              double lambda_f = 1.0,
                              double N0 = 400.0,
                              int grid_rows = 3,
                              int grid_cols = 4,
                              double brisk_uniformity_radius = 40.0,
                              size_t brisk_octaves = 0,
                              double brisk_absolute_threshold = 200.0,
                              size_t brisk_max_keypoints = 450);

}  // namespace okvis

#endif  // INCLUDE_OKVIS_EWG_IMAGE_UTILITY_HPP_
