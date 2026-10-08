/**
 * OKVIS2-X - Optional stereo DNN image enhancement (shared with Frontend pipeline).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef OKVIS_STEREO_DNN_IMAGE_ENHANCEMENT_HPP
#define OKVIS_STEREO_DNN_IMAGE_ENHANCEMENT_HPP

#include <opencv2/core.hpp>

#include <okvis/Parameters.hpp>

namespace okvis {

/**
 * Apply the same preprocessing as Frontend::detectAndDescribe for intensity images:
 * optional radial vignette, then AGCWD/CLAHE fusion.
 *
 * Uses \p fp as in YAML (not BO-tuned dynamic AGCWD/CLAHE weights from the SLAM thread).
 */
bool applyStereoDnnImageEnhancement(const cv::Mat& image_in,
                                   const FrontendParameters& fp,
                                   cv::Mat& image_out);

} // namespace okvis

#endif
