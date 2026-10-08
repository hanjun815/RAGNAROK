/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 */

/**
 * @file EwgImageUtility.cpp
 * @brief SFA-EWG (Spatial Feature-Aware Entropy-Weighted Gradient) image utility.
 */

#include <okvis/EwgImageUtility.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <brisk/brisk.h>
#include <opencv2/imgproc.hpp>

namespace okvis {

namespace {

cv::Mat toGrayU8(const cv::Mat& image_in) {
  if (image_in.empty()) {
    return cv::Mat();
  }
  cv::Mat src_u8;
  if (image_in.depth() == CV_8U) {
    src_u8 = image_in;
  } else {
    image_in.convertTo(src_u8, CV_8U);
  }
  cv::Mat gray;
  if (src_u8.channels() == 1) {
    gray = src_u8;
  } else if (src_u8.channels() == 3) {
    cv::cvtColor(src_u8, gray, cv::COLOR_BGR2GRAY);
  } else if (src_u8.channels() == 4) {
    cv::cvtColor(src_u8, gray, cv::COLOR_BGRA2GRAY);
  } else {
    cv::extractChannel(src_u8, gray, 0);
  }
  return gray;
}

// Compute the per-pixel local entropy H_local using a (patch x patch)
// neighbourhood and `num_bins` intensity bins. The result is CV_32FC1 with
// units of bits (entropy in base-2).
cv::Mat computeLocalEntropy(const cv::Mat& gray_u8, int patch, int num_bins) {
  const int rows = gray_u8.rows;
  const int cols = gray_u8.cols;
  cv::Mat H_local = cv::Mat::zeros(rows, cols, CV_32FC1);
  if (rows == 0 || cols == 0 || num_bins <= 0) {
    return H_local;
  }

  const int p = std::max(3, patch | 1);  // ensure odd >= 3
  const double inv_patch_total = 1.0 / static_cast<double>(p * p);

  // Quantize intensity into `num_bins`.
  const double quant_scale = static_cast<double>(num_bins) / 256.0;
  cv::Mat quantized(rows, cols, CV_8UC1);
  for (int r = 0; r < rows; ++r) {
    const uint8_t* src = gray_u8.ptr<uint8_t>(r);
    uint8_t* dst = quantized.ptr<uint8_t>(r);
    for (int c = 0; c < cols; ++c) {
      int q = static_cast<int>(src[c] * quant_scale);
      if (q >= num_bins) q = num_bins - 1;
      if (q < 0) q = 0;
      dst[c] = static_cast<uint8_t>(q);
    }
  }

  for (int b = 0; b < num_bins; ++b) {
    cv::Mat mask = (quantized == static_cast<uint8_t>(b));
    cv::Mat mask_f;
    mask.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);  // 0/1
    cv::Mat counts;
    cv::boxFilter(mask_f, counts, CV_32F, cv::Size(p, p),
                  cv::Point(-1, -1), /*normalize=*/false,
                  cv::BORDER_REPLICATE);
    for (int r = 0; r < rows; ++r) {
      const float* cp = counts.ptr<float>(r);
      float* hp = H_local.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const double cnt = static_cast<double>(cp[c]);
        if (cnt > 0.0) {
          const double pv = cnt * inv_patch_total;
          if (pv > 1e-12) {
            hp[c] -= static_cast<float>(pv * std::log2(pv));
          }
        }
      }
    }
  }
  return H_local;
}

struct BriskDetectorParams {
  double uniformity_radius = 0.0;
  size_t octaves = 0;
  double absolute_threshold = 0.0;
  size_t max_keypoints = 0;
};

inline bool operator==(const BriskDetectorParams& a, const BriskDetectorParams& b) {
  return a.uniformity_radius == b.uniformity_radius && a.octaves == b.octaves
         && a.absolute_threshold == b.absolute_threshold
         && a.max_keypoints == b.max_keypoints;
}

using BriskScaleSpaceDetector =
    brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>;

}  // namespace

double computeImageUtilityEWG(const cv::Mat& image_in,
                              double H_thres,
                              double alpha,
                              double tau,
                              int local_patch_size,
                              int num_intensity_bins,
                              double lambda_f,
                              double N0,
                              int grid_rows,
                              int grid_cols,
                              double brisk_uniformity_radius,
                              size_t brisk_octaves,
                              double brisk_absolute_threshold,
                              size_t brisk_max_keypoints) {
  cv::Mat gray = toGrayU8(image_in);
  if (gray.empty() || gray.total() == 0) {
    return 0.0;
  }
  const int rows = gray.rows;
  const int cols = gray.cols;

  constexpr double kHRefEps = 1e-12;
  constexpr double kPkEps = 1e-12;

  const int gr = std::max(1, grid_rows);
  const int gc = std::max(1, grid_cols);
  const int K_cells = gr * gc;
  const double N0_eff = std::max(1e-12, N0);

  const cv::Mat H_local =
      computeLocalEntropy(gray, local_patch_size, num_intensity_bins);

  double max_h = 0.0;
  cv::minMaxLoc(H_local, nullptr, &max_h, nullptr, nullptr, cv::Mat());
  const double H_ref = max_h;

  // Flat / saturation / degenerate entropy map → no score.
  if (!std::isfinite(H_ref) || H_ref <= kHRefEps) {
    return 0.0;
  }

  const double denom = H_ref + kHRefEps;

  cv::Mat H(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r) {
    const float* src = H_local.ptr<float>(r);
    float* dst = H.ptr<float>(r);
    for (int c = 0; c < cols; ++c) {
      const double hn = static_cast<double>(src[c]) / denom;
      dst[c] = static_cast<float>(std::clamp(hn, 0.0, 1.0));
    }
  }

  cv::Scalar mean_H_sc, std_H_sc;
  cv::meanStdDev(H, mean_H_sc, std_H_sc);
  const double mu = mean_H_sc[0];
  const double sigma = std::max(std_H_sc[0], 1e-6);
  const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
  const double inv_sigma = 1.0 / sigma;

  cv::Mat W(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r) {
    const float* hp = H.ptr<float>(r);
    float* wp = W.ptr<float>(r);
    for (int c = 0; c < cols; ++c) {
      const double dh = static_cast<double>(hp[c]) - mu;
      wp[c] = static_cast<float>(inv_sigma * std::exp(-dh * dh * inv_2sigma2));
    }
  }
  const double sum_w = cv::sum(W)[0];
  if (!std::isfinite(sum_w) || sum_w <= 1e-12) {
    return 0.0;
  }
  W /= static_cast<float>(sum_w);

  cv::Mat gx, gy;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
  cv::Mat grad_mag;
  cv::magnitude(gx, gy, grad_mag);

  const double weighted_grad_mean = static_cast<double>(cv::sum(W.mul(grad_mag))[0]);
  if (!std::isfinite(weighted_grad_mean)) {
    return 0.0;
  }

  const double inv_num_pixels = 1.0 / static_cast<double>(rows * cols);
  double U_ewg_pos = 0.0;
  for (int r = 0; r < rows; ++r) {
    const float* hp = H.ptr<float>(r);
    const float* wp = W.ptr<float>(r);
    const float* gp = grad_mag.ptr<float>(r);
    for (int c = 0; c < cols; ++c) {
      const double Hi = static_cast<double>(hp[c]);
      const double Wi = static_cast<double>(wp[c]);
      const double gi = static_cast<double>(gp[c]);
      const double pi_val = 2.0 / (1.0 + std::exp(-alpha * Hi + tau)) - 1.0;
      const double Mi = (Hi < H_thres) ? 1.0 : 0.0;
      U_ewg_pos += Wi * gi + pi_val * Mi * inv_num_pixels * weighted_grad_mean;
    }
  }
  if (!std::isfinite(U_ewg_pos)) {
    return 0.0;
  }
  U_ewg_pos = std::max(U_ewg_pos, 0.0);

  // Same detector type as Frontend::initialiseBriskFeatureDetectors() (BRISK scale-space + Harris).
  std::vector<cv::KeyPoint> brisk_kps;
  {
    const BriskDetectorParams det_params{brisk_uniformity_radius, brisk_octaves,
                                         brisk_absolute_threshold, brisk_max_keypoints};
    static thread_local std::unique_ptr<BriskScaleSpaceDetector> tls_brisk_det;
    static thread_local BriskDetectorParams tls_det_params{};
    static thread_local bool tls_det_valid = false;

    cv::Mat gray_det;
    if (gray.type() == CV_8UC1 && gray.isContinuous()) {
      gray_det = gray;
    } else {
      gray_det = gray.clone();
    }

    try {
      if (!tls_det_valid || !(tls_det_params == det_params)) {
        tls_brisk_det = std::make_unique<BriskScaleSpaceDetector>(
            det_params.uniformity_radius, det_params.octaves,
            det_params.absolute_threshold, det_params.max_keypoints);
        tls_det_params = det_params;
        tls_det_valid = true;
      }
      if (tls_brisk_det) {
        tls_brisk_det->detect(gray_det, brisk_kps);
      }
    } catch (...) {
      brisk_kps.clear();
      // Avoid a stuck invalid detector on repeated failures.
      tls_brisk_det.reset();
      tls_det_valid = false;
    }
  }

  const double N_feat = static_cast<double>(brisk_kps.size());
  const double Q_f = 1.0 - std::exp(-N_feat / N0_eff);

  double Q_s = 1.0;
  if (K_cells > 1) {
    Q_s = 0.0;
    std::vector<int> cell_counts(static_cast<size_t>(K_cells), 0);
    for (const cv::KeyPoint& kp : brisk_kps) {
      const int cell_r =
          std::clamp(static_cast<int>(kp.pt.y * gr / rows), 0, gr - 1);
      const int cell_c =
          std::clamp(static_cast<int>(kp.pt.x * gc / cols), 0, gc - 1);
      const int cell_idx = cell_r * gc + cell_c;
      ++cell_counts[static_cast<size_t>(cell_idx)];
    }
    double sum_n_eps = 0.0;
    for (int nk : cell_counts) {
      sum_n_eps += static_cast<double>(nk) + kPkEps;
    }
    if (sum_n_eps > 0.0 && std::isfinite(sum_n_eps)) {
      double shannon = 0.0;
      for (int nk : cell_counts) {
        const double pk = (static_cast<double>(nk) + kPkEps) / sum_n_eps;
        if (pk > 0.0) {
          shannon -= pk * std::log(pk);
        }
      }
      const double logK = std::log(static_cast<double>(K_cells));
      if (logK > 1e-15) {
        Q_s = shannon / logK;
      }
    }
    Q_s = std::clamp(Q_s, 0.0, 1.0);
  }

  const double lambda = std::max(0.0, lambda_f);
  const double U_sfa_ewg = U_ewg_pos * (1.0 + lambda * Q_f * Q_s);

  if (!std::isfinite(U_sfa_ewg)) {
    return 0.0;
  }
  return std::max(U_sfa_ewg, 0.0);
}

}  // namespace okvis
