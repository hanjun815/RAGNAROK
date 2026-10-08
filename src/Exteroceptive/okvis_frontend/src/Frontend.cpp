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
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#include "okvis/assert_macros.hpp"
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <thread>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <utility>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <glog/logging.h>

// DBoW2
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <DBoW2/DBoW2.h>
#pragma GCC diagnostic pop
#include <DBoW2/FBrisk.hpp>

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

#include <okvis/Frontend.hpp>
#include <okvis/StereoDnnImageEnhancement.hpp>
#include <okvis/EwgImageUtility.hpp>

// okvis ceres
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/ceres/ImuError.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>
#include <okvis/cameras/EucmCamera.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

#include <okvis/internal/Network.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const double kptrad = 0.09;
//cv::Ptr<cv::CLAHE> mClahe;

namespace {

// AGCWD: Adaptive Gamma Correction with Weighting Distribution.
// Operates on 8-bit intensity images (single channel).
cv::Mat agcwdEnhanceIntensityU8(const cv::Mat& intensity_in_u8, double weighting_param) {
  if (intensity_in_u8.empty() || intensity_in_u8.total() == 0) {
    return intensity_in_u8;
  }
  if (!(weighting_param > 0.0) || !std::isfinite(weighting_param)) {
    return intensity_in_u8;
  }

  cv::Mat intensity_u8;
  if (intensity_in_u8.type() == CV_8UC1) {
    intensity_u8 = intensity_in_u8;
  } else {
    intensity_in_u8.convertTo(intensity_u8, CV_8UC1);
  }

  constexpr int kHistSize = 256;
  const int channels[] = {0};
  int histSize[] = {kHistSize};
  float range[] = {0.f, 256.f};
  const float* ranges[] = {range};
  cv::Mat pdf;
  cv::calcHist(&intensity_u8, 1, channels, cv::Mat(), pdf, 1, histSize, ranges, true, false);
  pdf.convertTo(pdf, CV_32F);
  pdf /= static_cast<float>(intensity_u8.total());

  double minPdf = 0.0, maxPdf = 0.0;
  cv::minMaxLoc(pdf, &minPdf, &maxPdf);
  if (!(maxPdf > minPdf + 1.0e-12)) {
    return intensity_u8;
  }

  // Weighting distribution function + CDF.
  cv::Mat cdf(pdf.rows, pdf.cols, pdf.type());
  for (int i = 0; i < pdf.rows; ++i) {
    const float p = pdf.at<float>(i, 0);
    float pn = float((p - minPdf) / (maxPdf - minPdf));
    if (!std::isfinite(pn) || pn < 0.f) pn = 0.f;
    const float pw = float(maxPdf) * std::pow(pn, float(weighting_param));
    pdf.at<float>(i, 0) = pw;
    cdf.at<float>(i, 0) = (i == 0) ? pw : (cdf.at<float>(i - 1, 0) + pw);
  }
  const float cdf_end = cdf.at<float>(cdf.rows - 1, 0);
  if (!(cdf_end > 1.0e-12f)) {
    return intensity_u8;
  }
  cdf /= cdf_end;

  // Build LUT.
  cv::Mat lut(1, kHistSize, CV_8U);
  for (int i = 0; i < kHistSize; ++i) {
    // Avoid 0^0 (can happen when cdf(0)==1 for a fully dark image).
    if (i == 0) {
      lut.at<uchar>(0, i) = 0;
      continue;
    }
    float c = cdf.at<float>(i, 0);
    if (!std::isfinite(c)) c = 0.0f;
    if (c < 0.0f) c = 0.0f;
    if (c > 1.0f) c = 1.0f;
    const float gamma = 1.0f - c;
    const float in = float(i) / 255.0f;
    const float out = 255.0f * std::pow(in, gamma);
    lut.at<uchar>(0, i) = cv::saturate_cast<uchar>(out);
  }

  cv::Mat intensity_out_u8;
  cv::LUT(intensity_u8, lut, intensity_out_u8);
  return intensity_out_u8;
}

// CLAHE on 8-bit intensity images.
cv::Mat claheEnhanceIntensityU8(const cv::Mat& intensity_in_u8, double clip_limit, int tile_grid_size) {
  if (intensity_in_u8.empty() || intensity_in_u8.total() == 0) {
    return intensity_in_u8;
  }
  cv::Mat intensity_u8;
  if (intensity_in_u8.type() == CV_8UC1) {
    intensity_u8 = intensity_in_u8;
  } else {
    intensity_in_u8.convertTo(intensity_u8, CV_8UC1);
  }
  const double clip = (clip_limit > 0.0 && std::isfinite(clip_limit)) ? clip_limit : 2.0;
  const int grid = (tile_grid_size > 0) ? tile_grid_size : 8;
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, cv::Size(grid, grid));
  cv::Mat out;
  clahe->apply(intensity_u8, out);
  return out;
}

// Radial vignette correction using polynomial gain:
// gain(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6, with r in [0,1].
bool vignetteCorrectImageU8(const cv::Mat& image_in,
                            double k1,
                            double k2,
                            double k3,
                            cv::Mat& image_out) {
  image_out.release();
  if (image_in.empty() || image_in.total() == 0) {
    return false;
  }
  if (!std::isfinite(k1) || !std::isfinite(k2) || !std::isfinite(k3)) {
    return false;
  }

  cv::Mat image_u8;
  if (image_in.depth() == CV_8U) {
    image_u8 = image_in;
  } else {
    image_in.convertTo(image_u8, CV_8U);
  }
  if (image_u8.channels() == 4) {
    cv::Mat bgr;
    cv::cvtColor(image_u8, bgr, cv::COLOR_BGRA2BGR);
    image_u8 = bgr;
  }
  if (image_u8.channels() != 1 && image_u8.channels() != 3) {
    return false;
  }

  const int rows = image_u8.rows;
  const int cols = image_u8.cols;
  if (rows <= 1 || cols <= 1) {
    return false;
  }

  const double cx = 0.5 * double(cols - 1);
  const double cy = 0.5 * double(rows - 1);
  const double max_dx = std::max(cx, double(cols - 1) - cx);
  const double max_dy = std::max(cy, double(rows - 1) - cy);
  const double max_r = std::sqrt(max_dx * max_dx + max_dy * max_dy);
  if (!(max_r > 1.0e-9)) {
    return false;
  }

  cv::Mat gain(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y) {
    float* gptr = gain.ptr<float>(y);
    const double dy = (double(y) - cy) / max_r;
    for (int x = 0; x < cols; ++x) {
      const double dx = (double(x) - cx) / max_r;
      double r = std::sqrt(dx * dx + dy * dy);
      if (r > 1.0) {
        r = 1.0;
      }
      const double r2 = r * r;
      const double r4 = r2 * r2;
      const double r6 = r4 * r2;
      double g = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
      if (!std::isfinite(g)) {
        g = 1.0;
      }
      if (g < 0.0) {
        g = 0.0;
      }
      gptr[x] = static_cast<float>(g);
    }
  }

  if (image_u8.channels() == 1) {
    cv::Mat img_f;
    image_u8.convertTo(img_f, CV_32FC1);
    cv::multiply(img_f, gain, img_f);
    img_f.convertTo(image_out, CV_8UC1);
    return !image_out.empty();
  }

  cv::Mat img_f;
  image_u8.convertTo(img_f, CV_32FC3);
  std::vector<cv::Mat> channels;
  cv::split(img_f, channels);
  if (channels.size() != 3) {
    return false;
  }
  for (int c = 0; c < 3; ++c) {
    cv::multiply(channels[c], gain, channels[c]);
  }
  cv::merge(channels, img_f);
  img_f.convertTo(image_out, CV_8UC3);
  return !image_out.empty();
}

// Entropy of an 8-bit intensity image based on histogram (natural log).
double entropyU8(const cv::Mat& intensity_u8) {
  if (intensity_u8.empty() || intensity_u8.total() == 0) {
    return 0.0;
  }
  cv::Mat img;
  if (intensity_u8.type() == CV_8UC1) {
    img = intensity_u8;
  } else {
    intensity_u8.convertTo(img, CV_8UC1);
  }
  constexpr int kHistSize = 256;
  const int channels[] = {0};
  int histSize[] = {kHistSize};
  float range[] = {0.f, 256.f};
  const float* ranges[] = {range};
  cv::Mat hist;
  cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
  hist.convertTo(hist, CV_32F);
  hist /= static_cast<float>(img.total());
  double ent = 0.0;
  for (int i = 0; i < kHistSize; ++i) {
    const double p = static_cast<double>(hist.at<float>(i, 0));
    if (p > 0.0) {
      ent -= p * std::log(p);
    }
  }
  return ent;
}

cv::Mat toGrayU8(const cv::Mat& image_in) {
  if (image_in.empty()) {
    return cv::Mat();
  }
  cv::Mat img_u8;
  if (image_in.depth() == CV_8U) {
    img_u8 = image_in;
  } else {
    image_in.convertTo(img_u8, CV_8U);
  }
  if (img_u8.channels() == 1) {
    return img_u8;
  }
  if (img_u8.channels() == 3) {
    cv::Mat gray;
    cv::cvtColor(img_u8, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  if (img_u8.channels() == 4) {
    cv::Mat gray;
    cv::cvtColor(img_u8, gray, cv::COLOR_BGRA2GRAY);
    return gray;
  }
  return cv::Mat();
}

cv::Mat downsampleForObjective(const cv::Mat& image_in, int max_side = 640) {
  if (image_in.empty()) {
    return image_in;
  }
  const int rows = image_in.rows;
  const int cols = image_in.cols;
  const int side = std::max(rows, cols);
  if (side <= max_side) {
    return image_in;
  }
  const double scale = double(max_side) / double(side);
  cv::Mat out;
  cv::resize(image_in, out, cv::Size(), scale, scale, cv::INTER_AREA);
  return out;
}

struct BoSfaEwgConfig {
  double Hthres = 0.125;
  double alpha = 32.0;
  double tau = 4.0;
  int local_patch = 5;
  int num_bins = 8;
  double lambda_f = 1.0;
  double N0 = 500.0;
  int grid_rows = 3;
  int grid_cols = 4;
  // Same BRISK detector settings as Frontend::initialiseBriskFeatureDetectors().
  double brisk_uniformity_radius = 40.0;
  size_t brisk_octaves = 0;
  double brisk_absolute_threshold = 200.0;
  size_t brisk_max_keypoints = 450;
  // Keep BO objective EWG input scale consistent with runtime EWG policy.
  bool resize_enable = false;
  int resize_max_side = 640;
};

struct BoObjectiveTerms {
  /// `image_utility_weight` * mean(U_SFA_EWG) over available downsampled enhanced views.
  double image_utility = 0.0;
  double total = 0.0;
};

// Forward declaration (definition appears later in this anonymous namespace).
bool enhanceImageAgcwdClaheFusion(const cv::Mat& image_in,
                                  bool agcwd_enable, double agcwd_weighting_param,
                                  bool clahe_enable, double clahe_clip_limit, int clahe_tile_grid_size,
                                  cv::Mat& image_out);

// BO objective J(agcwd_w, clahe_clip) = w_util * mean(U_SFA_EWG_left, U_SFA_EWG_right)
double evaluateBoObjective(const cv::Mat& left_input,
                           const cv::Mat& right_input,
                           bool agcwd_enable,
                           bool clahe_enable,
                           double agcwd_w,
                           double clahe_clip,
                           int clahe_grid,
                           double image_utility_weight,
                           const BoSfaEwgConfig& sfa_ewg,
                           BoObjectiveTerms* terms_out) {
  BoObjectiveTerms terms;

  cv::Mat left_work =
      (sfa_ewg.resize_enable && sfa_ewg.resize_max_side > 0)
          ? downsampleForObjective(left_input, sfa_ewg.resize_max_side)
          : left_input;
  cv::Mat right_work =
      (sfa_ewg.resize_enable && sfa_ewg.resize_max_side > 0)
          ? downsampleForObjective(right_input, sfa_ewg.resize_max_side)
          : right_input;
  if (left_work.empty()) {
    if (terms_out) {
      *terms_out = terms;
    }
    return -1.0e9;
  }

  cv::Mat left_enhanced;
  if (!enhanceImageAgcwdClaheFusion(left_work, agcwd_enable, agcwd_w, clahe_enable,
                                    clahe_clip, clahe_grid, left_enhanced)) {
    left_enhanced = left_work;
  }
  cv::Mat right_enhanced;
  if (!right_work.empty()) {
    if (!enhanceImageAgcwdClaheFusion(right_work, agcwd_enable, agcwd_w, clahe_enable,
                                      clahe_clip, clahe_grid, right_enhanced)) {
      right_enhanced = right_work;
    }
  }

  auto sfaEw = [&](const cv::Mat& img) {
    return computeImageUtilityEWG(
        img, sfa_ewg.Hthres, sfa_ewg.alpha, sfa_ewg.tau, sfa_ewg.local_patch,
        sfa_ewg.num_bins, sfa_ewg.lambda_f, sfa_ewg.N0, sfa_ewg.grid_rows,
        sfa_ewg.grid_cols, sfa_ewg.brisk_uniformity_radius, sfa_ewg.brisk_octaves,
        sfa_ewg.brisk_absolute_threshold, sfa_ewg.brisk_max_keypoints);
  };
  double util_sum = 0.0;
  int util_count = 0;
  if (!left_enhanced.empty()) {
    const double Ul = sfaEw(left_enhanced);
    if (std::isfinite(Ul)) {
      util_sum += Ul;
      ++util_count;
    }
  }
  if (!right_enhanced.empty()) {
    const double Ur = sfaEw(right_enhanced);
    if (std::isfinite(Ur)) {
      util_sum += Ur;
      ++util_count;
    }
  }
  const double util_mean = (util_count > 0) ? (util_sum / double(util_count)) : 0.0;
  terms.image_utility = image_utility_weight * util_mean;
  terms.total = terms.image_utility;

  if (terms_out) {
    *terms_out = terms;
  }
  return terms.total;
}

double normalPdf(double z) {
  constexpr double kInvSqrt2Pi = 0.3989422804014327;
  return kInvSqrt2Pi * std::exp(-0.5 * z * z);
}

double normalCdf(double z) {
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

struct BoResult {
  double agcwd = 0.5;
  double clahe = 2.0;
  double objective = -1.0e9;
  BoObjectiveTerms terms;
};

BoResult runAgcwdClaheBayesOptimization(
    const cv::Mat& left_input,
    const cv::Mat& right_input,
    bool agcwd_enable,
    bool clahe_enable,
    int clahe_grid,
    double prev_agcwd_w,
    double prev_clahe_clip,
    int initial_samples,
    int iterations,
    int candidate_grid,
    double agcwd_min,
    double agcwd_max,
    double clahe_min,
    double clahe_max,
    double image_utility_weight,
    const BoSfaEwgConfig& sfa_ewg) {
  if (agcwd_min > agcwd_max) std::swap(agcwd_min, agcwd_max);
  if (clahe_min > clahe_max) std::swap(clahe_min, clahe_max);
  const double a0 = std::clamp(prev_agcwd_w, agcwd_min, agcwd_max);
  const double c0 = std::clamp(prev_clahe_clip, clahe_min, clahe_max);

  BoResult best;
  best.agcwd = a0;
  best.clahe = c0;

  std::vector<Eigen::Vector2d> X;
  std::vector<double> y;
  std::vector<BoObjectiveTerms> all_terms;
  X.reserve(size_t(std::max(1, initial_samples) + std::max(0, iterations) + 1));
  y.reserve(X.capacity());
  all_terms.reserve(X.capacity());

  auto eval_sample = [&](double a, double c) {
    BoObjectiveTerms t;
    const double obj = evaluateBoObjective(left_input, right_input, agcwd_enable, clahe_enable,
                                           a, c, clahe_grid, image_utility_weight, sfa_ewg, &t);
    X.emplace_back(a, c);
    y.push_back(obj);
    all_terms.push_back(t);
    if (obj > best.objective) {
      best.objective = obj;
      best.agcwd = a;
      best.clahe = c;
      best.terms = t;
    }
  };

  eval_sample(a0, c0); // Always include current parameter set.

  std::mt19937 rng(static_cast<unsigned int>(std::chrono::high_resolution_clock::now()
                                               .time_since_epoch().count()));
  std::uniform_real_distribution<double> randA(agcwd_min, agcwd_max);
  std::uniform_real_distribution<double> randC(clahe_min, clahe_max);

  const int n_init = std::max(1, initial_samples);
  for (int i = 0; i < n_init; ++i) {
    eval_sample(randA(rng), randC(rng));
  }

  const double l1 = std::max(1.0e-6, 0.25 * (agcwd_max - agcwd_min));
  const double l2 = std::max(1.0e-6, 0.25 * (clahe_max - clahe_min));
  constexpr double sigma_f = 1.0;
  constexpr double noise = 1.0e-6;
  constexpr double xi = 1.0e-3;

  auto kernel = [&](const Eigen::Vector2d& xa, const Eigen::Vector2d& xb) {
    const double da = (xa[0] - xb[0]) / l1;
    const double dc = (xa[1] - xb[1]) / l2;
    return sigma_f * sigma_f * std::exp(-0.5 * (da * da + dc * dc));
  };

  const int n_iter = std::max(0, iterations);
  const int grid_n = std::max(3, candidate_grid);
  for (int it = 0; it < n_iter; ++it) {
    const int n = int(X.size());
    Eigen::MatrixXd K(n, n);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        K(r, c) = kernel(X[size_t(r)], X[size_t(c)]);
      }
      K(r, r) += noise;
    }

    Eigen::LDLT<Eigen::MatrixXd> ldlt(K);
    if (ldlt.info() != Eigen::Success) {
      for (int d = 0; d < n; ++d) {
        K(d, d) += 1.0e-4;
      }
      ldlt.compute(K);
    }
    if (ldlt.info() != Eigen::Success) {
      eval_sample(randA(rng), randC(rng));
      continue;
    }

    Eigen::VectorXd yv(n);
    for (int i = 0; i < n; ++i) {
      yv(i) = y[size_t(i)];
    }
    const Eigen::VectorXd alpha = ldlt.solve(yv);
    const double y_best = best.objective;

    double best_ei = -1.0;
    Eigen::Vector2d x_next(randA(rng), randC(rng));
    const double step_a = (agcwd_max - agcwd_min) / double(std::max(1, grid_n - 1));
    const double step_c = (clahe_max - clahe_min) / double(std::max(1, grid_n - 1));
    std::uniform_real_distribution<double> jitter_a(-0.25 * step_a, 0.25 * step_a);
    std::uniform_real_distribution<double> jitter_c(-0.25 * step_c, 0.25 * step_c);
    for (int ia = 0; ia < grid_n; ++ia) {
      for (int ic = 0; ic < grid_n; ++ic) {
        double a = agcwd_min + step_a * double(ia) + jitter_a(rng);
        double c = clahe_min + step_c * double(ic) + jitter_c(rng);
        a = std::clamp(a, agcwd_min, agcwd_max);
        c = std::clamp(c, clahe_min, clahe_max);
        const Eigen::Vector2d x(a, c);
        Eigen::VectorXd k(n);
        for (int i = 0; i < n; ++i) {
          k(i) = kernel(x, X[size_t(i)]);
        }
        const double mu = k.dot(alpha);
        const Eigen::VectorXd v = ldlt.solve(k);
        const double var = std::max(1.0e-12, sigma_f * sigma_f + noise - k.dot(v));
        const double sigma = std::sqrt(var);
        const double z = (mu - y_best - xi) / sigma;
        const double ei = (mu - y_best - xi) * normalCdf(z) + sigma * normalPdf(z);
        if (ei > best_ei) {
          best_ei = ei;
          x_next = x;
        }
      }
    }
    eval_sample(x_next[0], x_next[1]);
  }

  return best;
}

// Paper-style weighted fusion coefficient u.
double fusionWeightU(double m1, double m2, double q1, double q2) {
  constexpr double kEps = 1.0e-12;
  const double denom_m = m1 + m2;
  const double denom_q = q1 + q2;
  const double term_m = (denom_m > kEps) ? (m1 / (2.0 * denom_m)) : 0.25;
  const double term_q = (denom_q > kEps) ? (q1 / (2.0 * denom_q)) : 0.25;
  double u = term_m + term_q;
  if (u < 0.0) u = 0.0;
  if (u > 1.0) u = 1.0;
  return u;
}

// Enhancement pipeline:
// - If only AGCWD: use AGCWD(V)
// - If only CLAHE: use CLAHE(V)
// - If both: fuse AGCWD(V) and CLAHE(V) using u(m,entropy)
//
// Gating policy: this function unconditionally applies the requested
// enhancement(s). Whether to call it is decided by the caller (e.g. the
// EWG-image-utility based gate in `detectAndDescribe`).
bool enhanceImageAgcwdClaheFusion(const cv::Mat& image_in,
                                 bool agcwd_enable, double agcwd_weighting_param,
                                 bool clahe_enable, double clahe_clip_limit, int clahe_tile_grid_size,
                                 cv::Mat& image_out) {
  image_out.release();
  if (image_in.empty()) {
    return false;
  }
  if (!agcwd_enable && !clahe_enable) {
    return false;
  }

  // Convert to 8-bit if needed (feature extractors expect 8-bit images).
  cv::Mat image_u8;
  if (image_in.depth() == CV_8U) {
    image_u8 = image_in;
  } else {
    image_in.convertTo(image_u8, CV_8U);
  }

  cv::Mat intensity_u8;
  cv::Mat hsv;
  std::vector<cv::Mat> hsv_channels;

  if (image_u8.channels() == 1) {
    intensity_u8 = image_u8;
  } else if (image_u8.channels() == 3) {
    cv::cvtColor(image_u8, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, hsv_channels);
    intensity_u8 = hsv_channels[2];
  } else if (image_u8.channels() == 4) {
    cv::Mat bgr;
    cv::cvtColor(image_u8, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, hsv_channels);
    intensity_u8 = hsv_channels[2];
    image_u8 = bgr;
  } else {
    return false;
  }

  if (intensity_u8.empty() || intensity_u8.total() == 0) {
    return false;
  }

  const bool do_agcwd = agcwd_enable && (agcwd_weighting_param > 0.0) && std::isfinite(agcwd_weighting_param);
  const bool do_clahe = clahe_enable;
  if (!do_agcwd && !do_clahe) {
    return false;
  }

  cv::Mat v_agcwd;
  cv::Mat v_clahe;
  if (do_agcwd) {
    v_agcwd = agcwdEnhanceIntensityU8(intensity_u8, agcwd_weighting_param);
  }
  if (do_clahe) {
    v_clahe = claheEnhanceIntensityU8(intensity_u8, clahe_clip_limit, clahe_tile_grid_size);
  }

  cv::Mat v_out;
  if (do_agcwd && do_clahe) {
    const double m1 = cv::mean(v_agcwd)[0];
    const double m2 = cv::mean(v_clahe)[0];
    const double q1 = entropyU8(v_agcwd);
    const double q2 = entropyU8(v_clahe);
    const double u = fusionWeightU(m1, m2, q1, q2);
    const double v = 1.0 - u;
    cv::addWeighted(v_agcwd, u, v_clahe, v, 0.0, v_out);
  } else if (do_agcwd) {
    v_out = v_agcwd;
  } else { // do_clahe
    v_out = v_clahe;
  }

  if (v_out.empty()) {
    return false;
  }

  if (image_u8.channels() == 1) {
    image_out = v_out;
    return true;
  }

  // Color: replace V channel and convert back.
  if (hsv_channels.size() != 3) {
    return false;
  }
  hsv_channels[2] = v_out;
  cv::merge(hsv_channels, hsv);
  cv::Mat bgr_out;
  cv::cvtColor(hsv, bgr_out, cv::COLOR_HSV2BGR);
  image_out = bgr_out;
  return !image_out.empty();
}

} // namespace

/// \brief Opaque stuct to use DBoW for loop closure.
class Frontend::DBoW {
 public:
  /// \brief Constructor with Vocabulary directory.
  /// \param dBowVocDir Vocabulary directory.
  DBoW(const std::string& dBowVocDir)
      : vocabulary(dBowVocDir+"/small_voc.yml.gz"),
        // false = do not use direct index.
        database(vocabulary, false, 0)
  {
  }
  /// \brief Constructor with Vocabulary.
  /// \param dBowVoc Vocabulary.
  DBoW(const DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>& dBowVoc)
      : vocabulary(dBowVoc),
      // false = do not use direct index.
      database(vocabulary, false, 0)
  {
  }

  DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> vocabulary; ///< BRISK Voc.
  DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> database; ///< BRISK Database.
  std::vector<uint64> poseIds; ///< The multiframe IDs corresponding to the dBow ones.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Constructor.
Frontend::Frontend(size_t numCameras, std::string dBowVocDir)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(40.0),
      briskDetectionAbsoluteThreshold_(200.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      keyframeInsertionOverlapThreshold_(0.55f),
      dBow_(new DBoW(dBowVocDir))
{
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
  }
  trackingLost_ = false;
  initialiseBriskFeatureDetectors();

#ifdef OKVIS_USE_NN
  // Deserialize the ScriptModule from a file using torch::jit::load().
  networks_.resize(numCameras);
  for (size_t i = 0; i < numCameras_; ++i) {
#ifdef OKVIS_USE_GPU
#ifdef OKVIS_USE_MPS
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCPU)));
    networks_[i]->to(torch::kMPS);
#else
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCUDA)));
    networks_[i]->to(torch::kCUDA);
#endif
#else
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCPU)));
    networks_[i]->to(torch::kCPU);
#endif
  }

#endif
}

Frontend::~Frontend() {
  std::thread bo_thread;
  {
    std::lock_guard<std::mutex> lock(agcwdMutex_);
    agcwdClaheBoStop_ = true;
    if (agcwdClaheBoThread_.joinable()) {
      bo_thread = std::move(agcwdClaheBoThread_);
    }
  }
  if (bo_thread.joinable()) {
    bo_thread.join();
  }
  endCnnThreads();
}

void Frontend::setAgcwdPreprocessing(bool enable, double weighting_param) {
  std::lock_guard<std::mutex> lock(agcwdMutex_);
  ++agcwdClaheBoJobSeq_;
  agcwdEnable_ = enable;
  agcwdWeightingParam_ = weighting_param;
  agcwdClaheBoCurrentAgcwd_ = weighting_param;
  if (!agcwdClaheBoCurrentInitialized_) {
    agcwdClaheBoCurrentClahe_ = claheClipLimit_;
    agcwdClaheBoCurrentInitialized_ = true;
  }
}

void Frontend::setClahePreprocessing(bool enable, double clip_limit, int tile_grid_size) {
  std::lock_guard<std::mutex> lock(agcwdMutex_);
  ++agcwdClaheBoJobSeq_;
  claheEnable_ = enable;
  claheClipLimit_ = clip_limit;
  claheTileGridSize_ = tile_grid_size;
  agcwdClaheBoCurrentClahe_ = clip_limit;
  if (!agcwdClaheBoCurrentInitialized_) {
    agcwdClaheBoCurrentAgcwd_ = agcwdWeightingParam_;
    agcwdClaheBoCurrentInitialized_ = true;
  }
}

void Frontend::setEwgImageUtilityConfig(double Hthres,
                                        double alpha,
                                        double tau,
                                        int local_patch,
                                        int num_bins,
                                        double lambda_f,
                                        double N0,
                                        int grid_rows,
                                        int grid_cols,
                                        bool ewg_resize_enable,
                                        int ewg_resize_max_side,
                                        bool fallback_use_preprocessed) {
  std::lock_guard<std::mutex> lock(agcwdMutex_);
  ewgHthres_ = Hthres;
  ewgAlpha_ = alpha;
  ewgTau_ = tau;
  ewgLocalPatch_ = std::max(3, local_patch);
  ewgNumBins_ = std::max(2, num_bins);
  ewgLambdaF_ = std::max(0.0, lambda_f);
  ewgN0_ = std::max(1.0e-12, N0);
  ewgGridRows_ = std::max(1, grid_rows);
  ewgGridCols_ = std::max(1, grid_cols);
  ewgResizeEnable_ = ewg_resize_enable;
  ewgResizeMaxSide_ = std::max(32, ewg_resize_max_side);
  ewgFallbackUsePreprocessed_ = fallback_use_preprocessed;
}

bool Frontend::getImageUtility(uint64_t frame_id,
                               std::vector<double>& utilities) const {
  std::lock_guard<std::mutex> lock(imageUtilityMutex_);
  auto it = imageUtilityByFrameId_.find(frame_id);
  if (it == imageUtilityByFrameId_.end()) {
    utilities.assign(numCameras_, std::numeric_limits<double>::quiet_NaN());
    return false;
  }
  utilities = it->second;
  if (utilities.size() < numCameras_) {
    utilities.resize(numCameras_, std::numeric_limits<double>::quiet_NaN());
  }
  bool any_recorded = false;
  for (double u : utilities) {
    if (std::isfinite(u)) {
      any_recorded = true;
      break;
    }
  }
  return any_recorded;
}

void Frontend::prepareImageUtilityFrameId(uint64_t frame_id) {
  std::lock_guard<std::mutex> lock(imageUtilityMutex_);
  // A dropped frame may leave its provisional entry behind. Let the first
  // present camera create a fresh NaN-filled vector; missing cameras stay NaN.
  // Erasing also avoids allocating entries when EWG is disabled.
  imageUtilityByFrameId_.erase(frame_id);
}

void Frontend::pruneImageUtilityCacheLocked(uint64_t frame_id) {
  constexpr uint64_t kKeepFrames = 5000;
  if (frame_id <= kKeepFrames) {
    return;
  }
  const uint64_t prune_before = frame_id - kKeepFrames;
  for (auto it = imageUtilityByFrameId_.begin(); it != imageUtilityByFrameId_.end();) {
    if (it->first < prune_before) {
      it = imageUtilityByFrameId_.erase(it);
    } else {
      ++it;
    }
  }
}

void Frontend::reassignImageUtilityFrameId(uint64_t previous_frame_id, uint64_t frame_id) {
  std::lock_guard<std::mutex> lock(imageUtilityMutex_);
  if (previous_frame_id != frame_id) {
    auto it = imageUtilityByFrameId_.find(previous_frame_id);
    if (it == imageUtilityByFrameId_.end()) {
      return;
    }
    // Detection precedes addStates(), so its provisional id is normally zero.
    // Move the complete per-camera vector after all detector threads have joined.
    auto utilities = std::move(it->second);
    imageUtilityByFrameId_.erase(it);
    imageUtilityByFrameId_[frame_id] = std::move(utilities);
  }
  pruneImageUtilityCacheLocked(frame_id);
}

bool Frontend::getImageUtilitySum(uint64_t frame_id,
                                  double& total,
                                  int& num_recorded) const {
  total = 0.0;
  num_recorded = 0;
  std::vector<double> utils;
  if (!getImageUtility(frame_id, utils)) {
    return false;
  }
  for (double u : utils) {
    if (std::isfinite(u)) {
      total += u;
      ++num_recorded;
    }
  }
  return num_recorded > 0;
}

void Frontend::setVignetteCorrection(bool enable, double k1, double k2, double k3) {
  std::lock_guard<std::mutex> lock(agcwdMutex_);
  vignetteCorrectionEnable_ = enable;
  vignetteCorrectionK1_ = k1;
  vignetteCorrectionK2_ = k2;
  vignetteCorrectionK3_ = k3;
}

void Frontend::setAgcwdClaheBayesOptConfig(bool enable,
                                           double utility_discrepancy_threshold,
                                           int initial_samples,
                                           int iterations,
                                           int candidate_grid,
                                           double agcwd_min,
                                           double agcwd_max,
                                           double clahe_min,
                                           double clahe_max,
                                           double image_utility_weight) {
  std::lock_guard<std::mutex> lock(agcwdMutex_);
  ++agcwdClaheBoJobSeq_;
  agcwdClaheBoEnable_ = enable;
  agcwdClaheBoUtilityDiscrepancyThreshold_ = std::max(0.0, utility_discrepancy_threshold);
  agcwdClaheBoInitialSamples_ = std::max(1, initial_samples);
  agcwdClaheBoIterations_ = std::max(0, iterations);
  agcwdClaheBoCandidateGrid_ = std::max(3, candidate_grid);
  agcwdClaheBoAgcwdMin_ = agcwd_min;
  agcwdClaheBoAgcwdMax_ = agcwd_max;
  agcwdClaheBoClaheMin_ = clahe_min;
  agcwdClaheBoClaheMax_ = clahe_max;
  agcwdClaheBoImageUtilityWeight_ = std::max(0.0, image_utility_weight);
  if (agcwdClaheBoAgcwdMin_ > agcwdClaheBoAgcwdMax_) {
    std::swap(agcwdClaheBoAgcwdMin_, agcwdClaheBoAgcwdMax_);
  }
  if (agcwdClaheBoClaheMin_ > agcwdClaheBoClaheMax_) {
    std::swap(agcwdClaheBoClaheMin_, agcwdClaheBoClaheMax_);
  }
  if (!agcwdClaheBoCurrentInitialized_) {
    agcwdClaheBoCurrentAgcwd_ = agcwdWeightingParam_;
    agcwdClaheBoCurrentClahe_ = claheClipLimit_;
    agcwdClaheBoCurrentInitialized_ = true;
  }
  agcwdClaheBoCurrentAgcwd_ =
      std::clamp(agcwdClaheBoCurrentAgcwd_, agcwdClaheBoAgcwdMin_, agcwdClaheBoAgcwdMax_);
  agcwdClaheBoCurrentClahe_ =
      std::clamp(agcwdClaheBoCurrentClahe_, agcwdClaheBoClaheMin_, agcwdClaheBoClaheMax_);
}

bool Frontend::getLastAgcwdImage(size_t cameraIndex, cv::Mat& outImage) const {
  std::lock_guard<std::mutex> lock(agcwdImagesMutex_);
  if (cameraIndex >= agcwdImages_.size()) {
    return false;
  }
  if (agcwdImages_[cameraIndex].empty()) {
    return false;
  }
  outImage = agcwdImages_[cameraIndex];
  return true;
}

bool Frontend::loadComponent(std::string filename,
                             const ImuParameters &imuParameters,
                             const cameras::NCameraSystem &nCameraSystem,
                             bool componentFixed)
{
  // create component
  components_.emplace_back(Component(imuParameters, nCameraSystem));
  componentsFixed_.push_back(componentFixed);

  // load it
  if (!components_.back().load(filename)) {
    return false;
  }

  // create component DBoW
  componentDBows_.emplace_back(std::unique_ptr<DBoW>(new DBoW(dBow_->vocabulary)));

  // fill component DBoW
  for (const auto &multiFrame : components_.back().multiFrames_) {
    // first get features
    std::vector<std::vector<uchar>> features(multiFrame.second->numKeypoints());
    int offset = 0;
    for (size_t im = 0; im < numCameras_; ++im) {
      for (size_t k = 0; k < multiFrame.second->numKeypoints(im); ++k) {
        features.at(k + offset).resize(48); // TODO: get 48 from feature
        memcpy(features.at(k + offset).data(),
               multiFrame.second->keypointDescriptor(im, k),
               48 * sizeof(uchar));

      }
      offset += multiFrame.second->numKeypoints(im);
    }
    // ... and add:
    componentDBows_.back()->database.add(features);
    componentDBows_.back()->poseIds.push_back(multiFrame.second->id());
  }

  return true;
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint>* keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_,
                        "Camera index exceeds number of cameras.")
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  // Optional preprocessing for low-light robustness.
  // Order: vignette correction -> AGCWD / CLAHE enhancement.
  bool vignette_enable = false;
  double vignette_k1 = 0.0;
  double vignette_k2 = 0.0;
  double vignette_k3 = 0.0;
  bool bo_enable = false;
  double bo_utility_discrepancy_threshold = 0.2;
  int bo_initial_samples = 4;
  int bo_iterations = 6;
  int bo_candidate_grid = 9;
  double bo_agcwd_min = 0.2;
  double bo_agcwd_max = 1.2;
  double bo_clahe_min = 1.0;
  double bo_clahe_max = 8.0;
  double bo_image_utility_weight = 1.0;
  bool agcwd_enable = false;
  double agcwd_w = 0.5;
  bool clahe_enable = false;
  double clahe_clip = 2.0;
  int clahe_grid = 8;
  double ewg_Hthres = 0.125;
  double ewg_alpha = 32.0;
  double ewg_tau = 4.0;
  int ewg_local_patch = 5;
  int ewg_num_bins = 8;
  double ewg_lambda_f = 1.0;
  double ewg_N0 = 500.0;
  int ewg_grid_rows = 3;
  int ewg_grid_cols = 4;
  bool ewg_resize_enable = false;
  int ewg_resize_max_side = 640;
  bool ewg_fallback_use_preprocessed = true;
  {
    std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
    vignette_enable = vignetteCorrectionEnable_;
    vignette_k1 = vignetteCorrectionK1_;
    vignette_k2 = vignetteCorrectionK2_;
    vignette_k3 = vignetteCorrectionK3_;
    bo_enable = agcwdClaheBoEnable_;
    bo_utility_discrepancy_threshold = agcwdClaheBoUtilityDiscrepancyThreshold_;
    bo_initial_samples = agcwdClaheBoInitialSamples_;
    bo_iterations = agcwdClaheBoIterations_;
    bo_candidate_grid = agcwdClaheBoCandidateGrid_;
    bo_agcwd_min = agcwdClaheBoAgcwdMin_;
    bo_agcwd_max = agcwdClaheBoAgcwdMax_;
    bo_clahe_min = agcwdClaheBoClaheMin_;
    bo_clahe_max = agcwdClaheBoClaheMax_;
    bo_image_utility_weight = agcwdClaheBoImageUtilityWeight_;
    agcwd_enable = agcwdEnable_;
    agcwd_w = bo_enable ? agcwdClaheBoCurrentAgcwd_ : agcwdWeightingParam_;
    clahe_enable = claheEnable_;
    clahe_clip = bo_enable ? agcwdClaheBoCurrentClahe_ : claheClipLimit_;
    clahe_grid = claheTileGridSize_;
    ewg_Hthres = ewgHthres_;
    ewg_alpha = ewgAlpha_;
    ewg_tau = ewgTau_;
    ewg_local_patch = ewgLocalPatch_;
    ewg_num_bins = ewgNumBins_;
    ewg_lambda_f = ewgLambdaF_;
    ewg_N0 = ewgN0_;
    ewg_grid_rows = ewgGridRows_;
    ewg_grid_cols = ewgGridCols_;
    ewg_resize_enable = ewgResizeEnable_;
    ewg_resize_max_side = ewgResizeMaxSide_;
    ewg_fallback_use_preprocessed = ewgFallbackUsePreprocessed_;
  }

  // BRISK detector settings (same as initialiseBriskFeatureDetectors()).
  const double brisk_ur = briskDetectionThreshold_;
  const size_t brisk_oct = briskDetectionOctaves_;
  const double brisk_abs = briskDetectionAbsoluteThreshold_;
  const size_t brisk_max = briskDetectionMaximumKeypoints_;

  // Raw utility is needed by the raw-image fallback or camera 0's BO history.
  // Otherwise the fallback utility computed on the processed image below is used
  // directly; a provisional raw score would only be overwritten.
  const bool raw_utility_required =
      !ewg_fallback_use_preprocessed
      || (cameraIndex == 0 && bo_enable && (agcwd_enable || clahe_enable));

  // Compute per-camera SFA-EWG on the raw image (optionally resized for speed) and cache it.
  double ewg_utility = std::numeric_limits<double>::quiet_NaN();
  if (raw_utility_required) {
    const cv::Mat raw_for_ewg = frameOut->image(cameraIndex);
    if (!raw_for_ewg.empty()) {
      const cv::Mat ewg_input =
          (ewg_resize_enable && ewg_resize_max_side > 0)
              ? downsampleForObjective(raw_for_ewg, ewg_resize_max_side)
              : raw_for_ewg;
      ewg_utility = computeImageUtilityEWG(ewg_input, ewg_Hthres, ewg_alpha, ewg_tau,
                                           ewg_local_patch, ewg_num_bins, ewg_lambda_f,
                                           ewg_N0, ewg_grid_rows, ewg_grid_cols, brisk_ur,
                                           brisk_oct, brisk_abs, brisk_max);
      // Store under the current (possibly provisional) frame id. ThreadedSlam
      // reassigns this cache entry after addStates() assigns the graph id.
      const uint64_t fid = frameOut->id();
      std::lock_guard<std::mutex> lockUtil(imageUtilityMutex_);
      auto& vec = imageUtilityByFrameId_[fid];
      if (vec.size() < numCameras_) {
        vec.resize(numCameras_, std::numeric_limits<double>::quiet_NaN());
      }
      if (cameraIndex < vec.size()) {
        vec[cameraIndex] = ewg_utility;
      }
      // Coarse pruning so the map cannot grow unbounded if some frames never
      // enter dataAssociationAndInitialization. Keep only the last ~5000 ids.
      pruneImageUtilityCacheLocked(fid);
    }
  }

  // Trigger-based asynchronous BO (camera 0 only, legacy AGCWD/CLAHE mode).
  // The current frame keeps using the last accepted parameters; the BO result
  // is applied to subsequent frames when the background job finishes.
  // Raw-image utility discrepancy trigger (symmetric relative difference):
  //   |U_real,t - U_real,t-1| / (|U_real,t-1| + |U_real,t|) > threshold.
  // U_real,t is raw-image SFA-EWG on camera 0 and is stored for the next frame.
  if (cameraIndex == 0 && bo_enable && (agcwd_enable || clahe_enable)) {
    const cv::Mat left_raw = frameOut->image(0);
    if (!left_raw.empty()) {
      double U_real_t = ewg_utility;
      if (!std::isfinite(U_real_t)) {
        const cv::Mat u_left_in =
            (ewg_resize_enable && ewg_resize_max_side > 0)
                ? downsampleForObjective(left_raw, ewg_resize_max_side)
                : left_raw;
        U_real_t = computeImageUtilityEWG(u_left_in, ewg_Hthres, ewg_alpha, ewg_tau,
                                          ewg_local_patch, ewg_num_bins, ewg_lambda_f,
                                          ewg_N0, ewg_grid_rows, ewg_grid_cols, brisk_ur,
                                          brisk_oct, brisk_abs, brisk_max);
      }

      bool has_prev_u = false;
      double prev_u = 0.0;
      {
        std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
        has_prev_u = agcwdClaheBoHasPrevRealUtility_;
        if (has_prev_u) {
          prev_u = agcwdClaheBoPrevRealUtility_;
        }
      }

      const double utility_abs_sum = std::abs(prev_u) + std::abs(U_real_t);
      const double utility_discrepancy =
          (has_prev_u && std::isfinite(U_real_t) && std::isfinite(prev_u) && utility_abs_sum > 1.0e-12)
              ? (std::abs(U_real_t - prev_u) / utility_abs_sum)
              : 0.0;
      const bool trigger_reopt = has_prev_u && std::isfinite(U_real_t)
                                 && std::isfinite(prev_u) && utility_abs_sum > 1.0e-12 &&
                                 (utility_discrepancy > bo_utility_discrepancy_threshold);
      if (trigger_reopt) {
        cv::Mat left_for_obj = left_raw;
        cv::Mat right_for_obj;
        if (frameOut->numFrames() > 1) {
          right_for_obj = frameOut->image(1);
        }
        if (vignette_enable) {
          cv::Mat left_v;
          if (vignetteCorrectImageU8(left_for_obj, vignette_k1, vignette_k2, vignette_k3, left_v)
              && !left_v.empty()) {
            left_for_obj = left_v;
          }
          if (!right_for_obj.empty()) {
            cv::Mat right_v;
            if (vignetteCorrectImageU8(right_for_obj, vignette_k1, vignette_k2, vignette_k3, right_v)
                && !right_v.empty()) {
              right_for_obj = right_v;
            }
          }
        }

        const cv::Mat left_job = left_for_obj.clone();
        const cv::Mat right_job = right_for_obj.empty() ? cv::Mat() : right_for_obj.clone();
        const auto frame_id = frameOut->id();
        const BoSfaEwgConfig bo_sfw{ewg_Hthres,
                                    ewg_alpha,
                                    ewg_tau,
                                    ewg_local_patch,
                                    ewg_num_bins,
                                    ewg_lambda_f,
                                    ewg_N0,
                                    ewg_grid_rows,
                                    ewg_grid_cols,
                                    brisk_ur,
                                    brisk_oct,
                                    brisk_abs,
                                    brisk_max,
                                    ewg_resize_enable,
                                    ewg_resize_max_side};

        bool bo_launched = false;
        bool bo_already_running = false;
        uint64_t bo_job_seq = 0;
        std::thread previous_bo_thread;
        if (!left_job.empty()) {
          bool should_start_new_bo = false;
          {
            std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
            if (agcwdClaheBoRunning_) {
              bo_already_running = true;
            } else if (!agcwdClaheBoStop_) {
              // Move out any finished joinable thread handle and join it *outside*
              // agcwdMutex_ to avoid lock-order deadlocks with the worker epilogue.
              if (agcwdClaheBoThread_.joinable()) {
                previous_bo_thread = std::move(agcwdClaheBoThread_);
              }
              agcwdClaheBoRunning_ = true;
              bo_job_seq = ++agcwdClaheBoJobSeq_;
              should_start_new_bo = true;
            }
          }

          if (previous_bo_thread.joinable()) {
            previous_bo_thread.join();
          }

          if (should_start_new_bo) {
            std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
            // Configuration may have changed while joining the old thread.
            if (!agcwdClaheBoStop_ && agcwdClaheBoRunning_ &&
                bo_job_seq == agcwdClaheBoJobSeq_) {
              agcwdClaheBoThread_ = std::thread(
                  [this, left_job, right_job,
                   agcwd_enable, clahe_enable, clahe_grid, agcwd_w, clahe_clip,
                   bo_initial_samples, bo_iterations, bo_candidate_grid,
                   bo_agcwd_min, bo_agcwd_max, bo_clahe_min, bo_clahe_max,
                   bo_image_utility_weight, bo_sfw,
                   bo_job_seq, frame_id, utility_discrepancy,
                   bo_utility_discrepancy_threshold, U_real_t, prev_u]() {
                    const BoResult bo = runAgcwdClaheBayesOptimization(
                        left_job, right_job,
                        agcwd_enable, clahe_enable, clahe_grid,
                        agcwd_w, clahe_clip, bo_initial_samples, bo_iterations,
                        bo_candidate_grid, bo_agcwd_min, bo_agcwd_max,
                        bo_clahe_min, bo_clahe_max,
                        bo_image_utility_weight, bo_sfw);

                    bool accepted = false;
                    bool stopped = false;
                    bool stale = false;
                    {
                      std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
                      stopped = agcwdClaheBoStop_;
                      stale = (bo_job_seq != agcwdClaheBoJobSeq_);
                      if (!stopped && !stale && agcwdClaheBoEnable_) {
                        agcwdClaheBoCurrentAgcwd_ = bo.agcwd;
                        agcwdClaheBoCurrentClahe_ = bo.clahe;
                        accepted = true;
                      }
                      agcwdClaheBoRunning_ = false;
                    }

                    if (accepted) {
                      LOG(INFO) << "[Enhancement BO][completed] mode=async"
                                << " frame_id=" << frame_id
                                << " job_seq=" << bo_job_seq
                                << " ewg_utility_discrepancy(relative)=" << utility_discrepancy
                                << " threshold=" << bo_utility_discrepancy_threshold
                                << " U_real_t=" << U_real_t
                                << " U_real_t_minus_1=" << prev_u
                                << " agcwd=" << bo.agcwd
                                << " clahe_clip=" << bo.clahe
                                << " J=" << bo.objective
                                << " image_utility=" << bo.terms.image_utility;
                    } else {
                      VLOG(1) << "[Enhancement BO][discarded] mode=async"
                              << " frame_id=" << frame_id
                              << " job_seq=" << bo_job_seq
                              << " stopped=" << stopped
                              << " stale=" << stale;
                    }
                  });
              bo_launched = true;
            } else {
              agcwdClaheBoRunning_ = false;
            }
          }
        }

        if (bo_launched) {
          LOG(INFO) << "[Enhancement BO][triggered] mode=async"
                    << " frame_id=" << frame_id
                    << " job_seq=" << bo_job_seq
                    << " ewg_utility_discrepancy(relative)=" << utility_discrepancy
                    << " threshold=" << bo_utility_discrepancy_threshold
                    << " U_real_t=" << U_real_t
                    << " U_real_t_minus_1=" << prev_u
                    << " current_agcwd=" << agcwd_w
                    << " current_clahe_clip=" << clahe_clip;
        } else if (bo_already_running) {
          VLOG(1) << "[Enhancement BO][skipped] reason=already_running"
                  << " frame_id=" << frame_id
                  << " ewg_utility_discrepancy(relative)=" << utility_discrepancy
                  << " threshold=" << bo_utility_discrepancy_threshold;
        }
      }

      {
        std::lock_guard<std::mutex> lockAgcwd(agcwdMutex_);
        if (std::isfinite(U_real_t)) {
          agcwdClaheBoPrevRealUtility_ = U_real_t;
          agcwdClaheBoHasPrevRealUtility_ = true;
        }
      }
    }
  }

  const bool do_enhancement = agcwd_enable || clahe_enable;
  const bool do_preprocessing = vignette_enable || do_enhancement;
  cv::Mat preprocess_img_orig;
  bool preprocessed_image_applied = false;
  if (do_preprocessing) {
    // Clear debug image for this camera to avoid publishing stale data when preprocessing cannot be applied.
    {
      std::lock_guard<std::mutex> lockImgs(agcwdImagesMutex_);
      if (agcwdImages_.size() != numCameras_) {
        agcwdImages_.resize(numCameras_);
      }
      agcwdImages_[cameraIndex] = cv::Mat();
    }

    preprocess_img_orig = frameOut->image(cameraIndex); // shallow copy (keeps original alive)
    cv::Mat preprocess_img = preprocess_img_orig;
    if (!preprocess_img.empty()) {
      if (vignette_enable) {
        cv::Mat vignette_out;
        if (vignetteCorrectImageU8(preprocess_img, vignette_k1, vignette_k2,
                                   vignette_k3, vignette_out)) {
          preprocess_img = vignette_out;
          frameOut->setImage(cameraIndex, preprocess_img);
          preprocessed_image_applied = true;
          std::lock_guard<std::mutex> lockImgs(agcwdImagesMutex_);
          if (agcwdImages_.size() != numCameras_) {
            agcwdImages_.resize(numCameras_);
          }
          agcwdImages_[cameraIndex] = preprocess_img;
        }
      }

      if (do_enhancement) {
        cv::Mat img_out;
        bool enhanced = false;
        enhanced = enhanceImageAgcwdClaheFusion(preprocess_img,
                                                agcwd_enable, agcwd_w,
                                                clahe_enable, clahe_clip, clahe_grid,
                                                img_out);
        if (enhanced) {
          // Store for debug/RViz visualisation.
          {
            std::lock_guard<std::mutex> lockImgs(agcwdImagesMutex_);
            if (agcwdImages_.size() != numCameras_) {
              agcwdImages_.resize(numCameras_);
            }
            agcwdImages_[cameraIndex] = img_out;
          }
          frameOut->setImage(cameraIndex, img_out);
          preprocessed_image_applied = true;
        }
      }
    }
  }

  // Fallback utility source:
  //  - false: use the raw-image utility already computed above.
  //  - true : recompute the utility on the current preprocessed image and overwrite the cache.
  if (ewg_fallback_use_preprocessed) {
    const cv::Mat fallback_img = frameOut->image(cameraIndex);
    if (!fallback_img.empty()) {
      const cv::Mat fallback_input =
          (ewg_resize_enable && ewg_resize_max_side > 0)
              ? downsampleForObjective(fallback_img, ewg_resize_max_side)
              : fallback_img;
      const double fallback_utility = computeImageUtilityEWG(
          fallback_input, ewg_Hthres, ewg_alpha, ewg_tau, ewg_local_patch, ewg_num_bins,
          ewg_lambda_f, ewg_N0, ewg_grid_rows, ewg_grid_cols, brisk_ur, brisk_oct,
          brisk_abs, brisk_max);
      const uint64_t fid = frameOut->id();
      std::lock_guard<std::mutex> lockUtil(imageUtilityMutex_);
      auto& vec = imageUtilityByFrameId_[fid];
      if (vec.size() < numCameras_) {
        vec.resize(numCameras_, std::numeric_limits<double>::quiet_NaN());
      }
      if (cameraIndex < vec.size()) {
        vec[cameraIndex] = fallback_utility;
      }
      // Keep one cache-pruning pass when the provisional raw pass was skipped.
      if (!raw_utility_required) {
        pruneImageUtilityCacheLocked(fid);
      }
    }
  }

  // Initialise only this camera while holding its featureDetectorMutexes_ lock.
  {
    const size_t i = cameraIndex;
    if(!std::static_pointer_cast<cv::BriskDescriptorExtractor>(
         descriptorExtractors_.at(i))->isCameraAware()) {
      cv::Mat rays;
      cv::Mat imageJacobians;
      bool success = frameOut->geometry(i)->getCameraAwarenessMaps(rays, imageJacobians);
      OKVIS_ASSERT_TRUE(Exception, success, "camera awareness maps not initialised");
      if (std::strcmp(frameOut->geometry(i)->type().c_str(), "EUCMCamera") == 0){
        std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_.at(i))->setCameraProperties(
                rays, imageJacobians, float(std::static_pointer_cast<const cameras::EucmCamera>(frameOut->geometry(i))->focalLengthU()));
      }
      else{
        std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_.at(i))->setCameraProperties(
                rays, imageJacobians, float(std::static_pointer_cast<const cameras::PinholeCameraBase>(frameOut->geometry(i))->focalLengthU()));
      }
    }
  }

  // ExtractionDirection == gravity direction in camera frame.
  // Default assumes world gravity is -Z. Optionally use externally provided gravity direction in
  // IMU/sensor frame (from GaRLILEO), mapped into the camera frame using T_SC.
  Eigen::Vector3d extractionDir_C(0.0, 0.0, -1.0);
  {
    std::lock_guard<std::mutex> lock(externalGravityMutex_);
    if (hasExternalGravity_) {
      const auto T_SC = frameOut->T_SC(cameraIndex);
      if (T_SC) {
        const Eigen::Vector3d g_C = T_SC->C().transpose() * externalGravity_S_;
        if (g_C.squaredNorm() > 1e-12) {
          extractionDir_C = g_C.normalized();
        }
      }
    } else {
      // fall back to default world gravity direction
      const Eigen::Vector3d g_in_W(0.0, 0.0, -1.0);
      const Eigen::Vector3d g_C = T_WC.inverse().C() * g_in_W;
      if (g_C.squaredNorm() > 1e-12) {
        extractionDir_C = g_C.normalized();
      }
    }
  }
  std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_[cameraIndex])
      ->setExtractionDirection(
        cv::Vec3f(float(extractionDir_C[0]),float(extractionDir_C[1]),float(extractionDir_C[2])));

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);
#ifdef OKVIS_USE_NN
  frameOut->setNetwork(cameraIndex, networks_[cameraIndex]);
#endif

  // detect
  frameOut->detect(cameraIndex);

  // extract
  frameOut->describe(cameraIndex);

  // precompute backprojections
  frameOut->computeBackProjections(cameraIndex);

  // Restore original image to avoid affecting other consumers (visualisation / CNN, etc.).
  if (preprocessed_image_applied) {
    frameOut->setImage(cameraIndex, preprocess_img_orig);
  }

  return true;
}

void Frontend::setExternalGravityInSensorFrame(const Eigen::Vector3d& g_S_unit) {
  std::lock_guard<std::mutex> lock(externalGravityMutex_);
  if (g_S_unit.squaredNorm() > 1e-12) {
    externalGravity_S_ = g_S_unit.normalized();
    hasExternalGravity_ = true;
  }
}

void Frontend::clearExternalGravity() {
  std::lock_guard<std::mutex> lock(externalGravityMutex_);
  hasExternalGravity_ = false;
}

void Frontend::setExternalEgoSpeed(double speed_mps) {
  std::lock_guard<std::mutex> lock(externalEgoSpeedMutex_);
  externalEgoSpeed_ = speed_mps;
  hasExternalEgoSpeed_ = true;
}

void Frontend::clearExternalEgoSpeed() {
  std::lock_guard<std::mutex> lock(externalEgoSpeedMutex_);
  hasExternalEgoSpeed_ = false;
}

void Frontend::setExternalRadarTargets(const okvis::Time& stamp,
                                       const okvis::RadarTargetReadings& targets) {
  std::lock_guard<std::mutex> lock(externalRadarTargetsMutex_);
  externalRadarTargetsStamp_ = stamp;
  externalRadarTargets_ = targets;
  hasExternalRadarTargets_ = !externalRadarTargets_.empty();
}

void Frontend::clearExternalRadarTargets() {
  std::lock_guard<std::mutex> lock(externalRadarTargetsMutex_);
  hasExternalRadarTargets_ = false;
  externalRadarTargets_.clear();
}

bool Frontend::verifyRecognisedPlace(const Estimator &estimator,
                                     const okvis::ViParameters &params,
                                     const std::shared_ptr<const MultiFrame>framesInOut,
                                     const std::shared_ptr<const MultiFrame> oldFrame,
                                     kinematics::Transformation &T_Sold_Snew,
                                     Eigen::Matrix<double, 6, 6>& H,
                                     double descriptorDistinctivenessAvgThreshold,
                                     int minInliers)
{
  // geometric verification:
  cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  std::map<LandmarkId, std::vector<const uchar *>> descriptors;
  AlignedMap<LandmarkId, Eigen::Vector4d> landmarks;
  int ctr = 0;
  opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter::Points points;
  std::map<KeypointIdentifier, uint64_t> matches;

  // find matchable points
  TimerSwitchable loopClosureDescriptorMatchingTimer("2.04 loop closure descriptor matching");
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    for (size_t kOld = 0; kOld < oldFrame->numKeypoints(im); ++kOld) {
      uint64_t lmId = oldFrame->landmarkId(im, kOld);
      if (lmId == 0) {
        continue;
      }
      // get the 3D points / descriptors from old frame
      const uchar *oldDescripor = oldFrame->keypointDescriptor(im, kOld);
      Eigen::Vector4d landmark;
      bool isInitialised = false;
      oldFrame->getLandmark(im, kOld, landmark, isInitialised);
      if (!isInitialised)
        continue;
      if (landmark.norm() < 1.0e-12) {
        continue; // bit of a hack, signals there was no associated 3d point
      }
#ifdef OKVIS_USE_NN
      if (params.frontend.use_cnn && oldFrame->isClassified(im)) {
        // make sure not to use sky or person points here
        cv::Mat classification;
        oldFrame->getClassification(im, kOld, classification);
        if (classification.at<float>(10) > 3.5f) { // Sky
          continue;
        }
        if (classification.at<float>(11) > 53.5f) { // Person
          continue;
        }
      }
#endif
      auto iter = descriptors.find(LandmarkId(lmId));
      if (iter != descriptors.end()) {
        // just add descriptor
        iter->second.push_back(oldDescripor);
      } else {
        descriptors[LandmarkId(lmId)].push_back(oldDescripor);
        landmarks[LandmarkId(lmId)] = landmark;
      }
    }
  }

  // match
  for (auto iter = landmarks.begin(); iter != landmarks.end(); ++iter) {
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {

      if(framesInOut->numKeypoints(im) == 0) {
        continue;
      }
      
      const uchar *ddata = framesInOut->keypointDescriptor(im, 0);
      const size_t K = framesInOut->numKeypoints(im);
      uint32_t distMin = briskMatchingThreshold_;
      size_t kMin = 0;
      for (const unsigned char* oldDescripor : descriptors.at(iter->first)) {
        for (size_t k = 0; k < K; ++k) {
          const uint32_t dist = brisk::Hamming::PopcntofXORed(ddata + 48 * k, oldDescripor, 3);
          if (dist < distMin) {
            distMin = dist;
            kMin = k;
          }
        }
      }
      // now get best match
      if (distMin < briskMatchingThreshold_) {
        ctr++;
        const KeypointIdentifier kid(framesInOut->id(), im, kMin);
        points[iter->first.value()] = iter->second;
        matches[kid] = iter->first.value();
      }
    }
  }

  loopClosureDescriptorMatchingTimer.stop();

  if (ctr < minInliers || points.size() < 8) {
    return false;
  }

  // run 3d2d RANSAC
  // create a AbsolutePoseSac problem and RANSAC
  opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter adapter(points,
                                                                      matches,
                                                                      framesInOut->cameraSystem(),
                                                                      framesInOut);
  typedef opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem<
    opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter>
    LoopclosureAbsoluteModel;
  opengv::sac::Ransac<LoopclosureAbsoluteModel> ransac;
  std::shared_ptr<LoopclosureAbsoluteModel> absposeproblem_ptr(
    new LoopclosureAbsoluteModel(adapter, LoopclosureAbsoluteModel::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 16;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  if (adapter.getNumberCorrespondences() < 7) {
    return false;
  }
  TimerSwitchable ransacLoopClosureTimer("2.05 loop closure ransacking");
  ransac.computeModel(0);
  const int numInliers = int(ransac.inliers_.size());
  const double inlierRatio = double(ransac.inliers_.size())
                             / double(adapter.getNumberCorrespondences());

  ransacLoopClosureTimer.stop();
  if (numInliers < minInliers || inlierRatio < 0.7) {
    return false;
  }
  // remember inliers
  const size_t numCorrespondences = adapter.getNumberCorrespondences();
  std::vector<bool> inliers(numCorrespondences, false);
  for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
    inliers.at(size_t(ransac.inliers_.at(i))) = true;
  }

  // check distinciveness of survived matches
  float sum = 0.0;
  size_t singletonDescriptorCameras = 0;
  for (size_t im = 0; im < numCameras_; ++im) {
    Eigen::Matrix<float, Eigen::Dynamic, 48 * 8> descriptorMatrix(ransac.inliers_.size(), 48 * 8);
    int inlierCtr = 0;
    int ctr2 = 0;
    for (const auto &match : matches) {
      if (match.first.cameraIndex != im) {
        ctr2++;
        continue;
      }
      const uchar *desc = framesInOut->keypointDescriptor(match.first.cameraIndex,
                                                          match.first.keypointIndex);
      if (inliers[ctr2]) {
        for (size_t b = 0; b < 48; b++) {
          for (size_t c = 0; c < 8; c++) {
            if (desc[b] & (1 << c)) {
              descriptorMatrix(inlierCtr, b * 8 + c) = 1.0;
            } else {
              descriptorMatrix(inlierCtr, b * 8 + c) = 0.0;
            }
          }
        }
        inlierCtr++;
      }
      ctr2++;
    }
    if (inlierCtr == 1) {
      ++singletonDescriptorCameras;
    }
    // A singleton has no sample variance. Give it no distinctiveness evidence;
    // dividing by rows-1 would produce NaN and bypass the comparison below.
    if (inlierCtr > 1) {
      descriptorMatrix.conservativeResize(inlierCtr, 48 * 8);
      Eigen::Matrix<float, 1, 48 * 8> stdev
        = ((descriptorMatrix.rowwise() - descriptorMatrix.colwise().mean()).colwise().squaredNorm()
           / (descriptorMatrix.rows() - 1))
            .cwiseSqrt();
      sum += float(inlierCtr) * stdev.sum();
    }
  }

  const float avg = sum / float(ransac.inliers_.size());
  if (descriptorDistinctivenessAvgThreshold >= 0.0
      && avg < float(descriptorDistinctivenessAvgThreshold)
      && int(ransac.inliers_.size()) < 20) {
    LOG(INFO) << framesInOut->id() << "->" << oldFrame->id() << " : "
              << "Rejecting loop closure due to indistincive descriptors (avg=" << avg
              << ", threshold=" << descriptorDistinctivenessAvgThreshold
              << ", ransac_inliers=" << numInliers
              << ", singleton_cameras=" << singletonDescriptorCameras << ")";
    return false;
  }

  // refine
  TimerSwitchable loopClosureRefinementTimer("2.06 loop closure pose refinement");
  const uint64_t frameId = framesInOut->id();
  Eigen::Matrix4d T_Sold_Snew_mat = Eigen::Matrix4d::Identity();
  T_Sold_Snew_mat.topLeftCorner<3, 4>() = ransac.model_coefficients_;
  T_Sold_Snew = kinematics::Transformation(T_Sold_Snew_mat);

  // set up ceres problem
  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  ::ceres::Problem quickSolver(problemOptions);
  ::ceres::CauchyLoss cauchyLoss(3);
  ceres::PoseManifold pose6dParameterisation;
  ceres::HomogeneousPointManifold homogeneousPointParameterisation;

  // parameters: sensor pose and extrinsics
  std::shared_ptr<ceres::PoseParameterBlock> pose(new ceres::PoseParameterBlock(T_Sold_Snew, 1));
  quickSolver.AddParameterBlock(pose->parameters(), 7, &pose6dParameterisation);
  std::vector<std::shared_ptr<ceres::PoseParameterBlock>> extrinsics;
  for (size_t i = 0; i < framesInOut->numFrames(); ++i) {
    kinematics::Transformation T_SC = estimator.extrinsics(StateId(frameId), uchar(i));
    extrinsics.push_back(
      std::shared_ptr<ceres::PoseParameterBlock>(new ceres::PoseParameterBlock(T_SC, i + 2)));
    quickSolver.AddParameterBlock(extrinsics.back()->parameters(), 7, &pose6dParameterisation);
    quickSolver.SetParameterBlockConstant(extrinsics.back()->parameters());
  }
  const size_t landmarkIdOffset = framesInOut->numFrames() + 2;
  std::map<size_t, std::shared_ptr<ceres::ParameterBlock>> lms;
  std::vector<std::shared_ptr<ceres::ReprojectionError2dBase>> reprojectionErrors;
  std::vector<std::pair<const double *, const double *>> extrinsicsAndLandmarks;

  // add error terms and create landmarks if necessary
  for (size_t k = 0; k < numCorrespondences; ++k) {
    if (inliers[k]) {
      // get the landmark id:
      size_t camIdx = size_t(adapter.camIndex(k));
      size_t keypointIdx = size_t(adapter.keypointIndex(k));
      KeypointIdentifier kid(frameId, camIdx, keypointIdx);
      uint64_t lmId = matches.at(kid);
      std::shared_ptr<ceres::ParameterBlock> landmark;
      if (!lms.count(lmId + landmarkIdOffset)) {
        landmark.reset(
          new ceres::HomogeneousPointParameterBlock(points.at(lmId), lmId + landmarkIdOffset));
        quickSolver.AddParameterBlock(landmark->parameters(), 4, &homogeneousPointParameterisation);
        quickSolver.SetParameterBlockConstant(landmark->parameters());
        lms[lmId + landmarkIdOffset] = landmark;
      } else {
        landmark = lms.at(lmId + landmarkIdOffset);
      }
      Eigen::Vector2d kp;
      if (framesInOut->getKeypoint(camIdx, keypointIdx, kp)) {
        double size = 1.0;
        framesInOut->getKeypointSize(camIdx, keypointIdx, size);
        switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>>
            reprojectionError(new ceres::ReprojectionError<
                              cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              framesInOut->geometryAs<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
                camIdx),
              camIdx,
              kp,
              64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::EquidistantDistortion>>>
            reprojectionError(
              new ceres::ReprojectionError<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
                framesInOut->geometryAs<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
                  camIdx),
                camIdx,
                kp,
                64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>>
            reprojectionError(new ceres::ReprojectionError<
                              cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              framesInOut->geometryAs<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
                camIdx),
              camIdx,
              kp,
              64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
        }
        extrinsicsAndLandmarks.push_back(
          std::pair<const double *, const double *>(landmark->parameters(),
                                                    extrinsics[camIdx]->parameters()));
      }
    }
  }

  // get solution
  ::ceres::Solver::Options solverOptions;
  ::ceres::Solver::Summary summary;
  solverOptions.num_threads = params.estimator.realtime_num_threads;
  solverOptions.max_num_iterations = params.estimator.realtime_max_iterations;
  solverOptions.minimizer_progress_to_stdout = false;
  ::ceres::Solve(solverOptions, &quickSolver, &summary);
  T_Sold_Snew = pose->estimate();

  // get uncertainty: lhs to be filled on the fly
  H = Eigen::Matrix<double, 6, 6>::Zero();
  int additionalOutliers = 0;
  for (size_t e = 0; e < reprojectionErrors.size(); ++e) {
    // fill lhs Hessian
    const double *pars[3];
    pars[0] = pose->parameters();
    pars[1] = extrinsicsAndLandmarks.at(e).first;
    pars[2] = extrinsicsAndLandmarks.at(e).second;
    const auto &reprojectionError = reprojectionErrors.at(e);
    double *jacobians[3];
    double *jacobiansMinimal[3];
    Eigen::Vector2d err;
    Eigen::Matrix<double, 2, 7> jacobian;
    Eigen::Matrix<double, 2, 6> jacobianMinimal;
    jacobians[0] = jacobian.data();
    jacobiansMinimal[0] = jacobianMinimal.data();
    jacobians[1] = nullptr;
    jacobiansMinimal[1] = nullptr;
    jacobians[2] = nullptr;
    jacobiansMinimal[2] = nullptr;
    reprojectionError->EvaluateWithMinimalJacobians(pars, err.data(), jacobians, jacobiansMinimal);  
    if (err.norm() > 3.0) {
      additionalOutliers++;
    } else {
      H += jacobianMinimal.transpose() * jacobianMinimal;
    }
  }

  const int numFinalInliers = int(ransac.inliers_.size())-additionalOutliers;
  const double finalInlierRatio = double(numFinalInliers)
                             / double(adapter.getNumberCorrespondences());
  if (numFinalInliers < minInliers || finalInlierRatio < 0.7) {
    return false;
  }

  LOG(INFO) << "[LC][Verified] current=" << framesInOut->id()
            << " matched=" << oldFrame->id()
            << " ransac_inliers=" << numInliers
            << " final_inliers=" << numFinalInliers
            << " final_inlier_ratio=" << finalInlierRatio
            << " descriptor_avg=" << avg
            << " singleton_cameras=" << singletonDescriptorCameras;
  loopClosureRefinementTimer.stop();
  return true;
}

namespace {
constexpr int kRadarPrNumRangeBins = 40;
constexpr int kRadarPrNumAzimuthBins = 40;
constexpr double kRadarPrRangeBinM = 0.5;
constexpr double kRadarPrAzimuthBinDeg = 3.0;
constexpr double kRadarPrAzimuthMinDeg = -60.0;
constexpr double kRadarPrAzimuthMaxDeg =
    kRadarPrAzimuthMinDeg + kRadarPrNumAzimuthBins * kRadarPrAzimuthBinDeg;
constexpr double kPi = 3.14159265358979323846;

using RadarPrDescriptor = std::array<float, kRadarPrNumAzimuthBins * kRadarPrNumRangeBins>;

inline RadarPrDescriptor buildRadarPrDescriptor(
    const okvis::RadarTargetReadings& targets,
    double intensity_min) {
  RadarPrDescriptor desc;
  desc.fill(0.0f);
  const double invRangeBin = 1.0 / kRadarPrRangeBinM;
  const double invAzBin = 1.0 / kRadarPrAzimuthBinDeg;
  for (const auto& t : targets) {
    const double intensity = t.intensity;
    if (!std::isfinite(intensity) || intensity <= intensity_min) {
      continue;
    }
    const double x = t.point_R.x();
    const double y = t.point_R.y();
    if (!std::isfinite(x) || !std::isfinite(y)) {
      continue;
    }
    const double range = std::sqrt(x * x + y * y);
    if (!std::isfinite(range) || range <= 0.0) {
      continue;
    }
    const int r_bin = int(std::floor(range * invRangeBin));
    if (r_bin < 0 || r_bin >= kRadarPrNumRangeBins) {
      continue;
    }
    const double az_deg = std::atan2(y, x) * (180.0 / kPi);
    if (az_deg < kRadarPrAzimuthMinDeg || az_deg >= kRadarPrAzimuthMaxDeg) {
      continue;
    }
    const int a_bin = int(std::floor((az_deg - kRadarPrAzimuthMinDeg) * invAzBin));
    if (a_bin < 0 || a_bin >= kRadarPrNumAzimuthBins) {
      continue;
    }
    const int idx = a_bin * kRadarPrNumRangeBins + r_bin;
    desc[size_t(idx)] = std::max(desc[size_t(idx)], float(intensity));
  }
  return desc;
}

inline double radarPrNorm(const RadarPrDescriptor& d) {
  double s = 0.0;
  for (float v : d) {
    s += double(v) * double(v);
  }
  return std::sqrt(s);
}

struct RadarPrSimilarityResult {
  double sim = 0.0;
  int shift = 0;
};

inline cv::Mat radarPrDescriptorToRangeAzimuthImage(const RadarPrDescriptor& d) {
  cv::Mat img(kRadarPrNumRangeBins, kRadarPrNumAzimuthBins, CV_32FC1);
  for (int r = 0; r < kRadarPrNumRangeBins; ++r) {
    float* row = img.ptr<float>(r);
    for (int a = 0; a < kRadarPrNumAzimuthBins; ++a) {
      row[a] = d[size_t(a * kRadarPrNumRangeBins + r)];
    }
  }
  return img;
}

inline RadarPrSimilarityResult radarPrFftCorrelationSimilarity(
    const RadarPrDescriptor& a,
    const RadarPrDescriptor& b) {
  RadarPrSimilarityResult out;
  const double na = radarPrNorm(a);
  const double nb = radarPrNorm(b);
  if (na < 1e-12 || nb < 1e-12) {
    return out;
  }

  cv::Mat img_a = radarPrDescriptorToRangeAzimuthImage(a);
  cv::Mat img_b = radarPrDescriptorToRangeAzimuthImage(b);
  cv::Mat fft_a;
  cv::Mat fft_b;
  cv::dft(img_a, fft_a, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
  cv::dft(img_b, fft_b, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

  cv::Mat cross_spectrum;
  cv::mulSpectrums(fft_a, fft_b, cross_spectrum, cv::DFT_ROWS, true);

  cv::Mat correlation_by_range;
  cv::idft(cross_spectrum, correlation_by_range,
           cv::DFT_ROWS | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

  double max_corr = -std::numeric_limits<double>::infinity();
  int best_shift = 0;
  for (int shift = 0; shift < kRadarPrNumAzimuthBins; ++shift) {
    double corr = 0.0;
    for (int r = 0; r < kRadarPrNumRangeBins; ++r) {
      corr += double(correlation_by_range.at<float>(r, shift));
    }
    if (corr > max_corr) {
      max_corr = corr;
      best_shift = shift;
    }
  }

  double best = max_corr / (na * nb + 1e-12);
  // Clamp to [0,1] for numerical safety. Inputs are non-negative intensities, so the
  // best azimuth-only circular correlation should be non-negative and bounded by Cauchy-Schwarz.
  if (!std::isfinite(best) || best < 0.0) best = 0.0;
  if (best > 1.0) best = 1.0;
  out.sim = best;
  out.shift = best_shift;
  return out;
}

}  // namespace

// filtered DBoW query result
int Frontend::getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                                    const std::vector<std::vector<uchar>> &features,
                                    std::vector<std::pair<StateId, double>> &stateIds) const
{
  DBoW2::QueryResults dBoWResult;
  dBow->database.query(features, dBoWResult, -1); // get all matches
  DBoW2::QueryResults dBoWResultOrig = dBoWResult;
  // sort ascending -- we want to match oldest...
  std::sort(dBoWResult.begin(),
            dBoWResult.end(),
            [](const DBoW2::Result &lhs, const DBoW2::Result &rhs) { return lhs.Id < rhs.Id; });

  // nonmax suppression
  std::set<uint64_t> ids;
  std::set<uint64_t> suppressedIds;
  const size_t numKeyframes = dBoWResultOrig.size();
  int nonmaxRadius = 5;
  for (size_t f = 0; f < numKeyframes; ++f) {
    const double score = dBoWResultOrig[f].Score;
    const uint64_t id = dBoWResultOrig[f].Id;
    if (id >= dBoWResult.size())
      continue;
    if (score < 0.375) {
      break;
    }
    // check suppressed:
    if (suppressedIds.count(f)) {
      continue;
    }
    // check maximum
    bool isMax = true;
    for (int a = std::max(0, int(id) - nonmaxRadius);
         a <= (std::min(int(numKeyframes-1), int(id) + nonmaxRadius));
         ++a) {
      if (dBoWResult[a].Score > score) {
        isMax = false;
      }
    }
    if (!isMax) {
      continue;
    }

    // suppress
    for (int a = std::max(0, int(id) - nonmaxRadius);
         a <= (std::min(int(numKeyframes-1), int(id) + nonmaxRadius));
         ++a) {
      suppressedIds.insert(a);
    }

    // use
    ids.insert(id);
  }

  for (size_t id : ids) {

    // start with oldest keyframe match
    const double p = dBoWResult.at(id).Score;

    // get old multiframe
    uint64_t poseId = dBow->poseIds.at(id);

    // output
    stateIds.push_back(std::make_pair(StateId(poseId),p));
  }

  return stateIds.size();
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
    Estimator &estimator, const okvis::ViParameters& params,
    std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) {

  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception, distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet")
  }
  int num3dMatches = 0;

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  double trackingQuality = 1.0;

  auto blockKeyframeDueToStationarySpeed = [&](const char* where)->bool {
    const double thr = params.garlileo.keyframe_stationary_speed_threshold;
    if (thr <= 0.0) {
      return false;
    }
    if (!isInitialized_) {
      return false;  // don't block keyframes during initialisation phase
    }
    double speed = 0.0;
    {
      std::lock_guard<std::mutex> lock(externalEgoSpeedMutex_);
      if (!hasExternalEgoSpeed_) {
        return false;
      }
      speed = externalEgoSpeed_;
    }
    if (speed >= thr) {
      return false;
    }

    static double last_log = 0.0;
    const double now = okvis::Time::now().toSec();
    if (now - last_log > 1.0) {
      LOG(INFO) << "[GaRLILEO][Keyframe] blocked (stationary): speed=" << speed
                << " m/s < thr=" << thr << " m/s"
                << " (skip overlap keyframe check) at " << where;
      last_log = now;
    }
    return true;
  };
  if (estimator.numFrames() > 1) {

    // match to all landmarks
    TimerSwitchable matchMapTimer("2.01 match to map");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        num3dMatches = matchToMap<cameras::EucmCamera>(estimator, params, framesInOut->id());
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchMapTimer.stop();

    // check tracking quality
    trackingQuality = estimator.trackingQuality(StateId(framesInOut->id()));
    const double trackingLostThr = std::max(0.0, params.garlileo.tracking_lost_quality_threshold);
    if (trackingQuality < trackingLostThr) {
      if(estimator.numFrames() == 2 && params.nCameraSystem.numCameras()==1) {
        // mono. we can't have matches at this point, so don't warn
      } else {
        if (num3dMatches >= 3) {
          LOG(WARNING) << "3d2d tracking weak: quality=" << trackingQuality
                       << ". Number of 3d2d-matches: " << num3dMatches;
        } else {
          LOG(WARNING) << "3d2d tracking lost. Number of 3d2d-matches: " << num3dMatches;
        }
      }
    }

    // do motion stereo
    if(!trackingLost_ || !isInitialized_) {
      bool rotationOnly = false;
      TimerSwitchable matchMotionStereoTimer("2.02 match motion stereo");
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          matchMotionStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          matchMotionStereo<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          matchMotionStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::NoDistortion: {
          matchMotionStereo<cameras::EucmCamera>(estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
      }
      if(!rotationOnly || num3dMatches>5) {
        if(!isInitialized_) {
          isInitialized_ = true;
          LOG(INFO) << "Initialized!";
        }
      }
      trackingQuality = estimator.trackingQuality(StateId(framesInOut->id()));
      matchMotionStereoTimer.stop();
    }
    //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after match motion stereo")

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    if(kfPrior){
      *asKeyframe = true;
     }
    else{
      if (blockKeyframeDueToStationarySpeed("keyframe_decision")) {
        *asKeyframe = false;
      } else {
      *asKeyframe = doWeNeedANewKeyframe(estimator, framesInOut);
      }
    }
  } else {
    *asKeyframe = true;  // first frame needs to be keyframe
  }

  // prepare features for place recognition
  std::vector<std::vector<uchar>> features(framesInOut->numKeypoints());
  // first, we are trying to match the database for loop closures
  int offset = 0;
  for (size_t im = 0; im < numCameras_; ++im) {
    for (size_t k = 0; k < framesInOut->numKeypoints(im); ++k) {
      features.at(k + offset).resize(48); // TODO: get 48 from feature
      memcpy(features.at(k + offset).data(),
             framesInOut->keypointDescriptor(im, k),
             48 * sizeof(uchar));
    }
    offset += framesInOut->numKeypoints(im);
  }

  /*MULTI-SESSION AND MULTI-AGENT*/
  if (!estimator.isLoopClosing() && !estimator.isLoopClosureAvailable()
      && !estimator.needsFullGraphOptimisation() && isInitialized_) {
    for (uint64_t c = 0; c < componentDBows_.size(); ++c) {
      TimerSwitchable matchDBoWTimer0("2.3.0 multi-session and multi-agent place recognition");
      std::vector<std::pair<StateId, double>> stateIds;
      getFilteredDBoWResult(componentDBows_.at(c), features, stateIds);
      matchDBoWTimer0.stop();

      int attempts = 0;

      for (const auto & id : stateIds) {

        double p = id.second;

        // get old multiframe
        MultiFramePtr oldMultiFrame = components_.at(c).multiFrames_.at(id.first);

        // stop after some amount of attempts
        if (attempts > std::max(10, int(componentDBows_.at(c)->poseIds.size() / 20)))
          break;
        if (p > 0.4) {
          kinematics::Transformation T_Sold_Snew;
          Eigen::Matrix<double, 6, 6> H;
          if (!verifyRecognisedPlace(estimator,
                                     params,
                                     framesInOut,
                                     oldMultiFrame,
                                     T_Sold_Snew,
                                     H,
                                     182.0,
                                     40)) {
            attempts++;
            continue;
          }
          attempts++;

          estimator.T_AiS_[StateId(framesInOut->id())][c] =
            components_.at(c).fullGraph_->pose(id.first) * T_Sold_Snew;

          break;
        }
      }
    }
  }

  /*LOOP CLOSURES*/
  if(params.estimator.do_loop_closures && !estimator.isLoopClosing()
      && !estimator.isLoopClosureAvailable()
      && !estimator.needsFullGraphOptimisation() && isInitialized_) {
    TimerSwitchable matchDBoWTimer("2.03 loop closure query");
    std::vector<std::pair<StateId, double>> stateIds;
    getFilteredDBoWResult(dBow_, features, stateIds);
    matchDBoWTimer.stop();

    // Optional: fuse radar-context score into DBoW score for candidate ranking.
    const bool useRadarPr = params.garlileo.place_recognition_use_radar;
    RadarPrDescriptor radarDescCurr;
    bool hasRadarDescCurr = false;
    okvis::RadarTargetReadings radarTargetsCurr;
    if (useRadarPr) {
      std::lock_guard<std::mutex> lock(externalRadarTargetsMutex_);
      if (hasExternalRadarTargets_) {
        radarTargetsCurr = externalRadarTargets_;
      }
    }
    if (useRadarPr && !radarTargetsCurr.empty()) {
      radarDescCurr =
          buildRadarPrDescriptor(radarTargetsCurr, params.garlileo.place_recognition_radar_min_intensity);
      hasRadarDescCurr = (radarPrNorm(radarDescCurr) > 1e-12);
    }
    double alpha = params.garlileo.place_recognition_radar_alpha;
    if (useRadarPr && hasRadarDescCurr) {
      // Reduce alpha when vision quality is low.
      const double q_track = std::clamp(trackingQuality / 0.5, 0.0, 1.0);
      const double q_match = std::clamp(double(num3dMatches) / 20.0, 0.0, 1.0);
      // alpha = std::clamp(alpha * q_track * q_match, 0.5, 1.0);
      alpha = std::clamp(alpha * q_track, 0.7, 1.0);
      static double last_log = 0.0;
      const double now = okvis::Time::now().toSec();
      if (now - last_log > 1.0) {
        LOG(INFO) << "[RadarPR] enabled: num_targets=" << radarTargetsCurr.size()
                  << " alpha=" << alpha
                  << " trackingQuality=" << trackingQuality
                  << " num3dMatches=" << num3dMatches;
        last_log = now;
      }
    }

    double imageUtilityMinForLc = std::numeric_limits<double>::quiet_NaN();
    bool lowFeaturesForLc = false;
    const bool imageUtilityGateForLc = params.garlileo.image_utility_threshold > 0.0;
    if (imageUtilityGateForLc) {
      std::vector<double> imageUtilities;
      if (getImageUtility(framesInOut->id(), imageUtilities)) {
        double minUtility = std::numeric_limits<double>::infinity();
        for (double u : imageUtilities) {
          if (std::isfinite(u)) {
            minUtility = std::min(minUtility, u);
          }
        }
        if (std::isfinite(minUtility)) {
          imageUtilityMinForLc = minUtility;
          lowFeaturesForLc = imageUtilityMinForLc < params.garlileo.image_utility_threshold;
        }
      }
    }
    const double trackingLostThr = std::max(0.0, params.garlileo.tracking_lost_quality_threshold);
    const bool trackingLostForLc = trackingQuality < trackingLostThr;
    const bool lcFallbackMode = lowFeaturesForLc || trackingLostForLc;
    if (lcFallbackMode) {
      static double last_log = 0.0;
      const double now = okvis::Time::now().toSec();
      if (now - last_log > 1.0) {
        LOG(WARNING) << "[LC][Fallback] enabled: trackingQuality=" << trackingQuality
                     << " tracking_thr=" << trackingLostThr
                     << " tracking_lost=" << trackingLostForLc
                     << " image_utility_min=" << imageUtilityMinForLc
                     << " image_utility_thr=" << params.garlileo.image_utility_threshold
                     << " low_features=" << lowFeaturesForLc
                     << " num3dMatches=" << num3dMatches;
        last_log = now;
      }
    }

    struct CandidateScore {
      StateId id;
      double s_img = 0.0;
      double s_rad = -1.0;
      double s = 0.0;
      bool has_visual = false;
      bool has_radar = false;
    };
    // Base candidate set from visual DBoW.
    std::vector<CandidateScore> candidates;
    candidates.reserve(stateIds.size());
    for (const auto& id : stateIds) {
      CandidateScore c;
      c.id = id.first;
      c.s_img = id.second;
      c.s = c.s_img;
      c.has_visual = true;
      if (useRadarPr && hasRadarDescCurr) {
        RadarPrDescriptor candDesc;
        bool hasCandDesc = false;
        {
          std::lock_guard<std::mutex> lock(radarPrDescriptorsMutex_);
          const auto it = radarPrDescriptors_.find(uint64_t(c.id.value()));
          if (it != radarPrDescriptors_.end()) {
            candDesc = it->second;
            hasCandDesc = true;
          }
        }
        if (hasCandDesc) {
          const RadarPrSimilarityResult sim_shift =
              radarPrFftCorrelationSimilarity(radarDescCurr, candDesc);
          c.s_rad = sim_shift.sim;
          c.has_radar = true;
          c.s = alpha * c.s_img + (1.0 - alpha) * c.s_rad;
        }
      }
      candidates.push_back(c);
    }
    if (useRadarPr && hasRadarDescCurr) {
      std::sort(candidates.begin(), candidates.end(), [](const CandidateScore& a, const CandidateScore& b) {
        return a.s > b.s;
      });
    }

    TimerSwitchable attemptLoopClosureTimer("2.07 attempt loop closure", true);
    // nonmax suppression
    size_t attempts = 0;
    for(const auto & cand : candidates) {
      const double p = cand.s;
      const double p_img = cand.s_img;
      if(attempts > std::max(size_t(10),dBow_->poseIds.size()/20)) break;

      const bool pass_visual_gate = cand.has_visual && (p > params.estimator.p_dbow);
      if (!pass_visual_gate) {
        continue;
      }

      const std::shared_ptr<const MultiFrame> oldFrame = estimator.multiFrame(cand.id);
      if (!oldFrame) {
        attempts++;
        continue;
      }
      // check if already existing loop closure or matching against current frame
      if(!estimator.isPoseGraphFrame(cand.id)) {
        continue;
      }
      if(estimator.isLoopClosureFrame(cand.id)) {
        continue;
      }
      if(estimator.isRecentLoopClosureFrame(cand.id)) {
        continue;
      }
      if(!estimator.isPlaceRecognitionFrame(cand.id)) {
        continue;
      }

      kinematics::Transformation T_Sold_Snew;
      Eigen::Matrix<double, 6, 6> H;
      bool verified = false;

      const double descriptorDistinctivenessAvgThreshold =
          lcFallbackMode
              ? params.garlileo.fallback_loop_closure_descriptor_distinctiveness_avg_threshold
              : 182.0;
      if (verifyRecognisedPlace(estimator, params, framesInOut, oldFrame, T_Sold_Snew, H,
                                descriptorDistinctivenessAvgThreshold, 10)) {
        verified = true;
      }
      if (!verified) {
        attempts++;
        continue;
      }
      attempts++;

      // enforce relative transformation
      attemptLoopClosureTimer.start();
      bool skipFullGraphOptimisation = false;
      const uint64_t frameId = framesInOut->id();
      const bool keepFullGraphRelativePoseConstraint =
          params.garlileo.fallback_loop_closure_persistent_enable
          && lcFallbackMode;
      const double fullGraphLoopInformationScale =
          lcFallbackMode
              ? std::max(1.0e-12, params.garlileo.fallback_loop_information_scale)
              : 100.0;
      const double persistentFullGraphLoopInformationScale =
          lcFallbackMode
              ? std::max(1.0e-12, params.garlileo.fallback_loop_persistent_information_scale)
              : fullGraphLoopInformationScale;
      const bool addLoopPathRelativePoseConstraints =
          lcFallbackMode
          && params.garlileo.fallback_loop_closure_path_constraints_enable;
      const int loopPathRelativePoseConstraintMode =
          lcFallbackMode
              ? params.garlileo.fallback_loop_closure_path_constraints_mode
              : 1;
      const int loopPathRelativePoseConstraintStride =
          lcFallbackMode
              ? std::max(1, params.garlileo.fallback_loop_closure_chain_constraint_stride)
              : 1;
      const bool weakenExistingLoopPathConstraints =
          lcFallbackMode
          && params.garlileo.fallback_loop_weaken_existing_constraints_enable;
      const double existingLoopPathConstraintScale =
          lcFallbackMode
              ? std::max(0.0, params.garlileo.fallback_loop_existing_constraint_scale)
              : 1.0;
      const double loopPathChainInformationScale =
          lcFallbackMode
              ? std::max(1.0e-12, params.garlileo.fallback_loop_chain_information_scale)
              : persistentFullGraphLoopInformationScale;
      bool loopClosureAttemptSuccessful =
          estimator.attemptLoopClosure(
            StateId(oldFrame->id()), StateId(frameId), T_Sold_Snew, H,
            skipFullGraphOptimisation,
            params.estimator.drift_percentage_heuristic,
            keepFullGraphRelativePoseConstraint,
            fullGraphLoopInformationScale,
            persistentFullGraphLoopInformationScale,
            addLoopPathRelativePoseConstraints,
            loopPathRelativePoseConstraintMode,
            loopPathRelativePoseConstraintStride,
            weakenExistingLoopPathConstraints,
            existingLoopPathConstraintScale,
            loopPathChainInformationScale);
      if(!loopClosureAttemptSuccessful) {
        LOG(INFO) << "unsuccessful loop closure frame "<< cand.id.value();
        attemptLoopClosureTimer.stop();
        continue;
      }
      attemptLoopClosureTimer.stop();
      if (lcFallbackMode
          && !params.garlileo.fallback_loop_closure_full_graph_optimisation_enable) {
        skipFullGraphOptimisation = true;
      }
      // bring back old landmarks
      TimerSwitchable addLoopClosureTimer("2.08 add loop closure");
      std::set<LandmarkId> loopClosureLandmarks;
      estimator.addLoopClosureFrame(StateId(oldFrame->id()), loopClosureLandmarks,
                                    skipFullGraphOptimisation);
      addLoopClosureTimer.stop();
      // properly match against all loopClosure frame points
      TimerSwitchable matchLoopClosureTimer("2.09 match loop closure");
      int loopClosureMatches = 0;
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          loopClosureMatches =
              matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
                  estimator, params, framesInOut->id(), &loopClosureLandmarks);
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          loopClosureMatches = matchToMap<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
              estimator, params, framesInOut->id(), &loopClosureLandmarks);
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          loopClosureMatches =
              matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
                  estimator, params, framesInOut->id(), &loopClosureLandmarks);
          break;
        }
        case okvis::cameras::NCameraSystem::NoDistortion: {
          loopClosureMatches = matchToMap<cameras::EucmCamera>(
                  estimator, params, framesInOut->id(), &loopClosureLandmarks);
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
      }
      const char* lc_source = lcFallbackMode ? "visual_fallback" : "visual";
      LOG(INFO) << "LOOP CLOSURE: current frame " << framesInOut->id()
                << ", matching to keyframe " << oldFrame->id() << ", "
                << loopClosureMatches << " matches, source=" << lc_source
                << ", s=" << p
                << " (p_img=" << p_img << ", s_rad=" << cand.s_rad
                << ", alpha=" << alpha
                << ", radar_score_used=" << cand.has_radar
                << ", lc_fallback=" << lcFallbackMode
                << ", low_features=" << lowFeaturesForLc
                << ", tracking_lost=" << trackingLostForLc
                << ", full_graph_lc_info_scale=" << fullGraphLoopInformationScale
                << ", persistent_full_graph_lc_info_scale="
                << persistentFullGraphLoopInformationScale
                << ", persistent_full_graph_constraint="
                << keepFullGraphRelativePoseConstraint
                << ", path_relative_constraints="
                << addLoopPathRelativePoseConstraints
                << ", path_constraint_mode="
                << loopPathRelativePoseConstraintMode
                << ", path_constraint_stride="
                << loopPathRelativePoseConstraintStride
                << ", weaken_existing_path_constraints="
                << weakenExistingLoopPathConstraints
                << ", existing_path_constraint_scale="
                << existingLoopPathConstraintScale
                << ", chain_path_constraint_info_scale="
                << loopPathChainInformationScale
                << ", full_graph_optimisation="
                << (!skipFullGraphOptimisation)
                << ").";

      matchLoopClosureTimer.stop();
      // re-decide keyframe:
      if(kfPrior){
        *asKeyframe = true;
      }
      else{
        if (blockKeyframeDueToStationarySpeed("keyframe_redecision_after_lc")) {
          *asKeyframe = false;
        } else {
        *asKeyframe = doWeNeedANewKeyframe(estimator, framesInOut);
        }
      }

      break; // only consider oldest keyframe match.
    }
    // if keyframe, we add to relocalisation database
    if(*asKeyframe && !kfPrior) {
      dBow_->database.add(features);
      dBow_->poseIds.push_back(framesInOut->id());
      if (useRadarPr && hasRadarDescCurr) {
        {
          std::lock_guard<std::mutex> lock(radarPrDescriptorsMutex_);
          radarPrDescriptors_[framesInOut->id()] = radarDescCurr;
        }
        {
          std::lock_guard<std::mutex> lock(radarPrTargetsMutex_);
          radarPrTargetsByKeyframe_[framesInOut->id()] = radarTargetsCurr;
        }
      }
    }
  }


#ifdef OKVIS_USE_NN
  // This needs to be after keyframe re-decision, otherwise we might delete a frame before the CNN
  // finishes. This could obviously be done in a smarter way though.
  if(params.frontend.use_cnn && isInitialized_) {
    if(*asKeyframe) {
      // clean up threads if still running
      std::set<StateId> toDelete;
      for(auto threads = cnnThreads_.begin(); threads != cnnThreads_.end(); ++threads) {
        bool deleting = true;
        for(size_t i=0; i<threads->second.size(); ++i) {
          //if(threads->second[i] && estimator.multiFrame(StateId(threads->first))->isClassified(i)) {
            threads->second[i]->join();
            delete threads->second[i];
            threads->second[i] = nullptr;
          //} else {
          //  deleting = false;
          //}
        }
        if(deleting) {
          toDelete.insert(StateId(threads->first));
        }
      }
      for(const auto & deleting : toDelete) {
        cnnThreads_.erase(deleting);
      }

      // launch classification in background
      cnnThreads_[StateId(framesInOut->id())] = std::vector<std::thread*>(
            params.nCameraSystem.numCameras(), nullptr);
      for (size_t i = 0; i < params.nCameraSystem.numCameras(); ++i) {
        cnnThreads_[StateId(framesInOut->id())].at(i) =
            new std::thread(&MultiFrame::computeClassifications,framesInOut.get(), i,
                      64*(framesInOut->image(i).cols/64), 64*(framesInOut->image(i).rows/64));
      }
    }
  }
#else
  OKVIS_ASSERT_TRUE(Exception, !params.frontend.use_cnn,
                    "Requested CNN classification, but not compiled with USE_NN option.")
#endif

  // do stereo match -- get new landmarks only when this is a keyframe
  if(*asKeyframe) {
    TimerSwitchable matchStereoTimer("2.10 match stereo");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        matchStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        matchStereo<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        matchStereo<okvis::cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        matchStereo<okvis::cameras::EucmCamera>(
                estimator, framesInOut, params, *asKeyframe);
        break;
        }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchStereoTimer.stop();
    //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after stereo")

  }

  // remove outliers, as the last matching step may have introduced some:
  switch (distortionType) {
  case okvis::cameras::NCameraSystem::RadialTangential: {
    removeOutliers<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::Equidistant: {
    removeOutliers<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::RadialTangential8: {
    removeOutliers<okvis::cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  default:
    OKVIS_THROW(Exception, "Unsupported distortion type.")
    break;
  } //ToDo:EUCM

#ifdef OKVIS_USE_NN
  if(params.frontend.use_cnn) {
    // remove matches into dynamic areas
    // get all landmarks
    MapPoints pointMap;
    estimator.getLandmarks(pointMap);
    for(MapPoints::iterator it = pointMap.begin(); it != pointMap.end(); ++it) {
      bool remove = false;
      if(it->second.classification == 10 || it->second.classification == 11) {
        remove = true;
      } else {
      for(auto& obs : it->second.observations) {
        if(!estimator.isKeyframe(StateId(obs.frameId))) continue;
        auto frame = estimator.multiFrame(StateId(obs.frameId));
        if(!frame->isClassified(obs.cameraIndex)) continue;
        cv::Mat classification;
        if(frame->getClassification(obs.cameraIndex, obs.keypointIndex, classification)) {
          if(classification.at<float>(10) > 3.5f) { // Sky
            remove = true;
            Eigen::Vector2d kpt;
            frame->getKeypoint(obs.cameraIndex, obs.keypointIndex, kpt);
            estimator.setLandmarkClassification(it->first, 10);
            break;
          }
          if(classification.at<float>(11) > 53.5f) { // Person
            remove = true;
            estimator.setLandmarkClassification(it->first, 11);
            break;
          }
        }
      }
      }
      if(remove) {
        std::set<KeypointIdentifier> observations = it->second.observations;
        for(auto& obs : observations) {
          estimator.setObservationInformation(
                StateId(obs.frameId), obs.cameraIndex, obs.keypointIndex,
                Eigen::Matrix2d::Identity()*0.0001);
        }
      }
    }
  }
#endif
  estimator.cleanUnobservedLandmarks();

  const double trackingLostThr = std::max(0.0, params.garlileo.tracking_lost_quality_threshold);
  return trackingQuality >= trackingLostThr;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(
    const okvis::ImuMeasurementDeque& imuMeasurements, const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS_propagated, okvis::SpeedAndBias& speedAndBiases,
    const okvis::Time& t_start, const okvis::Time& t_end, Eigen::Matrix<double, 15, 15>* covariance,
    Eigen::Matrix<double, 15, 15>* jacobian) const {

  if (imuMeasurements.size() < 2) {
    LOG(WARNING) << "- Skipping propagation as only one IMU measurement has been given to frontend."
                 << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated =
      okvis::ceres::ImuError::propagation(imuMeasurements, imuParams, T_WS_propagated,
                                          speedAndBiases, t_start, t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

void Frontend::endCnnThreads() {
  for(auto & threads : cnnThreads_) {
    for(auto & thread : threads.second) {
      if(thread) {
        thread->join();
        delete thread;
        thread = nullptr;
      }
    }
  }
}

void Frontend::clear()
{
  endCnnThreads();
  isInitialized_ = false;        // Is the pose initialised?
  dBow_->database.clear();
  dBow_->poseIds.clear(); // Store the multiframe IDs corresponsind to the dBow ones
  trackingLost_ = false; // Is the tracking currently lost?
  {
    std::lock_guard<std::mutex> lock(imageUtilityMutex_);
    imageUtilityByFrameId_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(externalRadarTargetsMutex_);
    hasExternalRadarTargets_ = false;
    externalRadarTargets_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(radarPrDescriptorsMutex_);
    radarPrDescriptors_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(radarPrTargetsMutex_);
    radarPrTargetsByKeyframe_.clear();
  }
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(const Estimator &estimator,
                                    std::shared_ptr<okvis::MultiFrame> currentFrame) {
  if (estimator.numFrames() < 4) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_) return false;

  int intersectionCount = 0;
  int unionCount = 0;

  size_t numKeypoints = 0;

  // go through all the frames and try to match the initialized keypoints
  std::set<uint64_t> lmIds;
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {
    const int rows = currentFrame->image(im).rows/10;
    const int cols = currentFrame->image(im).cols/10;

    cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

    const size_t numB = currentFrame->numKeypoints(im);
    numKeypoints += numB;
    const double radius = double(std::min(rows,cols))*kptrad;
    cv::KeyPoint keypoint;
    for (size_t k = 0; k < numB; ++k) {
      currentFrame->getCvKeypoint(im, k, keypoint);
      cv::circle(detections, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
      uint64_t lmId = currentFrame->landmarkId(im, k);
      if (lmId != 0) {
        cv::circle(matches, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        lmIds.insert(lmId);
      }
    }

    // IoU
    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(matches, detections, intersectionMask);
    cv::bitwise_or(matches, detections, unionMask);
    intersectionCount += cv::countNonZero(intersectionMask);
    unionCount += cv::countNonZero(unionMask);
  }

  double overlap = double(intersectionCount)/double(unionCount);

  std::set<StateId> allFrames = estimator.keyFrames();
  allFrames.insert(estimator.loopClosureFrames().begin(), estimator.loopClosureFrames().end());
  for(size_t age = 0; age < estimator.numFrames(); ++age) {
    auto id = estimator.stateIdByAge(age);
    if(!estimator.isInImuWindow(id)) {
      break;
    }
    if(estimator.isKeyframe(id)) {
      allFrames.insert(id);
    }
  }
  double overlapOthers = 0.0;
  for(auto frame : allFrames) {
    int intersectionCount = 0;
    int unionCount = 0;

    // go through all the frames and try to match the initialized keypoints
    auto otherFrame = estimator.multiFrame(frame);
    for (size_t im = 0; im < otherFrame->numFrames(); ++im) {
      const int rows = otherFrame->image(im).rows/10;
      const int cols = otherFrame->image(im).cols/10;

      cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
      cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

      const size_t numB = otherFrame->numKeypoints(im);

      const double radius = double(std::min(rows,cols))*kptrad;
      cv::KeyPoint keypoint;
      for (size_t k = 0; k < numB; ++k) {
        otherFrame->getCvKeypoint(im, k, keypoint);
        cv::circle(detections, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        uint64_t lmId = otherFrame->landmarkId(im, k);
        if (lmId != 0 && lmIds.count(lmId)) {
          cv::circle(matches, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        }
      }

      // IoU
      cv::Mat intersectionMask, unionMask;
      cv::bitwise_and(matches, detections, intersectionMask);
      cv::bitwise_or(matches, detections, unionMask);
      intersectionCount += cv::countNonZero(intersectionMask);
      unionCount += cv::countNonZero(unionMask);
    }

    overlapOthers = std::max(overlapOthers, double(intersectionCount)/double(unionCount));
  }

  overlap = std::min(overlapOthers, overlap);

  // take a decision
  if(numKeypoints < 7 * currentFrame->numFrames()) {
    // a respectable keyframe needs some detections...
    return false;
  }
  if (float(overlap) > keyframeInsertionOverlapThreshold_
      /*&& double(numMatches)/double(numKeypoints) > 0.35*/) {
    return false;
  } else {
    return true;
  }
}

// Match a new multiframe to existing keyframes
template <class CAMERA_GEOMETRY>
int Frontend::matchToMap(Estimator &estimator, const okvis::ViParameters& params,
                         const uint64_t currentFrameId,
                         const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively) {

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  // get all landmarks
  MapPoints pointMap;
  estimator.getLandmarks(pointMap);

  // these may be needed for loop-closure map fusion
  std::vector<LandmarkId> oldIds, newIds;

  // store the map to be matched
  std::vector<AlignedMap<LandmarkId, LandmarkToMatch>>
      landmarksToMatchVec(params.nCameraSystem.numCameras());

  // match
  int ctr = 0;
  std::vector<cv::Mat> descriptorPool(params.nCameraSystem.numCameras());
  kinematics::Transformation T_WS1 = estimator.pose(StateId(currentFrameId));
  double reprErr = 0.0;
  size_t reprErrContributors = 0;
  size_t reprErrMatches = 0;

  // Radar landmark support was removed. Keep hook as no-op for call-site simplicity.
  auto maybeBoostObservationInformation =
      [&](size_t, size_t, const MultiFramePtr&, const LandmarkId&, const Eigen::Vector3d&) {};

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {

    // the current frame to match
    const MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));

    const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
                            + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());
    const double reprThreshold = params.imu.use ? 3.0+f*0.06 : 3.0+f*0.34;

    const size_t numKeypoints = multiFrame->numKeypoints(im);
    if(numKeypoints == 0) {
      continue; // no points -- bad!
    }

    // for checks if in image
    const double maxU = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->imageWidth() + reprThreshold;
    const double maxV = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->imageHeight() + reprThreshold;

    // prepare landmarks as visible in this frame
    AlignedMap<LandmarkId, LandmarkToMatch> landmarksToMatch;
    const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
    const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
    const kinematics::Transformation T_CW1 = T_WC1.inverse();

    const double focalLength =
        multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
        + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV();

    // go through all landmarks
    const size_t numDescriptorsToKeep = 3; // use only best 3
    descriptorPool[im] = cv::Mat(
        int(numDescriptorsToKeep)*pointMap.size(), 48, CV_8UC1);
    uchar* dataPtr = descriptorPool[im].data;
    for(MapPoints::const_iterator it = pointMap.begin(); it != pointMap.end(); ++it) {
      if(loopClosureLandmarksToUseExclusively) {
        if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
          continue; // skip non-loop-closure points in this case
        }
      }
      // create landmark
      LandmarkToMatch landmarkToMatch;
      landmarkToMatch.is3d = false;

      // FoV check
      const Eigen::Vector4d hp_W = it->second.point;
      landmarkToMatch.p_W = hp_W.head<3>()/hp_W[3];
      const Eigen::Vector3d r_W = landmarkToMatch.p_W- T_WC1.r();
      const Eigen::Vector3d e_W = r_W.normalized();
      const double r = std::max(0.01, r_W.norm());

      const Eigen::Vector4d hp_C = T_CW1*hp_W;
      Eigen::Vector2d kp;
      const cameras::ProjectionStatus status = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)
                                                 ->projectHomogeneous(hp_C, &kp);
      if(status == cameras::ProjectionStatus::Invalid
          || status == cameras::ProjectionStatus::Behind) {
        continue;
      }

      if (kp[0] < -reprThreshold)
        continue;
      if (kp[1] < -reprThreshold)
        continue;
      if (kp[0] > maxU)
        continue;
      if (kp[1] > maxV)
        continue;

      landmarkToMatch.projection = kp;

      // distinguish whether to consider as 3D point or not.
      const double quality = it->second.quality;

      // obtain map point descriptor, and do some pruning.
      std::vector<double> bestScores(numDescriptorsToKeep, 1.0);
      landmarkToMatch.descriptors = cv::Mat(
          int(numDescriptorsToKeep), 48, CV_8UC1, dataPtr);
      landmarkToMatch.e_W.resize(3,numDescriptorsToKeep);
      landmarkToMatch.r_W.resize(3,numDescriptorsToKeep);
      landmarkToMatch.kids.reserve(numDescriptorsToKeep);
      const LandmarkId landmarkId = it->first;
      size_t descriptorCount = 0;
      for (auto obsiter = it->second.observations.rbegin();
           obsiter != it->second.observations.rend();
           ++obsiter) {

        const KeypointIdentifier kid = *obsiter;

        // remove some descriptors that are unlikely to match
        const kinematics::Transformation T_SC_old =
            *multiFrame->T_SC(kid.cameraIndex);
        const kinematics::Transformation T_WS_old =
            estimator.pose(StateId(kid.frameId));
        const kinematics::Transformation T_WC_old = T_WS_old * T_SC_old;
        const Eigen::Vector3d r_W_old = hp_W.head<3>()/hp_W[3] - T_WC_old.r();

        // check if 3D
        if(!landmarkToMatch.is3d) {
          const Eigen::Vector3d r_close_W = r_W-(0.2/focalLength/quality*r_W_old);
          const double cosA = r_W.normalized().dot(r_close_W.normalized());
          if(cosA > cos(10.0/focalLength)) {
            landmarkToMatch.is3d = true;
          }
        }

        // over 35 degree viewpoint change
        const double cosViewpointChnage = e_W.dot(r_W_old.normalized());
        if(cosViewpointChnage < cos(0.6)
            && !loopClosureLandmarksToUseExclusively) {
          continue;
        }

        // scale change over 50%
        const double scaleChange = fabs(r-r_W_old.norm())/r;
        if((scaleChange > 0.5)
            && !loopClosureLandmarksToUseExclusively) {
          continue;
        }

        const double score = 0.5*(acos(std::clamp(cosViewpointChnage, -1.0, 1.0))/0.6+scaleChange/0.5);
        double worstScore = 0.0;
        size_t worstIdx = 0;
        // find location in buffer to write to
        for(size_t n=0; n<numDescriptorsToKeep; ++n) {
          if(bestScores[n] > worstScore) {
            worstScore = bestScores[n];
            worstIdx = n;
          }
        }
        // store if better
        if(score<bestScores[worstIdx]) {
          // copy over descriptors
          const MultiFramePtr oldFrame =
              estimator.multiFrame(StateId(kid.frameId));
          std::memcpy(
              landmarkToMatch.descriptors.data+48*worstIdx,
              oldFrame->keypointDescriptor(
                  kid.cameraIndex, kid.keypointIndex), 48);

          // remember some other stuff for efficiency
          Eigen::Vector3d e_C;
          oldFrame->getBackProjection(kid.cameraIndex, kid.keypointIndex, e_C);
          landmarkToMatch.e_W.col(worstIdx) = T_WC_old.C()*e_C.normalized();
          landmarkToMatch.r_W.col(worstIdx) = T_WC_old.r();
          landmarkToMatch.kids.push_back(kid);

          // remember which were used
          descriptorCount = std::max(descriptorCount, worstIdx + 1);
          bestScores[worstIdx] = score;
        }
      }

      // No selected observation means the pool and ray columns are unwritten.
      // Exclude this landmark before either matching pass can read those bytes.
      if(descriptorCount == 0) {
        continue;
      }

      // crop unused bottom rows / right cols
      landmarkToMatch.descriptors = landmarkToMatch.descriptors(
          cv::Rect(0, 0, 48, descriptorCount));
      landmarkToMatch.e_W.conservativeResize(3, descriptorCount);
      landmarkToMatch.r_W.conservativeResize(3, descriptorCount);
      dataPtr += descriptorCount*48;

      // check classification
      if (it->second.classification == 10 || it->second.classification == 11) {
        landmarkToMatch.ignore = true;
      }

      // insert
      landmarksToMatch[landmarkId] = landmarkToMatch;
    }
    landmarksToMatchVec[im] = landmarksToMatch;

    // multithreaded matching
    const size_t num_matching_threads = size_t(params.frontend.num_matching_threads);

    std::vector<double> distances(numKeypoints,briskMatchingThreshold_);
    std::vector<LandmarkId> lmIds(numKeypoints);
    AlignedVector<Eigen::Vector4d> hps_W(numKeypoints, Eigen::Vector4d::Zero());
    std::vector<size_t> ctrs(num_matching_threads);
    std::vector<double> reprErrors(num_matching_threads);

    std::vector<std::thread*> threads(num_matching_threads, nullptr);
    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t] = new std::thread(
          &Frontend::matchToMapByThread<CAMERA_GEOMETRY>, this, t, num_matching_threads,
              std::cref(estimator), std::cref(params), currentFrameId,
              loopClosureLandmarksToUseExclusively, std::cref(T_WS1),
              std::cref(landmarksToMatch), numKeypoints,
              std::cref(pointMap), im, std::cref(multiFrame), std::ref(distances),
              std::ref(lmIds), std::ref(hps_W), std::ref(ctrs), std::ref(reprErrors));
    }

    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t]->join();
      delete threads[t];
      reprErr += reprErrors[t] * double(ctrs[t]);
      reprErrMatches += ctrs[t];
      if (ctrs[t] > 0) {
        ++reprErrContributors;
      }
    }

    // now insert observations
    for(size_t k = 0; k < numKeypoints; ++k) {
      uint64_t previousId = multiFrame->landmarkId(im,k);
      if(lmIds[k].isInitialised()) {

        if(previousId && loopClosureLandmarksToUseExclusively) {
          // remove
          estimator.removeObservation(StateId(currentFrameId), im, k);
          oldIds.push_back(LandmarkId(previousId));
          newIds.push_back(lmIds[k]);
        }

        multiFrame->setLandmarkId(im, k, lmIds[k].value());
        estimator.addObservation<CAMERA_GEOMETRY>(
              lmIds[k], StateId(currentFrameId), im, k);
        if (landmarksToMatch[lmIds[k]].ignore) {
          estimator.setObservationInformation(StateId(currentFrameId), im, k,
                                              Eigen::Matrix2d::Identity()*0.00001);
        }
        else {
          maybeBoostObservationInformation(im, k, multiFrame, lmIds[k],
                                          landmarksToMatch[lmIds[k]].p_W);
        }
        ctr++;
      }
    }
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "before ransac")
  // Average the final matched observations, independently of worker/camera
  // partitions. Discarded descriptor proposals supply no error evidence.
  reprErr = reprErrMatches > 0 ? reprErr / double(reprErrMatches) : 0.0;

  // remove outliers -- initialise pose only without IMU or when matching with large repr. err.
  MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));
  int numInitIter = 2;
  const bool ransacRemoveOutliers = true;
  bool runRansac = !params.imu.use;
  const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(0)->focalLengthU()
                        + multiFrame->geometryAs<CAMERA_GEOMETRY>(0)->focalLengthV());
  const double strictReprThreshold = 3.0 + f*0.006;
  if (reprErr > strictReprThreshold) {
    if (params.imu.use) {
      LOG(INFO) << "large reprojection error (" << reprErr << "): run RANSAC"
                << " valid_partitions=" << reprErrContributors
                << " total_partitions="
                << params.frontend.num_matching_threads * params.nCameraSystem.numCameras()
                << " matched_observations=" << reprErrMatches;
      runRansac = true;
    }
    numInitIter += 2;
  }
  bool secondRansac = false;
  if(runRansac) {
    const bool ransacSuccess = runRansac3d2d(estimator, multiFrame->cameraSystem(), multiFrame,
                                             runRansac, ransacRemoveOutliers);
    T_WS1 = estimator.pose(StateId(currentFrameId));
    if (!ransacSuccess) {
      numInitIter += 4;
      secondRansac = true;
    }
  }

  // do optimisation
  std::vector<StateId> updatedStatesRealtime;
  if(!loopClosureLandmarksToUseExclusively && ctr > 3) {
    estimator.optimiseRealtimeGraph(
        numInitIter, updatedStatesRealtime, params.estimator.realtime_num_threads,
        false, true, isInitialized_);
    /*int numInliers = */removeOutliers<CAMERA_GEOMETRY>(estimator,
                                    params.nCameraSystem,
                                    estimator.multiFrame(StateId(currentFrameId)));
    estimator.optimiseRealtimeGraph(
      2, updatedStatesRealtime, params.estimator.realtime_num_threads,
      false, true, isInitialized_);
    T_WS1 = estimator.pose(StateId(currentFrameId));
  }
  if (ctr <= 3 && isInitialized_) {
    secondRansac = true;
  }

  // now the non-initialised ones
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    // the current frame to match
    const MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));
    const size_t numKeypoints = multiFrame->numKeypoints(im);
    if(numKeypoints == 0) {
      continue; // no points -- bad!
    }

    // prepare landmarks as visible in this frame
    AlignedMap<LandmarkId, LandmarkToMatch> landmarksToMatch;
    const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
    const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
    const kinematics::Transformation T_CW1 = T_WC1.inverse();

    // multithreaded matching
    const size_t num_matching_threads = size_t(params.frontend.num_matching_threads);

    std::vector<double> distances(numKeypoints,briskMatchingThreshold_);
    std::vector<LandmarkId> lmIds(numKeypoints);
    AlignedVector<Eigen::Vector4d> hps_W(numKeypoints, Eigen::Vector4d::Zero());
    std::vector<size_t> ctrs(num_matching_threads);

    std::vector<std::thread*> threads(num_matching_threads, nullptr);
    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t] = new std::thread(
          &Frontend::matchToMapByThreadUnitialised<CAMERA_GEOMETRY>, this, t, num_matching_threads,
              std::cref(estimator), std::cref(params), currentFrameId,
              loopClosureLandmarksToUseExclusively, std::cref(T_WS1),
              std::cref(landmarksToMatchVec[im]), numKeypoints,
              std::cref(pointMap), im, std::cref(multiFrame), std::ref(distances),
              std::ref(lmIds), std::ref(hps_W), std::ref(ctrs));
    }

    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t]->join();
      delete threads[t];
    }

    // now insert observations
    for(size_t k = 0; k < numKeypoints; ++k) {
      uint64_t previousId = multiFrame->landmarkId(im,k);
      if(lmIds[k].isInitialised()) {

        if(previousId && loopClosureLandmarksToUseExclusively) {
          // remove
          estimator.removeObservation(StateId(currentFrameId), im, k);
          oldIds.push_back(LandmarkId(previousId));
          newIds.push_back(lmIds[k]);
        }

        // check bad reprojections into existing frames
        MapPoint2 mpt;
        estimator.getLandmark(lmIds[k], mpt);
        if (hps_W[k].norm() > 1.0e-22) {
          bool badReprojections = false;
          for (const auto &obs : mpt.observations) {
            Eigen::Vector2d ptp;
            Eigen::Vector2d pt;
            const auto &mf = estimator.multiFrame(StateId(obs.frameId));
            mf->getKeypoint(obs.cameraIndex, obs.keypointIndex, pt);
            const auto &cam = mf->geometryAs<CAMERA_GEOMETRY>(obs.cameraIndex);
            const kinematics::Transformation T_WS = estimator.pose(StateId(obs.frameId));
            const kinematics::Transformation T_SC = estimator.extrinsics(StateId(obs.frameId),
                                                                         obs.cameraIndex);
            //Eigen::Vector4d hpW = mpt.point;
            Eigen::Vector4d hpC = T_SC.inverse() * T_WS.inverse() * hps_W[k];
            auto s = cam->projectHomogeneous(hpC, &ptp);
            if (!(s == cameras::ProjectionStatus::Successful && (pt - ptp).norm() < 4.0)) {
              badReprojections = true;
              break;
            }
          }
          if (badReprojections) {
            continue;
          }
        }

        Eigen::Vector2d pt1;
        Eigen::Vector2d pt1p;
        multiFrame->getKeypoint(im, k, pt1);
        const auto &cam1 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im);
        if (hps_W[k].norm() > 1.0e-22 && !estimator.isLandmarkInitialised(lmIds[k])) { //ugly
          // check current reprojection
          auto s1 = cam1->projectHomogeneous(T_WC1.inverse() * hps_W[k], &pt1p);
          if (!(s1 == cameras::ProjectionStatus::Successful && (pt1 - pt1p).norm() < 4.0)) {
            continue;
          }

          // accept and set position
          estimator.setLandmark(lmIds[k], hps_W[k], true);
        } else {
          auto s1 = cam1->projectHomogeneous(T_WC1.inverse() * Eigen::Vector4d(mpt.point), &pt1p);
          if (!(s1 == cameras::ProjectionStatus::Successful && (pt1 - pt1p).norm() < 4.0)) {
            continue;
          }
        }

        // accept and set observation
        multiFrame->setLandmarkId(im, k, lmIds[k].value());
        estimator.addObservation<CAMERA_GEOMETRY>(
            lmIds[k], StateId(currentFrameId), im, k);
        if (landmarksToMatch[lmIds[k]].ignore) {
          estimator.setObservationInformation(StateId(currentFrameId), im, k,
                                              Eigen::Matrix2d::Identity()*0.00001);
        }
        else {
          maybeBoostObservationInformation(im, k, multiFrame, lmIds[k],
                                          landmarksToMatch[lmIds[k]].p_W);
        }
        ctr++;
      }
    }
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after non-initialised match to map")

  // merge landmarks, if loop-closure matching
  if(loopClosureLandmarksToUseExclusively) {
    estimator.mergeLandmarks(oldIds, newIds);
  }

  // final two steps optimisation
  if (secondRansac) {
    LOG(INFO) << "Running RANSAC also with uninitialised landmarks";
    const bool ransacSuccess = runRansac3d2d(estimator, multiFrame->cameraSystem(), multiFrame,
                                             secondRansac, ransacRemoveOutliers);
    T_WS1 = estimator.pose(StateId(currentFrameId));
    if (!ransacSuccess) {
      numInitIter += 4;
    }
    estimator.optimiseRealtimeGraph(
    numInitIter, updatedStatesRealtime, params.estimator.realtime_num_threads,
        false, true, isInitialized_);
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after match to map")

  return ctr;
}

// Match a new multiframe to existing keyframes:
template <class CAMERA_GEOMETRY>
void Frontend::matchToMapByThread(
    size_t threadIdx, size_t numThreads, const Estimator &estimator,
    const okvis::ViParameters& params, const uint64_t currentFrameId,
    const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
    const kinematics::Transformation& T_WS1,
    const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
    size_t numKeypoints, const MapPoints& pointMap,
    size_t im, const MultiFramePtr&  multiFrame, std::vector<double>& distances,
    std::vector<LandmarkId>& lmIds, AlignedVector<Eigen::Vector4d>& hps_W,
    std::vector<size_t>& ctrs,
    std::vector<double>& reprErrors) const {

  const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
  const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
  const kinematics::Transformation T_CW1 = T_WC1.inverse();

  const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
                   + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());

  const double reprojectionThreshold = params.imu.use ? 3.0+f*0.06 : 3.0+f*0.34;
  const double reprojectionThresholdSq = reprojectionThreshold * reprojectionThreshold;

  ctrs[threadIdx] = 0;

  // go through all landmarks
  const size_t segment = numKeypoints/numThreads;
  const size_t startK = segment*threadIdx;
  const size_t endK = threadIdx+1 == numThreads ? numKeypoints : startK + segment;
  std::vector<double> bestReprojectionErrorSq(endK - startK, 0.0);
  const uchar* ddata = multiFrame->keypointDescriptor(im, 0);
  Eigen::Matrix2Xd keypoints(2,numKeypoints);
  std::vector<bool> use(numKeypoints, true);
  for(size_t k = startK; k < endK; k++) {
    Eigen::Vector2d keypoint;
    multiFrame->getKeypoint(im, k, keypoint);
    keypoints.col(k) = keypoint;
    const uint64_t previousId = multiFrame->landmarkId(im,k);
    if(previousId&&!loopClosureLandmarksToUseExclusively) {
      use[k] = false; // I don't remember why this could happen -- just being paranoid.
      continue; // already matched
    }
  }
  // Each keypoint has its own winner. A sorted x range avoids testing every
  // keypoint for every landmark, while preserving landmark/descriptor order.
  // Use the original squared-distance arithmetic for the broad phase as well;
  // no approximate pixel bins or rounded radius can exclude a boundary match.
  std::vector<size_t> keypointOrder(endK - startK);
  bool canIndex = std::isfinite(reprojectionThresholdSq);
  for(size_t k = startK; k < endK; ++k) {
    keypointOrder[k - startK] = k;
    canIndex = canIndex && keypoints.col(k).allFinite();
  }
  if(canIndex) {
    std::sort(keypointOrder.begin(), keypointOrder.end(),
              [&](size_t a, size_t b) { return keypoints(0, a) < keypoints(0, b); });
  }
  for(auto it = landmarksToMatch.begin(); it != landmarksToMatch.end(); ++it) {

    if(!it->second.is3d) {
      continue;
    }

    if(loopClosureLandmarksToUseExclusively) {
      if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
        continue; // skip non-loop-closure points in this case
      }
    }

    // match all present descriptors
    const Eigen::Vector2d projection = it->second.projection;
    auto first = keypointOrder.begin();
    auto last = keypointOrder.end();
    if(canIndex && projection.allFinite()) {
      first = std::partition_point(first, last, [&](size_t k) {
        const double dx = projection[0] - keypoints(0, k);
        return keypoints(0, k) < projection[0] && dx * dx > reprojectionThresholdSq;
      });
      last = std::partition_point(first, last, [&](size_t k) {
        const double dx = projection[0] - keypoints(0, k);
        return keypoints(0, k) <= projection[0] || !(dx * dx > reprojectionThresholdSq);
      });
    }
    // Non-finite geometry uses the full range, retaining the original policy.
    for(auto candidate = first; candidate != last; ++candidate) {
      const size_t k = *candidate;

      if(!use[k]) {
        continue;
      }

      // also check image distance, unless tracking lost.
      const Eigen::Vector2d reprDist = projection - keypoints.col(k);
      const double reprDistSq = reprDist.dot(reprDist);
      if (reprDistSq > reprojectionThresholdSq) {
        continue;
      }

      const uchar* descriptorK = ddata + k*48;
      for(int d = 0; d<it->second.descriptors.rows; ++d) {
        const double dist = brisk::Hamming::PopcntofXORed(
            descriptorK,
            it->second.descriptors.data + d*48, 3);
        if(dist < distances[k]) {
          distances[k] = dist;
          lmIds[k] = it->first;
          bestReprojectionErrorSq[k - startK] = reprDistSq;
        }
      }
    }
  }
  // A descriptor can improve several times before its final landmark is
  // selected. Count that winner once, after all proposals have been visited.
  for(size_t k = startK; k < endK; ++k) {
    if(lmIds[k].isInitialised()) {
      ++ctrs[threadIdx];
      reprErrors[threadIdx] += std::sqrt(bestReprojectionErrorSq[k - startK]);
    }
  }
  reprErrors[threadIdx] = ctrs[threadIdx] > 0
      ? reprErrors[threadIdx] / double(ctrs[threadIdx]) : 0.0;
}

// Match a new multiframe to existing keyframes:
template <class CAMERA_GEOMETRY>
void Frontend::matchToMapByThreadUnitialised(
    size_t threadIdx, size_t numThreads, const Estimator &estimator,
    const okvis::ViParameters& params, const uint64_t currentFrameId,
    const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
    const kinematics::Transformation& T_WS1,
    const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
    size_t numKeypoints, const MapPoints& pointMap,
    size_t im, const MultiFramePtr&  multiFrame, std::vector<double>& distances,
    std::vector<LandmarkId>& lmIds, AlignedVector<Eigen::Vector4d>& hps_W,
    std::vector<size_t>& ctrs) const {

  const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
  const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
  const kinematics::Transformation T_CW1 = T_WC1.inverse();

  ctrs[threadIdx] = 0;

  const double focalLength =
      0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
      + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());

  // go through all landmarks
  const size_t segment = numKeypoints/numThreads;
  const size_t startK = segment*threadIdx;
  const size_t endK = threadIdx+1 == numThreads ? numKeypoints : startK + segment;
  const uchar* ddata = multiFrame->keypointDescriptor(im, 0);
  Eigen::Matrix3Xd e_Ws(3,numKeypoints);
  std::vector<uint64_t> previousIds(numKeypoints,0);
  std::vector<bool> use(numKeypoints,false);
  for(size_t k = startK; k < endK; k++) {
    Eigen::Vector3d e1_C;
    if(multiFrame->getBackProjection(im, k, e1_C)){
      const Eigen::Vector3d e1_W = T_WC1.C()*e1_C.normalized();
      e_Ws.col(k) = e1_W;
      const uint64_t previousId = multiFrame->landmarkId(im,k);
      previousIds[k] = previousId;
      if(previousId&&!loopClosureLandmarksToUseExclusively) {
        continue; // already matched
      }
      use[k] = true;
    }
  }
  const double sigma = 1.0/focalLength;
  const double cos6Sigma = cos(6.0*sigma);
  for(auto it = landmarksToMatch.begin(); it != landmarksToMatch.end(); ++it) {

    if(it->second.is3d) {
      continue;
    }

    if(loopClosureLandmarksToUseExclusively) {
      if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
        continue; // skip non-loop-closure points in this case
      }
    }

    // match all present descriptors
    for(size_t k = startK; k < endK; k++) {

      if(!use[k]) {
        continue;
      }

      // also check epipolar distance (later)
      const Eigen::Vector3d e1_W=e_Ws.col(k);
      const uchar* descriptorK = ddata + k*48;
      for(int d = 0; d<it->second.descriptors.rows; ++d) {
        const double dist = brisk::Hamming::PopcntofXORed(
            descriptorK, it->second.descriptors.data + d*48, 3);

        if(dist < distances[k]) {

          // epipolar distance check
          const Eigen::Vector3d e0_W = it->second.e_W.col(d);
          const Eigen::Vector3d r0_W = it->second.r_W.col(d);

          if(e0_W.dot(e1_W)<cos6Sigma) { // otherwise parallel... will be OK.
            const Eigen::Vector3d et_W = (T_WC1.r() - r0_W).normalized();
            const Eigen::Vector3d n0_W = e0_W.cross(et_W).normalized();
            const Eigen::Vector3d n1_W = e1_W.cross(et_W).normalized();
            if((n0_W.dot(n1_W) < cos6Sigma)) {
              continue; // not in epipolar plane
            }
            if((e0_W.cross(e1_W)).dot((n0_W + n1_W).normalized())>0.0) {
              continue; // divergent rays
            }
          }

          // try triangulation
          bool isValid = false;
          bool isParallel = false;
          Eigen::Vector4d hp_W = triangulation::triangulateFast(
              r0_W, e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

          if(!isValid) {
            continue;
          }

          // check if too close (out of focus)
          const Eigen::Vector3d p_W = hp_W.head<3>()/hp_W[3];
          if((p_W-r0_W).norm() < 0.2) {
            isValid = false;
          }
          if((p_W-T_WC1.r()).norm() < 0.2) {
            isValid = false;
          }
          if(!isValid) {
            continue;
          }

          if(it->first.value()==previousIds[k]) {
            ctrs[threadIdx]++; // still counts, already correct match...
            break; // the match is already done...
          }

          distances[k] = dist;
          lmIds[k] = it->first;
          ctrs[threadIdx]++;
          if(!isParallel) {
            hps_W[k] = hp_W;
          }
        }
      }
    }
  }
}

/// \brief Temporary match info storage.
struct MatchInfo {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector4d hp_W; ///< 3D point.
  size_t k1 = 0; ///< Match idx.
  bool matching = false; ///< Does it match?
  bool initialisable = false; ///< Initialisable?
  double quality = 0.0; ///< 3D quality.
};

template <class CAMERA_GEOMETRY>
int Frontend::matchMotionStereo(Estimator& estimator, const ViParameters &params,
                                 const uint64_t currentFrameId, bool& rotationOnly) {
  int retCtr = 0;
  rotationOnly = true;

  kinematics::Transformation T_WS1 = estimator.pose(StateId(currentFrameId));

  // find close frames
  TimerSwitchable matchMotionStereoTimer2("2.02.1 match motion stereo: prepare");
  std::set<StateId> allFrames;
  allFrames.insert(estimator.keyFrames().begin(), estimator.keyFrames().end());
  allFrames.insert(estimator.imuFrames().begin(), estimator.imuFrames().end());
  for(const auto & id : estimator.imuFrames()) {
    if(!estimator.isKeyframe(id)) {
      allFrames.erase(id);
    }
  }
  StateId previousFrameId = estimator.stateIdByAge(1);
  std::vector<std::pair<double, StateId>> overlaps;
  for(auto & id : allFrames) {
    if(id == previousFrameId) {
      overlaps.push_back(std::pair<double, StateId>(1.0, id));
      continue;
    }
    const double overlap = estimator.overlapFraction(
          estimator.multiFrame(previousFrameId),estimator.multiFrame(id));
    overlaps.push_back(std::pair<double, StateId>(overlap, id));
  }
  std::sort(overlaps.begin(), overlaps.end());
  std::vector<StateId> matchFrameIds;

  for(size_t i=0; i<overlaps.size(); ++i) {
    if(overlaps[overlaps.size()-i-1].first <= 1.0e-8) break;
    matchFrameIds.push_back(overlaps[overlaps.size()-i-1].second);
  }

  matchMotionStereoTimer2.stop();

  kinematics::Transformation T_WS0;
  bool firstFrame = true;
  for (auto olderFrameId : matchFrameIds) {
    T_WS0 = estimator.pose(olderFrameId);
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      const kinematics::Transformation T_SC0 = estimator.extrinsics(StateId(olderFrameId), im);
      const kinematics::Transformation T_SC1 = estimator.extrinsics(StateId(currentFrameId), im);
      const kinematics::Transformation T_WC0 = T_WS0 * T_SC0;
      const kinematics::Transformation T_WC1 = T_WS1 * T_SC1;
      // match
      MultiFramePtr multiFrame0 = estimator.multiFrame(olderFrameId);
      MultiFramePtr multiFrame1 = estimator.multiFrame(StateId(currentFrameId));
      const size_t k0Size = multiFrame0->numKeypoints(im);
      const size_t k1Size = multiFrame1->numKeypoints(im);
      const auto camera = multiFrame0->geometryAs<CAMERA_GEOMETRY>(im);
      const double f0 = 0.5* (camera->focalLengthU() + camera->focalLengthV());

      // preprocess matchable set, get descriptors close in memory
      std::vector<size_t> k1s;
      k1s.reserve(k1Size);
      cv::Mat desc1(k1Size, 48, CV_8UC1);
      for(size_t k1 = 0; k1 < k1Size; ++k1) {
        const uint64_t id1 = multiFrame1->landmarkId(im, k1);
        if(id1) {
          continue; // already matched
        }
        std::memcpy(
            desc1.data+48*k1s.size(),
            multiFrame1->keypointDescriptor(im, k1), 48);
        k1s.push_back(k1);
      }
      desc1 = desc1(cv::Rect(0,0,48,k1s.size()));

      AlignedVector<MatchInfo> matchInfos(k0Size);

      // vector container stores threads
      std::vector<std::thread> workers;
      for (size_t t = 0; t < size_t(params.frontend.num_matching_threads); t++) {
        workers.push_back(std::thread([this, t, k0Size, im, &multiFrame0, &estimator, f0, k1s,
                                      &T_WC0, &T_WC1, &multiFrame1, &olderFrameId, &matchInfos,
                                       &params, &camera, desc1]() {
          for(size_t k0 = t; k0 < k0Size; k0 += size_t(params.frontend.num_matching_threads)) {
            uint64_t id0 = multiFrame0->landmarkId(im, k0);
            if(id0) {
              if(!estimator.isLandmarkAdded(LandmarkId(id0))) {
                continue; // weird
              }
              if(estimator.isLandmarkInitialised(LandmarkId(id0))) {
                continue; // already matched
              }
            }

            uint32_t distances = briskMatchingThreshold_;
            bool initialisable = false;
            double quality = 0.0;
            Eigen::Vector4d hps_W(0,0,0,0);
            size_t k1_max=1000;

            // pre-fetch frame 0 stuff
            const uchar* d0 = multiFrame0->keypointDescriptor(im, k0);
            double size0;
            multiFrame0->getKeypointSize(im, k0, size0);
            Eigen::Vector2d pt0;
            multiFrame0->getKeypoint(im, k0, pt0);
            Eigen::Vector3d e0_C;
            if(!multiFrame0->getBackProjection(im, k0, e0_C)) continue;
            const Eigen::Vector3d e0_W = (T_WC0.C()*e0_C).normalized();
            const double sigma = size0/f0 * 0.125;

            if(estimator.isObserved(KeypointIdentifier{olderFrameId.value(), im, k0})) {
              continue; // already matched
            }

            for(size_t kk = 0; kk < k1s.size(); ++kk) {
              const size_t k1 = k1s[kk];
              const uint32_t dist = brisk::Hamming::PopcntofXORed(
                  d0, desc1.data+kk*48, 3);
              if(dist < distances) {
                // it's a match!

                // triangulate
                bool isValid = false;
                bool isParallel = false;
                Eigen::Vector3d e1_C;
                if(!multiFrame1->getBackProjection(im, k1, e1_C)) continue;
                const Eigen::Vector3d e1_W = (T_WC1.C()*e1_C).normalized();
                if(e0_W.dot(e1_W) < 0.5) continue;

                Eigen::Vector4d hp_W = triangulation::triangulateFast(
                      T_WC0.r(), e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

                if(!isValid) {
                  continue;
                }

                // check if too close (out of focus)
                const Eigen::Vector4d hp_C0 = (T_WC0.inverse()*hp_W);
                const Eigen::Vector4d hp_C1 = (T_WC1.inverse()*hp_W);

                if(e0_W.transpose()*e1_W < 0.8) {
                  isValid = false;
                }

                hp_W = hp_W/hp_W[3];
                if(hp_C0[2]/hp_C0[3] < 0.2) {
                  isValid = false;
                }
                if(hp_C1[2]/hp_C1[3] < 0.2) {
                  isValid = false;
                }

                // remember
                if(/*dist<distances && */isValid) {
                  k1_max = k1;
                  distances=dist;
                  quality = acos(std::clamp(
                      (hp_W.head<3>()-T_WC0.r()).normalized()
                          .dot((hp_W.head<3>()-T_WC1.r()).normalized()),
                      -1.0, 1.0));
                  hps_W = hp_W;
                  initialisable = !isParallel;
                }
              }
            }

            // add observations and initialise
            if(distances < briskMatchingThreshold_) {
              Eigen::Vector2d pt1p;
              Eigen::Vector2d pt1;
              multiFrame1->getKeypoint(im, k1_max, pt1);
              auto s1 = camera->projectHomogeneous(T_WC1.inverse()*hps_W, &pt1p);
              if(s1 == cameras::ProjectionStatus::Successful && (pt1-pt1p).norm()<4.0) {
                matchInfos[k0] = MatchInfo{hps_W, k1_max, true, initialisable, quality};
              }
            }
          }
        }));
      }

      // join all matcher threads
      std::for_each(workers.begin(), workers.end(), [](std::thread &worker) {
          worker.join();
      });

      // finally insert the actual matches
      for(size_t k0=0; k0<k0Size; ++k0) {
        const MatchInfo & mInfo = matchInfos.at(k0);
        uint64_t id0 = multiFrame0->landmarkId(im, k0);
        if(!mInfo.matching) {
          continue;
        }

        if(id0) {
          if(estimator.isLandmarkInitialised(LandmarkId(id0))) {
            continue; // already matched
          }
          if(!estimator.isLandmarkAdded(LandmarkId(id0))) {
            continue; // weird!
          }
        }
        if(estimator.isObserved(KeypointIdentifier{olderFrameId.value(), im, k0})) {
          continue; // already matched
        }

        const uint64_t id1 = multiFrame1->landmarkId(im, mInfo.k1);
        if(id1) {
          continue; // already matched
        }

        if(id0){
          MapPoint2 lm;
          estimator.getLandmark(LandmarkId(id0), lm);
          if(lm.quality<mInfo.quality) {
            estimator.setLandmark(LandmarkId(id0), mInfo.hp_W, mInfo.initialisable);
          }
        } else {
          // Optional motion-consistency gate: only block *new* landmark creation when
          // the correspondence violates epipolar consistency induced by the current
          // relative pose estimate T_C1_C0 = T_WC1.inverse() * T_WC0.
          //
          // This check does not rely on assumed depth. We back-project both keypoints
          // to bearing vectors (cam0, cam1), evaluate the epipolar residual
          //   r = x1^T [t]_x R x0,
          // and reject if |r| exceeds a threshold derived from the configured
          // pixel tolerance (max_pixel / focal_length).
          if (params.garlileo.landmark_motion_consistency_enable
              && params.garlileo.landmark_motion_consistency_max_pixel > 0.0) {
            Eigen::Vector3d e0_C_obs;
            Eigen::Vector3d e1_C_obs;
            if (multiFrame0->getBackProjection(im, k0, e0_C_obs)
                && multiFrame1->getBackProjection(im, mInfo.k1, e1_C_obs)) {
              const okvis::kinematics::Transformation T_C1_C0 = T_WC1.inverse() * T_WC0;
              const Eigen::Vector3d x0 = e0_C_obs.normalized();
              const Eigen::Vector3d x1 = e1_C_obs.normalized();
              const Eigen::Vector3d Rx0 = T_C1_C0.C() * x0;
              const Eigen::Vector3d txRx0 = T_C1_C0.r().cross(Rx0);
              const double epi_residual = std::abs(x1.dot(txRx0));
              const double focal_for_gate = std::max(
                  1.0,
                  0.5 * (multiFrame1->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
                         + multiFrame1->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV()));
              const double epi_threshold =
                  params.garlileo.landmark_motion_consistency_max_pixel / focal_for_gate;
              if (epi_residual > epi_threshold) {
                // Inconsistent epipolar geometry -> drop this match for new-landmark creation.
                continue;
              }
            }
          }
          id0 = estimator.addLandmark(mInfo.hp_W, mInfo.initialisable).value();
          multiFrame0->setLandmarkId(im, k0, id0);
          OKVIS_ASSERT_TRUE_DBG(Exception, estimator.isLandmarkAdded(LandmarkId(id0)),
                              id0<<" not added, bug")
          estimator.addObservation<CAMERA_GEOMETRY>(LandmarkId(id0), StateId(olderFrameId), im, k0);
        }

        multiFrame1->setLandmarkId(im, mInfo.k1, id0);
        estimator.addObservation<CAMERA_GEOMETRY>(
              LandmarkId(id0), StateId(currentFrameId), im, mInfo.k1);
        retCtr++;
      }
    }

    bool rotationOnly_tmp = false;
    static const bool removeOutliers = true;

    // do RANSAC 2D2D for initialization only
    const bool initialisePose =  (!isInitialized_);
    if(!isInitialized_) {
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId.value(), initialisePose,
                    removeOutliers, rotationOnly_tmp);
    }

    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }
  }

  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
template <class CAMERA_GEOMETRY>
void Frontend::matchStereo(Estimator &estimator, std::shared_ptr<okvis::MultiFrame> multiFrame,
                           const okvis::ViParameters& params, bool asKeyframe) {
  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  // needed later:
  kinematics::Transformation T_WS = estimator.pose(StateId(mfId));

  for (size_t im0 = 0; im0 < camNumber; im0++) {
    const kinematics::Transformation T_SC0 = *multiFrame->T_SC(im0);

    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) {
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap
      if (!multiFrame->hasOverlap(im0, im1)) {
        continue;
      }

      // useful later:
      const kinematics::Transformation T_SC1 = *multiFrame->T_SC(im1);
      const kinematics::Transformation T_WC0 = T_WS * T_SC0;
      const kinematics::Transformation T_WC1 = T_WS * T_SC1;

      {
        // match
        MultiFramePtr multiFrame = estimator.multiFrame(StateId(mfId));
        const size_t k0Size = multiFrame->numKeypoints(im0);
        const size_t k1Size = multiFrame->numKeypoints(im1);
        const auto camera0 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im0);
        const auto camera1 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im1);
        const double f0 = 0.5* (camera0->focalLengthU() + camera0->focalLengthV());
        const double f1 = 0.5* (camera1->focalLengthU() + camera1->focalLengthV());
        for(size_t k0 = 0; k0 < k0Size; ++k0) {

          double distances = briskMatchingThreshold_;
          bool initialisable=  false;
          Eigen::Vector4d hps_W;
          size_t k1_match = 0;

          for(size_t k1 = 0; k1 < k1Size; ++k1) {
            const auto dist = brisk::Hamming::PopcntofXORed(
                multiFrame->keypointDescriptor(im0, k0),
                  multiFrame->keypointDescriptor(im1, k1), 3);
            if(dist < distances) {
              // it's a match!
              double size0, size1;
              multiFrame->getKeypointSize(im0, k0, size0);
              multiFrame->getKeypointSize(im1, k1, size1);
              Eigen::Vector2d pt0, pt1;
              multiFrame->getKeypoint(im0, k0, pt0);
              multiFrame->getKeypoint(im1, k1, pt1);
              const double sigma = std::max(size0/f0, size1/f1) * 0.125;

              // triangulate
              bool isValid = false;
              bool isParallel = false;
              Eigen::Vector3d e0_C, e1_C;
              if(!multiFrame->getBackProjection(im0, k0, e0_C)) continue;
              if(!multiFrame->getBackProjection(im1, k1, e1_C)) continue;
              Eigen::Vector3d e0_W = (T_WC0.C()*e0_C).normalized();
              Eigen::Vector3d e1_W = (T_WC1.C()*e1_C).normalized();
              Eigen::Vector4d hp_W = triangulation::triangulateFast(
                    T_WC0.r(), e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

              // check if too close
              const Eigen::Vector4d hp_C0 = (T_WC0.inverse()*hp_W);
              const Eigen::Vector4d hp_C1 = (T_WC1.inverse()*hp_W);

              hp_W = hp_W/hp_W[3];

              if(hp_C0[2]/hp_C0[3] < 0.1) {
                isValid = false;
              }
              if(hp_C1[2]/hp_C1[3] <  0.1) {
                isValid = false;
              }

              if(e0_W.transpose()*e1_W < 0.8) {
                isValid = false;
              }

              // add observations and initialise
              if(isValid) {
                distances = dist;
                hps_W = hp_W;
                k1_match = k1;
                initialisable = !isParallel;
              }
            }
          }

          if(distances<briskMatchingThreshold_) {
            Eigen::Vector2d pt0, pt1;
            multiFrame->getKeypoint(im0, k0, pt0);
            multiFrame->getKeypoint(im1, k1_match, pt1);
            uint64_t lmId = 0;
            const uint64_t id0 = multiFrame->landmarkId(im0, k0); // may change!!
            uint64_t id1 = multiFrame->landmarkId(im1, k1_match);
            bool add0 = false;
            bool add1 = false;
            if(id0 && id1) {
              if (id0 != id1) {
                estimator.mergeLandmark(LandmarkId(id1), LandmarkId(id0));
                id1 = id0;
              }
              if(!estimator.isLandmarkInitialised(LandmarkId(id0))) {
                // only re-assess initialisation
                if(initialisable) {
                  //estimator.setLandmarkInitialized(id0, initialiseable);
                  estimator.setLandmark(LandmarkId(id0), hps_W, true); /// \todo check true
                }
              } // else we do nothing, because already initialised and matched.
            } else if(id1) {
              // only add observation into frame0
              lmId = id1;
              add0 = true;

            } else if(id0) {
              // only add observation into frame1
              lmId = id0;
              add1 = true;
            } else {
              if(!asKeyframe){
                continue; // we don't want to create new stuff from non-keyframes
              }
              add0 = true;
              add1 = true;
              // need new point

              lmId = estimator.addLandmark(hps_W, initialisable).value();
              OKVIS_ASSERT_TRUE_DBG(
                  Exception, estimator.isLandmarkAdded(LandmarkId(lmId)),
                  lmId<<" not added, bug")
            }
            if(add0) {
              // verify (again, because landmark may not have been reset)
              Eigen::Vector2d pt0p;
              MapPoint2 mapPoint;
              estimator.getLandmark(LandmarkId(lmId), mapPoint);
              Eigen::Vector4d hp_eff_W = mapPoint.point;
              auto s0 = camera0->projectHomogeneous(T_WC0.inverse()*hp_eff_W, &pt0p);
              if(s0 == cameras::ProjectionStatus::Successful && (pt0-pt0p).norm()<4.0) {
                // safe to add.
                multiFrame->setLandmarkId(im0, k0, lmId);
                estimator.addObservation<CAMERA_GEOMETRY>(LandmarkId(lmId), StateId(mfId), im0, k0);
              }
            }
            if(add1) {
              // verify (again, because landmark may not have been reset)
              Eigen::Vector2d pt1p;
              MapPoint2 mapPoint;
              estimator.getLandmark(LandmarkId(lmId), mapPoint);
              Eigen::Vector4d hp_eff_W = mapPoint.point;
              auto s1 = camera1->projectHomogeneous(T_WC1.inverse()*hp_eff_W, &pt1p);
              if(s1 == cameras::ProjectionStatus::Successful && (pt1-pt1p).norm()<4.0) {
                multiFrame->setLandmarkId(im1, k1_match, lmId);
                estimator.addObservation<CAMERA_GEOMETRY>(
                  LandmarkId(lmId), StateId(mfId), im1, k1_match);
              }
            }
          }
        }
      }
    }
  }

  // TODO: for more than 2 cameras check that there were no duplications!

  // TODO: ensure 1-1 matching.
}
template<class CAMERA_GEOMETRY>
int Frontend::removeOutliers(Estimator &estimator,
                             const okvis::cameras::NCameraSystem &nCameraSystem,
                             std::shared_ptr<okvis::MultiFrame> currentFrame)
{
  const size_t camNumber = currentFrame->numFrames();
  const uint64_t mfId = currentFrame->id();

  // needed later:
  kinematics::Transformation T_WS = estimator.pose(StateId(mfId));

  int ctr = 0;

  for (size_t im = 0; im < camNumber; im++) {
    const kinematics::Transformation T_SC = *currentFrame->T_SC(im);
    const kinematics::Transformation T_WCi = T_WS * T_SC;
    const kinematics::Transformation T_CiW = T_WCi.inverse();
    const size_t kSize = currentFrame->numKeypoints(im);
    for (size_t k = 0; k < kSize; ++k) {
      uint64_t lmId = currentFrame->landmarkId(im, k);
      if (lmId) {
        Eigen::Vector2d pt, proj;
        if (currentFrame->getKeypoint(im, k, pt)) {
          //Eigen::Vector3d e_Ci;
          //if (currentFrame->getBackProjection(im, k, e_Ci)) {
          MapPoint2 lm;
          if (estimator.getLandmark(LandmarkId(lmId), lm)) {
            bool remove = false;
            const Eigen::Vector4d hp_W = lm.point;
            Eigen::Vector4d hp_Ci = T_CiW * hp_W;
            const auto camera = currentFrame->geometryAs<CAMERA_GEOMETRY>(im);
            if (cameras::ProjectionStatus::Successful == camera->projectHomogeneous(hp_Ci, &proj)) {
              if ((proj - pt).norm() > 4.0) {
                remove = true;
              }
            } else {
              remove = true;
            }
            if (!remove) {
              ctr++;
            } else {
              estimator.removeObservation(StateId(mfId), im, k);
            }
          }
        }
      }
    }
  }
  return ctr;
}

// Perform 3D/2D RANSAC.
bool Frontend::runRansac3d2d(
    Estimator &estimator, const okvis::cameras::NCameraSystem& nCameraSystem,
    std::shared_ptr<okvis::MultiFrame> currentFrame, bool initializePose, bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return false;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(
    estimator, nCameraSystem, currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences < 10) return int(numCorrespondences);

  // create a RelativePoseSac problem and RANSAC
  typedef opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem<
        opengv::absolute_pose::FrameNoncentralAbsoluteAdapter> AbsoluteModel;
  opengv::sac::Ransac<AbsoluteModel> ransac;
  std::shared_ptr<AbsoluteModel> absposeproblem_ptr(
        new AbsoluteModel(adapter, AbsoluteModel::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 16;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);

  // deal with outliers and assign transformation
  numInliers = int(ransac.inliers_.size());
  if (numInliers >= 10 && double(ransac.inliers_.size())/double(numCorrespondences)>0.7) {
    // kick out outliers:
    if(removeOutliers) {
      std::vector<bool> inliers(numCorrespondences, false);
      for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
        inliers.at(size_t(ransac.inliers_.at(k))) = true;
      }

      for (size_t k = 0; k < numCorrespondences; ++k) {
        if (!inliers[k]) {
          // get the landmark id:
          size_t camIdx = size_t(adapter.camIndex(k));
          size_t keypointIdx = size_t(adapter.keypointIndex(k));

          // remove observation
          estimator.removeObservation(StateId(currentFrame->id()), camIdx, keypointIdx);
        }
      }
    }

    // assign transformation
    Eigen::Matrix4d T_WS_mat = Eigen::Matrix4d::Identity();
    T_WS_mat.topLeftCorner<3, 4>() = ransac.model_coefficients_;
    kinematics::Transformation T_WS = kinematics::Transformation(T_WS_mat);
    if(initializePose) {
      estimator.setPose(StateId(currentFrame->id()), T_WS);
    }
    return true;
  } else {
    LOG(INFO) << "RANSAC FAIL: " << numInliers << " inliers, ratio = "
              << double(ransac.inliers_.size())/double(numCorrespondences);
  }
  return false;
}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(Estimator &estimator, const okvis::ViParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose, bool removeOutliers, bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  int totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {
    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(estimator, params.nCameraSystem,
                                                        olderFrameId, im, currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few corresp. are inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem
        FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(
          new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = int(rotation_only_ransac.inliers_.size());
    float rotation_only_ratio = float(rotation_only_inliers) / float(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem
        FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
          new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;  //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = int(rel_pose_ransac.inliers_.size());
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8f) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rotationOnly = true;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(size_t(rotation_only_ransac.inliers_.at(k))) = true;
      }
    } else {
      if (rel_pose_inliers > 10 && rel_pose_ratio > 0.8f) {
        rel_pose_success = true;
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
        inliers.at(size_t(rel_pose_ransac.inliers_.at(k))) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(StateId(currentFrameId));
    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (removeOutliers && !inliers[k]) {
        uint64_t lmIdB = multiFrame->landmarkId(im, idxB);
        if(lmIdB !=0) {
          estimator.removeObservation(StateId(currentFrameId), im, idxB);
        }
      }
    }

    // initialize pose if necessary
    if (initializePose && !isInitialized_) {
      if (rel_pose_success) {
        //LOG(INFO) << "Initializing pose from 2D-2D RANSAC"; #Sebastian
      } else {
        //LOG(INFO) << "Initializing pose from 2D-2D RANSAC: orientation only";
      }
    }
  }

  if (rel_pose_success || rotation_only_success) {
    return totalInlierNumber;
  }

  rotationOnly = true;  // hack...
  return -1;

}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at
// startup.
void Frontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  //mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectors_.push_back(std::shared_ptr<cv::FeatureDetector>(
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
            briskDetectionThreshold_, briskDetectionOctaves_,
            briskDetectionAbsoluteThreshold_, briskDetectionMaximumKeypoints_)));
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>(
        new brisk::BriskDescriptorExtractor(
            briskDescriptionRotationInvariance_, briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->unlock();
  }
}

bool applyStereoDnnImageEnhancement(const cv::Mat& image_in,
                                   const FrontendParameters& fp,
                                   cv::Mat& image_out) {
  image_out.release();
  if (image_in.empty()) {
    return false;
  }
  cv::Mat work = image_in;
  if (fp.vignette_correction_enable) {
    cv::Mat v;
    if (vignetteCorrectImageU8(work, fp.vignette_correction_k1, fp.vignette_correction_k2,
                               fp.vignette_correction_k3, v)
        && !v.empty()) {
      work = v;
    }
  }
  const bool do_enh = fp.agcwd_enable || fp.clahe_enable;
  if (!do_enh) {
    image_out = work.clone();
    return true;
  }
  cv::Mat enhanced;
  bool ok = false;
  ok = enhanceImageAgcwdClaheFusion(work, fp.agcwd_enable, fp.agcwd_weighting_param,
                                    fp.clahe_enable, fp.clahe_clip_limit, fp.clahe_tile_grid_size,
                                    enhanced);
  if (ok && !enhanced.empty()) {
    image_out = enhanced;
    return true;
  }
  image_out = work.clone();
  return true;
}

}  // namespace okvis
