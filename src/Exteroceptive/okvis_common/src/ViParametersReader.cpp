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
 * @file ViParametersReader.cpp
 * @brief Source file for the VioParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <glog/logging.h>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/EucmCamera.hpp>

#include <opencv2/core/core.hpp>

#include <okvis/ViParametersReader.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {
/// \brief Prior sigma on b_g [rad/s] when the gyro bias is not estimated: tight enough to pin
/// b_g to g0 (6e-5 deg/s), loose enough to keep the prior information matrix well conditioned.
constexpr double kPinnedSigmaBg = 1.0e-6;
/// \brief Gyro drift noise density [rad/s^2/sqrt(Hz)] when the gyro bias is not estimated:
/// the b_g random walk over a 100 s sequence is then ~1e-9 rad/s, i.e. b_g stays at g0.
constexpr double kPinnedSigmaGwC = 1.0e-10;
/// \brief Prior sigma on b_a [m/s^2] when the accelerometer bias is not estimated.
constexpr double kPinnedSigmaBa = 1.0e-6;
/// \brief Accelerometer drift noise density [m/s^2/sqrt(Hz)] when b_a is not estimated.
constexpr double kPinnedSigmaAwC = 1.0e-10;
}  // namespace

namespace {
inline std::string trimCopy(std::string s) {
  auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
  return s;
}

inline bool parseBoolToken(std::string token, bool& out) {
  token = trimCopy(std::move(token));
  // remove quotes if present
  if (token.size() >= 2
      && ((token.front() == '"' && token.back() == '"')
          || (token.front() == '\'' && token.back() == '\''))) {
    token = token.substr(1, token.size() - 2);
  }
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (token == "0" || token == "false" || token == "no" || token == "n" || token == "off") {
    out = false;
    return true;
  }
  if (token == "1" || token == "true" || token == "yes" || token == "y" || token == "on") {
    out = true;
    return true;
  }
  return false;
}

// OpenCV FileStorage YAML parser can misinterpret inline comments containing ':' and turn a scalar
// into a MAP node. For a small set of boolean flags, fall back to a simple line-based parse.
inline bool tryParseBoolFromYamlText(const std::string& filename,
                                     const std::string& key,
                                     bool& out) {
  std::ifstream in(filename);
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    // strip comments
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = trimCopy(std::move(line));
    if (line.empty()) {
      continue;
    }
    // allow indentation
    if (line.rfind(key, 0) != 0) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string v = trimCopy(line.substr(colon + 1));
    if (v.empty()) {
      continue;
    }
    // take first token
    const auto sp = v.find_first_of(" \t");
    const std::string token = (sp == std::string::npos) ? v : v.substr(0, sp);
    return parseBoolToken(token, out);
  }
  return false;
}
}  // namespace

// The default constructor.
ViParametersReader::ViParametersReader()
    : readConfigFile_(false) {
}

// The constructor. This calls readConfigFile().
ViParametersReader::ViParametersReader(const std::string& filename) {
  // reads
  readConfigFile(filename);
}

// Read and parse a config file.
void ViParametersReader::readConfigFile(const std::string& filename) {

  // reads
  
  cv::FileStorage file(filename, cv::FileStorage::READ);

  OKVIS_ASSERT_TRUE(Exception, file.isOpened(),
                    "Could not open config file: " << filename)
  LOG(INFO) << "Opened configuration file: " << filename;

  // camera calibration
  std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if(!getCameraCalibration(calibrations, file)) {
    LOG(FATAL) << "Did not find any calibration!";
  }

  // Count the number of cameras for mapping and rectification to decide
  // whether stereo rectification for the stereo depth network is actually needed.
  size_t numRectifiedCamera = 0;
  for (const auto& calibration : calibrations) {
    if (calibration.cameraType.depthType.needRectify) {
      numRectifiedCamera ++;
    }
  }

  // Assign parameters for stereo rectification.
  double fov_scale;
  if (file["camera_parameters"]["fov_scale"].isReal()) {
    parseEntry(file["camera_parameters"], "fov_scale", fov_scale);
  }
  else {
    if (numRectifiedCamera != 0) {
      LOG(FATAL) << "Please set fov_scale for the stereo rectification.\n"
        << "If you don't want to run the stereo depth network, "
        << "please set mapping_rectification: false in okvis2.yaml.";
    }
  }

  std::vector<size_t> sidx;
  if (file["camera_parameters"]["deep_stereo_indices"].isSeq()) {
    cv::FileNode T_idx = file["camera_parameters"]["deep_stereo_indices"];
    for(auto iter = T_idx.begin(); iter != T_idx.end(); ++iter) {
      sidx.push_back(int(*iter));
    }
  }
  else {
    if (numRectifiedCamera != 0) {
      LOG(FATAL) << "Please set deep_stereo_indices for the stereo depth network.\n"
        << "If you don't want to run the stereo depth network, "
        << "please set mapping_rectification: false in okvis2.yaml.";
    }
  }
  if (sidx.size() == 2) {
    computeRectifyMap(calibrations[sidx[0]], calibrations[sidx[1]], sidx, fov_scale);
  } else {
    LOG(WARNING) << "No stereo rectification";
  }
  
  size_t camIdx = 0;
  for (size_t i = 0; i < calibrations.size(); ++i) {

    std::shared_ptr<const kinematics::Transformation> T_SC_okvis_ptr(
          new kinematics::Transformation(calibrations[i].T_SC.r(),
                                                calibrations[i].T_SC.q().normalized()));

    if(strcmp(calibrations[i].cameraModel.c_str(), "eucm") == 0){
      std::shared_ptr<okvis::cameras::EucmCamera> cam;
      cam.reset(new okvis::cameras::EucmCamera(calibrations[i].imageDimension[0],
                                               calibrations[i].imageDimension[1],
                                               calibrations[i].focalLength[0],
                                               calibrations[i].focalLength[1],
                                               calibrations[i].principalPoint[0],
                                               calibrations[i].principalPoint[1],
                                               calibrations[i].eucmParameters[0],
                                               calibrations[i].eucmParameters[1]));
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
              T_SC_okvis_ptr,
              std::static_pointer_cast<const okvis::cameras::CameraBase>(cam),
              okvis::cameras::NCameraSystem::NoDistortion, true,
              calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "EUCM camera " << camIdx
                << " with T_SC=\n" << s.str();
    }
    else if (strcmp(calibrations[i].distortionType.c_str(), "equidistant") == 0) {
      std::shared_ptr<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>> cam;
      cam.reset(new okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::EquidistantDistortion(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::Equidistant, true,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Equidistant pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (strcmp(calibrations[i].distortionType.c_str(), "radialtangential") == 0
               || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob") == 0) {
      std::shared_ptr<cameras::PinholeCamera<cameras::RadialTangentialDistortion>> cam;
      cam.reset(new cameras::PinholeCamera<
                  cameras::RadialTangentialDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::RadialTangentialDistortion(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::RadialTangential, true,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (strcmp(calibrations[i].distortionType.c_str(), "radialtangential8") == 0
               || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob8") == 0) {
      std::shared_ptr<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>> cam;
      cam.reset(new cameras::PinholeCamera<cameras::RadialTangentialDistortion8>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::RadialTangentialDistortion8(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3],
                    calibrations[i].distortionCoefficients[4],
                    calibrations[i].distortionCoefficients[5],
                    calibrations[i].distortionCoefficients[6],
                    calibrations[i].distortionCoefficients[7])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::RadialTangential8, true,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential 8 pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else {
      LOG(ERROR) << "unrecognized distortion type " << calibrations[i].distortionType;
    }
    ++camIdx;
  }

  if(file["lidar"].isMap()){
    viParameters_.lidar = okvis::LidarParameters(); 
    if(!getLiDARCalibration(file["lidar"], *viParameters_.lidar)){
      LOG(ERROR) << "Could not parse the LiDAR config file";
    } else {
      std::stringstream s;
      s << (*viParameters_.lidar).T_SL.T();
      LOG(INFO) << "Parsed LiDAR with the following characteristics: \n" 
                << "elevation_resolution_angle: " << std::to_string((*viParameters_.lidar).elevation_resolution_angle) << " \n"
                << "azimuth_resolution_angle: " << std::to_string((*viParameters_.lidar).azimuth_resolution_angle) << "\n"
                << "T_SL:\n " << s.str();
    }
  } else {
    LOG(INFO) << "No LiDAR declared";
  }

  //camera parameters.
  parseEntry(file["camera_parameters"], "timestamp_tolerance",
             viParameters_.camera.timestamp_tolerance);
  cv::FileNode T = file["camera_parameters"]["sync_cameras"];
  OKVIS_ASSERT_TRUE(
    Exception, T.isSeq(),
    "missing real array parameter " << "camera_parameters" << ": " << "sync_cameras")
  for(auto iter = T.begin(); iter != T.end(); ++iter) {
    viParameters_.camera.sync_cameras.insert(int(*iter));
  }
  parseEntry(file["camera_parameters"], "image_delay",
             viParameters_.camera.image_delay);
  cv::FileNode stereo_dnn_enh = file["camera_parameters"]["stereo_dnn_apply_image_enhancement"];
  if (stereo_dnn_enh.isInt() || stereo_dnn_enh.isString()) {
    parseEntry(file["camera_parameters"], "stereo_dnn_apply_image_enhancement",
               viParameters_.camera.stereo_dnn_apply_image_enhancement);
  } else {
    viParameters_.camera.stereo_dnn_apply_image_enhancement = false;
  }
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics",
             viParameters_.camera.online_calibration.do_extrinsics);
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics_final_ba",
             viParameters_.camera.online_calibration.do_extrinsics_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r",
             viParameters_.camera.online_calibration.sigma_r);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha",
             viParameters_.camera.online_calibration.sigma_alpha);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r_final_ba",
             viParameters_.camera.online_calibration.sigma_r_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha_final_ba",
             viParameters_.camera.online_calibration.sigma_alpha_final_ba);
  viParameters_.camera.stereo_indices = sidx;

  //IMU parameters.
  parseEntry(file["imu_parameters"], "use",
             viParameters_.imu.use);
  // Optional: allow disabling IMU preintegration residuals while still using IMU for propagation/initialisation.
  if (file["imu_parameters"]["use_preintegration_residual"].isInt()
      || file["imu_parameters"]["use_preintegration_residual"].isString()) {
    parseEntry(file["imu_parameters"], "use_preintegration_residual",
               viParameters_.imu.use_preintegration_residual);
  } else {
    viParameters_.imu.use_preintegration_residual = true;
  }
  Eigen::Matrix4d T_BS;
  parseEntry(file["imu_parameters"], "T_BS", T_BS);
  viParameters_.imu.T_BS = kinematics::Transformation(T_BS);
  parseEntry(file["imu_parameters"], "a_max",
             viParameters_.imu.a_max);
  parseEntry(file["imu_parameters"], "g_max",
             viParameters_.imu.g_max);
  parseEntry(file["imu_parameters"], "sigma_g_c",
             viParameters_.imu.sigma_g_c);
  parseEntry(file["imu_parameters"], "sigma_bg",
             viParameters_.imu.sigma_bg);
  parseEntry(file["imu_parameters"], "sigma_a_c",
             viParameters_.imu.sigma_a_c);
  parseEntry(file["imu_parameters"], "sigma_ba",
             viParameters_.imu.sigma_ba);
  parseEntry(file["imu_parameters"], "sigma_gw_c",
             viParameters_.imu.sigma_gw_c);
  parseEntry(file["imu_parameters"], "sigma_aw_c",
             viParameters_.imu.sigma_aw_c);
  parseEntry(file["imu_parameters"], "a0",
             viParameters_.imu.a0);
  parseEntry(file["imu_parameters"], "g0",
             viParameters_.imu.g0);
  parseEntry(file["imu_parameters"], "g",
             viParameters_.imu.g);
  parseEntry(file["imu_parameters"], "s_a",
             viParameters_.imu.s_a);

  // Optional: estimate the gyroscope bias at all? Defaults to false, i.e. the raw gyro readings
  // are used -- the same convention as GaRLILEO, which integrates the gyro with a constant bias.
  if (file["imu_parameters"]["estimate_gyro_bias"].isInt()
      || file["imu_parameters"]["estimate_gyro_bias"].isString()) {
    parseEntry(file["imu_parameters"], "estimate_gyro_bias",
               viParameters_.imu.estimate_gyro_bias);
  } else {
    viParameters_.imu.estimate_gyro_bias = false;
  }
  if (!viParameters_.imu.estimate_gyro_bias) {
    // b_g is a random-walk state of the 9-D speed/bias parameter block, so it cannot be held
    // constant per-component without a Ceres manifold (which okvis' marginalisation cannot
    // represent). Instead pin it: the initial prior fixes b_g at g0 and the (near-zero) drift
    // noise density keeps every ImuError from moving it. Both sigmas stay far enough above the
    // pseudo-inverse tolerance in ImuError that the information matrix stays well defined.
    viParameters_.imu.sigma_bg = kPinnedSigmaBg;
    viParameters_.imu.sigma_gw_c = kPinnedSigmaGwC;
    LOG(INFO) << "IMU gyro bias estimation disabled: b_g pinned to g0 = ["
              << viParameters_.imu.g0.transpose() << "] rad/s (sigma_bg=" << kPinnedSigmaBg
              << ", sigma_gw_c=" << kPinnedSigmaGwC << ").";
  }

  // Optional: estimate the accelerometer bias at all? Defaults to true, i.e. okvis' usual
  // behaviour; set to false to match GaRLILEO, which holds b_a at zero.
  if (file["imu_parameters"]["estimate_accel_bias"].isInt()
      || file["imu_parameters"]["estimate_accel_bias"].isString()) {
    parseEntry(file["imu_parameters"], "estimate_accel_bias",
               viParameters_.imu.estimate_accel_bias);
  } else {
    viParameters_.imu.estimate_accel_bias = true;
  }
  if (!viParameters_.imu.estimate_accel_bias) {
    // Same mechanism as for b_g above: pin rather than fix, so the graph structure and the
    // marginalisation are untouched.
    viParameters_.imu.sigma_ba = kPinnedSigmaBa;
    viParameters_.imu.sigma_aw_c = kPinnedSigmaAwC;
    LOG(INFO) << "IMU accelerometer bias estimation disabled: b_a pinned to a0 = ["
              << viParameters_.imu.a0.transpose() << "] m/s^2 (sigma_ba=" << kPinnedSigmaBa
              << ", sigma_aw_c=" << kPinnedSigmaAwC << ").";
  }

  // Parameters for detection etc.
  parseEntry(file["frontend_parameters"], "detection_threshold",
             viParameters_.frontend.detection_threshold);
  parseEntry(file["frontend_parameters"], "absolute_threshold",
             viParameters_.frontend.absolute_threshold);
  parseEntry(file["frontend_parameters"], "matching_threshold",
             viParameters_.frontend.matching_threshold);
  parseEntry(file["frontend_parameters"], "octaves",
             viParameters_.frontend.octaves);
  parseEntry(file["frontend_parameters"], "max_num_keypoints",
             viParameters_.frontend.max_num_keypoints);
  parseEntry(file["frontend_parameters"], "keyframe_overlap",
             viParameters_.frontend.keyframe_overlap);
  parseEntry(file["frontend_parameters"], "use_cnn",
             viParameters_.frontend.use_cnn);
  parseEntry(file["frontend_parameters"], "parallelise_detection",
             viParameters_.frontend.parallelise_detection);
  parseEntry(file["frontend_parameters"], "num_matching_threads",
             viParameters_.frontend.num_matching_threads);

  const auto n_vig_enable = file["frontend_parameters"]["vignette_correction_enable"];
  if (n_vig_enable.isInt() || n_vig_enable.isString()) {
    parseEntry(file["frontend_parameters"], "vignette_correction_enable",
               viParameters_.frontend.vignette_correction_enable);
  }
  const auto n_vig_k1 = file["frontend_parameters"]["vignette_correction_k1"];
  if (n_vig_k1.isReal() || n_vig_k1.isInt()) {
    parseEntry(file["frontend_parameters"], "vignette_correction_k1",
               viParameters_.frontend.vignette_correction_k1);
  }
  const auto n_vig_k2 = file["frontend_parameters"]["vignette_correction_k2"];
  if (n_vig_k2.isReal() || n_vig_k2.isInt()) {
    parseEntry(file["frontend_parameters"], "vignette_correction_k2",
               viParameters_.frontend.vignette_correction_k2);
  }
  const auto n_vig_k3 = file["frontend_parameters"]["vignette_correction_k3"];
  if (n_vig_k3.isReal() || n_vig_k3.isInt()) {
    parseEntry(file["frontend_parameters"], "vignette_correction_k3",
               viParameters_.frontend.vignette_correction_k3);
  }

  const auto n_bo_enable = file["frontend_parameters"]["agcwd_clahe_bo_enable"];
  if (n_bo_enable.isInt() || n_bo_enable.isString()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_enable",
               viParameters_.frontend.agcwd_clahe_bo_enable);
  }
  const auto n_bo_ud = file["frontend_parameters"]["agcwd_clahe_bo_utility_discrepancy_threshold"];
  if (n_bo_ud.isReal() || n_bo_ud.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_utility_discrepancy_threshold",
               viParameters_.frontend.agcwd_clahe_bo_utility_discrepancy_threshold);
  }
  const auto n_bo_init = file["frontend_parameters"]["agcwd_clahe_bo_initial_samples"];
  if (n_bo_init.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_initial_samples",
               viParameters_.frontend.agcwd_clahe_bo_initial_samples);
  }
  const auto n_bo_iter = file["frontend_parameters"]["agcwd_clahe_bo_iterations"];
  if (n_bo_iter.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_iterations",
               viParameters_.frontend.agcwd_clahe_bo_iterations);
  }
  const auto n_bo_grid = file["frontend_parameters"]["agcwd_clahe_bo_candidate_grid"];
  if (n_bo_grid.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_candidate_grid",
               viParameters_.frontend.agcwd_clahe_bo_candidate_grid);
  }
  const auto n_bo_amin = file["frontend_parameters"]["agcwd_clahe_bo_agcwd_min"];
  if (n_bo_amin.isReal() || n_bo_amin.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_agcwd_min",
               viParameters_.frontend.agcwd_clahe_bo_agcwd_min);
  }
  const auto n_bo_amax = file["frontend_parameters"]["agcwd_clahe_bo_agcwd_max"];
  if (n_bo_amax.isReal() || n_bo_amax.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_agcwd_max",
               viParameters_.frontend.agcwd_clahe_bo_agcwd_max);
  }
  const auto n_bo_cmin = file["frontend_parameters"]["agcwd_clahe_bo_clahe_min"];
  if (n_bo_cmin.isReal() || n_bo_cmin.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_clahe_min",
               viParameters_.frontend.agcwd_clahe_bo_clahe_min);
  }
  const auto n_bo_cmax = file["frontend_parameters"]["agcwd_clahe_bo_clahe_max"];
  if (n_bo_cmax.isReal() || n_bo_cmax.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_clahe_max",
               viParameters_.frontend.agcwd_clahe_bo_clahe_max);
  }
  const auto n_bo_iuw = file["frontend_parameters"]["agcwd_clahe_bo_image_utility_weight"];
  if (n_bo_iuw.isReal() || n_bo_iuw.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_clahe_bo_image_utility_weight",
               viParameters_.frontend.agcwd_clahe_bo_image_utility_weight);
  }

  // Optional: AGCWD image preprocessing.
  // Keep optional to avoid breaking existing configs.
  const auto n_agcwd_enable = file["frontend_parameters"]["agcwd_enable"];
  if (n_agcwd_enable.isInt() || n_agcwd_enable.isString()) {
    parseEntry(file["frontend_parameters"], "agcwd_enable",
               viParameters_.frontend.agcwd_enable);
  }
  const auto n_agcwd_w = file["frontend_parameters"]["agcwd_weighting_param"];
  if (n_agcwd_w.isReal() || n_agcwd_w.isInt()) {
    parseEntry(file["frontend_parameters"], "agcwd_weighting_param",
               viParameters_.frontend.agcwd_weighting_param);
  }

  const auto n_clahe_enable = file["frontend_parameters"]["clahe_enable"];
  if (n_clahe_enable.isInt() || n_clahe_enable.isString()) {
    parseEntry(file["frontend_parameters"], "clahe_enable",
               viParameters_.frontend.clahe_enable);
  }
  const auto n_clahe_clip = file["frontend_parameters"]["clahe_clip_limit"];
  if (n_clahe_clip.isReal() || n_clahe_clip.isInt()) {
    parseEntry(file["frontend_parameters"], "clahe_clip_limit",
               viParameters_.frontend.clahe_clip_limit);
  }
  const auto n_clahe_grid = file["frontend_parameters"]["clahe_tile_grid_size"];
  if (n_clahe_grid.isInt()) {
    parseEntry(file["frontend_parameters"], "clahe_tile_grid_size",
               viParameters_.frontend.clahe_tile_grid_size);
  }

  // ----- EWG (Entropy-Weighted Gradient) image utility ---------------------
  const auto n_ewg_Hthres = file["frontend_parameters"]["image_utility_ewg_Hthres"];
  if (n_ewg_Hthres.isReal() || n_ewg_Hthres.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_Hthres",
               viParameters_.frontend.image_utility_ewg_Hthres);
  }
  const auto n_ewg_alpha = file["frontend_parameters"]["image_utility_ewg_alpha"];
  if (n_ewg_alpha.isReal() || n_ewg_alpha.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_alpha",
               viParameters_.frontend.image_utility_ewg_alpha);
  }
  const auto n_ewg_tau = file["frontend_parameters"]["image_utility_ewg_tau"];
  if (n_ewg_tau.isReal() || n_ewg_tau.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_tau",
               viParameters_.frontend.image_utility_ewg_tau);
  }
  const auto n_ewg_patch = file["frontend_parameters"]["image_utility_ewg_local_patch"];
  if (n_ewg_patch.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_local_patch",
               viParameters_.frontend.image_utility_ewg_local_patch);
  }
  const auto n_ewg_bins = file["frontend_parameters"]["image_utility_ewg_num_bins"];
  if (n_ewg_bins.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_num_bins",
               viParameters_.frontend.image_utility_ewg_num_bins);
  }
  const auto n_ewg_lambda = file["frontend_parameters"]["image_utility_ewg_lambda_f"];
  if (n_ewg_lambda.isReal() || n_ewg_lambda.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_lambda_f",
               viParameters_.frontend.image_utility_ewg_lambda_f);
  }
  const auto n_ewg_n0 = file["frontend_parameters"]["image_utility_ewg_N0"];
  if (n_ewg_n0.isReal() || n_ewg_n0.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_N0",
               viParameters_.frontend.image_utility_ewg_N0);
  }
  const auto n_ewg_gr = file["frontend_parameters"]["image_utility_ewg_grid_rows"];
  if (n_ewg_gr.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_grid_rows",
               viParameters_.frontend.image_utility_ewg_grid_rows);
  }
  const auto n_ewg_gc = file["frontend_parameters"]["image_utility_ewg_grid_cols"];
  if (n_ewg_gc.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_grid_cols",
               viParameters_.frontend.image_utility_ewg_grid_cols);
  }
  const auto n_ewg_rs_en = file["frontend_parameters"]["image_utility_ewg_resize_enable"];
  if (n_ewg_rs_en.isInt() || n_ewg_rs_en.isString()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_resize_enable",
               viParameters_.frontend.image_utility_ewg_resize_enable);
  } else {
    (void)tryParseBoolFromYamlText(filename, "image_utility_ewg_resize_enable",
                                   viParameters_.frontend.image_utility_ewg_resize_enable);
  }
  const auto n_ewg_rs_ms = file["frontend_parameters"]["image_utility_ewg_resize_max_side"];
  if (n_ewg_rs_ms.isInt()) {
    parseEntry(file["frontend_parameters"], "image_utility_ewg_resize_max_side",
               viParameters_.frontend.image_utility_ewg_resize_max_side);
  }
  const auto n_ewg_fb_src =
      file["frontend_parameters"]["image_utility_fallback_use_preprocessed"];
  if (n_ewg_fb_src.isInt() || n_ewg_fb_src.isString()) {
    parseEntry(file["frontend_parameters"], "image_utility_fallback_use_preprocessed",
               viParameters_.frontend.image_utility_fallback_use_preprocessed);
  } else {
    (void)tryParseBoolFromYamlText(filename, "image_utility_fallback_use_preprocessed",
                                   viParameters_.frontend.image_utility_fallback_use_preprocessed);
  }

  // Parameters regarding the estimator.
  parseEntry(file["estimator_parameters"], "num_keyframes",
             viParameters_.estimator.num_keyframes);
  parseEntry(file["estimator_parameters"], "num_loop_closure_frames",
             viParameters_.estimator.num_loop_closure_frames);
  parseEntry(file["estimator_parameters"], "num_imu_frames",
             viParameters_.estimator.num_imu_frames);
  parseEntry(file["estimator_parameters"], "do_loop_closures",
             viParameters_.estimator.do_loop_closures);
  parseEntry(file["estimator_parameters"], "do_final_ba",
             viParameters_.estimator.do_final_ba);
  // Reset optional final-BA controls when reusing a reader with another config.
  viParameters_.estimator.final_ba_max_iterations = 100;
  viParameters_.estimator.final_ba_function_tolerance = 1.0e-6;
  const auto finalBaIterations = file["estimator_parameters"]["final_ba_max_iterations"];
  if (!finalBaIterations.empty()) {
    OKVIS_ASSERT_TRUE(Exception, finalBaIterations.isInt()
                      && static_cast<int>(finalBaIterations) > 0,
                      "final_ba_max_iterations must be a positive integer")
    viParameters_.estimator.final_ba_max_iterations = static_cast<int>(finalBaIterations);
  }
  const auto finalBaTolerance = file["estimator_parameters"]["final_ba_function_tolerance"];
  if (!finalBaTolerance.empty()) {
    OKVIS_ASSERT_TRUE(Exception, finalBaTolerance.isReal() || finalBaTolerance.isInt(),
                      "final_ba_function_tolerance must be numeric")
    const double tolerance = static_cast<double>(finalBaTolerance);
    OKVIS_ASSERT_TRUE(Exception, std::isfinite(tolerance) && tolerance >= 0.0 && tolerance < 1.0,
                      "final_ba_function_tolerance must be finite and in [0, 1)")
    viParameters_.estimator.final_ba_function_tolerance = tolerance;
  }
  parseEntry(file["estimator_parameters"], "enforce_realtime",
             viParameters_.estimator.enforce_realtime);
  parseEntry(file["estimator_parameters"], "realtime_min_iterations",
             viParameters_.estimator.realtime_min_iterations);
  parseEntry(file["estimator_parameters"], "realtime_max_iterations",
             viParameters_.estimator.realtime_max_iterations);
  parseEntry(file["estimator_parameters"], "realtime_time_limit",
             viParameters_.estimator.realtime_time_limit);
  parseEntry(file["estimator_parameters"], "realtime_num_threads",
             viParameters_.estimator.realtime_num_threads);
  parseEntry(file["estimator_parameters"], "full_graph_iterations",
             viParameters_.estimator.full_graph_iterations);
  parseEntry(file["estimator_parameters"], "full_graph_num_threads",
             viParameters_.estimator.full_graph_num_threads);
  parseEntry(file["estimator_parameters"], "p_dbow",
             viParameters_.estimator.p_dbow);
  parseEntry(file["estimator_parameters"], "drift_percentage_heuristic",
             viParameters_.estimator.drift_percentage_heuristic);

  // Some options for how and what to output.
  parseEntry(file["output_parameters"], "display_topview",
             viParameters_.output.display_topview);
  parseEntry(file["output_parameters"], "display_matches",
             viParameters_.output.display_matches);
  parseEntry(file["output_parameters"], "display_overhead",
             viParameters_.output.display_overhead);
  parseEntry(file["output_parameters"], "enable_submapping",
             viParameters_.output.enable_submapping);

  // GaRLILEO spline coupling (optional; auto-activates when /garlileo/spline_state is available)
  // A reused reader must not carry the optional camera policy into an older config.
  const GaRLILEOParameters reprojectionDefaults;
  viParameters_.garlileo.visual_fallback_reprojection_utility_adaptive =
      reprojectionDefaults.visual_fallback_reprojection_utility_adaptive;
  viParameters_.garlileo.visual_fallback_reprojection_information_scale_min =
      reprojectionDefaults.visual_fallback_reprojection_information_scale_min;
  viParameters_.garlileo.visual_fallback_reprojection_utility_low =
      reprojectionDefaults.visual_fallback_reprojection_utility_low;
  viParameters_.garlileo.visual_fallback_reprojection_utility_high =
      reprojectionDefaults.visual_fallback_reprojection_utility_high;
  if (file["garlileo_parameters"].isMap()) {
    const auto n_use_rotation = file["garlileo_parameters"]["use_rotation"];
    if (n_use_rotation.isInt() || n_use_rotation.isString()) {
      parseEntry(file["garlileo_parameters"], "use_rotation",
                 viParameters_.garlileo.use_rotation);
    } else {
      (void)tryParseBoolFromYamlText(filename, "use_rotation",
                                     viParameters_.garlileo.use_rotation);
    }
    if (file["garlileo_parameters"]["use_translation"].isInt()
        || file["garlileo_parameters"]["use_translation"].isString()) {
      parseEntry(file["garlileo_parameters"], "use_translation",
                 viParameters_.garlileo.use_translation);
    }
    if (file["garlileo_parameters"]["use_velocity"].isInt()
        || file["garlileo_parameters"]["use_velocity"].isString()) {
      parseEntry(file["garlileo_parameters"], "use_velocity",
                 viParameters_.garlileo.use_velocity);
    }
    if (file["garlileo_parameters"]["rotation_roll_pitch_std_deg"].isReal()
        || file["garlileo_parameters"]["rotation_roll_pitch_std_deg"].isInt()) {
      parseEntry(file["garlileo_parameters"], "rotation_roll_pitch_std_deg",
                 viParameters_.garlileo.rotation_roll_pitch_std_deg);
    }
    if (file["garlileo_parameters"]["rotation_yaw_std_deg"].isReal()
        || file["garlileo_parameters"]["rotation_yaw_std_deg"].isInt()) {
      parseEntry(file["garlileo_parameters"], "rotation_yaw_std_deg",
                 viParameters_.garlileo.rotation_yaw_std_deg);
    }
    if (file["garlileo_parameters"]["translation_std"].isReal()) {
      parseEntry(file["garlileo_parameters"], "translation_std",
                 viParameters_.garlileo.translation_std);
    }
    if (file["garlileo_parameters"]["velocity_std"].isReal()) {
      parseEntry(file["garlileo_parameters"], "velocity_std",
                 viParameters_.garlileo.velocity_std);
    }
    const auto n_std_scale_use_tq =
        file["garlileo_parameters"]["std_scale_use_tracking_quality"];
    if (n_std_scale_use_tq.isInt() || n_std_scale_use_tq.isString()) {
      parseEntry(file["garlileo_parameters"], "std_scale_use_tracking_quality",
                 viParameters_.garlileo.std_scale_use_tracking_quality);
    } else {
      (void)tryParseBoolFromYamlText(filename, "std_scale_use_tracking_quality",
                                     viParameters_.garlileo.std_scale_use_tracking_quality);
    }
    const auto n_std_scale_min_tq = file["garlileo_parameters"]["std_scale_min_tracking_quality"];
    if (n_std_scale_min_tq.isReal() || n_std_scale_min_tq.isInt()) {
      parseEntry(file["garlileo_parameters"], "std_scale_min_tracking_quality",
                 viParameters_.garlileo.std_scale_min_tracking_quality);
    }
    if (file["garlileo_parameters"]["max_time_diff"].isReal()) {
      parseEntry(file["garlileo_parameters"], "max_time_diff",
                 viParameters_.garlileo.max_time_diff);
    }
    if (file["garlileo_parameters"]["history_sec"].isReal()) {
      parseEntry(file["garlileo_parameters"], "history_sec",
                 viParameters_.garlileo.history_sec);
    }
    if (file["garlileo_parameters"]["apply_imu_extrinsic"].isInt()
        || file["garlileo_parameters"]["apply_imu_extrinsic"].isString()) {
      parseEntry(file["garlileo_parameters"], "apply_imu_extrinsic",
                 viParameters_.garlileo.apply_imu_extrinsic);
    }

    // Frontend pose prediction using GaRLILEO (optional)
    // NOTE: OpenCV FileStorage may parse these as MAP if inline comments contain ':'.
    // Fall back to raw-text parsing in that case.
    const auto n_frontend_use_garlileo = file["garlileo_parameters"]["frontend_use_garlileo"];
    if (n_frontend_use_garlileo.isInt() || n_frontend_use_garlileo.isString()) {
      parseEntry(file["garlileo_parameters"], "frontend_use_garlileo",
                 viParameters_.garlileo.frontend_use_garlileo);
    } else {
      (void)tryParseBoolFromYamlText(filename, "frontend_use_garlileo",
                                     viParameters_.garlileo.frontend_use_garlileo);
    }
    const auto n_frontend_use_rotation = file["garlileo_parameters"]["frontend_use_rotation"];
    if (n_frontend_use_rotation.isInt() || n_frontend_use_rotation.isString()) {
      parseEntry(file["garlileo_parameters"], "frontend_use_rotation",
                 viParameters_.garlileo.frontend_use_rotation);
    } else {
      (void)tryParseBoolFromYamlText(filename, "frontend_use_rotation",
                                     viParameters_.garlileo.frontend_use_rotation);
    }
    const auto n_frontend_use_translation = file["garlileo_parameters"]["frontend_use_translation"];
    if (n_frontend_use_translation.isInt() || n_frontend_use_translation.isString()) {
      parseEntry(file["garlileo_parameters"], "frontend_use_translation",
                 viParameters_.garlileo.frontend_use_translation);
    } else {
      (void)tryParseBoolFromYamlText(filename, "frontend_use_translation",
                                     viParameters_.garlileo.frontend_use_translation);
    }

    const auto n_frontend_use_garlileo_gravity = file["garlileo_parameters"]["frontend_use_garlileo_gravity"];
    if (n_frontend_use_garlileo_gravity.isInt() || n_frontend_use_garlileo_gravity.isString()) {
      parseEntry(file["garlileo_parameters"], "frontend_use_garlileo_gravity",
                 viParameters_.garlileo.frontend_use_garlileo_gravity);
    } else {
      (void)tryParseBoolFromYamlText(filename, "frontend_use_garlileo_gravity",
                                     viParameters_.garlileo.frontend_use_garlileo_gravity);
    }
    if (file["garlileo_parameters"]["frontend_max_age_sec"].isReal()) {
      parseEntry(file["garlileo_parameters"], "frontend_max_age_sec",
                 viParameters_.garlileo.frontend_max_age_sec);
    }
    if (file["garlileo_parameters"]["keyframe_stationary_speed_threshold"].isReal()) {
      parseEntry(file["garlileo_parameters"], "keyframe_stationary_speed_threshold",
                 viParameters_.garlileo.keyframe_stationary_speed_threshold);
    }

    // SFA-EWG image-utility based fallback gate. When >0 (and EWG is enabled),
    // (low_features) uses the minimum per-camera utility (worst stereo camera).
    const auto n_img_util_thr = file["garlileo_parameters"]["image_utility_threshold"];
    if (n_img_util_thr.isReal() || n_img_util_thr.isInt()) {
      parseEntry(file["garlileo_parameters"], "image_utility_threshold",
                 viParameters_.garlileo.image_utility_threshold);
    }
    const auto n_img_util_semi = file["garlileo_parameters"]["image_utility_semi_threshold"];
    if (n_img_util_semi.isReal() || n_img_util_semi.isInt()) {
      parseEntry(file["garlileo_parameters"], "image_utility_semi_threshold",
                 viParameters_.garlileo.image_utility_semi_threshold);
    }
    const auto n_img_util_sigk =
        file["garlileo_parameters"]["image_utility_blend_sigmoid_steepness"];
    if (n_img_util_sigk.isReal() || n_img_util_sigk.isInt()) {
      parseEntry(file["garlileo_parameters"], "image_utility_blend_sigmoid_steepness",
                 viParameters_.garlileo.image_utility_blend_sigmoid_steepness);
    }
    const auto n_img_util_hyst =
        file["garlileo_parameters"]["image_utility_blend_hysteresis"];
    if (n_img_util_hyst.isReal() || n_img_util_hyst.isInt()) {
      parseEntry(file["garlileo_parameters"], "image_utility_blend_hysteresis",
                 viParameters_.garlileo.image_utility_blend_hysteresis);
    }

    // Tracking lost threshold (trackingQuality < thr).
    const auto n_tracking_lost_thr = file["garlileo_parameters"]["tracking_lost_quality_threshold"];
    if (n_tracking_lost_thr.isReal() || n_tracking_lost_thr.isInt()) {
      parseEntry(file["garlileo_parameters"], "tracking_lost_quality_threshold",
                 viParameters_.garlileo.tracking_lost_quality_threshold);
    }
    const auto n_pose_fb_enable = file["garlileo_parameters"]["pose_fallback_enable"];
    if (n_pose_fb_enable.isInt() || n_pose_fb_enable.isString()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_enable",
                 viParameters_.garlileo.pose_fallback_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "pose_fallback_enable",
                                     viParameters_.garlileo.pose_fallback_enable);
    }
    const auto n_pose_fb_trans_std = file["garlileo_parameters"]["pose_fallback_translation_std"];
    if (n_pose_fb_trans_std.isReal() || n_pose_fb_trans_std.isInt()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_translation_std",
                 viParameters_.garlileo.pose_fallback_translation_std);
    }
    const auto n_pose_fb_rot_rp_std = file["garlileo_parameters"]["pose_fallback_rotation_roll_pitch_std_deg"];
    if (n_pose_fb_rot_rp_std.isReal() || n_pose_fb_rot_rp_std.isInt()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_rotation_roll_pitch_std_deg",
                 viParameters_.garlileo.pose_fallback_rotation_roll_pitch_std_deg);
    }
    const auto n_pose_fb_rot_yaw_std = file["garlileo_parameters"]["pose_fallback_rotation_yaw_std_deg"];
    if (n_pose_fb_rot_yaw_std.isReal() || n_pose_fb_rot_yaw_std.isInt()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_rotation_yaw_std_deg",
                 viParameters_.garlileo.pose_fallback_rotation_yaw_std_deg);
    }
    const auto n_pose_fb_weight = file["garlileo_parameters"]["pose_fallback_weight"];
    if (n_pose_fb_weight.isReal() || n_pose_fb_weight.isInt()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_weight",
                 viParameters_.garlileo.pose_fallback_weight);
    }
    const auto n_pose_fb_vel_std = file["garlileo_parameters"]["pose_fallback_velocity_std"];
    if (n_pose_fb_vel_std.isReal() || n_pose_fb_vel_std.isInt()) {
      parseEntry(file["garlileo_parameters"], "pose_fallback_velocity_std",
                 viParameters_.garlileo.pose_fallback_velocity_std);
    }

    // Visual fallback (optional): downweight camera reprojection factors when vision is unreliable.
    const auto n_vis_fb_down_en = file["garlileo_parameters"]["visual_fallback_downweight_enable"];
    if (n_vis_fb_down_en.isInt() || n_vis_fb_down_en.isString()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_downweight_enable",
                 viParameters_.garlileo.visual_fallback_downweight_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "visual_fallback_downweight_enable",
                                     viParameters_.garlileo.visual_fallback_downweight_enable);
    }
    const auto n_vis_fb_reproj_scale = file["garlileo_parameters"]["visual_fallback_reprojection_information_scale"];
    if (n_vis_fb_reproj_scale.isReal() || n_vis_fb_reproj_scale.isInt()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_reprojection_information_scale",
                 viParameters_.garlileo.visual_fallback_reprojection_information_scale);
    }
    const auto n_vis_fb_reproj_adaptive = file["garlileo_parameters"]["visual_fallback_reprojection_utility_adaptive"];
    if (n_vis_fb_reproj_adaptive.isInt() || n_vis_fb_reproj_adaptive.isString()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_reprojection_utility_adaptive",
                 viParameters_.garlileo.visual_fallback_reprojection_utility_adaptive);
    } else {
      (void)tryParseBoolFromYamlText(filename, "visual_fallback_reprojection_utility_adaptive",
                                    viParameters_.garlileo.visual_fallback_reprojection_utility_adaptive);
    }
    const auto n_vis_fb_reproj_min = file["garlileo_parameters"]["visual_fallback_reprojection_information_scale_min"];
    if (n_vis_fb_reproj_min.isReal() || n_vis_fb_reproj_min.isInt()) {
      viParameters_.garlileo.visual_fallback_reprojection_information_scale_min =
          static_cast<double>(n_vis_fb_reproj_min);
    }
    const auto n_vis_fb_reproj_low = file["garlileo_parameters"]["visual_fallback_reprojection_utility_low"];
    if (n_vis_fb_reproj_low.isReal() || n_vis_fb_reproj_low.isInt()) {
      viParameters_.garlileo.visual_fallback_reprojection_utility_low =
          static_cast<double>(n_vis_fb_reproj_low);
    }
    const auto n_vis_fb_reproj_high = file["garlileo_parameters"]["visual_fallback_reprojection_utility_high"];
    if (n_vis_fb_reproj_high.isReal() || n_vis_fb_reproj_high.isInt()) {
      viParameters_.garlileo.visual_fallback_reprojection_utility_high =
          static_cast<double>(n_vis_fb_reproj_high);
    }
    if (viParameters_.garlileo.visual_fallback_reprojection_utility_adaptive) {
      const auto& g = viParameters_.garlileo;
      OKVIS_ASSERT_TRUE(Exception,
          std::isfinite(g.visual_fallback_reprojection_information_scale)
          && std::isfinite(g.visual_fallback_reprojection_information_scale_min)
          && g.visual_fallback_reprojection_information_scale_min > 0.0
          && g.visual_fallback_reprojection_information_scale_min
              <= g.visual_fallback_reprojection_information_scale
          && std::isfinite(g.visual_fallback_reprojection_utility_low)
          && std::isfinite(g.visual_fallback_reprojection_utility_high)
          && g.visual_fallback_reprojection_utility_low >= 0.0
          && g.visual_fallback_reprojection_utility_high > g.visual_fallback_reprojection_utility_low,
          "Adaptive reprojection requires finite 0 < minimum_scale <= scale and 0 <= utility_low < utility_high")
    }
    // Visual fallback - additional camera-factor information scales (optional, default 1.0=no scaling).
    const auto n_vis_fb_relpose_scale = file["garlileo_parameters"]["visual_fallback_relpose_information_scale"];
    if (n_vis_fb_relpose_scale.isReal() || n_vis_fb_relpose_scale.isInt()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_relpose_information_scale",
                 viParameters_.garlileo.visual_fallback_relpose_information_scale);
    }
    const auto n_vis_fb_submap_scale = file["garlileo_parameters"]["visual_fallback_submap_information_scale"];
    if (n_vis_fb_submap_scale.isReal() || n_vis_fb_submap_scale.isInt()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_submap_information_scale",
                 viParameters_.garlileo.visual_fallback_submap_information_scale);
    }

    // Sigma inflation cannot represent zero information. Reject an enabled
    // zero/invalid scale instead of silently retaining full submap weight.
    if (viParameters_.garlileo.visual_fallback_downweight_enable) {
      const double submapScale =
          viParameters_.garlileo.visual_fallback_submap_information_scale;
      OKVIS_ASSERT_TRUE(Exception, std::isfinite(submapScale) && submapScale > 0.0,
          "Enabled visual fallback requires finite visual_fallback_submap_information_scale > 0")
    }

    // Landmark motion consistency gating (optional).
    const auto n_lmc_enable = file["garlileo_parameters"]["landmark_motion_consistency_enable"];
    if (n_lmc_enable.isInt() || n_lmc_enable.isString()) {
      parseEntry(file["garlileo_parameters"], "landmark_motion_consistency_enable",
                 viParameters_.garlileo.landmark_motion_consistency_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "landmark_motion_consistency_enable",
                                     viParameters_.garlileo.landmark_motion_consistency_enable);
    }
    const auto n_lmc_max_pixel = file["garlileo_parameters"]["landmark_motion_consistency_max_pixel"];
    if (n_lmc_max_pixel.isReal() || n_lmc_max_pixel.isInt()) {
      parseEntry(file["garlileo_parameters"], "landmark_motion_consistency_max_pixel",
                 viParameters_.garlileo.landmark_motion_consistency_max_pixel);
    }

    // Visual fallback - GaRLILEO between-factor enforcement (optional).
    const auto n_vis_fb_force_between = file["garlileo_parameters"]["visual_fallback_garlileo_force_between"];
    if (n_vis_fb_force_between.isInt() || n_vis_fb_force_between.isString()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_garlileo_force_between",
                 viParameters_.garlileo.visual_fallback_garlileo_force_between);
    } else {
      (void)tryParseBoolFromYamlText(filename, "visual_fallback_garlileo_force_between",
                                     viParameters_.garlileo.visual_fallback_garlileo_force_between);
    }
    const auto n_vis_fb_wait_max = file["garlileo_parameters"]["visual_fallback_garlileo_wait_max_sec"];
    if (n_vis_fb_wait_max.isReal() || n_vis_fb_wait_max.isInt()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_garlileo_wait_max_sec",
                 viParameters_.garlileo.visual_fallback_garlileo_wait_max_sec);
    }
    const auto n_vis_fb_wait_poll = file["garlileo_parameters"]["visual_fallback_garlileo_wait_poll_period_ms"];
    if (n_vis_fb_wait_poll.isInt() || n_vis_fb_wait_poll.isReal()) {
      parseEntry(file["garlileo_parameters"], "visual_fallback_garlileo_wait_poll_period_ms",
                 viParameters_.garlileo.visual_fallback_garlileo_wait_poll_period_ms);
    }

    // Between-factor backfill (optional): update recent GaRLILEO between constraints when late messages arrive.
    const auto n_between_bf_en = file["garlileo_parameters"]["between_backfill_enable"];
    if (n_between_bf_en.isInt() || n_between_bf_en.isString()) {
      parseEntry(file["garlileo_parameters"], "between_backfill_enable",
                 viParameters_.garlileo.between_backfill_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "between_backfill_enable",
                                     viParameters_.garlileo.between_backfill_enable);
    }
    const auto n_between_bf_pairs = file["garlileo_parameters"]["between_backfill_max_pairs"];
    if (n_between_bf_pairs.isInt() || n_between_bf_pairs.isReal()) {
      parseEntry(file["garlileo_parameters"], "between_backfill_max_pairs",
                 viParameters_.garlileo.between_backfill_max_pairs);
    }
    const auto n_between_bf_overwrite =
        file["garlileo_parameters"]["between_backfill_overwrite_on_quality_improvement"];
    if (n_between_bf_overwrite.isInt() || n_between_bf_overwrite.isString()) {
      parseEntry(file["garlileo_parameters"], "between_backfill_overwrite_on_quality_improvement",
                 viParameters_.garlileo.between_backfill_overwrite_on_quality_improvement);
    } else {
      (void)tryParseBoolFromYamlText(
          filename, "between_backfill_overwrite_on_quality_improvement",
          viParameters_.garlileo.between_backfill_overwrite_on_quality_improvement);
    }

    // Radar input for place recognition (loop closure) fusion.
    // Backward compatibility:
    //  - legacy single-radar keys (radar_support_max_time_diff,
    //    radar_POS_RinB/radar_SO3_RtoB, radar_T_BR) map to radar0.
    if (file["garlileo_parameters"]["radar_support_max_time_diff"].isReal()) {
      parseEntry(file["garlileo_parameters"], "radar_support_max_time_diff",
                 viParameters_.garlileo.place_recognition_radar_max_time_diff);
    }

    const auto n_radar0_enable = file["garlileo_parameters"]["place_recognition_radar0_enable"];
    if (n_radar0_enable.isInt() || n_radar0_enable.isString()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar0_enable",
                 viParameters_.garlileo.place_recognition_radar0_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "place_recognition_radar0_enable",
                                     viParameters_.garlileo.place_recognition_radar0_enable);
    }
    const auto n_radar1_enable = file["garlileo_parameters"]["place_recognition_radar1_enable"];
    if (n_radar1_enable.isInt() || n_radar1_enable.isString()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar1_enable",
                 viParameters_.garlileo.place_recognition_radar1_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "place_recognition_radar1_enable",
                                     viParameters_.garlileo.place_recognition_radar1_enable);
    }
    if (file["garlileo_parameters"]["place_recognition_radar_max_time_diff"].isReal()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar_max_time_diff",
                 viParameters_.garlileo.place_recognition_radar_max_time_diff);
    }
    if (file["garlileo_parameters"]["place_recognition_radar_scans_per_radar"].isInt()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar_scans_per_radar",
                 viParameters_.garlileo.place_recognition_radar_scans_per_radar);
    }

    // Place recognition fuses radar whenever at least one radar is enabled.
    viParameters_.garlileo.place_recognition_use_radar =
        viParameters_.garlileo.place_recognition_radar0_enable
        || viParameters_.garlileo.place_recognition_radar1_enable;
    if (file["garlileo_parameters"]["place_recognition_radar_alpha"].isReal()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar_alpha",
                 viParameters_.garlileo.place_recognition_radar_alpha);
    }
    if (file["garlileo_parameters"]["place_recognition_radar_min_intensity"].isReal()) {
      parseEntry(file["garlileo_parameters"], "place_recognition_radar_min_intensity",
                 viParameters_.garlileo.place_recognition_radar_min_intensity);
    }
    const auto n_fallback_lc_persistent =
        file["garlileo_parameters"]["fallback_loop_closure_persistent_enable"];
    if (n_fallback_lc_persistent.isInt() || n_fallback_lc_persistent.isString()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_closure_persistent_enable",
                 viParameters_.garlileo.fallback_loop_closure_persistent_enable);
    } else {
      (void)tryParseBoolFromYamlText(filename, "fallback_loop_closure_persistent_enable",
                                     viParameters_.garlileo.fallback_loop_closure_persistent_enable);
    }
    const auto n_fallback_lc_full_graph =
        file["garlileo_parameters"]["fallback_loop_closure_full_graph_optimisation_enable"];
    if (n_fallback_lc_full_graph.isInt() || n_fallback_lc_full_graph.isString()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_closure_full_graph_optimisation_enable",
                 viParameters_.garlileo.fallback_loop_closure_full_graph_optimisation_enable);
    } else {
      (void)tryParseBoolFromYamlText(
          filename, "fallback_loop_closure_full_graph_optimisation_enable",
          viParameters_.garlileo.fallback_loop_closure_full_graph_optimisation_enable);
    }
    const auto n_fallback_lc_path_constraints =
        file["garlileo_parameters"]["fallback_loop_closure_path_constraints_enable"];
    if (n_fallback_lc_path_constraints.isInt() || n_fallback_lc_path_constraints.isString()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_closure_path_constraints_enable",
                 viParameters_.garlileo.fallback_loop_closure_path_constraints_enable);
    } else {
      (void)tryParseBoolFromYamlText(
          filename, "fallback_loop_closure_path_constraints_enable",
          viParameters_.garlileo.fallback_loop_closure_path_constraints_enable);
    }
    if (file["garlileo_parameters"]["fallback_loop_closure_path_constraints_mode"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_closure_path_constraints_mode",
                 viParameters_.garlileo.fallback_loop_closure_path_constraints_mode);
    }
    if (file["garlileo_parameters"]["fallback_loop_closure_chain_constraint_stride"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_closure_chain_constraint_stride",
                 viParameters_.garlileo.fallback_loop_closure_chain_constraint_stride);
    }
    const auto n_fallback_lc_weaken_existing =
        file["garlileo_parameters"]["fallback_loop_weaken_existing_constraints_enable"];
    if (n_fallback_lc_weaken_existing.isInt() || n_fallback_lc_weaken_existing.isString()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_weaken_existing_constraints_enable",
                 viParameters_.garlileo.fallback_loop_weaken_existing_constraints_enable);
    } else {
      (void)tryParseBoolFromYamlText(
          filename, "fallback_loop_weaken_existing_constraints_enable",
          viParameters_.garlileo.fallback_loop_weaken_existing_constraints_enable);
    }
    if (file["garlileo_parameters"]["fallback_loop_existing_constraint_scale"].isReal()
        || file["garlileo_parameters"]["fallback_loop_existing_constraint_scale"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_existing_constraint_scale",
                 viParameters_.garlileo.fallback_loop_existing_constraint_scale);
    }
    if (file["garlileo_parameters"]["fallback_loop_information_scale"].isReal()
        || file["garlileo_parameters"]["fallback_loop_information_scale"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_information_scale",
                 viParameters_.garlileo.fallback_loop_information_scale);
    }
    if (file["garlileo_parameters"]["fallback_loop_persistent_information_scale"].isReal()
        || file["garlileo_parameters"]["fallback_loop_persistent_information_scale"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_persistent_information_scale",
                 viParameters_.garlileo.fallback_loop_persistent_information_scale);
    }
    if (file["garlileo_parameters"]["fallback_loop_chain_information_scale"].isReal()
        || file["garlileo_parameters"]["fallback_loop_chain_information_scale"].isInt()) {
      parseEntry(file["garlileo_parameters"], "fallback_loop_chain_information_scale",
                 viParameters_.garlileo.fallback_loop_chain_information_scale);
    }
    const auto n_fb_garlileo_after_only =
        file["garlileo_parameters"]["fallback_loop_garlileo_between_resume_after_path_only"];
    if (n_fb_garlileo_after_only.isInt() || n_fb_garlileo_after_only.isString()) {
      parseEntry(file["garlileo_parameters"],
                 "fallback_loop_garlileo_between_resume_after_path_only",
                 viParameters_.garlileo.fallback_loop_garlileo_between_resume_after_path_only);
    } else {
      (void)tryParseBoolFromYamlText(
          filename, "fallback_loop_garlileo_between_resume_after_path_only",
          viParameters_.garlileo.fallback_loop_garlileo_between_resume_after_path_only);
    }
    if (file["garlileo_parameters"]["fallback_loop_closure_descriptor_distinctiveness_avg_threshold"]
            .isReal()
        || file["garlileo_parameters"]["fallback_loop_closure_descriptor_distinctiveness_avg_threshold"]
               .isInt()) {
      parseEntry(file["garlileo_parameters"],
                 "fallback_loop_closure_descriptor_distinctiveness_avg_threshold",
                 viParameters_.garlileo.fallback_loop_closure_descriptor_distinctiveness_avg_threshold);
    }

    // Radar extrinsics (R->B).
    auto parseRadarExtrinsicFromPosQ =
        [&](const std::string& pos_key, const std::string& quat_key,
            okvis::kinematics::Transformation& out_T_BR, const std::string& tag) -> bool {
          if (!(file["garlileo_parameters"][pos_key].isSeq()
                && file["garlileo_parameters"][quat_key].isSeq())) {
            return false;
          }
          Eigen::Vector3d p;
          parseEntry(file["garlileo_parameters"], pos_key, p);
          Eigen::Vector4d q;
          q.setZero();
          {
            const cv::FileNode qnode = file["garlileo_parameters"][quat_key];
            if (qnode.isSeq() && qnode.size() == 4) {
              int idx = 0;
              for (auto it = qnode.begin(); it != qnode.end(); ++it, ++idx) {
                double v = 0.0;
                (*it) >> v;
                q(idx) = v;
              }
            } else {
              LOG(WARNING) << "[RadarPR] " << tag
                           << " quaternion must be [qx,qy,qz,qw]. Using identity.";
              q << 0.0, 0.0, 0.0, 1.0;
            }
          }
          const Eigen::Quaterniond q_RtoB(q(3), q(0), q(1), q(2)); // (w,x,y,z)
          Eigen::Matrix4d T_BR = Eigen::Matrix4d::Identity();
          T_BR.topLeftCorner<3,3>() = q_RtoB.normalized().toRotationMatrix();
          T_BR.topRightCorner<3,1>() = p;
          out_T_BR = kinematics::Transformation(T_BR);
          return true;
        };
    auto parseRadarExtrinsicFromT =
        [&](const std::string& T_key, okvis::kinematics::Transformation& out_T_BR,
            bool set_from_posq) {
          if (!file["garlileo_parameters"][T_key].isSeq()) {
            return;
          }
          Eigen::Matrix4d T_BR = Eigen::Matrix4d::Identity();
          parseEntry(file["garlileo_parameters"], T_key, T_BR);
          if ((T_BR - Eigen::Matrix4d::Identity()).norm() > 1e-12 || !set_from_posq) {
            out_T_BR = kinematics::Transformation(T_BR);
          }
        };

    // Radar0: first parse legacy keys for compatibility, then explicit radar0 keys override.
    bool radar0_set_from_posq =
        parseRadarExtrinsicFromPosQ("radar_POS_RinB", "radar_SO3_RtoB",
                                    viParameters_.garlileo.place_recognition_radar0_T_BR,
                                    "radar0");
    parseRadarExtrinsicFromT("radar_T_BR",
                             viParameters_.garlileo.place_recognition_radar0_T_BR,
                             radar0_set_from_posq);
    const bool radar0_set_from_posq_new =
        parseRadarExtrinsicFromPosQ("place_recognition_radar0_POS_RinB",
                                    "place_recognition_radar0_SO3_RtoB",
                                    viParameters_.garlileo.place_recognition_radar0_T_BR,
                                    "radar0");
    parseRadarExtrinsicFromT("place_recognition_radar0_T_BR",
                             viParameters_.garlileo.place_recognition_radar0_T_BR,
                             radar0_set_from_posq_new);

    // Radar1: explicit keys.
    const bool radar1_set_from_posq =
        parseRadarExtrinsicFromPosQ("place_recognition_radar1_POS_RinB",
                                    "place_recognition_radar1_SO3_RtoB",
                                    viParameters_.garlileo.place_recognition_radar1_T_BR,
                                    "radar1");
    parseRadarExtrinsicFromT("place_recognition_radar1_T_BR",
                             viParameters_.garlileo.place_recognition_radar1_T_BR,
                             radar1_set_from_posq);

    // Log frontend prediction config once to make it obvious what is active at runtime.
    LOG(INFO) << "[GaRLILEO][Frontend] config: frontend_use_garlileo="
              << std::boolalpha << viParameters_.garlileo.frontend_use_garlileo
              << " frontend_use_rotation=" << viParameters_.garlileo.frontend_use_rotation
              << " frontend_use_translation=" << viParameters_.garlileo.frontend_use_translation
              << " frontend_use_garlileo_gravity=" << viParameters_.garlileo.frontend_use_garlileo_gravity
              << " frontend_max_age_sec=" << viParameters_.garlileo.frontend_max_age_sec
              << " use_rotation=" << viParameters_.garlileo.use_rotation
              << " use_translation=" << viParameters_.garlileo.use_translation
              << " use_velocity=" << viParameters_.garlileo.use_velocity
              << " spline_state_topic=/garlileo/spline_state"
              << " keyframe_stationary_speed_threshold="
              << viParameters_.garlileo.keyframe_stationary_speed_threshold
              << " image_utility_threshold="
              << viParameters_.garlileo.image_utility_threshold
              << " image_utility_semi_threshold="
              << viParameters_.garlileo.image_utility_semi_threshold
              << " image_utility_blend_sigmoid_steepness="
              << viParameters_.garlileo.image_utility_blend_sigmoid_steepness
              << " image_utility_blend_hysteresis="
              << viParameters_.garlileo.image_utility_blend_hysteresis
              << " tracking_lost_quality_threshold="
              << viParameters_.garlileo.tracking_lost_quality_threshold
              << " pose_fallback_enable=" << viParameters_.garlileo.pose_fallback_enable
              << " pose_fallback_translation_std=" << viParameters_.garlileo.pose_fallback_translation_std
              << " rotation_roll_pitch_std_deg=" << viParameters_.garlileo.rotation_roll_pitch_std_deg
              << " rotation_yaw_std_deg=" << viParameters_.garlileo.rotation_yaw_std_deg
              << " std_scale_use_tracking_quality="
              << viParameters_.garlileo.std_scale_use_tracking_quality
              << " std_scale_min_tracking_quality="
              << viParameters_.garlileo.std_scale_min_tracking_quality
              << " pose_fallback_rotation_roll_pitch_std_deg="
              << viParameters_.garlileo.pose_fallback_rotation_roll_pitch_std_deg
              << " pose_fallback_rotation_yaw_std_deg="
              << viParameters_.garlileo.pose_fallback_rotation_yaw_std_deg
              << " pose_fallback_weight=" << viParameters_.garlileo.pose_fallback_weight
              << " pose_fallback_velocity_std=" << viParameters_.garlileo.pose_fallback_velocity_std
              << " visual_fallback_downweight_enable=" << viParameters_.garlileo.visual_fallback_downweight_enable
              << " visual_fallback_reprojection_information_scale=" << viParameters_.garlileo.visual_fallback_reprojection_information_scale
              << " visual_fallback_reprojection_utility_adaptive=" << viParameters_.garlileo.visual_fallback_reprojection_utility_adaptive
              << " visual_fallback_reprojection_information_scale_min=" << viParameters_.garlileo.visual_fallback_reprojection_information_scale_min
              << " visual_fallback_reprojection_utility_low=" << viParameters_.garlileo.visual_fallback_reprojection_utility_low
              << " visual_fallback_reprojection_utility_high=" << viParameters_.garlileo.visual_fallback_reprojection_utility_high
              << " between_backfill_enable=" << viParameters_.garlileo.between_backfill_enable
              << " between_backfill_max_pairs=" << viParameters_.garlileo.between_backfill_max_pairs
              << " between_backfill_overwrite_on_quality_improvement="
              << viParameters_.garlileo.between_backfill_overwrite_on_quality_improvement
              << " place_recognition_use_radar=" << viParameters_.garlileo.place_recognition_use_radar
              << " place_recognition_radar0_enable=" << viParameters_.garlileo.place_recognition_radar0_enable
              << " place_recognition_radar1_enable=" << viParameters_.garlileo.place_recognition_radar1_enable
              << " place_recognition_radar_max_time_diff=" << viParameters_.garlileo.place_recognition_radar_max_time_diff
              << " place_recognition_radar_scans_per_radar="
              << viParameters_.garlileo.place_recognition_radar_scans_per_radar
              << " place_recognition_radar_alpha=" << viParameters_.garlileo.place_recognition_radar_alpha
              << " place_recognition_radar_min_intensity=" << viParameters_.garlileo.place_recognition_radar_min_intensity
              << " fallback_loop_closure_persistent_enable="
              << viParameters_.garlileo.fallback_loop_closure_persistent_enable
              << " fallback_loop_closure_full_graph_optimisation_enable="
              << viParameters_.garlileo.fallback_loop_closure_full_graph_optimisation_enable
              << " fallback_loop_closure_path_constraints_enable="
              << viParameters_.garlileo.fallback_loop_closure_path_constraints_enable
              << " fallback_loop_closure_path_constraints_mode="
              << viParameters_.garlileo.fallback_loop_closure_path_constraints_mode
              << " fallback_loop_closure_chain_constraint_stride="
              << viParameters_.garlileo.fallback_loop_closure_chain_constraint_stride
              << " fallback_loop_weaken_existing_constraints_enable="
              << viParameters_.garlileo.fallback_loop_weaken_existing_constraints_enable
              << " fallback_loop_existing_constraint_scale="
              << viParameters_.garlileo.fallback_loop_existing_constraint_scale
              << " fallback_loop_information_scale="
              << viParameters_.garlileo.fallback_loop_information_scale
              << " fallback_loop_persistent_information_scale="
              << viParameters_.garlileo.fallback_loop_persistent_information_scale
              << " fallback_loop_chain_information_scale="
              << viParameters_.garlileo.fallback_loop_chain_information_scale
              << " fallback_loop_garlileo_between_resume_after_path_only="
              << viParameters_.garlileo.fallback_loop_garlileo_between_resume_after_path_only
              << " fallback_loop_closure_descriptor_distinctiveness_avg_threshold="
              << viParameters_.garlileo.fallback_loop_closure_descriptor_distinctiveness_avg_threshold;
  }

  // GPS Parameters
  if(file["gps_parameters"].isMap()){
    viParameters_.gps = okvis::GpsParameters(); 
    if(!getGpsCalibration(file["gps_parameters"], *viParameters_.gps)){
      LOG(ERROR) << "Could not parse the GPS config";
    } else {
      LOG(INFO) << "Parsed GPS with the following characteristics: \n" 
                << "\tdata_type: " << (*viParameters_.gps).type << " \n"
                << "\tr_SA: " << (*viParameters_.gps).r_SA.transpose() << " \n"
                << "\tyaw_error_threshold: " << std::to_string((*viParameters_.gps).yawErrorThreshold) << " \n"
                << "\trobust_gps_init: " << std::boolalpha << (*viParameters_.gps).robustGpsInit;
    }
  } else {
    LOG(INFO) << "No GPS declared";
  }

  // done!
  readConfigFile_ = true;
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, int& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isInt(),
                    "missing integer parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

void ViParametersReader::parseEntry(const cv::FileNode &file,
                                    std::string name, double& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isReal(),
                    "missing real parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

// Parses booleans from a cv::FileNode. OpenCV sadly has no implementation like this.
void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, bool& readValue) {
  OKVIS_ASSERT_TRUE(Exception, file[name].isInt() || file[name].isString(),
                    "missing boolean parameter " << file.name() << ": " << name)
  if (file[name].isInt()) {
    readValue = int(file[name]) != 0;
    return;
  }
  if (file[name].isString()) {
    std::string str = std::string(file[name]);
    // cut out first word. str currently contains everything including comments
    str = str.substr(0,str.find(" "));
    // transform it to all lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    /* from yaml.org/type/bool.html:
     * Booleans are formatted as English words
     * (“true”/“false”, “yes”/“no” or “on”/“off”)
     * for readability and may be abbreviated as
     * a single character “y”/“n” or “Y”/“N”. */
    if (str.compare("false")  == 0
        || str.compare("no")  == 0
        || str.compare("n")   == 0
        || str.compare("off") == 0) {
      readValue = false;
      return;
    }
    if (str.compare("true")   == 0
        || str.compare("yes") == 0
        || str.compare("y")   == 0
        || str.compare("on")  == 0) {
      readValue = true;
      return;
    }
    OKVIS_THROW(Exception, "Boolean with uninterpretable value " << str)
  }
  return;
}

void ViParametersReader::parseEntry(const cv::FileNode &file, std::string name,
                                    Eigen::Matrix4d& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2], T[3],
               T[4], T[5], T[6], T[7],
               T[8], T[9], T[10], T[11],
               T[12], T[13], T[14], T[15];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    Eigen::Vector3d& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    std::string& readValue) {
  cv::FileNode T = file[name];
  OKVIS_ASSERT_TRUE(Exception, T.isString(),
                    "missing string parameter " << file.name() << ": " << name)
  readValue = std::string(T);
}

bool ViParametersReader::getCameraCalibration(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileStorage& configurationFile) {

  bool success = getCalibrationViaConfig(calibrations, configurationFile["cameras"]);
  return success;
}

// Get the camera calibration via the configuration file.
bool ViParametersReader::getCalibrationViaConfig(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileNode cameraNode) const {

  calibrations.clear();
  bool gotCalibration = false;
  // first check if calibration is available in config file
  if (cameraNode.isSeq()
     && cameraNode.size() > 0) {
    size_t camIdx = 0;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      if ((*it).isMap()
          && (*it)["T_SC"].isSeq()
          && (*it)["image_dimension"].isSeq()
          && (*it)["image_dimension"].size() == 2
          && (*it)["distortion_coefficients"].isSeq()
          && (*it)["distortion_coefficients"].size() >= 4
          && (*it)["distortion_type"].isString()
          && (*it)["focal_length"].isSeq()
          && (*it)["focal_length"].size() == 2
          && (*it)["principal_point"].isSeq()
          && (*it)["principal_point"].size() == 2) {
        LOG(INFO) << "Found calibration in configuration file for camera " << camIdx;
        gotCalibration = true;
      } else {
        LOG(WARNING) << "Found incomplete calibration in configuration file for camera " << camIdx
                     << ". Will not use the calibration from the configuration file.";
        return false;
      }

      ++camIdx;
    }
  }
  else
    LOG(INFO) << "Did not find a calibration in the configuration file.";

  if (gotCalibration) {
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {

      CameraCalibration calib;

      if((std::string)((*it)["cam_model"]) == "pinhole"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode distortionCoefficientNode = (*it)["distortion_coefficients"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];

        // extrinsics
        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.distortionCoefficients.resize(int(distortionCoefficientNode.size()));
        for(int i=0; i<int(distortionCoefficientNode.size()); ++i) {
          calib.distortionCoefficients[i] = distortionCoefficientNode[i];
        }
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.distortionType = std::string((*it)["distortion_type"]);
        calib.cameraModel = "pinhole";
      } else if ( (std::string)((*it)["cam_model"]) == "eucm"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];
        cv::FileNode eucmParamNode= (*it)["eucm_parameters"];

        // extrinsics
        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = okvis::kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.eucmParameters << eucmParamNode[0], eucmParamNode[1];
        calib.distortionType = "None";
        calib.cameraModel = "eucm";
      }

      // parse additional stuff
      if((*it)["camera_type"].isString()){
        std::string camera_type = std::string((*it)["camera_type"]);
        if(camera_type.compare(0,4,"gray")==0) {
          calib.cameraType.isColour = false;
        } else {
          calib.cameraType.isColour = true;
        }
        if(camera_type.size()>=6){
          if(camera_type.compare(camera_type.size()-6,6,"+depth")==0) {
            calib.cameraType.depthType.isDepthCamera = true;
          } else {
            calib.cameraType.depthType.isDepthCamera = false;
          }
        }
      }
      if((*it)["slam_use"].isString()){
        std::string slam_use = std::string((*it)["slam_use"]);
        if(slam_use.compare(0,5,"okvis")==0) {
          calib.cameraType.isUsed = true;
        } else {
          calib.cameraType.isUsed = false;
        }
        if(slam_use.size()>=6){
          if(slam_use.compare(slam_use.size()-6,6,"-depth")==0) {
            calib.cameraType.depthType.createDepth = true;
          } else {
            calib.cameraType.depthType.createDepth = false;
          }
        }
        if(slam_use.size()>=8){
          if(slam_use.compare(slam_use.size()-8,8,"-virtual")==0) {
            calib.cameraType.depthType.createVirtual = true;
          } else {
            calib.cameraType.depthType.createVirtual = false;
          }
        }
      }
      if((*it)["sigma_pixels"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_pixels"]);
      }
      if((*it)["sigma_depth"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_depth"]);
      }

      if((*it)["mapping"].isString() && (std::string)((*it)["mapping"]) == "true"){
        calib.cameraType.isUsedMapping = true;
      }

      if((*it)["mapping_rectification"].isString() && (std::string)((*it)["mapping_rectification"]) == "true"){
        calib.cameraType.depthType.needRectify = true;
      }

      calibrations.push_back(calib);
    }
  }
  return gotCalibration;
}

bool ViParametersReader::getLiDARCalibration(const cv::FileNode& calibrationNode, okvis::LidarParameters& lidarParameters){
  
  if(!calibrationNode["T_SL"].isSeq() || !calibrationNode["elevation_resolution_angle"].isReal() || !calibrationNode["azimuth_resolution_angle"].isReal()){
    return false;
  }

  cv::FileNode T_SL_node = calibrationNode["T_SL"];
  Eigen::Matrix4d T_SL;
  T_SL << T_SL_node[0], T_SL_node[1], T_SL_node[2], T_SL_node[3],
          T_SL_node[4], T_SL_node[5], T_SL_node[6], T_SL_node[7],
          T_SL_node[8], T_SL_node[9], T_SL_node[10], T_SL_node[11],
          T_SL_node[12], T_SL_node[13], T_SL_node[14], T_SL_node[15];
  lidarParameters.T_SL = okvis::kinematics::Transformation(T_SL);
  lidarParameters.elevation_resolution_angle = calibrationNode["elevation_resolution_angle"];
  lidarParameters.azimuth_resolution_angle = calibrationNode["azimuth_resolution_angle"];

  return true;
}

bool ViParametersReader::getGpsCalibration(const cv::FileNode& calibrationNode, okvis::GpsParameters& gpsParameters){

  parseEntry(calibrationNode, "data_type",
             gpsParameters.type);
  parseEntry(calibrationNode, "r_SA",
             gpsParameters.r_SA);
  parseEntry(calibrationNode, "yaw_error_threshold",
             gpsParameters.yawErrorThreshold);
  parseEntry(calibrationNode, "robust_gps_init",
             gpsParameters.robustGpsInit);

  return true;
}

bool ViParametersReader::computeRectifyMap(CameraCalibration& leftCalibration,
                                           CameraCalibration& rightCalibration,
                                           const std::vector<size_t>& stereo_indices,
                                           const double& fov_scale) {
  // Precompute rectification map of stereo network for mapping
  if (leftCalibration.cameraType.depthType.needRectify && rightCalibration.cameraType.depthType.needRectify &&
      leftCalibration.cameraModel == "pinhole" && rightCalibration.cameraModel == "pinhole") {

    LOG(INFO) << "Found a stereo camera for rectification";

    if (leftCalibration.distortionType != "radialtangential" &&
        leftCalibration.distortionType != "equidistant") {
      LOG(ERROR) << "Radial-tangential or equidistant distortion models are supported in stereo rectification.";
      return false;
    }

    cv::Size rawSize = cv::Size(leftCalibration.imageDimension[0],
      leftCalibration.imageDimension[1]); // [width, height]
    int rectWidth = 512;
    const int rectHeight = 384;
    double aspectRatio = static_cast<double>(rawSize.width)/static_cast<double>(rawSize.height);
    // The native resolution of the depth networks (stereo & MVS) is 512x384. 
    // Deviate from this native resolution only for wide images.
    if (aspectRatio > 1.6) {
      rectWidth = static_cast<int>(aspectRatio*rectHeight);
      if (rectWidth % 8 != 0) {
        // Make sure that the image width divisible by 8.
        rectWidth += 8 - (rectWidth % 8);
      }
    }
    cv::Size rectSize(rectWidth, rectHeight);
    LOG(INFO) << "Original resolution: " << rawSize.width << "x" << rawSize.height
              <<",  Rectification resolution: " << rectSize.width << "x" << rectSize.height;
    double array_dcl[4] = {leftCalibration.distortionCoefficients[0],
                            leftCalibration.distortionCoefficients[1],
                            leftCalibration.distortionCoefficients[2],
                            leftCalibration.distortionCoefficients[3]};
    double array_dcr[4] = {rightCalibration.distortionCoefficients[0],
                            rightCalibration.distortionCoefficients[1],
                            rightCalibration.distortionCoefficients[2],
                            rightCalibration.distortionCoefficients[3]};

    double array_Kl[9] = {leftCalibration.focalLength[0], 0.0, leftCalibration.principalPoint[0],
                          0.0, leftCalibration.focalLength[1], leftCalibration.principalPoint[1],
                          0.0, 0.0, 1.0};
    double array_Kr[9] = {rightCalibration.focalLength[0], 0.0, rightCalibration.principalPoint[0],
                          0.0, rightCalibration.focalLength[1], rightCalibration.principalPoint[1],
                          0.0, 0.0, 1.0};

    Eigen::Matrix4d T_rl = rightCalibration.T_SC.T().inverse() * leftCalibration.T_SC.T();
    double array_Rrl[9] = {T_rl(0,0), T_rl(0,1), T_rl(0,2), 
                            T_rl(1,0), T_rl(1,1), T_rl(1,2),
                            T_rl(2,0), T_rl(2,1), T_rl(2,2)};
    double array_trl[3] = {T_rl(0,3), T_rl(1,3), T_rl(2,3)};

    cv::Mat Kl = cv::Mat(3, 3, CV_64F, array_Kl);
    cv::Mat Kr = cv::Mat(3, 3, CV_64F, array_Kr);
    cv::Mat dcl = cv::Mat(4, 1, CV_64F, array_dcl);
    cv::Mat dcr = cv::Mat(4, 1, CV_64F, array_dcr);
    cv::Mat R = cv::Mat(3, 3, CV_64F, array_Rrl);
    cv::Mat T = cv::Mat(3, 1, CV_64F, array_trl);

    // P1 is the rectified camera parameters, R1 is R_{rect}{unrect}
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat rectMapLeft1, rectMapLeft2, rectMapRight1, rectMapRight2;
    if (leftCalibration.distortionType == "radialtangential") {
      // alpha = 0 for only valid pixels after undistortion
      cv::stereoRectify(Kl, dcl, Kr, dcr, rawSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 
                        0, rectSize);
      cv::initUndistortRectifyMap(Kl, dcl, R1, P1, rectSize, CV_16SC2, rectMapLeft1, rectMapLeft2);
      cv::initUndistortRectifyMap(Kr, dcr, R2, P2, rectSize, CV_16SC2, rectMapRight1, rectMapRight2);
    }
    else if (leftCalibration.distortionType == "equidistant") {
      // balance = 0.0 and fov_scale from config for only valid pixels after undistortion
      cv::fisheye::stereoRectify(Kl, dcl, Kr, dcr, rawSize, R, T, R1, R2, P1, P2, Q,
                                  cv::CALIB_ZERO_DISPARITY, rectSize, 0.0, fov_scale);
      cv::fisheye::initUndistortRectifyMap(Kl, dcl, R1, P1, rectSize, CV_16SC2, rectMapLeft1, rectMapLeft2);
      cv::fisheye::initUndistortRectifyMap(Kr, dcr, R2, P2, rectSize, CV_16SC2, rectMapRight1, rectMapRight2);
    }

    Eigen::Matrix4d T1, T2;
    T1 << R1.at<double>(0,0), R1.at<double>(0,1), R1.at<double>(0,2), 0.0,
          R1.at<double>(1,0), R1.at<double>(1,1), R1.at<double>(1,2), 0.0,
          R1.at<double>(2,0), R1.at<double>(2,1), R1.at<double>(2,2), 0.0,
                        0.0,                 0.0,                0.0, 1.0;
    T2 << R2.at<double>(0,0), R2.at<double>(0,1), R2.at<double>(0,2), 0.0,
          R2.at<double>(1,0), R2.at<double>(1,1), R2.at<double>(1,2), 0.0,
          R2.at<double>(2,0), R2.at<double>(2,1), R2.at<double>(2,2), 0.0,
                        0.0,                 0.0,                0.0, 1.0;
    kinematics::Transformation T1_rect_unrect(T1);
    kinematics::Transformation T2_rect_unrect(T2);

    // Rectified left camera
    kinematics::Transformation T_SCl = leftCalibration.T_SC * T1_rect_unrect.inverse();
    std::shared_ptr<const kinematics::Transformation> T_SCl_okvis_ptr(
      new kinematics::Transformation(T_SCl.r(), T_SCl.q().normalized()));
    std::shared_ptr<cameras::PinholeCamera<cameras::NoDistortion>> rectify_caml;
    rectify_caml.reset(new cameras::PinholeCamera<cameras::NoDistortion>(
                        rectSize.width, rectSize.height,
                        P1.at<double>(0,0), P1.at<double>(1,1),
                        P1.at<double>(0,2), P1.at<double>(1,2),
                        cameras::NoDistortion()));
    rectify_caml->setRectifyMap(rectMapLeft1, rectMapLeft2);
    viParameters_.nCameraSystem.addRectifyCamera(stereo_indices[0], T_SCl_okvis_ptr, rectify_caml);

    // Rectified right camera
    kinematics::Transformation T_SCr = rightCalibration.T_SC * T2_rect_unrect.inverse();
        std::shared_ptr<const kinematics::Transformation> T_SCr_okvis_ptr(
      new kinematics::Transformation(T_SCr.r(), T_SCr.q().normalized()));
    std::shared_ptr<cameras::PinholeCamera<cameras::NoDistortion>> rectify_camr;
    rectify_camr.reset(new cameras::PinholeCamera<cameras::NoDistortion>(
                        rectSize.width, rectSize.height,
                        P2.at<double>(0,0), P2.at<double>(1,1),
                        P2.at<double>(0,2), P2.at<double>(1,2),
                        cameras::NoDistortion()));
    rectify_camr->setRectifyMap(rectMapRight1, rectMapRight2);
    viParameters_.nCameraSystem.addRectifyCamera(stereo_indices[1], T_SCr_okvis_ptr, rectify_camr);
    return true;
  } else {
    return false;
  }
}

}  // namespace okvis
