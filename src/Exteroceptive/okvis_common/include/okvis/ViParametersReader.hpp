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
 * @file ViParametersReader.hpp
 * @brief Header file for the ViParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIPARAMETERSREADER_HPP_
#define INCLUDE_OKVIS_VIPARAMETERSREADER_HPP_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop
#include <opencv2/calib3d.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class reads and parses config file.
 */
class ViParametersReader{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The default constructor.
  ViParametersReader();

  /**
   * @brief The constructor. This calls readConfigFile().
   * @param filename Configuration filename.
   */
  ViParametersReader(const std::string& filename);

  /// @brief Trivial destructor.
  ~ViParametersReader() = default;

  /**
   * @brief Read and parse a config file.
   *        To get the result call getParameters().
   * @param filename Configuration filename.
   */
  void readConfigFile(const std::string& filename);

  /**
   * @brief Get parameters.
   * @param[out] parameters A copy of the parameters.
   * @return True if parameters have been read from a configuration file. If it
   *         returns false then the variable \e parameters has not been changed.
   */
  bool getParameters(okvis::ViParameters& parameters) const{
    if(readConfigFile_)
      parameters = viParameters_;
    return readConfigFile_;
  }

 protected:

  /// @brief Struct that contains all the camera calibration information.
  struct CameraCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
    Eigen::Vector2i imageDimension;           ///< Image dimension. [pixels]
    Eigen::VectorXd distortionCoefficients;   ///< Distortion Coefficients.
    Eigen::Vector2d focalLength;              ///< Focal length.
    Eigen::Vector2d principalPoint;           ///< Principal point.
    std::string cameraModel;                  ///< camera model. ('pinhole' 'eucm')
    Eigen::Vector2d eucmParameters;           ///< alpha, beta eucm parameters

    /// \brief Distortion type. ('radialtangential' 'radialtangential8' 'equdistant')
    std::string distortionType;

    cameras::NCameraSystem::CameraType cameraType; ///< Some additional info about the camera.
  };

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, int& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode& file, std::string name, double& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, bool& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode& file, std::string name, Eigen::Matrix4d& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, Eigen::Vector3d& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, std::string& readValue);

  bool readConfigFile_; ///< If readConfigFile() has been called at least once this is true.
  okvis::ViParameters viParameters_;   ///< The parameters.

  /**
   * @brief Get the camera calibration. This looks for the calibration in the
   *        configuration file first. If this fails it will directly get the calibration
   *        from the sensor, if useDriver is set to true.
   * @remark Overload this function if you want to get the calibrations via ROS for example.
   * @param calibrations The calibrations.
   * @param configurationFile The config file.
   * @return True if reading of the calibration was successful.
   */
  bool getCameraCalibration(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileStorage& configurationFile);

  /**
   * @brief Compute rectified parameters and precomputed map for stereo camera
   * @param leftCalibration Rectified parameters will be updated in the left camera
   * @param rightCalibration Rectified parameters will be updated in the right camera
   * @param stereo_indices Left and right camera index defined in yaml file
   * @param fov_scale Divisor for new focal length, only used when fisheye stereo rectification
   * @return True if the stereo rectification is done.
   */
  bool computeRectifyMap(
      CameraCalibration& leftCalibration, CameraCalibration& rightCalibration,
      const std::vector<size_t>& stereo_indices, const double& fov_scale);

  /**
   * @brief Get the camera calibration via the configuration file.
   * @param[out] calibrations Read calibration.
   * @param[in] cameraNode File node pointing to the cameras sequence.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getCalibrationViaConfig(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileNode cameraNode) const;

  /**
   * @brief Get the LiDAR calibration via the configuration file
   * @param[in] cameraNode File node pointing to the LiDAR.
   * @param[out] lidarParameters Read lidar calibration.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getLiDARCalibration(const cv::FileNode& calibrationNode, okvis::LidarParameters& lidarParameters);

  /**
   * @brief Get the GPS calibration via the configuration file
   * @param[in] cameraNode File node pointing to the GPS.
   * @param[out] gpsParameters Read GPS calibration.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getGpsCalibration(const cv::FileNode& calibrationNode, okvis::GpsParameters& gpsParameters);

};

}

#endif /* INCLUDE_OKVIS_VIOPARAMETERSREADER_HPP_ */
