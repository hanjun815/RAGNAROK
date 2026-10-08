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

#ifndef STEREO_DEPTH_PROCESSOR_HPP
#define STEREO_DEPTH_PROCESSOR_HPP

#include <string>

#include <okvis/Parameters.hpp>
#include <okvis/mapTypedefs.hpp>
#include <okvis/SubmappingInterface.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/DeepLearningProcessor.hpp>
#include <okvis/Stereo2DepthProcessor.hpp>
#include <okvis/DepthFusionProcessor.hpp>
#include <okvis/mapTypedefs.hpp>

namespace okvis {

class Processor: public ::okvis::ViInterface {

public:

  /**
   * \brief Constructor.
   * \param parameters Parameters and settings.
   * @param dBowDir The directory to the DBoW vocabulary.
   */
  Processor(okvis::ViParameters& parameters, 
            okvis::DeepLearningProcessor* const dlProcessor,
            std::string dBowDir,
            const okvis::SupereightMapType::Config &mapConfig,
            const SupereightMapType::DataType::Config &dataConfig,
            const se::SubMapConfig &submapConfig);

  ~Processor() = default;

  /// @brief Display some visualisation.
  void display(std::map<std::string, cv::Mat> & images);

  /// @brief Print information about the current processing.
  void collectInfo();

  /// @brief Finish processing.
  void finish();

  /// \brief Indicats whether the add functions block. This only supports blocking.
  virtual void setBlocking(bool blocking) final;

  void setT_BS(const okvis::kinematics::Transformation& T_BS);

  /// \brief Runs main processing iteration, call in your main loop.
  bool processFrame();

  /// \name Add measurements to the algorithm.
  /**
   * \brief              Add a set of new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \return             Returns true normally. False, if the previous one has not been processed yet.
   */
  virtual bool addImages(const okvis::Time& /*stamp*/,
                         const std::map<size_t, cv::Mat> & /*images*/,
                         const std::map<size_t, cv::Mat> & /*depthImages*/
                         = std::map<size_t, cv::Mat>()) final {return false;}

  /// \name Add measurements to the algorithm.
  /**
   * \brief              Add a set of new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \return             Returns true normally. False, if the previous one has not been processed yet.
   */
  bool addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
                 const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages
                         = std::map<size_t, std::pair<okvis::Time, cv::Mat>>());

  /**
   * \brief          Add an IMU measurement.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return Returns true normally. False if the previous one has not been processed yet.
   */
  virtual bool addImuMeasurement(const okvis::Time & stamp,
                                 const Eigen::Vector3d & alpha,
                                 const Eigen::Vector3d & omega) final;

  /**
   * \brief          Add a LiDAR measurement.
   * \param stamp    The measurement timestamp.
   * \param rayMeasurement The ray measurement (cartesian coordinates).
   * \return Returns true normally. False if the previous one has not been processed yet.
   */
  virtual bool addLidarMeasurement(const okvis::Time & stamp,
                                   const Eigen::Vector3d & rayMeasurement) final;

  /**
   * @brief Adds a depth image image to the measurement Queue
   * @param[in] stamp The timestamp
   * @param[in] depthImage The depth image
   * 
   * @return True if successful
  */
  virtual bool addDepthMeasurement(const okvis::Time &stamp,
                                   const cv::Mat &depthImage,
                                   const std::optional<cv::Mat> &sigmaImage = std::nullopt) final;

  /// \brief          Add external GaRLILEO pseudo-measurement (pose/velocity) and forward to ThreadedSlam.
  virtual void addGaRLILEOPseudoMeasurement(const okvis::Time& stamp,
                                            const Eigen::Quaterniond& R_WS,
                                            const Eigen::Vector3d& v_body,
                                            const Eigen::Vector3d& p_W) final;

  /// \brief          Add external GaRLILEO gravity measurement and forward to ThreadedSlam.
  virtual void addGaRLILEOGravityMeasurement(const okvis::Time& stamp,
                                             const Eigen::Vector3d& g_W) final;

  /// \brief          Add radar targets measurement and forward to ThreadedSlam.
  virtual void addRadarTargetsMeasurement(const okvis::Time& stamp,
                                          const okvis::RadarTargetReadings& targets,
                                          int radar_index = 0) final;

  /// @brief Set function that handles submaps visualization (blocks version).
  void setSubmapCallback(const okvis::submapCallback &callback);

  /// @brief Set function that handles field slice visualization (blocks version).
  void setFieldSliceCallback(const okvis::fieldCallback &callback);

  /// @brief Set function that handles point cloud visulaization
  void setAlignmentPublishCallback(const okvis::alignmentPublishCallback &callback);

  /// @brief Set optimized graph callback.
  void setOptimizedGraphCallback(const okvis::ViInterface::OptimisedGraphCallback &callback);

  /// @brief Submapping interface.
  okvis::SubmappingInterface se_interface_;

  /// @brief OKVIS2 threads
  okvis::ThreadedSlam slam_;

private:
  // TODO: internal graph callback with se update and external callback
  void internalOptimizedGraphCallback(
    const okvis::State &state, 
    const okvis::TrackingState &trackingState,
    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> alignedMapPtr,
    std::shared_ptr<const okvis::MapPointVector> mapPointVectorPtr);

  DeepLearningProcessor* const deepLearningProcessor_;

  okvis::ViInterface::OptimisedGraphCallback optimizedGraphCallback_;
};

}

#endif //STEREO_DEPTH_PROCESSOR_HPP
