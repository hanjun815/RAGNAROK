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
 * @file XDatasetReader.hpp
 * @brief Header file for the XDatasetReader class. 99% taken of DatasetReader.hpp
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_XDATASETREADER_HPP_
#define INCLUDE_OKVIS_XDATASETREADER_HPP_

#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ViSensorBase.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief Reader class acting like a VI sensor.
/// @warning Make sure to use this in combination with synchronous
/// processing, as there is no throttling of the reading process.
    class XDatasetReader : public DatasetReaderBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

        /// @brief Disallow default construction.
        XDatasetReader() = delete;

        /// @brief Construct pointing to dataset.
        /// @param path The absolute or relative path to the dataset.
        /// @param deltaT Duration [s] to skip in the beginning.
        /// @param parameters ViParameters in the OKVIS2
        /// @param useLidar flag if Lidar is used
        XDatasetReader(const std::string& path, const Duration & deltaT = Duration(0.0),
                           ViParameters parameters = ViParameters(), bool use_lidar = false,
                           bool use_stereo_network = false, bool use_depth = false, bool use_rgb = false);

        /// @brief Destructor: stops streaming.
        virtual ~XDatasetReader();

        /// @brief (Re-)setting the dataset path.
        /// @param path The absolute or relative path to the dataset.
        /// @return True, if the dateset folder structure could be created.
        virtual bool setDatasetPath(const std::string & path) final;

        /// @brief Setting skip duration in the beginning.
        /// deltaT Duration [s] to skip in the beginning.
        virtual bool setStartingDelay(const okvis::Duration & deltaT) final;

        /// @brief Starts reading the dataset.
        /// @return True, if successful
        virtual bool startStreaming() final;

        /// @brief Stops reading the dataset.
        /// @return True, if successful
        virtual bool stopStreaming() final;

        /// @brief Check if currently reading the dataset.
        /// @return True, if reading.
        virtual bool isStreaming() final;

        /// @brief Get the completion fraction read already.
        /// @return Fraction read already.
        virtual double completion() const final;

        /// \brief Add Callback for receiving LiDAR measurements. // ToDo: maybe better to generalize ViSensorBase
        typedef std::function<
                bool(const okvis::Time &,
                     const Eigen::Vector3d & )> LidarCallback;

        /// \brief Add Callback for images in network.
        typedef std::function<
                bool(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> &,
                    const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & )> networkCallback;

        /// @brief Set the LiDAR callback
        /// @param lidarCallback The LiDAR callback to register.
        virtual void setLidarCallback(const LidarCallback & lidarCallback) final {
          lidarCallback_ = lidarCallback;
        }

        /// @brief Set the stereo callback for stereo network
        void setImagesNetworkCallback(const networkCallback& imagesNetworkCallback) {
            imagesNetworkCallback_ = imagesNetworkCallback;
        }

    private:

        /// @brief Main processing loop.
        void processing();
        std::thread processingThread_; ///< Thread running processing loop.

        std::string path_; ///< Dataset path.

        std::atomic_bool streaming_; ///< Are we streaming?
        std::atomic_int counter_; ///< Number of images read yet.
        size_t numImages_ = 0; ///< Number of images to read.

        Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].

        std::ifstream imuFile_; ///< Imu csv file.
        std::ifstream gpsFile_; ///< Gps csv file.
        std::ifstream lidarFile_; /// < LiDAR csv file, not yet used
        std::atomic_bool gpsFlag_; ///< Flag specifying if gps data is available / used.
        std::string gpsDataType_; ///< GPS data type: "cartesian" | "geodetic" | "geodetic-leica"
        std::atomic_bool lidarFlag_; ///< Flag if lidar is used
        std::atomic_bool networkFlag_; ///< Flag if depth (from stereo network) is used
        std::atomic_bool depthFlag_; ///< Flag if depth (from RGBD) is used
        std::atomic_bool rgbFlag_; ///< Flag specifying if RGB data should be used
        std::atomic_bool imuFlag_; ///< Flag specifying if imu data should be used
        LidarCallback lidarCallback_; ///< The registered lidar callback.
        networkCallback imagesNetworkCallback_; ///< Stereo image callback for netowrk

        /// @brief The times and names of all images by camera.
        std::vector < std::vector < std::pair<std::string, std::string> > > allImageNames_;
        std::vector < std::pair<std::string, std::string> > allDepthNames_;

        int numCameras_ = -1;
        size_t depthCameraId_ = 0;
        okvis::Time t_gps_; ///< Timestamp of the last gps signal received

    };

} // namespace okvis

#endif // INCLUDE_OKVIS_XDATASETREADER_HPP_
