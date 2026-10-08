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
 * @file LeicaDatasetReader.hpp
 * @brief Header file for the LeicaDatasetReader class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_LEICADATASETREADER_HPP
#define INCLUDE_OKVIS_LEICADATASETREADER_HPP

#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>

#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>

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
    class LeicaDatasetReader : public DatasetReaderBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

        /// @brief Disallow default construction.
        LeicaDatasetReader() = delete;

        /// @brief Construct pointing to dataset.
        /// @param path The absolute or relative path to the dataset.
        /// @param deltaT Duration [s] to skip in the beginning.
        /// @param numCameras The number of cameras to consider.
        LeicaDatasetReader(const std::string& path, const Duration & deltaT = Duration(0.0),
                           bool useBottom = true, bool useFront = true, bool useLeft = true, bool useRight = true, bool useTop = false,
                           const std::string cam_type = "eucm", const std::optional<GpsParameters>& gpsParameters = std::nullopt);

        /// @brief Destructor: stops streaming.
        virtual ~LeicaDatasetReader();

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

        /// @brief Set the LiDAR callback
        /// @param lidarCallback The LiDAR callback to register.
        virtual void setLidarCallback(const LidarCallback & lidarCallback) final {
          lidarCallback_ = lidarCallback;
        }

    protected:
        LidarCallback lidarCallback_; ///< The registered lidar callback.

    private:

        /// @brief Main processing loop.
        void processing();
        std::thread processingThread_; ///< Thread running processing loop.

        std::string path_; ///< Dataset path.

        std::atomic_bool streaming_; ///< Are we streaming?
        std::atomic_int counter_; ///< Number of images read yet.
        size_t numImages_ = 0; ///< Number of images to read.

        std::ifstream imuFile_; ///< Imu csv file.
        std::ifstream lidarFile_; /// < LiDAR csv file, not yet used

        Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].

        /// @brief The times and names of all images by camera.
        std::vector < std::vector < std::pair<uint64_t , std::string> > > allImageNames_;

        int numCameras_ = -1;
        bool useBottom_ = false;
        bool useFront_ = false;
        bool useLeft_ = false;
        bool useRight_ = false;
        bool useTop_ = false;
        std::string camType_;

        // gps stuff...
        std::ifstream gpsFile_; ///< Gps csv file.
        std::atomic_bool gpsFlag_; ///< Flag specifying if gps data is available / used.
        std::string gpsDataType_; ///< GPS data type: "cartesian" | "geodetic" | "geodetic-leica"
        okvis::Time t_gps_; ///< Timestamp of the last gps signal received

    };

}

#endif // INCLUDE_OKVIS_LEICADATASETREADER_HPP
