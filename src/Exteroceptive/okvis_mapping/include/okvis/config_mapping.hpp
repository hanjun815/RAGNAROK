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

#ifndef INCLUDE_OKVIS_LIDARMAPPING_CONFIG_HPP
#define INCLUDE_OKVIS_LIDARMAPPING_CONFIG_HPP



#include <se/map/map.hpp>
#include <se/sensor/sensor.hpp>
#include <se/sensor/lidar.hpp>
#include <se/tracker/tracker.hpp>



namespace se {

struct SubMapConfig{

    /** directory to save mesh files to
     */
    std::string resultsDirectory;

    /** Are debug outputs printed / exported?
    */
    bool write_mesh_output;

    /**
     * Downsampling rate of the depth sensor measurements, ToDo: implement for LiDAR
    */
    size_t sensorMeasurementDownsampling;

    /**
     * Downsampling factor of the depth image resolution
    */
    size_t depthImageResDownsampling;

    /**
     * How many keyframes conform a submap
    */
    size_t submapKfThreshold = 5;

    /**
     * Overlap ratio threshold for a new submap
    */
    float submapOverlapRatio = 0.4;

    /**
     * Minimum number of frames being integrated per submap
     */
    size_t submapMinFrames = 1;

    /**
     * Flag whether you're using map-to-map factors
     */
    bool useMap2MapFactors = false;

    /**
     * Flag whether you're using map-to-liveDepth factors
     */
    bool useMap2LiveFactors = false;

    /**
     * Number of submap factors per frame / submap
     */
    size_t numSubmapFactors = 200;

    /**
     * Resolution of voxel grid used for downsampling of point clouds for alignment
     */
    float voxelGridResolution = 0.1f;

    /**
     * LiDAR / Depth Sensor Error
     */
    float sensorError = 0.01f;

    /**
     * Flag whether you're using network predicted depth uncertainty
     */
    bool useUncertainty = false;

    /**
     * Scaling factor that every depth image value is multiplied with to convert the image to meters
     * We need to make sure to provide floating point images with meters as unit to SE2.
    */
    float depthScalingFactor = 1.0f;

    /**
     * When mapping, this value defines the minimum depth of the frustrum that will be mapped
    */
    float near_plane = 0.1;
    
    /**
     * When mapping, this value defines the maximum depth of the frustrum that will be mapped
    */
    float far_plane = 5.0;

    /** Default Constructor
     */
    SubMapConfig() : resultsDirectory("/home/"), write_mesh_output(false),
      sensorMeasurementDownsampling(1), depthImageResDownsampling(1){};

    /** Initializes the config from a YAML file. Data not present in the YAML file will be initialized
     * as in MapConfig::MapConfig().
     */
    SubMapConfig(const std::string& yaml_file)
    {
      // Open the file for reading.
      cv::FileStorage fs;
      try {
        if (!fs.open(yaml_file, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML)) {
          std::cerr << "Error: couldn't read configuration file " << yaml_file << "\n";
          return;
        }
      }
      catch (const cv::Exception& e) {
        // OpenCV throws if the file contains non-YAML data.
        std::cerr << "Error: invalid YAML in configuration file " << yaml_file << "\n";
        return;
      }

      // Get the node containing the general configuration.
      const cv::FileNode node = fs["general"];
      if (node.type() != cv::FileNode::MAP) {
        std::cerr << "Warning: using default map configuration, no \"map\" section found in "
                  << yaml_file << "\n";
        return;
      }

      // Read the config parameters.
      se::yaml::subnode_as_string(node, "results_directory", resultsDirectory);
      se::yaml::subnode_as_bool(node, "write_mesh_output", write_mesh_output);
      se::yaml::subnode_as_size_t(node, "sensor_measurement_downsampling", sensorMeasurementDownsampling);
      se::yaml::subnode_as_size_t(node, "depth_image_resolution_downsampling", depthImageResDownsampling);
      se::yaml::subnode_as_size_t(node, "submap_kf_threshold", submapKfThreshold);
      se::yaml::subnode_as_float(node, "submap_overlap_ratio", submapOverlapRatio);
      se::yaml::subnode_as_size_t(node, "submap_min_frames", submapMinFrames);
      se::yaml::subnode_as_bool(node, "use_map_to_map_factors", useMap2MapFactors);
      se::yaml::subnode_as_bool(node, "use_map_to_live_factors", useMap2LiveFactors);
      se::yaml::subnode_as_size_t(node, "n_factors_per_state", numSubmapFactors);
      se::yaml::subnode_as_float(node, "voxel_grid_resolution", voxelGridResolution);
      se::yaml::subnode_as_float(node, "sensor_error", sensorError);
      se::yaml::subnode_as_bool(node, "use_uncertainty", useUncertainty);
      se::yaml::subnode_as_float(node, "depth_scaling_factor", depthScalingFactor);
      se::yaml::subnode_as_float(node, "far_plane", far_plane);
      se::yaml::subnode_as_float(node, "near_plane", near_plane);
    }

    friend std::ostream& operator<<(std::ostream& os, const SubMapConfig& c)
    {
      os << str_utils::value_to_pretty_str(c.resultsDirectory, "results_directory") << " \n";
      os << str_utils::bool_to_pretty_str(c.write_mesh_output, "write_mesh_output") << " \n";
      os << str_utils::value_to_pretty_str(c.sensorMeasurementDownsampling, "sensor_measurement_downsampling") << " \n";
      os << str_utils::value_to_pretty_str(c.depthImageResDownsampling, "depth_image_res_downsampling") << " \n";
      os << str_utils::value_to_pretty_str(c.submapKfThreshold, "submap_kf_threshold") << " \n";
      os << str_utils::value_to_pretty_str(c.submapOverlapRatio, "submap_overlap_ratio") << " \n";
      os << str_utils::value_to_pretty_str(c.submapMinFrames, "submap_min_frames") << " \n";
      os << str_utils::bool_to_pretty_str(c.useMap2MapFactors, "use_map_to_map_factors") << " \n";
      os << str_utils::bool_to_pretty_str(c.useMap2LiveFactors, "use_map_to_live_factors") << " \n";
      os << str_utils::value_to_pretty_str(c.numSubmapFactors, "n_factors_per_state") << " \n";
      os << str_utils::value_to_pretty_str(c.voxelGridResolution, "voxel_grid_resolution") << " \n";
      os << str_utils::value_to_pretty_str(c.sensorError, "sensor_error") << " \n";
      os << str_utils::bool_to_pretty_str(c.useUncertainty, "use_uncertainty") << " \n";
      os << str_utils::value_to_pretty_str(c.depthScalingFactor, "depth_scaling_factor") << " \n";
      os << str_utils::value_to_pretty_str(c.far_plane, "far_plane") << " \n";
      os << str_utils::value_to_pretty_str(c.near_plane, "near_plane") << " \n";
      return os;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

template<typename MapT, typename SensorT>
struct Config {
    struct MapT::Config map;
    struct MapT::DataType::Config data;
    struct SensorT::Config sensor;

    /** Default initializes all configs.
     */
    Config();

    /** Initializes the config from a YAML file. Data not present in the YAML file will be
     * initialized as in Config::Config().
     */
    Config(const std::string& yaml_file);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const Config<MapT, SensorT>& c);



template<typename MapT, typename SensorT>
struct OkvisSubmapsConfig {
    SubMapConfig general;
    struct Config<MapT, SensorT> se_config;

    /** Default initializes all configs.
     */
    OkvisSubmapsConfig();

    /** Initializes the config from a YAML file. Data not present in the YAML file will be
     * initialized as in Config::Config().
     */
    OkvisSubmapsConfig(const std::string& yaml_file);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const OkvisSubmapsConfig<MapT, SensorT>& c);

} // namespace se

#include "okvis/impl/config_mapping_impl.hpp"

#endif // INCLUDE_OKVIS_LIDARMAPPING_CONFIG_HPP
