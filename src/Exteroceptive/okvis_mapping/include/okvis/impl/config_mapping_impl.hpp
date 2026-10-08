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

#ifndef INCLUDE_OKVIS_LIDARMAPPING_CONFIG_IMPL_HPP
#define INCLUDE_OKVIS_LIDARMAPPING_CONFIG_IMPL_HPP

#include "se/common/yaml.hpp"

namespace se {


template<typename MapT, typename SensorT>
Config<MapT, SensorT>::Config()
{
}



template<typename MapT, typename SensorT>
Config<MapT, SensorT>::Config(const std::string& yaml_file)
{
    map.readYaml(yaml_file);
    data.readYaml(yaml_file);
    sensor.readYaml(yaml_file);
}



template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const Config<MapT, SensorT>& c)
{
    os << "Data config -----------------------\n";
    os << c.data;
    os << "Map config ------------------------\n";
    os << c.map;
    os << "Sensor config ---------------------\n";
    os << c.sensor;
    return os;
}


template<typename MapT, typename SensorT>
OkvisSubmapsConfig<MapT, SensorT>::OkvisSubmapsConfig()
{
}



template<typename MapT, typename SensorT>
OkvisSubmapsConfig<MapT, SensorT>::OkvisSubmapsConfig(const std::string& yaml_file) :
        general(yaml_file), se_config(yaml_file)
{
}



template<typename MapT, typename SensorT>
std::ostream& operator<<(std::ostream& os, const OkvisSubmapsConfig<MapT, SensorT>& c)
{
  os << "General submap config -----------------------\n";
  os << c.general;
  os << "SE2 config -----------------------\n";
  os << c.se_config;
  return os;
}


} // namespace se

#endif // INCLUDE_OKVIS_LIDARMAPPING_CONFIG_IMPL_HPP
