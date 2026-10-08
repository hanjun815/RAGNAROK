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

#ifndef INCLUDE_OKVIS_MAP_TYPEDEFS_HPP
#define INCLUDE_OKVIS_MAP_TYPEDEFS_HPP

#include <se/supereight.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

    #if defined(OKVIS_COLIDMAP) && OKVIS_COLIDMAP
    typedef se::OccupancyColIdMap<> SupereightMapType;
    #else
    typedef se::OccupancyMap<se::Res::Multi> SupereightMapType;
    #endif
}

#endif //INCLUDE_OKVIS_MAP_TYPEDEFS_HPP
