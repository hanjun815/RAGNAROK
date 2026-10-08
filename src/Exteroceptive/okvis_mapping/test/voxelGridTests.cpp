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

#include <iostream>

#include <gtest/gtest.h>

#include <okvis/VoxelGridFilter.hpp>
#include <se/common/point_cloud_io.hpp>

/**
 * ToDo:
 * Check other functions such as Update and neighbor checking (only used voxel grid for downsampling so far) and
 * unused variables, pose updates, remove points far away etc.
 */

void save_point_cloud(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pointCloud,
                      std::string filename, const Eigen::Matrix4f & T_WS){
  se::Image<Eigen::Vector3f> pc(pointCloud.size(),1);
  for(size_t i = 0; i < pointCloud.size(); i++){
    pc[i] = pointCloud.at(i).cast<float>();
  }
  int test = save_point_cloud_vtk(pc, filename, Eigen::Isometry3f(T_WS));
}

TEST(VoxelGridFilter, downsampling) {
  // Create Random point cloud (cube of dimension dim and resolution res)
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> raw_point_cloud;
  constexpr double dim = 10.0;
  constexpr double res = 1.0;

  int n_of_points_per_dim = static_cast<int> (dim / res);

  double x,y,z = 0.0;

  while( x<= dim){
    while (y <= dim){
      while(z <=dim){
        Eigen::Vector3d point(x,y,z);
        raw_point_cloud.push_back(point);
        z += res;
      }
      z = 0.0;
      y += res;
    }
    y = 0.0;
    x += res;
  }

  // Check that created Point Cloud has correct size
  std::cout << "raw point cloud has size: " << raw_point_cloud.size() << std::endl;
  EXPECT_EQ(raw_point_cloud.size(), std::pow(n_of_points_per_dim+1, 3));


  // Downsampling with a resolution of 2.0m
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> downsampled_point_cloud;
  downsampled_point_cloud = okvis::VoxelDownSample(raw_point_cloud, 5.0);
  // Check that downsampled Point Cloud has correct size
  int n_of_downsampled_points_per_dim = static_cast<int> (dim / 5.0);
  std::cout << "downsampled point cloud has size: " << downsampled_point_cloud.size() << std::endl;
  EXPECT_EQ(downsampled_point_cloud.size(), std::pow(n_of_downsampled_points_per_dim+1, 3));
  //save_point_cloud(raw_point_cloud,"raw.vtk", Eigen::Matrix4f::Identity());
  //save_point_cloud(downsampled_point_cloud,"downsampled.vtk", Eigen::Matrix4f::Identity());

  EXPECT_LE(downsampled_point_cloud.size(), raw_point_cloud.size());

  // Check that original points are used
  for(auto pt: downsampled_point_cloud){
    int count = 0;

    for(auto raw_pt : raw_point_cloud){
      if((raw_pt-pt).norm() < 1e-07){
        count++;
      }
    }

    EXPECT_TRUE(count == 1);
  }
}

TEST(VoxelGridFilter, VoxelHashMap){

  /// Create very fine resolution point cloud (cube of dimension dim and resolution res)
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> raw_point_cloud;
  constexpr double dim = 10.0;
  constexpr double res = 0.5;
  int n_of_points_per_dim = static_cast<int> (dim / res);

  double x,y,z = 0.0;

  while( x<= dim){
    while (y <= dim){
      while(z <=dim){
        Eigen::Vector3d point(x,y,z);
        raw_point_cloud.push_back(point);
        z += res;
      }
      z = 0.0;
      y += res;
    }
    y = 0.0;
    x += res;
  }

  // Check that created Point Cloud has correct size
  std::cout << "Created point cloud of size " << raw_point_cloud.size() << "." << std::endl;
  EXPECT_EQ(raw_point_cloud.size(), std::pow(n_of_points_per_dim+1, 3));

  // Initialise VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
  okvis::VoxelHashMap map(2.0, 0.0, 1);

  // split raw point cloud into chunks of size 10
  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> chunks;
  for(size_t i = 0; i < raw_point_cloud.size(); i+=10) {
    auto last = std::min(i+10, raw_point_cloud.size());
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> chunk(raw_point_cloud.begin() + i, raw_point_cloud.begin() + last);
    chunks.emplace_back(chunk);
  }
  std::cout << "Split into " << chunks.size() << " chunks. Adding to voxel hash map." << std::endl;

  // Iterate through point cloud and add points
  for(auto chunk : chunks){
    map.AddPoints(chunk);
  }

  // Return final point cloud in voxel grid
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> downsampled_point_cloud;
  downsampled_point_cloud = map.Pointcloud();

  // Check Sizes
  std::cout << "downsampled point cloud of size " << downsampled_point_cloud.size() << "." << std::endl;
  EXPECT_LE(downsampled_point_cloud.size(), raw_point_cloud.size());
  int n_of_downsampled_points_per_dim = static_cast<int> (dim / 2.0);
  std::cout << "downsampled point cloud has size: " << downsampled_point_cloud.size() << std::endl;
  EXPECT_EQ(downsampled_point_cloud.size(), std::pow(n_of_downsampled_points_per_dim+1, 3));

  // Check that points are subset of original poitns
  for(auto pt: downsampled_point_cloud){
    int count = 0;
    for(auto raw_pt : raw_point_cloud){
      if((raw_pt-pt).norm() < 1e-07){
      count++;
      }
    }
    EXPECT_TRUE(count == 1);
  }

  // Visualize
  //save_point_cloud(raw_point_cloud,"raw.vtk", Eigen::Matrix4f::Identity());
  //save_point_cloud(downsampled_point_cloud,"downsampled.vtk", Eigen::Matrix4f::Identity());





}