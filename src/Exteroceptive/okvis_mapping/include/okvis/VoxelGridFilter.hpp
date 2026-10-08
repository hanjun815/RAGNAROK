/*
 * This code is an adaptation from the implementation of KISS-ICP
 * https://github.com/PRBonn/kiss-icp
 * Vizzo, Ignacio, et al.
 * "Kiss-icp: In defense of point-to-point icp–simple, accurate, and robust registration if done the right way."
 * IEEE Robotics and Automation Letters 8.2 (2023): 1029-1036.
*/

#ifndef INCLUDE_OKVIS_VOXELGRIDFILTER_HPP
#define INCLUDE_OKVIS_VOXELGRIDFILTER_HPP

#include <Eigen/Core>
#include <vector>
#include <Eigen/StdVector>

#include <okvis/kinematics/Transformation.hpp>

namespace okvis{

    using Voxel = Eigen::Vector3i;

    struct VoxelHash
    {
        size_t operator()(const Voxel &voxel) const{
          const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
          return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };

    template<typename T>
    using Point3 = Eigen::Matrix<T,3,1>;

    template<typename T>
    using Point3Cloud = std::vector<Point3<T>, Eigen::aligned_allocator<Point3<T>>> ;
    /**
     * Function for voxel grid based downsampling of a point cloud. Keeps original points instead of centroids.
     * This implementation is heavily inspired by the Kiss-ICP Code.
     * @param raw_point_cloud   The original point cloud
     * @param voxel_size        Size of voxels of used voxel grid filter
     * @return                  The downsampled point cloud
     */
    template<typename T>
    Point3Cloud<T> VoxelDownSample(const Point3Cloud<T>& raw_point_cloud, double voxel_size);

    template<typename T>
    Point3Cloud<T> VoxelDownSample(const Point3Cloud<T>& raw_point_cloud, double voxel_size){
      std::unordered_map<Voxel, Point3<T>, VoxelHash> grid; // ToDo: Aligned Allocator needed here?
      grid.reserve(raw_point_cloud.size());

      for(const auto &point : raw_point_cloud){
        // This seems to assume a world-aligned (0-based) voxel grid
        const auto voxel = Voxel((point/voxel_size).template cast<int>());
        if(grid.count(voxel) == 0){
          grid.insert({voxel, point});
        }
      }

      Point3Cloud<T> downsampled_point_cloud;
      downsampled_point_cloud.reserve(grid.size());
      for(const auto &[voxel, point] : grid){
        (void)voxel;
        downsampled_point_cloud.push_back(point);
      }

      return downsampled_point_cloud;

    }

    /**
     * Struct containing a voxel map that can be incrementally updated
     */
    struct VoxelHashMap {
        /// Some helpful typedefs
        using Vector3dVector = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
        using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

        using Voxel = Eigen::Vector3i;
        struct VoxelBlock {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            // buffer of points with a max limit of n_points
            Vector3dVector points;
            int num_points_;

            inline void AddPoint(const Eigen::Vector3d &point) {
              if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
            }
        };
        struct VoxelHash {
            size_t operator()(const Voxel &voxel) const {
              const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
              return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
            }
        };

        explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
                : voxel_size_(voxel_size),
                  max_distance_(max_distance),
                  max_points_per_voxel_(max_points_per_voxel) {}

        inline void Clear() { map_.clear(); }
        inline bool Empty() const { return map_.empty(); }
        void Update(const Vector3dVector &points, const Eigen::Vector3d &origin);
        void Update(const Vector3dVector &points, const okvis::kinematics::Transformation &pose);
        void AddPoints(const Vector3dVector &points);
        void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
        Vector3dVector Pointcloud() const;

        double voxel_size_;
        double max_distance_;
        int max_points_per_voxel_;
        std::unordered_map<Voxel, VoxelBlock, VoxelHash> map_;
    };

}

#endif //INCLUDE_OKVIS_VOXELGRIDFILTER_HPP
