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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/sized_cost_function.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#pragma GCC diagnostic pop
#include <gtest/gtest.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ceres/SubmapIcpError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>


#include <okvis/mapTypedefs.hpp>

#include <okvis/config_mapping.hpp>
#include <se/map/io/mesh_io.hpp>
#include <se/common/point_cloud_io.hpp>

#include <random>
#include <boost/filesystem.hpp>

/***
 * Note: For these tests to run successfully, we have to disable the "Ceres Hack" in the implementation of the SubmapIcpError
 */

// Get the absolute path of the current source file
boost::filesystem::path source_file(__FILE__);
// Get the project directory by navigating up
boost::filesystem::path project_dir = source_file.parent_path().parent_path().parent_path(); 
boost::filesystem::path config_path = project_dir / "config/unit_test/se2-icp.yaml";

void save_point_cloud(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pointCloud,
                      std::string filename, const Eigen::Matrix4f & T_WS){
  se::Image<Eigen::Vector3f> pc(pointCloud.size(),1);
  for(size_t i = 0; i < pointCloud.size(); i++){
    pc[i] = pointCloud.at(i).cast<float>();
  }

  int test = save_point_cloud_vtk(pc, filename, Eigen::Isometry3f(T_WS));

  if(test == 0) LOG(INFO) << "Correctly saved pointcloud in file " + filename;
  else LOG(INFO) << "Incorrectly saved pointcloud";
}

void downsamplePointCloud(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& originalPointCloud,
                          std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& downsampledPointCloud,
                          size_t num_of_points)
{
  std::sample(originalPointCloud.begin(), originalPointCloud.end(), std::back_inserter(downsampledPointCloud), num_of_points,
              std::mt19937{std::random_device{}()});
}

TEST(okvisTestSuite, SubmapAlignmentJacobian){
  /**
   * Create plane wall example
   */
  const double elevation_min = -10.0;
  const double elevation_max = 10.0;
  const double azimuth_min = -15.0;
  const double azimuth_max = 15.0;
  // angular resolution [degree]
  const double elevation_res = 0.1;
  const double azimuth_res = 0.1;
  // conversion degree <-> rad
  const double deg_to_rad = M_PI / 180.;
  // distance of plane wall [m]
  const double d = 15.0;
  // measurement uncertainty
  double sigma_measurement = 0.05;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_a;
  size_t num_points_elevation = std::floor((elevation_max - elevation_min) / elevation_res);
  size_t num_points_azimuth = std::floor((azimuth_max - azimuth_min) / azimuth_res);

  double elevation_angle = elevation_min;
  double azimuth_angle = azimuth_min;
  double x, y, z;
  x = d;
  for(size_t i = 0; i < num_points_elevation; i++){
    z = d*tan(elevation_angle * deg_to_rad);
    for(size_t j = 0; j < num_points_azimuth; j++){
      y = d*tan(azimuth_angle * deg_to_rad);
      // save point
      pointCloud_a.push_back(Eigen::Vector3d(x,y,z));
      // increase azimuth angle
      azimuth_angle+=azimuth_res;
    }
    azimuth_angle = azimuth_min;
    //increase elevation angle
    elevation_angle+=elevation_res;
  }
  std::cout << "Created " << pointCloud_a.size() << " points" << std::endl;

  /**
   * Create reference submap in frame {A}
   * Scenario: Plane wall
   */

  // ========= Config & I/O INITIALIZATION  =========

  // ========= Map INITIALIZATION  =========
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(config_path.string());

  okvis::SupereightMapType::DataType::Config dataConfig;
  dataConfig.readYaml(config_path.string());
  okvis::SupereightMapType map(mapConfig, dataConfig);

  // ========= Sensor INITIALIZATION  =========
  se::Lidar::Config sensorConfig;
  sensorConfig.readYaml(config_path.string());
  const se::Lidar sensor(sensorConfig);

  // Setup input, processed and output imgs
  Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity();

  // ========= Integrator INITIALIZATION  =========
  se::MapIntegrator integrator(map);

  auto measurementIter = pointCloud_a.begin();
  int frame = 0;
  std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                        Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>> ray_measurements;
  for(; measurementIter!=pointCloud_a.end(); measurementIter++) {
    ray_measurements.emplace_back(Eigen::Isometry3f(T_WS), (*measurementIter).cast<float>());
  }

  integrator.integrateRayBatch<se::Lidar>(frame, ray_measurements, sensor);
  std::cout << "Finished Integration. Saving submap (mesh & slice)" << std::endl;


  // Save mesh and slice for testing purposes
  //map.saveMesh("testMesh.ply");
  //map.saveMeshVoxel("testMeshVoxel.ply");
  //map.saveStructure("testStructure.ply");
  //map.saveFieldSlices("testSliceX.vtk", "testSliceY.vtk", "testSliceZ.vtk", Eigen::Vector3f(0.,0.,0.));

  /**
   * Check Jacobians in a linear region
   */

  Eigen::Vector3d p_b(15.0, 0., 0.);
  okvis::kinematics::Transformation T_WS_a, T_WS_b;
  T_WS_a.setIdentity();
  T_WS_b.setIdentity();
  okvis::ceres::PoseParameterBlock poseParameterBlock_jac0(T_WS_a,0,okvis::Time(0));
  okvis::ceres::PoseParameterBlock poseParameterBlock_jac1(T_WS_b,0,okvis::Time(0));

  // Set up parameter blocks and compute analytic jacobians
  double *parameters[2];
  parameters[0] = poseParameterBlock_jac0.parameters();
  parameters[1] = poseParameterBlock_jac1.parameters();
  double *jacobians[2];
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J0;
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J1;
  jacobians[0] = J0.data();
  jacobians[1] = J1.data();
  Eigen::Matrix<double, 1, 1> residual;

  ::ceres::CostFunction *cost_function = new okvis::ceres::SubmapIcpError(map, p_b, sigma_measurement);
  static_cast<okvis::ceres::SubmapIcpError *>(cost_function)->EvaluateWithMinimalJacobians(parameters,
                                                                                           residual.data(),
                                                                                           jacobians, NULL);

  // Numerical Jacobian w.r.t. T_WS_a
  double dx = 0.015; // Do not choose too fine for given map resolution
  double *tmp_jacobians[2];
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> tmpJ0;
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> tmpJ1;
  tmp_jacobians[0] = tmpJ0.data();
  tmp_jacobians[1] = tmpJ1.data();

  // w.r.t. pose {A}
  Eigen::Matrix<double, 1, 6> J0_numDiff;
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_0;
    Eigen::Matrix<double, 1, 1> residual_p;
    Eigen::Matrix<double, 1, 1> residual_m;
    dp_0.setZero();
    dp_0[i] = dx;
    okvis::ceres::PoseManifold::plus(parameters[0], dp_0.data(), parameters[0]);
    static_cast<okvis::ceres::SubmapIcpError *>(cost_function)->Evaluate(parameters, residual_p.data(), tmp_jacobians);
    poseParameterBlock_jac0.setEstimate(T_WS_a);
    parameters[0] = poseParameterBlock_jac0.parameters();// reset
    dp_0[i] = -dx;
    okvis::ceres::PoseManifold::plus(parameters[0], dp_0.data(), parameters[0]);
    static_cast<okvis::ceres::SubmapIcpError *>(cost_function)->Evaluate(parameters, residual_m.data(), tmp_jacobians);
    poseParameterBlock_jac0.setEstimate(T_WS_a);
    parameters[0] = poseParameterBlock_jac0.parameters(); // reset


    J0_numDiff.col(i) = (residual_p - residual_m) / (2. * dx);

  }

  // Use lift Jacobian for non-minimal Jacobian
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J0_numDiff_lift;
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> liftJac0;
  okvis::ceres::PoseManifold::minusJacobian(parameters[0], liftJac0.data());
  J0_numDiff_lift = J0_numDiff * liftJac0;

  EXPECT_LT((J0_numDiff_lift - J0).norm(), 0.1)
                << "J0: Jacobian Evaluation leads error  " << (J0_numDiff_lift - J0).norm() << " > 0.1"
                << std::endl;


  std::cout << "J0: Analytical Jacobian evaluates to: \n" << J0 << std::endl;
  std::cout << "J0: Numerical Jacobian evaluates to: \n" << J0_numDiff_lift << std::endl;
  // std::cout << "J0: J0_numDiff: \n" << J0_numDiff << std::endl;

  // Check Jacobian w.r.t. T_WS_b
  Eigen::Matrix<double, 1, 6> J1_numDiff;
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_1;
    Eigen::Matrix<double, 1, 1> residual_p;
    Eigen::Matrix<double, 1, 1> residual_m;
    dp_1.setZero();
    dp_1[i] = dx;
    okvis::ceres::PoseManifold::plus(parameters[1], dp_1.data(), parameters[1]);
    static_cast<okvis::ceres::SubmapIcpError *>(cost_function)->Evaluate(parameters, residual_p.data(), tmp_jacobians);
    poseParameterBlock_jac1.setEstimate(T_WS_b);
    parameters[1] = poseParameterBlock_jac1.parameters();// reset
    dp_1[i] = -dx;
    okvis::ceres::PoseManifold::plus(parameters[1], dp_1.data(), parameters[1]);
    static_cast<okvis::ceres::SubmapIcpError *>(cost_function)->Evaluate(parameters, residual_m.data(), tmp_jacobians);
    poseParameterBlock_jac1.setEstimate(T_WS_b);
    parameters[1] = poseParameterBlock_jac1.parameters(); // reset


    J1_numDiff.col(i) = (residual_p - residual_m) / (2. * dx);

  }

  // Use lift Jacobian for non-minimal Jacobian
  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J1_numDiff_lift;
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> liftJac1;
  okvis::ceres::PoseManifold::minusJacobian(parameters[1], liftJac1.data());
  J1_numDiff_lift = J1_numDiff * liftJac1;

  EXPECT_LT((J1_numDiff_lift - J1).norm(), 0.1)
                << "J1: Jacobian Evaluation leads error  " << (J1_numDiff_lift - J1).norm() << " > 0.1"
                << std::endl;

  std::cout << "J1: Jacobian evaluates to: \n" << J1 << std::endl;
  std::cout << "J1: Numerical Jacobian evaluates to: \n" << J1_numDiff_lift << std::endl;
}

TEST(okvisTestSuite, SubmapAlignmentOptimisation){
  /**
    * Create room corner example
    */
  const double angular_resolution = 0.25;
  const double deg_to_rad = M_PI / 180.;
  const double d = 5.0;
  double sigma_measurement = 0.05;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_a;
  size_t num_points = std::floor(90.0 / angular_resolution);

  double elevation_angle = -45.0;
  double azimuth_angle = -45.0;
  double x, y, z;

  // Wall looking frontward (fw: x)
  x = d;
  for(size_t i = 0; i < num_points; i++){
    z = d*tan(elevation_angle * deg_to_rad);
    for(size_t j = 0; j < num_points; j++){
      y = d*tan(azimuth_angle * deg_to_rad);
      // save point
      pointCloud_a.push_back(Eigen::Vector3d(x,y,z));
      // increase azimuth angle
      azimuth_angle+=angular_resolution;
    }
    azimuth_angle = -45.;
    //increase elevation angle
    elevation_angle+=angular_resolution;
  }

  // Wall looking left (left: y)
  elevation_angle = -45.0;
  azimuth_angle = -45.0;
  y = d;
  for(size_t i = 0; i < num_points; i++){
    z = d*tan(elevation_angle * deg_to_rad);
    for(size_t j = 0; j < num_points; j++){
      x = d*tan(azimuth_angle * deg_to_rad);
      // save point
      pointCloud_a.push_back(Eigen::Vector3d(x,y,z));
      // increase azimuth angle
      azimuth_angle+=angular_resolution;
    }
    azimuth_angle = -45.;
    //increase elevation angle
    elevation_angle+=angular_resolution;
  }

  // Wall looking down (down: -z)
  elevation_angle = -45.0;
  azimuth_angle = -45.0;
  z = -d;
  for(size_t i = 0; i < num_points; i++){
    y = d*tan(elevation_angle * deg_to_rad);
    for(size_t j = 0; j < num_points; j++){
      x = d*tan(azimuth_angle * deg_to_rad);
      // save point
      pointCloud_a.push_back(Eigen::Vector3d(x,y,z));
      // increase azimuth angle
      azimuth_angle+=angular_resolution;
    }
    azimuth_angle = -45.;
    //increase elevation angle
    elevation_angle+=angular_resolution;
  }


  // Save Point Cloud
  save_point_cloud(pointCloud_a, "cloud_a.vtk", Eigen::Matrix4f::Identity());
  std::cout << "Created " << pointCloud_a.size() << " points" << std::endl;

  // Transform point cloud with random transformation
  okvis::kinematics::Transformation T_Sa_Sb;
  T_Sa_Sb.setRandom(5.0, deg_to_rad * 10.0);
  // Transform Point Cloud
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_b(pointCloud_a.size());
  for(size_t i = 0; i < pointCloud_a.size(); i++){
    pointCloud_b.at(i) = (T_Sa_Sb.inverse().T() * pointCloud_a.at(i).homogeneous()).head<3>();
  }
  std::cout << "Transformed point cloud has size:  " << pointCloud_b.size() << " points" << std::endl;

  save_point_cloud(pointCloud_b, "cloud_b.vtk", Eigen::Matrix4f::Identity());

  /**
   * Create reference submap in frame {A}
   * Scenario: Corner of a room
   */


  // ========= Map INITIALIZATION  =========
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(config_path.string());

  okvis::SupereightMapType::DataType::Config dataConfig;
  dataConfig.readYaml(config_path.string());
  okvis::SupereightMapType map(mapConfig, dataConfig);

  // ========= Sensor INITIALIZATION  =========
  se::Lidar::Config sensorConfig;
  sensorConfig.readYaml(config_path.string());
  const se::Lidar sensor(sensorConfig);

  // Setup input, processed and output imgs
  Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity(); //< assume identity, no body-to-sensor transformation for testing

  // ========= Integrator INITIALIZATION  =========
  se::MapIntegrator integrator(map);

  auto measurementIter = pointCloud_a.begin();
  int frame = 0;
  std::vector<std::pair<Eigen::Isometry3f, Eigen::Vector3f>,
                        Eigen::aligned_allocator<std::pair<Eigen::Isometry3f, Eigen::Vector3f>>> ray_measurements;
  for(; measurementIter!=pointCloud_a.end(); measurementIter++) {
    ray_measurements.emplace_back(Eigen::Isometry3f(T_WS), (*measurementIter).cast<float>());
  }

  integrator.integrateRayBatch(frame, ray_measurements, sensor);  

  std::cout << "Finished Integration. Saving submap (mesh & slice)" << std::endl;
  // Save mesh and slice for testing purposes
//  map.saveMesh("testMesh.ply");
//  map.saveMeshVoxel("testMeshVoxel.ply");
//  map.saveStructure("testStructure.ply");
//  map.saveFieldSlices("testSliceX.vtk", "testSliceY.vtk", "testSliceZ.vtk", Eigen::Vector3f(0.,0.,0.));
//  map.saveFieldGradSlices("gradSliceX.vtk", "gradSliceY.vtk", "gradSliceZ.vtk", Eigen::Vector3f(0.,0.,0.));


  /**
   * Set up Ceres Problem
   */
  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership =
          ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership =
          ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
          ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  ::ceres::Problem problem(problemOptions);
  ::ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold=google::WARNING; // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  /**
   * Set up reference frame {A} of the submap
   */

  // General stuff and ground truth trafos
  okvis::ceres::PoseManifold pose6dParameterisation;
  // T_WS_a
  okvis::kinematics::Transformation T_WS_a;
  T_WS_a.setIdentity(); // Ground Truth
  okvis::ceres::PoseParameterBlock poseParameterBlock_a(T_WS_a,0,okvis::Time(0));
  problem.AddParameterBlock(poseParameterBlock_a.parameters(), okvis::ceres::PoseParameterBlock::Dimension, &pose6dParameterisation);
  // T_WS_b & T_Sa_Sb
  okvis::kinematics::Transformation T_WS_b;
  T_WS_b = T_WS_a * T_Sa_Sb;
  okvis::ceres::PoseParameterBlock poseParameterBlock_b(T_WS_b,0,okvis::Time(0));
  problem.AddParameterBlock(poseParameterBlock_b.parameters(), okvis::ceres::PoseParameterBlock::Dimension, &pose6dParameterisation);
  // Creat disturbance
  okvis::kinematics::Transformation T_dist;
  T_dist.setRandom(0.05, deg_to_rad * 1.0);

  // Create
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_b_disturbed;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_b_disturbed_downsampled;
  std::default_random_engine genx;
  genx.seed(1);
  std::normal_distribution<double> distributionx(0.0, sigma_measurement);
  std::default_random_engine geny;
  geny.seed(2);
  std::normal_distribution<double> distributiony(0.0, sigma_measurement);
  std::default_random_engine genz;
  genz.seed(3);
  std::normal_distribution<double> distributionz(0.0, sigma_measurement);

  // Outliers
  int outlier_count = 100; // count 20 = 5% outlier probability
  double outlier_amplitude = 5.0;
  std::default_random_engine outx;
  outx.seed(4);
  std::normal_distribution<double> outlierdistx(0.0, outlier_amplitude);
  std::default_random_engine outy;
  outy.seed(5);
  std::normal_distribution<double> outlierdisty(0.0, outlier_amplitude);
  std::default_random_engine outz;
  outz.seed(6);
  std::normal_distribution<double> outlierdistz(0.0, outlier_amplitude);

  // Add residual blocks
  int count = 0;
  for(auto p_b : pointCloud_b){
    double sigx = distributionx(genx);
    double sigy = distributiony(geny);
    double sigz = distributionz(genz);
    Eigen::Vector3d disturbed_point;
    disturbed_point = p_b;
    disturbed_point.x()+=sigx;
    disturbed_point.y()+=sigy;
    disturbed_point.z()+=sigz;

    if(count % outlier_count ==0){
      disturbed_point.x() += outlierdistx(outx);
      disturbed_point.y() -= outlierdisty(outy);
      disturbed_point.z() += outlierdistz(outz);
    }

    count++;
    pointCloud_b_disturbed.push_back(disturbed_point);

    //::ceres::CostFunction* cost_function = new okvis::ceres::SubmapIcpError(map, disturbed_point, sigma_measurement);
    //problem.AddResidualBlock(cost_function, NULL, poseParameterBlock_a.parameters(), poseParameterBlock_b.parameters());
  }
  save_point_cloud(pointCloud_b_disturbed, "cloud_disturbed.vtk", Eigen::Matrix4f::Identity());

  downsamplePointCloud(pointCloud_b_disturbed, pointCloud_b_disturbed_downsampled, 1000);
  std::cout << "Down-sampled point cloud to: " << pointCloud_b_disturbed_downsampled.size() << " points. Adding residuals..." << std::endl;

  // Downsample and add residual blocks
  for(Eigen::Vector3d pt : pointCloud_b_disturbed_downsampled){
    ::ceres::CostFunction* cost_function = new okvis::ceres::SubmapIcpError(map, pt, sigma_measurement);
    problem.AddResidualBlock(cost_function, NULL, poseParameterBlock_a.parameters(), poseParameterBlock_b.parameters());
  }


  // Problem 1: Set {A} const and disturb & optimize for {B}
  std::cout << "### Problem 1: Set {A} const and disturb & optimize for {B} ###" << std::endl;
  problem.SetParameterBlockConstant(poseParameterBlock_a.parameters());
  okvis::kinematics::Transformation T_WS_b_dist = T_WS_b * T_dist;
  poseParameterBlock_b.setEstimate(T_WS_b_dist);

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  Solve(options, &problem, &summary);

  std::cout << " ------ T_WSa (optimised) ------ \n" << poseParameterBlock_a.estimate().T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;
  std::cout << " ------ T_WSb ------ \n" << T_WS_b.T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;
  std::cout << " ------ T_WS_b_dist ------ \n" << T_WS_b_dist.T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;
  std::cout << " ------ T_WSb (optimised) ------ \n" << poseParameterBlock_b.estimate().T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;

  // Transform Point Cloud
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> opt;
  for(size_t i = 0; i < pointCloud_b_disturbed_downsampled.size(); i++){
    opt.push_back(T_WS_a.T3x4() * poseParameterBlock_b.estimate().T() * pointCloud_b_disturbed_downsampled.at(i).homogeneous());
  }
  std::cout << "Transformed point cloud has size:  " << opt.size() << " points" << std::endl;

  save_point_cloud(opt, "opt.vtk", Eigen::Matrix4f::Identity());

  // Translation Error
  EXPECT_LT((T_WS_b.r()-poseParameterBlock_b.estimate().r()).norm(), 1.5e-02) << "Translation Error too large!";
  // Orientation Error
  EXPECT_LT(2*(T_WS_b.q()*poseParameterBlock_b.estimate().q().inverse()).vec().norm(), 1e-02) << "Orientation Error too large!";

  // Problem 2: Set {B} const and disturb & optimize for {A}

  // transform disturbed point cloud into frame A for visualisation
  // Transform Point Cloud
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointCloud_dist_in_a;
  for(size_t i = 0; i < pointCloud_b_disturbed_downsampled.size(); i++){
    pointCloud_dist_in_a.push_back(T_WS_a.T3x4() * T_WS_b_dist.T() * pointCloud_b_disturbed_downsampled.at(i).homogeneous());
  }
  std::cout << "Transformed point cloud has size:  " << pointCloud_dist_in_a.size() << " points" << std::endl;

  save_point_cloud(pointCloud_dist_in_a, "pointCloud_dist_in_a.vtk", Eigen::Matrix4f::Identity());

  std::cout << "### Problem 2: Set {B} const and disturb & optimize for {A} ###" << std::endl;

  // Do the same fixing frame {B} and optimising frame {A}
  poseParameterBlock_b.setEstimate(T_WS_b);
  problem.SetParameterBlockConstant(poseParameterBlock_b.parameters());
  okvis::kinematics::Transformation T_WS_a_dist = T_WS_a * T_dist;
  poseParameterBlock_a.setEstimate(T_WS_a_dist);
  problem.SetParameterBlockVariable(poseParameterBlock_a.parameters());
  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  Solve(options, &problem, &summary);

  std::cout << " ------ T_WSa ------ \n" << T_WS_a.T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;
  std::cout << " ------ T_WS_a_dist ------ \n" << T_WS_a_dist.T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;
  std::cout << " ------ T_WSa (optimised) ------ \n" << poseParameterBlock_a.estimate().T() << std::endl;
  std::cout << " ------------------------------- " << std::endl;

  EXPECT_LT((T_WS_a.r()-poseParameterBlock_a.estimate().r()).norm(), 1.5e-02) << "Translation Error too large!";
  // Orientation Error
  EXPECT_LT(2*(T_WS_a.q()*poseParameterBlock_a.estimate().q().inverse()).vec().norm(), 1e-02) << "Orientation Error too large!";

}