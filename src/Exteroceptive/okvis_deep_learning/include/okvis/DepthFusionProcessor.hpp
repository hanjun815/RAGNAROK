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

#ifndef DEPTHFUSION_PROCESSOR_HPP
#define DEPTHFUSION_PROCESSOR_HPP

#include <map>
#include <thread>
#include <atomic>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <okvis/Measurements.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/QueuedTrajectory.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViSensorBase.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/Stereo2DepthProcessor.hpp>

namespace okvis {

class DepthFusionProcessor : public okvis::Stereo2DepthProcessor {

  struct StereoNetworkData {
    cv::Mat leftImage;
    cv::Mat rightImage; 
    torch::Tensor depth; // depth
    torch::Tensor sigma; // 1-sigma of depth
  };

  struct MultiViewData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat image;
    okvis::kinematics::Transformation T_WC;
  };

  typedef okvis::Measurement<MultiViewData> SrcFrameMeasurement;
  typedef okvis::Measurement<StereoNetworkData> StereoPrediction;

  struct VisualizationData {
    StereoMeasurement frame;
    std::vector<cv::Mat> srcframes;
    cv::Mat sparseDepthImage;
    cv::Mat stereoDepthImage;
    cv::Mat stereoSigmaImage;
    cv::Mat mvsDepthImage;
    cv::Mat mvsSigmaImage;
    cv::Mat fuseDepthImage;
    cv::Mat fuseSigmaImage;
  };

  struct ViStates {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::TrackingState trackingState; ///< The tracking state.
    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updatedStates; ///< Updated states (may be empty).
    std::shared_ptr<const okvis::MapPointVector> landmarks; ///< updated landmarks
  };

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

    DepthFusionProcessor(okvis::ViParameters &parameters, std::string modelDir);

    virtual ~DepthFusionProcessor();

    /// @brief Display stereo and motion images and sigma
    virtual void display(std::map<std::string, cv::Mat> &images) override;

    /// @brief Set the images callback before OKVIS2 graph optimization
    /// @param imagesCallback The depth and 1-sigma images callback to register.
    void setLiveDepthImageCallback(const LiveDepthCallback& liveDepthCallback) override final {
      liveDepthCallback_ = liveDepthCallback;
    }

    /**
     * @brief      Stores the state and keyframe updates provided by OKVIS
     *
     * @param[in]  latestState          The current OKVIS state
     * @param[in]  latestTrackingState  The current tracking state
     * @param[in]  keyframeStates       The state of the updated Keyframes
     * @param[in]  landmarks All the landmarks that the estimator updated.
     * @return     True when successful
     */
    bool stateUpdateCallback(const okvis::State &latestState,
                             const okvis::TrackingState &latestTrackingState,
                             std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updatedStates,
                             std::shared_ptr<const okvis::MapPointVector> landmarks) override final;

    /// @brief Check whether the processor is finished.
    bool finishedProcessing() override;

    void setFinished(){
      std::lock_guard<std::mutex> l(finishMutex_);
      shutdown_ = true;
      return;
    }

    bool isFinished(){
      std::lock_guard<std::mutex> l(finishMutex_);
      return shutdown_;
    }


  private:

    /// @brief Processing stereo images and according threads.
    void processingStereo();

    /// @brief Processing motion-stereo images and according threads.
    void processingMotion();
    
    /// @brief Function where the actual neural network predicts the depth from the stereo images
    /// @param sframe The stereo images measurement which will then be processed by the neural network
    void processStereoNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& sframe);

    /// @brief Function where the actual neural network predicts the depth from the multi-view images
    /// @param sframe The stereo images measurement which will then be processed by the neural network
    void processMVSnetwork(StereoPrediction& stereoPrediction);

    void updateTrajectory(ViStates& states);

    void setIntrinsicsTensors(const Eigen::VectorXd intrinsics);

    bool isSrcFrame(const okvis::kinematics::Transformation& T);
    
    std::thread stereoThread_;
    std::thread motionThread_;
    torch::jit::script::Module stereoModel_;
    torch::jit::script::Module mvsModel_;
    torch::Tensor curK0Tensor_; // [4,4] intrinsics for the half resolution
    torch::Tensor curInvK1Tensor_; // [4,4] inverse intrinsics for the quater resolution
    torch::Tensor srcK1Tensor_; // [N,4,4] source image intrinsics for the quater resolution
    LiveDepthCallback liveDepthCallback_; // callback to ThreadedSlam before optimization

    okvis::threadsafe::Queue<VisualizationData> visualisationsQueue_;
    okvis::threadsafe::Queue<ViStates> statesQueue_;
    okvis::threadsafe::Queue<StereoPrediction> stereoPredictionQueue_;
    std::mutex finishMutex_; // mutex for checking if the process has finished

    int imgHalfWidth_; // 0.5*imgWidth_
    int imgHalfHeight_; // 0.5*imgHeight_
    float principalU_;
    float principalV_;
    float fb_; // focalLength_*baseline_

    const int numSrc_ = 7; // the number of source view
    std::deque<SrcFrameMeasurement> srcFrame_; // Past frames for multi-view stereo
    okvis::Trajectory trajectory_; // this gives up-to-date keyframe pose and live pose
    okvis::kinematics::Transformation T_SC_; // T_{sensor(IMU)}{0th(left)-camera}
    okvis::kinematics::Transformation T_CS_; // Inverse of T_SC_
    std::map<okvis::Time, okvis::MapPointVector> landmarks_; // updated landmarks at corresponding stateId

    const int sensorMeasurementDownsampling_ = 1;
    int sensorMeasurementDownsamplingCounter_ = 0;
    unsigned int srcSelectCnt_ = 0;
    okvis::Duration imageTimeDelay_;  // Actual image time = nominal image time - imageTimeDelay_.
};
} // namespace okvis

#endif // DEPTHFUSION_NETWORK_HPP
