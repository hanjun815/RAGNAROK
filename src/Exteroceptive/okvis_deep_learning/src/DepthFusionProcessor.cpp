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

 #include <opencv2/core/core.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
 
 #include <glog/logging.h>
 #include <torch/cuda.h>

 #include <okvis/timing/Timer.hpp>
 
 #include <okvis/DepthFusionProcessor.hpp>
 #include <okvis/utils.hpp>
 
 namespace okvis {
 
 DepthFusionProcessor::DepthFusionProcessor(okvis::ViParameters &parameters, std::string modelDir) {

   stereoDnnApplyImageEnhancement_ = parameters.camera.stereo_dnn_apply_image_enhancement;
   stereoDnnFrontendParams_ = parameters.frontend;
   if (stereoDnnApplyImageEnhancement_) {
     LOG(INFO) << "Stereo DNN: frontend image enhancement enabled for network inputs "
                  "(stereo_dnn_apply_image_enhancement; uses frontend_parameters, not BO-tuned weights).";
   }
 
   // Remember the image time delay for compensation.
   imageTimeDelay_ = Duration(parameters.camera.image_delay);
 
   if (parameters.camera.stereo_indices.size() != 2) {
     throw std::runtime_error(
         "DepthFusionProcessor requires camera_parameters.deep_stereo_indices to list exactly two cameras [left, right].");
   }
 
   idLeft_ = parameters.camera.stereo_indices[0];
   idRight_ = parameters.camera.stereo_indices[1];
 
   Eigen::VectorXd intrinsics;
   kinematics::Transformation T_lr;
   if(!parameters.nCameraSystem.cameraType(idLeft_).depthType.needRectify) {
     needRectify_ = false;
     imgWidth_ = parameters.nCameraSystem.cameraGeometry(idLeft_)->imageWidth();
     imgHeight_ = parameters.nCameraSystem.cameraGeometry(idLeft_)->imageHeight();
     parameters.nCameraSystem.cameraGeometry(idLeft_)->getIntrinsics(intrinsics);
     T_lr = parameters.nCameraSystem.T_SC(idLeft_)->inverse() *
       (*parameters.nCameraSystem.T_SC(idRight_));
     T_SC_ = *parameters.nCameraSystem.T_SC(idLeft_);
   } else {
     needRectify_ = true;
     imgWidth_ = parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageWidth();
     imgHeight_ = parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageHeight();
     parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->getIntrinsics(intrinsics);
     T_lr = parameters.nCameraSystem.rectifyT_SC(idLeft_)->inverse() *
       (*parameters.nCameraSystem.rectifyT_SC(idRight_));
     T_SC_ = *parameters.nCameraSystem.rectifyT_SC(idLeft_);
   }
   T_CS_ = T_SC_.inverse();
 
   // Set internal camera parameters for disparity -> depth conversion.
   focalLength_ = static_cast<float>(intrinsics(0));  // left camera focal length
   principalU_ = static_cast<float>(intrinsics(2));
   principalV_ = static_cast<float>(intrinsics(3));
   baseline_ = T_lr.r().norm();
   fb_ = focalLength_ * baseline_;
   setIntrinsicsTensors(intrinsics);
   imgHalfWidth_ = static_cast<int>(0.5*imgWidth_);
   imgHalfHeight_ = static_cast<int>(0.5*imgHeight_);
 
   LOG(INFO) << "MVS network image width, height, cam params [pixel], and baseline [m] are " 
     << imgWidth_ << ", " << imgHeight_ << ", " << focalLength_ << ", " 
     << principalU_ << ", " << principalV_ << ", " << baseline_;
 
   // Load stereo & depth model
   try {
     stereoModel_ = torch::jit::load(modelDir + "/depth-model.pt", torch::kCUDA);
     // Adding torch::kCUDA gives cuda/cpu mix error (don't know why...)
     mvsModel_ = torch::jit::load(modelDir + "/mvs-model.pt");
 
     // Warm up forward pass for GPU.
     LOG(INFO) << "Warming up GPU ...";
     torch::Tensor leftTensor = torch::zeros({imgHeight_, imgWidth_}).to(torch::kU8).to(torch::kCUDA);
     torch::Tensor rightTensor = torch::zeros({imgHeight_, imgWidth_}).to(torch::kU8).to(torch::kCUDA);
     for(int i = 0; i < 10; i++) {
       LOG(INFO) << "GPU warmup for the stereo network iteration " << i;
       stereoModel_.forward({leftTensor, rightTensor});
     }
 
     torch::Tensor curImage = torch::zeros({imgHeight_, imgWidth_}).to(torch::kU8).to(torch::kCUDA); // [H,W]
     torch::Tensor srcImage = torch::zeros({numSrc_, imgHeight_, imgWidth_}).to(torch::kU8).to(torch::kCUDA); // [N,H,W]
     torch::Tensor curK0 = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA); // [4,4] for [0.5H,0.5W]
     torch::Tensor curInvK1 = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA); // [4,4] for [0.25H, 0.25W]
     torch::Tensor srcK1 = torch::eye({4}).to(torch::kFloat).repeat({numSrc_,1,1}).to(torch::kCUDA); // [N,4,4] for [0.25H, 0.25W]
     torch::Tensor curT_WC = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA); // [4,4]
     torch::Tensor srcT_WC = torch::eye({4}).to(torch::kFloat).repeat({numSrc_,1,1}).to(torch::kCUDA); // [N,4,4]
     torch::Tensor sparseDepth = torch::ones({imgHalfHeight_, imgHalfWidth_}).to(torch::kFloat).to(torch::kCUDA); // [0.5H,0.5W]
 
     for(int i = 0; i < 10; i++) {
       LOG(INFO) << "GPU warmup for the MVS network iteration " << i;
       mvsModel_.forward({curImage, srcImage, curK0, curInvK1, srcK1, curT_WC, srcT_WC, sparseDepth});
     }
 
     // Forward passes are asynchronous on CUDA. Do not signal readiness until
     // every warmup kernel has finished (and any CUDA error has surfaced).
     torch::cuda::synchronize();
     LOG(INFO) << "Done warming up GPU.";
 
   }
   catch (const c10::Error& e) {
     LOG(ERROR) << "Error loading or warming up the depth models: " << e.what();
     throw;
   }
 
   // Compute rectification map
   if (needRectify_) {
     parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)
       ->getRectifyMap(rectMapLeft0_, rectMapLeft1_);
     parameters.nCameraSystem.rectifyCameraGeometry(idRight_)
       ->getRectifyMap(rectMapRight0_, rectMapRight1_);
   }
 
   // Starting depth processing thread.
   shutdown_ = false;
   stereoThread_ = std::thread(&DepthFusionProcessor::processingStereo, this);
   motionThread_ = std::thread(&DepthFusionProcessor::processingMotion, this);
 }
 
 
 DepthFusionProcessor::~DepthFusionProcessor() {
   cameraMeasurementsQueue_.Shutdown();
   visualisationsQueue_.Shutdown();
   statesQueue_.Shutdown();
   stereoPredictionQueue_.Shutdown();
   shutdown_ = true;
   if(stereoThread_.joinable()) {
     stereoThread_.join();
   }
   if(motionThread_.joinable()) {
     motionThread_.join();
   }
 }
 
 
 void DepthFusionProcessor::setIntrinsicsTensors(const Eigen::VectorXd intrinsics) {
 
   curK0Tensor_ = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA);
   curK0Tensor_.index_put_({0,0}, static_cast<float>(0.5*intrinsics(0)));
   curK0Tensor_.index_put_({1,1}, static_cast<float>(0.5*intrinsics(1)));
   curK0Tensor_.index_put_({0,2}, static_cast<float>(0.5*intrinsics(2)));
   curK0Tensor_.index_put_({1,2}, static_cast<float>(0.5*intrinsics(3)));
 
   curInvK1Tensor_ = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA);
   curInvK1Tensor_.index_put_({0,0}, static_cast<float>(1.0f/(0.25*intrinsics(0))));
   curInvK1Tensor_.index_put_({1,1}, static_cast<float>(1.0f/(0.25*intrinsics(1))));
   curInvK1Tensor_.index_put_({0,2}, static_cast<float>(-intrinsics(2)/intrinsics(0)));
   curInvK1Tensor_.index_put_({1,2}, static_cast<float>(-intrinsics(3)/intrinsics(1)));
 
   torch::Tensor srcK1 = torch::eye({4}).to(torch::kFloat).to(torch::kCUDA);
   srcK1.index_put_({0,0}, static_cast<float>(0.25*intrinsics(0)));
   srcK1.index_put_({1,1}, static_cast<float>(0.25*intrinsics(1)));
   srcK1.index_put_({0,2}, static_cast<float>(0.25*intrinsics(2)));
   srcK1.index_put_({1,2}, static_cast<float>(0.25*intrinsics(3)));
   srcK1Tensor_ = srcK1.repeat({numSrc_,1,1});
 }
 
 
 void DepthFusionProcessor::processingStereo() {
   while(!isFinished()) {
     // StereoMeasurement sframe;
     std::map<size_t, std::vector<okvis::CameraMeasurement>> sframe;
     if (!cameraMeasurementsQueue_.Empty()) {
       if(blocking_){
         if (cameraMeasurementsQueue_.PopBlocking(&sframe)) {
           processStereoNetwork(sframe);
         }
       }
       else{
         if (cameraMeasurementsQueue_.PopNonBlocking(&sframe)) {
           processStereoNetwork(sframe);
         }
       }
     }
   }
 }
 
 
 void DepthFusionProcessor::processingMotion() {
   while(!isFinished()) {
     ViStates states;
     okvis::State latestTrajState;
     StereoPrediction oldestPrediction;
 
     while (!statesQueue_.Empty()) {
       if (!statesQueue_.PopBlocking(&states)) {
         return;  // shutdown did not provide a valid state
       }
       updateTrajectory(states);
     }
 
     if (!stereoPredictionQueue_.Empty()) {
       if (stereoPredictionQueue_.getCopyOfFront(&oldestPrediction)) {
 
         // Only this worker updates trajectory_; borrow its id set for this lookup.
         const std::set<StateId>& allIds = trajectory_.stateIds();
 
         if (!allIds.empty()) {
           StateId latestTrajId = *allIds.rbegin(); // get the most recently added id
           trajectory_.getState(latestTrajId, latestTrajState);
 
           // This makes processMVSnetwork executed after OKVIS2 optimization.
           if (oldestPrediction.timeStamp <= latestTrajState.timestamp) {
             if (!stereoPredictionQueue_.PopBlocking(&oldestPrediction)) {
               return;  // shutdown did not provide a valid prediction
             }
             processMVSnetwork(oldestPrediction);
           }
         }
       }
     }
 
   }
 }
 
 
 void DepthFusionProcessor::updateTrajectory(ViStates& states) {
   // Update Realtime Trajectory object
   std::set<okvis::StateId> affectedStateIds;
   trajectory_.update(states.trackingState, states.updatedStates, affectedStateIds);
 
   auto lastState = std::prev(states.updatedStates->end());
   if (states.trackingState.id == lastState->first) {
     landmarks_[lastState->second.timestamp] = *states.landmarks;
   }
 }
 
 
 void DepthFusionProcessor::processStereoNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
   okvis::TimerSwitchable tAll("DNN 1 Stereo depth");
 
   // Set the processing flag true.
   isProcessing_ = true;
 
   auto& frame0 = frames.at(idLeft_).front();
   auto& frame1 = frames.at(idRight_).front();

   convertStereoImageToGrayIfNeeded(frame0.measurement.image);
   convertStereoImageToGrayIfNeeded(frame1.measurement.image);
   applyStereoDnnEnhancementIfEnabled(frame0.measurement.image);
   applyStereoDnnEnhancementIfEnabled(frame1.measurement.image);

   // Stereo rectification
   if (needRectify_) {
     cv::Mat rectLeftImage, rectRightImage;
     cv::remap(frame0.measurement.image, rectLeftImage, rectMapLeft0_, rectMapLeft1_, cv::INTER_LINEAR);
     cv::remap(frame1.measurement.image, rectRightImage, rectMapRight0_, rectMapRight1_, cv::INTER_LINEAR);
     frame0.measurement.image = rectLeftImage;
     frame1.measurement.image = rectRightImage;
   }
 
   // cv::Mat to torch::Tensor for input images
   okvis::TimerSwitchable t0("DNN 1.1 cv::Mat to Tensor");
   auto leftTensor = cvMatToTensor(frame0.measurement.image).to(torch::kCUDA);
   auto rightTensor = cvMatToTensor(frame1.measurement.image).to(torch::kCUDA);
   // Please note that this t0 timing can be misleading if the GPU utilization is full from the MVS network.
   t0.stop();
 
   // Static stereo
   okvis::TimerSwitchable t1("DNN 1.2 Actual model inference");
   torch::Tensor outputStereo = stereoModel_.forward({leftTensor, rightTensor}).toTensor();
   t1.stop();
   torch::Tensor stereoDisparity = outputStereo.index({0,torch::indexing::Slice(),torch::indexing::Slice()}).squeeze();
   // The stereo network was over-confident during the training, so I had to multiply by 2.0.
   torch::Tensor stereoSigma = 2.0*outputStereo.index({1,torch::indexing::Slice(),torch::indexing::Slice()}).squeeze();
   torch::Tensor outputDepth = fb_ / stereoDisparity;
   torch::Tensor outputSigma = (outputDepth / stereoDisparity) * stereoSigma;
   outputSigma.index_put_({torch::isinf(outputSigma)}, 100.0f);
   outputSigma.index_put_({torch::isnan(outputSigma)}, 100.0f);
   // Set minimum sigma otherwise fused depth has nan value in a rare case.
   outputSigma.index_put_({outputSigma<0.01}, 0.01f);
   cv::Mat outputDepthMat = tensorToCvMatFloat(outputDepth.to(torch::kFloat).detach().cpu());
   cv::Mat outputSigmaMat = tensorToCvMatFloat(outputSigma.to(torch::kFloat).detach().cpu());
 
   StereoPrediction stereoOutput;
   stereoOutput.timeStamp = frame0.timeStamp;
   stereoOutput.measurement.leftImage = frame0.measurement.image;
   stereoOutput.measurement.rightImage = frame1.measurement.image;
   stereoOutput.measurement.depth = outputDepth;
   stereoOutput.measurement.sigma = outputSigma;
 
   // Queue for the MVS network.
   if (blocking_) {
     stereoPredictionQueue_.PushBlockingIfFull(stereoOutput, 10);
   }
   else {
     stereoPredictionQueue_.PushNonBlockingDroppingIfFull(stereoOutput, 10);
   }
 
   if(liveDepthCallback_) {
     liveDepthCallback_(frame0.timeStamp, outputDepthMat, outputSigmaMat);
   }
 
   // Set the processing flag true.
   isProcessing_ = false;
 
   tAll.stop();
 }
 
 
 void DepthFusionProcessor::processMVSnetwork(StereoPrediction& stereoPrediction){
   okvis::TimerSwitchable tAll("DNN 2 multi-view stereo");
   // Set the processing flag true.
   isProcessing_ = true;
 
   // The image delay should be compensated for the MVS network,
   // since the timestamp from stateUpdateCallback has been compensated.
   stereoPrediction.timeStamp = stereoPrediction.timeStamp - imageTimeDelay_;
 
   // For visualization
   cv::Mat sparseDepthVis;
   cv::cvtColor(stereoPrediction.measurement.leftImage, sparseDepthVis, cv::COLOR_GRAY2BGR);
 
   torch::Tensor mvsDepth, mvsSigma, fuseDepth, fuseSigma;
   torch::Tensor sparseDepth;
   bool isMvs = false;
   // Take the live (current) frame
   SrcFrameMeasurement liveFrame;
   liveFrame.timeStamp = stereoPrediction.timeStamp;
   liveFrame.measurement.image = stereoPrediction.measurement.leftImage;
   okvis::State liveState;
   if (trajectory_.getState(stereoPrediction.timeStamp, liveState)) {
     liveFrame.measurement.T_WC = liveState.T_WS * T_SC_;
   
     // Source frames are not full yet
     if (srcFrame_.size() < numSrc_) {
       // For the first frame
       if (srcFrame_.size() == 0) {
         srcFrame_.push_back(liveFrame);
       }
       else {
         // Compute baseline to decide to insert the live frame
         okvis::kinematics::Transformation T_Cs_Cl = 
           srcFrame_.back().measurement.T_WC.inverse() * liveFrame.measurement.T_WC;
 
         if (isSrcFrame(T_Cs_Cl)) {
           srcFrame_.push_back(liveFrame);
         }
       }
     }
     else { // When source frames are full
       // Make inputs -- images & sources poses
       okvis::TimerSwitchable t0("DNN 2.1 cv::Mat to Tensor");
       torch::Tensor liveTensor = cvMatToTensor(stereoPrediction.measurement.leftImage).to(torch::kCUDA);
       torch::Tensor srcTensors = torch::empty({numSrc_, imgHeight_, imgWidth_}).to(torch::kU8).to(torch::kCUDA);
       torch::Tensor T_WC_src = torch::eye({4}).to(torch::kFloat).repeat({numSrc_,1,1}).to(torch::kCUDA);
 
       for (int i = 0; i < srcFrame_.size(); i++) {
         torch::Tensor src_i = cvMatToTensor(srcFrame_[i].measurement.image).to(torch::kCUDA);
         srcTensors.index_put_({i,torch::indexing::Slice(),torch::indexing::Slice()}, src_i);
 
         // Update srcframe poses 
         okvis::State srcState_i;
         if (trajectory_.getState(srcFrame_[i].timeStamp, srcState_i)){
           srcFrame_[i].measurement.T_WC = srcState_i.T_WS * T_SC_;
         }
         torch::Tensor T_WC_i = eigenMatrixToTorch(srcFrame_[i].measurement.T_WC.T().cast<float>()).to(torch::kCUDA);
         T_WC_src.index_put_({i,torch::indexing::Slice(),torch::indexing::Slice()}, T_WC_i);
       }
 
       // live pose
       torch::Tensor T_WC_live = eigenMatrixToTorch(liveFrame.measurement.T_WC.T().cast<float>()).to(torch::kCUDA);
       t0.stop();
 
       // Sparse depth
       okvis::TimerSwitchable t1("DNN 2.2 make sparse depth");
       cv::Mat sparseMat = cv::Mat(imgHalfHeight_, imgHalfWidth_, CV_32F, 0.0f);
       okvis::kinematics::Transformation T_CW = liveFrame.measurement.T_WC.inverse();
       float half_f = 0.5*focalLength_;
       float half_cu = 0.5*(principalU_ + 0.5) - 0.5;
       float half_cv = 0.5*(principalV_ + 0.5) - 0.5;
       unsigned int numValidDepths = 0;
       // The MVS network was trained with the maximum 10m depth threhold. Use the same value here.
       const float maxDepth = 10.0;
       for (auto lm : landmarks_[liveFrame.timeStamp]) {
         Eigen::Vector4d p_W(lm.point(0,0), lm.point(1,0), lm.point(2,0), lm.point(3,0));
         Eigen::Vector4d p_C = T_CW * p_W;
         float Z_lm = p_C(2);
         if (Z_lm <= 0.4 || Z_lm > maxDepth) {
           continue;
         }
 
         int u_lm, v_lm;
         u_lm = static_cast<int>(half_f * (p_C(0)/Z_lm) + half_cu);
         v_lm = static_cast<int>(half_f * (p_C(1)/Z_lm) + half_cv);
 
         if (u_lm>0 && u_lm<imgHalfWidth_-1 && v_lm>0 && v_lm<imgHalfHeight_-1) {
           sparseMat.at<float>(v_lm, u_lm) = Z_lm;
           numValidDepths ++;
 
           float visZ = Z_lm;
           if (visZ > maxDepth) {
             visZ = maxDepth;
           }
           cv::circle(sparseDepthVis, cv::Point(2*u_lm,2*v_lm), 3, 
             cv::Scalar(0,static_cast<u_char>(visZ/(maxDepth)*255.0),0), -1);
         }
       }
       landmarks_.erase(liveFrame.timeStamp);
 
       auto options = torch::TensorOptions().dtype(torch::kFloat32);
       sparseDepth = torch::from_blob(
         sparseMat.data, {sparseMat.rows, sparseMat.cols}, options).to(torch::kCUDA);
       t1.stop();
 
       // Inference
       okvis::TimerSwitchable t2("DNN 2.3 Actual model inference");
       torch::Tensor modelOutput = mvsModel_.forward({liveTensor, srcTensors, curK0Tensor_,
         curInvK1Tensor_, srcK1Tensor_, T_WC_live, T_WC_src, sparseDepth}).toTensor();
       mvsDepth = modelOutput.index({0,torch::indexing::Slice(),torch::indexing::Slice()}).squeeze();
       // The MVS network was too over-confident during the training, so I had to multiply by 4.0.
       mvsSigma = 4.0*modelOutput.index({1,torch::indexing::Slice(),torch::indexing::Slice()}).squeeze();
       mvsSigma.index_put_({torch::isinf(mvsSigma)}, 100.0f);
       mvsSigma.index_put_({torch::isnan(mvsSigma)}, 100.0f);
       isMvs = true;
       t2.stop();
 
       // Fusion with stereo network
       okvis::TimerSwitchable t3("DNN 2.4 Depth fusion");
       torch::Tensor ivar_a = 1.0/stereoPrediction.measurement.sigma.square();
       torch::Tensor ivar_b = 1.0/mvsSigma.square();
       torch::Tensor var_fuse = 1.0/(ivar_a + ivar_b);
       fuseDepth = var_fuse * (ivar_a*stereoPrediction.measurement.depth + ivar_b*mvsDepth);
       fuseSigma = var_fuse.sqrt();
       cv::Mat outputDepthMat = tensorToCvMatFloat(fuseDepth.to(torch::kFloat).detach().cpu());
       cv::Mat outputSigmaMat = tensorToCvMatFloat(fuseSigma.to(torch::kFloat).detach().cpu());
       // Please note that this t3 timing can be misleading due to asynchronism of GPU.
       // For more proper timing, cudaDeviceSynchronize() should be placed before t2.stop(); with possible slowdown.
       t3.stop();
 
       // For the callback
       std::map<size_t, std::vector<okvis::CameraMeasurement>> frames;
       okvis::CameraMeasurement camMeasurement;
       // The image delay should be reversed, 
       // since the callback function is responsible for compensating image delay.
       camMeasurement.timeStamp = stereoPrediction.timeStamp + imageTimeDelay_;
       camMeasurement.sensorId = idLeft_;
       camMeasurement.measurement.depthImage = outputDepthMat.clone();
       camMeasurement.measurement.sigmaImage = outputSigmaMat.clone();
       frames[idLeft_] = {camMeasurement};
 
       // Callback
       if(imageCallback_) {
         imageCallback_(frames);
       }
 
       // Update source frames
       okvis::kinematics::Transformation T_Cs_Cl = 
         srcFrame_.back().measurement.T_WC.inverse() * liveFrame.measurement.T_WC;
 
       if (isSrcFrame(T_Cs_Cl)) {
         srcFrame_.push_back(liveFrame);
         srcFrame_.pop_front();
       }
     }
   }
 
   // When MVS is not available
   if (!isMvs) {
     cv::Mat outputDepthMat = tensorToCvMatFloat(stereoPrediction.measurement.depth.to(torch::kFloat).detach().cpu());
     cv::Mat outputSigmaMat = tensorToCvMatFloat(stereoPrediction.measurement.sigma.to(torch::kFloat).detach().cpu());
     std::map<size_t, std::vector<okvis::CameraMeasurement>> frames;
     okvis::CameraMeasurement camMeasurement;
     // The image delay should be reversed, 
     // since the callback function is responsible for compensating image delay.
     camMeasurement.timeStamp = stereoPrediction.timeStamp + imageTimeDelay_;
     camMeasurement.sensorId = idLeft_;
     camMeasurement.measurement.depthImage = outputDepthMat.clone();
     camMeasurement.measurement.sigmaImage = outputSigmaMat.clone();
     frames[idLeft_] = {camMeasurement};
 
     if(imageCallback_) {
       imageCallback_(frames);
     }
   }
 
   // Visualization
   VisualizationData visData;
   visData.frame.timeStamp = stereoPrediction.timeStamp;
   visData.frame.measurement.leftImage = stereoPrediction.measurement.leftImage;
   visData.frame.measurement.rightImage = stereoPrediction.measurement.rightImage;
   if (srcFrame_.size() == numSrc_) {
     for (size_t i = 0; i < numSrc_; i ++) {
       visData.srcframes.push_back(srcFrame_[i].measurement.image);
     }
   }
 
   visData.sparseDepthImage = sparseDepthVis;
   visData.stereoDepthImage = tensorToiDepthVis(stereoPrediction.measurement.depth);
   visData.stereoSigmaImage = tensorToiSigmaVis(stereoPrediction.measurement.depth, stereoPrediction.measurement.sigma);
 
   if (isMvs) {
     visData.mvsDepthImage = tensorToiDepthVis(mvsDepth);
     visData.mvsSigmaImage = tensorToiSigmaVis(mvsDepth, mvsSigma);
     visData.fuseDepthImage = tensorToiDepthVis(fuseDepth);
     visData.fuseSigmaImage = tensorToiSigmaVis(fuseDepth, fuseSigma);
   }
   else {
     visData.mvsDepthImage = cv::Mat::zeros(imgHeight_, imgWidth_, CV_8UC3);
     visData.mvsSigmaImage = cv::Mat::zeros(imgHeight_, imgWidth_, CV_8UC3);
     visData.fuseDepthImage = visData.stereoDepthImage;
     visData.fuseSigmaImage = visData.stereoSigmaImage;
   }
 
   visualisationsQueue_.PushNonBlockingDroppingIfFull(visData, 1);
 
   // Set the processing flag false.
   isProcessing_ = false;
 
   tAll.stop();
 }
 
 
 bool DepthFusionProcessor::isSrcFrame(const okvis::kinematics::Transformation& T) {
   float motionBaseline = T.r().norm(); // [m]
   float motionAngle = std::abs(std::acos(0.5*(T.C().trace()-1.0))); // [rad]
   if (srcSelectCnt_ % 2 == 0) {
     if (motionBaseline > 0.03 || motionAngle > 0.05) {
       srcSelectCnt_ ++;
       return true;
     }
   }
   else {
     if (motionBaseline > 0.10 || motionAngle > 0.35) {
       srcSelectCnt_ ++;
       return true;
     }
   }
   
   return false;
 }
 
 
 void DepthFusionProcessor::display(std::map<std::string, cv::Mat> &images) {
   if(visualisationsQueue_.Empty()) {
     return;
   }
   VisualizationData vis_data;
   if(visualisationsQueue_.PopNonBlocking(&vis_data)) {
     images["leftImage"] = vis_data.frame.measurement.leftImage;
     images["rightImage"] = vis_data.frame.measurement.rightImage;
     images["stereoDepth"] = vis_data.stereoDepthImage;
     images["stereoSigma"] = vis_data.stereoSigmaImage;
     images["mvsDepth"] = vis_data.mvsDepthImage;
     images["mvsSigma"] = vis_data.mvsSigmaImage;
     images["fuseDepth"] =  vis_data.fuseDepthImage;
     images["fuseSigma"] = vis_data.fuseSigmaImage;
     images["sparseDepth"] = vis_data.sparseDepthImage;
 
     if(vis_data.srcframes.size() > numSrc_-1) {
       for(size_t i = 0; i < vis_data.srcframes.size(); i++) {
         std::string srcName = "srcImage" + std::to_string(i);
         images[srcName] = vis_data.srcframes[i];
       }
     }
   }
 }
 
 
 bool DepthFusionProcessor::stateUpdateCallback(const okvis::State &latestState, const okvis::TrackingState &latestTrackingState,
                                      std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updatedStates,
                                      std::shared_ptr<const okvis::MapPointVector> landmarks) {
 
   ViStates states;
   states.trackingState = latestTrackingState;
   states.updatedStates = updatedStates;
   std::shared_ptr<okvis::MapPointVector> liveLandmarks;
   liveLandmarks.reset(new okvis::MapPointVector());
   for(const auto & lm : *landmarks) {
     if(lm.stateId == latestTrackingState.id.value()) {
       liveLandmarks->push_back(lm);
     }
   }
   states.landmarks = liveLandmarks;
 
   if(blocking_){
     statesQueue_.PushBlockingIfFull(states, 4);
     return true;
   }
   else{
     const int queue_size = 10;
     if(statesQueue_.PushNonBlockingDroppingIfFull(states, queue_size)) {
       LOG(WARNING) <<  "Updated states drop";
       return false;
     }
     else{
       return true;
     }
   }
 }
 
 bool DepthFusionProcessor::finishedProcessing() {
   size_t numQueues = cameraMeasurementsQueue_.Size() + statesQueue_.Size();
   if (numQueues == 0 && !isProcessing_) {
     while(!stereoPredictionQueue_.Empty()) {
       // Clear the queue, otherwise the processor could work after the final BA.
       StereoPrediction oldestPrediction;
       if (!stereoPredictionQueue_.PopBlocking(&oldestPrediction)) {
         return false;  // cancellation is not a completed drain
       }
     }
     return true;
   }
   else {
     return false;
   }
 }
 
 } // namespace srl
