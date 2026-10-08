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
 * @file ViSlamBackend.cpp
 * @brief Source file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <okvis/ViSlamBackend.hpp>
#include <okvis/PseudoInverse.hpp>
#include <okvis/Component.hpp>
#include <okvis/timing/Timer.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {


static const double minDeltaT = 2.0; // [sec]
static int numRealtimePoseGraphFrames = 12;
static const int numPoseGraphFrames = 12;

int ViSlamBackend::addCamera(const CameraParameters &cameraParameters)
{
  fullGraph_.addCamera(cameraParameters);
  return realtimeGraph_.addCamera(cameraParameters);
}

int ViSlamBackend::addImu(const ImuParameters &imuParameters)
{
  fullGraph_.addImu(imuParameters);
  return realtimeGraph_.addImu(imuParameters);
}

int ViSlamBackend::addGps(const GpsParameters &gpsParameters)
{
  fullGraph_.addGps(gpsParameters);
  return realtimeGraph_.addGps(gpsParameters);
}

bool ViSlamBackend::addGpsMeasurementsOnAllGraphs(GpsMeasurementDeque& inputgpsMeasurementDeque, ImuMeasurementDeque& imuMeasurementDeque){
  if(realtimeGraph_.gpsParametersVec_.empty()) {
    return false;
  }
  //  Check for valid GPS Measurements
  GpsMeasurementDeque gpsMeasurementDeque;
  if(realtimeGraph_.gpsParametersVec_.back().robustGpsInit){
    [[maybe_unused]] int valid = realtimeGraph_.checkValidGpsMeasurements(inputgpsMeasurementDeque, gpsMeasurementDeque);
  }
  else {
    gpsMeasurementDeque = inputgpsMeasurementDeque;
  }

  // (1) Check if GPS Measurements have to be added
  if(gpsMeasurementDeque.size() > 0){

    // (2) Check realtimeGraph_, if GPS needs to be re-initialized  
    bool needsGpsReInit = realtimeGraph_.needsGpsReInit(); // check if GPS extrinsics need to be re-initialised after long GPS dropout
    if(needsGpsReInit) {
      realtimeGraph_.reInitGpsExtrinsics();
    }

    // (3) Synchronize GPS Status
    fullGraph_.setGpsStatus(realtimeGraph_.getGpsStatus());

    // (4) Add Measurements to both graphs (if possible)
    if(!isLoopClosing_ && !isLoopClosureAvailable_){ // accessible => add measurements to all graphs
      // Add GPS Measurements to real time graph
      realtimeGraph_.addGpsMeasurements(gpsMeasurementDeque,imuMeasurementDeque,nullptr);
      fullGraph_.addGpsMeasurements(gpsMeasurementDeque, imuMeasurementDeque,nullptr);
    }
    else{ // not accessible => save state ids for measurements and buffer gps measurements

      std::deque<StateId> sids; // obtain state ids to add measurements to while buffering them for fullGraph_
      realtimeGraph_.addGpsMeasurements(gpsMeasurementDeque,imuMeasurementDeque,&sids);

      // GPS MEASUREMENTS BUFFERING
      for(size_t i = 0; i < gpsMeasurementDeque.size(); i++){
        addGpsBacklog_.push_back(AddGpsBacklog{sids.at(i), gpsMeasurementDeque.at(i), imuMeasurementDeque, needsGpsReInit});
      }
    }

    // (5) Initialization Considerations (successfull first rough initialization?)
    kinematics::Transformation T_GW_init;
    if(realtimeGraph_.initializationStrategy(T_GW_init)){
      realtimeGraph_.addGpsInitFactors();
      fullGraph_.addGpsInitFactors();
      realtimeGraph_.setGpsExtrinsics(T_GW_init);
      fullGraph_.setGpsExtrinsics(T_GW_init);
    }

    // (6) Try alignment if available / required
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      [[maybe_unused]] bool requires_gps_alignment = tryGpsAlignment();
    }

    // (7) compute observability based on internal measurements
    if(!gpsObservability_){
      gpsObservability_ = realtimeGraph_.isGpsObservable();
    }
    return true; // measurements added
  }
  else {
    return false; // no measurements could have been added to the graph
  }
}

bool ViSlamBackend::tryGpsAlignment(){
  Eigen::Vector3d posAlignVec;
  StateId gpsDropId;
  StateId alignId;
  okvis::kinematics::Transformation T_GW_new;

  bool needInitialAlign = realtimeGraph_.needsInitialGpsAlignment();
  if(needInitialAlign){
    fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
    addGpsAlignmentFrame(StateId(1));
    realtimeGraph_.resetInitialGpsAlignment();
    fullGraph_.resetInitialGpsAlignment();
    return true;
  }

  // Check for full alignments
  [[maybe_unused]] bool _ = fullGraph_.needsFullGpsAlignment(gpsDropId, alignId, T_GW_new);
  bool needFullAlign = realtimeGraph_.needsFullGpsAlignment(gpsDropId, alignId, T_GW_new);

  if(needFullAlign){

    attemptFullGpsAlignment(gpsDropId,alignId, T_GW_new);

    // Do brief realtime optimisation and synchronisation
    //TimerSwitchable gpsLoopOptimizeTimer("99 Initial GPS Loop Optimizer");
    std::vector<StateId> updatedStatesRealtime;
    optimiseRealtimeGraph(50, updatedStatesRealtime);
    //gpsLoopOptimizeTimer.stop();

    addGpsAlignmentFrame(gpsDropId);
    realtimeGraph_.resetFullGpsAlignment();
    fullGraph_.resetFullGpsAlignment();
    return true;
  }
  else{
    // check for position alignments
    bool needPosAlign = realtimeGraph_.needsPosGpsAlignment(gpsDropId, alignId, posAlignVec);
    if(needPosAlign){
      attemptPosGpsAlignment(gpsDropId,alignId, posAlignVec);
      addGpsAlignmentFrame(gpsDropId);
      realtimeGraph_.resetPosGpsAlignment();
      fullGraph_.resetPosGpsAlignment();
      return true;
    }
    else{
      return false;
    }
  }
}

bool ViSlamBackend::addStates(MultiFramePtr multiFrame, const ImuMeasurementDeque &imuMeasurements,
                              bool asKeyframe)
{
  AuxiliaryState auxiliaryState;
  auxiliaryState.isImuFrame = true;
  auxiliaryState.isKeyframe = asKeyframe;
  if(multiFrames_.empty()) {
    // initialise
    OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "not allowed")
    const StateId id = realtimeGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                                          multiFrame->cameraSystem());
    multiFrame->setId(id.value());
    fullGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                              multiFrame->cameraSystem());
    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; // for internal book-keeping
    imuFrames_.insert(id);
    return (id.value()==1 && id.isInitialised());
  } else {
    const StateId id = realtimeGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements,
                                                         asKeyframe);
    // obtain potentially optimised extrinsics
    for (size_t i = 0; i < multiFrame->numFrames(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        const kinematics::Transformation T_SCi = extrinsics(id, i);
        multiFrame->setExtrinsics(i, T_SCi);
      }
    }

    multiFrame->setId(id.value());

    if(isLoopClosing_ || isLoopClosureAvailable_) {
      addStatesBacklog_.push_back(AddStatesBacklog{multiFrame->timestamp(), id, imuMeasurements});
      touchedStates_.insert(id);
    } else {
      // safe to add to the full graph
      fullGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements, asKeyframe);
    }

    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; // for internal book-keeping
    imuFrames_.insert(id);
    return id.isInitialised();
  }
}

void ViSlamBackend::printStates(StateId stateId, std::ostream &buffer) const
{
  const ViGraph::State & state = realtimeGraph_.states_.at(stateId);
  if(state.isKeyframe) {
    buffer << "KF ";
  }
  buffer << "pose: ";
  if (state.pose->fixed()) buffer << "(";
  buffer << "id=" << stateId.value() << ":";
  if (state.pose->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "speedAndBias: ";
  if (state.speedAndBias->fixed()) buffer << "(";
  buffer << "id=" << state.speedAndBias->id() << ":";
  if (state.speedAndBias->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "extrinsics: ";
  for (size_t i = 0; i < state.extrinsics.size(); ++i) {
    uint64_t id = state.extrinsics.at(i)->id();
    if (state.extrinsics.at(i)->fixed()) buffer << "(";
    buffer << "id=" << id << ":";
    if (state.extrinsics.at(i)->fixed()) buffer << ")";
    if (i+1 < state.extrinsics.size()) buffer << ", ";
  }
  buffer << std::endl;
}

bool ViSlamBackend::getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const
{
  return realtimeGraph_.getLandmark(landmarkId, mapPoint);
}


size_t ViSlamBackend::getLandmarks(MapPoints & landmarks) const
{
  return realtimeGraph_.getLandmarks(landmarks);
}

double ViSlamBackend::trackingQuality(StateId id) const
{
  const MultiFramePtr frame = multiFrame(id);
  const size_t numFrames = frame->numFrames();
  std::set<LandmarkId> landmarks;
  std::vector<cv::Mat> matchesImg(numFrames);

  // remember matched points
  int intersectionCount = 0;
  int unionCount = 0;
  int matchedPoints = 0;
  for (size_t im = 0; im < numFrames; ++im) {
    const int rows = frame->image(im).rows/10;
    const int cols = frame->image(im).cols/10;
    const double radius = double(std::min(rows,cols))*kptradius_;
    matchesImg.at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
    const size_t num = frame->numKeypoints(im);
    cv::KeyPoint keypoint;
    for (size_t k = 0; k < num; ++k) {
      frame->getCvKeypoint(im, k, keypoint);
      uint64_t lmId = frame->landmarkId(im, k);
      if (lmId != 0 && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {
        // make sure these are observed elsewhere
        for(const auto & obs : realtimeGraph_.landmarks_.at(LandmarkId(lmId)).observations) {
          if(obs.first.frameId != id.value()) {
            matchedPoints++;
            cv::circle(matchesImg.at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                       cv::FILLED);
            break;
          }
        }
      }
    }
    // one point per image does not count.
    const int pointArea = int(radius*radius*M_PI);
    intersectionCount += std::max(0,cv::countNonZero(matchesImg.at(im)) - pointArea);
    unionCount += rows*cols - pointArea;
  }
  return matchedPoints < 8 ? 0.0 : double(intersectionCount)/double(unionCount);
}

bool ViSlamBackend::setKeyframe(StateId id, bool isKeyframe) {
  auxiliaryStates_.at(id).isKeyframe = isKeyframe;
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    fullGraph_.setKeyframe(id, isKeyframe);
  }
  return realtimeGraph_.setKeyframe(id, isKeyframe);
}

bool ViSlamBackend::addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &landmark,
                                bool isInitialised)
{
  bool success = realtimeGraph_.addLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    success &= fullGraph_.addLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}

LandmarkId ViSlamBackend::addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  const LandmarkId lmId = realtimeGraph_.addLandmark(homogeneousPoint, initialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(lmId);
  } else {
    fullGraph_.addLandmark(lmId, homogeneousPoint, initialised);
  }
  return lmId;
}

bool ViSlamBackend::setLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark,
                                bool isInitialised) {
  bool success = realtimeGraph_.setLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    fullGraph_.setLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}
bool ViSlamBackend::setLandmarkClassification(LandmarkId landmarkId, int classification) {
  if(realtimeGraph_.landmarks_.count(landmarkId)==0) {
    return false;
  }
  realtimeGraph_.landmarks_.at(landmarkId).classification = classification;
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    fullGraph_.landmarks_.at(landmarkId).classification = classification;
  }
  return true;
}

bool ViSlamBackend::setObservationInformation(
    StateId stateId, size_t camIdx, size_t keypointIdx, const Eigen::Matrix2d & information) {
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.observations_.at(kid).errorTerm->setInformation(information);
  }
  realtimeGraph_.observations_.at(kid).errorTerm->setInformation(information);
  return true;
}

bool ViSlamBackend::scaleObservationInformation(
    StateId stateId, size_t camIdx, size_t keypointIdx, double information_scale) {
  if (!std::isfinite(information_scale) || information_scale <= 0.0) {
    return false;
  }
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  const auto it = realtimeGraph_.observations_.find(kid);
  if (it == realtimeGraph_.observations_.end()) {
    return false;
  }
  // Reuse the lookup above. The fallback path calls this for every observed
  // keypoint; going through setObservationInformation repeats the same lookup
  // (twice during a full-graph solve).
  const Eigen::Matrix2d information = information_scale * it->second.errorTerm->information();
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(it->second.landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.observations_.at(kid).errorTerm->setInformation(information);
  }
  it->second.errorTerm->setInformation(information);
  return true;
}

size_t ViSlamBackend::setFrameReprojectionInformationScale(
    StateId stateId, double informationScale) {
  if(!std::isfinite(informationScale) || informationScale <= 0.0) {
    return 0;
  }
  const auto frame = realtimeGraph_.states_.find(stateId);
  const auto auxiliary = auxiliaryStates_.find(stateId);
  if(frame == realtimeGraph_.states_.end() || auxiliary == auxiliaryStates_.end()) {
    return 0;
  }
  const double ratio = informationScale / auxiliary->second.reprojectionInformationScale;
  if(!std::isfinite(ratio) || ratio <= 0.0 || ratio == 1.0) {
    return 0;
  }
  // Remember even a frame with no current observations: triangulation can add
  // its first observation during a later frame's motion-stereo matching.
  auxiliary->second.reprojectionInformationScale = informationScale;
  size_t updated = 0;
  for(const auto& observation : frame->second.observations) {
    const auto& key = observation.first;
    if(scaleObservationInformation(stateId, key.cameraIndex, key.keypointIndex, ratio)) {
      ++updated;
    }
  }
  return updated;
}

int ViSlamBackend::scaleVisualTwoPoseGraphErrors(
    ViGraphEstimator& graph, double weightScale) {
  int touched = 0;
  // Both graph insertion/removal paths maintain the same link at both endpoints.
  // Visit only the lower-ID endpoint, preserving the previous traversal order
  // without allocating a visited set for every frame and graph.
  for(auto& kv : graph.states_) {
    auto& state = kv.second;
    for(auto& link : state.twoPoseLinks) {
      if(link.first < kv.first) {
        continue;
      }
      if(!link.second.errorTerm) {
        continue;
      }
      if(isSuppressedLoopPathTwoPoseConstraint(link.second.state0, link.second.state1)) {
        link.second.errorTerm->setWeightScale(0.0);
        continue;
      }
      link.second.errorTerm->setWeightScale(weightScale);
      ++touched;
    }
    for(auto& link : state.twoPoseConstLinks) {
      if(link.first < kv.first) {
        continue;
      }
      if(!link.second.errorTerm) {
        continue;
      }
      if(isSuppressedLoopPathTwoPoseConstraint(link.second.state0, link.second.state1)) {
        link.second.errorTerm->setWeightScale(0.0);
        continue;
      }
      link.second.errorTerm->setWeightScale(weightScale);
      ++touched;
    }
  }
  return touched;
}

int ViSlamBackend::scaleAllTwoPoseGraphErrors(double weightScale) {
  // This is the visual-fallback scale only. Fallback-loop-suppressed visual
  // relative-pose constraints stay disabled and are not restored by reset-to-1.0.
  requestedVisualTwoPoseWeightScale_ = weightScale;
  int touched = scaleVisualTwoPoseGraphErrors(realtimeGraph_, weightScale);
  // A flag check alone misses the interval before a newly launched worker sets
  // isLoopClosing_. The same mutex is held across every full-graph solve.
  std::unique_lock<std::mutex> weightLock(fullGraphTwoPoseWeightMutex_, std::try_to_lock);
  if(weightLock.owns_lock() && !isLoopClosing_ && !isLoopClosureAvailable_) {
    touched += scaleVisualTwoPoseGraphErrors(fullGraph_, weightScale);
  }
  // Otherwise synchronization applies the latest request after merging graphs.
  return touched;
}

bool ViSlamBackend::removeObservation(StateId stateId, size_t camIdx,
                                      size_t keypointIdx)
{
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.removeObservation(kid);
  }
  bool success = realtimeGraph_.removeObservation(kid);
  multiFrames_.at(stateId)->setLandmarkId(camIdx, keypointIdx, 0);
  return success;
}

bool ViSlamBackend::convertToPoseGraphMst(const std::set<StateId> & framesToConvert,
                                          const std::set<StateId> & framesToConsider,
                                          std::set<StateId> & affectedFrames) {
  // remember landmarks in frames (transformed to sensor frame)
  for(auto pose : framesToConvert) {
    ViGraph::State& state = realtimeGraph_.states_.at(pose);
    kinematics::Transformation T_SW = state.pose->estimate().inverse();
    MultiFramePtr mFrame = multiFrames_.at(pose);
    for(size_t i=0; i<mFrame->numFrames(); ++i) {
      for(size_t k=0; k<mFrame->numKeypoints(i); ++k) {
        uint64_t lmId = mFrame->landmarkId(i,k);
        if(lmId && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {
          Eigen::Vector4d landmark = realtimeGraph_.landmark(LandmarkId(lmId));
          mFrame->setLandmark(i, k, T_SW*landmark,
                              realtimeGraph_.isLandmarkInitialised(LandmarkId(lmId)));
        }
      }
    }
  }

  std::vector<ViGraphEstimator::PoseGraphEdge> poseGraphEdges;
  std::vector<std::pair<StateId,StateId>> removedTwoPoseErrors;
  std::vector<KeypointIdentifier> removedObservations;
  realtimeGraph_.convertToPoseGraphMst(
        framesToConvert, framesToConsider, &poseGraphEdges, &removedTwoPoseErrors,
        &removedObservations);

  for(auto iter = poseGraphEdges.begin(); iter != poseGraphEdges.end();) {
    if(isSuppressedLoopPathTwoPoseConstraint(iter->referenceId, iter->otherId)) {
      removeTwoPoseLinkIfPresent(realtimeGraph_, iter->referenceId, iter->otherId);
      iter = poseGraphEdges.erase(iter);
    } else {
      ++iter;
    }
  }

  // remember affected frames
  for (auto addedEdge : poseGraphEdges) {
    affectedFrames.insert(addedEdge.otherId);
    affectedFrames.insert(addedEdge.referenceId);
  }
  for (auto removedEdge : removedTwoPoseErrors) {
    affectedFrames.insert(removedEdge.first);
    affectedFrames.insert(removedEdge.second);
  }

  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & poseGraphEdge : poseGraphEdges) {
      touchedStates_.insert(poseGraphEdge.referenceId);
      touchedStates_.insert(poseGraphEdge.otherId);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      touchedStates_.insert(removedTwoPoseError.first);
      touchedStates_.insert(removedTwoPoseError.second);
    }
    for(auto obs : removedObservations) {
      touchedStates_.insert(StateId(obs.frameId));
      if(fullGraph_.observations_.count(obs)) {
        LandmarkId lmId = fullGraph_.observations_.at(obs).landmarkId;
        if(lmId.isInitialised()) {
          touchedLandmarks_.insert(fullGraph_.observations_.at(obs).landmarkId); /// \todo hack, fix
        }
      }
    }
  } else {
    // replicate fullGraph_
    for(auto obs : removedObservations) {
      fullGraph_.removeObservation(obs);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      removeTwoPoseConstLinkIfPresent(
          fullGraph_, removedTwoPoseError.first, removedTwoPoseError.second);
    }
    for(const auto & poseGraphEdge : poseGraphEdges) {
      if(isSuppressedLoopPathTwoPoseConstraint(
             poseGraphEdge.referenceId, poseGraphEdge.otherId)) {
        continue;
      }
      fullGraph_.addExternalTwoPoseLink(
            poseGraphEdge.poseGraphError->cloneTwoPoseGraphErrorConst(),
            poseGraphEdge.referenceId, poseGraphEdge.otherId);

    }
  }

  return true;
}

int ViSlamBackend::expandKeyframe(StateId keyframe)
{
  OKVIS_ASSERT_TRUE(Exception, imuFrames_.count(keyframe) == 0, "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, keyFrames_.count(keyframe) || loopClosureFrames_.count(keyframe),
                    "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(keyframe).twoPoseLinks.size()>0,
                    "must be frontier frame")
  int ctr = 0;
  std::set<LandmarkId> lms;
  std::set<StateId> cnncts;
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(keyframe, &lms, &cnncts, &allObservations);

  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(cnncts.begin(), cnncts.end());
    touchedLandmarks_.insert(lms.begin(), lms.end());
  } else {
    fullGraph_.removeTwoPoseConstLinks(keyframe);
    for(auto lm : lms) {
      const auto & landmark = realtimeGraph_.landmarks_.at(lm);
      if(!fullGraph_.landmarkExists(lm)) {
        fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
        fullGraph_.setLandmarkQuality(lm, landmark.quality);
      }
    }
    for(const auto & obs : allObservations) {
      const bool useCauchy = obs.lossFunction != nullptr;
      LandmarkId landmarkId(obs.hPoint->id());
      fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                             obs.keypointIdentifier, useCauchy);
    }
  }

  for(auto cnnct : cnncts) {
    if(loopClosureFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      keyFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
    if(keyFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      loopClosureFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
  }

  return ctr;
}

void ViSlamBackend::eliminateImuFrames(size_t numImuFrames, std::set<StateId> & /*affectedFrames*/)
{
  std::set<StateId> imuFrames = imuFrames_; // copy so we can later delete
  for(auto id : imuFrames) {
    if(imuFrames_.size() <= numImuFrames) {
      break; // done -- reduced the IMU frames
    }
    if(realtimeGraph_.states_.at(id).isKeyframe) {
      // move to the keyframes that we keep
      imuFrames_.erase(id);
      keyFrames_.insert(id);
      components_.at(currentComponentIdx_).poseIds.insert(id);
      if(loopClosureFrames_.count(id)) {
        loopClosureFrames_.erase(id); // make sure the two sets are not intersecting
      }
      auxiliaryStates_.at(id).isImuFrame = false;
    } else {
      StateId refId = mostOverlappedStateId(id, false);
      auto observations = realtimeGraph_.states_.at(id).observations;
      realtimeGraph_.removeAllObservations(id);
      realtimeGraph_.eliminateStateByImuMerge(id, refId);

      // also remove from loop closure frames
      if(loopClosureFrames_.count(id)) {
        OKVIS_THROW(Exception, "bug")
      }

      // also manage the full graph, if possible
      if(isLoopClosing_ || isLoopClosureAvailable_) {
        for(const auto & obs : observations) {
          touchedLandmarks_.insert(obs.second.landmarkId);
        }
        eliminateStates_[id] = refId;
      } else {
        fullGraph_.removeAllObservations(id); /// \todo make more efficient (copy over)
        fullGraph_.eliminateStateByImuMerge(id, refId); /// \todo make more efficient (copy over)
      }
      imuFrames_.erase(id);
      multiFrames_.erase(id);
      auxiliaryStates_.erase(id);
    }
  }
  // Realtime optimisation may run concurrently with a background full-graph
  // optimisation. Ceres Problem mutation is not thread-safe, so this thread must
  // only touch the realtime graph; the full graph reapplies these persistent
  // constraints inside optimiseFullGraph()/synchroniseRealtimeAndFullGraph().
  reapplyPersistentLoopPathRelativePoseConstraints(true, false);
}

std::pair<uint64_t, uint64_t> ViSlamBackend::orderedStatePairKey(
    StateId poseId0, StateId poseId1) const
{
  return {std::min(poseId0.value(), poseId1.value()),
          std::max(poseId0.value(), poseId1.value())};
}

void ViSlamBackend::setFallbackLoopGarlileoBetweenPathBounds(StateId pose_i, StateId pose_j)
{
  fallbackLoopGarlileoPathIdLo_ = std::min(pose_i.value(), pose_j.value());
  fallbackLoopGarlileoPathIdHi_ = std::max(pose_i.value(), pose_j.value());
  fallbackLoopGarlileoPathBoundsActive_ = true;
  LOG(INFO) << "[FallbackLC] GaRLILEO between resume-after-path: active segment id ["
            << fallbackLoopGarlileoPathIdLo_ << ", " << fallbackLoopGarlileoPathIdHi_ << "]";
}

void ViSlamBackend::clearFallbackLoopGarlileoBetweenPathBounds()
{
  if(fallbackLoopGarlileoPathBoundsActive_) {
    LOG(INFO) << "[FallbackLC] GaRLILEO between resume-after-path: cleared segment bounds";
  }
  fallbackLoopGarlileoPathBoundsActive_ = false;
  fallbackLoopGarlileoPathIdLo_ = 0;
  fallbackLoopGarlileoPathIdHi_ = 0;
}

bool ViSlamBackend::suppressGarlileoBetweenForFallbackLoopPathPair(StateId poseId0,
                                                                 StateId poseId1) const
{
  if(!fallbackLoopGarlileoPathBoundsActive_) {
    return false;
  }
  const uint64_t a = poseId0.value();
  const uint64_t b = poseId1.value();
  const uint64_t lo = fallbackLoopGarlileoPathIdLo_;
  const uint64_t hi = fallbackLoopGarlileoPathIdHi_;
  auto inClosed = [&](uint64_t v) { return v >= lo && v <= hi; };
  const bool bothStrictlyAfter = (a > hi && b > hi);
  if(bothStrictlyAfter) {
    return false;
  }
  const bool bothStrictlyBefore = (a < lo && b < lo);
  if(bothStrictlyBefore) {
    return false;
  }
  return inClosed(a) || inClosed(b);
}

bool ViSlamBackend::isProtectedLoopPathRelativePoseConstraint(
    StateId poseId0, StateId poseId1) const
{
  std::lock_guard<std::mutex> lock(loopPathProtectionMutex_);
  return protectedLoopPathRelativePoseConstraints_.count(
      orderedStatePairKey(poseId0, poseId1)) > 0;
}

bool ViSlamBackend::isLoopPathRelativePoseConstraintInstalled(
    StateId poseId0, StateId poseId1, bool fullGraph) const
{
  std::lock_guard<std::mutex> lock(loopPathProtectionMutex_);
  const auto& installed = fullGraph
      ? protectedLoopPathRelativePoseConstraintsFullGraph_
      : protectedLoopPathRelativePoseConstraintsRealtime_;
  return installed.count(orderedStatePairKey(poseId0, poseId1)) > 0;
}

void ViSlamBackend::protectLoopPathRelativePoseConstraint(
    StateId poseId0, StateId poseId1,
    bool installedRealtime, bool installedFullGraph)
{
  std::lock_guard<std::mutex> lock(loopPathProtectionMutex_);
  const auto key = orderedStatePairKey(poseId0, poseId1);
  protectedLoopPathRelativePoseConstraints_.insert(key);
  if(installedRealtime) {
    protectedLoopPathRelativePoseConstraintsRealtime_.insert(key);
  }
  if(installedFullGraph) {
    protectedLoopPathRelativePoseConstraintsFullGraph_.insert(key);
  }
}

void ViSlamBackend::unprotectLoopPathRelativePoseConstraint(
    StateId poseId0, StateId poseId1)
{
  std::lock_guard<std::mutex> lock(loopPathProtectionMutex_);
  const auto key = orderedStatePairKey(poseId0, poseId1);
  protectedLoopPathRelativePoseConstraints_.erase(key);
  protectedLoopPathRelativePoseConstraintsRealtime_.erase(key);
  protectedLoopPathRelativePoseConstraintsFullGraph_.erase(key);
}

bool ViSlamBackend::isStatePairInSuppressedLoopPath(
    StateId poseId0, StateId poseId1) const
{
  if(poseId0 == poseId1) {
    return false;
  }
  for(const auto& snapshot : suppressedLoopPathConstraintSnapshots_) {
    bool found0 = false;
    bool found1 = false;
    for(const StateId stateId : snapshot.stateIds) {
      found0 = found0 || stateId == poseId0;
      found1 = found1 || stateId == poseId1;
      if(found0 && found1) {
        return true;
      }
    }
  }
  return false;
}

bool ViSlamBackend::isSuppressedLoopPathRelativePoseConstraint(
    StateId poseId0, StateId poseId1) const
{
  return suppressedLoopPathRelativePoseConstraints_.count(
      orderedStatePairKey(poseId0, poseId1)) > 0
      || isStatePairInSuppressedLoopPath(poseId0, poseId1);
}

bool ViSlamBackend::isSuppressedLoopPathTwoPoseConstraint(
    StateId poseId0, StateId poseId1) const
{
  return suppressedLoopPathTwoPoseConstraints_.count(
      orderedStatePairKey(poseId0, poseId1)) > 0
      || isStatePairInSuppressedLoopPath(poseId0, poseId1);
}

bool ViSlamBackend::removeTwoPoseLinkIfPresent(
    ViGraphEstimator& graph, StateId poseId0, StateId poseId1)
{
  if(graph.states_.count(poseId0) == 0 || graph.states_.count(poseId1) == 0) {
    return false;
  }
  auto& state0 = graph.states_.at(poseId0);
  auto& state1 = graph.states_.at(poseId1);
  if(state0.twoPoseLinks.count(poseId1) == 0) {
    return false;
  }
  const auto link = state0.twoPoseLinks.at(poseId1);
  if(link.residualBlockId != nullptr) {
    graph.problem_->RemoveResidualBlock(link.residualBlockId);
  }
  state0.twoPoseLinks.erase(poseId1);
  state1.twoPoseLinks.erase(poseId0);
  return true;
}

bool ViSlamBackend::removeTwoPoseConstLinkIfPresent(
    ViGraphEstimator& graph, StateId poseId0, StateId poseId1)
{
  if(graph.states_.count(poseId0) == 0 || graph.states_.count(poseId1) == 0) {
    return false;
  }
  auto& state0 = graph.states_.at(poseId0);
  auto& state1 = graph.states_.at(poseId1);
  if(state0.twoPoseConstLinks.count(poseId1) == 0) {
    return false;
  }
  const auto link = state0.twoPoseConstLinks.at(poseId1);
  if(link.residualBlockId != nullptr) {
    graph.problem_->RemoveResidualBlock(link.residualBlockId);
  }
  state0.twoPoseConstLinks.erase(poseId1);
  state1.twoPoseConstLinks.erase(poseId0);
  return true;
}

void ViSlamBackend::addSuppressedLoopPathConstraintSnapshot(
    const std::vector<StateId>& stateIds)
{
  if(stateIds.size() < 2) {
    return;
  }
  LoopPathConstraintSuppressionSnapshot snapshot;
  snapshot.stateIds = stateIds;
  suppressedLoopPathConstraintSnapshots_.push_back(snapshot);
}

std::pair<size_t, size_t> ViSlamBackend::suppressLoopPathConstraintsInGraph(
    ViGraphEstimator& graph, const std::vector<StateId>& stateIds)
{
  if(stateIds.size() < 2) {
    return {0, 0};
  }

  std::set<StateId> pathIdSet(stateIds.begin(), stateIds.end());
  std::set<std::pair<uint64_t, uint64_t>> visitedRelative;
  std::set<std::pair<uint64_t, uint64_t>> visitedTwoPose;
  std::vector<std::pair<StateId, StateId>> relativePairsToRemove;
  std::vector<std::pair<StateId, StateId>> twoPosePairsToRemove;

  for(const StateId stateId : stateIds) {
    if(graph.states_.count(stateId) == 0) {
      continue;
    }
    const auto& state = graph.states_.at(stateId);
    for(const auto& link : state.relativePoseLinks) {
      const StateId otherStateId = link.first;
      if(pathIdSet.count(otherStateId) == 0) {
        continue;
      }
      const auto key = orderedStatePairKey(stateId, otherStateId);
      if(!visitedRelative.insert(key).second) {
        continue;
      }
      if(isPersistentFullGraphRelativePoseConstraint(stateId, otherStateId)
         || isProtectedLoopPathRelativePoseConstraint(stateId, otherStateId)) {
        continue;
      }
      suppressedLoopPathRelativePoseConstraints_.insert(key);
      relativePairsToRemove.emplace_back(stateId, otherStateId);
    }
    for(const auto& link : state.twoPoseLinks) {
      const StateId otherStateId = link.first;
      if(pathIdSet.count(otherStateId) == 0) {
        continue;
      }
      const auto key = orderedStatePairKey(stateId, otherStateId);
      if(!visitedTwoPose.insert(key).second) {
        continue;
      }
      suppressedLoopPathTwoPoseConstraints_.insert(key);
      twoPosePairsToRemove.emplace_back(stateId, otherStateId);
    }
    for(const auto& link : state.twoPoseConstLinks) {
      const StateId otherStateId = link.first;
      if(pathIdSet.count(otherStateId) == 0) {
        continue;
      }
      const auto key = orderedStatePairKey(stateId, otherStateId);
      if(!visitedTwoPose.insert(key).second) {
        continue;
      }
      suppressedLoopPathTwoPoseConstraints_.insert(key);
      twoPosePairsToRemove.emplace_back(stateId, otherStateId);
    }
  }

  size_t relativeRemoved = 0;
  for(const auto& pair : relativePairsToRemove) {
    if(graph.states_.count(pair.first) != 0
       && graph.states_.count(pair.second) != 0
       && graph.states_.at(pair.first).relativePoseLinks.count(pair.second) != 0) {
      graph.removeRelativePoseConstraint(pair.first, pair.second);
      ++relativeRemoved;
    }
  }

  size_t twoPoseRemoved = 0;
  for(const auto& pair : twoPosePairsToRemove) {
    if(removeTwoPoseLinkIfPresent(graph, pair.first, pair.second)) {
      ++twoPoseRemoved;
    }
    if(removeTwoPoseConstLinkIfPresent(graph, pair.first, pair.second)) {
      ++twoPoseRemoved;
    }
  }

  return {relativeRemoved, twoPoseRemoved};
}

void ViSlamBackend::applySuppressedLoopPathConstraintRemovals(
    bool updateRealtimeGraph, bool updateFullGraph)
{
  if(!updateRealtimeGraph && !updateFullGraph) {
    return;
  }
  size_t realtimeRelativeRemoved = 0;
  size_t realtimeTwoPoseRemoved = 0;
  size_t fullRelativeRemoved = 0;
  size_t fullTwoPoseRemoved = 0;
  for(const auto& snapshot : suppressedLoopPathConstraintSnapshots_) {
    if(updateRealtimeGraph) {
      const auto removed = suppressLoopPathConstraintsInGraph(
          realtimeGraph_, snapshot.stateIds);
      realtimeRelativeRemoved += removed.first;
      realtimeTwoPoseRemoved += removed.second;
    }
    if(updateFullGraph) {
      const auto removed = suppressLoopPathConstraintsInGraph(
          fullGraph_, snapshot.stateIds);
      fullRelativeRemoved += removed.first;
      fullTwoPoseRemoved += removed.second;
    }
  }
  if(realtimeRelativeRemoved + realtimeTwoPoseRemoved
     + fullRelativeRemoved + fullTwoPoseRemoved > 0) {
    LOG(INFO) << "Suppressed fallback loop path constraints"
              << " realtime(relative=" << realtimeRelativeRemoved
              << ", two_pose=" << realtimeTwoPoseRemoved << ")"
              << " full(relative=" << fullRelativeRemoved
              << ", two_pose=" << fullTwoPoseRemoved << ").";
  }
}

bool ViSlamBackend::addOrUpdateProtectedLoopPathRelativePoseConstraint(
    StateId poseId0, StateId poseId1,
    const kinematics::Transformation& T_S0S1,
    const Eigen::Matrix<double, 6, 6>& information,
    bool updateRealtimeGraph, bool updateFullGraph)
{
  bool updatedRealtime = false;
  bool updatedFullGraph = false;
  if(updateRealtimeGraph
     && realtimeGraph_.states_.count(poseId0) != 0
     && realtimeGraph_.states_.count(poseId1) != 0) {
    const bool hasLink =
        realtimeGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1) != 0;
    if(hasLink && isLoopPathRelativePoseConstraintInstalled(poseId0, poseId1, false)) {
      // The persistent fallback-loop chain is immutable once created. Avoid
      // remove/add churn on every realtime optimisation step; Ceres Problem
      // mutation is not thread-safe with the background full-graph optimiser.
    } else {
      if(hasLink) {
        realtimeGraph_.removeRelativePoseConstraint(poseId0, poseId1);
      }
      realtimeGraph_.addRelativePoseConstraint(poseId0, poseId1, T_S0S1, information);
      updatedRealtime = true;
    }
  }

  if(updateFullGraph
     && fullGraph_.states_.count(poseId0) != 0
     && fullGraph_.states_.count(poseId1) != 0) {
    const bool hasLink =
        fullGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1) != 0;
    if(hasLink && isLoopPathRelativePoseConstraintInstalled(poseId0, poseId1, true)) {
      // See realtime-graph branch above.
    } else {
      if(hasLink) {
        fullGraph_.removeRelativePoseConstraint(poseId0, poseId1);
      }
      fullGraph_.addRelativePoseConstraint(poseId0, poseId1, T_S0S1, information);
      updatedFullGraph = true;
    }
  }

  if(updatedRealtime || updatedFullGraph) {
    // A protected pair in one graph can still hold an old sensor edge in the
    // other graph. Only record installations performed by this invocation.
    protectLoopPathRelativePoseConstraint(
        poseId0, poseId1, updatedRealtime, updatedFullGraph);
  }
  return updatedRealtime || updatedFullGraph;
}

void ViSlamBackend::addPersistentLoopPathConstraintSnapshot(
    const std::vector<StateId>& stateIds,
    const AlignedVector<kinematics::Transformation>& T_WS,
    const Eigen::Matrix<double, 6, 6>& information,
    double informationScale, int stride,
    bool weakenExistingConstraints, double existingConstraintScale)
{
  (void)weakenExistingConstraints;
  (void)existingConstraintScale;
  if(stateIds.size() < 2 || stateIds.size() != T_WS.size()) {
    return;
  }
  LoopPathConstraintSnapshot snapshot;
  snapshot.stateIds = stateIds;
  snapshot.T_WS = T_WS;
  snapshot.information = information;
  snapshot.informationScale =
      std::isfinite(informationScale) && informationScale > 0.0
          ? informationScale
          : 100.0;
  snapshot.stride = std::max(1, stride);
  persistentLoopPathConstraintSnapshots_.push_back(snapshot);
}

void ViSlamBackend::reapplyPersistentLoopPathRelativePoseConstraints(
    bool updateRealtimeGraph, bool updateFullGraph)
{
  if(!updateRealtimeGraph && !updateFullGraph) {
    return;
  }

  applySuppressedLoopPathConstraintRemovals(updateRealtimeGraph, updateFullGraph);

  size_t constraintsAdded = 0;
  for(const auto& snapshot : persistentLoopPathConstraintSnapshots_) {
    if(snapshot.stateIds.size() < 2 || snapshot.stateIds.size() != snapshot.T_WS.size()) {
      continue;
    }

    auto collectSurvivingIndices = [&](auto& graph) {
      std::vector<size_t> survivingIndices;
      survivingIndices.reserve(snapshot.stateIds.size());
      for(size_t i = 0; i < snapshot.stateIds.size(); ++i) {
        if(graph.states_.count(snapshot.stateIds[i]) != 0) {
          survivingIndices.push_back(i);
        }
      }
      return survivingIndices;
    };

    std::vector<size_t> realtimeSurvivingIndices;
    std::vector<size_t> fullGraphSurvivingIndices;
    if(updateRealtimeGraph) {
      realtimeSurvivingIndices = collectSurvivingIndices(realtimeGraph_);
    }
    if(updateFullGraph) {
      fullGraphSurvivingIndices = collectSurvivingIndices(fullGraph_);
    }

    auto addChainConstraints =
        [&](const std::vector<size_t>& survivingIndices,
            bool updateRealtime, bool updateFull) {
      if(survivingIndices.size() < 2) {
        return;
      }
      const int stride = std::max(1, snapshot.stride);
      const Eigen::Matrix<double, 6, 6> chainInformation =
          snapshot.informationScale * snapshot.information;
      size_t startSurvivor = 0;
      size_t skippedSinceLastConstraint = 0;
      for(size_t survivor = 1; survivor < survivingIndices.size(); ++survivor) {
        const bool isEndpoint = survivor + 1 == survivingIndices.size();
        ++skippedSinceLastConstraint;
        if(!isEndpoint && skippedSinceLastConstraint < size_t(stride)) {
          continue;
        }

        const size_t startIndex = survivingIndices[startSurvivor];
        const size_t endIndex = survivingIndices[survivor];
        const StateId startId = snapshot.stateIds[startIndex];
        const StateId endId = snapshot.stateIds[endIndex];
        if(startId == endId) {
          continue;
        }
        const kinematics::Transformation T_Sstart_Send =
            snapshot.T_WS[startIndex].inverse() * snapshot.T_WS[endIndex];
        if(addOrUpdateProtectedLoopPathRelativePoseConstraint(
               startId, endId, T_Sstart_Send, chainInformation,
               updateRealtime, updateFull)) {
          ++constraintsAdded;
        }
        startSurvivor = survivor;
        skippedSinceLastConstraint = 0;
      }
    };

    if(updateRealtimeGraph) {
      addChainConstraints(realtimeSurvivingIndices, true, false);
    }
    if(updateFullGraph) {
      addChainConstraints(fullGraphSurvivingIndices, false, true);
    }
  }

  if(constraintsAdded > 0) {
    LOG(INFO) << "Reapplied " << constraintsAdded
              << " persistent fallback loop path chain constraints"
              << " (realtime=" << updateRealtimeGraph
              << ", full_graph=" << updateFullGraph << ").";
  }
}

bool ViSlamBackend::addRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                              const kinematics::Transformation &T_S0S1,
                                              const Eigen::Matrix<double, 6, 6>& information,
                                              bool overwriteProtected)
{
  if(realtimeGraph_.states_.count(poseId0) == 0 || realtimeGraph_.states_.count(poseId1) == 0) {
    return false;
  }
  if(isProtectedLoopPathRelativePoseConstraint(poseId0, poseId1)
     && !overwriteProtected) {
    LOG(INFO) << "Skip overwriting protected fallback loop-chain relative pose constraint "
              << poseId0.value() << " <-> " << poseId1.value() << ".";
    return false;
  }
  if(isSuppressedLoopPathRelativePoseConstraint(poseId0, poseId1)
     && !overwriteProtected) {
    LOG(INFO) << "Skip adding suppressed fallback loop-path relative pose constraint "
              << poseId0.value() << " <-> " << poseId1.value() << ".";
    return false;
  }

  // If a constraint already exists, overwrite by removing and re-adding.
  if(realtimeGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1)) {
    realtimeGraph_.removeRelativePoseConstraint(poseId0, poseId1);
  }
  realtimeGraph_.addRelativePoseConstraint(poseId0, poseId1, T_S0S1, information);

  if(isLoopClosing_ || isLoopClosureAvailable_) {
    // fullGraph_ not safely accessible; mark for later sync.
    touchedStates_.insert(poseId0);
    touchedStates_.insert(poseId1);
    return true;
  }

  if(fullGraph_.states_.count(poseId0) == 0 || fullGraph_.states_.count(poseId1) == 0) {
    // Should not usually happen outside loop-closing, but keep consistent by syncing later.
    touchedStates_.insert(poseId0);
    touchedStates_.insert(poseId1);
    return true;
  }

  if(fullGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1)) {
    if(isProtectedLoopPathRelativePoseConstraint(poseId0, poseId1)
       && !overwriteProtected) {
      return true;
    }
    fullGraph_.removeRelativePoseConstraint(poseId0, poseId1);
  }
  fullGraph_.addRelativePoseConstraint(poseId0, poseId1, T_S0S1, information);
  return true;
}

bool ViSlamBackend::removeRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                                 bool removeProtected)
{
  if(isProtectedLoopPathRelativePoseConstraint(poseId0, poseId1)
     && !removeProtected) {
    LOG(INFO) << "Skip removing protected fallback loop-chain relative pose constraint "
              << poseId0.value() << " <-> " << poseId1.value() << ".";
    return false;
  }

  erasePersistentFullGraphRelativePoseConstraint(poseId0, poseId1);
  if(removeProtected) {
    unprotectLoopPathRelativePoseConstraint(poseId0, poseId1);
  }

  if (realtimeGraph_.states_.count(poseId0) == 0 || realtimeGraph_.states_.count(poseId1) == 0) {
    return false;
  }

  bool removed = false;
  if (realtimeGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1)) {
    realtimeGraph_.removeRelativePoseConstraint(poseId0, poseId1);
    removed = true;
  }

  if (isLoopClosing_ || isLoopClosureAvailable_) {
    // fullGraph_ not safely accessible; mark for later sync.
    touchedStates_.insert(poseId0);
    touchedStates_.insert(poseId1);
    return removed;
  }

  if (fullGraph_.states_.count(poseId0) == 0 || fullGraph_.states_.count(poseId1) == 0) {
    touchedStates_.insert(poseId0);
    touchedStates_.insert(poseId1);
    return removed;
  }

  if (fullGraph_.states_.at(poseId0).relativePoseLinks.count(poseId1)) {
    fullGraph_.removeRelativePoseConstraint(poseId0, poseId1);
    removed = true;
  }
  return removed;
}

bool ViSlamBackend::isPersistentFullGraphRelativePoseConstraint(StateId poseId0,
                                                                StateId poseId1) const
{
  for(const auto& constraint : persistentFullGraphRelativePoseConstraints_) {
    const bool sameDirection = constraint.pose_i == poseId0 && constraint.pose_j == poseId1;
    const bool oppositeDirection = constraint.pose_i == poseId1 && constraint.pose_j == poseId0;
    if(sameDirection || oppositeDirection) {
      return true;
    }
  }
  return false;
}

void ViSlamBackend::addOrUpdatePersistentFullGraphRelativePoseConstraint(
    const RelPoseInfo& constraint)
{
  RelPoseInfo persistentConstraint = constraint;
  persistentConstraint.keepInFullGraph = true;
  persistentConstraint.informationScale =
      std::isfinite(persistentConstraint.persistentInformationScale)
      && persistentConstraint.persistentInformationScale > 0.0
          ? persistentConstraint.persistentInformationScale
          : persistentConstraint.informationScale;
  for(auto& item : persistentFullGraphRelativePoseConstraints_) {
    const bool sameDirection = item.pose_i == persistentConstraint.pose_i
                               && item.pose_j == persistentConstraint.pose_j;
    const bool oppositeDirection = item.pose_i == persistentConstraint.pose_j
                                  && item.pose_j == persistentConstraint.pose_i;
    if(sameDirection || oppositeDirection) {
      item = persistentConstraint;
      return;
    }
  }
  persistentFullGraphRelativePoseConstraints_.push_back(persistentConstraint);
}

void ViSlamBackend::erasePersistentFullGraphRelativePoseConstraint(StateId poseId0,
                                                                   StateId poseId1)
{
  for(auto it = persistentFullGraphRelativePoseConstraints_.begin();
      it != persistentFullGraphRelativePoseConstraints_.end();) {
    const bool sameDirection = it->pose_i == poseId0 && it->pose_j == poseId1;
    const bool oppositeDirection = it->pose_i == poseId1 && it->pose_j == poseId0;
    if(sameDirection || oppositeDirection) {
      it = persistentFullGraphRelativePoseConstraints_.erase(it);
    } else {
      ++it;
    }
  }
}

void ViSlamBackend::reapplyPersistentFullGraphRelativePoseConstraints()
{
  for(const auto& item : persistentFullGraphRelativePoseConstraints_) {
    if(fullGraph_.states_.count(item.pose_i) == 0
       || fullGraph_.states_.count(item.pose_j) == 0) {
      continue;
    }
    if(fullGraph_.states_.at(item.pose_i).relativePoseLinks.count(item.pose_j)) {
      fullGraph_.removeRelativePoseConstraint(item.pose_i, item.pose_j);
    }
    const double informationScale =
        std::isfinite(item.informationScale) && item.informationScale > 0.0
            ? item.informationScale
            : 100.0;
    fullGraph_.addRelativePoseConstraint(item.pose_i, item.pose_j, item.T_Si_Sj,
                                         informationScale * item.information);
  }
}

bool ViSlamBackend::addPoseConstraint(StateId poseId,
                                      const kinematics::Transformation &T_WS,
                                      const Eigen::Matrix<double, 6, 6>& information)
{
  if (realtimeGraph_.states_.count(poseId) == 0) {
    return false;
  }

  // Only add when graphs are accessible to keep both consistent (same policy as velocity/speed priors).
  if (isLoopClosing_ || isLoopClosureAvailable_) {
    return false;
  }

  realtimeGraph_.addPoseConstraint(poseId, T_WS, information);
  if (fullGraph_.states_.count(poseId)) {
    fullGraph_.addPoseConstraint(poseId, T_WS, information);
  }
  return true;
}

bool ViSlamBackend::addVelocityConstraint(StateId poseId,
                                          const Eigen::Vector3d& v_S,
                                          const Eigen::Matrix3d& information)
{
  if(realtimeGraph_.states_.count(poseId) == 0) {
    return false;
  }

  // only add when graphs are accessible to keep both consistent
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    return false;
  }

  realtimeGraph_.addVelocityConstraint(poseId, v_S, information);
  if(fullGraph_.states_.count(poseId)) {
    fullGraph_.addVelocityConstraint(poseId, v_S, information);
  }
  return true;
}

bool ViSlamBackend::removeVelocityConstraints(StateId poseId)
{
  if (realtimeGraph_.states_.count(poseId) == 0) {
    return false;
  }
  // Only modify when both graphs are safely accessible to keep consistency.
  if (isLoopClosing_ || isLoopClosureAvailable_) {
    return false;
  }
  realtimeGraph_.removeVelocityConstraints(poseId);
  if (fullGraph_.states_.count(poseId)) {
    fullGraph_.removeVelocityConstraints(poseId);
  }
  return true;
}

bool ViSlamBackend::setVelocityPrior(StateId poseId, const Eigen::Vector3d& v_S,
                                    double information, bool enabled)
{
  // The lock also covers the interval before the full-graph worker sets its flag.
  // A rejected replacement must leave the old prior intact in both graphs.
  std::unique_lock<std::mutex> graphLock(fullGraphTwoPoseWeightMutex_, std::try_to_lock);
  if(!graphLock.owns_lock() || isLoopClosing_ || isLoopClosureAvailable_
     || realtimeGraph_.states_.count(poseId) == 0
     || fullGraph_.states_.count(poseId) == 0) {
    return false;
  }
  auto update = [&](ViGraphEstimator& graph) {
    graph.removeVelocityConstraints(poseId);
    if(enabled) {
      graph.addVelocityConstraint(poseId, v_S, Eigen::Matrix3d::Identity() * information);
    }
  };
  update(realtimeGraph_);
  update(fullGraph_);
  return true;
}

bool ViSlamBackend::applyStrategy(size_t numKeyframes,
                                  size_t numLoopClosureFrames,
                                  size_t numImuFrames,
                                  std::set<StateId> &affectedFrames,
                                  bool expand)
{
  // check/handle lost
  TimerSwitchable t1("7.1 check/handle lost");
  if(imuFrames_.size()>1 && !keyFrames_.empty()) {
    auto iter = auxiliaryStates_.rbegin();
    const double quality = trackingQuality(iter->first);
    if (quality < 0.01) {
      if (quality > 0.00001) {
        LOG(WARNING) << "Tracking quality weak: quality=" << quality;
      } else {
        LOG(WARNING) << "TRACKING FAILURE: quality=" << quality;
      }
      /// \todo: lost component handling currently disabled. Re-introduce!
    }
  }
  t1.stop();

  // first eliminate the IMU frames that are too much and that are not keyframes
  TimerSwitchable t2("7.2 eliminate IMU frames");
  eliminateImuFrames(numImuFrames, affectedFrames);
  t2.stop();

  // now tackle keyframes, if necessary
  StateId currentKeyframeId = currentKeyframeStateId();
  StateId currentFrameId = realtimeGraph_.currentStateId();

  if(!currentKeyframeId.isInitialised()) {
    return true; /// \todo fix this
  }

  // now eliminate keyframes
  bool keyFrameEliminated = false;
  int ctrPg = 0;
  if(keyFrames_.size() > numKeyframes) {
    TimerSwitchable t3("7.3 convert to posegraph");
    while(keyFrames_.size() > numKeyframes) {
      // find keyframe with least common observations with current frame or current keyframe
      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto keyFrame : keyFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, keyFrame),
                                   realtimeGraph_.covisibilities(currentKeyframeId, keyFrame));
        if(keyFrame == *keyFrames_.begin()){
          if(coObs>=2) {
            continue; // spare
          }
        }
        if(coObs < minObservations) {
          minObservations = coObs;
          minId = keyFrame;
        }
      }
      StateId maxId;
      int maxCoObs=0;
      std::set<StateId> framesToConsider;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {
          // this is a frontier node
          framesToConsider.insert(frameId);
        }
      }

      // convert it.
      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; // flag it to be posegraph frame
      keyFrames_.erase(minId);
      for(auto id : keyFrames_) {
        auxiliaryStates_.at(minId).recentLoopClosureFrames.insert(id); // don't re-add immediately.
      }
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      keyFrameEliminated = true;
      if(maxCoObs == 0) {
        // handle weird case that might happen with (almost) no matches
        realtimeGraph_.removeAllObservations(minId);
        // also manage the full graph, if possible
        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if(convertToPosegraphFrames.size() > 0) {
        bool success = convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider,
                                             affectedFrames);
        ctrPg++;
        OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(minId).observations.size()==0,
                          "observations at ID " << minId.value() << " , success=" << int(success))
        if(ctrPg>=3) {
          break; // max 3 at the time...
        }
      }
    }
    t3.stop();
  }

  // freeze old states
  TimerSwitchable t4("7.4 freezing");
  if(keyFrameEliminated) {
    auto iter = auxiliaryStates_.rbegin();
    StateId oldestKeyFrameId;
    for (size_t p = 0; p<(numKeyframes+numImuFrames) && iter!=auxiliaryStates_.rend(); ++p) {
      iter++;
    }
    if(iter!=auxiliaryStates_.rend()) {
      oldestKeyFrameId = iter->first;
    }

    int ctr = 0;
    for(auto iter = realtimeGraph_.states_.find(oldestKeyFrameId);
        iter != realtimeGraph_.states_.end(); --iter) {
      if(ctr==numRealtimePoseGraphFrames) {
        Time freezeTime = realtimeGraph_.states_.rbegin()->second.timestamp;
        while((freezeTime - iter->second.timestamp).toSec() < minDeltaT) {
          if(iter==realtimeGraph_.states_.begin()) {
            break;
          }
          --iter;
        }
        if(iter!=realtimeGraph_.states_.begin()) {
          // freezing of poses
          // make sure not to go and unfreeze old stuff
          // -- messes up synchronising if concurrentlz loop-optimising
          const StateId freezeId = std::max(lastFreeze_, iter->first);
          // freezing of poses
          lastFreeze_ = freezeId;
          if(realtimeGraph_.states_.find(freezeId) != realtimeGraph_.states_.begin()) {
            realtimeGraph_.removeSubmapAlignmentConstraintsUntil(freezeId);
            realtimeGraph_.freezePosesUntil(freezeId);
          }
          // freezing of speed/bias
          realtimeGraph_.freezeSpeedAndBiasesUntil(freezeId);
        }
        break;
      }
      if(iter==realtimeGraph_.states_.begin()) {
        break;
      }
      ctr++;
    }
  }
  t4.stop();

  // now tackle loop closure frames
  int ctrLc = 0;
  if(loopClosureFrames_.size() > 0) {
    TimerSwitchable t5("7.5 convert to posegraph loop-closure frames");
    do {

      // find keyframe with least common observations with current frame or current keyframe
      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto loopClosureFrame : loopClosureFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, loopClosureFrame),
                                realtimeGraph_.covisibilities(currentKeyframeId, loopClosureFrame));
        //size_t coObs = realtimeGraph_.covisibilities(currentFrameId, loopClosureFrame);
        if(coObs < minObservations) {
          minObservations = coObs;
          minId = loopClosureFrame;
        }
      }

      // eliminate no covisibility in any case
      if(minObservations!=0 && loopClosureFrames_.size()<=numLoopClosureFrames) {
        break;
      }

      std::set<StateId> framesToConsider;
      StateId maxId;
      int maxCoObs = 0;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {
          // this is a frontier node
          framesToConsider.insert(frameId);
        }
      }

      // convert it.
      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; // flag it to be posegraph frame
      loopClosureFrames_.erase(minId);
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      if(maxCoObs == 0) {
        // handle weird case that might happen with (almost) no matches
        realtimeGraph_.removeAllObservations(minId);
        // also manage the full graph, if possible
        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if(convertToPosegraphFrames.size() > 0 && maxCoObs > 0) {
        convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider, affectedFrames);
        ++ctrLc;
        if(ctrLc>=3) {
          break; // max 3 at the time.
        }
      }
    } while (loopClosureFrames_.size() > numLoopClosureFrames);
    t5.stop();
  }

  /// \todo the following would need re-optimisation of landmarks...
  // expand frontier, if necessary
  if(expand && ctrLc<3 && ctrPg<3) {
    TimerSwitchable t6("7.6 expand");
    currentKeyframeId = currentKeyframeStateId();
    if(currentKeyframeId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentKeyframeId).twoPoseLinks.size()>0) {
        expandKeyframe(currentKeyframeId);
      }
    }
    const StateId currentLoopclosureFrameId = currentLoopclosureStateId();
    if(currentLoopclosureFrameId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentLoopclosureFrameId).twoPoseLinks.size()>0) {
        expandKeyframe(currentLoopclosureFrameId);
      }
    }
    t6.stop();
  }

  return true;
}

void ViSlamBackend::optimiseRealtimeGraph(
  int numIter, std::vector<StateId> & updatedStates, int numThreads, bool verbose,
  bool onlyNewestState, bool isInitialised)
{

  //OKVIS_ASSERT_TRUE(Exception, areLandmarksInFrontOfCameras(), "before optimisation")

  // fix current position, if not initialised
  std::unique_ptr<ceres::PoseError> initialFixation;
  ::ceres::ResidualBlockId initialFixationId = nullptr;
  if(!isInitialised){
    Eigen::Matrix<double, 6, 1> informationDiag = Eigen::Matrix<double, 6, 1>::Ones();
    informationDiag[0] = 1.0e8;
    informationDiag[1] = 1.0e8;
    informationDiag[2] = 1.0e8;
    informationDiag[3] = 0.0;
    informationDiag[4] = 0.0;
    informationDiag[5] = 0.0;
    initialFixation.reset(
      new ceres::PoseError(realtimeGraph_.states_.at(StateId(1)).pose->estimate(), informationDiag));
    initialFixationId = realtimeGraph_.problem_->AddResidualBlock(
        initialFixation.get(), nullptr,
        realtimeGraph_.states_.rbegin()->second.pose->parameters());
    kinematics::Transformation T_WS(
        realtimeGraph_.states_.at(StateId(1)).pose->estimate().r(),
        realtimeGraph_.states_.rbegin()->second.pose->estimate().q());
    realtimeGraph_.setPose(realtimeGraph_.states_.rbegin()->first, T_WS);
  }

  // freeze if requested
  bool frozen = false;
  StateId unfreezeId;
  if(onlyNewestState) {
    // paranoid: find last frozen
    for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
        ++riter) {
      if(riter->second.pose->fixed()) {
        break;
      } else {
        unfreezeId = riter->first;
      }
    }

    auto riter = realtimeGraph_.states_.rbegin();
    riter++;
    if(riter != realtimeGraph_.states_.rend()) {
      realtimeGraph_.freezePosesUntil(riter->first);
      realtimeGraph_.freezeSpeedAndBiasesUntil(riter->first);
      frozen = true;
    }
    for(const auto & lm : realtimeGraph_.landmarks_) {
      realtimeGraph_.problem_->SetParameterBlockConstant(lm.second.hPoint->parameters());
    }

    // freeze extrinsics
    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockConstant(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }
  }

  // Realtime optimisation may run concurrently with a background full-graph
  // optimisation. Ceres Problem mutation is not thread-safe, so this thread must
  // only touch the realtime graph; the full graph reapplies these persistent
  // constraints inside optimiseFullGraph()/synchroniseRealtimeAndFullGraph().
  reapplyPersistentLoopPathRelativePoseConstraints(true, false);

  // run the optimiser
  realtimeGraph_.options_.linear_solver_type = ::ceres::DENSE_SCHUR;
  realtimeGraph_.options_.initial_trust_region_radius = realtimeTrustRegionRadius_.initial(
      realtimeGraph_.options_.min_trust_region_radius,
      realtimeGraph_.options_.max_trust_region_radius, onlyNewestState, isInitialised);
  realtimeGraph_.optimise(numIter, numThreads, verbose);
  const auto& realtimeSummary = realtimeGraph_.summary();
  realtimeTrustRegionRadius_.observe(
      realtimeSummary.iterations.empty() ? 0.0 : realtimeSummary.iterations.back().trust_region_radius,
      realtimeSummary.IsSolutionUsable(), onlyNewestState, isInitialised);

  // unfreeze if necessary
  if(onlyNewestState) {
    // unfreeze extrinsics
    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockVariable(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }

    if(frozen) {
      realtimeGraph_.unfreezePosesFrom(unfreezeId);
      realtimeGraph_.unfreezeSpeedAndBiasesFrom(unfreezeId);
    }
    if(onlyNewestState) {
      for(const auto & lm : realtimeGraph_.landmarks_) {
        realtimeGraph_.problem_->SetParameterBlockVariable(lm.second.hPoint->parameters());
      }
    }

    // undo initial fixation
    if(initialFixation && initialFixationId) {
      realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
    }

    // adopt pose change
    auto riter = realtimeGraph_.states_.rbegin();
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
    }

    // check consistency -- currently disabled
    //if(!isLoopClosing_ && !isLoopClosureAvailable_) {
    //  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.isSynched(fullGraph_), "not synched");
    //}
    return;
  }

  // import landmarks
  if(!onlyNewestState) {
    realtimeGraph_.updateLandmarks();
  }

  // also copy states to observationless and fullGraph (if possible)
  for (auto id : updatedStatesLoopClosureAttempt_) {
    updatedStates.push_back(id); // remember that these were also updated (from initialisation)
  }
  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()) {
      break; // done, don't need to iterate further...
    }
    if(!updatedStatesLoopClosureAttempt_.count(riter->first)) {
       updatedStates.push_back(riter->first);
    }
    /// consciously ignoring extrinsics here. \todo generalise for non-fixed extrinsics...
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
      fullState.T_GW->setEstimate(riter->second.T_GW->estimate());
    }
  }

  // Check if GPS Trafo observable
  if(!gpsObservability_){ // gps not yet observable, copy estimate to other graphs if possible
      //  Copy T_GW estimates to observationLessGraph_ and fullGaph_
      // for fullGraph first check if possible -->  if(!isLoopClosing_ && !isLoopClosureAvailable_)
      if(!isLoopClosing_ && !isLoopClosureAvailable_){
          fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
      }
  }
  else{ // GPS is observable; check if fullGraph_ available and T_GW already fixed
      if(!realtimeGraph_.isGpsFixed()){ // realtimeGraph_ not yet fixed
          if(!isLoopClosing_ && !isLoopClosureAvailable_){ // fullGraph_ accessible
              realtimeGraph_.freezeGpsExtrinsics();
              fullGraph_.setGpsExtrinsics(realtimeGraph_.T_GW());
              fullGraph_.freezeGpsExtrinsics();
          }
      }
  }
  updatedStatesLoopClosureAttempt_.clear(); // processed now, so clear

  // ... and landmarks to fullGraph (if possible)
  if(!isLoopClosing_ && !isLoopClosureAvailable_) {
    for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
        ++iter) {
      fullGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                             iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  // finally adopt stuff from realtime optimisation results in full and observation-less graphs
  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(!riter->second.previousImuLink.errorTerm) {
      continue;
    }
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      if (fullGraph_.imuParametersVec_.at(0).use && fullGraph_.imuParametersVec_.at(0).use_preintegration_residual) {
        std::static_pointer_cast<ceres::ImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(
            *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            riter->second.previousImuLink.errorTerm));
      }
    }
    if(riter->second.pose->fixed()) {
      break;
    }
  }

  // unfreeze if necessary
  if(initialFixation && initialFixationId) {
    realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
  }

  // check consistency -- currently disabled
  //if(!isLoopClosing_ && !isLoopClosureAvailable_) {
  //  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.isSynched(fullGraph_), "not synched");
  //}

  //OKVIS_ASSERT_TRUE(Exception, areLandmarksInFrontOfCameras(), "after optimisation")
}

bool ViSlamBackend::setOptimisationTimeLimit(double timeLimit, int minIterations)
{
  //fullGraph_.setOptimisationTimeLimit(timeLimit, 1); /// \todo Fix hack!
  return realtimeGraph_.setOptimisationTimeLimit(timeLimit, minIterations);
}

bool ViSlamBackend::isLandmarkAdded(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkAdded(landmarkId);
}

bool ViSlamBackend::isLandmarkInitialised(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkInitialised(landmarkId);
}

bool ViSlamBackend::setPose(StateId id, const kinematics::TransformationCacheless &pose)
{
  bool success = realtimeGraph_.setPose(id, pose);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setPose(id, pose);
  }
  return success;
}

bool ViSlamBackend::setSpeedAndBias(StateId id, const SpeedAndBias &speedAndBias)
{
  bool success = realtimeGraph_.setSpeedAndBias(id, speedAndBias);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setSpeedAndBias(id, speedAndBias);
  }
  return success;
}

bool ViSlamBackend::setExtrinsics(StateId id, uchar camIdx,
                                  const kinematics::TransformationCacheless & extrinsics)
{
  bool success = realtimeGraph_.setExtrinsics(id, camIdx, extrinsics);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setExtrinsics(id, camIdx, extrinsics);
  }
  return success;
}

bool ViSlamBackend::isInImuWindow(StateId id) const
{
  return auxiliaryStates_.at(id).isImuFrame;
}

Time ViSlamBackend::timestamp(StateId id) const
{
  return realtimeGraph_.timestamp(id);
}

void ViSlamBackend::drawOverheadImage(cv::Mat &image, int idx) const
{
  const ViGraphEstimator& graph = idx==0 ? realtimeGraph_ : fullGraph_;
  static const double borderPixels = 50;
  // first find max/min
  Eigen::Vector3d min(-0.05,-0.05,-0.05);
  Eigen::Vector3d max(0.05,0.05,0.05);
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {
    const Eigen::Vector3d pos = iter->second.pose->estimate().r();
    for(int i=0; i<3; ++i) {
      if(pos[i] > max[i]) {
        max[i] = pos[i];
      }
      if(pos[i] < min[i]) {
        min[i] = pos[i];
      }
    }
  }
  const Eigen::Vector3d centre = 0.5*(max+min);
  const double scale = std::min((image.cols-2.0*borderPixels)/(max[0]-min[0]),
      (image.rows-2.0*borderPixels)/(max[1]-min[1]));

  // landmarks
  for(auto iter = graph.landmarks_.begin(); iter != graph.landmarks_.end(); ++iter) {
    const Eigen::Vector4d hp = iter->second.hPoint->estimate();
    const Eigen::Vector3d pos = (hp.head<3>()/hp[3]-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    const double quality = iter->second.quality; /// \todo save/retrieve real quality
    if(cvPos.x > 0.0 && cvPos.y > 0.0 && cvPos.x < (image.cols-1) && cvPos.y < (image.rows-1)) {
        cv::circle(image, cvPos, 1, cv::Scalar(0,std::min(255,20+int(quality/0.03*225.0)),0),
                                               cv::FILLED, cv::LINE_AA);
    }
  }

  // Gps Measurements 
  for(auto iter = graph.states_.begin(); iter!= graph.states_.end(); ++iter){
    for(auto mIter = iter->second.GpsFactors.begin(); mIter != iter->second.GpsFactors.end(); ++mIter) {
      Eigen::Vector3d gps_point_in_W = (graph.T_GW().inverse().T() * mIter->errorTerm->measurement().homogeneous()).head<3>();
      Eigen::Vector3d gps_point_in_img = (gps_point_in_W - centre) * scale;
      cv::Point2d cvPos(gps_point_in_img[0]+image.cols*0.5, -gps_point_in_img[1]+image.rows*0.5);
      cv::drawMarker(image, cvPos, cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 12, 3);
    }
  }


  // draw co-observations (need to re-compute as cannot call computeCovisibilities in const method)
  std::map<uint64_t, std::map<uint64_t, int>> coObservationCounts;
  for(auto iter=graph.landmarks_.begin(); iter!=graph.landmarks_.end(); ++iter) {
    auto obs = iter->second.observations;
    std::set<uint64> covisibilities;
    for(auto obsiter=obs.begin(); obsiter!=obs.end(); ++obsiter) {
      covisibilities.insert(obsiter->first.frameId);
    }
    for(auto i0=covisibilities.begin(); i0!=covisibilities.end(); ++i0) {
      for(auto i1=covisibilities.begin(); i1!=covisibilities.end(); ++i1) {
        if(*i1>=*i0) {
          continue;
        }
        if(coObservationCounts.find(*i0)==coObservationCounts.end()) {
          coObservationCounts[*i0][*i1] = 1;
        } else {
          if (coObservationCounts.at(*i0).find(*i1)==coObservationCounts.at(*i0).end()) {
            coObservationCounts.at(*i0)[*i1] = 1;
          } else {
            coObservationCounts.at(*i0).at(*i1)++;
          }
        }
      }
    }
  }

  for(auto i0=coObservationCounts.begin(); i0!=coObservationCounts.end(); ++i0) {
    for(auto i1=coObservationCounts.at(i0->first).begin();
        i1!=coObservationCounts.at(i0->first).end(); ++i1) {
      if(!graph.states_.count(StateId(i0->first))) continue;
      if(!graph.states_.count(StateId(i1->first))) continue;
      const Eigen::Vector3d p0 = graph.states_.at(StateId(i0->first)).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(StateId(i1->first)).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 50.0 + std::min(205, i1->second*2);
      cv::line(image, cvPos0, cvPos1, cv::Scalar(0,brightness, brightness), 1, cv::LINE_AA);
    }
  }

  // draw old frames
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);

    // draw pose graph error
    for(auto piter=iter->second.twoPoseLinks.begin();
        piter != iter->second.twoPoseLinks.end(); piter++) {
      auto refId = piter->second.errorTerm->referencePoseId();
      auto otherId = piter->second.errorTerm->otherPoseId();
      if(refId<otherId) {
        // draw only forward.
        const Eigen::Vector3d p0 = graph.states_.at(refId).pose->estimate().r();
        const Eigen::Vector3d p1 = graph.states_.at(otherId).pose->estimate().r();
        const Eigen::Vector3d pos0 = (p0-centre)*scale;
        const Eigen::Vector3d pos1 = (p1-centre)*scale;
        cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
        cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
        double brightness = 50.0 + std::min(205.0, piter->second.errorTerm->strength());
        cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2,brightness, brightness), 3,
                 cv::LINE_AA);
      }
    }
    // draw pose graph error const
    for(auto piter=iter->second.twoPoseConstLinks.begin(); piter !=
        iter->second.twoPoseConstLinks.end(); piter++) {
      const Eigen::Vector3d p0 = graph.states_.at(iter->first).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(piter->first).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 200;
      cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2.0,brightness, brightness/2.0), 3,
               cv::LINE_AA);
    }
  }

  // draw frames
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,0,0);
    auto colour2 = cv::Scalar(255,255,255);
    if(!iter->second.isKeyframe) {
      colour1 = cv::Scalar(127,127,127);
    }
    if(idx<=2) {
        if(keyFrames_.count(iter->first)) {
          colour1 = cv::Scalar(255,255,0);
        }
        const bool isPoseGraphFrame = auxiliaryStates_.at(iter->first).isPoseGraphFrame;
        if(isPoseGraphFrame) {
          colour2 = cv::Scalar(127,127,127);
          if(iter->second.pose->fixed()) {
            colour2 = cv::Scalar(25,25,200);
          }
        }
    }

    // draw IMU error
    if(iter->second.nextImuLink.errorTerm) {
      auto iterNext = iter;
      iterNext++;
      const Eigen::Vector3d nextPos = (iterNext->second.pose->estimate().r()-centre)*scale;
      cv::Point2d nextCvPos(nextPos[0]+image.cols*0.5, -nextPos[1]+image.rows*0.5);
      cv::line(image, cvPos, nextCvPos, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    // draw point
    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour2, 1, cv::LINE_AA);
    std::stringstream stream;
    stream << iter->first.value();
    cv::putText(image, stream.str(), cvPos+cv::Point2d(6,3),
                cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  }

  if(idx >2) {
      // some text:
      auto T_WS = graph.states_.rbegin()->second.pose->estimate();
      const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
      std::stringstream postext;
      postext << "position = ["
              << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
      cv::putText(image, postext.str(), cv::Point(15,15),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream veltext;
      veltext << "velocity = ["
              << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
      cv::putText(image, veltext.str(), cv::Point(15,35),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream gyroBiasText;
      gyroBiasText << "gyro bias = ["
                   << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
      cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream accBiasText;
      accBiasText << "acc bias = ["
                  << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
      cv::putText(image, accBiasText.str(), cv::Point(15,75),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      return;
  }

  // check loopClosureFrame co-observability
  std::map<StateId, size_t> counts;
  for(auto iter = loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    counts[*iter] = 0;
  }
  const StateId cId = currentStateId();
  for (auto pit = graph.landmarks_.begin(); pit != graph.landmarks_.end(); ++pit) {
    const auto& residuals = pit->second.observations;
    bool isObservedInNewFrame = false;
    std::set<uint64_t> loopClosureFramesObserved;
    for (const auto& r : residuals) {
      uint64_t poseId = r.first.frameId;
      if(cId.value() == poseId) {
        isObservedInNewFrame = true;
      }
      if(counts.find(StateId(poseId)) != counts.end()) {
        loopClosureFramesObserved.insert(poseId);
      }
    }

    if(isObservedInNewFrame) {
      for(auto iter = loopClosureFramesObserved.begin(); iter!=loopClosureFramesObserved.end();
          ++iter) {
        counts.at(StateId(*iter))++;
      }
    }
  }

  // draw loop closure frames
  for(auto iter=loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    const Eigen::Vector3d pos = (graph.states_.at(*iter).pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,255);
    auto colour2 = cv::Scalar(255,0,0);

    // draw point
    cv::circle(image, cvPos, 5, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, int(6.0f+float(counts.at(*iter))/10.0f), colour2,
               int(1+counts.at(*iter))/5, cv::LINE_AA);
  }

  // current position always on top
  auto T_WS = graph.states_.rbegin()->second.pose->estimate();
  const Eigen::Vector3d pos = (T_WS.r()-centre)*scale;
  cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
  auto colour1 = cv::Scalar(127,127,127);
  auto colour2 = cv::Scalar(255,255,255);
  if(graph.states_.rbegin()->second.isKeyframe) {
    colour1 = cv::Scalar(255,255,0);
  }
  cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
  cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);

  // current keyframe
  StateId kfId = currentKeyframeStateId();
  if(kfId.isInitialised()) {
    kinematics::Transformation T_WS_kf = graph.states_.at(kfId).pose->estimate();
    const Eigen::Vector3d pos = (T_WS_kf.r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,127);
    auto colour2 = cv::Scalar(255,255,255);

    cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
  }

  // some text:
  const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
  std::stringstream postext;
  postext << "position = [" << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
  cv::putText(image, postext.str(), cv::Point(15,15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream veltext;
  veltext << "velocity = ["
          << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
  cv::putText(image, veltext.str(), cv::Point(15,35),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream gyroBiasText;
  gyroBiasText << "gyro bias = ["
               << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
  cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream accBiasText;
  accBiasText << "acc bias = ["
              << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
  cv::putText(image, accBiasText.str(), cv::Point(15,75),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}

bool ViSlamBackend::getObservedIds(StateId id, std::set<StateId> &observedIds) const
{
  auto iter = realtimeGraph_.states_.find(id);
  if (iter == realtimeGraph_.states_.end()) {
    return false;
  }

  // insert from pose graph edges
  const auto &state = iter->second;
  for (const auto &link : state.twoPoseLinks) {
    observedIds.insert(link.second.state0);
    observedIds.insert(link.second.state1);
  }

  // insert currently co-visible
  if (keyFrames_.count(id) || imuFrames_.count(id) || loopClosureFrames_.count(id)) {
    std::set<LandmarkId> landmarks;
    for (const auto &obs : state.observations) {
      landmarks.insert(obs.second.landmarkId);
    }
    for (const auto &lmId : landmarks) {
      for (const auto &obs : realtimeGraph_.landmarks_.at(lmId).observations) {
        observedIds.insert(StateId(obs.first.frameId));
      }
    }
  }

  // insert frames connected by IMU (if available)
  auto next = iter;
  next++;
  if (next != realtimeGraph_.states_.end()) {
    observedIds.insert(next->first);
  }
  if (iter != realtimeGraph_.states_.begin()) {
    auto previous = iter;
    previous--;
    observedIds.insert(previous->first);
  }

  return true;
}

bool ViSlamBackend::isLoopClosureFrame(StateId frameId) const {
  return loopClosureFrames_.count(frameId) > 0;
}

bool ViSlamBackend::isRecentLoopClosureFrame(StateId frameId) const {
  for(auto id : keyFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  for(auto id : loopClosureFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  return false;
}

void ViSlamBackend::addLoopClosureFrame(StateId loopClosureFrameId,
                                        std::set<LandmarkId> & loopClosureLandmarks,
                                        bool skipFullGraphOptimisation)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_,
                    "trying to add loop closure while loop closing")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "trying to add loop closure while loop closing")

  // remember loop closure frame
  if(loopClosureFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add previously added loop-closure frame";
    return;
  }
  if(keyFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add current keyframe as loop-closure frame" << std::endl;
    return;
  }

  loopClosureFrames_.insert(loopClosureFrameId); // sort/remove later
  currentLoopClosureFrames_.insert(loopClosureFrameId);

  // undo pose graph errors into observations
  std::set<StateId> connectedStates;
  OKVIS_ASSERT_TRUE_DBG(Exception, auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame,
                    "must be pose graph frame")
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(
        loopClosureFrameId, &loopClosureLandmarks, &connectedStates, &allObservations);
  for(auto lm : loopClosureLandmarks) {
    const auto & landmark = realtimeGraph_.landmarks_.at(lm);
    if(!fullGraph_.landmarkExists(lm)) {
      fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
      fullGraph_.setLandmarkQuality(lm, landmark.quality);
    }
  }
  fullGraph_.removeTwoPoseConstLinks(loopClosureFrameId);
  for(const auto& obs : allObservations) {
    const bool useCauchy = obs.lossFunction != nullptr;
    LandmarkId landmarkId(obs.hPoint->id());
    fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                           obs.keypointIdentifier, useCauchy);
  }

  // add also connected frames to loop closure frames
  // flag no longer pose graph frames
  for(auto connectedState : connectedStates) {
    if(auxiliaryStates_.at(connectedState).isPoseGraphFrame
       && keyFrames_.count(connectedState)==0) {
      loopClosureFrames_.insert(connectedState);
      currentLoopClosureFrames_.insert(connectedState);
      auxiliaryStates_.at(connectedState).isPoseGraphFrame = false;
    } // else already in active window and considered
  }
  auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame = false;

  // remember oldest Id / freeze / unfreeze
  if(!loopClosureFrames_.empty()) {
    StateId oldestId = loopClosureFrameId;
    if(fullGraph_.states_.count(oldestId)) {
      StateId oldestIdToSetVariable = auxiliaryStates_.at(oldestId).loopId;
      // also remember this throughout the created loop
      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable);
          iter != auxiliaryStates_.end(); ++iter) {
        iter->second.loopId = oldestIdToSetVariable;
      }

      // apply the freeze/unfreeze
      Time oldestT = fullGraph_.states_.at(oldestId).timestamp;
      int ctr = 0;
      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable); ; --iter) {
        if(ctr == numPoseGraphFrames || iter==auxiliaryStates_.begin()) {
          while((oldestT - fullGraph_.timestamp(iter->first)).toSec()<minDeltaT) {
            if(iter==auxiliaryStates_.begin()) {
              break;
            }
            --iter;
          }

          if (lastFreeze_.isInitialised()) {
            // ensure we go back at least as far as lastFreeze
            if (iter->first > lastFreeze_) {
              while (iter->first > lastFreeze_) {
                if (iter == auxiliaryStates_.begin()) {
                  break;
                }
                --iter;
              }
            }
          }

          fullGraph_.unfreezePosesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezePosesUntil(iter->first);
          }
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezeSpeedAndBiasesUntil(iter->first);
          }
          break;
        }
        if(iter==auxiliaryStates_.begin()) {
          fullGraph_.unfreezePosesFrom(iter->first);
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          break;
        }
        ctr++;
      }
    }
  }

  // remember as recent loop closures
  for(auto id : keyFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }
  for(auto id : loopClosureFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }

  // signal we need optimisation
  if(!skipFullGraphOptimisation) {
    needsFullGraphOptimisation_ = true;
  }
}

void ViSlamBackend::addGpsAlignmentFrame(StateId gpsLossFrameId) {

  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "trying to add gps loop closure while loop closing")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "trying to add gps loop closure while loop closing")
  LOG(INFO) << "[GPS] Kicking off full graph optimisation due to gps loop closure, starting at state id = " << gpsLossFrameId.value()
            << " up to current state id  = " << fullGraph_.states_.rbegin()->first.value();

  // remember oldest Id / freeze / unfreeze
  StateId oldestIdToSetVariable = gpsLossFrameId;

  Time oldestT = fullGraph_.states_.at(oldestIdToSetVariable).timestamp;
  int ctr = 0;

  //
  for(auto iter = fullGraph_.states_.find(oldestIdToSetVariable) ; ; --iter){
      if(ctr == numPoseGraphFrames || iter == fullGraph_.states_.begin()){
          while((oldestT - fullGraph_.timestamp(iter->first)).toSec()<minDeltaT) {
            if(iter==fullGraph_.states_.begin()) {
              break;
            }
            --iter;
          }
          fullGraph_.unfreezePosesFrom(iter->first);
          if(iter!=fullGraph_.states_.begin()) {
            fullGraph_.freezePosesUntil(iter->first);
          }
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          if(iter!=fullGraph_.states_.begin()) {
            fullGraph_.freezeSpeedAndBiasesUntil(iter->first);
          }
          break;
      }
      if(iter==fullGraph_.states_.begin()) {
        fullGraph_.unfreezePosesFrom(iter->first);
        fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
        break;
      }
      ctr++; // count states that will be optimised during global alignment
  }

  // signal we need optimisation
  needsFullGraphOptimisation_ = true;
}

bool ViSlamBackend::synchroniseRealtimeAndFullGraph(std::vector<StateId> &updatedStates)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "cannot synchronise while loop-closing")
  realtimeTrustRegionRadius_.reset();

  // first, we move the loop-closure frames into the set of regular keyframes...
  keyFrames_.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  loopClosureFrames_.clear();

  // remove landmarks in full graph that have disappeared
  auto landmarks = fullGraph_.landmarks_;
  for(auto iter = landmarks.begin(); iter != landmarks.end(); ++iter) {
    if(realtimeGraph_.landmarks_.count(iter->first) == 0) {
      fullGraph_.removeLandmark(iter->first);
    }
  }

  // compute pose change
  StateId oldFrameId;
  for(auto iter = fullGraph_.states_.crbegin(); iter != fullGraph_.states_.crend(); ++iter) {
    if(realtimeGraph_.states_.count(iter->first)) {
      oldFrameId = iter->first;
      break;
    }
  }

  const kinematics::Transformation T_WS_old = realtimeGraph_.pose(oldFrameId);
  const kinematics::Transformation T_WS_new = fullGraph_.pose(oldFrameId);
  const kinematics::Transformation T_Wnew_Wold = T_WS_new * T_WS_old.inverse();

  // process added new states
  for(const auto& addState : addStatesBacklog_) {
    bool asKeyframe = false;
    const bool existsInRealtimeGraph = realtimeGraph_.states_.count(addState.id) !=0;
    if(existsInRealtimeGraph) {
      if(realtimeGraph_.states_.at(addState.id).isKeyframe) {
        asKeyframe = true;
      }
    }
    fullGraph_.addStatesPropagate(addState.timestamp, addState.imuMeasurements, asKeyframe);
    kinematics::Transformation T_WS;
    SpeedAndBias speedAndBias;
    if(!existsInRealtimeGraph) {
      const auto& anystate = realtimeGraph_.anyState_.at(addState.id);
      const kinematics::Transformation T_Sk_S = anystate.T_Sk_S;
      const kinematics::Transformation T_WSk =
          T_Wnew_Wold*realtimeGraph_.states_.at(anystate.keyframeId).pose->estimate();
      T_WS = T_WSk*T_Sk_S;
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias.head<3>() = T_WSk.C() * anystate.v_Sk;
    } else {
      const kinematics::Transformation T_S0S1 =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse()*
          realtimeGraph_.states_.at(addState.id).pose->estimate();
      T_WS = fullGraph_.states_.at(oldFrameId).pose->estimate()*T_S0S1;
      const SpeedAndBias speedAndBias_old =
          realtimeGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      const Eigen::Vector3d v_S =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse().C()
          *speedAndBias_old.head<3>();
      speedAndBias.head<3>() = T_WS.C()*v_S;
    }
    fullGraph_.setPose(addState.id, T_WS);
    fullGraph_.setSpeedAndBias(addState.id, speedAndBias);
  }
  addStatesBacklog_.clear();

  for(const auto& eliminateState : eliminateStates_) {
    fullGraph_.removeAllObservations(eliminateState.first);
    /// \todo make more efficient (copy over)
    fullGraph_.eliminateStateByImuMerge(eliminateState.first, eliminateState.second);
    /// \todo make more efficient (copy over)
  }
  eliminateStates_.clear();

  for(const auto& eliminateState : eliminateStates_) {
    const auto iter = fullGraph_.states_.find(eliminateState.second);
    if(iter != fullGraph_.states_.end()) {
      if(iter->second.previousImuLink.errorTerm) {
        if (fullGraph_.imuParametersVec_.at(0).use && fullGraph_.imuParametersVec_.at(0).use_preintegration_residual) {
          std::static_pointer_cast<ceres::ImuError>(iter->second.previousImuLink.errorTerm)
            ->syncFrom(*std::static_pointer_cast<ceres::ImuError>(
              realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        } else {
          std::static_pointer_cast<ceres::PseudoImuError>(iter->second.previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        }
      }
    }
  }
  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {
      break;
    }
    auto & errorTerm = realtimeGraph_.states_.at(riter->first).previousImuLink.errorTerm;
    if(errorTerm) {
      if (fullGraph_.imuParametersVec_.at(0).use && fullGraph_.imuParametersVec_.at(0).use_preintegration_residual) {
        std::static_pointer_cast<ceres::ImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::PseudoImuError>(riter->second.previousImuLink.errorTerm));
      }
    }
  }

  // update new landmarks with pose change and insert into full graph
  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
      ++iter) {
    if(fullGraph_.landmarks_.count(iter->first) == 0) {
      realtimeGraph_.setLandmark(iter->first, T_Wnew_Wold * iter->second.hPoint->estimate());
      fullGraph_.addLandmark(
            iter->first, iter->second.hPoint->estimate(), iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  // ----- gps stuff begin -----
  // Process buffered gps measurements
  for(auto addGpsMeas : addGpsBacklog_){
      bool stillExistsInRealtimeGraph = fullGraph_.states_.count(addGpsMeas.id) !=0;
      if(stillExistsInRealtimeGraph) {
          if(addGpsMeas.reInitFlag){
            fullGraph_.reInitGpsExtrinsics();
            }
          fullGraph_.addGpsMeasurement(addGpsMeas.id, addGpsMeas.gpsMeasurement, addGpsMeas.imuMeasurements);
      }
  }
  addGpsBacklog_.clear();
  // only have to do sth if fullGraph_.T_GW has not yet been fixed
  if(!fullGraph_.isGpsFixed()){
      const kinematics::Transformation T_GW_new = fullGraph_.T_GW(); // fullGraph Optimisation Result
      // copy results
      realtimeGraph_.setGpsExtrinsics(T_GW_new);
      if(gpsObservability_){ // copy results and FIX
          LOG(INFO) << "[GPS] Fixing T_GW: \n" << fullGraph_.T_GW().T3x4();
          realtimeGraph_.setGpsExtrinsics(T_GW_new);
          fullGraph_.freezeGpsExtrinsics();
          realtimeGraph_.freezeGpsExtrinsics();
      }
  }
  // ----- gps stuff end -----

  // ----- Submap Alignment Begin -----
  for(auto alignmentTerm : addSubmapAlignmentBacklog_){
    bool stillExistsInRealtimeGraph = fullGraph_.states_.count(StateId(alignmentTerm.frame_B_id)) !=0;
    if(stillExistsInRealtimeGraph) {
      DLOG(INFO) << "reversely adding measurements to fullGraph at state " << alignmentTerm.frame_B_id
                << " and submap at " << alignmentTerm.frame_A_id << std::endl;
      fullGraph_.addSubmapAlignmentConstraints(alignmentTerm.submap_ptr, alignmentTerm.frame_A_id,
                                               alignmentTerm.frame_B_id, alignmentTerm.pointCloud,
                                               alignmentTerm.sensorError, alignmentTerm.isLidar,
                                               alignmentTerm.robustFunction);
    }
  }
  addSubmapAlignmentBacklog_.clear();
  // ----- Submap Alignment End -----

  // copy the result over now
  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {
      /// \todo remove these for safety
      OKVIS_ASSERT_TRUE(Exception, riter->second.pose->estimate().T()
                        ==realtimeGraph_.pose(riter->first).T(), "O-O")
      OKVIS_ASSERT_TRUE(Exception, riter->second.speedAndBias->estimate()
                        ==realtimeGraph_.speedAndBias(riter->first), "O-O")
      break;
    } else {
      updatedStates.push_back(riter->first);
    }
    if(realtimeGraph_.states_.count(riter->first) == 0) {
      OKVIS_THROW(Exception, "impossible: state not present")
      // new state not yet added
      continue;
    }
    realtimeGraph_.setPose(riter->first, riter->second.pose->estimate());
    realtimeGraph_.setSpeedAndBias(riter->first, riter->second.speedAndBias->estimate());
    for(size_t i = 0; i<riter->second.extrinsics.size(); ++i) {
      if(!riter->second.extrinsics.at(i)->fixed()) {
        realtimeGraph_.setExtrinsics(riter->first, uchar(i),
                                     riter->second.extrinsics.at(i)->estimate());
      }
    }
  }

  // update landmarks
  for(auto iter = fullGraph_.landmarks_.begin(); iter != fullGraph_.landmarks_.end(); ++iter) {
    OKVIS_ASSERT_TRUE_DBG(Exception, realtimeGraph_.landmarkExists(iter->first), "not allowed")
    realtimeGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                               iter->second.hPoint->initialized());
    realtimeGraph_.setLandmarkQuality(iter->first, iter->second.quality);
  }

  // process touched landmarks/observations
  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmark full vs realtime graph")
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }
    // remove all
    auto observations = fullGraph_.landmarks_.at(lm).observations;
    for(const auto & obs : observations) {
      fullGraph_.removeObservation(obs.first);
    }
  }
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }
    // copy over
    for(const auto & obs : realtimeGraph_.landmarks_.at(lm).observations) {
      fullGraph_.addExternalObservation(obs.second.errorTerm, lm, obs.first, true);
    }
  }

  // remove all other edges
  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; // later deleted
    }
    auto relativePoselinks = fullGraph_.states_.at(stateId).relativePoseLinks;
    for(const auto& link : relativePoselinks) {
      if(isPersistentFullGraphRelativePoseConstraint(link.second.state0, link.second.state1)
         || isProtectedLoopPathRelativePoseConstraint(link.second.state0, link.second.state1)) {
        continue;
      }
      fullGraph_.removeRelativePoseConstraint(link.second.state0, link.second.state1);
    }
    auto twoPoselinks = fullGraph_.states_.at(stateId).twoPoseConstLinks;
    for(const auto& link : twoPoselinks) {
      fullGraph_.removeTwoPoseConstLink(link.second.state0, link.second.state1);
    }
  }

  // re-add
  std::set<::ceres::ResidualBlockId> addedRelativePoseLinks;
  std::set<::ceres::ResidualBlockId> addedTwoPoseLinks;
  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; // later deleted
    }
    for(const auto& relPoseLink : realtimeGraph_.states_.at(stateId).relativePoseLinks) {
      if(addedRelativePoseLinks.count(relPoseLink.second.residualBlockId)==0) {
        const StateId state0 = relPoseLink.second.state0;
        const StateId state1 = relPoseLink.second.state1;
        if(fullGraph_.states_.count(state0) == 0 || fullGraph_.states_.count(state1) == 0) {
          addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
          continue;
        }
        if(isSuppressedLoopPathRelativePoseConstraint(state0, state1)
           && !isPersistentFullGraphRelativePoseConstraint(state0, state1)
           && !isProtectedLoopPathRelativePoseConstraint(state0, state1)) {
          addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
          continue;
        }
        if(fullGraph_.states_.at(state0).relativePoseLinks.count(state1)) {
          if(isPersistentFullGraphRelativePoseConstraint(state0, state1)
             || isProtectedLoopPathRelativePoseConstraint(state0, state1)) {
            addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
            continue;
          }
          fullGraph_.removeRelativePoseConstraint(state0, state1);
        }
        fullGraph_.addRelativePoseConstraint(
              state0, state1,
              relPoseLink.second.errorTerm->T_AB(),
              relPoseLink.second.errorTerm->information());
        addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
      }
    }
    for(const auto& twoPoseLink : realtimeGraph_.states_.at(stateId).twoPoseLinks) {
      if(addedTwoPoseLinks.count(twoPoseLink.second.residualBlockId)==0) {
        if(isSuppressedLoopPathTwoPoseConstraint(
               twoPoseLink.second.state0, twoPoseLink.second.state1)) {
          addedTwoPoseLinks.insert(twoPoseLink.second.residualBlockId);
          continue;
        }
        fullGraph_.addExternalTwoPoseLink(
              twoPoseLink.second.errorTerm->cloneTwoPoseGraphErrorConst(),
              twoPoseLink.second.state0, twoPoseLink.second.state1);
        addedTwoPoseLinks.insert(twoPoseLink.second.residualBlockId);
      }
    }
  }
  reapplyPersistentLoopPathRelativePoseConstraints(true, true);
  reapplyPersistentFullGraphRelativePoseConstraints();
  touchedStates_.clear();
  touchedLandmarks_.clear();

  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmarks full vs realtime graph")
  OKVIS_ASSERT_TRUE(Exception, fullGraph_.states_.size() == realtimeGraph_.states_.size(),
                    "inconsistent states full vs realtime graph")

  /*if(!realtimeGraph_.isSynched(fullGraph_)) {
    std::cout << "not synched" << std::endl;
    OKVIS_THROW(Exception, "not synched")
  }*/

  // The solve has finished. Reapply after clone/merge so both old and newly
  // inserted edges use the latest frame request; suppressed paths remain zero.
  {
    std::lock_guard<std::mutex> weightLock(fullGraphTwoPoseWeightMutex_);
    const double weightScale = requestedVisualTwoPoseWeightScale_.load();
    scaleVisualTwoPoseGraphErrors(realtimeGraph_, weightScale);
    scaleVisualTwoPoseGraphErrors(fullGraph_, weightScale);
  }

  // loop closure processed
  currentLoopClosureFrames_.clear();
  isLoopClosureAvailable_ = false;

  return true;
}

int ViSlamBackend::cleanUnobservedLandmarks() {
  // check consistency -- currently disabled
  //if (!isLoopClosing_ && !isLoopClosureAvailable_) {
  //  OKVIS_ASSERT_TRUE(Exception,
  //                    realtimeGraph_.isSynched(fullGraph_),
  //                    "trying to clean unobserved landmarks while loop closing")
  //}
  std::map<LandmarkId, std::set<KeypointIdentifier>> removed;
  int removed1 = realtimeGraph_.cleanUnobservedLandmarks(&removed);
  for(const auto & rem : removed) {
    for(const auto & obs : rem.second) {
      // note: it can happen (rarely) that a landmark gets cleaned but then re-added,
      // so the respective frame observations might be missed in the synchronisation.
      multiFrame(StateId(obs.frameId))->setLandmarkId(obs.cameraIndex, obs.keypointIndex, 0);
    }
  }
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & rem : removed) {
      touchedLandmarks_.insert(rem.first);
      for(auto obs : rem.second) {
        // note: it can happen (rarely) that a landmark gets cleaned but then re-added,
        // so the respective frame observations might be missed in the synchronisation.
        touchedStates_.insert(StateId(obs.frameId));
      }
    }
    return removed1; /// \todo This can be done with some refactoring
  } else {
    int removed0 = fullGraph_.cleanUnobservedLandmarks();
    OKVIS_ASSERT_TRUE(Exception, removed0 == removed1, "landmarks cleaned inconsistent!")
    return removed1;
  }

}

bool ViSlamBackend::mergeLandmark(const LandmarkId &fromId, const LandmarkId &intoId)
{
  bool success = (realtimeGraph_.mergeLandmark(fromId, intoId, multiFrames_));
  // also reset associated keypoints
  auto observations = realtimeGraph_.landmarks_.at(intoId).observations;
  for (const auto &observation : observations) {
    multiFrames_.at(StateId(observation.first.frameId))
      ->setLandmarkId(observation.first.cameraIndex,
                      observation.first.keypointIndex,
                      intoId.value());
  }

  if (isLoopClosing_ || isLoopClosureAvailable_) {
    for (const auto &obs : observations) {
      touchedStates_.insert(StateId(obs.first.frameId));
    }
    touchedLandmarks_.insert(fromId);
    touchedLandmarks_.insert(intoId);
  } else {
    // now merge
    success &= fullGraph_.mergeLandmark(fromId, intoId, multiFrames_);
  }

  return success;
}

int ViSlamBackend::mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, fromIds.size() == intoIds.size(), "vectors must be same lengths")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "loop closure not finished, cannot merge landmarks")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  std::map<LandmarkId, LandmarkId> changes; // keep track of changes
  int ctr = 0;
  for(size_t i = 0; i < fromIds.size(); ++i) {
    // check if the fromId hasn't already been changed
    while(changes.count(fromIds.at(i))) {
      fromIds.at(i) = changes.at(fromIds.at(i));
    }
    // check if the intoId hasn't already been changed
    while(changes.count(intoIds.at(i))) {
      intoIds.at(i) = changes.at(intoIds.at(i));
    }
    // check if the change hasn't been indirectly applied already
    if(fromIds.at(i) == intoIds.at(i)) {
      continue; //this has already been done.
    }

    // now merge
    if(realtimeGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_)) {
      ctr++;
    }
    fullGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_);
    changes[fromIds.at(i)] = intoIds.at(i);

    // also reset associated keypoints
    auto observations = realtimeGraph_.landmarks_.at(intoIds.at(i)).observations;
    for(const auto & observation : observations) {
      multiFrames_.at(StateId(observation.first.frameId))->setLandmarkId(
            observation.first.cameraIndex, observation.first.keypointIndex, intoIds.at(i).value());
    }
  }

  return ctr;
}

void ViSlamBackend::optimiseFullGraph(int numIter, ::ceres::Solver::Summary &summary,
                                      int numThreads, bool verbose, double functionTolerance)
{
  OKVIS_ASSERT_TRUE(Exception, std::isfinite(functionTolerance)
                    && functionTolerance >= 0.0 && functionTolerance < 1.0,
                    "Full graph function tolerance must be finite and in [0, 1)")
  // Freeze visual two-pose weights for this solve, including direct final-BA calls.
  // Frame processing uses try_lock and retains the latest request without waiting.
  std::unique_lock<std::mutex> weightLock(fullGraphTwoPoseWeightMutex_);
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "can't optimise posegraph; already loop-closing")
  DLOG(INFO) << "Starting fullGraph_ optimisation.";
  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = true;
  // Also consume requests deferred by try_lock (which may fail spuriously).
  // Later requests are cached for synchronization, never applied during Solve.
  scaleVisualTwoPoseGraphErrors(fullGraph_, requestedVisualTwoPoseWeightScale_.load());
  if(!fullGraphRelativePoseConstraints_.empty()) {
    AlignedVector<RelPoseInfo> temporaryFullGraphConstraints;
    for(auto & item : fullGraphRelativePoseConstraints_) {
      if(fullGraph_.states_.count(item.pose_i) == 0
         || fullGraph_.states_.count(item.pose_j) == 0) {
        continue;
      }
      if(fullGraph_.states_.at(item.pose_i).relativePoseLinks.count(item.pose_j)) {
        fullGraph_.removeRelativePoseConstraint(item.pose_i, item.pose_j);
      }
      const double informationScale =
          std::isfinite(item.informationScale) && item.informationScale > 0.0
              ? item.informationScale
              : 100.0;
      fullGraph_.addRelativePoseConstraint(item.pose_i, item.pose_j, item.T_Si_Sj,
                                           informationScale * item.information);
      if(item.keepInFullGraph) {
        const double persistentInformationScale =
            std::isfinite(item.persistentInformationScale)
            && item.persistentInformationScale > 0.0
                ? item.persistentInformationScale
                : informationScale;
        addOrUpdatePersistentFullGraphRelativePoseConstraint(item);
        LOG(INFO) << "Keeping persistent loop-closure relative pose constraint "
                  << item.pose_i.value() << " -> " << item.pose_j.value()
                  << " in full graph with initial_information_scale=" << informationScale
                  << ", persistent_information_scale=" << persistentInformationScale << ".";
      } else {
        temporaryFullGraphConstraints.push_back(item);
      }
    }
    fullGraph_.options().function_tolerance = 0.001;
    reapplyPersistentLoopPathRelativePoseConstraints(false, true);
    fullGraph_.optimise(numIter/3, numThreads, verbose);
    for(auto & item : temporaryFullGraphConstraints) {
      if(fullGraph_.states_.count(item.pose_i) != 0
         && fullGraph_.states_.at(item.pose_i).relativePoseLinks.count(item.pose_j)) {
        fullGraph_.removeRelativePoseConstraint(item.pose_i, item.pose_j);
      }
    }
    fullGraphRelativePoseConstraints_.clear();
  }
  reapplyPersistentLoopPathRelativePoseConstraints(false, true);
  reapplyPersistentFullGraphRelativePoseConstraints();
  fullGraph_.options().function_tolerance = functionTolerance;

  fullGraph_.optimise(numIter, numThreads, verbose);
  summary = fullGraph_.summary();
  fullGraph_.updateLandmarks(); /// \todo check to do better

  // Unlock before publishing completion: synchronization may now import results
  // and apply deferred weights, but never while Ceres is evaluating these errors.
  weightLock.unlock();
  isLoopClosureAvailable_ = true;
  isLoopClosing_ = false;
}

void ViSlamBackend::doFinalBa(
    int numIter, ::ceres::Solver::Summary &summary,
    std::set<StateId> & updatedStatesBa, double extrinsicsPositionUncertainty,
    double extrinsicsOrientationUncertainty, int numThreads, bool verbose,
    double functionTolerance)
{
  OKVIS_ASSERT_TRUE(Exception, numIter > 0, "Final BA iteration cap must be positive")
  OKVIS_ASSERT_TRUE(Exception, std::isfinite(functionTolerance)
                    && functionTolerance >= 0.0 && functionTolerance < 1.0,
                    "Final BA function tolerance must be finite and in [0, 1)")
  // Final BA can use a stricter stopping criterion without changing background
  // or realtime solves. Restore the prior graph option even on an exception.
  struct RestoreFunctionTolerance {
    ::ceres::Solver::Options& options;
    double previous;
    ~RestoreFunctionTolerance() { options.function_tolerance = previous; }
  } restoreTolerance{fullGraph_.options_, fullGraph_.options_.function_tolerance};
  fullGraph_.options_.function_tolerance = functionTolerance;

  // convert all posegraph edges
  int i=0;
  for(const auto & state : fullGraph_.states_) {
    if(state.second.twoPoseConstLinks.size()>0) {
      if(keyFrames_.count(state.first) == 0 && loopClosureFrames_.count(state.first) == 0) {
        keyFrames_.insert(state.first);
      }
      std::cout << "\rConstructing VI-BA problem... "
                << std::round(10000.0*double(i)/double(fullGraph_.states_.size()))/100.0 << "%";
      expandKeyframe(state.first);
      i++;
    }
  }
  fullGraph_.cleanUnobservedLandmarks();
  std::cout << "\rConstructing VI-BA problem... " << "100.00%" << std::endl;

  // unfreeze
  fullGraph_.unfreezePosesFrom(StateId(1));
  fullGraph_.unfreezeSpeedAndBiasesFrom(StateId(1));

  // Disabling final calibration must also fix extrinsics that online
  // calibration left variable. Preserve their current estimates in both passes.
  const bool optimiseExtrinsics = extrinsicsPositionUncertainty > 0.0
      && extrinsicsOrientationUncertainty > 0.0;
  if(!optimiseExtrinsics) {
    fullGraph_.setExtrinsicsFixed();
  }

  // unfreeze GPS extrinsics
  if(!fullGraph_.gpsParametersVec_.empty()){
    fullGraph_.unfreezeGpsExtrinsics();
  }

  // make sure IMU errors are reintegrated if needed
  ceres::ImuError::redoPropagationAlways = true;

  // optimise
  std::cout << "Running optimisation (verbose="
            << (verbose?"true)":"false)...") << std::endl;
  optimiseFullGraph(numIter, summary,numThreads, verbose, functionTolerance);

  // Release initialization priors, but keep disabled biases anchored at g0/a0.
  // Near-zero random walk constrains differences only; without this absolute
  // anchor, final BA could move all nominally fixed biases together.
  fullGraph_.removeSpeedAndBiasPriorExceptFixedBiases(StateId(1));

  // remove extrinsics fixation
  if(optimiseExtrinsics) {
    std::cout << "Running optimisation again without priors & fixation (verbose="
              << (verbose?"true)":"false)...") << std::endl;
    fullGraph_.setExtrinsicsVariable();
    fullGraph_.softConstrainExtrinsics(
          extrinsicsPositionUncertainty, extrinsicsOrientationUncertainty);
  } else {
    std::cout << "Running optimisation again without priors (verbose="
              << (verbose?"true)":"false)...") << std::endl;
  }

  // optimise
  optimiseFullGraph(numIter, summary,numThreads, verbose, functionTolerance);

  cv::Mat img=cv::Mat::zeros(2000,2000,CV_8UC3);
  drawOverheadImage(img, 2);
  cv::imwrite("fullBa.png", img);

  // print
  const auto & state = fullGraph_.states_.rbegin()->second;
  for(size_t i = 0; i < state.extrinsics.size(); ++i) {
    std::cout << "extrinsics T_SC" << i << ":" << std::endl;
    std::cout << std::setprecision(15) << state.extrinsics.at(i)->estimate().T() << std::endl;
  }
  Eigen::Matrix<double,6,1> biases = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    biases += s.second.speedAndBias->estimate().tail<6>();
  }
  biases = (1.0/double(fullGraph_.states_.size()))*biases;
  std::cout << "biases: " << std::setprecision(15) << biases.transpose() << std::endl;
  Eigen::Matrix<double,6,1> biasesStd = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    Eigen::Matrix<double,6,1> diff = (s.second.speedAndBias->estimate().tail<6>()-biases);
    biasesStd += diff.cwiseAbs2();
  }
  biasesStd /= double(fullGraph_.states_.size());  // population variance across states
  std::cout << "biases stdev: " << biasesStd.cwiseSqrt().transpose() << std::endl;

  // Prepare histogram plotting: < -5 | -5 < -4 | ... | -1 < 0 | 0 < 1 | ... | > 5 => 12 bins
  // index: floor(x) + 6
  // save per camera
  std::vector<Eigen::Matrix<int, 12, 1>, Eigen::aligned_allocator<Eigen::Matrix<int, 12, 1>>> errorHistograms;
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    Eigen::Matrix<int, 12, 1> errHist;
    errHist.setZero();
    errorHistograms.push_back(errHist);
  }

  // some plotting (currently disabled)
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> outliers;
  std::vector<int> inlierCtrs;
  std::vector<int> outlierCtrs;
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    inlierCtrs.push_back(0);
    outlierCtrs.push_back(0);
    images.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
    outliers.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
  }

  for(const auto & obs : fullGraph_.observations_) {
    auto frame = multiFrames_.at(StateId(obs.first.frameId));
    cv::KeyPoint kpt;
    frame->getCvKeypoint(obs.first.cameraIndex, obs.first.keypointIndex, kpt);
    double* params[3];
    params[0] = fullGraph_.states_.at(StateId(obs.first.frameId)).pose->parameters();
    params[1] = fullGraph_.landmarks_.at(obs.second.landmarkId).hPoint->parameters();
    params[2] = fullGraph_.states_.at(StateId(obs.first.frameId)).extrinsics.at(
          obs.first.cameraIndex)->parameters();
    Eigen::Vector2d err;
    obs.second.errorTerm->Evaluate(params, err.data(), nullptr);
    double badness = sqrt(err.transpose()*err)/3.0; // Chi2 threshold 9
    if(badness>1.0) {
      cv::circle(outliers.at(obs.first.cameraIndex), kpt.pt, 1, cv::Scalar(255,0,0),
                 cv::FILLED, cv::LINE_AA);
      outlierCtrs.at(obs.first.cameraIndex)++;
    } else {
      cv::Scalar colour(0,std::max(0.0,255.0*(1.0-badness)),std::min(255.0,255.0*badness));
      cv::circle(images.at(obs.first.cameraIndex), kpt.pt, 1, colour, cv::FILLED, cv::LINE_AA);
      inlierCtrs.at(obs.first.cameraIndex)++;
    }

    // histogram filling
    double errNorm = err.norm();
    int idx = static_cast<int> (std::floor(errNorm)) + 6;
    if(idx > 11)
      idx = 11;
    else if(idx < 0)
      idx = 0;
    errorHistograms.at(obs.first.cameraIndex)[idx] += 1;

  }

  /*// show debug outlier image -- disabled
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    std::stringstream namestr;
    namestr << "Reprojection error image " << im << " (" << inlierCtrs.at(im) << ")";
    cv::imshow(namestr.str(), images[im]);
    std::stringstream outlierstr;
    outlierstr << "Outliers image " << im << " (" << outlierCtrs.at(im) << ")";
    cv::imshow(outlierstr.str(), outliers[im]);
    // histogram printing
    std::cout << "Cam " << im << " reprojection error histogram: " << errorHistograms.at(im).transpose() << std::endl;
  }*/

  // synchronise and communicate that we used all states
  std::vector<StateId> updatedStates;
  synchroniseRealtimeAndFullGraph(updatedStates);
  for(const auto & state : fullGraph_.states_) {
    updatedStatesBa.insert(state.first);
  }

  // get rid of unobserved landmarks
  cleanUnobservedLandmarks();
}

bool ViSlamBackend::saveMap(std::string path)
{
  // save in g2o format
  std::string g2oPath = path.substr(0, path.size() - 4) + ".g2o";
  Component component(fullGraph_.imuParametersVec_[0],
                      multiFrames_.at(StateId(1))->cameraSystem(),
                      fullGraph_,
                      multiFrames_);
  component.save(g2oPath);

  // ... and as own format
  std::ofstream file(path);
  if(!file.good()) {
    return false;
  }
  
  std::ios init(nullptr);
  init.copyfmt(file);
  
  // first we write all the landmarks
  file << "landmarks:" << std::endl;
  for(const auto & landmark : fullGraph_.landmarks_) {
    if(landmark.second.quality > 0.001) {
      // ID
      file << landmark.first.value() << ",";
      // save position
      Eigen::Vector4d hposition = landmark.second.hPoint->estimate();
      Eigen::Vector3d position = hposition.head<3>()/hposition[3];
      file << position[0] << "," << position[1] << "," << position[2];
      file << std::endl;
    }
  }
  
  // next write the frames with keypoint descriptors
  fullGraph_.computeCovisibilities();
  for(const auto & state : fullGraph_.states_) {
    file << "frame: " << state.first.value() << ", covisibilities: ";
    std::vector<std::pair<StateId, int>> covisibilities;
    for(const auto & state2 : fullGraph_.states_) {
      int c = fullGraph_.covisibilities(state.first, state2.first);
      if(c > 0) {
        covisibilities.push_back(std::make_pair(state2.first, c));
      }
    }
    std::sort(covisibilities.begin(), covisibilities.end(), 
         [](const std::pair<StateId, int> & left, const std::pair<StateId, int> & right){
           return left.second > right.second;});
    for(const auto & covisibility : covisibilities) {
      file << covisibility.first.value() << " ";
    }
    file << std::endl;
    for(const auto & obs : state.second.observations) {
      const auto & landmark = fullGraph_.landmarks_.at(obs.second.landmarkId);
      if(landmark.quality > 0.001) {
        file << obs.first.keypointIndex << "," << obs.second.landmarkId.value() << ","; // kpt&lm ID
        Eigen::Vector4d hposition = landmark.hPoint->estimate();
        Eigen::Vector3d position = hposition.head<3>()/hposition[3];
        file << position[0] << "," << position[1] << "," << position[2] << ","; // lm 3D pos redund.
        // retrieve descriptor
        const unsigned char* descriptor =
            multiFrames_.at(state.first)->keypointDescriptor(
              obs.first.cameraIndex, obs.first.keypointIndex);
        for(size_t i=0; i<48; ++i) {
          file << std::setfill('0') << std::setw(2) << std::hex << uint32_t(descriptor[i]);
        }
        file.copyfmt(init); // reset formatting
        file << std::endl;
      }
    }
  }

  return true;
}

bool ViSlamBackend::writeFinalCsvTrajectory(const std::string &csvFileName, bool rpg) const {
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success = csvFile.good();
  if (!success) {
    return false;
  }

  // write description
  if (rpg) {
    csvFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
  } else {
    csvFile << "timestamp" << ", " << "p_WS_W_x" << ", " << "p_WS_W_y" << ", "
            << "p_WS_W_z" << ", " << "q_WS_x" << ", " << "q_WS_y" << ", "
            << "q_WS_z" << ", " << "q_WS_w" << ", " << "v_WS_W_x" << ", "
            << "v_WS_W_y" << ", " << "v_WS_W_z" << ", " << "b_g_x" << ", "
            << "b_g_y" << ", " << "b_g_z" << ", " << "b_a_x" << ", " << "b_a_y"
            << ", " << "b_a_z" << ", " << "NrGps" << ", " << "SID" << ", " << "gpsMode" << std::endl;
  }
  for (auto iter = realtimeGraph_.anyState_.begin(); iter != realtimeGraph_.anyState_.end(); ++iter) {
    Eigen::Vector3d p_WS_W;
    Eigen::Quaterniond q_WS;
    SpeedAndBias speedAndBiases;
    std::stringstream time;
    size_t nrGps;
    size_t gpsMode;
    StateId sid;

    if (iter->second.keyframeId.isInitialised()) {
      // reconstruct from close keyframe
      const ViGraph::State &keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      kinematics::Transformation T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      p_WS_W = T_WS.r();
      q_WS = T_WS.q();
      speedAndBiases.head<3>() = keyframeState.pose->estimate().C() * iter->second.v_Sk;
      speedAndBiases.tail<6>() = keyframeState.speedAndBias->estimate().tail<6>();
      time << iter->second.timestamp.sec << std::setw(9) << std::setfill('0')
            << iter->second.timestamp.nsec;
      nrGps = keyframeState.GpsFactors.size();
      gpsMode = keyframeState.gpsMode;
      sid = iter->first;
    } else {
      // read from state
      const ViGraph::State &state = realtimeGraph_.states_.at(iter->first);
      p_WS_W = state.pose->estimate().r();
      q_WS = state.pose->estimate().q();
      speedAndBiases = state.speedAndBias->estimate();
      time << state.timestamp.sec << std::setw(9) << std::setfill('0')
            << state.timestamp.nsec;
      nrGps = state.GpsFactors.size();
      gpsMode = state.gpsMode;
      sid = iter->first;
    }
    if (rpg) {
      csvFile << std::setprecision(19) << iter->second.timestamp.toSec() << " "
              << p_WS_W[0] << " " << p_WS_W[1] << " " << p_WS_W[2] << " "
              << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " " << q_WS.w() << std::endl;
    } else {
      csvFile << time.str() << ", " << std::scientific
              << std::setprecision(18) << p_WS_W[0] << ", " << p_WS_W[1] << ", "
              << p_WS_W[2] << ", " << q_WS.x() << ", " << q_WS.y() << ", "
              << q_WS.z() << ", " << q_WS.w() << ", " << speedAndBiases[0] << ", "
              << speedAndBiases[1] << ", " << speedAndBiases[2] << ", "
              << speedAndBiases[3] << ", " << speedAndBiases[4] << ", "
              << speedAndBiases[5] << ", " << speedAndBiases[6] << ", "
              << speedAndBiases[7] << ", " << speedAndBiases[8] << ", "
              << nrGps << ", " << sid.value() << ", " << gpsMode << ", "
              << iter->second.keyframeId.value() << std::endl;
    }
  }

  // done, close.
  csvFile.close();
  return success;
}

bool ViSlamBackend::writeGlobalCsvTrajectory(const std::string &csvFileName) const
{
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success =  csvFile.good();
  if(!success) {
    return false;
  }

  //OKVIS_ASSERT_TRUE(Exception, fullGraph_.isSynched(realtimeGraph_), "DA FCK")

  kinematics::Transformation T_GW;

  // write description
  csvFile << "timestamp" << ", " << "p_GA_G_x" << ", " << "p_GA_G_y" << ", "
               << "p_GA_G_z" << std::endl;
  for(auto iter=realtimeGraph_.anyState_.begin(); iter!=realtimeGraph_.anyState_.end(); ++iter) {
    Eigen::Vector3d p_GA_G;
    std::stringstream time;
    kinematics::Transformation T_WS;

    if(iter->second.keyframeId.isInitialised()) {
      // reconstruct from close keyframe
      const ViGraph::State& keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      time << iter->second.timestamp.sec << std::setw(9) << std::setfill('0')
           << iter->second.timestamp.nsec;
    } else {
      // read from state
      const ViGraph::State& state = realtimeGraph_.states_.at(iter->first);
      T_WS = state.pose->estimate();
      time << state.timestamp.sec << std::setw(9) << std::setfill('0')
           << state.timestamp.nsec;
    }

    T_GW = realtimeGraph_.T_GW();
    p_GA_G =  T_GW.C() * (T_WS.r() + T_WS.C()* realtimeGraph_.gpsParametersVec_.back().r_SA ) + T_GW.r();
    csvFile << time.str() << ", " << std::scientific
        << std::setprecision(18) << p_GA_G[0] << ", " << p_GA_G[1] << ", "
        << p_GA_G[2] <<  std::endl;
  }

  // done, close.
  csvFile.close();
  return success;
}

bool ViSlamBackend::attemptLoopClosure(StateId pose_i, StateId pose_j,
                                       const kinematics::Transformation& T_Si_Sj,
                                       const Eigen::Matrix<double, 6, 6>& information,
                                       bool & skipFullGraphOptimisation,
                                       double driftPercentageHeuristic,
                                       bool keepFullGraphRelativePoseConstraint,
                                       double fullGraphRelativePoseInformationScale,
                                       double persistentFullGraphRelativePoseInformationScale,
                                       bool addLoopPathRelativePoseConstraints,
                                       int loopPathRelativePoseConstraintMode,
                                       int loopPathRelativePoseConstraintStride,
                                       bool weakenExistingLoopPathConstraints,
                                       double existingLoopPathConstraintScale,
                                       double loopPathChainInformationScale)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  // check if poses exist, signal unsuccessful otherwise
  if(realtimeGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(realtimeGraph_.states_.count(pose_j) == 0) {
    return false;
  }
  if(fullGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(fullGraph_.states_.count(pose_j) == 0) {
    return false;
  }

  int numSteps = 0; // number of steps to distribute error over
  StateId lastLoopId = auxiliaryStates_.at(pose_i).loopId;
  for(auto iter = fullGraph_.states_.find(pose_i);
      iter != fullGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    if(lastLoopId!=loopId) {
      lastLoopId = loopId;
      numSteps++;
    }
    if(iter->first == pose_j) {
      break;
    }
  }
  StateId firstId = auxiliaryStates_.at(pose_i).loopId;
  lastLoopId = firstId;
  double distanceTravelled = 0.0;
  double distanceTravelled2 = 0.0; // just for the heuristic
  kinematics::Transformation T_WS_i = fullGraph_.states_.at(pose_i).pose->estimate();
  kinematics::Transformation T_WS_last = fullGraph_.states_.at(pose_i).pose->estimate();
  auto iter = fullGraph_.states_.find(pose_i);
  std::vector<double> distances;
  ++iter;
  Eigen::Vector3d distanceTravelledVec(0.0,0.0,0.0);
  if(pose_i != lastLoopId && fullGraph_.states_.count(lastLoopId) != 0) {
    distanceTravelledVec +=
        (fullGraph_.states_.at(lastLoopId).pose->estimate().r()-T_WS_i.r());
    distanceTravelled2 += distanceTravelledVec.norm();

  }
  for(; iter != fullGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    kinematics::Transformation T_WS_j = iter->second.pose->estimate();
    T_WS_last = T_WS_j;
    if(lastLoopId!=loopId) {
      const Eigen::Vector3d dsVec = (T_WS_j.r()-T_WS_i.r());
      const double ds = dsVec.norm();
      lastLoopId = loopId;
      distances.push_back(ds);
      distanceTravelledVec += dsVec;
      distanceTravelled += ds;
      distanceTravelled2 += ds;
      T_WS_i = T_WS_j;
    }
    if(iter->first == pose_j) {
      break;
    }
  }

  skipFullGraphOptimisation = false;
  {

    // we don't optimise, but just re-align the new part to the old part.
    // we distribute the inconsistency along the loop (i.e. what should be variable)
    // according to changing loopIds (i.e. leaving previously closed loops rigid)

    const kinematics::Transformation T_WSi = fullGraph_.pose(pose_i);
    const kinematics::Transformation T_WSj_old= fullGraph_.pose(pose_j);
    const kinematics::Transformation T_Si_Sj_old = T_WSi.inverse() * T_WSj_old;
    const kinematics::Transformation T_WSj_new = T_WSi * T_Si_Sj;
    kinematics::Transformation T_Wnew_Wold_final = T_WSj_new * T_WSj_old.inverse();

    // compute rotation increment for the later averaging
    Eigen::AngleAxisd aa(T_Wnew_Wold_final.q());
    aa.angle() = aa.angle() * (1.0 / double(numSteps));
    const kinematics::Transformation T_WW(Eigen::Vector3d(0.0,0.0,0.0), Eigen::Quaterniond(aa));

    // compute rotation adjustments
    kinematics::Transformation T_WS_prev = fullGraph_.pose(pose_i);
    kinematics::Transformation T_WS = fullGraph_.pose(pose_i);
    auto iter = fullGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    for( ; iter != fullGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {
        T_WS = T_WW * T_WS * T_SS;
        lastLoopId = loopId;
      } else {
        T_WS = T_WS * T_SS;
      }
      if(iter->first == pose_j) {
        break;
      }
    }

    // compute translation error to be distributed later
    const Eigen::Vector3d dr_W = T_WSj_new.r()-T_WS.r();

    // heuristic verification: check relative trajectory errors
    const double relPositionError = dr_W.norm()/(distanceTravelled2);
    const double relOrientationError =
        T_WSj_new.q().angularDistance(T_WSj_old.q())/double(numSteps);
    const double relPositionErrorBudget = // [m/m]
            driftPercentageHeuristic/100.0 + // 1.35% position bias default
            0.02*distanceTravelledVec.norm()/distanceTravelled2 + // 2% scale error
            0.08/sqrt(numSteps); // position noise, 8% stdev per step
    const double relOrientationErrorBudget =
        0.0004 + 0.004/sqrt(numSteps); // bias and noise, in rad/step
    if(relPositionError > relPositionErrorBudget
        || relOrientationError > relOrientationErrorBudget
        || numSteps < 1) {

      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (heuristic consistency).";
      LOG(INFO) << "Rel. pos. err. " << relPositionError << " vs budget "
          << relPositionErrorBudget << " m/m, rel. or. err. "
          << relOrientationError << " vs budget "
          << relOrientationErrorBudget << " rad/kf";
      LOG(INFO) << "dist. travelled " << distanceTravelled2 << " m, no. steps " << numSteps;

      return false;
    }

    // do not accept uncertain loop closures
    Eigen::Matrix<double, 6, 6> P;
    okvis::PseudoInverse::symm(information, P);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(P.topLeftCorner<3, 3>());
    Eigen::Vector3d eigenvalues = saes.eigenvalues();
    double sigma = sqrt(eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
    if (sigma > 0.1 && 3.0 * sigma > relPositionErrorBudget * distanceTravelled2) {
      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (reloc. 3-sigma = " << 3.0*sigma << " m vs budget "
                   << relPositionErrorBudget*distanceTravelled2 << ") ";
      return false;
    }

    // compute full adjustments
    T_WS_prev = fullGraph_.pose(pose_i);
    T_WS = fullGraph_.pose(pose_i);
    iter = fullGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    int ctr = 0;
    double r = 0.0;
    for( ; iter != fullGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {
        // we weight distance adjustments by distance travelled, and rotation uniformly.
        r +=  distances.at(size_t(ctr)) / distanceTravelled;
        T_WS = T_WW *  T_WS * T_SS;
        lastLoopId = loopId;
        ++ctr;
      } else {
        T_WS = T_WS * T_SS;
      }
      const kinematics::Transformation T_WS_set(T_WS.r() + r*dr_W, T_WS.q());
      const kinematics::Transformation T_Wnew_Wold = T_WS_set * T_WSk_old.inverse();
      SpeedAndBias speedAndBias = iter->second.speedAndBias->estimate();
      const Eigen::Vector3d v_Wold = speedAndBias.head<3>();
      speedAndBias.head<3>() = T_Wnew_Wold.C() * v_Wold;
      fullGraph_.setPose(iter->first, T_WS_set);
      fullGraph_.setSpeedAndBias(iter->first, speedAndBias);
      if(realtimeGraph_.states_.count(iter->first) != 0) {
        realtimeGraph_.setPose(iter->first, T_WS_set);
        realtimeGraph_.setSpeedAndBias(iter->first, speedAndBias);
        updatedStatesLoopClosureAttempt_.insert(iter->first);
      }
      if(iter->first == pose_j) {
        break;
      }
    }

    // update landmarks
    for(auto iter = realtimeGraph_.landmarks_.begin();
        iter != realtimeGraph_.landmarks_.end(); ++iter) {
      // TODO: check if this is always right!
      Eigen::Vector4d hPointNew = T_Wnew_Wold_final * iter->second.hPoint->estimate();
      realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
      fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    }
  }

  bool fallbackLoopPathChainComplete = false;
  if(addLoopPathRelativePoseConstraints) {
    std::vector<StateId> correctedPathIds;
    for(auto pathIter = fullGraph_.states_.find(pose_i);
        pathIter != fullGraph_.states_.end(); ++pathIter) {
      const StateId stateId = pathIter->first;
      if(fullGraph_.states_.count(stateId) != 0) {
        correctedPathIds.push_back(stateId);
      }
      if(stateId == pose_j) {
        break;
      }
    }

    const bool hasCompletePath =
        correctedPathIds.size() >= 2
        && correctedPathIds.front() == pose_i
        && correctedPathIds.back() == pose_j;
    if(!hasCompletePath) {
      LOG(WARNING) << "Skip fallback loop path constraints: incomplete corrected path from "
                   << pose_i.value() << " to " << pose_j.value() << ".";
    } else {
      AlignedVector<kinematics::Transformation> correctedPathPoses;
      correctedPathPoses.reserve(correctedPathIds.size());
      for(const StateId stateId : correctedPathIds) {
        correctedPathPoses.push_back(fullGraph_.pose(stateId));
      }
      const int pathMode =
          (loopPathRelativePoseConstraintMode >= 0
           && loopPathRelativePoseConstraintMode <= 2)
              ? loopPathRelativePoseConstraintMode
              : 1;
      const int pathStride = std::max(1, loopPathRelativePoseConstraintStride);
      const bool addAnchorConstraints = pathMode == 0 || pathMode == 2;
      const bool addChainConstraints = pathMode == 1 || pathMode == 2;
      const double anchorInformationScale =
          std::isfinite(persistentFullGraphRelativePoseInformationScale)
          && persistentFullGraphRelativePoseInformationScale > 0.0
              ? persistentFullGraphRelativePoseInformationScale
              : 100.0;
      const double chainInformationScale =
          std::isfinite(loopPathChainInformationScale)
          && loopPathChainInformationScale > 0.0
              ? loopPathChainInformationScale
              : anchorInformationScale;
      const Eigen::Matrix<double, 6, 6> anchorInformation =
          anchorInformationScale * information;
      const Eigen::Matrix<double, 6, 6> chainInformation =
          chainInformationScale * information;

      if(weakenExistingLoopPathConstraints) {
        addSuppressedLoopPathConstraintSnapshot(correctedPathIds);
        const auto realtimeRemoved =
            suppressLoopPathConstraintsInGraph(realtimeGraph_, correctedPathIds);
        std::pair<size_t, size_t> fullGraphRemoved{0, 0};
        if(!isLoopClosing_ && !isLoopClosureAvailable_) {
          fullGraphRemoved =
              suppressLoopPathConstraintsInGraph(fullGraph_, correctedPathIds);
        } else {
          for(const StateId stateId : correctedPathIds) {
            touchedStates_.insert(stateId);
          }
        }
        LOG(INFO) << "Removed fallback loop path constraints before corrected-chain insertion"
                  << " realtime(relative=" << realtimeRemoved.first
                  << ", two_pose=" << realtimeRemoved.second << ")"
                  << " full(relative=" << fullGraphRemoved.first
                  << ", two_pose=" << fullGraphRemoved.second << ").";
      }

      size_t anchorConstraintsAdded = 0;
      if(addAnchorConstraints) {
        std::set<StateId> anchorPathIds;
        StateId pathLastLoopId = firstId;
        for(size_t pathIndex = 1; pathIndex < correctedPathIds.size(); ++pathIndex) {
          const StateId stateId = correctedPathIds[pathIndex];
          if(auxiliaryStates_.count(stateId) == 0) {
            continue;
          }
          const StateId loopId = auxiliaryStates_.at(stateId).loopId;
          const bool loopBoundary = pathLastLoopId != loopId;
          const bool loopEndpoint = stateId == pose_j;
          if(loopBoundary || loopEndpoint) {
            anchorPathIds.insert(stateId);
          }
          if(loopBoundary) {
            pathLastLoopId = loopId;
          }
        }
        anchorPathIds.erase(pose_i);

        const kinematics::Transformation T_WSi_path = fullGraph_.pose(pose_i);
        for(const StateId pathId : anchorPathIds) {
          if(realtimeGraph_.states_.count(pathId) == 0 || pathId == pose_i) {
            continue;
          }
          const kinematics::Transformation T_Si_Sk =
              T_WSi_path.inverse() * fullGraph_.pose(pathId);
          if(addRelativePoseConstraint(pose_i, pathId, T_Si_Sk, anchorInformation, true)) {
            protectLoopPathRelativePoseConstraint(pose_i, pathId);
            ++anchorConstraintsAdded;
          }
        }
      }

      size_t chainConstraintsAdded = 0;
      if(addChainConstraints) {
        if(keepFullGraphRelativePoseConstraint) {
          addPersistentLoopPathConstraintSnapshot(
              correctedPathIds, correctedPathPoses, information,
              chainInformationScale, pathStride,
              weakenExistingLoopPathConstraints, existingLoopPathConstraintScale);
          reapplyPersistentLoopPathRelativePoseConstraints(true, true);
        }
        StateId chainStartId = correctedPathIds.front();
        size_t skippedSinceLastConstraint = 0;
        for(size_t pathIndex = 1; pathIndex < correctedPathIds.size(); ++pathIndex) {
          const StateId chainEndId = correctedPathIds[pathIndex];
          const bool isEndpoint = chainEndId == pose_j;
          ++skippedSinceLastConstraint;
          if(!isEndpoint && skippedSinceLastConstraint < size_t(pathStride)) {
            continue;
          }
          if(chainStartId == chainEndId) {
            continue;
          }
          const kinematics::Transformation T_Sstart_Send =
              fullGraph_.pose(chainStartId).inverse() * fullGraph_.pose(chainEndId);
          if(addRelativePoseConstraint(
                 chainStartId, chainEndId, T_Sstart_Send, chainInformation, true)) {
            protectLoopPathRelativePoseConstraint(chainStartId, chainEndId);
            ++chainConstraintsAdded;
          }
          chainStartId = chainEndId;
          skippedSinceLastConstraint = 0;
        }
      }

      LOG(INFO) << "Added fallback loop path relative-pose constraints: anchor="
                << anchorConstraintsAdded << " chain=" << chainConstraintsAdded
                << " mode=" << pathMode << " stride=" << pathStride
                << " anchor_information_scale=" << anchorInformationScale
                << " chain_information_scale=" << chainInformationScale << ".";
      fallbackLoopPathChainComplete = true;
    }
  }

  if(fallbackLoopPathChainComplete) {
    setFallbackLoopGarlileoBetweenPathBounds(pose_i, pose_j);
  } else {
    clearFallbackLoopGarlileoBetweenPathBounds();
  }

  // remember this frame closed a loop
  auxiliaryStates_.at(pose_j).closedLoop = true;

  const RelPoseInfo loopRelativePoseConstraint{
      T_Si_Sj, information, pose_i, pose_j,
      keepFullGraphRelativePoseConstraint,
      fullGraphRelativePoseInformationScale,
      persistentFullGraphRelativePoseInformationScale};
  fullGraphRelativePoseConstraints_.push_back(loopRelativePoseConstraint);
  if(keepFullGraphRelativePoseConstraint) {
    addOrUpdatePersistentFullGraphRelativePoseConstraint(loopRelativePoseConstraint);
  }

  //const kinematics::Transformation T_SiSj_estimated = (pose(pose_i).inverse() * pose(pose_j));
  //OKVIS_ASSERT_TRUE(Exception,
  //                  (T_SiSj_estimated.inverse() * T_Si_Sj).T().isIdentity(1.0e-6),
  //                  "We are f**ked: T_SiSj_estimated = \n" << T_SiSj_estimated.T() << std::endl <<
  //                  "T_Si_Sj = \n" << T_Si_Sj.T())

  return true;
}

bool ViSlamBackend::attemptFullGpsAlignment(StateId pose_i , StateId pose_j, const okvis::kinematics::Transformation& T_GW_new){
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "loop closure not finished, cannot merge landmarks")

  // check if poses exist, signal unsuccessful otherwise
  if(realtimeGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(realtimeGraph_.states_.count(pose_j) == 0) {
    return false;
  }

  // Obtain different T_GW estimates and compute relative transformations
  okvis::kinematics::Transformation T_GW_old = realtimeGraph_.T_GW(pose_i);
  LOG(INFO) << "[GPS Loop Closure] Between states "
            << pose_i.value() << " and " << pose_j.value() << " while most recent state is "
            << realtimeGraph_.states_.rbegin()->first.value() << "\n"
            << "T_GW_old:\n" << T_GW_old.T3x4() << "\n"
            << "T_GW_new:\n" << T_GW_new.T3x4();

  // XXX: What we actually want to do is to transform from "new to old" (because we assume the global frame should not move)! Misleading names here when comparing to attemptLoopClosure().
  okvis::kinematics::Transformation T_WSj = realtimeGraph_.pose(pose_j);
  okvis::kinematics::Transformation T_WSi = realtimeGraph_.pose(pose_i);

  const kinematics::Transformation T_Wold_Wnew_final = T_GW_old.inverse() * T_GW_new;
  const kinematics::Transformation T_WSj_old = T_Wold_Wnew_final * T_WSj;


  // optimise -- if sensible to do it already now.
  int numVariableStates = 0; // number of varying states
  for(auto iter = realtimeGraph_.states_.find(pose_i);
      iter != realtimeGraph_.states_.end(); ++iter) {
    numVariableStates++;
  }

  StateId firstId = auxiliaryStates_.at(pose_i).loopId;
  StateId lastId;
  StateId id;
  lastId = firstId;

  double distanceTravelled = 0.0;
  kinematics::Transformation T_WS_i = realtimeGraph_.states_.at(pose_i).pose->estimate();
  auto iter = realtimeGraph_.states_.find(pose_i);
  std::vector<double> distances;
  ++iter;
  for(; iter != realtimeGraph_.states_.end(); ++iter) {
    kinematics::Transformation T_WS_j = iter->second.pose->estimate();
    const double ds = (T_WS_j.r()-T_WS_i.r()).norm();
    distances.push_back(ds);
    distanceTravelled += ds;
    T_WS_i = T_WS_j;
  }

  // compute rotation increment for the later averaging
  Eigen::AngleAxisd aa(T_Wold_Wnew_final.q());
  aa.angle() = aa.angle() * (1.0 / double(numVariableStates));
  const kinematics::Transformation T_WW(Eigen::Vector3d(0.0,0.0,0.0), Eigen::Quaterniond(aa));
  kinematics::Transformation T_WS_prev = realtimeGraph_.pose(pose_i);
  kinematics::Transformation T_WS = realtimeGraph_.pose(pose_i);
  iter = realtimeGraph_.states_.find(pose_i);
  iter++;
  for( ; iter != realtimeGraph_.states_.end(); ++iter) {
    const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
    const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
    T_WS_prev = T_WSk_old;
    T_WS = T_WW *  T_WS * T_SS; // XXX: needs investigation
  }

  // T_WSj_old: Should be pose of state j after alignment
  const Eigen::Vector3d dr_W = T_WSj_old.r()-T_WS.r();

  // compute full adjustments
  T_WS_prev = realtimeGraph_.pose(pose_i);
  T_WS = realtimeGraph_.pose(pose_i);
  iter = realtimeGraph_.states_.find(pose_i);
  iter++;
  int ctr = 0;
  double r = 0.0;
  for( ; iter != realtimeGraph_.states_.end(); ++iter) {
    const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
    const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
    T_WS_prev = T_WSk_old;
    // we weight distance adjustments by distance travelled, and rotation uniformly.
    r +=  distances.at(ctr) / distanceTravelled;
    T_WS = T_WW * T_WS * T_SS; // XXX: unvestigate why this is different
    ++ctr;
    const kinematics::Transformation T_WS_set(T_WS.r() + r*dr_W, T_WS.q());
    const kinematics::Transformation T_Wnew_Wold = T_WS_set * T_WSk_old.inverse(); /// XXX: this now uses the wrong namings but should do the correct thing
    SpeedAndBias speedAndBias = iter->second.speedAndBias->estimate();
    speedAndBias.head<3>() = (T_Wnew_Wold.C() * speedAndBias.head<3>()).eval();
    realtimeGraph_.setPose(iter->first, T_WS_set);
    fullGraph_.setPose(iter->first, T_WS_set);
    realtimeGraph_.setSpeedAndBias(iter->first, speedAndBias);
    fullGraph_.setSpeedAndBias(iter->first, speedAndBias);
  }

  /// update landmarks
  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end(); ++iter) {
    Eigen::Vector4d hPointNew = T_Wold_Wnew_final * iter->second.hPoint->estimate();
    realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
  }

  return true;
}

bool ViSlamBackend::attemptPosGpsAlignment(StateId pose_i , StateId pose_j, const Eigen::Vector3d& posAlignVec){

    OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
    OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_, "loop closure not finished, cannot merge landmarks")

    // check if poses exist, signal unsuccessful otherwise
    if(realtimeGraph_.states_.count(pose_i) == 0) {
      return false;
    }
    if(realtimeGraph_.states_.count(pose_j) == 0) {
      return false;
    }
    Eigen::Vector3d r_Wold_Wnew = posAlignVec;

    // freeze / unfreeze
    StateId oldestId=pose_i;

    fullGraph_.unfreezePosesFrom(pose_i);
    fullGraph_.freezePosesUntil(pose_i);

    fullGraph_.unfreezeSpeedAndBiasesFrom(pose_i);
    fullGraph_.freezeSpeedAndBiasesUntil(pose_i);

    // optimise -- if sensible to do it already now.
    int numVariableStates = 0; // number of varying states
    for(auto iter = realtimeGraph_.states_.find(oldestId);
        iter != realtimeGraph_.states_.end(); ++iter) {
      numVariableStates++;
      if(iter->first == pose_j){
          break;
        }
    }

    StateId lastId;
    StateId id;
    lastId = oldestId;

    double fullDistanceTravelled = 0.0;
    std::vector<double> distances; // save distance travelled per step
    auto iter = realtimeGraph_.states_.find(pose_i);
    ++iter;
    for(; iter != realtimeGraph_.states_.end(); ++iter) {
      id = iter->first;
      if(id.value() > pose_j.value())
        break;
      double ds = (realtimeGraph_.pose(id).r()-realtimeGraph_.pose(lastId).r()).norm();
      distances.push_back(ds);
      lastId = id;
      fullDistanceTravelled += ds;
    }

    // Now apply position adjustments based on distance travelled
    size_t counter = 0; // counter to iterate distances vector
    iter = realtimeGraph_.states_.find(pose_i);
    lastId = oldestId;
    ++iter;
    double distanceTravelled = 0.0;
    okvis::kinematics::Transformation T_WS;

    for(; iter != realtimeGraph_.states_.end(); ++iter) {
      id = iter->first;
      //std::cout << "Adjusting state " << id.value() << std::endl;
      if(id.value() <= pose_j.value()){

          distanceTravelled += distances.at(counter);

          // position adjustment
          Eigen::Vector3d dr = distanceTravelled/fullDistanceTravelled * r_Wold_Wnew;
          // now apply position adjustment
          T_WS = realtimeGraph_.pose(id);
          okvis::kinematics::Transformation T_WS_set(T_WS.r()+dr,T_WS.q());
          realtimeGraph_.setPose(iter->first, T_WS_set);
          fullGraph_.setPose(iter->first, T_WS_set);
          counter+=1;
        }
      else{
          // now apply position adjustment
          //std::cout << "Adjustment is done with a rigid factor of 1 " << std::endl;
          T_WS = realtimeGraph_.pose(id);
          okvis::kinematics::Transformation T_WS_set(T_WS.r() + r_Wold_Wnew,T_WS.q());
          realtimeGraph_.setPose(iter->first, T_WS_set);
          fullGraph_.setPose(iter->first, T_WS_set);
        }
    }

  /// update landmarks
  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end(); ++iter) {
    Eigen::Vector4d r_Wold_Wnew_hom(r_Wold_Wnew(0) , r_Wold_Wnew(1) , r_Wold_Wnew(2) , 0.0);
    Eigen::Vector4d hPointNew = r_Wold_Wnew_hom + iter->second.hPoint->estimate();
    realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
  }

  return true;
}

bool ViSlamBackend::addSubmapAlignmentConstraints(const SupereightMapType* submap_ptr,
                                                  const uint64_t &frame_A_id, const uint64_t frame_B_id,
                                                  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pointCloud,
                                                  std::vector<float> sensorError, bool isLidar, const std::string robustFunction) {
  if(!isLoopClosing_ && !isLoopClosureAvailable_){
    fullGraph_.addSubmapAlignmentConstraints(submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction);
  }
  else
  {
    addSubmapAlignmentBacklog_.push_back(AddSubmapAlignmentBacklog{submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction});
    LOG(INFO) << "full Graph is being optimized. don't add alignment constraints.";
  }
  return realtimeGraph_.addSubmapAlignmentConstraints(submap_ptr, frame_A_id, frame_B_id, pointCloud, sensorError, isLidar, robustFunction);
}

StateId ViSlamBackend::currentKeyframeStateId(bool considerLoopClosureFrames) const
{
  StateId currentFrame = currentStateId();
  return mostOverlappedStateId(currentFrame, considerLoopClosureFrames);
}

StateId ViSlamBackend::mostOverlappedStateId(StateId frame, bool considerLoopClosureFrames) const
{
  std::set<StateId> allFrames;
  allFrames.insert(keyFrames_.begin(), keyFrames_.end());
  allFrames.insert(imuFrames_.begin(), imuFrames_.end());
  allFrames.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  if(!considerLoopClosureFrames) {
    for(const auto & id : currentLoopClosureFrames_)
      allFrames.erase(id);
  }
  StateId returnId;
  double overlap = 0.0;
  for(auto id : allFrames) {
    if(id==frame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(frame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==frame) {
    OKVIS_THROW(Exception, "most overlapped frame with itself: makes no sense")
  }
  return returnId;
}

StateId ViSlamBackend::currentLoopclosureStateId() const
{
  StateId currentFrame = currentStateId();
  StateId returnId;
  double overlap = 0.0;
  for(auto id : loopClosureFrames_) {
    if(id==currentFrame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(currentFrame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==currentFrame) {
    OKVIS_THROW(Exception, "current frame is loopclosure frame: makes no sense")
  }
  if(overlap>0.5) {
    return returnId;
  }
  return StateId();
}

int ViSlamBackend::prunePlaceRecognitionFrames() {
  realtimeGraph_.computeCovisibilities();
  std::set<StateId> allFrames;
  allFrames.insert(keyFrames_.begin(), keyFrames_.end());
  allFrames.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  int ctr=0;
  if(keyFrames_.size() == 0) {
    return 0;
  }
  StateId id0 = *keyFrames_.rbegin();
  for(auto id1 : allFrames) {
    if(id0.value()<=id1.value()) {
      continue;
    }
    if(!auxiliaryStates_.at(id0).isPlaceRecognitionFrame) {
      continue;
    }
    if(!auxiliaryStates_.at(id1).isPlaceRecognitionFrame) {
      continue;
    }
    if(realtimeGraph_.covisibilities(id0, id1) < 10) {
      continue;
    }
    const double overlap = overlapFraction(multiFrames_.at(id0), multiFrames_.at(id1));
    if(overlap > 0.6) {
      ctr++;
      // prune the newer frame
      auxiliaryStates_.at(id0).isPlaceRecognitionFrame = false;
      break;
    }
  }
  return ctr;
}

void ViSlamBackend::clear()
{
  realtimeTrustRegionRadius_.reset();
  requestedVisualTwoPoseWeightScale_ = 1.0;
  // clear underlying graphs
  realtimeGraph_.clear();
  fullGraph_.clear();

  multiFrames_.clear();

  auxiliaryStates_.clear(); // Store information about states.
  currentComponentIdx_ = 0; // The index of the current component.

  loopClosureFrames_.clear(); // All the current loop closure frames.

  imuFrames_.clear(); // All the current IMU frames.
  keyFrames_.clear(); // All the current keyframes.

  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = false;
  isLoopClosureAvailable_ = false;
  components_.resize(1);

  addStatesBacklog_.clear(); // Backlog of states to add to fullGraph_.
  eliminateStates_.clear(); // States eliminated in realtimeGraph_.
  touchedStates_.clear(); // States modified in realtimeGraph_.
  touchedLandmarks_.clear(); // Landmarks modified in realtimeGraph_.

  fullGraphRelativePoseConstraints_.clear(); // Relative pose constraints.
  persistentFullGraphRelativePoseConstraints_.clear(); // Full-graph-only loop relative pose constraints.
  persistentLoopPathConstraintSnapshots_.clear(); // Persistent corrected loop path chain constraints.
  suppressedLoopPathConstraintSnapshots_.clear(); // Corrected fallback loop path constraints suppressed across sync.
  {
    std::lock_guard<std::mutex> lock(loopPathProtectionMutex_);
    protectedLoopPathRelativePoseConstraints_.clear();
    protectedLoopPathRelativePoseConstraintsRealtime_.clear();
    protectedLoopPathRelativePoseConstraintsFullGraph_.clear();
  }
  suppressedLoopPathRelativePoseConstraints_.clear(); // Removed fallback loop path relative-pose constraints.
  suppressedLoopPathTwoPoseConstraints_.clear(); // Removed fallback loop path visual relative-pose constraints.

  clearFallbackLoopGarlileoBetweenPathBounds();

  lastFreeze_ = StateId(); // Store up to where the realtimeGraph_ states were fixed.
}

double ViSlamBackend::overlapFraction(const MultiFramePtr frameA,
                                      const MultiFramePtr frameB) const {

  OKVIS_ASSERT_TRUE(Exception, frameA->numFrames() == frameB->numFrames(),
                    "must be same number of frames")
  const size_t numFrames = frameA->numFrames();
  const MultiFramePtr frames[2] = {frameA, frameB};

  std::set<LandmarkId> landmarks[2];
  std::vector<cv::Mat> detectionsImg[2];
  detectionsImg[0].resize(numFrames);
  detectionsImg[1].resize(numFrames);
  std::vector<cv::Mat> matchesImg[2];
  matchesImg[0].resize(numFrames);
  matchesImg[1].resize(numFrames);

  // paint detection images and remember matched points
  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      detectionsImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      matchesImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      const size_t num = frames[f]->numKeypoints(im);
      cv::KeyPoint keypoint;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        cv::circle(detectionsImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                   cv::FILLED);
        uint64_t lmId = frames[f]->landmarkId(im, k);
        if (lmId != 0) {
          landmarks[f].insert(LandmarkId(lmId));
        }
      }
    }
  }

  // find matches
  std::set<LandmarkId> matches;
  std::set_intersection(
      landmarks[0].begin(), landmarks[0].end(),landmarks[1].begin(),
      landmarks[1].end(), std::inserter(matches,matches.begin()));

  // without matches there will be no overlap
  if(matches.size() == 0) {
    return 0.0;
  }

  // draw match images
  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::KeyPoint keypoint;
      const size_t num = frames[f]->numKeypoints(im);
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        if (matches.count(LandmarkId(frames[f]->landmarkId(im, k)))) {
          cv::circle(matchesImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                     cv::FILLED);
        }
      }
    }
  }

  // IoU
  double overlap[2];
  for(size_t f=0; f<2; ++f) {
    int intersectionCount = 0;
    int unionCount = 0;
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::Mat intersectionMask, unionMask;
      cv::bitwise_and(matchesImg[f].at(im), detectionsImg[f].at(im), intersectionMask);
      cv::bitwise_or(matchesImg[f].at(im), detectionsImg[f].at(im), unionMask);
      intersectionCount += cv::countNonZero(intersectionMask);
      unionCount += cv::countNonZero(unionMask);
    }
    overlap[f] = double(intersectionCount)/double(unionCount);
  }

  return std::min(overlap[0], overlap[1]);
}

void ViSlamBackend::writeLidarDebugStatisticsCsv(const std::string& csvFilePrefix)
{
  realtimeGraph_.writeLidarDebugStatisticsCsv(csvFilePrefix);
}

}  // namespace okvis
