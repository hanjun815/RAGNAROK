// GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
// SPDX-License-Identifier: MIT
#ifndef GARLILEO_INCREMENTAL_WINDOW_READINESS_H
#define GARLILEO_INCREMENTAL_WINDOW_READINESS_H

#include <cstddef>

namespace garlileo {

// Recovery is limited in sensor time, so pausing playback does not expire it.
// 0.5s accommodates several 15Hz scans (the observed recovery needs 0.205s),
// while bounding retry windows to roughly 50 IMU samples at the dataset's 100Hz.
// This is an explicit recovery policy, not the publication lag/retention margin.
inline constexpr double kMaxRadarRecoveryHorizonSec = 0.5;

inline bool radarRecoveryHorizonExceeded(
        double windowStart, double commonWindowEnd, double latestImuTime,
        bool useAnyRadar, bool windowWasDeferred, double minimumWindowDurationSec) {
    if (!useAnyRadar) return false;
    // Bound accumulated retries. Also bound a frozen endpoint ONLY when the
    // existing duration guard has no usable window. A ready, never-deferred
    // window must not be rejected just because newer IMU data arrived first.
    return (windowWasDeferred
            && latestImuTime - windowStart > kMaxRadarRecoveryHorizonSec)
        || (commonWindowEnd - windowStart < minimumWindowDurationSec
            && latestImuTime - commonWindowEnd > kMaxRadarRecoveryHorizonSec);
}

// A window shorter than this holds only one or two IMU samples, so a normal
// 20-30 ms gap at 100 Hz (Co-RaL) reads as below 50 Hz; wait for more samples.
inline constexpr double kMinImuRateWindowSec = 0.1;

enum class IncrementalWindowReadiness {
    Ready,
    WaitForImuSamples,
    WaitForRadarTargets,
    FatalImuRate,
};

// Called after IncrementalOptimization establishes a positive window duration.
// Radar counts are individual valid targets, not PointCloud2 scan messages.
// Preserve the existing thresholds and IMU failure policy; low radar target
// density is retryable because subsequent scans can complete this same window.
inline IncrementalWindowReadiness classifyIncrementalWindow(
        double durationSec, std::size_t imuSamples,
        bool useRadar0, std::size_t radar0Targets,
        bool useRadar1, std::size_t radar1Targets) {
    if (static_cast<double>(imuSamples) / durationSec < 50.0) {
        return durationSec < kMinImuRateWindowSec ? IncrementalWindowReadiness::WaitForImuSamples
                                                  : IncrementalWindowReadiness::FatalImuRate;
    }
    if ((useRadar0 && static_cast<double>(radar0Targets) / durationSec < 10.0)
        || (useRadar1 && static_cast<double>(radar1Targets) / durationSec < 10.0)) {
        return IncrementalWindowReadiness::WaitForRadarTargets;
    }
    return IncrementalWindowReadiness::Ready;
}

}  // namespace garlileo
#endif
