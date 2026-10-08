#pragma once

#include <algorithm>
#include <cmath>

namespace okvis {
namespace ceres {

/// Trust-region radius of the realtime solves. Full realtime solves continue the
/// radius of the previous full realtime solve; frontend solves always start from
/// kInitialRadius and do not update this state. Owners must reset it when
/// replacing the graph.
class RealtimeTrustRegionRadius {
 public:
  /// Initial radius of the realtime and frontend solves (Ceres default: 1e4).
  static constexpr double kInitialRadius = 1.0e12;

  void reset() { previous_ = 0.0; }

  double initial(double minimum, double maximum, bool onlyNewest, bool initialized) const {
    if (onlyNewest || !initialized || previous_ <= 0.0) {
      return kInitialRadius;
    }
    return std::clamp(previous_, minimum, maximum);
  }

  void observe(double radius, bool usable, bool onlyNewest, bool initialized) {
    if (onlyNewest) {
      return;
    }
    if (!initialized || !usable || !std::isfinite(radius) || radius <= 0.0) {
      reset();
      return;
    }
    previous_ = radius;
  }

 private:
  double previous_ = 0.0;
};

}  // namespace ceres
}  // namespace okvis
