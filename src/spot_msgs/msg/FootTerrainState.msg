# Frame name in which terrain vectors are expressed.
string frame_name

# Estimated friction coefficient (mu) for the local ground patch.
float64 ground_mu_est

# Foot slip distance estimate.
bool has_foot_slip_distance_rt_frame
geometry_msgs/Vector3 foot_slip_distance_rt_frame

# Foot slip velocity estimate.
bool has_foot_slip_velocity_rt_frame
geometry_msgs/Vector3 foot_slip_velocity_rt_frame

# Estimated terrain contact normal.
bool has_ground_contact_normal_rt_frame
geometry_msgs/Vector3 ground_contact_normal_rt_frame

# Visual terrain surface penetration statistics.
float64 visual_surface_ground_penetration_mean
float64 visual_surface_ground_penetration_std
