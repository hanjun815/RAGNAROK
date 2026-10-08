# FootState layout of SPOT_ego_Velocity, used to record the GaRLILEO and Co-RaL
# datasets under the spot_msgs/msg/FootStateArray type name. GaRLILEO uses it
# only to decode those recordings.

# Contact
uint8 CONTACT_UNKNOWN = 0
uint8 CONTACT_MADE = 1
uint8 CONTACT_LOST = 2

geometry_msgs/Point foot_position_rt_body
uint8 contact
