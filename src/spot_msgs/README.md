# spot_msgs

Foot state message definitions for Boston Dynamics Spot, originating from Co-RaL.
GaRLILEO subscribes to the foot state topic as `spot_msgs/msg/FootStateArray`, and
the dataset bags record that topic under this type name, so this package must be
built before replaying them.

- `FootStateArray`: header and the state of each foot (RAGNAROK dataset).
- `FootState`: foot position in the body frame, contact state, and terrain data.
- `FootTerrainState`: terrain estimates reported by Spot.

The GaRLILEO and Co-RaL datasets were recorded with the earlier
[SPOT_ego_Velocity](https://github.com/SangwooJung98/SPOT_ego_Velocity) layout,
which has no terrain data, under the same `spot_msgs/msg/FootStateArray` type name.
`EgoVelocityFootStateArray` and `EgoVelocityFootState` reproduce that layout so
GaRLILEO can decode those recordings; they are not published on any topic.
