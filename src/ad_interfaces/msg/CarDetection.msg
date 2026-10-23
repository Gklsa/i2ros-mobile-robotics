std_msgs/Header header

float32 confidence
float32 distance

float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

# False when this candidate was filtered out (too close/far, off-axis, or
# inside a waypoint zone's left-side ROI exclusion) - see reject_reason.
bool accepted
string reject_reason

# World-frame position (meters, "world" TF frame), back-projected from the
# depth camera at the bbox center. Only valid when has_world_position is
# true (depth/camera_info not yet received, or depth invalid/out of range,
# leaves world_x/y/z at 0). Approximate: derived from the RGB-left pixel
# location sampled into the depth camera's own image, so it carries a small
# parallax error equal to the ~0.2m RGB-left/depth camera baseline - same
# caveat as TrafficLightDetection's world position.
bool has_world_position
float32 world_x
float32 world_y
float32 world_z
