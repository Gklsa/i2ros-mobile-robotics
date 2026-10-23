std_msgs/Header header

uint8 UNKNOWN=0
uint8 RED=1
uint8 YELLOW=2
uint8 GREEN=3

uint8 state
string label
float32 confidence

# Image-plane position in pixels, relative to the full RGB image.
float32 center_x
float32 center_y
float32 radius

float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

# Normalized full-image coordinates. 0.0 is left/top, 1.0 is right/bottom.
float32 normalized_x
float32 normalized_y

# Shape/color features for downstream filtering.
float32 area
float32 color_ratio
float32 circularity
float32 apparent_size

# True when this bulb was found inside a detected traffic-light housing/frame
# (a dark rectangular body, or a bright/colored signal casing, each with bulb
# holes) rather than as a bare color blob. Decision/Planning can use this to
# prefer structurally-confirmed candidates over bare blobs that may be signs,
# billboards, or tail lights.
bool has_housing

# World-frame position (meters, "world" TF frame), back-projected from the
# depth camera at the bbox center. Only valid when has_world_position is
# true (depth/camera_info not yet received, or depth invalid/out of range,
# leaves world_x/y/z at 0). Approximate: derived from the RGB-left pixel
# location sampled into the depth camera's own image, so it carries a small
# parallax error equal to the ~0.2m RGB-left/depth camera baseline.
bool has_world_position
float32 world_x
float32 world_y
float32 world_z
