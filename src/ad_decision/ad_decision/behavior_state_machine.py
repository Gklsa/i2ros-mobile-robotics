# Copyright 2026 i2ros Team 6
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Behavior decision node: traffic lights, front obstacles, and target speed.

Traffic lights are handled from the per-light array on
``/perception/traffic_light_detections`` rather than the scalar
``/perception/traffic_light_state``. The scalar is a plain argmax over
per-color confidence (see TrafficLightDetector._select_output_state), so a
bright cross-traffic green can outrank our own red, and it carries no
position, which makes distance-based deceleration impossible.

Consequences of consuming the array instead:

* The array is published *unfiltered* - neither the detector's
  ``traffic_light_confidence_threshold`` (used there as an on/off switch,
  set to 1.1 globally in params.yaml) nor its per-waypoint overrides apply
  to it. Both gates are therefore re-implemented here as
  ``red_light_confidence_threshold`` and ``red_light_waypoint_zones``.
* Every ``red_light_*`` parameter is deliberately distinct from perception's
  ``traffic_light_*`` parameters. This keeps behavior thresholds independent
  from the waypoint-gated detector configuration in the shared ``/**`` block.

The red-light behavior is a small state machine of its own: track the
nearest relevant red ahead in the world frame, ramp the target speed down so
the car reaches zero at the stop point, then *latch* the stop. The latch is
what makes a stop hold - at the stop line the light is usually too high and
too close to stay in the camera frame, so the detection disappears while the
light is still red. The latch releases on a green at the same light, on
leaving the intersection's waypoint zone, or on a timeout.
"""

from enum import Enum
import math
import struct
from typing import List, Optional, Tuple

from ad_interfaces.msg import CarDetections, TrafficLightDetections, TrafficLightState
from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import Bool, Float32, Int32, String

# (world_xyz, forward_m, confidence, normalized_x) for a single accepted
# detection. normalized_x is the image-plane center (0 left, 1 right) straight
# from the detector - unlike world_xyz it involves no depth sample, so it
# stays trustworthy when the depth back-projection is off by metres.
LightObservation = Tuple[Tuple[float, float, float], float, float, float]
# (zone_low, zone_high, waypoint_index, world_x, world_y).  The explicit
# world coordinates let decision start braking before waypoint_manager has
# advanced far enough to publish the stop waypoint itself.
FixedStopPoint = Tuple[int, int, int, float, float]
# (zone_low, zone_high, world_x, world_y, world_z).  While such a zone is
# active, this is the only traffic-light head allowed to affect decision.
MappedTrafficLight = Tuple[int, int, float, float, float]


class BehaviorState(str, Enum):
    CRUISE = 'CRUISE'
    APPROACH_RED_LIGHT = 'APPROACH_RED_LIGHT'
    STOP_FOR_RED_LIGHT = 'STOP_FOR_RED_LIGHT'
    FOLLOW_VEHICLE = 'FOLLOW_VEHICLE'
    EMERGENCY_BRAKE = 'EMERGENCY_BRAKE'
    ROUTE_COMPLETE = 'ROUTE_COMPLETE'


class TrafficLightTarget:
    """The one light the decision node is currently reacting to."""

    def __init__(self, world_xyz: Tuple[float, float, float], stamp) -> None:
        self.world_xyz = world_xyz
        self.last_seen = stamp
        self.confidence = 0.0
        self.forward_m: Optional[float] = None
        self.latched = False
        self.latched_at = None
        # Set once a detection of this head was seen close to the image
        # center while still far enough away for the geometry to be
        # well-conditioned. Grants immunity from the lateral stop gate,
        # whose input is the depth-derived world position - at waypoint 2 a
        # head dead-center in the image measured lateral +8.4 m and was
        # crawled past on red purely because of that bad localization.
        self.image_confirmed = False


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BehaviorStateMachine(Node):
    """Behavior decision node for traffic lights and front obstacles."""

    def __init__(self) -> None:
        super().__init__('behavior_state_machine')

        self.declare_parameter('pose_topic', '/OurCar/CoM/pose')
        self.declare_parameter('twist_topic', '/OurCar/CoM/twist')
        self.declare_parameter(
            'traffic_light_detections_topic',
            '/perception/traffic_light_detections',
        )
        self.declare_parameter(
            'traffic_light_camera_info_topic',
            '/OurCar/Sensors/RGBCameraLeft/camera_info',
        )
        self.declare_parameter('waypoint_index_topic', '/planning/waypoint_index')
        self.declare_parameter('next_waypoint_topic', '/planning/next_waypoint')
        self.declare_parameter('obstacle_points_topic', '/perception/pcl/points_preprocessed')
        self.declare_parameter('front_car_distance_topic', '/perception/front_car_distance')
        self.declare_parameter('car_detections_topic', '/perception/car_detections')
        self.declare_parameter('decision_state_topic', '/decision/state')
        self.declare_parameter('decision_target_speed_topic', '/decision/target_speed')
        self.declare_parameter('decision_stop_request_topic', '/decision/stop_request')
        self.declare_parameter('decision_emergency_brake_topic', '/decision/emergency_brake')
        self.declare_parameter(
            'decision_traffic_light_distance_topic',
            '/decision/traffic_light_distance',
        )

        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('target_speed', 5.0)
        self.declare_parameter('follow_speed', 2.0)
        self.declare_parameter('minimum_follow_speed', 0.5)
        self.declare_parameter('obstacle_timeout_sec', 0.5)
        self.declare_parameter('follow_distance', 10.0)
        self.declare_parameter('emergency_distance', 4.0)
        self.declare_parameter('emergency_ttc_sec', 1.5)
        self.declare_parameter('obstacle_lateral_limit', 2.0)
        self.declare_parameter('obstacle_min_forward_distance', 0.5)
        self.declare_parameter('obstacle_max_forward_distance', 25.0)
        self.declare_parameter('obstacle_min_vertical', -1.5)
        self.declare_parameter('obstacle_max_vertical', 0.8)
        self.declare_parameter('pointcloud_sample_step', 4)

        # --- Front obstacles from the depth cloud --------------------------
        # /perception/pcl/points_preprocessed is the *raw* depth cloud: the
        # obstacle detector publishes the identical message on `.../points`
        # and `.../points_preprocessed`, so no ground removal, no wall
        # removal, no clustering has happened and every gate is ours. The
        # gates below are named like the red_light_* block so that nothing in
        # the shared `/**` section of params.yaml can reach them.
        self.declare_parameter('front_obstacle_pointcloud_enabled', True)
        # The shared obstacle_timeout_sec (0.5 s) is shorter than the gap
        # between messages on *either* obstacle source - the depth cloud is
        # unpacked in a Python loop and the car YOLO runs at well under 2 Hz.
        # The distance therefore expired between messages and reappeared on
        # the next one, which at 10 Hz produced the alternating
        # `front obstacle 5.9 m` / `front obstacle none` state flapping (and
        # with it a target speed oscillating between 0.0 and 3.5 m/s).
        self.declare_parameter('front_obstacle_timeout_sec', 1.5)
        # Half-width of the corridor we treat as "our lane", in metres. The
        # shared obstacle_lateral_limit of 2.0 m reaches into the oncoming
        # lane, so a car parked there stopped us at waypoint 90. The vehicle
        # is ~1.8 m wide, so 1.2 m keeps a real in-lane obstacle while a
        # 3 m-away oncoming car falls outside.
        self.declare_parameter('front_obstacle_corridor_half_width', 1.2)
        # Bend the corridor towards the current waypoint instead of scanning
        # straight along the camera axis. In a corner the camera points at
        # the outside wall/guardrail, which a straight corridor reads as a
        # front obstacle at ~10 m and follows down to a standstill (waypoint
        # 107 stalled facing a building with an empty road curving away;
        # waypoint 98 stuck on a right-hand corner the same way). The skew
        # angle is the bearing to /planning/next_waypoint, clamped below.
        self.declare_parameter('front_obstacle_corridor_follow_waypoint', True)
        self.declare_parameter('front_obstacle_corridor_max_skew_deg', 30.0)
        # Emergency brake hysteresis: engage at emergency_distance, release
        # only beyond this. Without a gap the state chatters on the boundary.
        self.declare_parameter('front_obstacle_emergency_release_distance', 6.0)
        # Same idea for FOLLOW_VEHICLE: engage at follow_distance, release
        # only this much beyond it. At waypoint 106 a wall read 9.8-10.4 m
        # across consecutive cloud scans and the state flipped
        # CRUISE/FOLLOW_VEHICLE every publish cycle for four seconds.
        self.declare_parameter('front_obstacle_follow_release_margin_m', 1.0)
        # Safety valve, mirroring red_light_latch_timeout_sec: never hold a
        # stop on an obstacle that never moves. At waypoint 84 a 5.4 m reading
        # held the car for 19 s with nothing able to break the deadlock. After
        # this long, drop to a creep so the car can steer clear. The
        # controller's reverse/unstick recovery was removed in edf3c65, so
        # this creep is now the *only* mechanism that gets a wedged car
        # moving again - do not disable it.
        self.declare_parameter('front_obstacle_stop_timeout_sec', 6.0)
        # A single noisy depth sample used to be enough for a full emergency
        # brake, because the corridor scan returned its *minimum* forward
        # distance. Require a real cluster, then take a low percentile of it.
        self.declare_parameter('front_obstacle_min_points', 25)
        self.declare_parameter('front_obstacle_distance_percentile', 10.0)
        # Consecutive control cycles the emergency condition must hold.
        self.declare_parameter('front_obstacle_emergency_confirm_count', 3)
        # The car YOLO's scalar distance carries no lateral position, and its
        # own image-angle gate (car_max_lateral_angle_ratio 0.45) reaches
        # well into the oncoming lane: at waypoint 64 an oncoming car at
        # 15.8 m was reported and braked for. The only lateral information
        # available here is the depth-cloud corridor scan (half-width
        # front_obstacle_corridor_half_width = "directly ahead of us"), so
        # only trust a car reading when the corridor also sees something at a
        # similar distance. An off-lane car leaves the corridor empty (or on
        # a far wall) and the reading is discarded. Falls back to trusting
        # the car reading when the point-cloud channel is disabled.
        self.declare_parameter('front_obstacle_car_requires_cloud_match', True)
        self.declare_parameter('front_obstacle_car_cloud_match_tolerance_m', 4.0)
        # Since 9450b0b perception also publishes /perception/car_detections
        # with a world position per car, so each car can be gated on its
        # actual vehicle-frame lateral offset - a direct answer instead of
        # the cloud-corroboration workaround above (which stays as the
        # fallback for cars whose depth sample failed). A car whose center
        # is within this many metres of our centerline overlaps our lane.
        self.declare_parameter('front_car_lateral_limit', 1.8)
        # Waypoint zones where the emergency-brake *hard stop* is suppressed;
        # FOLLOW_VEHICLE slowdown stays active, so a close obstacle still
        # caps the speed near minimum_follow_speed instead of freezing the
        # car. Requested by control for the right turn at waypoint 98: the
        # trajectory clears the guardrail, but the rail enters the emergency
        # corridor mid-turn and the stop-hold-creep cycle wedged the car -
        # made worse now that the controller's reverse recovery is gone
        # (edf3c65). Scoped to the turns, not a global off-switch. 133-135
        # and 140-142 added on request too: repeated false emergencies there
        # (an oncoming car flickering at 1.5 m, a kerb-side cloud reading at
        # 2.4 m) kept wedging the car; FOLLOW slowdown still covers a real
        # car ahead in those stretches.
        self.declare_parameter(
            'front_obstacle_emergency_suppress_zones', '97-100,133-135,140-142'
        )
        # Places where static scenery enters the depth-cloud follow corridor.
        # This suppresses FOLLOW_VEHICLE only; emergency braking remains
        # available unless the same waypoint is also in the emergency list.
        self.declare_parameter('front_obstacle_follow_suppress_zones', '')
        # Below this speed an emergency brake brakes nothing - the car is
        # already stopped - so the hard stop is published as stop_request
        # only and the emergency flag is dropped. (This used to also protect
        # the controller's reverse recovery from being cancelled; that
        # recovery is gone since edf3c65, but the flag still gates controller
        # behavior, so keep the semantics.)
        self.declare_parameter('front_obstacle_emergency_min_speed', 0.4)

        # --- End of route ---------------------------------------------------
        # Nothing upstream signals "route complete": waypoint_manager._next_index
        # returns the same index once it runs out of waypoints, so it keeps
        # publishing the final one, the planner keeps aiming at it, and with no
        # red light or obstacle in sight this node keeps asking for cruise
        # speed. The car then drives past the last waypoint and rolls away.
        #
        # Detected without knowing the route: the target waypoint sits *behind*
        # us while the index has stopped advancing. Mid-route the manager
        # switches as soon as we pass a waypoint, so that combination only
        # happens at the end of the route (or on a planning failure, where
        # stopping is also the right answer). Waiting at a red light does not
        # trigger it - the index freezes, but the target stays ahead.
        self.declare_parameter('route_end_stop_enabled', True)
        self.declare_parameter('route_end_behind_distance', 1.0)
        self.declare_parameter('route_end_stall_sec', 3.0)

        # --- Startup ------------------------------------------------------
        # The detector loads YOLO weights long after the rest of the stack is
        # up, and the spawn sits inside the first light zone (waypoints 1-5).
        # Without this hold the car launches blind at cruise speed and meets
        # its first red at 14 m - the stop then lands in the middle of the
        # junction. Hold the car until the first message arrives on
        # /perception/traffic_light_detections (any content counts: it only
        # proves the detector is alive), or until the timeout, so a crashed
        # detector cannot pin the car at spawn forever.
        self.declare_parameter('startup_hold_for_traffic_lights', True)
        self.declare_parameter('startup_hold_timeout_sec', 30.0)
        self.declare_parameter('startup_mapped_light_confirm_count', 2)

        # --- Red-light handling -------------------------------------------
        self.declare_parameter('red_light_enabled', True)
        # Calibrated against the detector's *YOLO bulb* path, which is what
        # publishes colored detections whenever ultralytics is importable and
        # traffic_light_yolo_classify_mode is "bulb" (the default):
        #
        #   confidence = clip(0.4 * yolo_conf + 0.6 * quality, 0, 0.9)
        #
        # and `quality` is always 0.9 there, because a bulb whose vertical
        # position falls outside its expected band is discarded rather than
        # downgraded. With traffic_light_yolo_confidence_threshold = 0.35 that
        # puts every RED/GREEN detection in [0.68, 0.90].
        #
        # So 0.55 passes every real bulb-mode detection, while still rejecting
        # the HSV/housing fallback's much noisier 0.35-0.54 band - the band
        # that produced the phantom "red light 44.0 m ahead, confidence 0.42"
        # at waypoint 92 while ultralytics was missing. If YOLO ever drops out
        # again the effect is a loud `low_conf` reject log instead of a
        # phantom stop.
        self.declare_parameter('red_light_confidence_threshold', 0.55)
        # Detections arrive at camera rate (~2 Hz, slower with YOLO at
        # imgsz=960), so this must span several frames. The shared
        # `traffic_light_timeout_sec` of 1.0 s would expire between frames.
        self.declare_parameter('red_light_detections_timeout_sec', 3.0)
        # Comma-separated waypoint indices / inclusive ranges where the route
        # actually has traffic lights; empty string disables zone gating.
        # Mirrors the zones in params.yaml's traffic_light_waypoint_overrides
        # and is needed here because the detections array carries no zone
        # gating of its own, while the rest of the scene produces frequent
        # red false positives (signs, brake lights, facade reflections).
        self.declare_parameter(
            'red_light_waypoint_zones',
            '1-5,70-77,87-101,126-131,145-150',
        )
        # Once a red is confirmed we keep approaching it on odometry alone:
        # its world position is fixed, and the detector drops frames (HSV
        # fallback, ~2 Hz camera). Dropping the target on the first gap is
        # what produced "Red light lost before stopping; resuming cruise"
        # 5 s after acquiring a 0.92-confidence red at 25 m.
        self.declare_parameter('red_light_target_hold_sec', 12.0)
        # Geometry gates, in the vehicle frame (x forward, y left, z up).
        self.declare_parameter('red_light_min_forward_distance', 1.0)
        self.declare_parameter('red_light_max_forward_distance', 45.0)
        # Far detections are back-projected from a single depth sample, so a
        # bbox whose depth lands on the background reads tens of metres too
        # far (a 0.42-confidence "red light 44.0 m ahead" in the waypoint-92
        # log). Beyond red_light_far_distance_m demand a stronger detection.
        # Measured far-detection confidences on this route cluster tightly:
        # 0.72, 0.77, 0.80, 0.81, 0.82, 0.84. At 0.80 the gate sat in the
        # middle of that cluster and passed or rejected essentially at random
        # (waypoint 145 logged `far_low_conf(40.5m @ 0.80)` one second before
        # tracking the same light at 0.82). Put it below the cluster instead,
        # so it only excludes detections that are genuinely weak.
        self.declare_parameter('red_light_far_distance_m', 28.0)
        self.declare_parameter('red_light_far_confidence_threshold', 0.70)
        # Heads in this world are pole-mounted on the far corner of the
        # junction, so on the approach their measured lateral offset is
        # routinely 5-13 m (waypoints 73-76 logged off_corridor rejects at
        # 7.3, 9.5, 12.5 and 13.0 m for the junction's own light - it never
        # got tracked and the car drove through at cruise speed). 7.0 was
        # sized for mast-arm heads hanging over the lane; 13.5 accepts the
        # corner poles. Discriminating our head from the crossing road's is
        # NOT this gate's job anymore - red_light_stop_max_lateral_m decides
        # stop-vs-crawl at the stop point, and a tracked-but-off-axis head
        # costs a crawl-through, not a false stop.
        self.declare_parameter('red_light_lateral_limit', 13.5)
        # Once latched we are stopped at the line and the head is high, close
        # and off to the side, so its single-sample depth back-projection puts
        # it well outside the driving corridor: at waypoint 75 the same light
        # we had just tracked at 9.2 m started reading `off_corridor(lat
        # 10.5m)`, which meant no green could ever reach the release logic and
        # the stop ended on the 20 s timeout. While latched, widen the gate and
        # let world-position association (red_light_green_release_radius) do
        # the discrimination instead of the corridor.
        self.declare_parameter('red_light_latched_lateral_limit', 15.0)
        self.declare_parameter('red_light_min_height_offset', -2.0)
        self.declare_parameter('red_light_max_height_offset', 15.0)
        # Stop this far short of the light's own position. Larger values also
        # keep the light inside the camera frame while stopped, which is what
        # lets the green release fire instead of the timeout. At 6.0 every
        # latched stop in the logs sat with the housing clipped at the top of
        # the image (bulb color unclassifiable, so no green could release the
        # latch and every red cost the full 20 s timeout - waypoints 4, 150).
        self.declare_parameter('red_light_stop_offset', 8.0)
        # Per-zone overrides ("lo-hi:metres,..."): heads mounted on the far
        # side of a wide junction need a larger offset, or "8 m before the
        # light" is still inside the junction - waypoint 77 stopped in the
        # middle of the road that way. 12.0 for 70-77 is a first guess;
        # measure against the stop line and tune.
        self.declare_parameter('red_light_stop_offset_overrides', '70-77:12.0')
        # Fixed map stop points ("lo-hi:waypoint:x:y,...").  In these zones
        # perception still decides whether the light is red, but its noisy
        # depth-derived world position no longer decides where the car stops.
        # Waypoint 75 is (226.454, 6.347) in waypoints.yaml.  Keeping the
        # coordinates here means braking can begin while wp71-74 is active;
        # /planning/next_waypoint would not expose wp75 until too late.
        self.declare_parameter(
            'red_light_fixed_stop_points',
            '70-77:75:226.454:6.347',
        )
        # Optional map-anchored signal heads ("lo-hi:x:y:z,...").  In a
        # mapped zone, only the colored detection closest to the static
        # head's projected image x is accepted.  The fixed map position then
        # replaces the unreliable depth point for distance and association.
        self.declare_parameter('red_light_mapped_heads', '')
        self.declare_parameter('red_light_mapped_head_max_x_error', 0.06)
        self.declare_parameter('red_light_stop_tolerance', 0.7)
        # Only commit a stop to a head within this *lateral offset* of our
        # axis. This used to be a bearing gate (35 deg), but bearing diverges
        # at close range: our own corner-pole head sits ~5 m lateral, which
        # at the 6-7 m stop distance is a bearing of 37-40 deg - the gate
        # refused the stop at waypoint 130 (lateral +5.3 m, bearing +38 deg)
        # and the car crawled through the red. Lateral offset does not grow
        # as we close in: own heads latch at |lat| 0.5-5.6 m in the logs,
        # heads that belong to other approaches read 7 m and beyond. Beyond
        # the limit the head gets a crawl, not a stop, same as before.
        self.declare_parameter('red_light_stop_max_lateral_m', 6.5)
        # Image-center confirmation: a red seen within this offset of the
        # image center (normalized, 0.5 = half the image width) while still
        # at least red_light_image_confirm_min_forward_m away is *our* head,
        # whatever the depth back-projection claims - image position needs
        # no depth. Such a target may always be stopped for; the lateral
        # gate above then only vetoes heads that never looked centered.
        # 0.15, not 0.25: our own heads confirm at x=0.46-0.56 (offset
        # <=0.06), while 0.25 let the *cross-street* pole head at waypoint 2
        # through at x=0.27 (offset 0.23) and blessed a stop at a green
        # light. Its measured lateral (+8.3 m) was correct all along.
        self.declare_parameter('red_light_image_center_confirm_offset', 0.15)
        self.declare_parameter('red_light_image_confirm_min_forward_m', 10.0)
        self.declare_parameter('red_light_off_axis_speed', 1.5)
        # How far past red_light_stop_offset we still accept as a stop. Beyond
        # this the light was found too late and stopping would only block the
        # junction, so crawl through instead.
        self.declare_parameter('red_light_max_stop_overshoot', 1.0)
        # Once the CoM is this far beyond a mapped fixed stop point, the car
        # has committed to the junction. Ignore later color changes until the
        # waypoint zone is exited instead of crawling or stopping mid-junction.
        self.declare_parameter('red_light_passed_fixed_stop_margin_m', 0.5)
        # Comfort deceleration for the approach ramp: v = sqrt(2*a*s).
        self.declare_parameter('red_light_decel', 0.8)
        # Consecutive *detection messages* (not control cycles) that must
        # agree before a red starts being tracked.
        self.declare_parameter('red_light_confirm_count', 2)
        # Only a head essentially straight ahead may become a NEW target:
        # measured lateral within this limit, or seen near the image center
        # (red_light_image_center_confirm_offset - depth-independent, covers
        # a centered head whose localization is off). Side heads are never
        # acquired at all; see select_red's lock-on contract.
        self.declare_parameter('red_light_acquire_lateral_limit', 4.0)
        # Two observations count as the same physical light when their world
        # positions are within this horizontal radius. 3.0, not 6.0: at
        # waypoint 97 neighboring heads sit ~5.3 m apart (world 128.4/3.4 vs
        # 129.1/-1.9), so 6.0 merged them and the locked target wandered from
        # head to head across the junction, re-latching each time.
        self.declare_parameter('red_light_associate_radius', 3.0)
        # Greens are associated a little more loosely than reds, but only a
        # little: neighboring heads at one junction sit ~5.3 m apart here, so
        # the old 8-12 m radii let *another* head's green release a stop at a
        # head that was still red (waypoint 2 crossed on red that way, and at
        # waypoint 150 a left-side green released the stop). Re-sightings of
        # the same head land within ~1 m in this run's logs, so 4.0 has
        # margin without reaching the neighbors.
        self.declare_parameter('red_light_green_release_radius', 4.0)
        # ... and when even that fails, optionally any green that passes the
        # same corridor/height/confidence gates releases a latched stop.
        # Default OFF: at waypoint 150 this fallback let a *left-side* green
        # release the stop while our own head was still red. With the
        # tightened same-head association above, a genuine green at our head
        # releases normally; when perception truly cannot re-see the head,
        # the latch timeout is the intended (if slow) way out.
        self.declare_parameter('red_light_any_green_releases_latch', False)
        # Our own approach's pole-mounted heads measure lateral +4.4 to
        # +5.0 m in this world (waypoints 88, 92, 150), so the previous
        # 3.0 m gate rejected our own green and left the latch to time out
        # at a green light. 6.0 clears the pole heads; the crossing road's
        # heads localize farther out than that (7-11 m in the logs).
        self.declare_parameter('red_light_green_release_lateral_limit', 6.0)
        # Consecutive detection messages a green must persist for before it
        # releases a stop. Load-bearing since the detector dropped its own
        # temporal filtering (see green_release_confirmed).
        self.declare_parameter('red_light_green_release_confirm_count', 2)
        # Dropping a *tracked, not yet latched* red on a green demands more
        # frames than releasing a latched stop. The lights here cycle fast,
        # and worse, the bulb color flickers: at waypoint 74 a red tracked
        # for 8 s "turned green" for all of 1.2 s and was red again - no real
        # green phase is that short. Two frames (~1 s at camera rate) passed
        # that flicker and threw away the tracked target; by re-acquisition
        # the light was 6.8 m away, far too close to stop for. Three frames
        # (~1.5 s) outlasts the flicker while a genuine green phase (~8 s
        # here) still drops the target promptly.
        self.declare_parameter('red_light_green_drop_confirm_count', 3)
        self.declare_parameter('red_light_require_housing', False)
        # Safety valve: never sit at a latched stop longer than this.
        self.declare_parameter('red_light_latch_timeout_sec', 20.0)
        # After a release, ignore reds for this long, so the head we are
        # driving under (and any opposite-facing head at the same
        # intersection) cannot immediately re-latch us inside the junction.
        self.declare_parameter('red_light_pass_grace_sec', 8.0)
        # ... unless *the same head we just cleared* went red again while its
        # stop point is still comfortably ahead. The lights here cycle in
        # seconds: at waypoint 2 the stop released on green and the light was
        # red again before the car reached the junction - the blanket grace
        # blinded it and it crossed on red. Same-head-only matters: under the
        # 87-101 overpass the heads sit ~10 m apart, and letting *any*
        # stoppable red cancel the grace made the car re-stop at every head
        # in the row ("crossed the junction in several installments").
        self.declare_parameter('red_light_grace_red_override_margin_m', 2.0)
        # Grace used instead when a *tracked, not yet latched* red is dropped
        # because it turned green: we are still short of the stop line, so if
        # the light cycles back to red before we arrive we must be able to
        # re-acquire it immediately. Applying the full pass grace here is what
        # ran the light at waypoint 130 (green drop at 30 m, reds ignored for
        # 8 s, re-tracked at 11 m - too late) and forced the last-moment stop
        # at waypoint 150 (green drop at 42.7 m, re-tracked at exactly
        # grace-expiry, 13.1 m out).
        self.declare_parameter('red_light_green_drop_grace_sec', 1.0)

        self.node_started_at = self.get_clock().now()
        self.startup_hold_released = False
        self.mission_started_at = None
        self.mission_completion_time_sec: Optional[float] = None
        self.mission_completion_reported = False
        self.startup_mapped_light_state = TrafficLightState.UNKNOWN
        self.startup_mapped_light_count = 0
        self.startup_mapped_light_ready = False
        self.current_pose = None
        self.current_speed = 0.0
        self.latest_detections: List = []
        self.latest_detections_time = None
        self.latest_rejects: List[str] = []
        self.latest_waypoint_index: Optional[int] = None
        self.waypoint_index_changed_at = None
        self.next_waypoint_pose = None
        self.route_finished = False
        self.nearest_front_obstacle_m: Optional[float] = None
        self.latest_obstacle_time = None
        self.nearest_front_car_m: Optional[float] = None
        self.latest_front_car_time = None
        # True while the stored car distance came from a laterally-gated
        # per-car detection (no cloud corroboration needed then).
        self.front_car_laterally_gated = False
        # Once the per-car array is flowing, the scalar topic is redundant
        # (same detector, no lateral information) and must not overwrite it.
        self.car_detections_seen = False
        self.current_state = BehaviorState.CRUISE
        self.emergency_confirm_count = 0
        self.emergency_engaged = False
        self.follow_engaged = False
        self.obstacle_stop_since = None
        self.obstacle_stop_released = False

        self.red_light_zones = self.parse_waypoint_zones(
            str(self.get_parameter('red_light_waypoint_zones').value)
        )
        self.emergency_suppress_zones = self.parse_waypoint_zones(
            str(self.get_parameter('front_obstacle_emergency_suppress_zones').value)
        )
        self.follow_suppress_zones = self.parse_waypoint_zones(
            str(self.get_parameter('front_obstacle_follow_suppress_zones').value)
        )
        self.stop_offset_overrides = self.parse_zone_values(
            str(self.get_parameter('red_light_stop_offset_overrides').value)
        )
        self.fixed_stop_points = self.parse_fixed_stop_points(
            str(self.get_parameter('red_light_fixed_stop_points').value)
        )
        self.mapped_traffic_lights = self.parse_mapped_traffic_lights(
            str(self.get_parameter('red_light_mapped_heads').value)
        )
        self.traffic_light_camera_info: Optional[CameraInfo] = None
        self.red_target: Optional[TrafficLightTarget] = None
        self.red_confirm_count = 0
        self.green_confirm_count = 0
        self.ignore_red_until = None
        # Once a latched stop is released, ignore every further color change
        # in that same intersection zone until the route leaves it.  This is
        # stronger and more deterministic than a fixed-duration grace: a
        # short light cycle must not re-latch the car halfway through wp1-5
        # (or any of the other configured junction zones).
        self.ignore_red_zone: Optional[Tuple[int, int]] = None
        # World position of the most recently released target, for the
        # same-head check in grace_overridden_by_red.
        self.last_released_xyz: Optional[Tuple[float, float, float]] = None

        self.state_pub = self.create_publisher(
            String,
            self.get_parameter('decision_state_topic').value,
            10,
        )
        self.target_speed_pub = self.create_publisher(
            Float32,
            self.get_parameter('decision_target_speed_topic').value,
            10,
        )
        self.stop_request_pub = self.create_publisher(
            Bool,
            self.get_parameter('decision_stop_request_topic').value,
            10,
        )
        self.emergency_brake_pub = self.create_publisher(
            Bool,
            self.get_parameter('decision_emergency_brake_topic').value,
            10,
        )
        # Distance to the tracked light (NaN when none), for rqt_plot tuning.
        self.traffic_light_distance_pub = self.create_publisher(
            Float32,
            self.get_parameter('decision_traffic_light_distance_topic').value,
            10,
        )

        self.create_subscription(
            PoseStamped,
            self.get_parameter('pose_topic').value,
            self.pose_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            self.get_parameter('twist_topic').value,
            self.twist_callback,
            10,
        )
        self.create_subscription(
            TrafficLightDetections,
            self.get_parameter('traffic_light_detections_topic').value,
            self.traffic_light_detections_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('traffic_light_camera_info_topic').value,
            self.traffic_light_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Int32,
            self.get_parameter('waypoint_index_topic').value,
            self.waypoint_index_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('next_waypoint_topic').value,
            self.next_waypoint_callback,
            10,
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter('obstacle_points_topic').value,
            self.obstacle_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32,
            self.get_parameter('front_car_distance_topic').value,
            self.front_car_distance_callback,
            10,
        )
        self.create_subscription(
            CarDetections,
            self.get_parameter('car_detections_topic').value,
            self.car_detections_callback,
            10,
        )

        publish_rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / publish_rate_hz, self.publish_decision)

        self.get_logger().info(
            'Behavior state machine ready. Red-light waypoint zones: '
            f'{self.red_light_zones or "all waypoints"}'
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def twist_callback(self, msg: TwistStamped) -> None:
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z
        self.current_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    def traffic_light_detections_callback(self, msg: TrafficLightDetections) -> None:
        self.latest_detections = list(msg.detections)
        self.latest_detections_time = self.get_clock().now()
        # Per-frame work (confirmation counting, association, green release)
        # runs here rather than in the 10 Hz control timer: the timer would
        # re-count the same cached camera frame several times over and make
        # red_light_confirm_count meaningless.
        self.process_detections(self.latest_detections_time)

    def traffic_light_camera_info_callback(self, msg: CameraInfo) -> None:
        self.traffic_light_camera_info = msg

    def waypoint_index_callback(self, msg: Int32) -> None:
        index = int(msg.data)
        if index != self.latest_waypoint_index:
            self.latest_waypoint_index = index
            self.waypoint_index_changed_at = self.get_clock().now()

    def next_waypoint_callback(self, msg: PoseStamped) -> None:
        self.next_waypoint_pose = msg

    def obstacle_callback(self, msg: PointCloud2) -> None:
        if not bool(self.get_parameter('front_obstacle_pointcloud_enabled').value):
            self.nearest_front_obstacle_m = None
            self.latest_obstacle_time = None
            return
        distance = self.find_nearest_front_obstacle(msg)
        if distance is None:
            # "No cluster in this frame" is a missing measurement, not a
            # cleared road - see front_car_distance_callback.
            return
        self.nearest_front_obstacle_m = distance
        self.latest_obstacle_time = self.get_clock().now()

    def car_detections_callback(self, msg: CarDetections) -> None:
        """
        Per-car detections with world positions (perception 9450b0b).

        A car with a world position is kept only when its vehicle-frame
        lateral offset overlaps our lane (front_car_lateral_limit) - the
        real fix for the oncoming-lane braking the scalar topic could not
        distinguish. Cars without a world position fall back to the old
        cloud-corroboration path. An empty frame is a missing measurement,
        not a cleared road (same policy as the other obstacle sources).
        """
        self.car_detections_seen = True
        min_forward = float(self.get_parameter('obstacle_min_forward_distance').value)
        lateral_limit = float(self.get_parameter('front_car_lateral_limit').value)

        gated_nearest: Optional[float] = None
        ungated_nearest: Optional[float] = None
        for detection in msg.detections:
            if not detection.accepted or not math.isfinite(detection.distance):
                continue
            distance = float(detection.distance)
            offset = None
            if detection.has_world_position:
                offset = self.vehicle_frame_offset((
                    float(detection.world_x),
                    float(detection.world_y),
                    float(detection.world_z),
                ))
            if offset is not None:
                forward_m, lateral_m, _height = offset
                if forward_m < min_forward:
                    continue
                if abs(lateral_m) > lateral_limit:
                    continue
                if gated_nearest is None or distance < gated_nearest:
                    gated_nearest = distance
            else:
                if ungated_nearest is None or distance < ungated_nearest:
                    ungated_nearest = distance

        if gated_nearest is not None:
            self.nearest_front_car_m = gated_nearest
            self.front_car_laterally_gated = True
            self.latest_front_car_time = self.get_clock().now()
        elif ungated_nearest is not None:
            self.nearest_front_car_m = ungated_nearest
            self.front_car_laterally_gated = False
            self.latest_front_car_time = self.get_clock().now()

    def front_car_distance_callback(self, msg: Float32) -> None:
        if self.car_detections_seen:
            # The per-car array carries the same distances plus lateral
            # information; letting the scalar overwrite it would reintroduce
            # the ungated readings.
            return
        if not math.isfinite(msg.data):
            # Both detectors drop in and out frame to frame, and overwriting
            # the last reading with "nothing" on every miss made the fused
            # distance alternate between a value and None at the full 10 Hz
            # control rate, dragging the target speed between 3.5 and ~1 m/s
            # for most of the route. Treat a miss as no new information and
            # let the previous reading expire through
            # front_obstacle_timeout_sec instead.
            return
        self.nearest_front_car_m = float(msg.data)
        self.front_car_laterally_gated = False
        self.latest_front_car_time = self.get_clock().now()

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def publish_decision(self) -> None:
        now = self.get_clock().now()
        self.update_red_light_target(now)

        state, target_speed, stop_request, emergency_brake = self.evaluate()

        if state == BehaviorState.ROUTE_COMPLETE:
            self.report_mission_completion()

        # Without this the only visible symptom of a bad obstacle gate is the
        # car crawling or refusing to move, with nothing in the log to say
        # which branch did it.
        if state != self.current_state:
            front_distance, front_source = self.current_front_obstacle()
            front_text = (
                'none' if front_distance is None
                else f'{front_distance:.1f} m ({front_source})'
            )
            self.get_logger().info(
                f'{self.current_state.value} -> {state.value} '
                f'(target {target_speed:.1f} m/s, front obstacle {front_text}, '
                f'waypoint {self.latest_waypoint_index})'
            )

        self.current_state = state
        self.publish_string(self.state_pub, state.value)
        self.publish_float(self.target_speed_pub, target_speed)
        self.publish_bool(self.stop_request_pub, stop_request)
        self.publish_bool(self.emergency_brake_pub, emergency_brake)
        self.publish_float(
            self.traffic_light_distance_pub,
            self.red_target.forward_m
            if self.red_target is not None and self.red_target.forward_m is not None
            else float('nan'),
        )

    def evaluate(self) -> Tuple[BehaviorState, float, bool, bool]:
        cruise_speed = float(self.get_parameter('target_speed').value)

        if self.route_end_reached():
            return BehaviorState.ROUTE_COMPLETE, 0.0, True, False

        if self.startup_hold_active():
            return BehaviorState.CRUISE, 0.0, True, False

        if self.should_emergency_brake():
            if self.obstacle_stop_expired():
                # Blocked long enough that holding still is not working.
                # Creep instead: it lets the controller steer, and it lets its
                # unstick recovery arm (which a stop_request suppresses).
                creep = min(
                    cruise_speed,
                    float(self.get_parameter('minimum_follow_speed').value),
                )
                return BehaviorState.FOLLOW_VEHICLE, creep, False, False
            # Hold the hard stop, but only assert emergency_brake while there
            # is speed left to scrub off - see
            # front_obstacle_emergency_min_speed.
            hard_brake = self.current_speed > float(
                self.get_parameter('front_obstacle_emergency_min_speed').value
            )
            return BehaviorState.EMERGENCY_BRAKE, 0.0, True, hard_brake

        self.obstacle_stop_since = None
        self.obstacle_stop_released = False

        red_stop, red_speed = self.red_light_command(cruise_speed)
        if red_stop:
            return BehaviorState.STOP_FOR_RED_LIGHT, 0.0, True, False

        state = BehaviorState.CRUISE
        target_speed = cruise_speed
        if self.should_follow_vehicle():
            state = BehaviorState.FOLLOW_VEHICLE
            target_speed = self.compute_follow_speed(cruise_speed)
        if red_speed is not None and red_speed < target_speed:
            state = BehaviorState.APPROACH_RED_LIGHT
            target_speed = red_speed

        return state, target_speed, False, False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup_hold_active(self) -> bool:
        """Hold at spawn until the detector, and any mapped spawn light, is ready."""
        if self.startup_hold_released:
            return False
        if not bool(self.get_parameter('startup_hold_for_traffic_lights').value):
            self.startup_hold_released = True
            self.start_mission_timer()
            return False

        mapped_head = self.current_mapped_traffic_light()
        detector_ready = self.latest_detections_time is not None
        waypoint_ready = self.latest_waypoint_index is not None
        mapped_light_ready = mapped_head is None or self.startup_mapped_light_ready
        if detector_ready and waypoint_ready and mapped_light_ready:
            release_reason = (
                'the mapped traffic light has a confirmed color'
                if mapped_head is not None
                else 'the traffic-light detector is publishing'
            )
            self.get_logger().info(
                f'{release_reason.capitalize()}; releasing the startup hold.'
            )
            self.startup_hold_released = True
            self.start_mission_timer()
            return False

        elapsed = (self.get_clock().now() - self.node_started_at).nanoseconds * 1e-9
        timeout = float(self.get_parameter('startup_hold_timeout_sec').value)
        if elapsed > timeout:
            missing = (
                'a confirmed color from the mapped traffic light'
                if mapped_head is not None and detector_ready
                else 'traffic-light detections'
            )
            self.get_logger().warn(
                f'No {missing} after {timeout:.0f}s; departing without it. '
                'Check the detector and mapped-head calibration.'
            )
            self.startup_hold_released = True
            self.start_mission_timer()
            return False
        waiting_for = (
            'the mapped traffic light to have a confirmed red/green color'
            if mapped_head is not None and detector_ready
            else 'traffic-light detections and the initial waypoint'
        )
        self.get_logger().info(
            f'Holding at spawn while waiting for {waiting_for} '
            f'({elapsed:.0f}s elapsed).',
            throttle_duration_sec=5.0,
        )
        return True

    def start_mission_timer(self, now=None) -> None:
        """Start timing when the vehicle is first released from its spawn hold."""
        if self.mission_started_at is not None:
            return
        self.mission_started_at = now if now is not None else self.get_clock().now()
        self.get_logger().info('Mission timer started; the vehicle is released.')

    def report_mission_completion(self, now=None) -> None:
        """Log the elapsed mission time once when the route is completed."""
        if self.mission_completion_reported:
            return
        self.mission_completion_reported = True

        if self.mission_started_at is None:
            self.get_logger().warn(
                'Route completed before the mission timer was started; '
                'completion time is unavailable.'
            )
            return

        completed_at = now if now is not None else self.get_clock().now()
        elapsed = (completed_at - self.mission_started_at).nanoseconds * 1e-9
        self.mission_completion_time_sec = max(0.0, elapsed)
        self.get_logger().info(
            f'Mission completed in {self.mission_completion_time_sec:.2f} seconds.'
        )

    # ------------------------------------------------------------------
    # End of route
    # ------------------------------------------------------------------

    def route_end_reached(self) -> bool:
        """Report whether the route has run out of waypoints ahead of us."""
        if not bool(self.get_parameter('route_end_stop_enabled').value):
            return False
        if self.route_finished:
            return True
        if self.next_waypoint_pose is None or self.waypoint_index_changed_at is None:
            return False
        if self.is_fresh(
            self.waypoint_index_changed_at,
            float(self.get_parameter('route_end_stall_sec').value),
        ):
            # The index is still advancing, so there is route left.
            return False

        position = self.next_waypoint_pose.pose.position
        offset = self.vehicle_frame_offset((position.x, position.y, position.z))
        if offset is None:
            return False
        behind_distance = float(self.get_parameter('route_end_behind_distance').value)
        if offset[0] > -behind_distance:
            return False

        self.get_logger().warn(
            f'Final waypoint {self.latest_waypoint_index} is {-offset[0]:.1f} m behind '
            'us and the route is not advancing; holding a stop. Nothing upstream '
            'publishes a route-complete signal.'
        )
        self.route_finished = True
        return True

    # ------------------------------------------------------------------
    # Red light: per-frame association (runs on every detections message)
    # ------------------------------------------------------------------

    def process_detections(self, now) -> None:
        reds, greens = self.scan_detections()
        target = self.red_target
        best_red = self.select_red(reds, target)

        if target is not None and target.latched:
            self.log_latched_detections(reds, greens)
            # A red still visible at our own light outranks any green in the
            # frame (cross traffic, the next intersection down the road).
            if best_red is not None:
                # Keep following the matched head even after latching.  The
                # depth back-projection drifts by 1-2 m between frames at the
                # wp87-101 junction: the locked target stayed at its original
                # 8.6 m position while the same red walked through 9.0, 10.4
                # and 12.1 m.  When it turned green at 12.1 m it then fell
                # outside green_release_radius and the car waited for the
                # timeout.  Incremental same-head matching has already passed
                # select_red's associate-radius gate, so refreshing here keeps
                # the red->green transition associable without accepting an
                # arbitrary green in the frame.
                world_xyz, forward_m, confidence, _normalized_x = best_red
                target.world_xyz = world_xyz
                target.forward_m = forward_m
                target.confidence = confidence
                target.last_seen = now
                self.green_confirm_count = 0
            elif self.green_release_confirmed(target, greens):
                self.release_target(target, now, 'green light at the tracked head')
            return

        if not self.red_light_active(now):
            if best_red is not None and self.grace_overridden_by_red(now, best_red):
                self.get_logger().info(
                    f'Red light {best_red[1]:.1f} m ahead during the pass grace: '
                    'still room to stop for it (the light has cycled back to '
                    f'red), honoring it (waypoint {self.latest_waypoint_index}).'
                )
                self.ignore_red_until = None
            else:
                best_red = None

        if best_red is None:
            # Note that reds may still be non-empty here: unmatched, *farther*
            # heads. Those must not block a green release - at waypoint 95 the
            # crossing road's heads face our camera almost head-on and read
            # red while our own pole head reads green, so "any red in frame
            # blocks the release" left the car standing at a green light until
            # the latch timeout. A red that matches the tracked head itself
            # returns non-None from select_red and is handled below.
            if not reds:
                # Only a genuinely empty frame breaks a confirmation run.
                self.red_confirm_count = 0
            if target is not None and self.green_release_confirmed(target, greens):
                self.release_target(
                    target,
                    now,
                    'tracked light turned green',
                    grace_sec=float(
                        self.get_parameter('red_light_green_drop_grace_sec').value
                    ),
                )
            return

        # Mirror the latched branch: while *our own* head is visibly red
        # (best_red matched or superseded the target), any green in the frame
        # belongs to someone else.
        self.green_confirm_count = 0

        world_xyz, forward_m, confidence, normalized_x = best_red
        radius = float(self.get_parameter('red_light_associate_radius').value)
        refreshing = target is not None and self.same_light(target.world_xyz, world_xyz, radius)

        if not refreshing:
            # Acquiring a new head needs confirmation; refreshing one we are
            # already tracking must not, or a head that only shows up on
            # alternate frames would never get its distance updated.
            self.red_confirm_count += 1
            if self.red_confirm_count < int(
                self.get_parameter('red_light_confirm_count').value
            ):
                return
            target = TrafficLightTarget(world_xyz, now)
            self.red_target = target
            self.get_logger().info(
                f'Tracking red light {forward_m:.1f} m ahead '
                f'({self.describe_bearing(world_xyz)}, confidence {confidence:.2f}, '
                f'waypoint {self.latest_waypoint_index}).'
            )

        target.world_xyz = world_xyz
        target.forward_m = forward_m
        target.confidence = confidence
        target.last_seen = now

        if not target.image_confirmed and forward_m >= float(
            self.get_parameter('red_light_image_confirm_min_forward_m').value
        ) and abs(normalized_x - 0.5) <= float(
            self.get_parameter('red_light_image_center_confirm_offset').value
        ):
            target.image_confirmed = True
            self.get_logger().info(
                f'Red light confirmed in-lane by image position (x={normalized_x:.2f} '
                f'at {forward_m:.1f} m, waypoint {self.latest_waypoint_index}); '
                'the lateral stop gate will not veto it.'
            )

    def select_red(
        self,
        reds: List[LightObservation],
        target: Optional[TrafficLightTarget],
    ) -> Optional[LightObservation]:
        """
        Pick which red to react to under a lock-on contract.

        One junction, one head, once: only a head essentially straight ahead
        (red_acquirable) may become a new target, and a locked target is
        never switched to another head. Junctions here carry 3-4 heads only
        ~5 m apart running different phases, and every laxer policy failed
        in turn: nearest-first tracked the cross street's head (waypoint 2,
        stopped at our own green); switching to closer/in-lane heads made
        the target wander across the junction and re-latch at each head
        (waypoints 97, 130); side heads' greens released stops early
        (waypoint 150). Side heads are now ignored outright - they neither
        stop us nor release us. After the locked head is passed or released,
        the pass grace and zone gating keep the junction quiet behind us.
        """
        if not reds:
            return None

        if target is not None:
            # Lock-on: a frame either refreshes the same head or contributes
            # nothing. Never switch a locked target to another head - at
            # waypoints 97/130 the target wandered across the junction's
            # heads and re-latched at each one in turn.
            radius = float(self.get_parameter('red_light_associate_radius').value)
            matched = [
                observation
                for observation in reds
                if self.same_light(target.world_xyz, observation[0], radius)
            ]
            if matched:
                return min(matched, key=lambda observation: observation[1])
            return None

        acquirable = [obs for obs in reds if self.red_acquirable(obs)]
        if not acquirable:
            return None
        return min(acquirable, key=lambda observation: observation[1])

    def red_acquirable(self, observation: LightObservation) -> bool:
        """Check whether this red may become a new straight-ahead target."""
        world_xyz, _forward, _confidence, normalized_x = observation
        offset = self.vehicle_frame_offset(world_xyz)
        if offset is not None and abs(offset[1]) <= float(
            self.get_parameter('red_light_acquire_lateral_limit').value
        ):
            return True
        return abs(normalized_x - 0.5) <= float(
            self.get_parameter('red_light_image_center_confirm_offset').value
        )

    def scan_detections(self) -> Tuple[List[LightObservation], List[LightObservation]]:
        """Split the latest frame into the relevant reds and all greens."""
        self.latest_rejects = []
        if not self.is_fresh(
            self.latest_detections_time,
            float(self.get_parameter('red_light_detections_timeout_sec').value),
        ):
            return [], []
        if self.current_pose is None:
            self.get_logger().warn(
                'No vehicle pose yet; traffic lights cannot be localized.',
                throttle_duration_sec=5.0,
            )
            return [], []

        reds: List[LightObservation] = []
        greens: List[LightObservation] = []
        rejects: List[str] = []
        mapped_head = self.current_mapped_traffic_light()
        if mapped_head is not None:
            expected_x = self.mapped_head_normalized_x(mapped_head)
            candidates = []
            if expected_x is not None:
                for detection in self.latest_detections:
                    accepted = self.accept_mapped_detection(
                        detection,
                        mapped_head,
                        expected_x,
                        rejects,
                    )
                    if accepted is not None:
                        error = abs(float(detection.normalized_x) - expected_x)
                        candidates.append(
                            (error, -float(detection.confidence), detection, accepted)
                        )
            if candidates:
                _error, _negative_confidence, detection, accepted = min(
                    candidates,
                    key=lambda candidate: (candidate[0], candidate[1]),
                )
                self.update_startup_mapped_light_confirmation(detection.state)
                if detection.state == TrafficLightState.RED:
                    reds.append(accepted)
                else:
                    greens.append(accepted)
            else:
                self.update_startup_mapped_light_confirmation(
                    TrafficLightState.UNKNOWN
                )
                if expected_x is None:
                    rejects.append('mapped_head_behind')
        else:
            for detection in self.latest_detections:
                accepted = self.accept_detection(detection, rejects)
                if accepted is None:
                    continue
                if detection.state == TrafficLightState.RED:
                    reds.append(accepted)
                elif detection.state == TrafficLightState.GREEN:
                    greens.append(accepted)

        self.latest_rejects = rejects

        # Without this, a mis-tuned gate is indistinguishable from "perception
        # saw nothing" - both just look like the car driving through a red.
        if rejects and not reds and not greens and self.in_red_light_zone():
            self.get_logger().info(
                f'No usable traffic light in frame (waypoint {self.latest_waypoint_index}): '
                + ', '.join(rejects[:5]),
                throttle_duration_sec=5.0,
            )
        return reds, greens

    def update_startup_mapped_light_confirmation(self, state: int) -> None:
        """Require stable mapped-head color before releasing a spawn-zone hold."""
        if self.startup_hold_released or self.startup_mapped_light_ready:
            return
        if state not in (TrafficLightState.RED, TrafficLightState.GREEN):
            self.startup_mapped_light_state = TrafficLightState.UNKNOWN
            self.startup_mapped_light_count = 0
            return
        if state == self.startup_mapped_light_state:
            self.startup_mapped_light_count += 1
        else:
            self.startup_mapped_light_state = state
            self.startup_mapped_light_count = 1
        required = max(
            1,
            int(self.get_parameter('startup_mapped_light_confirm_count').value),
        )
        if self.startup_mapped_light_count >= required:
            self.startup_mapped_light_ready = True
            color = 'red' if state == TrafficLightState.RED else 'green'
            self.get_logger().info(
                f'Mapped spawn light confirmed {color} for '
                f'{self.startup_mapped_light_count} frames.'
            )

    def accept_mapped_detection(
        self,
        detection,
        mapped_head: MappedTrafficLight,
        expected_x: float,
        rejects: List[str],
    ) -> Optional[LightObservation]:
        """Bind one image detection to a static, map-anchored signal head."""
        if detection.state not in (TrafficLightState.RED, TrafficLightState.GREEN):
            rejects.append(f'uncolored({detection.confidence:.2f})')
            return None
        confidence = float(detection.confidence)
        if confidence < float(
            self.get_parameter('red_light_confidence_threshold').value
        ):
            rejects.append(f'low_conf({confidence:.2f})')
            return None
        if (
            bool(self.get_parameter('red_light_require_housing').value)
            and not detection.has_housing
        ):
            rejects.append('no_housing')
            return None

        normalized_x = float(detection.normalized_x)
        x_error = abs(normalized_x - expected_x)
        max_error = float(
            self.get_parameter('red_light_mapped_head_max_x_error').value
        )
        if x_error > max_error:
            rejects.append(
                f'other_head(x={normalized_x:.2f}, expected={expected_x:.2f})'
            )
            return None

        world_xyz = (mapped_head[2], mapped_head[3], mapped_head[4])
        offset = self.vehicle_frame_offset(world_xyz)
        if offset is None:
            rejects.append('no_pose')
            return None
        forward_m, _lateral_m, height_m = offset
        if forward_m < float(self.get_parameter('red_light_min_forward_distance').value):
            rejects.append(f'behind({forward_m:.1f}m)')
            return None
        if forward_m > float(self.get_parameter('red_light_max_forward_distance').value):
            rejects.append(f'too_far({forward_m:.1f}m)')
            return None
        if forward_m > float(self.get_parameter('red_light_far_distance_m').value) and (
            confidence
            < float(self.get_parameter('red_light_far_confidence_threshold').value)
        ):
            rejects.append(f'far_low_conf({forward_m:.1f}m @ {confidence:.2f})')
            return None
        if height_m < float(self.get_parameter('red_light_min_height_offset').value) or (
            height_m > float(self.get_parameter('red_light_max_height_offset').value)
        ):
            rejects.append(f'bad_height({height_m:.1f}m)')
            return None
        return world_xyz, forward_m, confidence, normalized_x

    def log_latched_detections(
        self,
        reds: List[LightObservation],
        greens: List[LightObservation],
    ) -> None:
        """
        Report what the camera sees while a stop is latched.

        scan_detections' reject log stays quiet whenever anything at all was
        accepted, so a latch that ends on the timeout leaves no trace of why
        no green arrived. Three stops in one run ended that way, one of them
        with the head almost dead ahead (bearing -4 deg) - which rules out the
        corridor gate and leaves nothing in the log pointing at the real
        cause. This closes that gap.
        """
        parts = [
            f'{name} {forward_m:.1f}m {self.describe_bearing(world_xyz)} '
            f'conf {confidence:.2f}'
            for name, observations in (('RED', reds), ('GREEN', greens))
            for world_xyz, forward_m, confidence, _normalized_x in observations
        ]
        if self.latest_rejects:
            parts.append('rejected: ' + ', '.join(self.latest_rejects[:4]))
        if not parts:
            # This method runs from the detections callback, so an empty
            # report means the detector published a frame with zero
            # detections - the light is out of frame or missed at close
            # range, not gated away here.
            parts.append('empty detections frame (light not seen by perception)')
        self.get_logger().info(
            f'Latched at waypoint {self.latest_waypoint_index}: ' + '; '.join(parts),
            throttle_duration_sec=3.0,
        )

    def describe_bearing(self, world_xyz: Tuple[float, float, float]) -> str:
        """Render a light's lateral offset and bearing for the log."""
        offset = self.vehicle_frame_offset(world_xyz)
        if offset is None:
            return 'bearing unknown'
        forward_m, lateral_m, _height = offset
        bearing_deg = math.degrees(math.atan2(lateral_m, max(0.1, forward_m)))
        return f'lateral {lateral_m:+.1f} m, bearing {bearing_deg:+.0f} deg'

    def accept_detection(self, detection, rejects: List[str]) -> Optional[LightObservation]:
        """
        Accept a detection as a light that applies to us, or reject it.

        Rejects anything that is not a confidently classified light standing
        ahead of the car, inside the lane corridor, at a plausible height.
        Detections without a world position are dropped: without depth there
        is no distance to brake against, and reacting to them would bring
        back the "stop for any red pixel anywhere" behavior.
        """
        if detection.state not in (TrafficLightState.RED, TrafficLightState.GREEN):
            # Not silent: at waypoint 4 the latched log claimed "empty
            # detections frame" while the debug image showed a detected
            # housing clipped at the top of the frame - the box was there,
            # its bulb color just could not be classified. That distinction
            # (detector dead vs. light half out of frame) decides who has to
            # fix it.
            rejects.append(f'uncolored({detection.confidence:.2f})')
            return None
        if detection.confidence < float(
            self.get_parameter('red_light_confidence_threshold').value
        ):
            rejects.append(f'low_conf({detection.confidence:.2f})')
            return None
        if (
            bool(self.get_parameter('red_light_require_housing').value)
            and not detection.has_housing
        ):
            rejects.append('no_housing')
            return None
        if not detection.has_world_position:
            rejects.append('no_depth')
            return None

        world_xyz = (
            float(detection.world_x),
            float(detection.world_y),
            float(detection.world_z),
        )
        if not all(math.isfinite(value) for value in world_xyz):
            rejects.append('bad_world_xyz')
            return None

        offset = self.vehicle_frame_offset(world_xyz)
        if offset is None:
            rejects.append('no_pose')
            return None

        forward_m, lateral_m, height_m = offset
        if forward_m < float(self.get_parameter('red_light_min_forward_distance').value):
            rejects.append(f'behind({forward_m:.1f}m)')
            return None
        if forward_m > float(self.get_parameter('red_light_max_forward_distance').value):
            rejects.append(f'too_far({forward_m:.1f}m)')
            return None
        if forward_m > float(self.get_parameter('red_light_far_distance_m').value) and (
            detection.confidence
            < float(self.get_parameter('red_light_far_confidence_threshold').value)
        ):
            rejects.append(f'far_low_conf({forward_m:.1f}m @ {detection.confidence:.2f})')
            return None
        latched = self.red_target is not None and self.red_target.latched
        lateral_limit = float(
            self.get_parameter(
                'red_light_latched_lateral_limit' if latched
                else 'red_light_lateral_limit'
            ).value
        )
        if abs(lateral_m) > lateral_limit:
            rejects.append(f'off_corridor(lat {lateral_m:.1f}m)')
            return None
        if height_m < float(self.get_parameter('red_light_min_height_offset').value) or (
            height_m > float(self.get_parameter('red_light_max_height_offset').value)
        ):
            rejects.append(f'bad_height({height_m:.1f}m)')
            return None
        return (
            world_xyz,
            forward_m,
            float(detection.confidence),
            float(detection.normalized_x),
        )

    def green_release_confirmed(
        self,
        target: TrafficLightTarget,
        greens: List[LightObservation],
    ) -> bool:
        """
        Debounce green-based releases over consecutive detection messages.

        The detector no longer debounces anything: params.yaml sets
        traffic_light_history_size / _confirm_count / _track_history /
        _track_min_count all to 1 ("responsiveness-first ... accept some
        additional flicker"). A single flickered green must therefore not be
        able to release a red-light stop from here. Dropping a not-yet-latched
        target demands one frame more than releasing a latched stop - see
        red_light_green_drop_confirm_count.
        """
        if not self.green_releases_target(target, greens):
            self.green_confirm_count = 0
            return False
        self.green_confirm_count += 1
        count_param = (
            'red_light_green_release_confirm_count' if target.latched
            else 'red_light_green_drop_confirm_count'
        )
        return self.green_confirm_count >= max(
            1,
            int(self.get_parameter(count_param).value),
        )

    def green_releases_target(
        self,
        target: TrafficLightTarget,
        greens: List[LightObservation],
    ) -> bool:
        if not greens:
            return False

        radius = float(self.get_parameter('red_light_green_release_radius').value)
        if any(
            self.same_light(target.world_xyz, world_xyz, radius)
            for world_xyz, _forward, _confidence, _normalized_x in greens
        ):
            return True

        if not target.latched or not bool(
            self.get_parameter('red_light_any_green_releases_latch').value
        ):
            return False

        # Fallback for a latched stop whose red was localized badly enough
        # that no green ever associates with it. Held to a much narrower
        # corridor than accept_detection's: cross-traffic heads sit off to the
        # side, and letting one of those release us is the failure this whole
        # node exists to avoid.
        lateral_limit = float(
            self.get_parameter('red_light_green_release_lateral_limit').value
        )
        for world_xyz, _forward, _confidence, _normalized_x in greens:
            offset = self.vehicle_frame_offset(world_xyz)
            if offset is not None and abs(offset[1]) <= lateral_limit:
                return True
        return False

    # ------------------------------------------------------------------
    # Red light: per-cycle tracking and command
    # ------------------------------------------------------------------

    def update_red_light_target(self, now) -> None:
        target = self.red_target
        if target is None:
            return

        # Keep the distance current between camera frames: the car keeps
        # moving, and close to the stop point most frames no longer contain
        # the light at all.
        offset = self.vehicle_frame_offset(target.world_xyz)
        target.forward_m = offset[0] if offset is not None else None

        if target.latched:
            if not self.in_red_light_zone():
                self.release_target(target, now, 'left the intersection waypoint zone')
                return
            timeout = float(self.get_parameter('red_light_latch_timeout_sec').value)
            if target.latched_at is not None and not self.is_fresh(target.latched_at, timeout):
                if self.current_mapped_traffic_light() is not None:
                    self.get_logger().warn(
                        f'Red-light stop held {timeout:.0f}s without seeing green at '
                        'the configured map head; continuing to hold. A missing or '
                        'uncolored mapped light is not permission to run a red light.',
                        throttle_duration_sec=5.0,
                    )
                    return
                self.get_logger().warn(
                    f'Red-light stop held {timeout:.0f}s without seeing green; '
                    'releasing and ignoring further changes in this intersection. '
                    'Check traffic-light perception at this intersection.'
                )
                self.release_target(target, now, 'latch timeout')
            return

        fixed_stop_overshoot = self.current_fixed_stop_overshoot()
        if fixed_stop_overshoot is not None:
            self.suppress_reds_after_fixed_stop_line(
                fixed_stop_overshoot,
                target=target,
            )
            return

        # A confirmed red is a fixed point in the world, so a gap in the
        # detections is not a reason to cancel the stop - we can keep closing
        # on it with odometry alone. Give up only if we have not seen it for
        # red_light_target_hold_sec, or if we have already driven past it.
        # In a fixed-stop zone the perceived light position is deliberately
        # not authoritative: the depth back-projection can jump behind the
        # vehicle even though the mapped stop point is still ahead.  Let the
        # fixed-stop command/overshoot handling own that decision instead.
        if (
            self.current_fixed_stop_point() is None
            and target.forward_m is not None
            and target.forward_m < float(
                self.get_parameter('red_light_min_forward_distance').value
            )
        ):
            # Must go through release_target: this is exactly the case
            # red_light_pass_grace_sec exists for. Clearing the target inline
            # skipped the grace window, so at waypoint 131 the *other* head of
            # the junction we were driving through re-latched us 2 s later,
            # 1.6 m past its own stop point, in the middle of the junction.
            self.release_target(target, now, 'we have driven past it')
            return

        if not self.is_fresh(
            target.last_seen,
            float(self.get_parameter('red_light_target_hold_sec').value),
        ):
            self.get_logger().info(
                'Red light not re-detected within the hold window; resuming cruise.'
            )
            self.red_target = None
            self.red_confirm_count = 0

    def red_light_command(self, cruise_speed: float) -> Tuple[bool, Optional[float]]:
        """Return (hard stop requested, speed cap imposed by the red light)."""
        target = self.red_target
        if target is None:
            return False, None
        if target.latched:
            return True, 0.0
        remaining, fixed_stop = self.red_light_stop_remaining(target.forward_m)
        if remaining is None:
            # Tracked, but not localizable this cycle (no pose): hold a
            # conservative cap rather than a stop.
            return False, min(cruise_speed, 1.5)

        if remaining <= float(self.get_parameter('red_light_stop_tolerance').value):
            # Found too late to stop where a stop belongs. Latching here parks
            # the car inside the junction and helps nobody: waypoint 131
            # acquired a head at 4.4 m with a 6.0 m stop offset, stopped 1.6 m
            # past the line, and then sat out the full 20 s timeout there.
            max_overshoot = float(
                self.get_parameter('red_light_max_stop_overshoot').value
            )
            if -remaining > max_overshoot:
                stop_text = (
                    f'fixed waypoint {fixed_stop[0]}'
                    if fixed_stop is not None
                    else 'depth-derived stop point'
                )
                self.get_logger().info(
                    f'Red light found too late to stop for ({-remaining:.1f} m past '
                    f'{stop_text}, active waypoint {self.latest_waypoint_index}); '
                    'crawling past.',
                    throttle_duration_sec=5.0,
                )
                return False, min(
                    cruise_speed,
                    float(self.get_parameter('red_light_off_axis_speed').value),
                )

            # A head far off to the side is not the one governing the lane we
            # are in - most often it belongs to another approach of a junction
            # we are already turning through. Committing a stop to it parks
            # the car mid-corner. Crawl past instead of latching. Judged by
            # lateral offset, not bearing - see red_light_stop_max_lateral_m.
            offset = self.vehicle_frame_offset(target.world_xyz)
            lateral_limit = float(
                self.get_parameter('red_light_stop_max_lateral_m').value
            )
            if offset is not None and abs(offset[1]) > lateral_limit:
                if target.image_confirmed:
                    # The head was seen at the image center from a distance,
                    # so the outsized lateral is a localization error, not a
                    # cross head. Stop for it - and leave a trace for the
                    # perception folks, because a centered light measuring
                    # metres of lateral offset means the depth
                    # back-projection is broken (waypoint 2: +8.4 m).
                    self.get_logger().warn(
                        'Stopping for an image-confirmed red light despite '
                        f'{self.describe_bearing(target.world_xyz)} - its world '
                        'position is suspect; check the depth back-projection.'
                    )
                else:
                    self.get_logger().info(
                        'Not stopping for a red light off to the side '
                        f'({self.describe_bearing(target.world_xyz)}, limit '
                        f'{lateral_limit:.1f} m lateral, waypoint '
                        f'{self.latest_waypoint_index}); crawling past.',
                        throttle_duration_sec=5.0,
                    )
                    return False, min(
                        cruise_speed,
                        float(self.get_parameter('red_light_off_axis_speed').value),
                    )

            target.latched = True
            target.latched_at = self.get_clock().now()
            if fixed_stop is not None:
                stop_waypoint, stop_x, stop_y = fixed_stop
                self.get_logger().info(
                    f'Stopping for red light at fixed waypoint {stop_waypoint} '
                    f'({stop_x:.3f}, {stop_y:.3f}); perceived light geometry is '
                    'not used for stop distance.'
                )
            else:
                self.get_logger().info(
                    f'Stopping for red light ({target.forward_m:.1f} m to the light, '
                    f'{self.describe_bearing(target.world_xyz)}).'
                )
            return True, 0.0

        decel = max(0.1, float(self.get_parameter('red_light_decel').value))
        return False, min(cruise_speed, math.sqrt(2.0 * decel * remaining))

    def release_target(
        self,
        target: TrafficLightTarget,
        now,
        reason: str,
        grace_sec: Optional[float] = None,
    ) -> None:
        """
        Drop the tracked light and ignore reds for a grace window.

        The default grace is red_light_pass_grace_sec, sized for driving
        *through* a junction (past-the-light drops, latch releases). Callers
        releasing a light we have not reached yet must pass a much shorter
        grace, or a red that reappears on the approach is ignored until it is
        too close to stop for.
        """
        released_from_latch = target.latched
        released_zone = self.current_red_light_zone() if released_from_latch else None
        if released_from_latch:
            self.get_logger().info(f'Releasing red-light stop: {reason}.')
        else:
            self.get_logger().info(f'Dropping tracked red light: {reason}.')
        self.last_released_xyz = target.world_xyz
        self.red_target = None
        self.red_confirm_count = 0
        self.green_confirm_count = 0
        if grace_sec is None:
            grace_sec = float(self.get_parameter('red_light_pass_grace_sec').value)
        self.ignore_red_until = now + Duration(seconds=grace_sec)
        if released_zone is not None:
            self.ignore_red_zone = released_zone
            self.get_logger().info(
                'Ignoring traffic-light changes until leaving waypoint zone '
                f'{released_zone[0]}-{released_zone[1]}.'
            )

    def red_light_active(self, now) -> bool:
        if not bool(self.get_parameter('red_light_enabled').value):
            return False
        if self.red_zone_suppressed():
            return False
        fixed_stop_overshoot = self.current_fixed_stop_overshoot()
        if fixed_stop_overshoot is not None:
            self.suppress_reds_after_fixed_stop_line(fixed_stop_overshoot)
            return False
        if self.ignore_red_until is not None and now < self.ignore_red_until:
            return False
        return self.in_red_light_zone()

    def current_fixed_stop_overshoot(self) -> Optional[float]:
        """Return metres beyond the current fixed stop line, once committed."""
        if not self.in_red_light_zone() or self.current_fixed_stop_point() is None:
            return None
        remaining, _fixed_stop = self.red_light_stop_remaining(None)
        margin = float(
            self.get_parameter('red_light_passed_fixed_stop_margin_m').value
        )
        if remaining is None or remaining >= -margin:
            return None
        return -remaining

    def suppress_reds_after_fixed_stop_line(
        self,
        overshoot_m: float,
        target: Optional[TrafficLightTarget] = None,
    ) -> None:
        """Ignore an intersection once its fixed stop line is behind the car."""
        zone = self.current_red_light_zone()
        if zone is None:
            return
        if self.ignore_red_zone != zone:
            self.get_logger().info(
                f'Fixed red-light stop point is {overshoot_m:.1f} m behind; '
                f'ignoring traffic-light changes until leaving waypoint zone '
                f'{zone[0]}-{zone[1]}.'
            )
        self.ignore_red_zone = zone
        if target is None:
            return
        self.red_target = None
        self.red_confirm_count = 0
        self.green_confirm_count = 0

    def grace_overridden_by_red(self, now, best_red: LightObservation) -> bool:
        """
        Report whether a red seen during the pass grace must be honored.

        Three conditions: the *grace* is what is suppressing reds (not the
        zone gating or the enable switch); the red is the same head we just
        released (a different head down the road is the case the grace
        exists for); and its stop point is still far enough ahead to stop
        for properly - see red_light_grace_red_override_margin_m.
        """
        if not bool(self.get_parameter('red_light_enabled').value):
            return False
        if self.red_zone_suppressed():
            return False
        if not self.in_red_light_zone():
            return False
        if self.ignore_red_until is None or now >= self.ignore_red_until:
            return False
        if self.last_released_xyz is None or not self.same_light(
            self.last_released_xyz,
            best_red[0],
            float(self.get_parameter('red_light_associate_radius').value),
        ):
            return False
        remaining, _fixed_stop = self.red_light_stop_remaining(best_red[1])
        return remaining is not None and remaining > float(
            self.get_parameter('red_light_grace_red_override_margin_m').value
        )

    def in_red_light_zone(self) -> bool:
        if not self.red_light_zones:
            return True
        if self.latest_waypoint_index is None:
            # Waypoint manager not up yet - do not gate on a signal we have
            # never received.
            return True
        return any(
            low <= self.latest_waypoint_index <= high
            for low, high in self.red_light_zones
        )

    def current_red_light_zone(self) -> Optional[Tuple[int, int]]:
        """Return the configured intersection zone containing the car."""
        if self.latest_waypoint_index is None:
            return None
        for low, high in self.red_light_zones:
            if low <= self.latest_waypoint_index <= high:
                return low, high
        return None

    def red_zone_suppressed(self) -> bool:
        """Suppress a released junction until its waypoint zone is exited."""
        zone = self.ignore_red_zone
        if zone is None:
            return False
        index = self.latest_waypoint_index
        if index is None or zone[0] <= index <= zone[1]:
            return True
        self.get_logger().info(
            f'Left released traffic-light zone {zone[0]}-{zone[1]}; '
            'traffic-light handling re-enabled.'
        )
        self.ignore_red_zone = None
        return False

    def parse_zone_values(self, raw: str) -> List[Tuple[int, int, float]]:
        """Parse "lo-hi:value,lo-hi:value" into (low, high, value) entries."""
        entries: List[Tuple[int, int, float]] = []
        for chunk in (raw or '').split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                zone_text, value_text = chunk.split(':', 1)
                value = float(value_text)
                if '-' in zone_text:
                    low_text, high_text = zone_text.split('-', 1)
                    low, high = int(low_text), int(high_text)
                else:
                    low = high = int(zone_text)
            except ValueError:
                self.get_logger().warn(
                    f'Ignoring unparsable zone:value entry: {chunk!r}'
                )
                continue
            entries.append((min(low, high), max(low, high), value))
        return entries

    def parse_fixed_stop_points(self, raw: str) -> List[FixedStopPoint]:
        """Parse "lo-hi:waypoint:x:y,..." fixed map stop-point entries."""
        entries: List[FixedStopPoint] = []
        for chunk in (raw or '').split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                zone_text, waypoint_text, x_text, y_text = chunk.split(':')
                waypoint = int(waypoint_text)
                x = float(x_text)
                y = float(y_text)
                if '-' in zone_text:
                    low_text, high_text = zone_text.split('-', 1)
                    low, high = int(low_text), int(high_text)
                else:
                    low = high = int(zone_text)
            except ValueError:
                self.get_logger().warn(
                    f'Ignoring unparsable fixed stop-point entry: {chunk!r}'
                )
                continue
            entries.append((min(low, high), max(low, high), waypoint, x, y))
        return entries

    def parse_mapped_traffic_lights(self, raw: str) -> List[MappedTrafficLight]:
        """Parse "lo-hi:x:y:z,..." static signal-head entries."""
        entries: List[MappedTrafficLight] = []
        for chunk in (raw or '').split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                zone_text, x_text, y_text, z_text = chunk.split(':')
                x, y, z = float(x_text), float(y_text), float(z_text)
                if '-' in zone_text:
                    low_text, high_text = zone_text.split('-', 1)
                    low, high = int(low_text), int(high_text)
                else:
                    low = high = int(zone_text)
            except ValueError:
                self.get_logger().warn(
                    f'Ignoring unparsable mapped traffic light: {chunk!r}'
                )
                continue
            entries.append((min(low, high), max(low, high), x, y, z))
        return entries

    def current_mapped_traffic_light(self) -> Optional[MappedTrafficLight]:
        """Return the sole map head enabled for the active waypoint zone."""
        if self.latest_waypoint_index is None:
            return None
        for entry in self.mapped_traffic_lights:
            if entry[0] <= self.latest_waypoint_index <= entry[1]:
                return entry
        return None

    def mapped_head_normalized_x(
        self,
        mapped_head: MappedTrafficLight,
    ) -> Optional[float]:
        """Project a static map head to horizontal image coordinates."""
        world_xyz = (mapped_head[2], mapped_head[3], mapped_head[4])
        offset = self.vehicle_frame_offset(world_xyz)
        if offset is None or offset[0] <= 0.1:
            return None
        forward_m, lateral_m, _height_m = offset

        camera_info = self.traffic_light_camera_info
        if camera_info is not None and camera_info.width > 0 and camera_info.k[0] > 0.0:
            fx_normalized = float(camera_info.k[0]) / float(camera_info.width)
            cx_normalized = float(camera_info.k[2]) / float(camera_info.width)
        else:
            # Unity is configured at 640x480 with a 90-degree vertical FOV:
            # fx = height/2 = 240, hence fx/width = 0.375.
            fx_normalized = 0.375
            cx_normalized = 0.5
        return cx_normalized - fx_normalized * lateral_m / forward_m

    def current_fixed_stop_point(self) -> Optional[Tuple[int, float, float]]:
        """Return the mapped (waypoint, x, y) stop point for the active zone."""
        if self.latest_waypoint_index is None:
            return None
        for low, high, waypoint, x, y in self.fixed_stop_points:
            if low <= self.latest_waypoint_index <= high:
                return waypoint, x, y
        return None

    def red_light_stop_remaining(
        self,
        light_forward_m: Optional[float],
    ) -> Tuple[Optional[float], Optional[Tuple[int, float, float]]]:
        """
        Return signed metres to the applicable red-light stop point.

        A configured fixed point wins over perception geometry.  Its forward
        projection is signed, so passing wp75 produces a negative overshoot
        and cannot make the distance grow again as an unsigned Euclidean
        distance would.
        """
        fixed_stop = self.current_fixed_stop_point()
        if fixed_stop is not None:
            _waypoint, x, y = fixed_stop
            if self.current_pose is None:
                return None, fixed_stop
            z = float(self.current_pose.pose.position.z)
            offset = self.vehicle_frame_offset((x, y, z))
            return (offset[0] if offset is not None else None), fixed_stop
        if light_forward_m is None:
            return None, None
        return light_forward_m - self.current_stop_offset(), None

    def current_stop_offset(self) -> float:
        if self.latest_waypoint_index is not None:
            for low, high, value in self.stop_offset_overrides:
                if low <= self.latest_waypoint_index <= high:
                    return value
        return float(self.get_parameter('red_light_stop_offset').value)

    def parse_waypoint_zones(self, raw: str) -> List[Tuple[int, int]]:
        zones: List[Tuple[int, int]] = []
        for chunk in (raw or '').split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                if '-' in chunk:
                    low_text, high_text = chunk.split('-', 1)
                    low, high = int(low_text), int(high_text)
                else:
                    low = high = int(chunk)
            except ValueError:
                self.get_logger().warn(
                    f'Ignoring unparsable red_light_waypoint_zones entry: {chunk!r}'
                )
                continue
            zones.append((min(low, high), max(low, high)))
        return zones

    @staticmethod
    def same_light(
        first: Tuple[float, float, float],
        second: Tuple[float, float, float],
        radius: float,
    ) -> bool:
        return math.dist(first[:2], second[:2]) <= radius

    def vehicle_frame_offset(
        self,
        world_xyz: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Express a world point as (forward, left, up) relative to the car."""
        if self.current_pose is None:
            return None
        position = self.current_pose.pose.position
        yaw = quaternion_to_yaw(self.current_pose.pose.orientation)
        dx = world_xyz[0] - position.x
        dy = world_xyz[1] - position.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            dx * cos_yaw + dy * sin_yaw,
            -dx * sin_yaw + dy * cos_yaw,
            world_xyz[2] - position.z,
        )

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------

    def should_follow_vehicle(self) -> bool:
        front_distance = self.current_front_obstacle_distance()
        if front_distance is None:
            self.follow_engaged = False
            return False
        if self.in_follow_suppress_zone():
            _distance, source = self.current_front_obstacle()
            self.get_logger().info(
                f'FOLLOW_VEHICLE suppressed at waypoint '
                f'{self.latest_waypoint_index} (front obstacle '
                f'{front_distance:.1f} m ({source})).',
                throttle_duration_sec=5.0,
            )
            self.follow_engaged = False
            return False
        follow_distance = float(self.get_parameter('follow_distance').value)
        if self.follow_engaged:
            follow_distance += float(
                self.get_parameter('front_obstacle_follow_release_margin_m').value
            )
        self.follow_engaged = front_distance <= follow_distance
        return self.follow_engaged

    def should_emergency_brake(self) -> bool:
        """Report a *confirmed* emergency; call exactly once per control cycle."""
        raw = self.raw_emergency_condition()
        if self.in_emergency_suppress_zone():
            if raw:
                distance, source = self.current_front_obstacle()
                distance_text = 'unknown' if distance is None else f'{distance:.1f} m'
                self.get_logger().info(
                    f'Emergency brake suppressed in waypoint zone (waypoint '
                    f'{self.latest_waypoint_index}, front obstacle {distance_text} '
                    f'({source})); FOLLOW_VEHICLE slowdown still applies.',
                    throttle_duration_sec=5.0,
                )
            self.emergency_confirm_count = 0
            self.emergency_engaged = False
            return False
        if raw:
            self.emergency_confirm_count += 1
        else:
            self.emergency_confirm_count = 0
        return self.emergency_confirm_count >= max(
            1,
            int(self.get_parameter('front_obstacle_emergency_confirm_count').value),
        )

    def in_emergency_suppress_zone(self) -> bool:
        if not self.emergency_suppress_zones or self.latest_waypoint_index is None:
            return False
        return any(
            low <= self.latest_waypoint_index <= high
            for low, high in self.emergency_suppress_zones
        )

    def in_follow_suppress_zone(self) -> bool:
        if not self.follow_suppress_zones or self.latest_waypoint_index is None:
            return False
        return any(
            low <= self.latest_waypoint_index <= high
            for low, high in self.follow_suppress_zones
        )

    def raw_emergency_condition(self) -> bool:
        front_distance = self.current_front_obstacle_distance()
        if front_distance is None:
            self.emergency_engaged = False
            return False

        if self.emergency_engaged:
            release_distance = float(
                self.get_parameter('front_obstacle_emergency_release_distance').value
            )
            if front_distance <= release_distance:
                return True
            self.emergency_engaged = False

        triggered = front_distance <= float(self.get_parameter('emergency_distance').value)
        if not triggered and self.current_speed > 0.1:
            ttc = front_distance / self.current_speed
            triggered = ttc <= float(self.get_parameter('emergency_ttc_sec').value)

        self.emergency_engaged = triggered
        return triggered

    def obstacle_stop_expired(self) -> bool:
        """Report whether an obstacle has held us still for too long."""
        timeout = float(self.get_parameter('front_obstacle_stop_timeout_sec').value)
        if timeout <= 0.0:
            return False
        if self.obstacle_stop_released:
            return True
        if self.current_speed > float(
            self.get_parameter('front_obstacle_emergency_min_speed').value
        ):
            # Still decelerating; the stop is doing its job.
            self.obstacle_stop_since = None
            return False
        if self.obstacle_stop_since is None:
            self.obstacle_stop_since = self.get_clock().now()
            return False
        if self.is_fresh(self.obstacle_stop_since, timeout):
            return False

        distance, source = self.current_front_obstacle()
        distance_text = 'unknown' if distance is None else f'{distance:.1f} m'
        self.get_logger().warn(
            f'Held still {timeout:.0f}s by a front obstacle at {distance_text} '
            f'({source}); creeping instead. Check whether it is really in our lane.'
        )
        self.obstacle_stop_released = True
        return True

    def compute_follow_speed(self, target_speed: float) -> float:
        front_distance = self.current_front_obstacle_distance()
        if front_distance is None:
            return target_speed

        follow_speed = float(self.get_parameter('follow_speed').value)
        minimum_follow_speed = float(self.get_parameter('minimum_follow_speed').value)
        emergency_distance = float(self.get_parameter('emergency_distance').value)
        follow_distance = float(self.get_parameter('follow_distance').value)

        if follow_distance <= emergency_distance:
            return max(0.0, min(follow_speed, target_speed))

        ratio = (
            (front_distance - emergency_distance)
            / (follow_distance - emergency_distance)
        )
        ratio = max(0.0, min(1.0, ratio))
        scaled_speed = minimum_follow_speed + ratio * (follow_speed - minimum_follow_speed)
        return max(0.0, min(scaled_speed, target_speed))

    def current_front_obstacle(self) -> Tuple[Optional[float], str]:
        """
        Nearest front obstacle and which source reported it.

        The source matters for triage: the depth cloud sees walls and kerbs,
        while the car detector's own gate is an image-angle one
        (car_max_lateral_angle_ratio) that reaches into the oncoming lane.
        """
        nearest: Optional[float] = None
        source = 'none'
        if self.nearest_front_obstacle_m is not None and self.is_obstacle_fresh():
            nearest = self.nearest_front_obstacle_m
            source = 'cloud'
        if self.nearest_front_car_m is not None and self.is_front_car_fresh():
            trusted = self.front_car_laterally_gated or self.car_reading_confirmed_ahead(
                self.nearest_front_car_m, nearest
            )
            if trusted:
                if nearest is None or self.nearest_front_car_m < nearest:
                    nearest = self.nearest_front_car_m
                    source = 'car'
        return nearest, source

    def car_reading_confirmed_ahead(
        self,
        car_m: float,
        cloud_m: Optional[float],
    ) -> bool:
        """
        Check a car-YOLO distance against the in-lane depth corridor.

        The scalar car distance has no lateral position, so this corridor
        cross-check is the only lateral gate decision can apply to it - see
        front_obstacle_car_requires_cloud_match.
        """
        if not bool(
            self.get_parameter('front_obstacle_car_requires_cloud_match').value
        ):
            return True
        if not bool(self.get_parameter('front_obstacle_pointcloud_enabled').value):
            # No corridor scan to check against; keep the reading rather than
            # silently disabling car braking altogether.
            return True
        if cloud_m is None:
            return False
        tolerance = float(
            self.get_parameter('front_obstacle_car_cloud_match_tolerance_m').value
        )
        return abs(car_m - cloud_m) <= tolerance

    def current_front_obstacle_distance(self) -> Optional[float]:
        return self.current_front_obstacle()[0]

    def find_nearest_front_obstacle(self, cloud: PointCloud2) -> Optional[float]:
        x_offset = self.field_offset(cloud, 'x')
        y_offset = self.field_offset(cloud, 'y')
        z_offset = self.field_offset(cloud, 'z')
        if x_offset is None or y_offset is None or z_offset is None or cloud.point_step <= 0:
            return None

        endian = '>' if cloud.is_bigendian else '<'
        point_count = int(cloud.width * cloud.height)
        sample_step = max(1, int(self.get_parameter('pointcloud_sample_step').value))
        lateral_limit = float(
            self.get_parameter('front_obstacle_corridor_half_width').value
        )
        min_forward = float(self.get_parameter('obstacle_min_forward_distance').value)
        max_forward = float(self.get_parameter('obstacle_max_forward_distance').value)
        min_vertical = float(self.get_parameter('obstacle_min_vertical').value)
        max_vertical = float(self.get_parameter('obstacle_max_vertical').value)

        # Cloud axes are camera-optical (x right, y down, z forward); the
        # path bearing is left-positive, so a point on the path at depth z
        # sits at x = -z * tan(bearing).
        skew_tan = math.tan(self.corridor_skew_rad())

        forward_distances: List[float] = []
        data = cloud.data
        for index in range(0, point_count, sample_step):
            base = index * cloud.point_step
            if base + max(x_offset, y_offset, z_offset) + 4 > len(data):
                break

            x = struct.unpack_from(endian + 'f', data, base + x_offset)[0]
            y = struct.unpack_from(endian + 'f', data, base + y_offset)[0]
            z = struct.unpack_from(endian + 'f', data, base + z_offset)[0]
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            if abs(x + z * skew_tan) > lateral_limit:
                continue
            if y < min_vertical or y > max_vertical:
                continue
            if z < min_forward or z > max_forward:
                continue
            forward_distances.append(z)

        # Reporting the single closest sample makes one bad depth pixel
        # indistinguishable from a car: it braked on noise. Demand enough
        # points to call it a surface, then report a low percentile of them.
        min_points = max(1, int(self.get_parameter('front_obstacle_min_points').value))
        if len(forward_distances) < min_points:
            return None

        forward_distances.sort()
        percentile = min(
            100.0,
            max(0.0, float(self.get_parameter('front_obstacle_distance_percentile').value)),
        )
        index = int(round((percentile / 100.0) * (len(forward_distances) - 1)))
        return forward_distances[index]

    def corridor_skew_rad(self) -> float:
        """
        Bearing (left-positive) the obstacle corridor should bend towards.

        Follows /planning/next_waypoint so that in a corner the corridor
        tracks the road instead of the building the camera happens to face.
        Returns 0 when disabled or when the waypoint is unusable (missing,
        or so close that its bearing is meaningless).
        """
        if not bool(
            self.get_parameter('front_obstacle_corridor_follow_waypoint').value
        ):
            return 0.0
        if self.next_waypoint_pose is None:
            return 0.0
        position = self.next_waypoint_pose.pose.position
        offset = self.vehicle_frame_offset((position.x, position.y, position.z))
        if offset is None or offset[0] < 2.0:
            return 0.0
        max_skew = math.radians(
            float(self.get_parameter('front_obstacle_corridor_max_skew_deg').value)
        )
        bearing = math.atan2(offset[1], offset[0])
        return max(-max_skew, min(max_skew, bearing))

    @staticmethod
    def field_offset(cloud: PointCloud2, name: str) -> Optional[int]:
        for field in cloud.fields:
            if field.name == name:
                return int(field.offset)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_obstacle_fresh(self) -> bool:
        return self.is_fresh(
            self.latest_obstacle_time,
            float(self.get_parameter('front_obstacle_timeout_sec').value),
        )

    def is_front_car_fresh(self) -> bool:
        return self.is_fresh(
            self.latest_front_car_time,
            float(self.get_parameter('front_obstacle_timeout_sec').value),
        )

    def is_fresh(self, stamp, timeout_sec: float) -> bool:
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        return age <= timeout_sec

    @staticmethod
    def publish_string(publisher, value: str) -> None:
        msg = String()
        msg.data = value
        publisher.publish(msg)

    @staticmethod
    def publish_float(publisher, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    @staticmethod
    def publish_bool(publisher, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BehaviorStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
