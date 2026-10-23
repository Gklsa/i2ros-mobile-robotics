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

import math
from typing import List, Sequence, Tuple

import rclpy
from ad_interfaces.srv import GetNextWaypoint
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from rclpy.node import Node
from std_msgs.msg import Int32


def yaw_to_quaternion(yaw: float) -> Quaternion:
    msg = Quaternion()
    msg.z = math.sin(yaw * 0.5)
    msg.w = math.cos(yaw * 0.5)
    return msg


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointManager(Node):
    """Select the next short-term goal from a predefined ordered waypoint list."""

    def __init__(self) -> None:
        super().__init__('waypoint_manager')

        self.declare_parameter('pose_topic', '/OurCar/CoM/pose')
        self.declare_parameter('next_waypoint_topic', '/planning/next_waypoint')
        self.declare_parameter('active_waypoint_index_topic', '/planning/active_waypoint_index')
        self.declare_parameter('waypoint_index_topic', '/planning/waypoint_index')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('waypoint_reach_threshold', 3.0)
        self.declare_parameter('waypoint_loop', True)
        self.declare_parameter('initial_waypoint_index', 0)
        self.declare_parameter('initialize_from_nearest_waypoint', False)
        self.declare_parameter(
            'waypoint_triplets',
            [
                -28.0, -1.0, 1.20,
                -24.0, 14.0, 0.55,
                -12.0, 24.0, 0.10,
                2.0, 25.0, 0.00,
                18.0, 23.0, -0.35,
                27.0, 13.0, -1.40,
                24.0, 2.0, 3.14,
                10.0, 1.0, 3.14,
                0.0, -8.0, -2.30,
                -15.0, -11.0, -2.85,
                -26.0, -6.0, 1.55,
            ],
        )
        self.declare_parameter('publish_rate_hz', 5.0)

        self.waypoints = self._load_waypoints()
        self.current_pose = None
        self.active_index = 0
        self.initialized_from_pose = False
        self.reach_threshold = float(self.get_parameter('waypoint_reach_threshold').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.loop_enabled = bool(self.get_parameter('waypoint_loop').value)

        self.waypoint_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('next_waypoint_topic').value,
            10,
        )
        self.active_index_pub = self.create_publisher(
            Int32,
            self.get_parameter('active_waypoint_index_topic').value,
            10,
        )
        self.index_pub = self.create_publisher(
            Int32,
            self.get_parameter('waypoint_index_topic').value,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('pose_topic').value,
            self.pose_callback,
            10,
        )
        self.create_service(GetNextWaypoint, 'get_next_waypoint', self.handle_get_next_waypoint)

        publish_rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / publish_rate_hz, self.publish_current_waypoint)

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} planning waypoints and waiting for vehicle pose.'
        )

    def _load_waypoints(self) -> List[Tuple[float, float, float]]:
        triplets = list(self.get_parameter('waypoint_triplets').value)
        if len(triplets) < 3 or len(triplets) % 3 != 0:
            raise ValueError('waypoint_triplets must contain x, y, yaw triples.')

        waypoints = []
        for index in range(0, len(triplets), 3):
            x, y, yaw = float(triplets[index]), float(triplets[index + 1]), float(triplets[index + 2])
            waypoints.append((x, y, yaw))
        return waypoints

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg
        current_xy = (msg.pose.position.x, msg.pose.position.y)
        if not self.initialized_from_pose:
            self._initialize_active_index(current_xy)

        self._advance_if_reached(current_xy)

    def _initialize_active_index(self, current_xy: Sequence[float]) -> None:
        if bool(self.get_parameter('initialize_from_nearest_waypoint').value):
            self.active_index = self._nearest_waypoint_index(current_xy)
        else:
            requested_index = int(self.get_parameter('initial_waypoint_index').value)
            self.active_index = max(0, min(requested_index, len(self.waypoints) - 1))
        self.initialized_from_pose = True
        self.get_logger().info(
            f'Initialized planning route at waypoint index {self.active_index}.'
        )

    def _nearest_waypoint_index(self, xy: Sequence[float]) -> int:
        best_index = 0
        best_distance = float('inf')
        for index, (wx, wy, _) in enumerate(self.waypoints):
            distance = math.hypot(wx - xy[0], wy - xy[1])
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _advance_if_reached(self, xy: Sequence[float]) -> None:
        if not self.waypoints:
            return

        while True:
            reached = (
                self._distance_to_waypoint(self.active_index, xy)
                <= self._effective_reach_threshold()
            )
            missed = self._passed_active_waypoint(xy)
            if not reached and not missed:
                return

            next_index = self._next_index(self.active_index)
            if next_index == self.active_index:
                return

            self.active_index = next_index
            self.get_logger().info(f'Switched to waypoint index {self.active_index}.')

    def _next_index(self, index: int) -> int:
        next_index = index + 1
        if next_index >= len(self.waypoints):
            return 0 if self.loop_enabled else index
        return next_index

    def _distance_to_waypoint(self, index: int, xy: Sequence[float]) -> float:
        wx, wy, _ = self.waypoints[index]
        return math.hypot(wx - xy[0], wy - xy[1])

    def _effective_reach_threshold(self) -> float:
        if self.active_index in (96, 97, 98, 99):
            return max(self.reach_threshold, 8.0)
        return self.reach_threshold

    def _passed_active_waypoint(self, xy: Sequence[float]) -> bool:
        if self.active_index <= 0:
            return False

        px, py, _ = self.waypoints[self.active_index - 1]
        wx, wy, _ = self.waypoints[self.active_index]
        vx = wx - px
        vy = wy - py
        segment_length_sq = vx * vx + vy * vy
        if segment_length_sq < 1e-6:
            return False

        progress = ((xy[0] - px) * vx + (xy[1] - py) * vy) / segment_length_sq
        return progress > 1.05

    def _build_waypoint_pose(self, waypoint: Tuple[float, float, float]) -> Pose:
        pose = Pose()
        pose.position.x = waypoint[0]
        pose.position.y = waypoint[1]
        pose.orientation = yaw_to_quaternion(waypoint[2])
        return pose

    def current_target_pose(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.pose = self._build_waypoint_pose(self.waypoints[self.active_index])
        return msg

    def publish_current_waypoint(self) -> None:
        if not self.initialized_from_pose:
            return
        self.waypoint_pub.publish(self.current_target_pose())
        index_msg = Int32()
        index_msg.data = self.active_index
        self.active_index_pub.publish(index_msg)
        self.index_pub.publish(index_msg)

    def handle_get_next_waypoint(self, request, response):
        current_xy = (request.current_pose.position.x, request.current_pose.position.y)
        if not self.initialized_from_pose:
            self._initialize_active_index(current_xy)
        else:
            self._advance_if_reached(current_xy)

        target = self._build_waypoint_pose(self.waypoints[self.active_index])
        response.target_pose = target
        response.success = True
        response.message = f'Active waypoint index: {self.active_index}, yaw={quaternion_to_yaw(target.orientation):.2f}'
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointManager()
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
