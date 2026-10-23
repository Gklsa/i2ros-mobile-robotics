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
from ad_interfaces.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TwistStamped
from nav_msgs.msg import Path
from rclpy.node import Node


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


class TrajectoryPlanner(Node):
    """Generate a smooth local trajectory from the current pose to the active waypoint."""

    def __init__(self) -> None:
        super().__init__('trajectory_planner')

        self.declare_parameter('pose_topic', '/OurCar/CoM/pose')
        self.declare_parameter('twist_topic', '/OurCar/CoM/twist')
        self.declare_parameter('next_waypoint_topic', '/planning/next_waypoint')
        self.declare_parameter('reference_path_topic', '/planning/reference_path')
        self.declare_parameter('trajectory_topic', '/planning/trajectory')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('trajectory_point_spacing', 1.0)
        self.declare_parameter('minimum_point_count', 15)
        self.declare_parameter('maximum_point_count', 80)
        self.declare_parameter('target_speed', 5.0)
        self.declare_parameter('min_turn_speed', 1.5)
        self.declare_parameter('max_lateral_accel', 2.5)
        self.declare_parameter('hermite_tangent_ratio', 0.6)
        self.declare_parameter('minimum_tangent_length', 4.0)

        self.current_pose = None
        self.current_speed = 0.0
        self.target_waypoint = None

        self.path_pub = self.create_publisher(
            Path,
            self.get_parameter('reference_path_topic').value,
            10,
        )
        self.trajectory_pub = self.create_publisher(
            Trajectory,
            self.get_parameter('trajectory_topic').value,
            10,
        )

        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value, self.pose_callback, 10)
        self.create_subscription(TwistStamped, self.get_parameter('twist_topic').value, self.twist_callback, 10)
        self.create_subscription(PoseStamped, self.get_parameter('next_waypoint_topic').value, self.target_callback, 10)
        publish_rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / publish_rate_hz, self.publish_plan)

        self.get_logger().info('Trajectory planner waiting for current pose and target waypoint.')

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def twist_callback(self, msg: TwistStamped) -> None:
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z
        self.current_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    def target_callback(self, msg: PoseStamped) -> None:
        self.target_waypoint = msg

    def publish_plan(self) -> None:
        if self.current_pose is None or self.target_waypoint is None:
            return

        samples = self.generate_samples(self.current_pose.pose, self.target_waypoint.pose)
        if len(samples) < 2:
            return

        speeds = self.generate_speed_profile(samples)
        self.path_pub.publish(self.build_path(samples))
        self.trajectory_pub.publish(self.build_trajectory(samples, speeds))

    def generate_samples(self, start_pose: Pose, goal_pose: Pose) -> List[Tuple[float, float, float]]:
        start = (start_pose.position.x, start_pose.position.y)
        goal = (goal_pose.position.x, goal_pose.position.y)
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 0.5:
            return [(start[0], start[1], quaternion_to_yaw(start_pose.orientation))]

        point_spacing = max(0.2, float(self.get_parameter('trajectory_point_spacing').value))
        min_points = max(2, int(self.get_parameter('minimum_point_count').value))
        max_points = max(min_points, int(self.get_parameter('maximum_point_count').value))
        num_points = int(distance / point_spacing) + 1
        num_points = max(min_points, min(max_points, num_points))

        start_yaw = quaternion_to_yaw(start_pose.orientation)
        goal_yaw = quaternion_to_yaw(goal_pose.orientation)
        tangent_ratio = float(self.get_parameter('hermite_tangent_ratio').value)
        tangent_length = max(
            float(self.get_parameter('minimum_tangent_length').value),
            distance * tangent_ratio,
        )
        m0 = (math.cos(start_yaw) * tangent_length, math.sin(start_yaw) * tangent_length)
        m1 = (math.cos(goal_yaw) * tangent_length, math.sin(goal_yaw) * tangent_length)

        samples = []
        for idx in range(num_points):
            s = idx / (num_points - 1)
            h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0
            h10 = s * s * s - 2.0 * s * s + s
            h01 = -2.0 * s * s * s + 3.0 * s * s
            h11 = s * s * s - s * s

            x = h00 * start[0] + h10 * m0[0] + h01 * goal[0] + h11 * m1[0]
            y = h00 * start[1] + h10 * m0[1] + h01 * goal[1] + h11 * m1[1]
            samples.append((x, y, 0.0))

        oriented_samples = []
        for idx, (x, y, _) in enumerate(samples):
            if idx == len(samples) - 1:
                prev = samples[idx - 1]
                yaw = math.atan2(y - prev[1], x - prev[0])
            else:
                nxt = samples[idx + 1]
                yaw = math.atan2(nxt[1] - y, nxt[0] - x)
            oriented_samples.append((x, y, yaw))
        return oriented_samples

    def generate_speed_profile(self, samples: Sequence[Tuple[float, float, float]]) -> List[float]:
        target_speed = float(self.get_parameter('target_speed').value)
        min_turn_speed = float(self.get_parameter('min_turn_speed').value)
        max_lateral_accel = max(0.1, float(self.get_parameter('max_lateral_accel').value))
        speeds = [target_speed for _ in samples]

        for idx in range(1, len(samples) - 1):
            curvature = self.compute_curvature(samples[idx - 1], samples[idx], samples[idx + 1])
            if curvature <= 1e-6:
                continue
            curvature_speed = math.sqrt(max_lateral_accel / curvature)
            speeds[idx] = clamp(curvature_speed, min_turn_speed, target_speed)

        speeds[0] = min(target_speed, max(self.current_speed, min_turn_speed))
        return speeds

    def compute_curvature(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> float:
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        if min(a, b, c) < 1e-6:
            return 0.0

        twice_area = abs(
            p0[0] * (p1[1] - p2[1]) +
            p1[0] * (p2[1] - p0[1]) +
            p2[0] * (p0[1] - p1[1])
        )
        return 2.0 * twice_area / (a * b * c)

    def compute_path_length(self, samples: Sequence[Tuple[float, float, float]]) -> float:
        total = 0.0
        for idx in range(1, len(samples)):
            total += math.hypot(samples[idx][0] - samples[idx - 1][0], samples[idx][1] - samples[idx - 1][1])
        return total

    def build_path(self, samples: Sequence[Tuple[float, float, float]]) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter('world_frame').value)
        for x, y, yaw in samples:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = yaw_to_quaternion(yaw)
            msg.poses.append(pose)
        return msg

    def build_trajectory(
        self,
        samples: Sequence[Tuple[float, float, float]],
        speeds: Sequence[float],
    ) -> Trajectory:
        trajectory = Trajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = str(self.get_parameter('world_frame').value)
        for (x, y, yaw), speed in zip(samples, speeds):
            point = TrajectoryPoint()
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.orientation = yaw_to_quaternion(yaw)
            point.target_speed = float(speed)
            trajectory.points.append(point)
        return trajectory


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryPlanner()
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
