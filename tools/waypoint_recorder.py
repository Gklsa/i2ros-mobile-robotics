#!/usr/bin/env python3
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__("waypoint_recorder")
        default_output_dir = Path(__file__).resolve().parent

        self.declare_parameter("topic", "/OurCar/CoM/pose")
        self.declare_parameter("min_distance", 5.0)
        self.declare_parameter(
            "output", str(default_output_dir / "recorded_waypoints.yaml")
        )
        self.declare_parameter(
            "csv_output", str(default_output_dir / "recorded_waypoints.csv")
        )
        self.declare_parameter("reach_threshold", 6.0)

        self.topic = self.get_parameter("topic").value
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.output = Path(str(self.get_parameter("output").value))
        self.csv_output = Path(str(self.get_parameter("csv_output").value))
        self.reach_threshold = float(self.get_parameter("reach_threshold").value)

        self.points = []
        self.last_xy = None
        self.closed = False

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.csv_output.parent.mkdir(parents=True, exist_ok=True)

        self.output.write_text(
            "waypoint_manager:\n"
            "  ros__parameters:\n"
            "    waypoint_loop: true\n"
            f"    waypoint_reach_threshold: {self.reach_threshold:.1f}\n"
            "    publish_rate_hz: 5.0\n"
            "    waypoint_triplets:\n"
            "      [\n"
        )
        self.csv_output.write_text("index,x,y,yaw\n")

        self.create_subscription(PoseStamped, self.topic, self.pose_callback, 10)

        self.get_logger().info(
            f"Recording {self.topic}; min_distance={self.min_distance:.1f}m; "
            f"YAML={self.output}; CSV={self.csv_output}"
        )
        self.get_logger().info("Drive one clean lap. Press Ctrl-C when finished.")

    def pose_callback(self, msg):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = yaw_from_quaternion(msg.pose.orientation)

        if self.last_xy is not None:
            distance = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
            if distance < self.min_distance:
                return

        self.points.append((x, y, yaw))
        self.last_xy = (x, y)
        index = len(self.points) - 1

        with self.output.open("a") as f:
            f.write(f"        {x:.3f}, {y:.3f}, {yaw:.3f},\n")

        with self.csv_output.open("a") as f:
            f.write(f"{index},{x:.6f},{y:.6f},{yaw:.6f}\n")

        self.get_logger().info(f"#{index:03d}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")

    def close_yaml(self):
        if self.closed:
            return
        with self.output.open("a") as f:
            f.write("      ]\n")
        self.closed = True


def main():
    rclpy.init()
    node = WaypointRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_yaml()
        node.get_logger().info(f"Saved {len(node.points)} waypoints to {node.output}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
