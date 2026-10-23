#!/usr/bin/env python3
import math
import re
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from simulation.msg import VehicleControl


WAYPOINT_FILE = Path(__file__).resolve().parents[1] / "src/ad_planning/config/waypoints.yaml"


def load_waypoints():
    text = WAYPOINT_FILE.read_text()
    block = text.split("waypoint_triplets:", 1)[1]
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", block)]
    return [
        (numbers[i], numbers[i + 1], numbers[i + 2])
        for i in range(0, len(numbers) - 2, 3)
    ]


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RuntimeDriveDebug(Node):
    def __init__(self):
        super().__init__("runtime_drive_debug")
        self.waypoints = load_waypoints()
        self.pose = None
        self.twist = None
        self.target = None
        self.command = None

        self.create_subscription(PoseStamped, "/OurCar/CoM/pose", self.pose_cb, 10)
        self.create_subscription(TwistStamped, "/OurCar/CoM/twist", self.twist_cb, 10)
        self.create_subscription(PoseStamped, "/planning/next_waypoint", self.target_cb, 10)
        self.create_subscription(VehicleControl, "/car_command", self.command_cb, 10)
        self.create_timer(0.5, self.print_snapshot)

    def pose_cb(self, msg):
        self.pose = msg

    def twist_cb(self, msg):
        self.twist = msg

    def target_cb(self, msg):
        self.target = msg

    def command_cb(self, msg):
        self.command = msg

    def nearest_waypoint_index(self, x, y):
        best_index = 0
        best_distance = float("inf")
        for index, (wx, wy, _) in enumerate(self.waypoints):
            distance = math.hypot(wx - x, wy - y)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index, best_distance

    def print_snapshot(self):
        if self.pose is None or self.twist is None or self.target is None or self.command is None:
            return

        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        tx = self.target.pose.position.x
        ty = self.target.pose.position.y
        speed = math.hypot(self.twist.twist.linear.x, self.twist.twist.linear.y)
        active_index, target_error = self.nearest_waypoint_index(tx, ty)
        car_index, car_error = self.nearest_waypoint_index(px, py)
        distance_to_target = math.hypot(tx - px, ty - py)
        yaw = yaw_from_quaternion(self.pose.pose.orientation)

        print(
            f"active={active_index:03d} car_nearest={car_index:03d} "
            f"pos=({px:7.2f},{py:6.2f}) yaw={yaw:5.2f} "
            f"target=({tx:7.2f},{ty:6.2f}) dist={distance_to_target:5.2f} "
            f"speed={speed:4.2f} throttle={self.command.throttle:5.3f} "
            f"steer={self.command.steering:6.3f} brake={self.command.brake:5.3f} "
            f"target_err={target_error:4.2f} car_err={car_error:4.2f}",
            flush=True,
        )


def main():
    rclpy.init()
    node = RuntimeDriveDebug()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
