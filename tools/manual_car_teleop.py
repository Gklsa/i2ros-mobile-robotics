#!/usr/bin/env python3
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from simulation.msg import VehicleControl


HELP = """
Manual waypoint teleop

  w / up       one short throttle pulse
  s / down     one short brake pulse
  a / left     one short left steering pulse
  d / right    one short right steering pulse
  r            center steering now
  space        one stronger brake pulse
  x            stop all commands now
  q            quit
"""


class ManualCarTeleop(Node):
    def __init__(self):
        super().__init__("manual_car_teleop")
        self.declare_parameter("command_topic", "/car_command")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("max_throttle", 0.04)
        self.declare_parameter("max_steering", 0.15)
        self.declare_parameter("brake_strength", 0.30)
        self.declare_parameter("strong_brake_strength", 0.60)
        self.declare_parameter("pulse_duration", 0.18)

        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_steering = float(self.get_parameter("max_steering").value)
        self.brake_strength = float(self.get_parameter("brake_strength").value)
        self.strong_brake_strength = float(
            self.get_parameter("strong_brake_strength").value
        )
        self.pulse_duration = float(self.get_parameter("pulse_duration").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)

        topic = self.get_parameter("command_topic").value
        self.pub = self.create_publisher(VehicleControl, topic, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_command)

        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        self.throttle_until = 0.0
        self.steering_until = 0.0
        self.brake_until = 0.0
        self.quit_requested = False
        self.old_terminal = None
        self.last_key = "none"

        self.get_logger().info(
            f"Publishing pulse manual controls to {topic}; "
            f"pulse_duration={self.pulse_duration:.2f}s"
        )

    def clamp(self, value, low, high):
        return min(max(value, low), high)

    def handle_key(self, key):
        now = time.monotonic()
        self.last_key = key

        if key in ("w", "W", "UP"):
            self.throttle = self.max_throttle
            self.throttle_until = now + self.pulse_duration
            self.brake_until = 0.0
        elif key in ("s", "S", "DOWN"):
            self.throttle_until = 0.0
            self.brake = self.brake_strength
            self.brake_until = now + self.pulse_duration
        elif key in ("a", "A", "LEFT"):
            self.steering = -self.max_steering
            self.steering_until = now + self.pulse_duration
        elif key in ("d", "D", "RIGHT"):
            self.steering = self.max_steering
            self.steering_until = now + self.pulse_duration
        elif key == "r":
            self.steering = 0.0
            self.steering_until = 0.0
        elif key == " ":
            self.throttle_until = 0.0
            self.brake = self.strong_brake_strength
            self.brake_until = now + self.pulse_duration
        elif key == "x":
            self.throttle = 0.0
            self.steering = 0.0
            self.brake = 0.0
            self.throttle_until = 0.0
            self.steering_until = 0.0
            self.brake_until = 0.0
        elif key == "q":
            self.quit_requested = True

    def read_key(self):
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None

        char = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if char != "\x1b":
            return char

        rest = ""
        if select.select([sys.stdin], [], [], 0.01)[0]:
            rest += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if select.select([sys.stdin], [], [], 0.01)[0]:
            rest += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        arrows = {
            "[A": "UP",
            "[B": "DOWN",
            "[C": "RIGHT",
            "[D": "LEFT",
        }
        return arrows.get(rest)

    def publish_command(self):
        key = self.read_key()
        if key is not None:
            self.handle_key(key)

        now = time.monotonic()
        throttle = self.throttle if now < self.throttle_until else 0.0
        steering = self.steering if now < self.steering_until else 0.0
        brake = self.brake if now < self.brake_until else 0.0

        if brake > 0.0:
            throttle = 0.0

        msg = VehicleControl()
        msg.throttle = float(throttle)
        msg.steering = float(steering)
        msg.brake = float(brake)
        msg.reserved = 0.0
        self.pub.publish(msg)

        print(
            f"\rthrottle={throttle: .2f} steering={steering: .2f} "
            f"brake={brake: .2f} last_key={self.last_key:>5}    ",
            end="",
            flush=True,
        )


def main():
    rclpy.init()
    node = ManualCarTeleop()
    print(HELP)
    old_terminal = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop = VehicleControl()
        node.pub.publish(stop)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
