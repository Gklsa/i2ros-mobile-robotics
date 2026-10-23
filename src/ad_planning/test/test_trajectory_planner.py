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

"""Functional unit tests for the local trajectory planner."""

import math

from geometry_msgs.msg import Pose
import pytest
import rclpy

from ad_planning.trajectory_planner import (
    TrajectoryPlanner,
    clamp,
    quaternion_to_yaw,
    yaw_to_quaternion,
)


@pytest.fixture(scope='module')
def planner():
    """Create one ROS node for the planner algorithm tests."""
    if not rclpy.ok():
        rclpy.init()
    node = TrajectoryPlanner()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _pose(x: float, y: float, yaw: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.orientation = yaw_to_quaternion(yaw)
    return pose


def test_math_helpers_round_trip_and_clamp():
    """Quaternion conversion and bounds helpers retain expected values."""
    yaw = -1.2
    assert quaternion_to_yaw(yaw_to_quaternion(yaw)) == pytest.approx(yaw)
    assert clamp(-1.0, 0.0, 2.0) == 0.0
    assert clamp(3.0, 0.0, 2.0) == 2.0


def test_generated_path_has_requested_endpoints(planner):
    """Hermite sampling connects the current pose to the active waypoint."""
    samples = planner.generate_samples(
        _pose(1.0, 2.0, 0.0),
        _pose(21.0, 7.0, math.pi / 4.0),
    )

    assert 15 <= len(samples) <= 80
    assert samples[0][:2] == pytest.approx((1.0, 2.0))
    assert samples[-1][:2] == pytest.approx((21.0, 7.0))
    assert planner.compute_path_length(samples) > 20.0


def test_curvature_reduces_speed(planner):
    """Curves reduce speed while the planner remains behavior-independent."""
    curved_samples = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, math.pi / 2.0),
    ]
    planner.current_speed = 3.0
    moving_speeds = planner.generate_speed_profile(curved_samples)

    assert moving_speeds[1] < float(planner.get_parameter('target_speed').value)
    assert all(speed > 0.0 for speed in moving_speeds)
