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

"""Behavior tests for the autonomous-driving decision state machine."""

import pytest
from rclpy.time import Time


@pytest.fixture
def behavior_node():
    """Create a fresh decision node and expose its state classes."""
    from ad_decision.behavior_state_machine import (
        BehaviorState,
        BehaviorStateMachine,
        TrafficLightTarget,
    )
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    node = BehaviorStateMachine()
    yield node, BehaviorState, TrafficLightTarget
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _set_nominal_conditions(monkeypatch, node):
    """Disable higher-priority branches before testing one behavior."""
    monkeypatch.setattr(node, 'route_end_reached', lambda: False)
    monkeypatch.setattr(node, 'startup_hold_active', lambda: False)
    monkeypatch.setattr(node, 'should_emergency_brake', lambda: False)
    monkeypatch.setattr(node, 'red_light_command', lambda _speed: (False, None))
    monkeypatch.setattr(node, 'should_follow_vehicle', lambda: False)


def test_front_vehicle_enters_follow_state(behavior_node, monkeypatch):
    """A vehicle inside the follow range reduces the cruise target."""
    node, behavior_state, _target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    monkeypatch.setattr(node, 'should_follow_vehicle', lambda: True)
    monkeypatch.setattr(node, 'current_front_obstacle_distance', lambda: 7.0)

    state, speed, stop, emergency = node.evaluate()

    assert state == behavior_state.FOLLOW_VEHICLE
    assert 0.0 < speed < float(node.get_parameter('target_speed').value)
    assert not stop
    assert not emergency


def test_close_vehicle_requests_emergency_brake(behavior_node, monkeypatch):
    """A confirmed close obstacle requests both a stop and hard braking."""
    node, behavior_state, _target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    node.current_speed = 2.0
    monkeypatch.setattr(node, 'should_emergency_brake', lambda: True)
    monkeypatch.setattr(node, 'obstacle_stop_expired', lambda: False)

    state, speed, stop, emergency = node.evaluate()

    assert state == behavior_state.EMERGENCY_BRAKE
    assert speed == 0.0
    assert stop
    assert emergency


def test_red_light_approach_reduces_speed(behavior_node, monkeypatch):
    """A red light ahead selects the lower deceleration target."""
    node, behavior_state, _target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    monkeypatch.setattr(node, 'red_light_command', lambda _speed: (False, 1.25))

    state, speed, stop, emergency = node.evaluate()

    assert state == behavior_state.APPROACH_RED_LIGHT
    assert speed == pytest.approx(1.25)
    assert not stop
    assert not emergency


def test_red_light_stop_point_holds_vehicle(behavior_node, monkeypatch):
    """Reaching a red stop point latches a zero-speed stop request."""
    node, behavior_state, _target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    monkeypatch.setattr(node, 'red_light_command', lambda _speed: (True, 0.0))

    state, speed, stop, emergency = node.evaluate()

    assert state == behavior_state.STOP_FOR_RED_LIGHT
    assert speed == 0.0
    assert stop
    assert not emergency


def test_matching_green_releases_stop_and_resumes_cruise(
    behavior_node,
    monkeypatch,
):
    """A confirmed green at the tracked head releases the red-light latch."""
    node, behavior_state, target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    now = node.get_clock().now()
    target = target_type((10.0, 2.0, 5.0), now)
    target.latched = True
    node.red_target = target
    green = [((10.0, 2.0, 5.0), 10.0, 0.95, 0.5)]
    confirmation_count = max(
        1,
        int(node.get_parameter('red_light_green_release_confirm_count').value),
    )

    for _index in range(confirmation_count - 1):
        assert not node.green_release_confirmed(target, green)
    assert node.green_release_confirmed(target, green)
    node.release_target(target, now, 'confirmed matching green in behavior test')

    state, speed, stop, emergency = node.evaluate()
    assert node.red_target is None
    assert state == behavior_state.CRUISE
    assert speed == pytest.approx(float(node.get_parameter('target_speed').value))
    assert not stop
    assert not emergency


def test_route_completion_has_highest_stop_priority(behavior_node, monkeypatch):
    """A completed route always produces a non-emergency terminal stop."""
    node, behavior_state, _target_type = behavior_node
    _set_nominal_conditions(monkeypatch, node)
    monkeypatch.setattr(node, 'route_end_reached', lambda: True)

    state, speed, stop, emergency = node.evaluate()

    assert state == behavior_state.ROUTE_COMPLETE
    assert speed == 0.0
    assert stop
    assert not emergency


def test_mission_timer_records_completion_only_once(behavior_node):
    """Mission timing starts after release and is frozen at first completion."""
    node, _behavior_state, _target_type = behavior_node
    node.start_mission_timer(Time(nanoseconds=2_000_000_000))

    node.report_mission_completion(Time(nanoseconds=14_340_000_000))

    assert node.mission_completion_reported
    assert node.mission_completion_time_sec == pytest.approx(12.34)

    node.report_mission_completion(Time(nanoseconds=30_000_000_000))
    assert node.mission_completion_time_sec == pytest.approx(12.34)
