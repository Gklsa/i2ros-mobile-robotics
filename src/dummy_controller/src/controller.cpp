// Copyright 2026 i2ros Team 6
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "controller.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace
{
double normalizeAngle(double angle)
{
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }

    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }

    return angle;
}
}

Controller::Controller()
    : has_pose_(false),
      has_twist_(false),
      has_waypoint_(false),
      has_waypoint_index_(false),
      stop_requested_(false),
      emergency_brake_(false),
      has_decision_input_(false),
      current_x_(0.0),
      current_y_(0.0),
      current_yaw_(0.0),
        current_speed_(0.0),
        waypoint_x_(0.0),
	    waypoint_y_(0.0),
        active_waypoint_index_(0),
	    target_speed_(3.0),
	    has_trajectory_(false),
	    last_steering_(0.0f),
        last_throttle_(0.0f)
{
}

bool Controller::isStale(const std::chrono::steady_clock::time_point& last) const
{
    const auto elapsed = std::chrono::steady_clock::now() - last;
    return std::chrono::duration<double>(elapsed).count() > kWatchdogTimeoutSec;
}

bool Controller::inputsAreStale() const
{
    if (has_pose_ && isStale(last_pose_time_)) {
        return true;
    }
    if (has_twist_ && isStale(last_twist_time_)) {
        return true;
    }
    if (has_decision_input_ && isStale(last_decision_time_)) {
        return true;
    }
    return false;
}


void Controller::updateTrajectory(const ad_interfaces::msg::Trajectory& msg)
{
    latest_trajectory_ = msg;
    has_trajectory_ = !msg.points.empty();
}


void Controller::updatePose(const geometry_msgs::msg::PoseStamped& msg)
{
    current_x_ = msg.pose.position.x;
    current_y_ = msg.pose.position.y;

    const auto& q = msg.pose.orientation;

    // quaternion -> yaw
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    current_yaw_ = std::atan2(siny_cosp, cosy_cosp);

    has_pose_ = true;
    last_pose_time_ = std::chrono::steady_clock::now();
}

void Controller::updateTwist(const geometry_msgs::msg::TwistStamped& msg)
{
    current_speed_ = std::sqrt(
        msg.twist.linear.x * msg.twist.linear.x +
        msg.twist.linear.y * msg.twist.linear.y
    );

    has_twist_ = true;
    last_twist_time_ = std::chrono::steady_clock::now();
}

void Controller::updateWaypoint(const geometry_msgs::msg::PoseStamped& msg)
{
    waypoint_x_ = msg.pose.position.x;
    waypoint_y_ = msg.pose.position.y;
    has_waypoint_ = true;
}

void Controller::updateActiveWaypointIndex(int index)
{
    active_waypoint_index_ = index;
    has_waypoint_index_ = true;
}

void Controller::updateTargetSpeed(const std_msgs::msg::Float32& msg)
{
    target_speed_ = msg.data;
    has_decision_input_ = true;
    last_decision_time_ = std::chrono::steady_clock::now();
}

void Controller::updateStopRequest(const std_msgs::msg::Bool& msg)
{
    stop_requested_ = msg.data;
    has_decision_input_ = true;
    last_decision_time_ = std::chrono::steady_clock::now();
}

void Controller::updateEmergencyBrake(const std_msgs::msg::Bool& msg)
{
    emergency_brake_ = msg.data;
    has_decision_input_ = true;
    last_decision_time_ = std::chrono::steady_clock::now();
}


float Controller::getThrottle()
{
    if (!has_twist_) {
        last_throttle_ = 0.0f;
        return last_throttle_;
    }

    if (emergency_brake_ || stop_requested_) {
        last_throttle_ = 0.0f;
        return last_throttle_;
    }

    const bool in_turn_zone = isTightTurnZone();
    const double desired_speed = limitedTargetSpeed();
    const double kp = in_turn_zone ? 0.18 : 0.2;
    const double speed_error = desired_speed - current_speed_;

    if (speed_error <= 0.0) {
        last_throttle_ = 0.0f;
        return last_throttle_;
    }

    const double steering_load =
        std::abs(last_steering_) / 0.6;

    const bool catch_up_zone =
        activeWaypointInRange(17, 60)||activeWaypointInRange(99,103);

    const double max_throttle = in_turn_zone
        ? (steering_load > 0.35 ? 0.240 : 0.220)
        : catch_up_zone
            ?(steering_load > 0.7 ? 0.35 : 0.70)
            :(steering_load > 0.7 ? 0.22 : 0.35);

    const double min_turn_throttle = in_turn_zone
        ? (current_speed_ < 0.15 ? 0.220 : (current_speed_ < 0.60 ? 0.1 : 0.0))
        : 0.0;

    last_throttle_ = static_cast<float>(
        std::clamp(kp * speed_error, min_turn_throttle, max_throttle)
    );

    return last_throttle_;
}

float Controller::getBrake()
{
    if (!has_twist_) {
        return 0.0f;
    }

    if (emergency_brake_) {
        return 1.0f;
    }

    if (stop_requested_) {
        return 0.8f;
    }

    const bool in_turn_zone = isTightTurnZone();
    const double desired_speed = limitedTargetSpeed();
    const double kp_brake = in_turn_zone ? 0.12 : 0.3;
    const double speed_error = desired_speed - current_speed_;

    if (speed_error >= (in_turn_zone ? -0.45 : 0.0)) {
        return 0.0f;
    }

    return static_cast<float>(
        std::clamp(-kp_brake * speed_error, 0.0, in_turn_zone ? 0.12 : 0.8)
    );
}

float Controller::getSteering()
{
    if (!has_pose_) {
        return 0.0f;
    }

    if (auto target = selectTrajectoryTarget()) {
        target_speed_ = target->speed;
        last_steering_ =
            computePurePursuitSteering(target->x, target->y);
        return last_steering_;
    }

    if (!has_waypoint_) {
        last_steering_ = 0.0f;
        return last_steering_;
    }

    last_steering_ = computeWaypointSteering();
    return last_steering_;
}

std::optional<Controller::TargetPoint>
Controller::selectTrajectoryTarget() const
{
    if (!has_trajectory_ || latest_trajectory_.points.empty()) {
        return std::nullopt;
    }

    const auto& points = latest_trajectory_.points;

    const double cos_yaw = std::cos(current_yaw_);
    const double sin_yaw = std::sin(current_yaw_);

    // ---------------------------------------------------------
    // 1. Find the trajectory point closest to the vehicle.
    // Prefer points that are not clearly behind the vehicle.
    // ---------------------------------------------------------
    std::size_t nearest_index = 0;
    double nearest_distance = std::numeric_limits<double>::max();
    double nearest_lateral = 0.0;
    bool found_forward_point = false;

    for (std::size_t i = 0; i < points.size(); ++i) {
        const double x = points[i].pose.position.x;
        const double y = points[i].pose.position.y;

        const double dx = x - current_x_;
        const double dy = y - current_y_;

        const double forward =
            cos_yaw * dx + sin_yaw * dy;

        const double lateral =
            -sin_yaw * dx + cos_yaw * dy;

        const double distance =
            std::hypot(forward, lateral);

        // Ignore points that are clearly behind the vehicle.
        if (forward < -1.0) {
            continue;
        }

        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_lateral = lateral;
            nearest_index = i;
            found_forward_point = true;
        }
    }

    if (!found_forward_point) {
        return std::nullopt;
    }

    // ---------------------------------------------------------
    // 2. Determine lookahead.
    // ---------------------------------------------------------
    double min_lookahead;
    double max_lookahead;

    if (isWaypoint118TurnZone()) {
        min_lookahead = 1.6;
        max_lookahead = 3.0;
    } else if (isWaypoint128TurnZone()) {
        // More aggressive correction around waypoint 130.
        min_lookahead = 1.2;
        max_lookahead = 2.4;
    } else if (isWaypoint104TurnZone()) {
        min_lookahead = 2.2;
        max_lookahead = 3.8;
    } else if (isWaypoint99EntryDelayZone()) {
        min_lookahead = 2.5;
        max_lookahead = 4.0;
    } else {
        min_lookahead = 4.0;
        max_lookahead = 10.0;
    }

    double lookahead = std::clamp(
        4.0 + current_speed_ * 0.8,
        min_lookahead,
        max_lookahead
    );

    // If the vehicle has already deviated significantly from the trajectory,
    // use a shorter lookahead so that Pure Pursuit corrects laterally faster.
    const double lateral_error = std::abs(nearest_lateral);

    if (isWaypoint128TurnZone()) {
        if (lateral_error > 1.5) {
            lookahead = 1.2;
        } else if (lateral_error > 0.8) {
            lookahead = std::min(lookahead, 1.6);
        } else {
            lookahead = std::min(lookahead, 2.2);
        }
    }

    // ---------------------------------------------------------
    // 3. Move forward along the trajectory by arc length.
    // ---------------------------------------------------------
    double accumulated_distance = 0.0;

    for (std::size_t i = nearest_index;
         i + 1 < points.size();
         ++i)
    {
        const double x1 = points[i].pose.position.x;
        const double y1 = points[i].pose.position.y;
        const double x2 = points[i + 1].pose.position.x;
        const double y2 = points[i + 1].pose.position.y;

        accumulated_distance += std::hypot(
            x2 - x1,
            y2 - y1
        );

        if (accumulated_distance >= lookahead) {
            return TargetPoint{
                x2,
                y2,
                points[i + 1].target_speed
            };
        }
    }

    // Use the final trajectory point if the remaining trajectory
    // is shorter than the requested lookahead.
    const auto& last = points.back();

    return TargetPoint{
        last.pose.position.x,
        last.pose.position.y,
        last.target_speed
    };
}

double Controller::limitedTargetSpeed() const
{
    // Demo acceleration zone:
    // accelerate between waypoint 17 and 30 to catch the leading vehicle.
    if (activeWaypointInRange(17, 60)||activeWaypointInRange(99,103)) {
        return 10.0;
    }

    if (!isTightTurnZone()) {
        return target_speed_;
    }

    // Keep the narrow corner slow, then restore normal speed after leaving it.
    const double steering_abs = std::abs(last_steering_);
    double speed_limit = isTightTurnCommitZone() ? 0.90 : 1.10;

    if (steering_abs > 0.55) {
        speed_limit = std::min(speed_limit, 0.78);
    } else if (steering_abs > 0.30) {
        speed_limit = std::min(speed_limit, 0.90);
    } else if (steering_abs > 0.15) {
        speed_limit = std::min(speed_limit, 1.00);
    }

    return std::clamp(target_speed_, 0.0, speed_limit);
}

bool Controller::isTightTurnZone() const
{
    return isWaypoint99TurnZone() ||
           isWaypoint104TurnZone() ||
           isWaypoint118TurnZone() ||
           isWaypoint128TurnZone();
}

bool Controller::isTightTurnCommitZone() const
{
    return isWaypoint99CommitTurnZone() ||
           isWaypoint104TurnZone() ||
           isWaypoint118CommitTurnZone() ||
           isWaypoint128TurnZone();
}

bool Controller::activeWaypointInRange(int first, int last) const
{
    return has_waypoint_index_ &&
           active_waypoint_index_ >= first &&
           active_waypoint_index_ <= last;
}

bool Controller::isWaypoint99TurnZone() const
{
    return activeWaypointInRange(93, 100);
}

bool Controller::isWaypoint99EntryDelayZone() const
{
    // Waypoint 96 is still the approach. Hold a medium steering cap until the
    // active target reaches waypoint 97.
    return activeWaypointInRange(93, 96);
}

bool Controller::isWaypoint99CommitTurnZone() const
{
    // Waypoints 97-100 are the actual narrow bend; waypoint 96 is only setup.
    return activeWaypointInRange(97, 100);
}

bool Controller::isWaypoint104TurnZone() const
{
    // Waypoints 104-108 are the second narrow bend at the top of the route.
    return activeWaypointInRange(104, 108);
}

bool Controller::isWaypoint118TurnZone() const
{
    // Waypoints 118-122 are the third narrow bend before the southbound road.
    return activeWaypointInRange(118, 122);
}

bool Controller::isWaypoint118CommitTurnZone() const
{
    // Slow down from waypoint 118, but only start the large turn near waypoint 119.
    return activeWaypointInRange(119, 122);
}

bool Controller::isWaypoint128TurnZone() const
{
    // Waypoints 128-131 are the fourth narrow bend into the westbound road.
    return activeWaypointInRange(128, 131);
}

double Controller::steeringLimit() const
{
    if (!isTightTurnZone()) {
        return 0.6;
    }

    if (isWaypoint99EntryDelayZone()) {
        return 0.30;
    }

    if (isWaypoint118CommitTurnZone()) {
        return 0.88;
    }

    if (isWaypoint128TurnZone()) {
        return 0.82;
    }

    return 0.75;
}

double Controller::tightTurnMinimumSteering() const
{
    if (isWaypoint118CommitTurnZone()) {
        return 0.80;
    }

    if (isWaypoint104TurnZone() || isWaypoint128TurnZone()) {
        return 0.68;
    }

    return 0.55;
}

float Controller::computePurePursuitSteering(double target_x, double target_y) const
{
    const double dx = target_x - current_x_;
    const double dy = target_y - current_y_;
    const double cos_yaw = std::cos(current_yaw_);
    const double sin_yaw = std::sin(current_yaw_);
    const double x_local = cos_yaw * dx + sin_yaw * dy;
    const double y_local = -sin_yaw * dx + cos_yaw * dy;
    const double lookahead_sq = std::max(1.0, x_local * x_local + y_local * y_local);

    const double wheelbase = 2.7;
    const double ros_steering = std::atan2(2.0 * wheelbase * y_local, lookahead_sq);

    // Unity's positive steering turns the car toward negative ROS yaw, so invert ROS steering.
    double unity_steering = -ros_steering;
    if (isTightTurnCommitZone() && std::abs(unity_steering) > 0.05) {
        unity_steering = std::copysign(
            std::max(std::abs(unity_steering), tightTurnMinimumSteering()),
            unity_steering
        );
    }

    const double steering_limit = steeringLimit();
    return static_cast<float>(
        std::clamp(unity_steering, -steering_limit, steering_limit)
    );
}

float Controller::computeWaypointSteering() const
{
    const double dx = waypoint_x_ - current_x_;
    const double dy = waypoint_y_ - current_y_;
    const double target_yaw = std::atan2(dy, dx);
    const double heading_error = normalizeAngle(target_yaw - current_yaw_);

    const double steering_limit = steeringLimit();
    const double ros_steering =
        std::clamp(0.75 * heading_error, -steering_limit, steering_limit);
    double unity_steering = -ros_steering;
    if (isTightTurnCommitZone() && std::abs(unity_steering) > 0.05) {
        unity_steering = std::copysign(
            std::max(std::abs(unity_steering), tightTurnMinimumSteering()),
            unity_steering
        );
    }

    return static_cast<float>(
        std::clamp(unity_steering, -steering_limit, steering_limit)
    );
}

