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

#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <ad_interfaces/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <optional>

class Controller
{
public:
    Controller();

    void updateTrajectory(const ad_interfaces::msg::Trajectory& msg);
    void updatePose(const geometry_msgs::msg::PoseStamped& msg);
    void updateTwist(const geometry_msgs::msg::TwistStamped& msg);
    void updateWaypoint(const geometry_msgs::msg::PoseStamped& msg);
    void updateActiveWaypointIndex(int index);
    void updateTargetSpeed(const std_msgs::msg::Float32& msg);
    void updateStopRequest(const std_msgs::msg::Bool& msg);
    void updateEmergencyBrake(const std_msgs::msg::Bool& msg);


    bool inputsAreStale() const;

    float getThrottle();
    float getSteering();
    float getBrake();

private:
    bool has_pose_;
    bool has_twist_;
    bool has_waypoint_;
    bool has_waypoint_index_;
    bool stop_requested_;
    bool emergency_brake_;
    bool has_decision_input_;

    static constexpr double kWatchdogTimeoutSec = 2.0;
    std::chrono::steady_clock::time_point last_pose_time_;
    std::chrono::steady_clock::time_point last_twist_time_;
    std::chrono::steady_clock::time_point last_decision_time_;

    double current_x_;
    double current_y_;
    double current_yaw_;
    double current_speed_;

    double waypoint_x_;
    double waypoint_y_;
    int active_waypoint_index_;

    double target_speed_;

    ad_interfaces::msg::Trajectory latest_trajectory_;
    bool has_trajectory_;
    float last_steering_;
    float last_throttle_;

    struct TargetPoint
    {
        double x;
        double y;
        double speed;
    };

    std::optional<TargetPoint> selectTrajectoryTarget() const;
    bool isStale(const std::chrono::steady_clock::time_point& last) const;
    bool activeWaypointInRange(int first, int last) const;
    double limitedTargetSpeed() const;
    bool isTightTurnZone() const;
    bool isTightTurnCommitZone() const;
    bool isWaypoint99TurnZone() const;
    bool isWaypoint99EntryDelayZone() const;
    bool isWaypoint99CommitTurnZone() const;
    bool isWaypoint104TurnZone() const;
    bool isWaypoint118TurnZone() const;
    bool isWaypoint118CommitTurnZone() const;
    bool isWaypoint128TurnZone() const;
    double steeringLimit() const;
    double tightTurnMinimumSteering() const;
    float computePurePursuitSteering(double target_x, double target_y) const;
    float computeWaypointSteering() const;
};

#endif
