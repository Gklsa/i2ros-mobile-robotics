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

#include <rclcpp/rclcpp.hpp>

#include <simulation/msg/vehicle_control.hpp>
#include <ad_interfaces/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>

#include "controller.hpp"

constexpr float loop_interval = 0.05f;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("dummy_controller_node");
    Controller controller;

    auto pub = node->create_publisher<simulation::msg::VehicleControl>(
        "/car_command",
        10
    );

    auto pose_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/OurCar/CoM/pose",
        10,
        [&controller](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
            controller.updatePose(*msg);
        }
    );

    auto twist_sub = node->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/OurCar/CoM/twist",
        10,
        [&controller](const geometry_msgs::msg::TwistStamped::SharedPtr msg)
        {
            controller.updateTwist(*msg);
        }
    );

    auto waypoint_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/planning/next_waypoint",
        10,
        [&controller](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
            controller.updateWaypoint(*msg);
        }
    );

    auto waypoint_index_sub = node->create_subscription<std_msgs::msg::Int32>(
        "/planning/active_waypoint_index",
        10,
        [&controller](const std_msgs::msg::Int32::SharedPtr msg)
        {
            controller.updateActiveWaypointIndex(msg->data);
        }
    );

    auto trajectory_sub = node->create_subscription<ad_interfaces::msg::Trajectory>(
        "/planning/trajectory",
        10,
        [&controller](const ad_interfaces::msg::Trajectory::SharedPtr msg)
        {
            controller.updateTrajectory(*msg);
        }
    );

    auto target_speed_sub = node->create_subscription<std_msgs::msg::Float32>(
        "/decision/target_speed",
        10,
        [&controller](const std_msgs::msg::Float32::SharedPtr msg)
        {
            controller.updateTargetSpeed(*msg);
        }
    );

    auto stop_request_sub = node->create_subscription<std_msgs::msg::Bool>(
        "/decision/stop_request",
        10,
        [&controller](const std_msgs::msg::Bool::SharedPtr msg)
        {
            controller.updateStopRequest(*msg);
        }
    );

    auto emergency_brake_sub = node->create_subscription<std_msgs::msg::Bool>(
        "/decision/emergency_brake",
        10,
        [&controller](const std_msgs::msg::Bool::SharedPtr msg)
        {
            controller.updateEmergencyBrake(*msg);
        }
    );

    rclcpp::Rate loop_rate(1.0 / loop_interval);

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        simulation::msg::VehicleControl msg;
        msg.steering = controller.getSteering();
        msg.throttle = controller.getThrottle();
        msg.brake = controller.getBrake();
        msg.reserved = 0.0f;

        if (controller.inputsAreStale()) {
            RCLCPP_WARN_THROTTLE(
                node->get_logger(), *node->get_clock(), 1000,
                "Controller watchdog: pose/twist/decision inputs are stale, "
                "forcing a stop (brake) until fresh data arrives."
            );
        }

        pub->publish(msg);
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
