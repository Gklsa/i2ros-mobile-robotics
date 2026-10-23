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

#include <gtest/gtest.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

#include "controller.hpp"

TEST(ControllerTest, HoldsOutputsWithoutSensorInput)
{
  Controller controller;

  EXPECT_FLOAT_EQ(controller.getThrottle(), 0.0F);
  EXPECT_FLOAT_EQ(controller.getBrake(), 0.0F);
  EXPECT_FLOAT_EQ(controller.getSteering(), 0.0F);
}

TEST(ControllerTest, AcceleratesTowardDecisionTargetSpeed)
{
  Controller controller;
  geometry_msgs::msg::TwistStamped twist;
  std_msgs::msg::Float32 target_speed;
  target_speed.data = 2.0F;

  controller.updateTwist(twist);
  controller.updateTargetSpeed(target_speed);

  EXPECT_GT(controller.getThrottle(), 0.0F);
  EXPECT_FLOAT_EQ(controller.getBrake(), 0.0F);
}

TEST(ControllerTest, StopAndEmergencyRequestsOverrideLongitudinalControl)
{
  Controller controller;
  geometry_msgs::msg::TwistStamped twist;
  twist.twist.linear.x = 2.0;
  controller.updateTwist(twist);

  std_msgs::msg::Bool stop;
  stop.data = true;
  controller.updateStopRequest(stop);
  EXPECT_FLOAT_EQ(controller.getThrottle(), 0.0F);
  EXPECT_FLOAT_EQ(controller.getBrake(), 0.8F);

  stop.data = false;
  controller.updateStopRequest(stop);
  std_msgs::msg::Bool emergency;
  emergency.data = true;
  controller.updateEmergencyBrake(emergency);
  EXPECT_FLOAT_EQ(controller.getThrottle(), 0.0F);
  EXPECT_FLOAT_EQ(controller.getBrake(), 1.0F);
}

TEST(ControllerTest, SteersTowardWaypointUsingUnitySignConvention)
{
  Controller controller;
  geometry_msgs::msg::PoseStamped pose;
  pose.pose.orientation.w = 1.0;
  controller.updatePose(pose);

  geometry_msgs::msg::PoseStamped waypoint;
  waypoint.pose.position.x = 10.0;
  waypoint.pose.position.y = 2.0;
  controller.updateWaypoint(waypoint);

  const float steering = controller.getSteering();
  EXPECT_LT(steering, 0.0F);
  EXPECT_GE(steering, -0.6F);
}
