cmake_minimum_required(VERSION 3.8)
project(dummy_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(simulation REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ad_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(include)

add_executable(dummy_controller_node
  src/dummy_controller_node.cpp
  src/controller.cpp
)

ament_target_dependencies(dummy_controller_node
  rclcpp
  simulation
  geometry_msgs
  ad_interfaces
  std_msgs
)

install(TARGETS
  dummy_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(controller_test
    test/controller_test.cpp
    src/controller.cpp
  )
  target_include_directories(controller_test PRIVATE include)
  ament_target_dependencies(controller_test
    geometry_msgs
    ad_interfaces
    std_msgs
  )
endif()

ament_package()
