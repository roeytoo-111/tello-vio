cmake_minimum_required(VERSION 3.5)
project(orbslam2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Default to Release. Colcon leaves CMAKE_BUILD_TYPE empty unless told
# otherwise, which compiles this wrapper at -O0. The heavy lifting happens
# inside libORB_SLAM2.so (built Release by scripts/orbslam.sh), so the penalty
# is a couple of milliseconds per frame rather than a catastrophe -- but it is
# a couple of milliseconds out of a 33 ms budget for no reason at all.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "orbslam2: CMAKE_BUILD_TYPE was empty; defaulting to Release")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # -Wpedantic is deliberately absent: ORB-SLAM2's headers are not pedantic-clean
  # and the resulting wall of warnings hides our own.
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

find_package(Boost COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

# Configuration files are useful even when the node itself cannot be built --
# ORB-SLAM2's System constructor calls exit(-1) if it cannot open the settings
# file, so shipping it is what makes `ros2 run orbslam2 mono <voc> <cfg>` work.
install(FILES config.yaml DESTINATION share/${PROJECT_NAME})

# ORB_SLAM2 is an external (non-ROS) library and may not be installed.
# Two independent reasons to skip the node: the user asked us to
# (ORB_SLAM2_ENABLE=OFF), or we could not find the library.
#
# The previous form put only the find_package inside `if(ORB_SLAM2_ENABLE)` and
# left add_executable outside it, so passing OFF -- the documented off switch --
# skipped the *discovery* and then tried to build anyway, failing on a missing
# System.h. The flag did the exact opposite of its name.
option(ORB_SLAM2_ENABLE "Build the ORB-SLAM2 wrapper node" ON)

set(_build_node FALSE)
if(ORB_SLAM2_ENABLE)
  find_package(ORB_SLAM2 QUIET)
  if(ORB_SLAM2_FOUND)
    set(_build_node TRUE)
  else()
    message(WARNING
      "ORB_SLAM2 not found; skipping the orbslam2 node. "
      "Set ORB_SLAM2_ROOT_DIR (or install ORB_SLAM2) and rebuild to enable.")
  endif()
else()
  message(STATUS "ORB_SLAM2_ENABLE=OFF; skipping the orbslam2 node as requested.")
endif()

if(NOT _build_node)
  ament_package()
  return()
endif()

add_executable(mono
  src/monocular/mono.cpp
  src/monocular/monocular-slam-node.cpp
)

# SYSTEM: ORB-SLAM2 and Eigen headers are third-party, and their warnings would
# otherwise drown ours.
target_include_directories(mono SYSTEM PRIVATE
  ${ORB_SLAM2_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
)
target_include_directories(mono PRIVATE ${message_filters_INCLUDE_DIRS})

target_link_libraries(mono
  ${ORB_SLAM2_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)

ament_target_dependencies(mono
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  visualization_msgs
  tf2_ros
  cv_bridge
  message_filters
)

install(TARGETS mono DESTINATION lib/${PROJECT_NAME})
ament_package()
