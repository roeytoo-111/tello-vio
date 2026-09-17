cmake_minimum_required(VERSION 3.5)
project(chase_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# ROS 2 interface generators use PYTHON_EXECUTABLE (ament_cmake_python).
# Some systems accidentally pick an old /usr/local python; prefer /usr/bin.
# (Same guard as tello_msg, which hit this.)
if(EXISTS "/usr/bin/python3")
  set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python interpreter" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DroneDetection.msg"
  DEPENDENCIES std_msgs builtin_interfaces
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
