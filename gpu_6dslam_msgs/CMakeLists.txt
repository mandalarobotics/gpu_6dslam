cmake_minimum_required(VERSION 2.8.3)
project(gpu_6dslam_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs )
#find_package(catkin REQUIRED COMPONENTS message_generation std_msgs geometry_msgs)



add_service_files(
   FILES
   regsiterAll.srv
 )
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
 )
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES m3d_driver
  CATKIN_DEPENDS message_runtime std_msgs
#  DEPENDS system_lib
)
