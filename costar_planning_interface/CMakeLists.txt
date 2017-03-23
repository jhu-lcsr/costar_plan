cmake_minimum_required(VERSION 2.8.3)
project(costar_planning_interface)


add_definitions(-g -std=c++0x -pthread -O1)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  costar_plan_msgs
  rospy
  
  dmp
  #costar_experiments
  #costar_tasks

  kdl_parser
  urdf

  # TF
  tf
  tf_conversions

  # MoveIt stuff
  moveit_core
  moveit_msgs
  moveit_ros_perception
  moveit_ros_planning_interface
  moveit_ros_planning

  # messages
  sensor_msgs
  trajectory_msgs

  # ros stuff
  rosbag

  # action
  actionlib
)

## System dependencies are found with CMake's conventions
find_package(orocos_kdl REQUIRED)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS python thread)
find_package(PythonLibs 2.7 REQUIRED)

find_package(PkgConfig REQUIRED)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ltl_tree_search
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${orocos_kdl_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${EIGEN3_INCLUDE_DIRS} )

set(PY_SOURCES
  cpp/costar_planner.cpp
  cpp/collision_map.cpp
)

set(SOURCES 
  cpp/costar_planner.cpp
  cpp/collision_map.cpp
  cpp/features.cpp
  cpp/test_features.cpp
  cpp/training_features.cpp
  cpp/skill.cpp
  cpp/visualize.cpp
  cpp/robot_kinematics.cpp
  cpp/task_model.cpp
  cpp/instantiated_skill.cpp
  cpp/instantiated_skill_init.cpp

  cpp/traj/trajectory_distribution.cpp
  cpp/traj/dmp_trajectory_distribution.cpp

  cpp/utils/commander.cpp
  cpp/utils/params.cpp

  cpp/dist/utils.cc
)

set(PY_HEADERS
  include/costar_task_plan/costar_planner.h
  include/costar_task_plan/collision_map.h
)

set(HEADERS
  include/costar_task_plan/costar_planner.h
  include/costar_task_plan/collision_map.h
  include/costar_task_plan/test_features.h
  include/costar_task_plan/skill.h
  include/costar_task_plan/training_features.h
  include/costar_task_plan/features.h
  include/costar_task_plan/visualize.h
  include/costar_task_plan/trajectory_distribution.h
  include/costar_task_plan/dmp_trajectory_distribution.h
  include/costar_task_plan/robot_kinematics.h
  include/costar_task_plan/task_model.h
  include/costar_task_plan/instantiated_skill.h

  include/costar_task_plan/utils/commander.h
  include/costar_task_plan/utils/params.h

  include/costar_task_plan/dist/gmm.h
  include/costar_task_plan/dist/normal.h
  include/costar_task_plan/dist/utils.h
)

## Declare a cpp library
add_library(costar_planner ${SOURCES} ${HEADERS})
add_library(pycostar_planner ${PY_SOURCES} ${PY_HEADERS})

SET(PY_PROJECT_NAME costar_task_plan)
SET(PY_PACKAGE_DIR python/costar_task_plan)

set_target_properties(pycostar_planner
  PROPERTIES COMPILE_FLAGS -DGEN_PYTHON_BINDINGS
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
#add_python_export_library(${PY_PROJECT_NAME} ${PY_PACKAGE_DIR}
#  cpp/costar_planner.cpp
#  include/costar_task_plan/costar_planner.h
#)

## Declare a cpp executable
add_executable(costar_plan_node cpp/test/costar_plan_node.cpp)
add_executable(moveit_robot_model_test cpp/test/moveit_robot_test.cpp)

## Add cmake target dependencies of the executable/libname,rary
## as an example, message headers may need to be generated before nodes
add_dependencies(costar_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(costar_plan_node costar_planner)
#add_dependencies(features_test costar_planner)

## Specify libraries to link a library or executable target against
target_link_libraries(costar_planner
  ${orocos_kdl_LIBRARIES} ${catkin_LIBRARIES} ${GCOP_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES} ${PYTHON_LIBRARIES}
)
target_link_libraries(pycostar_planner
  ${orocos_kdl_LIBRARIES} ${catkin_LIBRARIES} ${GCOP_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES} ${PYTHON_LIBRARIES}
  #${catkin_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
)

SET(ALL_LIBRARIES 
  costar_planner ${orocos_kdl_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES} ${PYTHON_LIBRARIES}
)

target_link_libraries(costar_plan_node ${ALL_LIBRARIES})
target_link_libraries(moveit_robot_model_test ${ALL_LIBRARIES})

## Declare a C++ library
# add_library(ltl_tree_search
#   cpp/${PROJECT_NAME}/ltl_tree_search.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(ltl_tree_search ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(ltl_tree_search_node cpp/ltl_tree_search_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(ltl_tree_search_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(ltl_tree_search_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_#ython_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ltl_tree_search ltl_tree_search_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ltl_tree_search.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
