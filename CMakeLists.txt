cmake_minimum_required(VERSION 2.8.6)
project(diffqc)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++17)
add_compile_options(-std=c++17)
add_compile_options(-Wdeprecated-declarations)

add_definitions(-DSOFTCON_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")

message("Flags : " ${CMAKE_CXX_FLAGS})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)


add_subdirectory(pybind11)
pybind11_add_module(diffqc diffqc.cc)

find_package(Eigen3 3.3.3 REQUIRED)
find_package(Boost REQUIRED)
if (EIGEN3_FOUND)
   include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if (Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIR})
endif()

target_compile_definitions(diffqc PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
