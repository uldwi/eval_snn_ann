# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.22)

# ESP-IDF manages optimization through sdkconfig and its own compiler flags.
# Its imported component targets do not define CMake Debug/Release locations;
# IDE build profiles must therefore leave the generic build type unset.
if(CMAKE_BUILD_TYPE)
    message(STATUS "ESP-IDF uses sdkconfig for build mode; clearing CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE "" CACHE STRING "" FORCE)
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(eval_snn_ann)
