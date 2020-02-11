cmake_minimum_required(VERSION 3.15)
project(dwisy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)
include_directories(include)

add_executable(dwisy main.cpp)
target_link_libraries(dwisy ${OpenCV_LIBS})