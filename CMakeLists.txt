cmake_minimum_required(VERSION 3.15)
project(dwisy)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(dwisy main.cpp)
target_link_libraries(dwisy ${OpenCV_LIBS})