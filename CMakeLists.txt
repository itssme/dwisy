cmake_minimum_required(VERSION 3.15)
project(dwisy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)
include_directories(include)

add_executable(dwisy main.cpp include/utils.h)
target_link_libraries(dwisy ${OpenCV_LIBS})

if (NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    set(RESOURCE_FILES res/haarcascade_frontalface_default.xml)
    foreach(item IN LISTS RESOURCE_FILES)
        message(STATUS ${item})
        file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/${item}" DESTINATION "${CMAKE_BINARY_DIR}/res")
    endforeach()
endif()
