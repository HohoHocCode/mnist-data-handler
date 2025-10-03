cmake_minimum_required(VERSION 3.10)
project(mnist_ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Thu thập các file nguồn
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/data.cpp
    ${CMAKE_SOURCE_DIR}/src/data_handler.cpp
    ${CMAKE_SOURCE_DIR}/main.cpp
)

add_executable(mnist_ml ${SOURCES})