cmake_minimum_required(VERSION 3.10)

# Tên project
project(MNISTHandler)

# Sử dụng C++17
set(CMAKE_CXX_STANDARD 17)

# Đặt thư mục include
include_directories(include)

# Thêm các file nguồn
set(SOURCES
    src/data.cpp
    src/data_handler.cpp
)

# Tạo static library từ các source
add_library(MNISTHandlerLib ${SOURCES})

# Nếu không có main.cpp, bạn có thể tạo library
# add_library(MNISTHandlerLib ${SOURCES})
