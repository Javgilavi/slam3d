cmake_minimum_required(VERSION 3.16)
project(slam3d_stella_driver CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(stella_vslam REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(slam3d_stella_driver main.cc)
target_link_libraries(slam3d_stella_driver PRIVATE stella_vslam::stella_vslam ${OpenCV_LIBS})
# spdlog / nlohmann json headers vendored by stella_vslam
target_include_directories(slam3d_stella_driver PRIVATE
  ${STELLA_SRC_DIR}/3rd/spdlog/include ${STELLA_SRC_DIR}/3rd/json/include)
