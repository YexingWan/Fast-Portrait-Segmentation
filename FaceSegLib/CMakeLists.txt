cmake_minimum_required(VERSION 3.0)
project(FaceSegLib VERSION 1.0)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${MNN_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS} ${MNN_LIBRARY_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/ FaceSegLibSRC)

add_library(FaceSegLib SHARED ${FaceSegLibSRC})

target_link_libraries(FaceSegLib MNN ${OpenCV_LIBS})
