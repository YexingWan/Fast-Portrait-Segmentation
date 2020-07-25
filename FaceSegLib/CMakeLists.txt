cmake_minimum_required(VERSION 3.0)
project(FaceSegLib VERSION 1.0)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/ FaceSegLibSRC)

add_library(FaceSegLib SHARED ${FaceSegLibSRC})

# target_link_libraries(FaceSegLib MNN)
