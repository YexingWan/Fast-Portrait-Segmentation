cmake_minimum_required(VERSION 3.0)

project(FaceSeg VERSION 1.0)
set(CMAKE_C_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

option(BUILD_TESTS "NOTE:whether to build test program" OFF)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${MNN_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS} ${MNN_LIBRARY_DIRS})

add_subdirectory(${CMAKE_SOURCE_DIR}/FaceSegLib)
include_directories(${CMAKE_SOURCE_DIR}/FaceSegLib/include)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/FaceSeg/ FaceSegSRC)

add_executable(FaceSeg ${FaceSegSRC})

target_link_libraries(FaceSeg FaceSegLib MNN ${OpenCV_LIBS})

if(BUILD_TESTS)
  aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/test/ testPreprocessSRC)
  add_executable(testPreprocess ${testPreprocessSRC})
  target_link_libraries(testPreprocess FaceSegLib MNN ${OpenCV_LIBS})
endif()
