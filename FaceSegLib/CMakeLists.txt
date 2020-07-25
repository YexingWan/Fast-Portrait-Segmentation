cmake_minimum_required(3.0)
project(FaceSegLib 1.0)


include_directories(${CMAKE_CURRENTLIST_DIR}/include)
aux_source_directory(${CMAKE_CURRENTLIST_DIR}/src FaceSegLibSRC)

add_library(FaceSegLib SHARED ${FaceSegLibSRC})

# target_link_libraries()