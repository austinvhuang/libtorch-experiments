cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nca)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(nca nca.cpp)
target_link_libraries(nca "${TORCH_LIBRARIES}")
set_property(TARGET nca PROPERTY CXX_STANDARD 14)
