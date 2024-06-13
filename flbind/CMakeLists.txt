cmake_minimum_required(VERSION 3.10)
project(flbind)


find_package(Eigen3 REQUIRED)

add_subdirectory(deps/pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(flbind
  src/core.cpp
)

include_directories(flbind ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(flbind PRIVATE Eigen3::Eigen)
