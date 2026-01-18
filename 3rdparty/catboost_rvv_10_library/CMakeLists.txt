cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(catboost_rvv)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(cmake/function.cmake)

add_subdirectory(catboost_rvv_lib)
add_subdirectory(mainCalcIndexesBasic)
add_subdirectory(mainCalculateLeafValues)
add_subdirectory(mainBinarizeFloats)
add_subdirectory(mainL2SqrDistance)