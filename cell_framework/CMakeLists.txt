cmake_minimum_required(VERSION 3.26)
project(CellFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API 1)

# Directory for generated headers
set(GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

add_custom_target(generate_headers
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_headers.cmake
    BYPRODUCTS ${GENERATED_INCLUDE_DIR}/Cell/Core.hpp
    VERBATIM
)

add_library(CellCore)

target_sources(CellCore
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/Cell/Core.ixx
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/Cell/Core.cpp
)

target_include_directories(CellCore PUBLIC ${GENERATED_INCLUDE_DIR})

add_dependencies(CellCore generate_headers)

add_executable(cell_app main.cpp)

target_link_libraries(cell_app PRIVATE CellCore)

add_dependencies(cell_app generate_headers)
