cmake_minimum_required(VERSION 3.1...3.16)

project(srl-projection-install-test VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(SRLProjection REQUIRED)

# Compile a simple executable to ensure that SRLProjection was correctly installed system-wide
add_executable(${PROJECT_NAME} "srl_projection_install_test.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE SRL::Projection)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

