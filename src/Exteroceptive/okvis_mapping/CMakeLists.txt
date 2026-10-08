cmake_minimum_required(VERSION 3.8...3.16)

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

set(LIB_NAME okvis_mapping)
add_library(${LIB_NAME}
  src/VoxelGridFilter.cpp
  src/SubmappingUtils.cpp
  src/LidarMotionUndistortion.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GLOG_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    ${GLOG_LIBRARIES}
    Eigen3::Eigen
    SRL::Supereight2
    SRL::Projection
    okvis::Common
    okvis::Util
    okvis::Timing
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)

if(USE_COLIDMAP)
  target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_COLIDMAP)
endif()
add_library(okvis::Mapping ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE COMPONENT lib
  LIBRARY COMPONENT lib
)
install(DIRECTORY include/okvis TYPE INCLUDE COMPONENT dev FILES_MATCHING PATTERN "*.hpp")

# testing
if(BUILD_TESTS)
        if(APPLE)
                add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
        else()
                add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
        endif(APPLE)
        enable_testing()
        set(PROJECT_TEST_NAME ${LIB_NAME}_test)
        add_executable(${PROJECT_TEST_NAME}
                test/runTests.cpp
                test/voxelGridTests.cpp
        )
        target_link_libraries(${PROJECT_TEST_NAME}
                ${LIB_NAME}
                gmock
                gtest
        )
        target_compile_options(${PROJECT_TEST_NAME}
                PUBLIC
                ${OKVIS_PUBLIC_CXX_FLAGS}
                PRIVATE
                ${OKVIS_PRIVATE_CXX_FLAGS}
        )
        add_test(test ${PROJECT_TEST_NAME})
endif()