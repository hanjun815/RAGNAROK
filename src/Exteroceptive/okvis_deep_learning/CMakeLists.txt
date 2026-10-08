cmake_minimum_required(VERSION 3.5)
project(okvis_deep_learning)

find_package(Eigen3 REQUIRED)
if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV COMPONENTS core imgproc features2d calib3d dnn REQUIRED)

set(LIB_NAME okvis_deep_learning)

add_library(${LIB_NAME} src/Processor.cpp)
if(USE_NN)
    target_sources(${LIB_NAME}
        PRIVATE
            src/Stereo2DepthProcessor.cpp 
            src/DepthFusionProcessor.cpp)
endif()

target_include_directories(${LIB_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS} 
    ${Torch_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    okvis::Ceres
    okvis::Common
    okvis::CV
    okvis::Frontend
    okvis::Kinematics
    okvis::Mapping
    okvis::MultisensorProcessing
    okvis::Time
    okvis::Timing
    okvis::Util
)

if(USE_NN)
    target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
    )
endif()


add_library(okvis::DeepLearning ALIAS ${LIB_NAME})
install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE COMPONENT lib
  LIBRARY COMPONENT lib
)
install(DIRECTORY include/okvis TYPE INCLUDE COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
