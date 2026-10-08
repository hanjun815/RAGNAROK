cmake_minimum_required(VERSION 3.1...3.16)

project(srl-projection VERSION 1.0.0 LANGUAGES C CXX)

# Options
option(SRL_PROJECTION_BUILD_TESTS "Enable testing" ON)


# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of CMAKE_BUILD_TYPE for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
# Warn if the build type is not Release
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}. Performance may be terrible.")
else()
  message(STATUS "CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}.")
endif()

# Create the library ###############################################################################
# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS highgui imgproc core features2d)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
		${EIGEN3_INCLUDE_DIR}
		${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  INTERFACE
		${OpenCV_LIBS}
)

# Add an alias so that the library can be used inside the build tree, e.g. when testing
add_library(SRL::Projection ALIAS ${PROJECT_NAME})
# This is required so that the exported target has the name Projection and not srl-projection
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME Projection)



# Run tests ########################################################################################
if(SRL_PROJECTION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()



# Install the library ##############################################################################
set(PACKAGE_NAME SRLProjection)
# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)
# Install includes
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
# The directory where the .cmake files will be installed
if(WIN32 AND NOT CYGWIN)
  set(CMAKE_INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
  set(CMAKE_INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
  set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()

# Create a Config.cmake file
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)

# Create a ConfigVersion.cmake file
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)

# Export the targets to a script
install(EXPORT ${PROJECT_NAME}-targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    SRL::
  DESTINATION
    "${CMAKE_INSTALL_CMAKEDIR}"
)
# Install the headers
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# Install the Config and ConfigVersion files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)



# Export from the build tree #######################################################################
export(EXPORT ${PROJECT_NAME}-targets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Targets.cmake"
	NAMESPACE SRL::
)
# Register package in user's package registry
export(PACKAGE ${PACKAGE_NAME})

