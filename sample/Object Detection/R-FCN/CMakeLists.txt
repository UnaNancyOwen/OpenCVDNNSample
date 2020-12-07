cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( R-FCN util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "R-FCN" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( R-FCN ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://1drv.ws/u/s!AoN7vygOjLIQqUWHpY67oaC7mopf )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/rfcn_models.tar.gz"
  EXPECTED_HASH SHA256=A4BE59F5C05D2EDE32B0C132BF0AB95A0C878586DDAC1971761C0110CC84D0FF
  SHOW_PROGRESS
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzfv "${CMAKE_CURRENT_LIST_DIR}/rfcn_models.tar.gz"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/rfcn_models/resnet50_rfcn_final.caffemodel" "${CMAKE_CURRENT_LIST_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/rfcn_pascal_voc_resnet50.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/rfcn_pascal_voc_resnet50.prototxt"
  EXPECTED_HASH SHA256=1C0D8B6E4D3BE63AD7D67A1FE7DD8E5CDDFC7B77A724EEA75BE1239CEAC5A170
  SHOW_PROGRESS
)

