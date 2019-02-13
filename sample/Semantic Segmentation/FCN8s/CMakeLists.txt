cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( FCN8s util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "FCN8s" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( FCN8s ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/fcn8s-heavy-pascal.caffemodel"
  EXPECTED_HASH SHA256=A2C5E90540BB12FFEDCA4867796DB59D7CD61C2FF4ED4AEE9F0F3A0713BAC3AA
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn8s/deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/fcn8s-heavy-pascal.prototxt"
  EXPECTED_HASH SHA256=4E87632ABA2CDA27E7DC810A6197D6F66FCD631045A595DEC6BB4A4C90B7E9CA
  SHOW_PROGRESS
)

