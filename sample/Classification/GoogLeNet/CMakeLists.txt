cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( GoogLeNet util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "GoogLeNet" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( GoogLeNet ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/bvlc_googlenet.caffemodel"
  EXPECTED_HASH SHA256=6F7101E3A2183738A7125A0C5021BA82A1FEB4228C5CA0924D991B6DAF6F6FAD
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/deploy.prototxt"
  EXPECTED_HASH SHA256=FA36CD4112A3240A29600A5AA35333F31BEC33DAA1A6085B0AF308A9F50F0C50
  SHOW_PROGRESS
)
