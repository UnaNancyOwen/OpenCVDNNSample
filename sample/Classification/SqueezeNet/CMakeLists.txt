cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( SqueezeNet util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "SqueezeNet" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( SqueezeNet ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/squeezenet_v1.1.caffemodel"
  EXPECTED_HASH SHA256=72B912ACE512E8621F8FF168A7D72AF55910D3C7C9445AF8DFBFF4C2EE960142
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/deploy.prototxt"
  EXPECTED_HASH SHA256=D041BFB2AB4B32FDA4FF6C6966684132F2924E329916AA5BFE9285C6B23E3D1C
  SHOW_PROGRESS
)
