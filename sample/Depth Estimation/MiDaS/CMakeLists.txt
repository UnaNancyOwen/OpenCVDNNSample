cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( MiDaS util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "MiDaS" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( MiDaS ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.onnx )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/model-f6b98070.onnx"
  EXPECTED_HASH SHA256=360583846cc3d0bac9c1735313a32140956b1badf20badc6ab9caf662136587d
  SHOW_PROGRESS
)

set( MODEL https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/model-small.onnx"
  EXPECTED_HASH SHA256=2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58
  SHOW_PROGRESS
)
