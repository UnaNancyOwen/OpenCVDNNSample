cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( ENet util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "ENet" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( ENet ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/e-lab/ENet-training/releases/download/v1.cs/model-cityscapes.net )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/model-cityscapes.net"
  EXPECTED_HASH SHA256=8DC0A04A47109B4D526E6DCAEF9E60D4EBB90C853DD5FD5BAFC882BB68299ECB
  SHOW_PROGRESS
)






