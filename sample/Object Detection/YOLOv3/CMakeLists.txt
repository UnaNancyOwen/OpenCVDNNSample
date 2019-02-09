cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( YOLOv3 util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "YOLOv3" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( YOLOv3 ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://pjreddie.com/media/files/yolov3.weights )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov3.weights"
  EXPECTED_HASH SHA256=523E4E69E1D015393A1B0A441CEF1D9C7659E3EB2D7E15F793F060A21B32F297
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov3.cfg"
  EXPECTED_HASH SHA256=22489EA38575DFA36C67A90048E8759576416A79D32DC11E15D2217777B9A953
  SHOW_PROGRESS
)
