cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( YOLOv4 util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "YOLOv4" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( YOLOv4 ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov4.weights"
  EXPECTED_HASH SHA256=e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562
  SHOW_PROGRESS
)


# Download Config
set( CONFIG https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov4.cfg"
  EXPECTED_HASH SHA256=a6d0f8e5c62cc8378384f75a8159b95fa2964d4162e33351b00ac82e0fc46a34
  SHOW_PROGRESS
)
