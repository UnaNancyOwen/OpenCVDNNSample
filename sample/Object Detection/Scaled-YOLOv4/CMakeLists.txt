cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( Scaled-YOLOv4 util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "Scaled-YOLOv4" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( Scaled-YOLOv4 ${OpenCV_LIBS} )
endif()

# Download Model
# Please download pre-trained model from this url.
# https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp#testing
#set( MODEL https://drive.google.com/u/0/uc?id=1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL )
#file( DOWNLOAD 
#  "${MODEL}"
#  "${CMAKE_CURRENT_LIST_DIR}/yolov4-csp.weights"
#  EXPECTED_HASH SHA256=1652B8C5ADA032EA03B74A719B4D804971EA5BBFB55D16C54C861A7BD48FC960
#  SHOW_PROGRESS
#)

set( MODEL https://drive.google.com/u/0/uc?id=1XLVy_DMjvhhmHucSypL3zeDGDdmCGrsu )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov4-tiny.weights"
  EXPECTED_HASH SHA256=CF9FBFD0F6D4869B35762F56100F50ED05268084078805F0E7989EFE5BB8CA87
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/WongKinYiu/ScaledYOLOv4/yolov4-csp/models/yolov4-csp.cfg )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov4-csp.cfg"
  EXPECTED_HASH SHA256=064bb5d2ba156c2b98ae251752e39dd539892987eec663469fd38a8fc0c4bb7a
  SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/WongKinYiu/ScaledYOLOv4/yolov4-tiny/cfg/yolov4-tiny.cfg )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/yolov4-tiny.cfg"
  EXPECTED_HASH SHA256=DCD80349511A005FA5B7FF93B996351F817FADB4932CC2AFBC9AB3E9B4F5F67A
  SHOW_PROGRESS
)

