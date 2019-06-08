cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( MobileNet-SSD util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "MobileNet-SSD" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( MobileNet-SSD ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://drive.google.com/uc?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/MobileNetSSD_deploy.caffemodel"
  EXPECTED_HASH SHA256=761C86FBAE3D8361DD454F7C740A964F62975ED32F4324B8B85994EDEC30F6AF
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/MobileNetSSD_deploy.prototxt"
  EXPECTED_HASH SHA256=E781559C4F5BEAEC2A486CCD952AF5B6FA408E9498761BF5F4FB80B4E9F0D25E
  SHOW_PROGRESS
)
