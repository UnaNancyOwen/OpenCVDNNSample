cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( Mask-RCNN util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "Mask-RCNN" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( Mask-RCNN ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
  EXPECTED_HASH SHA256=4F57861C2FF1C87E92495789EE266E41A7DC79A038521D93A153BE403B0B601F
  SHOW_PROGRESS
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzfv "${CMAKE_CURRENT_LIST_DIR}/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" "${CMAKE_CURRENT_LIST_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
  EXPECTED_HASH SHA256=090923144D81B2E442093F965BE27ECCDA11CE5F781EAF1C7EC76932CE99641E
  SHOW_PROGRESS
)

