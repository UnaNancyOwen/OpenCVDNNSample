cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( VGG16-SSD util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "VGG16-SSD" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( VGG16-SSD ${OpenCV_LIBS} )
endif()

# Download Model and Config
set( MODEL_AND_CONFIG https://drive.google.com/uc?authuser=0&id=0BzKzrI_SkD1_TkFPTEQ1Z091SUE )
file( DOWNLOAD 
  "${MODEL_AND_CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/models_VGGNet_VOC0712Plus_SSD_300x300_ft.tar.gz"
  EXPECTED_HASH SHA256=E3EB9794A33EB77E6798F396DF453123C249A1DF554E42A3302EB4AA20A8F2EE
  SHOW_PROGRESS
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzfv "${CMAKE_CURRENT_LIST_DIR}/models_VGGNet_VOC0712Plus_SSD_300x300_ft.tar.gz"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel" "${CMAKE_CURRENT_LIST_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/models/VGGNet/VOC0712Plus/SSD_300x300_ft/deploy.prototxt" "${CMAKE_CURRENT_LIST_DIR}/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.prototxt"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

set( MODEL_AND_CONFIG https://drive.google.com/uc?authuser=0&id=0BzKzrI_SkD1_NVVNdWdYNEh1WTA )
file( DOWNLOAD 
  "${MODEL_AND_CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/models_VGGNet_VOC0712Plus_SSD_512x512_ft.tar.gz"
  EXPECTED_HASH SHA256=07C98B7867DD800F55DC46F939826A0FD72B7096B11E5D612C328850B62BF6DF
  SHOW_PROGRESS
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzfv "${CMAKE_CURRENT_LIST_DIR}/models_VGGNet_VOC0712Plus_SSD_512x512_ft.tar.gz"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/models/VGGNet/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel" "${CMAKE_CURRENT_LIST_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_LIST_DIR}/models/VGGNet/VOC0712Plus/SSD_512x512_ft/deploy.prototxt" "${CMAKE_CURRENT_LIST_DIR}/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.prototxt"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

