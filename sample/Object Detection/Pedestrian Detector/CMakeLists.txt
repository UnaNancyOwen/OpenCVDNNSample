cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
# common.hpp, ocv_common.hpp, cnn.hpp/cpp, core.hpp/cpp, and detector.hpp/cpp are distributed in Open Model Zoo under the Apache License v2.0.
# https://github.com/opencv/open_model_zoo/tree/master/demos/pedestrian_tracker_demo
project( Sample )
add_executable( PedestrianDetector common.hpp ocv_common.hpp cnn.hpp cnn.cpp core.hpp core.cpp detector.hpp detector.cpp main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "PedestrianDetector" )

# Find Package
message( WARNING "This sample requere OpenCV, and Intel Inference Engine (OpenVINO)." )
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

# Intel Inference Engine (OpenVINO)
set( InferenceEngine_DIR "C:/Program Files (x86)/IntelSWTools/openvino/inference_engine/share" CACHE PATH "Path to Inference Engine config directory." )
find_package( InferenceEngine REQUIRED )

if( OpenCV_FOUND AND InferenceEngine_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  include_directories( ${InferenceEngine_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( PedestrianDetector ${OpenCV_LIBS} )
  target_link_libraries( PedestrianDetector ${InferenceEngine_LIBRARIES} )
  target_link_libraries( PedestrianDetector IE::ie_cpu_extension )

  # Add Post Build Event
  add_custom_command(
    TARGET PedestrianDetector 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:ie_cpu_extension>
        $<TARGET_FILE_DIR:PedestrianDetector>/$<TARGET_FILE_NAME:ie_cpu_extension>
  )
endif()

# Download License
set( LICENSE https://raw.githubusercontent.com/opencv/open_model_zoo/master/LICENSE )
file( DOWNLOAD 
  "${LICENSE}"
  "${CMAKE_CURRENT_LIST_DIR}/LICENSE"
  EXPECTED_HASH SHA256=C71D239DF91726FC519C6EB72D318EC65820627232B2F796219E87DCF35D0AB4
  SHOW_PROGRESS
)

# Download Model
set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/person-detection-retail-0013/FP32/person-detection-retail-0013.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/person-detection-retail-0013.bin"
  EXPECTED_HASH SHA256=3D2C6572E6B6F845A1F2C23C32DE958333F40B4EE6C3448201F00450A5B8C8D3
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/person-detection-retail-0013/FP16/person-detection-retail-0013.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/person-detection-retail-0013-fp16.bin"
  EXPECTED_HASH SHA256=5B65DDC561B86E48420F7942CD5762DEEBC8691D1D11DC7567842A8B4548BF93
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/person-detection-retail-0013/FP32/person-detection-retail-0013.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/person-detection-retail-0013.xml"
  EXPECTED_HASH SHA256=0F6FC8657B45BA7216CADA7546BA8BDD489E357578EE017386DAD555B34E1ED6
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/person-detection-retail-0013/FP16/person-detection-retail-0013.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/person-detection-retail-0013-fp16.xml"
  EXPECTED_HASH SHA256=F0A9F7D819DD79A4E03EE77D530AD822035817121ABEFF547DCACC736F0CE206
  SHOW_PROGRESS
)
