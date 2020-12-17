cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
# common.hpp, human_pose.hpp/cpp, human_pose_estimator.hpp/cpp, and peak.hpp/cpp are distributed in Open Model Zoo under the Apache License v2.0.
# https://github.com/opencv/open_model_zoo/tree/master/demos/human_pose_estimation_demo
project( Sample )
add_executable( LightweightOpenPose common.hpp human_pose.hpp human_pose.cpp human_pose_estimator.hpp human_pose_estimator.cpp peak.hpp peak.cpp main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "LightweightOpenPose" )

# Find Package
message( WARNING "This sample requere OpenCV, and Intel Inference Engine (OpenVINO 2020.x or later)." )
# OpenCV
find_package( OpenCV REQUIRED )

# Intel Inference Engine (OpenVINO)
set( InferenceEngine_DIR "C:/Program Files (x86)/IntelSWTools/openvino/inference_engine/share" CACHE PATH "Path to Inference Engine config directory." )
find_package( InferenceEngine REQUIRED )

if( OpenCV_FOUND AND InferenceEngine_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  include_directories( ${InferenceEngine_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( LightweightOpenPose ${OpenCV_LIBS} )
  target_link_libraries( LightweightOpenPose ${InferenceEngine_LIBRARIES} )
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
set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001.bin"
  EXPECTED_HASH SHA256=9790DA47ABF072C509FF843695A7B9B96283729731BA4D711FB50201F376B2FC
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001-fp16.bin"
  EXPECTED_HASH SHA256=48C607A55B22B540C38B6ED295EBA211AAA6CD4ACD47776F0A1259829CB51196
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/INT8/human-pose-estimation-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001-int8.bin"
  EXPECTED_HASH SHA256=0D252F0FD98C598FB38EAA907FA8B3D15CF2D420D743649B4B6C5DBDBBE5F1AC
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001.xml"
  EXPECTED_HASH SHA256=93CB893D1FFC0BB7BB6825BD0CC64F626057DA1E16E22ED3835D753F2526E068
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001-fp16.xml"
  EXPECTED_HASH SHA256=5DBD78DEC8EA4A9283BB78640A3BA2C3878535286471ED7B80BCE30070684046
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/human-pose-estimation-0001/INT8/human-pose-estimation-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/human-pose-estimation-0001-int8.xml"
  EXPECTED_HASH SHA256=A77EB5AD3423D41B5F26ACF01C8549DB892CBB32F903342D6C1DC7640E1902C4
  SHOW_PROGRESS
)
