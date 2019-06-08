cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( HeadPoseEstimation util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "HeadPoseEstimation" )

# Find Package
# OpenCV
message( WARNING "This sample requere OpenCV with Intel Inference Engine (OpenVINO)." )
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( HeadPoseEstimation ${OpenCV_LIBS} )
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
set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001.bin"
  EXPECTED_HASH SHA256=E4047E643BD39D97288DC5D22ABE8EAD850E05AC1BC44605443BBB2ABFC2E246
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001-fp16.bin"
  EXPECTED_HASH SHA256=535A6AF806999E22CCA5E4071E55841A694A6B60370A6F8FB3B9D0CDA5F81C41
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/INT8/head-pose-estimation-adas-0001.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001-int8.bin"
  EXPECTED_HASH SHA256=ABC0FB7F08E782301C4FFCCF9FC5CE626E608A1E7F9C179BAF6513519FA38416
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001.xml"
  EXPECTED_HASH SHA256=5CC24273DB0AEDBF44EE5671135228CEA7187FB5BEC81BE216AF74AAC3D5E8D9
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001-fp16.xml"
  EXPECTED_HASH SHA256=D570C7B43A20428AC41915C6B915860D6A141D0057B0091C22FE014B79EE5EA7
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/head-pose-estimation-adas-0001/INT8/head-pose-estimation-adas-0001.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/head-pose-estimation-adas-0001-int8.xml"
  EXPECTED_HASH SHA256=3D820D5AEDE971AEF3B760328BFE1EDD96D395B272622CEA816C573003AE2B89
  SHOW_PROGRESS
)

# Download Other
set( CASCADE https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml )
file( DOWNLOAD 
  "${CASCADE}"
  "${CMAKE_CURRENT_LIST_DIR}/haarcascade_frontalface_default.xml"
  EXPECTED_HASH SHA256=0F7D4527844EB514D4A4948E822DA90FBB16A34A0BBBBC6ADC6498747A5AAFB0
  SHOW_PROGRESS
)