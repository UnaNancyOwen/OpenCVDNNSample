cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( LandmarksRegressor util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "LandmarksRegressor" )

# Find Package
# OpenCV
message( WARNING "This sample requere OpenCV with Intel Inference Engine (OpenVINO)." )
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( LandmarksRegressor ${OpenCV_LIBS} )
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
set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/landmarks-regression-retail-0009.bin"
  EXPECTED_HASH SHA256=46795837D35E8199B7C5B57E1F76297827BF516A150C0D5643197D8C325F1DBC
  SHOW_PROGRESS
)

set( MODEL https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/landmarks-regression-retail-0009-fp16.bin"
  EXPECTED_HASH SHA256=5D74C26CBB836B3DE358AB05D4CBD92C4EB713DC74484CFF9DE82B2DEB3D8527
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/landmarks-regression-retail-0009.xml"
  EXPECTED_HASH SHA256=DB495316E343DC5CDB3EDAF6CA5853C4A478C96772EB6DA042CCDDDEEEFD0936
  SHOW_PROGRESS
)

set( CONFIG https://download.01.org/opencv/2019/open_model_zoo/R1/20190404_140900_models_bin/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/landmarks-regression-retail-0009-fp16.xml"
  EXPECTED_HASH SHA256=BCF274A704031D24AF7FDD7F868F052E346CF96029B6B6E85CDC168BEDDBE401
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
