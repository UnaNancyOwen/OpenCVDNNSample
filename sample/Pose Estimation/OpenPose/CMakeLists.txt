cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( OpenPose util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "OpenPose" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( OpenPose ${OpenCV_LIBS} )
endif()

# Download License
set( LICENSE https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/LICENSE )
file( DOWNLOAD 
  "${LICENSE}"
  "${CMAKE_CURRENT_LIST_DIR}/LICENSE"
  EXPECTED_HASH SHA256=BE0B0FC17147B2C5F85FDFDE41EA8CD08DACEF8EA846082A3F727672104718A6
  SHOW_PROGRESS
)
message( WARNING "The OpenPose license is only allowed to use for non-commercial research by academic or non-profit organization. Please read OpenPose license. (LICENSE)" )

# Download Model
set( MODEL http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/pose_iter_440000.caffemodel"
  EXPECTED_HASH SHA256=b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549
  SHOW_PROGRESS
)

set( MODEL http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/pose_iter_160000.caffemodel"
  EXPECTED_HASH SHA256=2EA61D8549A3386972BA9A310468F961E7157D94C5EE2529A51B3901BB82EE21
  SHOW_PROGRESS
)

set( MODEL http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/pose_iter_102000.caffemodel"
  EXPECTED_HASH SHA256=F313AA5EF031AC91606E1A0383DAA14BE241A6E1A91C1155283DD28503C79C2D
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/openpose_pose_coco.prototxt"
  EXPECTED_HASH SHA256=57944874090b19b9dd175beae7cbecf4f03e1023b6d63b64cff632166a627e23
  SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/openpose_pose_mpi_faster_4_stages.prototxt"
  EXPECTED_HASH SHA256=08EB22720A1AC373D78EDBE7657C6866367084FF1FE0CBFDBCFBCA590CD57572
  SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/pose_deploy.prototxt"
  EXPECTED_HASH SHA256=D722796D87F251AADF8EB1AF9DCA03B3732E8A1D1BB524961A29C8E5073A8251
  SHOW_PROGRESS
)