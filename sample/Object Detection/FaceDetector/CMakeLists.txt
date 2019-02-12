cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
add_executable( FaceDetector util.h main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "FaceDetector" )

# Find Package
# OpenCV
set( OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "Path to OpenCV config directory." )
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( FaceDetector ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/opencv_face_detector.caffemodel"
  EXPECTED_HASH SHA256=2A56A11A57A4A295956B0660B4A3D76BBDCA2206C4961CEA8EFE7D95C7CB2F2D
  SHOW_PROGRESS
)

set( MODEL https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/opencv_face_detector_fp16.caffemodel"
  EXPECTED_HASH SHA256=510FFD2471BD81E3FCC88A5BEB4EAE4FB445CCF8333EBC54E7302B83F4158A76
  SHOW_PROGRESS
)

set( MODEL https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb )
file( DOWNLOAD 
  "${MODEL}"
  "${CMAKE_CURRENT_LIST_DIR}/opencv_face_detector_uint8.pb"
  EXPECTED_HASH SHA256=5C71D752EF2CBF2F457AC82FDD580FCB2522FD04C5EFDAED18EB6D9E2843FBED
  SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/opencv_face_detector.prototxt"
  EXPECTED_HASH SHA256=F62621CAC923D6F37BD669298C428BB7EE72233B5F8C3389BB893E35EBBCF795
  SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt )
file( DOWNLOAD 
  "${CONFIG}"
  "${CMAKE_CURRENT_LIST_DIR}/opencv_face_detector.pbtxt"
  EXPECTED_HASH SHA256=5397808362FC6FB7C4044484848E297B22449D913A267FA7F8BD9D86DCE29144
  SHOW_PROGRESS
)
