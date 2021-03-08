cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Download Class
set( HEADER https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/prj-opencv-cpp/cpp/cv_dnn_centerface.h )
file( DOWNLOAD 
    "${HEADER}"
    "${CMAKE_CURRENT_LIST_DIR}/cv_dnn_centerface.h"
    EXPECTED_HASH SHA256=0cdb8c78ab3d4ccff320e7eb9333733e6e37a577322589a36e9ac050546cfeed
    SHOW_PROGRESS
)

set( SOURCE https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/prj-opencv-cpp/cpp/cv_dnn_centerface.cpp )
file( DOWNLOAD 
    "${SOURCE}"
    "${CMAKE_CURRENT_LIST_DIR}/cv_dnn_centerface.cpp"
    EXPECTED_HASH SHA256=2fa7eae4636318e4b93d2c8a83dc19aff53cb67e91faa13d57e53cb7bb5bda1e
    SHOW_PROGRESS
)

# Create Project
project( Sample )
add_executable( CenterFace cv_dnn_centerface.h cv_dnn_centerface.cpp main.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "CenterFace" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
    # Additional Include Directories
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    # Additional Dependencies
    target_link_libraries( CenterFace ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx )
file( DOWNLOAD 
    "${MODEL}"
    "${CMAKE_CURRENT_LIST_DIR}/centerface.onnx"
    EXPECTED_HASH SHA256=77e394b51108381b4c4f7b4baf1c64ca9f4aba73e5e803b2636419578913b5fe
    SHOW_PROGRESS
)
