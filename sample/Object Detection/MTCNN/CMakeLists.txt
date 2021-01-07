cmake_minimum_required( VERSION 3.6 )

# Require C++11 (or later)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( Sample )
file( GLOB MTCNN_HEADER "mtcnn/*.h")
file( GLOB MTCNN_SOURCE "mtcnn/*.cpp")
add_executable( MTCNN main.cpp ${MTCNN_HEADER} ${MTCNN_SOURCE} )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "MTCNN" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
    # Additional Include Directories
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    # Additional Dependencies
    target_link_libraries( MTCNN ${OpenCV_LIBS} )
endif()

# Download Model
set( MODEL https://github.com/DuinoDu/mtcnn/raw/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det1.caffemodel )
file( DOWNLOAD 
    "${MODEL}"
    "${CMAKE_CURRENT_LIST_DIR}/det1.caffemodel"
    EXPECTED_HASH SHA256=d6085e7f48ba7e6b6f1b58964595f6bce5b97bcc4866751f7b4bdc98f920c096
    SHOW_PROGRESS
)

set( MODEL https://github.com/DuinoDu/mtcnn/raw/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det2.caffemodel )
file( DOWNLOAD 
    "${MODEL}"
    "${CMAKE_CURRENT_LIST_DIR}/det2.caffemodel"
    EXPECTED_HASH SHA256=39b20f7a57bb8176cc9466cea4dfd52da6a6f876de60c7ab222a309f2d0ca08c
    SHOW_PROGRESS
)

set( MODEL https://github.com/DuinoDu/mtcnn/raw/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det3.caffemodel )
file( DOWNLOAD 
    "${MODEL}"
    "${CMAKE_CURRENT_LIST_DIR}/det3.caffemodel"
    EXPECTED_HASH SHA256=9d6098829a4d6d318f37cec42142465637fafe4c673f2e93b69495bf7ca23d2d
    SHOW_PROGRESS
)

# Download Config
set( CONFIG https://raw.githubusercontent.com/DuinoDu/mtcnn/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det1.prototxt )
file( DOWNLOAD 
    "${CONFIG}"
    "${CMAKE_CURRENT_LIST_DIR}/det1.prototxt"
    EXPECTED_HASH SHA256=adc1756d8515d3ca3a6a186c0fadab66fcae04bd8d3c6388e2fe8797a626dde4
    SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/DuinoDu/mtcnn/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det2.prototxt )
file( DOWNLOAD 
    "${CONFIG}"
    "${CMAKE_CURRENT_LIST_DIR}/det2.prototxt"
    EXPECTED_HASH SHA256=077686e89e606354f425366afdb2018777d93c6450b50e2c12301f8a97f6bb47
    SHOW_PROGRESS
)

set( CONFIG https://raw.githubusercontent.com/DuinoDu/mtcnn/db5bd8f02023f8d37913140fd2bf2749c2dbf266/model/det3.prototxt )
file( DOWNLOAD 
    "${CONFIG}"
    "${CMAKE_CURRENT_LIST_DIR}/det3.prototxt"
    EXPECTED_HASH SHA256=a8385a3aad241acf5902b79466f9a359ea9f03a3b6dcbe1e1efa050908cf7d04
    SHOW_PROGRESS
)
