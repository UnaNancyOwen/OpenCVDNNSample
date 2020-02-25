#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "util.h"

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Read Class Name List and Color Table
    const std::string list = "../voc.names";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read FCN8s
    const std::string model  = "../fcn8s-heavy-pascal.caffemodel";
    const std::string config = "../fcn8s-heavy-pascal.prototxt";
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    /*
        Please see list of supported combinations backend/target.
        https://docs.opencv.org/4.2.0/db/d30/classcv_1_1dnn_1_1Net.html#a9dddbefbc7f3defbe3eeb5dc3d3483f4
    */

    // Set Preferable Backend
    net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_OPENCL );

    while( true ){
        // Read Frame
        cv::Mat frame;
        capture >> frame;
        if( frame.empty() ){
            cv::waitKey( 0 );
            break;
        }
        if( frame.channels() == 4 ){
            cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // FCN8s ( Scale : 1.f, Size : 500 x 500, Mean Subtraction : ( 0, 0, 0 ), Channels Order : BGR )
        cv::Mat blob = cv::dnn::blobFromImage( frame, 1.f, cv::Size( 500, 500 ), cv::Scalar( 0, 0, 0 ), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Draw Detection Output
        for( cv::Mat& detection : detections ){
            // 500 x 500 x C
            const int32_t class_ids = detection.size[1];
            const int32_t rows      = detection.size[2];
            const int32_t cols      = detection.size[3];

            cv::Mat max_class_ids = cv::Mat::zeros( rows, cols, CV_8UC1 );
            cv::Mat max_scores    = cv::Mat( rows, cols, CV_32FC1, detection.data ); // Temp

            // Transform Matrix to CxWxH from 1xCxWxH
            std::vector<int32_t> size = { class_ids, rows, cols };
            cv::Mat mat( static_cast<int32_t>( size.size() ), &size[0], CV_32F, detection.ptr<float>() );

            // Retrieve Class_Ids Map
            for( int32_t class_id = 1; class_id < class_ids; class_id++ ){
                for( int32_t y = 0; y < rows; y++ ){
                    for( int32_t x = 0; x < cols; x++ ){
                        const float score = mat.at<float>( class_id, y, x );
                        const float max_score = max_scores.at<float>( y, x );
                        if( max_score > score ){
                            continue;
                        }

                        max_scores.at<float>( y, x ) = score;
                        max_class_ids.at<uint8_t>( y, x ) = class_id;
                    }
                }
            }

            // Retrieve Segmentation Map
            cv::Mat segmentation = cv::Mat( rows, cols, CV_8UC3 );
            for( int32_t y = 0; y < rows; y++ ){
                for( int32_t x = 0; x < cols; x++ ){
                    const uint8_t class_id = max_class_ids.at<uint8_t>( y, x );
                    const cv::Scalar color = colors[class_id];
                    segmentation.at<cv::Vec3b>( y, x ) = cv::Vec3b( color[0], color[1], color[2] );
                }
            }

            // Alpha Blending Image and Segmentation Map for Visualize
            constexpr double alpha = 0.1;
            constexpr double beta  = 1.0 - alpha;
            cv::addWeighted( resize_frame, alpha, segmentation, beta, 0.0, resize_frame );

            // Resize Original Image
            cv::resize( resize_frame, frame, frame.size() );
        }

        // Show Image
        cv::imshow( "Segmentation", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
