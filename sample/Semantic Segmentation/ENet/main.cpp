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
    const std::string list = "../cityscapes.names";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read ENet
    const std::string model = "../model-cityscapes.net";
    cv::dnn::Net net = cv::dnn::readNet( model );
    if( net.empty() ){
        return -1;
    }

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    while( true ){
        // Read Frame
        cv::Mat frame;
        capture >> frame;
        if( frame.empty() ){
            continue;
        }
        if( frame.channels() == 4 ){
            cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // ENet Cityscapes ( Scale : 1 / 255, Size : 1024 x 512, Mean Subtraction : ( 0, 0, 0 ), Channels Order : RGB )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 1024, 512 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f / 255.f, cv::Size( 1024, 512 ), cv::Scalar( 0, 0, 0 ), true, false );

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
