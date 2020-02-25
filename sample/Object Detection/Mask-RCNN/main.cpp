#include <iostream>
#include <string>
#include <vector>
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
    const std::string list = "../coco.names";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read Mask-RCNN
    const std::string model  = "../frozen_inference_graph.pb";
    const std::string config = "../mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
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
        // Mask-RCNN ( Scale : 1.f, Size : Free Size, Mean Subtraction : ( 0, 0, 0 ), Channels Order : RGB )
        cv::Mat blob = cv::dnn::blobFromImage( frame, 1.f, cv::Size( resize_frame.cols, resize_frame.rows ), cv::Scalar( 0, 0, 0 ), true, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        const std::vector<cv::String> output_blobs = { "detection_out_final", "detection_masks" };
        std::vector<cv::Mat> detections;
        net.forward( detections, output_blobs );

        // Draw Detection Output and Sigmoid
        cv::Mat detection = detections[0]; // detection_out_final
        cv::Mat mask      = detections[1]; // detection_masks

        // Transform Matrix to Nx7 from 1x1xNx7
        const std::vector<int32_t> size = { detection.size[2], detection.size[3] };
        cv::Mat mat( static_cast<int32_t>( size.size() ), &size[0], CV_32F, detection.ptr<float>() );
        for( int32_t i = 0; i < mat.rows; i++ ){
            const cv::Mat object = mat.row( i );

            // Retrieve Class Index
            const int32_t class_id = static_cast<int32_t>( object.at<float>( 1 ) );

            // Retrieve and Check Confidence
            const float confidence = object.at<float>( 2 );
            std::cout << confidence << std::endl;

            // Check Confidence
            constexpr float threshold = 0.5;
            if( threshold > confidence ){
                continue;
            }

            // Retrieve Object Position
            const int32_t x      = static_cast<int32_t>( object.at<float>( 3 ) * frame.cols );
            const int32_t y      = static_cast<int32_t>( object.at<float>( 4 ) * frame.rows );
            const int32_t width  = static_cast<int32_t>( object.at<float>( 5 ) * frame.cols ) - x;
            const int32_t height = static_cast<int32_t>( object.at<float>( 6 ) * frame.rows ) - y;
            const cv::Rect rectangle = cv::Rect( x, y, width, height );

            // Retrieve Object Mask
            const std::vector<int32_t> size = { mask.size[2], mask.size[3] };
            cv::Mat object_mask( static_cast<int32_t>( size.size() ), &size[0], CV_32F, mask.ptr<float>( i, class_id ) );

            // ReSize Mask to Object Size from 15x15
            cv::resize( object_mask, object_mask, cv::Size( rectangle.width, rectangle.height ) );

            // Threshold Mask
            constexpr double mask_threshold = 0.3;
            cv::threshold( object_mask, object_mask, mask_threshold, 1.0, cv::THRESH_TOZERO );
            object_mask.convertTo( object_mask, CV_8U, 255.0 );

            // Draw Mask
            const cv::Scalar color = colors[class_id];
            constexpr double alpha = 0.6;
            constexpr double beta  = 1.0 - alpha;
            const cv::Mat mask_color = alpha * frame( rectangle ) + beta * color;
            mask_color.copyTo( frame( rectangle ), object_mask );

            // Draw Rectangle
            constexpr int32_t thickness = 3;
            cv::rectangle( frame, rectangle, color, thickness );
        }

        // Show Image
        cv::imshow( "Object Detection", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
