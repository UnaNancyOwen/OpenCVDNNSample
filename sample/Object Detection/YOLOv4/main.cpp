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

    // Read Darknet
    const std::string model   = "../yolov4.weights";
    const std::string config  = "../yolov4.cfg";
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
        // YOLO v3 ( Scale : 1 / 255, Size : 416 x 416, Mean Subtraction : ( 0.0, 0.0, 0.0 ), Channels Order : RGB )
        cv::Mat blob = cv::dnn::blobFromImage( frame, 1 / 255.f, cv::Size( 416, 416 ), cv::Scalar(), true, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Draw Region
        std::vector<int32_t> class_ids; std::vector<float> confidences; std::vector<cv::Rect> rectangles;
        for( cv::Mat& detection : detections ){
            for( int32_t i = 0; i < detection.rows; i++ ){
                // Retrieve Region
                //   region0 [x_center][y_center][width][height][class0 confidence][class1 confidence][class2 confidence]...
                //   region1 [x_center][y_center][width][height][class0 confidence][class1 confidence][class2 confidence]...
                //   region2 [x_center][y_center][width][height][class0 confidence][class1 confidence][class2 confidence]...
                //   ...
                cv::Mat region = detection.row( i );

                // Retrieve Max Confidence and Class Index
                cv::Mat scores = region.colRange( 5, detection.cols );
                cv::Point class_id;
                double confidence;
                cv::minMaxLoc( scores, 0, &confidence, 0, &class_id );

                // Check Confidence
                constexpr float threshold = 0.2;
                if( threshold > confidence ){
                    continue;
                }

                // Retrieve Object Position
                const int32_t x_center = static_cast<int32_t>( region.at<float>( 0 ) * frame.cols );
                const int32_t y_center = static_cast<int32_t>( region.at<float>( 1 ) * frame.rows );
                const int32_t width    = static_cast<int32_t>( region.at<float>( 2 ) * frame.cols );
                const int32_t height   = static_cast<int32_t>( region.at<float>( 3 ) * frame.rows );
                const cv::Rect rectangle  = cv::Rect( x_center - ( width / 2 ), y_center - ( height / 2 ), width, height );

                // Add Class ID, Confidence, Rectangle
                class_ids.push_back( class_id.x );
                confidences.push_back( confidence );
                rectangles.push_back( rectangle );
            }
        }

        // Remove Overlap Rectangles using Non-Maximum Suppression
        constexpr float confidence_threshold = 0.5; // Confidence
        constexpr float nms_threshold = 0.5; // IoU (Intersection over Union)
        std::vector<int32_t> indices;
        cv::dnn::NMSBoxes( rectangles, confidences, confidence_threshold, nms_threshold, indices );

        // Draw Rectangle
        for( const int32_t& index : indices ){
            const cv::Rect rectangle = rectangles[index];
            const cv::Scalar color = colors[class_ids[index]];
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
