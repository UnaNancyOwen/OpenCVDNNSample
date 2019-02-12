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

    // Read Face Detector
    const std::string model  = "../opencv_face_detector.caffemodel";
    const std::string config = "../opencv_face_detector.prototxt";
    /*
    // Read Face Detector FP16
    const std::string model  = "../opencv_face_detector_fp16.caffemodel";
    const std::string config = "../opencv_face_detector.prototxt";
    */
    /*
    // Read Face Detector UINT8
    const std::string model  = "../opencv_face_detector_uint8.pb";
    const std::string config = "../opencv_face_detector.pbtxt";
    */
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    // Set Preferable Target
    // Face Detector is faster CPU than GPU.
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
        // Face Detector ( Scale : 1.f, Size : 300 x 300, Mean Subtraction : ( 104, 177, 123 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 300, 300 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 300, 300 ), cv::Scalar( 104, 177, 123 ), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Draw Detection Output
        for( cv::Mat& detection : detections ){
            // Transform Matrix to Nx7 from 1x1xNx7
            const std::vector<int32_t> size = { detection.size[2], detection.size[3] };
            cv::Mat mat( static_cast<int32_t>( size.size() ), &size[0], CV_32F, detection.ptr<float>() );
            for( int32_t i = 0; i < mat.rows; i++ ){
                // Retrieve Object
                //   object0 [batch_id][class_id][confidence][left][top][right][bottom]
                //   object1 [batch_id][class_id][confidence][left][top][right][bottom]
                //   object2 [batch_id][class_id][confidence][left][top][right][bottom]
                //   ...
                const cv::Mat object = mat.row( i );

                // Retrieve and Check Confidence
                const float confidence = object.at<float>( 2 );
                constexpr float threshold = 0.2;
                if( threshold > confidence ){
                    continue;
                }

                // Retrieve Object Position
                const int32_t x      = static_cast<int32_t>( ( object.at<float>( 3 ) * frame.cols ) );
                const int32_t y      = static_cast<int32_t>( ( object.at<float>( 4 ) * frame.rows ) );
                const int32_t width  = static_cast<int32_t>( ( object.at<float>( 5 ) * frame.cols ) ) - x;
                const int32_t height = static_cast<int32_t>( ( object.at<float>( 6 ) * frame.rows ) ) - y;
                const cv::Rect rectangle = cv::Rect( x, y, width, height );

                // Draw Rectangle
                const cv::Scalar color = cv::Scalar( 0, 0, 255 );
                constexpr int32_t thickness = 3;
                cv::rectangle( frame, rectangle, color, thickness );
            }
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
