#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "util.h"

// Gender Estimation
std::string estimate_gender( cv::dnn::Net& net, cv::Mat& frame )
{
    if( net.empty() ){
        return "";
    }

    if( frame.empty() ){
        return "";
    }

    // Create Blob from Input Image
    // Gender-Net ( Scale : 1.f, Size : 227x227, Mean Subtraction : ( 78.4263377603, 87.7689143744, 114.895847746 ), Channels Order : BGR )
    cv::Mat resize_frame;
    cv::resize( frame, resize_frame, cv::Size( 227, 227 ) );
    cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 227, 227 ), cv::Scalar( 78.4263377603, 87.7689143744, 114.895847746 ), false, false );

    // Set Input Blob
    net.setInput( blob );

    // Run Forward Network
    std::vector<float> confidences = net.forward();

    // Retrive Gender
    std::vector<std::string> gender_list = { "Male", "Female" };
    int32_t index = std::distance( confidences.begin(), std::max_element( confidences.begin(), confidences.end() ) );

    return gender_list[index];
}

// Age Estimation
std::string estimate_age( cv::dnn::Net& net, cv::Mat& frame )
{
    if( net.empty() ){
        return "";
    }

    if( frame.empty() ){
        return "";
    }

    // Create Blob from Input Image
    // Age-Net ( Scale : 1.f, Size : 227x227, Mean Subtraction : ( 78.4263377603, 87.7689143744, 114.895847746 ), Channels Order : BGR )
    cv::Mat resize_frame;
    cv::resize( frame, resize_frame, cv::Size( 227, 227 ) );
    cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 227, 227 ), cv::Scalar( 78.4263377603, 87.7689143744, 114.895847746 ), false, false );

    // Set Input Blob
    net.setInput( blob );

    // Run Forward Network
    std::vector<float> confidences = net.forward();

    // Retrieve Age
    std::vector<std::string> age_list = { "0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-" };
    int32_t index = std::distance( confidences.begin(), std::max_element( confidences.begin(), confidences.end() ) );

    return age_list[index];
}

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Read Face Detector
    const std::string config = "../opencv_face_detector.prototxt";
    const std::string model  = "../opencv_face_detector.caffemodel";
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    /*
    const std::string config = "../opencv_face_detector.prototxt";
    const std::string model  = "../opencv_face_detector_fp16.caffemodel";
    cv::dnn::Net net = cv::dnn::readNetFromCaffe( config, model );
    if( net.empty() ){
        return -1;
    }
    */

    /*
    const std::string config = "../opencv_face_detector.pbtxt";
    const std::string model  = "../opencv_face_detector_uint8.pb";
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow( model, config );
    if( net.empty() ){
        return -1;
    }
    */

    // Read Gender-Net
    const std::string gender_config = "../deploy_gender.prototxt";
    const std::string gender_model = "../gender_net.caffemodel";
    cv::dnn::Net gender_net = cv::dnn::readNet( gender_model, gender_config );
    if( gender_net.empty() ){
        return -1;
    }

    // Read Age-Net
    const std::string age_config = "../deploy_age.prototxt";
    const std::string age_model = "../age_net.caffemodel";
    cv::dnn::Net age_net = cv::dnn::readNet( age_model, age_config );
    if( age_net.empty() ){
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
            cv::waitKey( 0 );
            break;
        }
        if( frame.channels() == 4 ){
            cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // Face Detector ( Scale : 1.f, Size : 300x300, Mean Subtraction : ( 104, 177, 123 ), Channels Order : BGR )
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
            cv::Mat mat( detection.size[2], detection.size[3], CV_32F, detection.ptr<float>() );
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

                // Estimate Gender and Age
                constexpr int32_t padding = 20;
                const cv::Rect roi = cv::Rect( x - padding, y - padding, width + ( padding * 2 ), height + ( padding * 2 ) );
                if( ( roi & cv::Rect( 0, 0, frame.cols, frame.rows ) ) == roi ){
                    // Retrieve Estimated Gender and Age
                    const std::string gender = estimate_gender( gender_net, frame( roi ) );
                    const std::string age    = estimate_age( age_net, frame( roi ) );

                    // Draw  Estimated Gender and Age
                    const std::string attribute = gender + " (" + age + ")";
                    constexpr double font_scale = 0.7;
                    constexpr int32_t font_thickness = 2;
                    cv::putText( frame, attribute, cv::Point( rectangle.x, rectangle.y - 10 ), cv::FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness );
                }
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
