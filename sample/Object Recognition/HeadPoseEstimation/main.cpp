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

    // Read Head Pose Estimation
    const std::string model  = "../head-pose-estimation-adas-0001.bin";
    const std::string config = "../head-pose-estimation-adas-0001.xml";
    /*
    // Read Head Pose Estimation FP16
    const std::string model  = "../head-pose-estimation-adas-0001-fp16.bin";
    const std::string config = "../head-pose-estimation-adas-0001-fp16.xml";
    */
    /*
    // Read Head Pose Estimation INT8
    const std::string model  = "../head-pose-estimation-adas-0001-int8.bin";
    const std::string config = "../head-pose-estimation-adas-0001-int8.xml";
    */
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // Face Detection based Haar Cascades
    // https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html
    cv::CascadeClassifier cascade;
    cascade.load( "../haarcascade_frontalface_default.xml" );
    /*
    // Face Detection based Deep Learning
    // https://github.com/UnaNancyOwen/OpenCVDNNSample/tree/master/sample/Object%20Detection/FaceDetector
    */

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

        // Retrieve Tightly Cropped Face Image using Face Detection
        std::vector<cv::Rect> faces;
        cascade.detectMultiScale( frame, faces, 1.1, 3, 0, cv::Size( 100, 100 ) );

        for( int32_t i = 0; i < faces.size(); i++ ){
            cv::Mat face_frame = cv::Mat( frame, faces[i] );

            // Create Blob from Input Image
            // Landmarks Detector ( Scale : 1.f, Size : 60 x 60, Mean Subtraction : ( 120.0, 110.0, 104.0 ), Channels Order : BGR )
            // The input image for this model is a tightly cropped face. Please detect the face in some way before using this model.
            cv::Mat resize_frame;
            cv::resize( face_frame, resize_frame, cv::Size( 60, 60 ) );
            cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 60, 60 ), cv::Scalar( 120.0, 110.0, 104.0 ), false, false );

            // Set Input Blob
            net.setInput( blob );

            // Run Forward Network
            std::vector<cv::Mat> detections;
            net.forward( detections, getOutputsNames( net ) );

            // Retrieve Estimated Head Pose
            // http://web3d.jondgoodwin.com/recipes/pitch.png
            const float pitch = detections[0].at<float>( 0 ); // Pitch [-70, 70] (Degree)
            const float roll  = detections[1].at<float>( 0 ); // Roll  [-70, 70] (Degree)
            const float yow   = detections[2].at<float>( 0 ); // Yow   [-90, 90] (Degree)

            // Draw Estimated Head Pose
            const cv::String pose  = cv::format( "( %.1f, %.1f, %.1f )", pitch, roll, yow );
            const cv::Point point  = cv::Point( faces[i].x, faces[i].y - 10 );
            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            constexpr double scale = 0.5;
            constexpr int32_t thickness = 2;
            cv::putText( frame, pose, point, cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, scale, color, thickness );
            cv::rectangle( frame, faces[i], color, thickness );
        }

        // Show Image
        cv::imshow( "Head Pose Estimation", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
