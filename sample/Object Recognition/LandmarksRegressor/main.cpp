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

    // Read Landmarks Regressor
    const std::string model  = "../landmarks-regression-retail-0009.bin";
    const std::string config = "../landmarks-regression-retail-0009.xml";
    /*
    // Read Landmarks Regressor FP16
    const std::string model  = "../landmarks-regression-retail-0009-fp16.bin";
    const std::string config = "../landmarks-regression-retail-0009-fp16.xml";
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
            // Landmarks Detector ( Scale : 1.f, Size : 48 x 48, Mean Subtraction : ( 0, 0, 0 ), Channels Order : BGR )
            // The input image for this model is a tightly cropped face. Please detect the face in some way before using this model.
            cv::Mat blob = cv::dnn::blobFromImage( frame, 1.f, cv::Size( 48, 48 ), cv::Scalar(), false, false );

            // Set Input Blob
            net.setInput( blob );

            // Run Forward Network
            std::vector<cv::Mat> detections;
            net.forward( detections, getOutputsNames( net ) );

            // Draw Detection Output
            for( cv::Mat& detection : detections ){
                // Transform Matrix to 1x10 from 1x10x32x32
                const std::vector<int32_t> size = { detection.size[0], detection.size[1] };
                cv::Mat mat( static_cast<int32_t>( size.size() ), &size[0], CV_32F, detection.ptr<float>() );
                for( int32_t i = 0; i < mat.rows; i++ ){
                    // Retrieve Object
                    //   object0 [x_0][y_0][x_1][y_1][x_2][y_2][x_3][y_3][x_4][y_4]
                    const cv::Mat object = mat.row( i );

                    // Left Eye, Right Eye, Nose Tip, Left Mouth-Corner, Right Mouth-Corner
                    for( int32_t j = 0; j < object.cols; j += 2 ){
                        // Retrieve Object Position
                        const int32_t x = static_cast<int32_t>( ( object.at<float>( j + 0 ) * face_frame.cols ) );
                        const int32_t y = static_cast<int32_t>( ( object.at<float>( j + 1 ) * face_frame.rows ) );

                        // Draw Object Point
                        const cv::Point point = cv::Point( x, y );
                        const cv::Scalar color = cv::Scalar( 255, 0, 0 );
                        constexpr int32_t radius = 5;
                        cv::circle( face_frame, point, radius, color, -1 );
                    }
                }
            }

            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            constexpr int32_t thickness = 2;
            cv::rectangle( frame, faces[i], color, thickness );
        }

        // Show Image
        cv::imshow( "Face Landmarks Regressor", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
