#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "cv_dnn_centerface.h"

enum LandMarks
{
    EYE_RIGHT   = 0,
    EYE_LEFT    = 1,
    NOSE        = 2,
    MOUTH_RIGHT = 3,
    MOUTH_LEFT  = 4,
    NUM_POINTS
};

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Create Detector
    const std::string model = "../centerface.onnx";
    Centerface centerface( model, 640, 480 );

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

        // Detect Faces
        std::vector<FaceInfo> faces;
        centerface.detect( frame, faces );

        // Draw Faces and Landmarks
        for( const FaceInfo& face : faces ){
            // Draw Bounding Box
            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            const cv::Rect rectangle = cv::Rect( face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1 );
            cv::rectangle( frame, rectangle, color );

            // Draw Landmarks
            for( int32_t i = 0; i < LandMarks::NUM_POINTS; i++ ){
                const cv::Point point = cv::Point( face.landmarks[2 * i], face.landmarks[2 * i + 1] );
                constexpr int32_t radius = 3;
                cv::circle( frame, point, radius, color );
            }
        }

        // Show Image
        cv::imshow( "Face Detection", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
