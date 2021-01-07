#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "mtcnn/detector.h"

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Set Network Config
    mtcnn::ProposalNetwork::Config proposal_config;
    proposal_config.caffemodel = "../det1.caffemodel";
    proposal_config.prototext  = "../det1.prototxt";
    proposal_config.score_threshold   = 0.6f;
    proposal_config.overlap_threshold = 0.7f;

    mtcnn::RefineNetwork::Config refine_config;
    refine_config.caffemodel = "../det2.caffemodel";
    refine_config.prototext  = "../det2.prototxt";
    refine_config.score_threshold   = 0.7f;
    refine_config.overlap_threshold = 0.7f;

    mtcnn::OutputNetwork::Config output_config;
    output_config.caffemodel = "../det3.caffemodel";
    output_config.prototext  = "../det3.prototxt";
    output_config.score_threshold   = 0.7f;
    output_config.overlap_threshold = 0.7f;

    // Create Detector
    mtcnn::Detector detector( proposal_config, refine_config, output_config );

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
        std::vector<mtcnn::Face> faces;
        faces = detector.detect( frame, 20.0f, 0.709f );

        // Draw Faces and Landmarks
        for( const mtcnn::Face& face : faces ){
            const cv::Rect rectangle = face.rectangle.getRect();
            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            cv::rectangle( frame, rectangle, color );

            for( int32_t i = 0; i < mtcnn::NUM_POINTS; i++ ){
                const cv::Point point = { static_cast<int32_t>( face.points[2 * i] ), static_cast<int32_t>( face.points[2 * i + 1] ) };
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
