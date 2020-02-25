#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
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
    const std::string list = "../imagenet.names";
    const std::vector<std::string> classes = readClassNameList( list );

    // Read GoogLeNet
    const std::string model  = "../bvlc_googlenet.caffemodel";
    const std::string config = "../deploy.prototxt";
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
        // GoogLeNet ( Scale : 1.f, Size : 224 x 224, Mean Subtraction : ( 104, 117, 123 ), Channels Order : BGR )
        cv::Mat blob = cv::dnn::blobFromImage( frame, 1.f, cv::Size( 224, 224 ), cv::Scalar( 104, 117, 123 ), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Draw Softmax
        for( cv::Mat& detection : detections ){
            // Softmax Cx1
            //   [class0 confidence][class1 confidence][class2 confidence]...

            // Retrieve Top N Confidence Indices using ArgSort
            std::vector<int32_t> indices( detection.size[1] );
            std::iota( indices.begin(), indices.end(), 0 );
            std::sort( indices.begin(), indices.end(), [&detection]( const int32_t index_1, const int32_t index_2 ) {
                return detection.at<float>( index_1 ) > detection.at<float>( index_2 );
            } );

            // Draw Higher Confidences Class
            for( int32_t i = 0; i < 5; i++ ){
                // Retrieve Index and Confidence
                const int32_t index = indices[i];
                const double confidence = detection.at<float>( index );

                // Check Confidence
                constexpr float threshold = 0.05;
                if( threshold > confidence ){
                    break;
                }

                // Draw Result ( Rank : Class Name (Confidence) )
                std::ostringstream oss;
                oss << i + 1 << " : "<< classes[index] << " (" << std::fixed << std::setprecision( 5 ) << confidence << ")";
                constexpr int32_t offset = 50;
                cv::putText( frame, oss.str(), cv::Point( 50, 50 + ( offset * i ) ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar( 0, 0, 255 ), 2 );
                //std::cout << oss.str() << std::endl;
            }
        }

        // Show Image
        cv::imshow( "Classification", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
