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
    if( !capture.isOpened() )
    {
        return -1;
    }

    // Read Network
    const std::string model = "../model-f46da743.onnx";
    cv::dnn::Net net = cv::dnn::readNet( model );
    if( net.empty() )
    {
        return -1;
    }

    // Set Preferable Backend and Target
    net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    while( true ){
        // Read Frame
        cv::Mat input;
        capture >> input;
        if( input.empty() ){
            cv::waitKey( 0 );
            break;
        }
        if( input.channels() == 4 ){
            cv::cvtColor( input, input, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // MiDaS ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        cv::Mat blob = cv::dnn::blobFromImage( input, 1 / 255.f, cv::Size( 384, 384 ), cv::Scalar( 123.675, 116.28, 103.53 ), true, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        cv::Mat output = net.forward( getOutputsNames( net )[0] );

        // Convert Size to 384x384 from 1x384x384
        const std::vector<int32_t> size = { output.size[1], output.size[2] };
        output = cv::Mat( static_cast< int32_t >( size.size() ), &size[0], CV_32F, output.ptr<float>() );

        // Resize Output Image to Input Image Size
        cv::resize( output, output, input.size() );

        // Visualize Output Image
        // 1. Normalize ( 0.0 - 1.0 )
        // 2. Scaling ( 0 - 255 )
        double min, max;
        cv::minMaxLoc( output, &min, &max );
        const double range = max - min;
        output.convertTo( output, CV_32F, 1.0 / range, -( min / range ) );
        output.convertTo( output, CV_8U, 255.0 );

        // Show Image
        cv::imshow( "input", input );
        cv::imshow( "output", output );

        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}