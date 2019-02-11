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
    const std::string list = "../voc.names";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read R-FCN
    const std::string model  = "../resnet50_rfcn_final.caffemodel";
    const std::string config = "../rfcn_pascal_voc_resnet50.prototxt";
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_OPENCL );

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
        // R-FCN ( Scale : 1.f, Size : 800 x 600, Mean Subtraction : ( 102.9801, 115.9465, 122.7717 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 800, 600 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 800, 600 ), cv::Scalar( 102.9801, 115.9465, 122.7717 ), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Set Input Im_Info for R-FCN, Faster-RCNN
        cv::Mat im_info = ( cv::Mat_<float>( 1, 3 ) << 800, 600, 1.6f );
        net.setInput( im_info, "im_info" );

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

                // Retrieve Class Index
                const int32_t class_id = static_cast<int32_t>( object.at<float>( 1 ) );

                // Retrieve and Check Confidence
                const float confidence = object.at<float>( 2 );
                constexpr float threshold = 0.2;
                if( threshold > confidence ){
                    continue;
                }

                // Retrieve Object Position
                // R-FCN output is input image size. It is necessary adjust to original image size for visualize.
                const int32_t x      = static_cast<int32_t>( ( object.at<float>( 3 ) * frame.cols ) / 800 );
                const int32_t y      = static_cast<int32_t>( ( object.at<float>( 4 ) * frame.rows ) / 600 );
                const int32_t width  = static_cast<int32_t>( ( object.at<float>( 5 ) * frame.cols ) / 800 ) - x;
                const int32_t height = static_cast<int32_t>( ( object.at<float>( 6 ) * frame.rows ) / 600 ) - y;
                const cv::Rect rectangle = cv::Rect( x, y, width, height );

                // Draw Rectangle
                const cv::Scalar color = colors[class_id];
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
