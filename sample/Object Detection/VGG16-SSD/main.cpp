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

    // Read VGG16-SSD 300
    const std::string model  = "../VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel";
    const std::string config = "../VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.prototxt";
    /*
    // Read VGG16-SSD 512
    const std::string model  = "../VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel";
    const std::string config = "../VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.prototxt";
    */
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    // Set Preferable Target
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
        // VGG16-SSD 300 ( Scale : 1.f, Size : 300 x 300, Mean Subtraction : ( 102.9801, 115.9465, 122.7717 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 300, 300 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 300, 300 ), cv::Scalar( 102.9801, 115.9465, 122.7717 ), false, false );

        /*
        // VGG16-SSD 512 ( Scale : 1.f, Size : 512 x 512 Mean Subtraction : ( 102.9801, 115.9465, 122.7717 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 512, 512 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 512, 512 ), cv::Scalar( 102.9801, 115.9465, 122.7717 ), false, false );
        */

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

                // Retrieve Class Index
                const int32_t class_id = static_cast< int32_t >( object.at<float>( 1 ) );

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
