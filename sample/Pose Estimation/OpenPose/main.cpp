#include <iostream>
#include <string>
#include <vector>
#include <limits>
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

    // Read OpenPose COCO (18 parts)
    const std::string model  = "../pose_iter_440000.caffemodel";
    const std::string config = "../openpose_pose_coco.prototxt";
    /*
    // Read OpenPose MPI (16 parts)
    const std::string model  = "../pose_iter_160000.caffemodel";
    const std::string config = "../openpose_pose_mpi_faster_4_stages.prototxt";
    */
    /*
    // Read OpenPose Hand
    const std::string model  = "../pose_iter_102000.caffemodel";
    const std::string config = "../pose_deploy.prototxt";
    */
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
        // OpenPose ( Scale : 1 / 255, Size : 368 x 368, Mean Subtraction : ( 0, 0, 0 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 368, 368 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f / 255.f, cv::Size( 368, 368 ), cv::Scalar( 0, 0, 0 ), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Draw Parts Position
        // NOTE: This is a visualize pose estimation with assumption single human (hand) scene.
        for( cv::Mat& detection : detections ){
            // Retrieve Num of Parts and Heat Map Size
            const int32_t nparts = detection.size[1];
            const int32_t rows   = detection.size[2];
            const int32_t cols   = detection.size[3];

            // Find Parts Position
            const cv::Point2f scale( static_cast<float>( frame.cols ) / cols, static_cast<float>( frame.rows ) / rows );
            for( int32_t i = 0; i < nparts; i++ ){
                // Create Heat Map of Parts
                cv::Mat heat_map( rows, cols, CV_32F, detection.ptr<float>( 0, i ) );

                // Most Reliable Parts Position from Heat Map
                cv::Point point;
                double confidence = std::numeric_limits<double>::lowest();
                cv::minMaxLoc( heat_map, 0, &confidence, 0, &point );

                // Check Confidence
                constexpr double threshold = 0.5;
                if( threshold > confidence ){
                    continue;
                }

                // Draw Parts Position
                const cv::Point position( static_cast<int32_t>( point.x * scale.x ), static_cast<int32_t>( point.y * scale.y ) );
                constexpr int32_t radius = 3;
                cv::circle( frame, position, radius, cv::Scalar( 0, 0, 255 ), -1 );
            }
        }

        // Show Image
        cv::imshow( "OpenPose", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
