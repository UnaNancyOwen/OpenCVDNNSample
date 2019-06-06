#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include <inference_engine.hpp>
#include <ext_list.hpp>

std::unique_ptr<ObjectDetector>
CreateDetector( const std::string& model, const std::string& config )
{
    if( model.empty() || config.empty() ){
        std::exit( EXIT_FAILURE );
    }

    // Create Detector Config
    DetectorConfig detector_config( config, model );

    // Create Inference Plugin
    const std::string device = "CPU";
    InferenceEngine::InferencePlugin inference_plugin = InferenceEngine::PluginDispatcher().getPluginByDevice( device );
    inference_plugin.AddExtension( std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>() );
    inference_plugin.SetConfig( { { InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES } } );

    // Create Detector
    std::unique_ptr<ObjectDetector> detector = std::make_unique<ObjectDetector>( detector_config, inference_plugin );

    return detector;
}

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Read Person Detection
    const std::string detector_model  = "../person-detection-retail-0013.bin";
    const std::string detector_config = "../person-detection-retail-0013.xml";
    /*
    // Read Person Detection (FP16)
    const std::string detector_model  = "../person-detection-retail-0013-fp16.bin";
    const std::string detector_config = "../person-detection-retail-0013-fp16.xml";
    */

    std::unique_ptr<ObjectDetector> detector = CreateDetector( detector_model, detector_config );

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

        // Submit Frame to Detector
        // Frame Index is not necessary for detection process. It will be used in tracking process.
        static int32_t frame_index = 0;
        detector->submitFrame( frame, frame_index++ );

        // Wait for Detection Results
        detector->waitAndFetchResults();

        // Retrieve Detected Results
        TrackedObjects detections = detector->getResults();

        // Draw Detected Person
        for( const TrackedObject& detection : detections ){
            const cv::Rect rect = detection.rect;
            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            constexpr int32_t thickness = 3;
            cv::rectangle( frame, rect, color, thickness );
        }

        // Show Image
        cv::imshow( "Pedestrian Detector", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
