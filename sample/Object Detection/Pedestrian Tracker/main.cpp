#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "tracker.hpp"
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

std::unique_ptr<PedestrianTracker>
CreateTracker( const std::string& model, const std::string& config )
{
    if( model.empty() || config.empty() ){
        std::exit( EXIT_FAILURE );
    }

    // Create Person Tracker
    TrackerParams params;
    std::unique_ptr<PedestrianTracker> tracker = std::make_unique<PedestrianTracker>( params );

    // Create Fast Descriptor and Fast Distance
    std::shared_ptr<IImageDescriptor> descriptor_fast = std::make_shared<ResizedImageDescriptor>( cv::Size( 16, 32 ), cv::InterpolationFlags::INTER_LINEAR );
    std::shared_ptr<IDescriptorDistance> distance_fast = std::make_shared<MatchTemplateDistance>();
    tracker->set_descriptor_fast( descriptor_fast );
    tracker->set_distance_fast( distance_fast );

    // Create Strong Descriptor and Strong Distance
    CnnConfig cnn_config( config, model );
    cnn_config.max_batch_size = 16;

    const std::string device = "CPU";
    InferenceEngine::InferencePlugin inference_plugin = InferenceEngine::PluginDispatcher().getPluginByDevice( device );
    inference_plugin.AddExtension( std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>() );
    inference_plugin.SetConfig( { { InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES } } );

    std::shared_ptr<IImageDescriptor> descriptor = std::make_shared<DescriptorIE>( cnn_config, inference_plugin );
    std::shared_ptr<IDescriptorDistance> distance = std::make_shared<CosDistance>( descriptor->size() );
    tracker->set_descriptor_strong( descriptor );
    tracker->set_distance_strong( distance );

    return tracker;
}

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    const double fps = capture.get( cv::CAP_PROP_FPS );
    std::vector<cv::Scalar> colors;
    for( int32_t i = 0; i < 100; i++ ){
        const cv::Scalar color( 
            static_cast<uchar>( 255.0 * rand() / RAND_MAX ),
            static_cast<uchar>( 255.0 * rand() / RAND_MAX ),
            static_cast<uchar>( 255.0 * rand() / RAND_MAX ) );
        colors.push_back( color );
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

    // Read Person Tracker
    const std::string tracker_model = "../person-reidentification-retail-0031.bin";
    const std::string tracker_config = "../person-reidentification-retail-0031.xml";
    /*
    // Read Person Detection (FP16)
    const std::string tracker_model  = "../person-reidentification-retail-0031-fp16.bin";
    const std::string tracker_config = "../person-reidentification-retail-0031-fp16.xml";
    */

    std::unique_ptr<PedestrianTracker> tracker = CreateTracker( tracker_model, tracker_config );

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

        // Tracking Process
        static uint32_t index = 0;
        const uint64_t time_stamp = static_cast<uint64_t>( 1000.0 / fps * index++ );
        tracker->Process( frame, detections, time_stamp );

        // Draw Tracked Person
        // Position History
        std::unordered_map<size_t, std::vector<cv::Point>>& active_tracks = tracker->GetActiveTracks();
        for( const std::pair<const size_t, std::vector<cv::Point>>& active_track : active_tracks ){
            const size_t id = active_track.first;
            const Track& track = tracker->tracks().at( id );
            if( track.lost ){
                continue;
            }

            const std::vector<cv::Point> centers = active_track.second;
            for( int32_t i = 1; i < centers.size(); i++ ){
                const cv::Scalar color = colors[id % colors.size()];
                constexpr int32_t thickness = 2;
                cv::line( frame, centers[i - 1], centers[i], color, thickness );
            }
        }

        // Rectangle
        for( const TrackedObject& detection : tracker->TrackedDetections() ){
            const size_t id = detection.object_id;
            const Track& track = tracker->tracks().at( id );
            if( track.lost ){
                continue;
            }

            const cv::Rect rect = detection.rect;
            const cv::Scalar color = colors[id % colors.size()];
            constexpr int32_t rect_thickness = 3;
            cv::rectangle( frame, rect, color, rect_thickness );

            const cv::String info = cv::format( "ID %d", id );
            constexpr double font_scale = 0.5;
            constexpr int32_t font_thickness = 2;
            cv::putText( frame, info, cv::Point( rect.x, rect.y ), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness );
        }

        // Show Image
        cv::imshow( "Pedestrian Tracker", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
