/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#include "onet.h"
#include "helpers.h"

constexpr int32_t INPUT_WIDTH  = 48;
constexpr int32_t INPUT_HEIGHT = 48;

constexpr float IMAGE_SCALE = 1.0f / 128.0f;
constexpr float IMAGE_MEAN  = 127.5f;

mtcnn::OutputNetwork::OutputNetwork( const mtcnn::OutputNetwork::Config& config )
    : score_threshold( config.score_threshold ),
      overlap_threshold( config.overlap_threshold )
{
    net = cv::dnn::readNet( config.caffemodel, config.prototext );
    if( net.empty() ){
        throw std::invalid_argument("invalid prototext or caffemodel");
    }
}

mtcnn::OutputNetwork::OutputNetwork(){}

std::vector<mtcnn::Face> mtcnn::OutputNetwork::run( const cv::Mat& image, const std::vector<mtcnn::Face>& faces )
{
    const cv::Size size = cv::Size( INPUT_WIDTH, INPUT_HEIGHT );

    std::vector<mtcnn::Face> results;

    for( const mtcnn::Face& face : faces ){
        cv::Mat roi = mtcnn::crop( image, face.rectangle.getRect() );
        cv::resize(roi, roi, size, 0.0, 0.0, cv::INTER_AREA );
        cv::Mat blob = cv::dnn::blobFromImage( roi, IMAGE_SCALE, cv::Size(), cv::Scalar( IMAGE_MEAN, IMAGE_MEAN, IMAGE_MEAN ), false );
        net.setInput( blob, "data" );

        const std::vector<cv::String> output_names{ "conv6-2", "conv6-3", "prob1" };
        std::vector<cv::Mat> outputs;
        net.forward( outputs, output_names );

        cv::Mat regressions = outputs[0];
        cv::Mat landmark    = outputs[1];
        cv::Mat scores      = outputs[2];

        const float* regressions_data = reinterpret_cast<float*>( regressions.data );
        const float* scores_data      = reinterpret_cast<float*>( scores.data );
        const float* landmark_data    = reinterpret_cast<float*>( landmark.data );

        if( scores_data[1] < score_threshold ){
            continue;
        }

        mtcnn::Face result = face;
        result.score = scores_data[1];
        for( int32_t i = 0; i < 4; i++ ){
            result.regression[i] = regressions_data[i];
        }

        const float width  = result.rectangle.x2 - result.rectangle.x1 + 1.0f;
        const float height = result.rectangle.y2 - result.rectangle.y1 + 1.0f;

        for( int32_t p = 0; p < mtcnn::NUM_POINTS; p++ ){
            result.points[2 * p + 0] = result.rectangle.x1 + ( landmark_data[p + mtcnn::NUM_POINTS] * width ) - 1;
            result.points[2 * p + 1] = result.rectangle.y1 + ( landmark_data[p] * height ) - 1;
        }

        results.push_back( result );
    }

    mtcnn::Face::apply_regression( results, true );
    results = mtcnn::Face::nms( results, overlap_threshold, true );

    return results;
}
