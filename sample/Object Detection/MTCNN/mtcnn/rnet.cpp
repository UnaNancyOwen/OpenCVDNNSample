/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#include "rnet.h"
#include "helpers.h"

constexpr int32_t INPUT_WIDTH  = 24;
constexpr int32_t INPUT_HEIGHT = 24;

constexpr float IMAGE_SCALE = 1.0f / 128.0f;
constexpr float IMAGE_MEAN  = 127.5f;

mtcnn::RefineNetwork::RefineNetwork( const mtcnn::RefineNetwork::Config& config )
    : score_threshold( config.score_threshold ),
      overlap_threshold( config.overlap_threshold )
{
    net = cv::dnn::readNet( config.caffemodel, config.prototext );
    if( net.empty() ){
        throw std::invalid_argument("invalid prototext or caffemodel");
    }
}

mtcnn::RefineNetwork::~RefineNetwork(){}

std::vector<mtcnn::Face> mtcnn::RefineNetwork::run( const cv::Mat &image, const std::vector<mtcnn::Face>& faces )
{
    const cv::Size size = cv::Size( INPUT_WIDTH, INPUT_HEIGHT );

    std::vector<cv::Mat> inputs;
    for( const mtcnn::Face& face : faces ){
        cv::Mat roi = mtcnn::crop( image, face.rectangle.getRect() );
        cv::resize( roi, roi, size, 0.0, 0.0, cv::INTER_AREA );
        inputs.push_back( roi );
    }
    cv::Mat blob = cv::dnn::blobFromImages( inputs, IMAGE_SCALE, cv::Size(), cv::Scalar( IMAGE_MEAN, IMAGE_MEAN, IMAGE_MEAN ), false );
    net.setInput( blob, "data" );

    const std::vector<cv::String> output_names{ "conv5-2", "prob1" };
    std::vector<cv::Mat> outputs;
    net.forward( outputs, output_names );

    cv::Mat regressions = outputs[0];
    cv::Mat scores      = outputs[1];

    std::vector<mtcnn::Face> results;

    const float* regressions_data = reinterpret_cast<float*>( regressions.data );
    const float* scores_data      = reinterpret_cast<float*>( scores.data );

    for( int32_t k = 0; k < faces.size(); k++ ){
        if( scores_data[2 * k + 1] < score_threshold ){
            continue;
        }

        mtcnn::Face result = faces[k];
        result.score = scores_data[2 * k + 1];
        for( int32_t i = 0; i < 4; i++ ){
            result.regression[i] = regressions_data[4 * k + i];
        }

        results.push_back( result );
    }

    results = mtcnn::Face::nms( results, overlap_threshold );
    mtcnn::Face::apply_regression( results, true );
    mtcnn::Face::rectanglees_to_squares( results );

    return results;
}
