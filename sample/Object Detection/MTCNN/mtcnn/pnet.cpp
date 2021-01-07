/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#include "pnet.h"

constexpr float   P_NET_WINDOW_SIZE = 12.0f;
constexpr int32_t P_NET_STRIDE      = 2;

constexpr float IMAGE_SCALE = 1.0f / 128.0f;
constexpr float IMAGE_MEAN  = 127.5f;

mtcnn::ProposalNetwork::ProposalNetwork( const mtcnn::ProposalNetwork::Config &config )
    : score_threshold( config.score_threshold ),
      overlap_threshold( config.overlap_threshold )
{
    net = cv::dnn::readNet( config.caffemodel, config.prototext );
    if( net.empty() ) {
        throw std::invalid_argument("invalid prototext or caffemodel");
    }
}

mtcnn::ProposalNetwork::~ProposalNetwork(){}

std::vector<mtcnn::Face> mtcnn::ProposalNetwork::make_faces( const cv::Mat& scores, const cv::Mat& regressions, const float scale_factor, const float threshold )
{
    const int32_t width  = scores.size[3];
    const int32_t height = scores.size[2];
    const int32_t size   = width * height;

    const float* regressions_data = reinterpret_cast<float*>( regressions.data );
    const float* scores_data      = reinterpret_cast<float*>( scores.data );
    scores_data += size;

    std::vector<mtcnn::Face> faces;
    for( int32_t i = 0; i < size; i++ ){
        if( scores_data[i] < threshold ){
            continue;
        }

        const int32_t y = i / width;
        const int32_t x = i - width * y;

        mtcnn::Face face;
        BoundingBox& bounding_box = face.rectangle;
        bounding_box.x1 = static_cast<float>( x * P_NET_STRIDE ) / scale_factor;
        bounding_box.y1 = static_cast<float>( y * P_NET_STRIDE ) / scale_factor;
        bounding_box.x2 = static_cast<float>( x * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.0f) / scale_factor;
        bounding_box.y2 = static_cast<float>( y * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.0f) / scale_factor;
        face.regression[0] = regressions_data[i + 0 * size];
        face.regression[1] = regressions_data[i + 1 * size];
        face.regression[2] = regressions_data[i + 2 * size];
        face.regression[3] = regressions_data[i + 3 * size];
        face.score = scores_data[i];

        faces.push_back( face );
    }

    return faces;
}

std::vector<mtcnn::Face> mtcnn::ProposalNetwork::run( const cv::Mat& image, const float min_size, const float scale_factor )
{
    std::vector<mtcnn::Face> results;
    const float max_size = static_cast<float>(std::min(image.rows, image.cols));
    float face_size = min_size;

    while( face_size <= max_size ){
        float current_scale = P_NET_WINDOW_SIZE / face_size;
        int32_t height = static_cast<int32_t>( std::ceil( image.rows * current_scale ) );
        int32_t width  = static_cast<int32_t>( std::ceil( image.cols * current_scale ) );
        cv::Mat blob = cv::dnn::blobFromImage( image, IMAGE_SCALE, cv::Size( width, height ), cv::Scalar( IMAGE_MEAN, IMAGE_MEAN, IMAGE_MEAN), false );
        net.setInput( blob, "data" );

        const std::vector<cv::String> output_names{ "conv4-2", "prob1" };
        std::vector<cv::Mat> outputs;
        net.forward( outputs, output_names );

        cv::Mat regressions = outputs[0];
        cv::Mat scores      = outputs[1];

        std::vector<mtcnn::Face> faces =  make_faces( scores, regressions, current_scale, score_threshold );
        if( !faces.empty() ){
            faces = mtcnn::Face::nms( faces, 0.5f );
        }
        if( !faces.empty() ){
            results.insert( results.end(), faces.begin(), faces.end() );
        }

        face_size /= scale_factor;
    }

    if( !results.empty() ){
        results = mtcnn::Face::nms( results, overlap_threshold );
        if( !results.empty() ){
            mtcnn::Face::apply_regression( results, false);
            mtcnn::Face::rectanglees_to_squares( results );
        }
    }

    return results;
}
