/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#include "detector.h"

mtcnn::Detector::Detector( const mtcnn::ProposalNetwork::Config& p_config, const mtcnn::RefineNetwork::Config& r_config, const mtcnn::OutputNetwork::Config& o_config )
{
    p_net = std::make_unique<mtcnn::ProposalNetwork>( p_config );
    r_net = std::make_unique<mtcnn::RefineNetwork>( r_config );
    o_net = std::make_unique<mtcnn::OutputNetwork>( o_config );
}

std::vector<mtcnn::Face> mtcnn::Detector::detect( const cv::Mat& image, const float min_size, const float scale_factor )
{
    if( image.empty() ){
        return std::vector<mtcnn::Face>();
    }

    cv::Mat rgb_image;
    if( image.channels() == 3 ){
        cv::cvtColor( image, rgb_image, cv::COLOR_BGR2RGB );
    }
    else if( image.channels() == 4 ){
        cv::cvtColor( image, rgb_image, cv::COLOR_BGRA2RGB );
    }
    else{
        throw std::runtime_error( "not support this image format!" );
    }

    rgb_image.convertTo( rgb_image, CV_32FC3 );
    rgb_image = rgb_image.t();

    // Proposal Network
    // find the initial set of faces
    std::vector<mtcnn::Face> faces = p_net->run( rgb_image, min_size, scale_factor );

    if( faces.empty() ){
        return faces;
    }

    // Refine Network
    // refine the output of the proposal network
    faces = r_net->run( rgb_image, faces );

    if( faces.empty() ){
        return faces;
    }

    // Output Network
    // find the face parts
    faces = o_net->run( rgb_image, faces );

    for( Face& face : faces ){
        std::swap( face.rectangle.x1, face.rectangle.y1 );
        std::swap( face.rectangle.x2, face.rectangle.y2 );
        for( int32_t p = 0; p < mtcnn::NUM_POINTS; p++ ){
            std::swap( face.points[2 * p], face.points[2 * p + 1] );
        }
    }

    return faces;
}
