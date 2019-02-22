#ifndef __UTIL__
#define __UTIL__

#include <vector>
#include <string>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Get Output Layers Name
std::vector<std::string> getOutputsNames( const cv::dnn::Net& net )
{
    static std::vector<std::string> names;
    if( names.empty() ){
        std::vector<int32_t> out_layers = net.getUnconnectedOutLayers();
        std::vector<std::string> layers_names = net.getLayerNames();
        names.resize( out_layers.size() );
        for( size_t i = 0; i < out_layers.size(); ++i ){
            names[i] = layers_names[out_layers[i] - 1];
        }
    }
    return names;
}

// Get Output Layer Type
std::string getOutputLayerType( cv::dnn::Net& net )
{
    const std::vector<int32_t> out_layers = net.getUnconnectedOutLayers();
    const std::string output_layer_type = net.getLayer( out_layers[0] )->type;
    return output_layer_type;
}

// Read Class Name List
std::vector<std::string> readClassNameList( const std::string list_path )
{
    std::vector<std::string> classes;
    std::ifstream ifs( list_path );
    if( !ifs.is_open() ){
        return classes;
    }
    std::string class_name = "";
    while( std::getline( ifs, class_name ) ){
        classes.push_back( class_name );
    }
    return classes;
}

// Get Class Color Table for Visualize
std::vector<cv::Scalar> getClassColors( const int32_t number_of_colors )
{
    cv::RNG random;
    std::vector<cv::Scalar> colors;
    for( int32_t i = 0; i < number_of_colors; i++ ){
        cv::Scalar color( random.uniform( 0, 255 ), random.uniform( 0, 255 ), random.uniform( 0, 255 ) );
        colors.push_back( color );
    }
    return colors;
}

#endif // __UTIL__
