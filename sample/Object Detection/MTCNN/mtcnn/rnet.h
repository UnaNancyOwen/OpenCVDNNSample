/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_RNET__
#define __MTCNN_RNET__

#include "face.h"
#include <opencv2/dnn.hpp>

namespace mtcnn
{
    class RefineNetwork
    {
        public:
            struct Config
            {
                std::string prototext;
                std::string caffemodel;
                float score_threshold;
                float overlap_threshold;
            };

        private:
            cv::dnn::Net net;
            float score_threshold;
            float overlap_threshold;

        public:
            RefineNetwork( const mtcnn::RefineNetwork::Config& config );
            ~RefineNetwork();

        private:
            RefineNetwork( const mtcnn::RefineNetwork& r_net ) = delete;
            RefineNetwork& operator=( const mtcnn::RefineNetwork& r_net ) = delete;

        public:
            std::vector<mtcnn::Face> run( const cv::Mat& image, const std::vector<mtcnn::Face>& faces );
    };
}

#endif // __MTCNN_RNET__
