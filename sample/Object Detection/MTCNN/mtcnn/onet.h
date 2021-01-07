/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_ONET__
#define __MTCNN_ONET__

#include "face.h"
#include <opencv2/dnn.hpp>

namespace mtcnn
{
    class OutputNetwork
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
            OutputNetwork( const mtcnn::OutputNetwork::Config& config );
            OutputNetwork();

        private:
            OutputNetwork( const mtcnn::OutputNetwork& o_net ) = delete;
            OutputNetwork& operator=( const mtcnn::OutputNetwork& o_net ) = delete;

        public:
            std::vector<mtcnn::Face> run( const cv::Mat& image, const std::vector<mtcnn::Face>& faces );
    };
}

#endif // __MTCNN_ONET__
