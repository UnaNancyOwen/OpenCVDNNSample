/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_PNET__
#define __MTCNN_PNET__

#include "face.h"
#include <opencv2/dnn.hpp>

namespace mtcnn
{
    class ProposalNetwork
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

        private:
            std::vector<mtcnn::Face> make_faces( const cv::Mat& scores, const cv::Mat& regressions, const float scale_factor, const float threshold );

        public:
            ProposalNetwork( const mtcnn::ProposalNetwork::Config& config );
            ~ProposalNetwork();

        private:
            ProposalNetwork( const mtcnn::ProposalNetwork& p_net ) = delete;
            ProposalNetwork& operator=( const mtcnn::ProposalNetwork& p_net ) = delete;

        public:
            std::vector<mtcnn::Face> run( const cv::Mat& image, const float min_size, const float scale_factor );
    };
}

#endif // __MTCNN_PNET__
