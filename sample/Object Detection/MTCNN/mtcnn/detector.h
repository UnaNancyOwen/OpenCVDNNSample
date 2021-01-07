/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_DETECTOR__
#define __MTCNN_DETECTOR__

#include "face.h"
#include "onet.h"
#include "pnet.h"
#include "rnet.h"

namespace mtcnn
{
    class Detector
    {
    private:
        std::unique_ptr<mtcnn::ProposalNetwork> p_net;
        std::unique_ptr<mtcnn::RefineNetwork> r_net;
        std::unique_ptr<mtcnn::OutputNetwork> o_net;

    public:
        Detector( const mtcnn::ProposalNetwork::Config& p_config, const mtcnn::RefineNetwork::Config& r_config, const mtcnn::OutputNetwork::Config& o_config );
        std::vector<mtcnn::Face> detect( const cv::Mat& image, const float min_size, const float scale_factor );
    };
}

#endif // __MTCNN_DETECTOR__
