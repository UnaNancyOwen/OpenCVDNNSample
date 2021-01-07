/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_HELPERS__
#define __MTCNN_HELPERS__

#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

namespace mtcnn
{
    inline cv::Mat crop( const cv::Mat& image, cv::Rect rectangle )
    {
        cv::Mat crop_image = cv::Mat::zeros( rectangle.height, rectangle.width, image.type() );
        const int32_t dx = std::abs( std::min( 0, rectangle.x ) );
        if( dx > 0 ){
            rectangle.x = 0;
        }
        rectangle.width -= dx;
        const int32_t dy = std::abs( std::min( 0, rectangle.y ) );
        if( dy > 0 ){
            rectangle.y = 0;
        }
        rectangle.height -= dy;
        int32_t dw = std::abs( std::min( 0, image.cols - 1 - ( rectangle.x + rectangle.width ) ) );
        rectangle.width -= dw;
        int32_t dh = std::abs( std::min( 0, image.rows - 1 - ( rectangle.y + rectangle.height ) ) );
        rectangle.height -= dh;
        if( rectangle.width > 0 && rectangle.height > 0 ){
            image( rectangle ).copyTo( crop_image(cv::Range( dy, dy + rectangle.height ), cv::Range( dx, dx + rectangle.width ) ) );
        }
        return crop_image;
    }
}

#endif // __MTCNN_HELPERS__