/*
 Acknowledgements
 ----------------
 This mtcnn implementation based on the implementation by Kapil Sachdeva (@ksachdeva).
 Thank you for your great implementation!
 https://github.com/ksachdeva/opencv-mtcnn
*/

#ifndef __MTCNN_FACE__
#define __MTCNN_FACE__

#include <numeric>
#include <opencv2/opencv.hpp>

namespace mtcnn
{
    constexpr int32_t NUM_REGRESSIONS = 4;
    constexpr int32_t NUM_POINTS = 5;

    class BoundingBox
    {
        public:
            float x1;
            float y1;
            float x2;
            float y2;

        public:
            cv::Rect getRect() const
            {
                return cv::Rect( static_cast<int32_t>( x1 ), static_cast<int32_t>( y1 ), static_cast<int32_t>( x2 - x1 ), static_cast<int32_t>( y2 - y1 ) );
            }

            BoundingBox getSquare() const
            {
                BoundingBox rectangle;
                const float width  = x2 - x1;
                const float height = y2 - y1;
                const float side   = std::max( width, height );
                rectangle.x1 = static_cast<float>( static_cast<int32_t>( x1 + ( width  - side ) * 0.5f ) );
                rectangle.y1 = static_cast<float>( static_cast<int32_t>( y1 + ( height - side ) * 0.5f ) );
                rectangle.x2 = static_cast<float>( static_cast<int32_t>( rectangle.x1 + side ) );
                rectangle.y2 = static_cast<float>( static_cast<int32_t>( rectangle.y1 + side ) );
                return rectangle;
            }
    };

    class Face
    {
        public:
            BoundingBox rectangle;
            float score;
            float regression[mtcnn::NUM_REGRESSIONS];
            float points[2 * mtcnn::NUM_POINTS];

        public:
            static void apply_regression( std::vector<mtcnn::Face>& faces, const bool add_one = false )
            {
                const float offset = add_one ? 1.0f : 0.0f;
                for( int32_t i = 0; i < faces.size(); i++ )
                {
                    const float width  = faces[i].rectangle.x2 - faces[i].rectangle.x1 + offset;
                    const float height = faces[i].rectangle.y2 - faces[i].rectangle.y1 + offset;
                    faces[i].rectangle.x1 = faces[i].rectangle.x1 + faces[i].regression[1] * width;
                    faces[i].rectangle.y1 = faces[i].rectangle.y1 + faces[i].regression[0] * height;
                    faces[i].rectangle.x2 = faces[i].rectangle.x2 + faces[i].regression[3] * width;
                    faces[i].rectangle.y2 = faces[i].rectangle.y2 + faces[i].regression[2] * height;
                }
            }

            static void rectanglees_to_squares( std::vector<mtcnn::Face> &faces )
            {
                for (size_t i = 0; i < faces.size(); ++i) {
                    faces[i].rectangle = faces[i].rectangle.getSquare();
                }
            }

            static std::vector<mtcnn::Face> nms( std::vector<mtcnn::Face>& faces, const float overlap_threshold, const bool use_min = false )
            {
                std::vector<mtcnn::Face> faces_nms;
                if( faces.empty() ){
                    return faces_nms;
                }

                std::sort( faces.begin(), faces.end(), []( const mtcnn::Face &f1, const mtcnn::Face &f2 ){ return f1.score > f2.score; } );

                std::vector<int32_t> indices( faces.size() );
                std::iota( std::begin( indices ), std::end( indices ), 0 );

                while( indices.size() > 0 ){
                    const int32_t index = indices[0];
                    faces_nms.push_back( faces[index] );
                    std::vector<int32_t> temporary_indices = indices;
                    indices.clear();

                    for( int32_t i = 1; i < temporary_indices.size(); i++ ){
                        const int32_t temporary_index = temporary_indices[i];
                        const float inter_x1 = std::max( faces[index].rectangle.x1, faces[temporary_index].rectangle.x1 );
                        const float inter_y1 = std::max( faces[index].rectangle.y1, faces[temporary_index].rectangle.y1 );
                        const float inter_x2 = std::min( faces[index].rectangle.x2, faces[temporary_index].rectangle.x2 );
                        const float inter_y2 = std::min( faces[index].rectangle.y2, faces[temporary_index].rectangle.y2 );

                        const float width  = std::max( 0.f, ( inter_x2 - inter_x1 + 1 ) );
                        const float height = std::max( 0.f, ( inter_y2 - inter_y1 + 1 ) );
                        const float inter_area = width * height;

                        const float area1 = ( faces[index].rectangle.x2 - faces[index].rectangle.x1 + 1 ) *
                                            ( faces[index].rectangle.y2 - faces[index].rectangle.y1 + 1 );
                        const float area2 = ( faces[temporary_index].rectangle.x2 - faces[temporary_index].rectangle.x1 + 1 ) *
                                            ( faces[temporary_index].rectangle.y2 - faces[temporary_index].rectangle.y1 + 1 );

                        const float overlap = use_min ? inter_area / std::min( area1, area2 ) : inter_area / ( area1 + area2 - inter_area );
                        if( overlap <= overlap_threshold ){
                            indices.push_back( temporary_index );
                        }
                    }
                }

                return faces_nms;
            }
    };
}

#endif // __MTCNN_FACE__
