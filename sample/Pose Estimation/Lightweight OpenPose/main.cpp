#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "human_pose_estimator.hpp"

namespace human_pose_estimation
{
    // Joints
    enum Joints
    {
        NOSE           = 0,
        SPINE_SHOULDER = 1,
        SHOULDER_RIGHT = 2,
        ELBOW_RIGHT    = 3,
        HAND_RIGHT     = 4,
        SHOULDER_LEFT  = 5,
        ELBOW_LEFT     = 6,
        HAND_LEFT      = 7,
        HIP_RIGHT      = 8,
        KNEE_RIGHT     = 9,
        FOOT_RIGHT     = 10,
        HIP_LEFT       = 11,
        KNEE_LEFT      = 12,
        FOOT_LEFT      = 13,
        EYE_RIGHT      = 14,
        EYE_LEFT       = 15,
        EAR_RIGHT      = 16,
        EAR_LEFT       = 17,
        JOINT_COUNT    = 18
    };
}

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Read Lightweight OpenPose
    const std::string model  = "../human-pose-estimation-0001.bin";
    const std::string config = "../human-pose-estimation-0001.xml";
    /*
    // Read Lightweight OpenPose (FP16)
    const std::string model  = "../human-pose-estimation-0001-fp16.bin";
    const std::string config = "../human-pose-estimation-0001-fp16.xml";
    */
    /*
    // Read Lightweight OpenPose (INT8)
    const std::string model  = "../human-pose-estimation-0001-int8.bin";
    const std::string config = "../human-pose-estimation-0001-int8.xml";
    */

    const std::string device = "CPU";
    const bool        report = false;
    human_pose_estimation::HumanPoseEstimator estimator( config, device, report );

    while( true ){
        // Read Frame
        cv::Mat frame;
        capture >> frame;
        if( frame.empty() ){
            cv::waitKey( 0 );
            break;
        }
        if( frame.channels() == 4 ){
            cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        }

        // Estimate Human Pose
        std::vector<human_pose_estimation::HumanPose> poses = estimator.estimate( frame );

        // Draw Estimated Human Pose
        // Color Table
        const std::vector<cv::Scalar> colors = {
            cv::Scalar( 255,   0,   0 ), cv::Scalar( 255,  85,   0 ), cv::Scalar( 255, 170,   0 ),
            cv::Scalar( 255, 255,   0 ), cv::Scalar( 170, 255,   0 ), cv::Scalar(  85, 255,   0 ),
            cv::Scalar(   0, 255,   0 ), cv::Scalar(   0, 255,  85 ), cv::Scalar(   0, 255, 170 ),
            cv::Scalar(   0, 255, 255 ), cv::Scalar(   0, 170, 255 ), cv::Scalar(   0,  85, 255 ),
            cv::Scalar(   0,   0, 255 ), cv::Scalar(  85,   0, 255 ), cv::Scalar( 170,   0, 255 ),
            cv::Scalar( 255,   0, 255 ), cv::Scalar( 255,   0, 170 ), cv::Scalar( 255,   0,  85 )
        };

        // Draw Joints
        const cv::Point2f absent( -1.0f, -1.0f );
        for( const human_pose_estimation::HumanPose& pose : poses ){
            for( int32_t i = 0; i < pose.keypoints.size(); i++ ) {
                if( pose.keypoints[i] == absent ){
                    continue;
                }

                constexpr int32_t radius = 4;
                constexpr int32_t thickness = -1;
                cv::circle( frame, pose.keypoints[i], radius, colors[i], thickness );
            }
        }

        // Joints Pair List
        const std::vector<std::pair<int32_t, int32_t>> bones = {
            { human_pose_estimation::Joints::SPINE_SHOULDER, human_pose_estimation::Joints::HIP_RIGHT      },
            { human_pose_estimation::Joints::SPINE_SHOULDER, human_pose_estimation::Joints::HIP_LEFT       },
            { human_pose_estimation::Joints::HIP_RIGHT     , human_pose_estimation::Joints::KNEE_RIGHT     },
            { human_pose_estimation::Joints::HIP_LEFT      , human_pose_estimation::Joints::KNEE_LEFT      },
            { human_pose_estimation::Joints::KNEE_RIGHT    , human_pose_estimation::Joints::FOOT_RIGHT     },
            { human_pose_estimation::Joints::KNEE_LEFT     , human_pose_estimation::Joints::FOOT_LEFT      },
            { human_pose_estimation::Joints::SPINE_SHOULDER, human_pose_estimation::Joints::SHOULDER_RIGHT },
            { human_pose_estimation::Joints::SPINE_SHOULDER, human_pose_estimation::Joints::SHOULDER_LEFT  },
            { human_pose_estimation::Joints::SHOULDER_RIGHT, human_pose_estimation::Joints::ELBOW_RIGHT    },
            { human_pose_estimation::Joints::SHOULDER_LEFT , human_pose_estimation::Joints::ELBOW_LEFT     },
            { human_pose_estimation::Joints::ELBOW_RIGHT   , human_pose_estimation::Joints::HAND_RIGHT     },
            { human_pose_estimation::Joints::ELBOW_LEFT    , human_pose_estimation::Joints::HAND_LEFT      },
            { human_pose_estimation::Joints::SPINE_SHOULDER, human_pose_estimation::Joints::NOSE           },
            { human_pose_estimation::Joints::NOSE          , human_pose_estimation::Joints::EYE_RIGHT      },
            { human_pose_estimation::Joints::NOSE          , human_pose_estimation::Joints::EYE_LEFT       },
            { human_pose_estimation::Joints::EYE_RIGHT     , human_pose_estimation::Joints::EAR_RIGHT      },
            { human_pose_estimation::Joints::EYE_LEFT      , human_pose_estimation::Joints::EAR_LEFT       }
        };

        // Draw Bones
        for( const human_pose_estimation::HumanPose& pose : poses ){
            for( const std::pair<int32_t, int32_t>& bone : bones ){
                const std::pair<cv::Point2f, cv::Point2f> points( pose.keypoints[bone.first], pose.keypoints[bone.second] );
                if( points.first == absent || points.second == absent ){
                    continue;
                }

                const float mean_x = ( points.first.x + points.second.x ) / 2.0f;
                const float mean_y = ( points.first.y + points.second.y ) / 2.0f;
                const cv::Point2f difference = points.first - points.second;
                const float length = std::sqrt( difference.x * difference.x + difference.y * difference.y );
                const int32_t angle = static_cast<int32_t>( std::atan2( difference.y, difference.x ) * 180 / CV_PI );
                constexpr int32_t bone_width = 4;
                std::vector<cv::Point> polygon;
                cv::ellipse2Poly( cv::Point2d( mean_x, mean_y ), cv::Size2d( length / 2, bone_width ), angle, 0, 360, 1, polygon );
                cv::fillConvexPoly( frame, polygon, colors[bone.second] );
            }
        }

        // Show Image
        cv::imshow( "Lightweight OpenPose", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
