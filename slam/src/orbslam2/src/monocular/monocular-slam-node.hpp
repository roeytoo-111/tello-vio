#ifndef MONOCULAR_SLAM_NODE_HPP
#define MONOCULAR_SLAM_NODE_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

/**
 * ROS 2 wrapper around ORB-SLAM2 monocular.
 *
 * The original wrapper called TrackMonocular() every frame and then discarded
 * the pose it returned: no TF, no Odometry, no PoseStamped. A SLAM node that
 * publishes no pose cannot localise anything, which made the whole package
 * decorative. This version publishes the pose, in the right frame, and says so
 * when tracking is lost instead of going quiet.
 *
 * Frame conventions
 * -----------------
 * ORB-SLAM2 works entirely in the *optical* convention (x right, y down,
 * z forward) and defines its world frame as the first keyframe's camera pose.
 * ROS uses FLU/ENU (REP-103). Publishing ORB-SLAM2's numbers into a frame
 * called "map" without rotating them -- as the map-point publisher used to --
 * puts the map on its side in RViz and makes every consumer wrong.
 *
 * The node therefore applies a fixed rotation R_ros_optical when publishing,
 * so that:
 *   - `map` is a ROS-convention frame (z up), and
 *   - `camera_optical` remains the optical-convention camera frame that
 *     sensor_msgs/CameraInfo and image geometry expect.
 */
class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM2::System *pSLAM, const std::string &strVocFile,
                      const std::string &strSettingsFile);
    ~MonocularSlamNode();

    /** Stop the SLAM threads and persist the trajectory. Idempotent. */
    void ShutdownAndSave(const std::string &trajectory_path = "KeyFrameTrajectory.txt");

private:
    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);

    void UpdateSLAMState();
    void UpdateMapState();

    /** Publish TF + PoseStamped + Odometry from the current Tcw. */
    void PublishPose(const rclcpp::Time &stamp);
    void PublishFrame(const rclcpp::Time &stamp);
    void PublishMapPoints(const rclcpp::Time &stamp);
    void PublishState();

    void InitializeMarkersPublisher(const std::string &strSettingPath);

    cv::Mat DrawFrame();
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    ORB_SLAM2::System *m_SLAM;
    std::atomic<bool> m_shutdown_requested{false};

    std::mutex mMutex;

    cv_bridge::CvImagePtr m_cvImPtr;
    cv::Mat m_last_image;      ///< guarded by mMutex; used by DrawFrame
    cv::Mat Tcw;               ///< world -> camera, empty when tracking failed

    // Frames and behaviour, all parameters.
    std::string m_map_frame;
    std::string m_camera_frame;
    std::string m_image_topic;
    bool m_publish_tf{true};
    double m_map_publish_period{0.5};
    int m_max_map_points{20000};

    /// Rotation taking ORB-SLAM2's optical axes into ROS axes.
    cv::Matx33d m_R_ros_optical;

    rclcpp::Time m_last_map_publish;

    int N{0};
    std::vector<cv::KeyPoint> mvCurrentKeys;
    std::vector<bool> mvbMap, mvbVO;
    int mnTracked{0}, mnTrackedVO{0};
    std::vector<cv::KeyPoint> mvIniKeys;
    std::vector<int> mvIniMatches;

    std::vector<ORB_SLAM2::KeyFrame *> mvKeyFrames;
    std::vector<ORB_SLAM2::MapPoint *> mvMapPoints;
    std::vector<ORB_SLAM2::MapPoint *> mvRefMapPoints;

    visualization_msgs::msg::Marker mPoints;
    visualization_msgs::msg::Marker mReferencePoints;

    float mPointSize{0.02f};

    int mState{ORB_SLAM2::Tracking::SYSTEM_NOT_READY};
    int m_prev_state{-1};
    bool mbOnlyTracking{false};
    std::size_t m_n_keyframes{0};
    std::size_t m_n_mappoints{0};
    std::uint64_t m_n_images{0};
    std::uint64_t m_n_tracked_ok{0};

    nav_msgs::msg::Path m_path;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_annotated_image_publisher;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr m_map_publisher;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_publisher;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_publisher;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_publisher;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_state_publisher;
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
};

#endif  // MONOCULAR_SLAM_NODE_HPP
