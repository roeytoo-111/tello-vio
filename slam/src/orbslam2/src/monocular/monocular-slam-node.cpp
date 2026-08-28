#include "monocular-slam-node.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"

using ImageMsg = sensor_msgs::msg::Image;
using MarkerMsg = visualization_msgs::msg::Marker;
using PointMsg = geometry_msgs::msg::Point;
using PoseMsg = geometry_msgs::msg::PoseStamped;
using OdomMsg = nav_msgs::msg::Odometry;

namespace
{

/** Rotation matrix -> quaternion (x, y, z, w), Shepperd's branch-max method. */
void RotationToQuaternion(const cv::Matx33d &R, double q[4])
{
    const double t = R(0, 0) + R(1, 1) + R(2, 2);
    if (t > 0.0) {
        const double s = std::sqrt(t + 1.0) * 2.0;
        q[3] = 0.25 * s;
        q[0] = (R(2, 1) - R(1, 2)) / s;
        q[1] = (R(0, 2) - R(2, 0)) / s;
        q[2] = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        q[3] = (R(2, 1) - R(1, 2)) / s;
        q[0] = 0.25 * s;
        q[1] = (R(0, 1) + R(1, 0)) / s;
        q[2] = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        q[3] = (R(0, 2) - R(2, 0)) / s;
        q[0] = (R(0, 1) + R(1, 0)) / s;
        q[1] = 0.25 * s;
        q[2] = (R(1, 2) + R(2, 1)) / s;
    } else {
        const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        q[3] = (R(1, 0) - R(0, 1)) / s;
        q[0] = (R(0, 2) + R(2, 0)) / s;
        q[1] = (R(1, 2) + R(2, 1)) / s;
        q[2] = 0.25 * s;
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 1e-12) {
        for (int i = 0; i < 4; ++i) q[i] /= n;
    }
}

const char *StateName(int s)
{
    switch (s) {
    case ORB_SLAM2::Tracking::SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
    case ORB_SLAM2::Tracking::NO_IMAGES_YET:    return "NO_IMAGES_YET";
    case ORB_SLAM2::Tracking::NOT_INITIALIZED:  return "NOT_INITIALIZED";
    case ORB_SLAM2::Tracking::OK:               return "OK";
    case ORB_SLAM2::Tracking::LOST:             return "LOST";
    default:                                    return "UNKNOWN";
    }
}

}  // namespace

MonocularSlamNode::MonocularSlamNode(ORB_SLAM2::System *pSLAM,
                                     const std::string & /*strVocFile*/,
                                     const std::string &strSettingsFile)
: Node("orbslam"), m_SLAM(pSLAM), m_last_map_publish(0, 0, RCL_ROS_TIME)
{
    m_image_topic = this->declare_parameter<std::string>("image_topic", "/image_raw");
    m_map_frame = this->declare_parameter<std::string>("map_frame", "map");
    m_camera_frame = this->declare_parameter<std::string>("camera_frame", "camera_optical");
    m_publish_tf = this->declare_parameter<bool>("publish_tf", true);
    m_map_publish_period = this->declare_parameter<double>("map_publish_period", 0.5);
    m_max_map_points = this->declare_parameter<int>("max_map_points", 20000);

    // ORB-SLAM2's world frame inherits the optical convention of the first
    // keyframe (x right, y down, z forward). Rotating by Rz(-90) Rx(-90) turns
    // it into a ROS-convention frame (x forward, y left, z up), which is what
    // anything named "map" is required to be.
    m_R_ros_optical = cv::Matx33d(0, 0, 1,
                                  -1, 0, 0,
                                  0, -1, 0);

    m_image_subscriber = this->create_subscription<ImageMsg>(
        m_image_topic, rclcpp::SensorDataQoS(),
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));

    m_annotated_image_publisher =
        this->create_publisher<ImageMsg>("~/annotated_frame", rclcpp::SensorDataQoS());
    m_map_publisher = this->create_publisher<MarkerMsg>("~/map", rclcpp::QoS(2));
    m_pose_publisher = this->create_publisher<PoseMsg>("~/pose", rclcpp::QoS(10));
    m_odom_publisher = this->create_publisher<OdomMsg>("~/odom", rclcpp::QoS(10));
    m_path_publisher = this->create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1));
    m_state_publisher =
        this->create_publisher<std_msgs::msg::String>("~/tracking_state", rclcpp::QoS(10));
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    m_path.header.frame_id = m_map_frame;
    mState = ORB_SLAM2::Tracking::SYSTEM_NOT_READY;
    mbOnlyTracking = false;

    InitializeMarkersPublisher(strSettingsFile);

    RCLCPP_INFO(this->get_logger(),
                "orbslam: subscribing %s, publishing %s -> %s",
                m_image_topic.c_str(), m_map_frame.c_str(), m_camera_frame.c_str());
}

MonocularSlamNode::~MonocularSlamNode()
{
    try {
        ShutdownAndSave("KeyFrameTrajectory.txt");
    } catch (...) {
        // Never throw from a destructor.
    }
}

void MonocularSlamNode::ShutdownAndSave(const std::string &trajectory_path)
{
    if (m_shutdown_requested.exchange(true)) {
        return;
    }
    if (!m_SLAM) {
        return;
    }

    const auto kfs = m_SLAM->GetAllKeyFrames();
    const auto mps = m_SLAM->GetAllMapPoints();
    RCLCPP_INFO(this->get_logger(),
                "shutdown: %zu keyframes, %zu map points, %lu images "
                "(%lu tracked OK) -> %s",
                kfs.size(), mps.size(),
                static_cast<unsigned long>(m_n_images),
                static_cast<unsigned long>(m_n_tracked_ok),
                trajectory_path.c_str());

    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM(trajectory_path);
}

// --------------------------------------------------------------------------- //

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    if (m_shutdown_requested.load()) {
        return;
    }

    try {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // ORB-SLAM2 is fed grayscale so the RGB/BGR question never arises: ORB is a
    // binary intensity-comparison descriptor, and converting here removes an
    // entire class of encoding bug at no cost.
    cv::Mat im_gray;
    const int ch = m_cvImPtr->image.channels();
    if (ch == 1) {
        im_gray = m_cvImPtr->image;
    } else if (ch == 3) {
        cv::cvtColor(m_cvImPtr->image, im_gray, cv::COLOR_BGR2GRAY);
    } else if (ch == 4) {
        cv::cvtColor(m_cvImPtr->image, im_gray, cv::COLOR_BGRA2GRAY);
    } else {
        RCLCPP_ERROR(this->get_logger(), "unsupported image channels: %d", ch);
        return;
    }

    const rclcpp::Time stamp(msg->header.stamp);
    ++m_n_images;

    // TrackMonocular returns an EMPTY Mat when tracking fails. Treating that as
    // a pose (or dereferencing it) is the standard way this wrapper breaks the
    // moment the drone flies past a blank wall.
    cv::Mat Tcw_new = m_SLAM->TrackMonocular(im_gray, stamp.seconds());

    {
        std::lock_guard<std::mutex> lock(mMutex);
        Tcw = Tcw_new;
        m_cvImPtr->image.copyTo(m_last_image);
    }

    UpdateSLAMState();
    UpdateMapState();
    PublishState();

    if (!Tcw_new.empty()) {
        ++m_n_tracked_ok;
        PublishPose(stamp);
    }

    if (m_annotated_image_publisher->get_subscription_count() > 0) {
        PublishFrame(stamp);
    }

    // Serialising the whole map into Markers costs O(map size) and runs on the
    // tracking thread. Doing it per frame turns a 20k-point map into tens of
    // milliseconds of work at 30 Hz, stealing the budget from tracking itself.
    // Rate-limit it, and skip it entirely when nobody is subscribed.
    if (m_map_publisher->get_subscription_count() > 0) {
        const auto now = this->now();
        if ((now - m_last_map_publish).seconds() >= m_map_publish_period) {
            m_last_map_publish = now;
            PublishMapPoints(now);
        }
    }
}

// --------------------------------------------------------------------------- //

void MonocularSlamNode::PublishPose(const rclcpp::Time &stamp)
{
    cv::Mat Tcw_local;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (Tcw.empty()) {
            return;
        }
        Tcw_local = Tcw.clone();
    }

    // ORB-SLAM2 returns Tcw: world -> camera, as CV_32F. The camera pose in the
    // world is its inverse: Rwc = Rcw^T, twc = -Rcw^T * tcw.
    cv::Matx33d Rcw;
    cv::Vec3d tcw;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rcw(i, j) = static_cast<double>(Tcw_local.at<float>(i, j));
        }
        tcw(i) = static_cast<double>(Tcw_local.at<float>(i, 3));
    }
    const cv::Matx33d Rwc = Rcw.t();
    const cv::Vec3d twc = -(Rwc * tcw);

    // Rotate out of ORB-SLAM2's optical world frame into a ROS-convention one.
    const cv::Matx33d R_map_cam = m_R_ros_optical * Rwc;
    const cv::Vec3d t_map_cam = m_R_ros_optical * twc;

    double q[4];
    RotationToQuaternion(R_map_cam, q);

    PoseMsg pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = m_map_frame;
    pose.pose.position.x = t_map_cam(0);
    pose.pose.position.y = t_map_cam(1);
    pose.pose.position.z = t_map_cam(2);
    pose.pose.orientation.x = q[0];
    pose.pose.orientation.y = q[1];
    pose.pose.orientation.z = q[2];
    pose.pose.orientation.w = q[3];
    m_pose_publisher->publish(pose);

    OdomMsg odom;
    odom.header = pose.header;
    odom.child_frame_id = m_camera_frame;
    odom.pose.pose = pose.pose;
    // Monocular SLAM has no metric scale, so a covariance in metres would be
    // meaningless. Flag it unavailable rather than inventing numbers: the
    // scale comes from tello_vio/map_align, not from here.
    odom.pose.covariance[0] = -1.0;
    odom.twist.covariance[0] = -1.0;
    m_odom_publisher->publish(odom);

    if (m_publish_tf) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header = pose.header;
        tf.child_frame_id = m_camera_frame;
        tf.transform.translation.x = t_map_cam(0);
        tf.transform.translation.y = t_map_cam(1);
        tf.transform.translation.z = t_map_cam(2);
        tf.transform.rotation = pose.pose.orientation;
        m_tf_broadcaster->sendTransform(tf);
    }

    if (m_path_publisher->get_subscription_count() > 0) {
        m_path.header.stamp = stamp;
        m_path.poses.push_back(pose);
        if (m_path.poses.size() > 5000) {
            m_path.poses.erase(m_path.poses.begin());
        }
        m_path_publisher->publish(m_path);
    }
}

void MonocularSlamNode::PublishState()
{
    if (mState != m_prev_state) {
        RCLCPP_INFO(this->get_logger(), "tracking state: %s -> %s",
                    StateName(m_prev_state), StateName(mState));
        m_prev_state = mState;
    }
    if (m_state_publisher->get_subscription_count() == 0) {
        return;
    }
    std_msgs::msg::String msg;
    std::ostringstream os;
    os << StateName(mState) << " keyframes=" << m_n_keyframes
       << " mappoints=" << m_n_mappoints
       << " matches=" << mnTracked << " vo_matches=" << mnTrackedVO;
    msg.data = os.str();
    m_state_publisher->publish(msg);
}

// --------------------------------------------------------------------------- //

void MonocularSlamNode::InitializeMarkersPublisher(const std::string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    if (fSettings.isOpened() && !fSettings["Viewer.PointSize"].empty()) {
        mPointSize = static_cast<float>(fSettings["Viewer.PointSize"]);
    }
    if (mPointSize <= 0.0f) {
        mPointSize = 0.02f;
    }

    mPoints.header.frame_id = m_map_frame;
    mPoints.ns = "MapPoints";
    mPoints.id = 0;
    mPoints.type = MarkerMsg::POINTS;
    mPoints.scale.x = mPointSize;
    mPoints.scale.y = mPointSize;
    mPoints.pose.orientation.w = 1.0;
    mPoints.action = MarkerMsg::ADD;
    mPoints.color.a = 1.0;

    mReferencePoints = mPoints;
    mReferencePoints.ns = "MapPoints";
    mReferencePoints.id = 1;
    mReferencePoints.color.r = 1.0f;
    mReferencePoints.color.a = 1.0;
}

void MonocularSlamNode::UpdateSLAMState()
{
    std::lock_guard<std::mutex> lock(mMutex);

    ORB_SLAM2::Frame currentFrame = m_SLAM->GetCurrentFrame();
    mState = m_SLAM->GetTrackingState();

    mvCurrentKeys = currentFrame.mvKeys;
    N = static_cast<int>(mvCurrentKeys.size());
    mvbVO.assign(N, false);
    mvbMap.assign(N, false);
    mnTracked = 0;
    mnTrackedVO = 0;

    if (mState == ORB_SLAM2::Tracking::NOT_INITIALIZED) {
        mvIniKeys = m_SLAM->GetInitialKeys();
        mvIniMatches = m_SLAM->GetInitialMatches();
    } else if (mState == ORB_SLAM2::Tracking::OK) {
        for (int i = 0; i < N; ++i) {
            ORB_SLAM2::MapPoint *pMP = currentFrame.mvpMapPoints[i];
            if (pMP && !currentFrame.mvbOutlier[i]) {
                if (pMP->Observations() > 0) {
                    mvbMap[i] = true;
                    ++mnTracked;
                } else {
                    mvbVO[i] = true;
                    ++mnTrackedVO;
                }
            }
        }
    }
}

void MonocularSlamNode::UpdateMapState()
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (m_SLAM->IsMapOptimized()) {
        mvKeyFrames = m_SLAM->GetAllKeyFrames();
        mvMapPoints = m_SLAM->GetAllMapPoints();
        mvRefMapPoints = m_SLAM->GetReferenceMapPoints();
        m_n_keyframes = mvKeyFrames.size();
        m_n_mappoints = mvMapPoints.size();
    }
}

// --------------------------------------------------------------------------- //

void MonocularSlamNode::PublishFrame(const rclcpp::Time &stamp)
{
    cv::Mat im = DrawFrame();
    if (im.empty()) {
        return;
    }
    cv_bridge::CvImage rosImage;
    rosImage.image = im;
    // Carry the *source* image's stamp, not the publish instant: an annotated
    // frame stamped "now" cannot be time-aligned with anything, and the offset
    // is exactly the SLAM processing time, which is what you would want to see.
    rosImage.header.stamp = stamp;
    rosImage.header.frame_id = m_camera_frame;
    rosImage.encoding = "bgr8";
    m_annotated_image_publisher->publish(*rosImage.toImageMsg());
}

cv::Mat MonocularSlamNode::DrawFrame()
{
    cv::Mat im;
    std::vector<cv::KeyPoint> vIniKeys, vCurrentKeys;
    std::vector<int> vMatches;
    std::vector<bool> vbVO, vbMap;
    int state;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (m_last_image.empty()) {
            return cv::Mat();
        }
        m_last_image.copyTo(im);
        state = mState;
        vCurrentKeys = mvCurrentKeys;
        if (state == ORB_SLAM2::Tracking::NOT_INITIALIZED) {
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        } else if (state == ORB_SLAM2::Tracking::OK) {
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
    }

    if (im.channels() < 3) {
        cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }

    if (state == ORB_SLAM2::Tracking::NOT_INITIALIZED) {
        const std::size_t n = std::min(vMatches.size(), vIniKeys.size());
        for (std::size_t i = 0; i < n; ++i) {
            const int j = vMatches[i];
            if (j >= 0 && j < static_cast<int>(vCurrentKeys.size())) {
                cv::line(im, vIniKeys[i].pt, vCurrentKeys[j].pt, cv::Scalar(0, 255, 0));
            }
        }
    } else if (state == ORB_SLAM2::Tracking::OK) {
        const float r = 5.0f;
        // Bound the loop by the *shortest* of the three vectors. They are
        // filled under the same lock and should agree, but an out-of-range
        // index here is a segfault in the middle of a flight.
        const std::size_t n =
            std::min({vCurrentKeys.size(), vbVO.size(), vbMap.size()});
        for (std::size_t i = 0; i < n; ++i) {
            if (!vbVO[i] && !vbMap[i]) {
                continue;
            }
            const cv::Point2f pt1(vCurrentKeys[i].pt.x - r, vCurrentKeys[i].pt.y - r);
            const cv::Point2f pt2(vCurrentKeys[i].pt.x + r, vCurrentKeys[i].pt.y + r);
            const cv::Scalar c = vbMap[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
            cv::rectangle(im, pt1, pt2, c);
            cv::circle(im, vCurrentKeys[i].pt, 2, c, -1);
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);
    return imWithInfo;
}

void MonocularSlamNode::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    std::stringstream s;
    if (nState == ORB_SLAM2::Tracking::NO_IMAGES_YET) {
        s << " WAITING FOR IMAGES";
    } else if (nState == ORB_SLAM2::Tracking::NOT_INITIALIZED) {
        s << " TRYING TO INITIALIZE ";
    } else if (nState == ORB_SLAM2::Tracking::OK) {
        s << (mbOnlyTracking ? "LOCALIZATION | " : "SLAM MODE |  ");
        s << "KFs: " << m_n_keyframes << ", MPs: " << m_n_mappoints
          << ", Matches: " << mnTracked;
        if (mnTrackedVO > 0) {
            s << ", + VO matches: " << mnTrackedVO;
        }
    } else if (nState == ORB_SLAM2::Tracking::LOST) {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    } else if (nState == ORB_SLAM2::Tracking::SYSTEM_NOT_READY) {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline = 0;
    const cv::Size textSize =
        cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
    imText.rowRange(im.rows, imText.rows) =
        cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

// --------------------------------------------------------------------------- //

void MonocularSlamNode::PublishMapPoints(const rclcpp::Time &stamp)
{
    std::vector<ORB_SLAM2::MapPoint *> all, ref;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        all = mvMapPoints;
        ref = mvRefMapPoints;
    }
    if (all.empty()) {
        return;
    }

    const std::set<ORB_SLAM2::MapPoint *> spRefMPs(ref.begin(), ref.end());

    mPoints.points.clear();
    mReferencePoints.points.clear();
    mPoints.points.reserve(std::min<std::size_t>(all.size(), m_max_map_points));

    // Decimate rather than truncate when the map outgrows the budget: taking
    // the first N points would show only the oldest corner of the map, which
    // looks like SLAM having stopped working.
    const std::size_t stride =
        (m_max_map_points > 0 && all.size() > static_cast<std::size_t>(m_max_map_points))
            ? (all.size() / static_cast<std::size_t>(m_max_map_points)) + 1
            : 1;

    for (std::size_t i = 0; i < all.size(); i += stride) {
        ORB_SLAM2::MapPoint *mp = all[i];
        if (!mp || mp->isBad() || spRefMPs.count(mp)) {
            continue;
        }
        const cv::Mat pos = mp->GetWorldPos();
        if (pos.empty()) {
            continue;
        }
        // Same optical -> ROS rotation as the pose. Publishing raw ORB-SLAM2
        // coordinates into a frame called "map" lays the map on its side.
        const cv::Vec3d p = m_R_ros_optical * cv::Vec3d(pos.at<float>(0),
                                                        pos.at<float>(1),
                                                        pos.at<float>(2));
        PointMsg q;
        q.x = p(0);
        q.y = p(1);
        q.z = p(2);
        mPoints.points.push_back(q);
    }

    for (ORB_SLAM2::MapPoint *mp : spRefMPs) {
        if (!mp || mp->isBad()) {
            continue;
        }
        const cv::Mat pos = mp->GetWorldPos();
        if (pos.empty()) {
            continue;
        }
        const cv::Vec3d p = m_R_ros_optical * cv::Vec3d(pos.at<float>(0),
                                                        pos.at<float>(1),
                                                        pos.at<float>(2));
        PointMsg q;
        q.x = p(0);
        q.y = p(1);
        q.z = p(2);
        mReferencePoints.points.push_back(q);
    }

    mPoints.header.stamp = stamp;
    mReferencePoints.header.stamp = stamp;
    m_map_publisher->publish(mPoints);
    m_map_publisher->publish(mReferencePoints);
}
