#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "monocular-slam-node.hpp"

#include "System.h"

namespace
{
bool Readable(const std::string &path)
{
    std::ifstream f(path);
    return f.good();
}
}  // namespace

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    if (argc < 3) {
        std::cerr << "\nUsage: ros2 run orbslam2 mono <vocabulary> <settings>\n\n"
                  << "  vocabulary : ORB_SLAM2/Vocabulary/ORBvoc.txt\n"
                  << "  settings   : e.g. "
                     "$(ros2 pkg prefix orbslam2)/share/orbslam2/config.yaml\n\n"
                  << "The settings file MUST match the resolution actually being\n"
                  << "published on the image topic. If you launched the driver with\n"
                  << "video_scale != 1.0, scale Camera.fx/fy/cx/cy to match.\n"
                  << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    const std::string voc_path = argv[1];
    const std::string cfg_path = argv[2];

    // ORB_SLAM2::System calls exit(-1) from its constructor when it cannot open
    // either file, before any ROS logging exists -- so the user sees a bare
    // "Failed to open settings file" with no context. Check first and say
    // something useful, including the two paths that are wrong in the shipped
    // docs and launch file more often than not.
    if (!Readable(voc_path)) {
        std::cerr << "ERROR: cannot read vocabulary file: " << voc_path << "\n"
                  << "Unpack it with:  tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary/\n"
                  << std::endl;
        rclcpp::shutdown();
        return 2;
    }
    if (!Readable(cfg_path)) {
        std::cerr << "ERROR: cannot read settings file: " << cfg_path << "\n"
                  << "The packaged one lives at "
                     "$(ros2 pkg prefix orbslam2)/share/orbslam2/config.yaml\n"
                  << std::endl;
        rclcpp::shutdown();
        return 2;
    }

    std::cout << "orbslam2: loading vocabulary (this takes ~10-30 s)..." << std::endl;
    ORB_SLAM2::System SLAM(voc_path, cfg_path, ORB_SLAM2::System::MONOCULAR);

    auto node = std::make_shared<MonocularSlamNode>(&SLAM, voc_path, cfg_path);
    rclcpp::spin(node);

    // Persist the trajectory before ROS is torn down, so the shutdown log line
    // still reaches the user.
    char cwd[PATH_MAX];
    const std::string out_path =
        (getcwd(cwd, sizeof(cwd)) != nullptr)
            ? (std::string(cwd) + "/KeyFrameTrajectory.txt")
            : std::string("KeyFrameTrajectory.txt");
    node->ShutdownAndSave(out_path);

    rclcpp::shutdown();
    return 0;
}
