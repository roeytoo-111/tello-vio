#!/bin/bash
# Build the workspace. Run from anywhere; paths are resolved from this script.
#
# The previous version did `cd ../workspace && rm -rf build install log`, which
# depended entirely on the caller's working directory: run it from the wrong
# place and it recursively deleted three directories somewhere else. Paths are
# now absolute, and the destructive step is opt-in.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ROS's setup.bash references unset variables (AMENT_TRACE_SETUP_FILES,
# AMENT_CURRENT_PREFIX, ...), which `set -u` turns into a fatal error. Sourcing
# it is the one place we have to relax nounset.
ros_source() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

CLEAN="${CLEAN:-0}"
PACKAGES="${PACKAGES:-}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    ros_source /opt/ros/humble/setup.bash
  else
    echo "ERROR: no ROS 2 environment. source /opt/ros/humble/setup.bash first." >&2
    exit 1
  fi
fi
echo " - ROS_DISTRO=${ROS_DISTRO}"

cd "${REPO}"

if [[ "${CLEAN}" == "1" ]]; then
  echo " - CLEAN=1: removing ${REPO}/{build,install,log}"
  rm -rf "${REPO}/build" "${REPO}/install" "${REPO}/log"
fi

echo " - Resolving dependencies"
rosdep install -i --from-paths workspace/src slam/src --rosdistro "${ROS_DISTRO}" -y || \
  echo " ! rosdep reported problems; continuing (ORB-SLAM2 is not a rosdep key)"

# ORB_SLAM2_ROOT_DIR is optional: without it the orbslam2 package configures,
# warns, and skips building its node, so the rest of the workspace still builds.
if [[ -n "${ORB_SLAM2_ROOT_DIR:-}" ]]; then
  echo " - ORB_SLAM2_ROOT_DIR=${ORB_SLAM2_ROOT_DIR}"
elif [[ -d "${REPO}/libs/ORB_SLAM2" ]]; then
  export ORB_SLAM2_ROOT_DIR="${REPO}/libs/ORB_SLAM2"
  echo " - ORB_SLAM2_ROOT_DIR=${ORB_SLAM2_ROOT_DIR} (auto-detected)"
else
  echo " - ORB-SLAM2 not present; the orbslam2 node will be skipped"
fi

echo " - Building"
if [[ -n "${PACKAGES}" ]]; then
  # shellcheck disable=SC2086
  colcon build --symlink-install --packages-select ${PACKAGES}
else
  colcon build --symlink-install
fi

echo
echo "Done. Activate with:"
echo "  source ${REPO}/install/setup.bash"
