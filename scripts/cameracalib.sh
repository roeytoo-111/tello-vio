#!/bin/bash
# Camera intrinsic calibration for the Tello, on ROS 2 Humble.
#
# Usage:
#   ./scripts/cameracalib.sh              # 8x6 inner corners, 25 mm squares
#   SIZE=7x9 SQUARE=0.020 ./scripts/cameracalib.sh
#
# --size is the number of INNER corners, not squares: a board with 9x7 squares
# has 8x6 inner corners. Getting this wrong is the most common reason the tool
# never detects the board.
#
# --square is the printed square edge in METRES. Measure it on the actual
# printout: printer scaling of a few percent maps straight into a few percent
# of scale error in everything downstream.
set -euo pipefail

SIZE="${SIZE:-8x6}"
SQUARE="${SQUARE:-0.025}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  # `set -u` vs ROS's setup.bash: it reads unset variables, so relax nounset
  # for the duration of the source.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

if ! ros2 pkg prefix camera_calibration >/dev/null 2>&1; then
  echo " - Installing camera_calibration"
  sudo apt update
  sudo apt install -y "ros-${ROS_DISTRO}-camera-calibration" \
                      "ros-${ROS_DISTRO}-camera-calibration-parsers" \
                      "ros-${ROS_DISTRO}-camera-info-manager"
fi

cat <<'TIPS'
Collect views that actually constrain the model:
  - fill the frame corners, not just the centre (that is what fits distortion)
  - vary distance across the usable range
  - TILT the board substantially -- a board held flat and parallel gives a
    confident, wrong focal length because focal length and distance trade off
  - keep it still at each capture; the Tello's rolling shutter smears motion
Press CALIBRATE when all four bars are green, then SAVE.
Results land in /tmp/calibrationdata.tar.gz -> copy ost.yaml over
workspace/src/tello/resource/ost.yaml and rebuild.
TIPS

exec ros2 run camera_calibration cameracalibrator \
  --size "${SIZE}" --square "${SQUARE}" \
  --no-service-check \
  image:=/image_raw camera:=/camera
