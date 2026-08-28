#!/bin/bash
# Install ROS 2 Humble and this project's dependencies on Ubuntu 22.04 (jammy).
#
# The previous version installed ROS 2 Foxy (Ubuntu 20.04) into a repository
# that targets Humble, using the deprecated apt-key mechanism. Both are fixed.
set -euo pipefail

SUDO=""
if (( EUID != 0 )); then SUDO="sudo"; fi

CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-}")"
if [[ "${CODENAME}" != "jammy" ]]; then
  echo " ! This repository targets Ubuntu 22.04 (jammy) + ROS 2 Humble."
  echo " ! Detected '${CODENAME}'. Continuing, but expect package mismatches."
fi

echo " - Locale"
$SUDO apt update
$SUDO apt install -y locales curl gnupg2 lsb-release software-properties-common
$SUDO locale-gen en_US en_US.UTF-8
$SUDO update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo " - ROS 2 apt source (signed-by keyring; apt-key is deprecated)"
$SUDO add-apt-repository -y universe
$SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${CODENAME} main" | \
  $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo " - ROS 2 Humble"
$SUDO apt update
$SUDO apt install -y \
  ros-humble-desktop ros-dev-tools \
  ros-humble-cv-bridge ros-humble-vision-opencv \
  ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
  ros-humble-camera-calibration ros-humble-camera-calibration-parsers \
  ros-humble-camera-info-manager ros-humble-image-transport \
  ros-humble-rqt-image-view ros-humble-rqt-robot-monitor \
  ros-humble-diagnostic-msgs \
  python3-colcon-common-extensions python3-rosdep python3-argcomplete \
  python3-numpy python3-opencv python3-yaml \
  build-essential cmake git

echo " - rosdep"
$SUDO rosdep init 2>/dev/null || true
rosdep update

echo " - Python dependencies (user install; use a venv if you prefer)"
python3 -m pip install --user --upgrade djitellopy av

if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc 2>/dev/null; then
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  echo " - Added the Humble setup to ~/.bashrc"
fi

echo
echo "Done. Open a new shell (or 'source /opt/ros/humble/setup.bash'), then:"
echo "  ./scripts/build.sh"
