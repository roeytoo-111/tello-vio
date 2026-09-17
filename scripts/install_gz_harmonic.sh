#!/usr/bin/env bash
# Install Gazebo Harmonic (gz-sim8 LTS) + the ROS 2 Humble bridge on
# Ubuntu 22.04 (Jammy) -- Tier B of docs/drone_chasing_rl/
# sim_training_architecture.md. Needs sudo.
#
# Verified facts this script encodes (packages.osrfoundation.org index +
# gazebosim.org/docs/harmonic, read 2026-09-17):
#   * Harmonic binaries support Jammy; the gz-harmonic metapackage brings
#     the Python bindings (python3-gz-transport13, python3-gz-msgs10) that
#     GzChaseEnv needs.
#   * The Humble bridge for Harmonic is ros-humble-ros-gzharmonic, hosted
#     in the SAME osrf repo (not packages.ros.org), and it CONFLICTS with
#     ros-humble-ros-gz / ros-humble-ros-gzgarden (Fortress/Garden
#     variants) -- remove those first if present.
#   * Gazebo Classic 11 (this machine's current install) is EOL and unused
#     by Tier B; it can coexist, nothing here removes it.
set -euo pipefail

if [ "$(lsb_release -cs)" != "jammy" ]; then
    echo "This script targets Ubuntu 22.04 (jammy); got $(lsb_release -cs)." >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y curl lsb-release gnupg

sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update

if dpkg -l ros-humble-ros-gz 2>/dev/null | grep -q '^ii'; then
    echo "Removing conflicting ros-humble-ros-gz (Fortress variant)..."
    sudo apt-get remove -y 'ros-humble-ros-gz*'
fi

sudo apt-get install -y gz-harmonic ros-humble-ros-gzharmonic

echo
echo "=== verify ==="
gz sim --versions
python3 -c "from gz.transport13 import Node; from gz.msgs10.world_control_pb2 import WorldControl; print('gz python bindings OK')"
echo
echo "Next (in this order -- sim_training_architecture.md 3.2):"
echo "  1. python3 -m chase_sim_gz.sign_test      # yaw sign: gz issue #2657 is OPEN"
echo "  2. python3 -m chase_sim_gz.calibrate_lag  # match measured T_lag within 10%"
echo "  3. ros2 launch chase_sim_gz tier_b.launch.py   # full ROS graph rehearsal"
