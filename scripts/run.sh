#!/bin/bash
# Launch the full VIO stack. Paths resolved from this script, not from $PWD.
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


if [[ ! -f "${REPO}/install/setup.bash" ]]; then
  echo "ERROR: workspace not built. Run ./scripts/build.sh first." >&2
  exit 1
fi
ros_source "${REPO}/install/setup.bash"

exec ros2 launch tello_vio vio.launch.py "$@"
