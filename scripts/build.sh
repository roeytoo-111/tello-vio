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

# Static analysis before building. py_compile only checks SYNTAX, so an
# undefined name (e.g. `np.` where the module imports `numpy`) sails through
# it, installs cleanly, and raises NameError at runtime on the drone. pyflakes
# catches that without importing the module, which matters because the ROS
# nodes cannot be imported outside a ROS context.
if python3 -c "import pyflakes" 2>/dev/null; then
  echo " - Static analysis (pyflakes)"
  PYFILES=$(find "${REPO}/workspace/src" "${REPO}/scripts" "${REPO}/docs" \
      -name '*.py' -not -path '*/build/*' -not -path '*/install/*' \
      -not -path '*/__pycache__/*' 2>/dev/null)
  if [[ -n "${PYFILES}" ]]; then
    # shellcheck disable=SC2086
    if ! python3 -m pyflakes ${PYFILES}; then
      echo " ! pyflakes reported problems (see above). Fix them before flying:" >&2
      echo " !   an undefined name here becomes a crash on the drone." >&2
      exit 1
    fi
  fi
else
  echo " - Skipping static analysis (pip install pyflakes to enable)"
fi

echo " - Building"
if [[ -n "${PACKAGES}" ]]; then
  # shellcheck disable=SC2086
  colcon build --symlink-install --packages-select ${PACKAGES}
else
  colcon build --symlink-install
fi

# Construct the driver against a stub drone. py_compile and pyflakes both pass
# on code that raises at RUNTIME (a bad attribute, a wrong constructor arg), and
# the ROS nodes cannot be imported inside the normal pytest run because ROS's
# pytest plugins conflict with pytest 9. Running it here, after the build and
# with the workspace sourced, is the one place this check fits.
if [[ -f "${REPO}/install/setup.bash" ]]; then
  echo " - Driver construction check"
  # shellcheck disable=SC1091
  ( ros_source "${REPO}/install/setup.bash"
    if ! python3 "${REPO}/scripts/driver_ctor_check.py" > /tmp/ctor_check.log 2>&1; then
      echo " ! Driver failed to construct. It would die on launch:" >&2
      tail -15 /tmp/ctor_check.log >&2
      exit 1
    fi
    echo "   OK" ) || exit 1
fi

echo
echo "Done. Activate with:"
echo "  source ${REPO}/install/setup.bash"
