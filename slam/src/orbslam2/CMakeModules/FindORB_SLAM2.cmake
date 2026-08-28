#
# Find ORB_SLAM2 library and headers.
#
# This repo's install script (`scripts/orbslam.sh`) exports ORB_SLAM2_ROOT_DIR.
# We use that as the primary hint, but also fall back to common prefixes.
#
# Sets:
#   ORB_SLAM2_FOUND
#   ORB_SLAM2_INCLUDE_DIRS
#   ORB_SLAM2_LIBRARIES
#

include(FindPackageHandleStandardArgs)

set(_ORB_SLAM2_HINTS "")

if(DEFINED ORB_SLAM2_ROOT_DIR AND NOT ORB_SLAM2_ROOT_DIR STREQUAL "")
  list(APPEND _ORB_SLAM2_HINTS "${ORB_SLAM2_ROOT_DIR}")
endif()

if(DEFINED ENV{ORB_SLAM2_ROOT_DIR} AND NOT "$ENV{ORB_SLAM2_ROOT_DIR}" STREQUAL "")
  list(APPEND _ORB_SLAM2_HINTS "$ENV{ORB_SLAM2_ROOT_DIR}")
endif()

list(APPEND _ORB_SLAM2_HINTS
  "/usr/local"
  "/usr"
  "/opt"
)

find_path(_ORB_SLAM2_ROOT
  # We need the project root (some headers include Thirdparty/* with relative paths).
  NAMES Thirdparty/DBoW2/DBoW2/BowVector.h
  HINTS ${_ORB_SLAM2_HINTS}
)

find_path(_ORB_SLAM2_INCLUDE
  # Different forks install headers either as:
  #   include/ORB_SLAM2/System.h   (wrapper-style)
  # or
  #   include/System.h            (upstream-style)
  NAMES ORB_SLAM2/System.h System.h
  HINTS ${_ORB_SLAM2_HINTS}
  PATH_SUFFIXES include Include
)

set(ORB_SLAM2_INCLUDE_DIRS "")
if(_ORB_SLAM2_ROOT)
  list(APPEND ORB_SLAM2_INCLUDE_DIRS "${_ORB_SLAM2_ROOT}")
endif()
if(_ORB_SLAM2_INCLUDE)
  list(APPEND ORB_SLAM2_INCLUDE_DIRS "${_ORB_SLAM2_INCLUDE}")
  # NAMES above accepts either layout, but the wrapper sources include
  # "System.h" flatly. For a fork that installs headers under
  # include/ORB_SLAM2/, adding only the parent leaves those includes
  # unresolvable: configure succeeds and the *compile* fails, which is a
  # confusing place to discover a search-path problem. Add the nested
  # directory too so both layouts actually work.
  if(EXISTS "${_ORB_SLAM2_INCLUDE}/ORB_SLAM2/System.h")
    list(APPEND ORB_SLAM2_INCLUDE_DIRS "${_ORB_SLAM2_INCLUDE}/ORB_SLAM2")
  endif()
endif()

find_library(ORB_SLAM2_LIBRARIES
  NAMES ORB_SLAM2 libORB_SLAM2 orbslam2 liborbslam2
  HINTS ${_ORB_SLAM2_HINTS}
  PATH_SUFFIXES lib lib64 build/lib build/lib64
)

# ORB_SLAM2 is typically built together with Thirdparty/DBoW2 and Thirdparty/g2o.
# The wrapper executable may need to link these DSOs explicitly.
find_library(DBOW2_LIBRARY
  NAMES DBoW2 libDBoW2
  HINTS ${_ORB_SLAM2_HINTS}
  PATH_SUFFIXES lib lib64 build/lib build/lib64
)

find_library(G2O_LIBRARY
  NAMES g2o libg2o
  HINTS ${_ORB_SLAM2_HINTS}
  PATH_SUFFIXES Thirdparty/g2o/lib Thirdparty/g2o/build/lib Thirdparty/g2o/build/lib64
)

if(ORB_SLAM2_LIBRARIES)
  set(_ORB_SLAM2_ALL_LIBS "${ORB_SLAM2_LIBRARIES}")
  if(DBOW2_LIBRARY)
    list(APPEND _ORB_SLAM2_ALL_LIBS "${DBOW2_LIBRARY}")
  endif()
  if(G2O_LIBRARY)
    list(APPEND _ORB_SLAM2_ALL_LIBS "${G2O_LIBRARY}")
  endif()
  set(ORB_SLAM2_LIBRARIES "${_ORB_SLAM2_ALL_LIBS}")
endif()

find_package_handle_standard_args(ORB_SLAM2
  REQUIRED_VARS ORB_SLAM2_INCLUDE_DIRS ORB_SLAM2_LIBRARIES
)

mark_as_advanced(ORB_SLAM2_INCLUDE_DIRS ORB_SLAM2_LIBRARIES)

