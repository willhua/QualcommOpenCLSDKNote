#!/bin/bash
#--------------------------------------------------------------------------------------
# File: build_android.sh
# Desc:
#
# Author:      QUALCOMM
#
#               Copyright (c) 2017 QUALCOMM Technologies, Inc.
#                         All Rights Reserved.
#                      QUALCOMM Proprietary/GTDR
#--------------------------------------------------------------------------------------
set -e

if [[ -z "$ANDROID_NDK" ]]; then
   echo "Please set \$ANDROID_NDK to the root of your standalone Android ndk build tree"
   exit 1
fi

if [[ -z "$OPEN_CL_LIB" ]]; then
   echo "Please set \$OPEN_CL_LIB to the path to libOpenCL.so (e.g. /path/to/libOpenCL.so)"
   exit 1
fi

if [[ -z "$ION_INCLUDE_PATH" ]]; then
   echo "Please set \$ION_INCLUDE_PATH to the directory containing ION headers."
   exit 1
fi

if [[ "$#" -lt 1 ]]; then
   echo "Error: invalid number of arguments: $#"
   echo "Usage: $0 <BITNESS> [args-to-pass-to-make]"
   echo "       BITNESS: 32 or 64"
   exit 1
fi

BITNESS=$1

if [[ "64" == "$BITNESS" ]]; then
    ANDROID_ABI="arm64-v8a"
    ANDROID_TOOLCHAIN="aarch64-linux-android-clang"
    NDK_ARCH=arm64
elif [[ "32" == "$BITNESS" ]]; then
    ANDROID_ABI="armeabi-v7a"
    ANDROID_TOOLCHAIN="arm-linux-androideabi-clang"
    NDK_ARCH=arm
else
    echo "Invalid bitness!: $BITNESS"
    exit 1
fi

if [ ! -d "android-cmake" ]; then
    echo "Couldn't find `pwd`/android-cmake, please install it to this directory."
    exit 1
fi

# Creates an android standalone toolchain in this dir, for use with android-cmake
ANDROID_STANDALONE_TOOLCHAIN=`pwd`/android_standalone_toolchain_$ANDROID_ABI
if [ ! -d "$ANDROID_STANDALONE_TOOLCHAIN" ]; then
    $ANDROID_NDK/build/tools/make-standalone-toolchain.sh \
        --install-dir=$ANDROID_STANDALONE_TOOLCHAIN \
        --arch=$NDK_ARCH \
        --platform=android-21 \
        --toolchain=$ANDROID_TOOLCHAIN
fi

BUILD_DIR=bld_android_$BITNESS

mkdir --parents $BUILD_DIR
cd $BUILD_DIR
cmake \
  --debug-trycompile \
  -DANDROID=True \
  -DCMAKE_TOOLCHAIN_FILE=../android-cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DANDROID_SO_UNDEFINED=ON \
  -DANDROID_ABI=$ANDROID_ABI \
  -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
  -DOPEN_CL_LIB=$OPEN_CL_LIB \
  -g "Unix Makefiles" ../

# Passes extra cmd line arguments to make
make "${@:2}"
