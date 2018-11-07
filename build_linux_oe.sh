if ! [[ "$OSTYPE" =~ linux ]]; then
  echo "Not a Linux System "
  exit 1
fi

if [[ -z "$OE_ROOT" ]]; then
   echo "Please set \$OE_ROOT to your root directory"
   exit 1
fi

#if [[ -z "$OPEN_CL_LIB" ]]; then
#   echo "Please set \$OPEN_CL_LIB"
#   exit 1
#fi

if [[ "$#" -lt 2 ]]; then
   echo "Error: invalid number of arguments: $#"
   echo "Usage: $0 <OE_TARGET> <BITNESS> [args-to-pass-to-make]"
   echo "       OE_TARGET: target platform (8053, 8096, 8074)"
   echo "       BITNESS: 32 or 64"
   exit 1
fi

OE_TARGET=$1
ARCH=$2

if [[ "64" == "$ARCH" ]]; then
   ARM_ARCH="arm64v8"
elif [[ "32" == "$ARCH" ]]; then
   ARM_ARCH="armv7"
else
   echo "Invalid bitness!: $ARCH"
   exit 1
fi

OE_TOOLCHAIN_FILE="toolchain/linux_embedded/linux_embedded-toolchain.cmake"
if [[ ! -f $OE_TOOLCHAIN_FILE ]]; then
    echo "Can't find toolchain file: $OE_TOOLCHAIN_FILE"
    exit 1
fi

BUILD_DIR=bld_linux_oe_$ARCH

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake  \
   --debug-trycompile \
   -DLINUX_OE=YES \
   -DCMAKE_TOOLCHAIN_FILE=$OE_TOOLCHAIN_FILE \
   -DOE_ROOT=$OE_ROOT \
   -DCMAKE_ARM_COMPILER=YES \
   -DNEEDS_TO_LINK_PTHREAD=YES \
   -DARCH=$ARCH \
   -DOPEN_CL_LIB=$OPEN_CL_LIB \
   -g "Unix Makefiles" ../
make "${@:3}"
