set(TARGET apq8053)
set(ARCH 64)

INCLUDE(CMakeForceCompiler)
set(CMAKE_SYSTEM_NAME Linux)

set(OE_HOST_SYSROOT "poky/build/tmp-glibc/sysroots/x86_64-linux")

if(${ARCH} STREQUAL "32")
set(OE_TARGET_SYSROOT "$ENV{OE_ROOT}/poky/build/tmp-glibc/sysroots/lib32-${TARGET}")
set(CMAKE_C_COMPILER ${OE_ROOT}/${OE_HOST_SYSROOT}/usr/bin/arm-oemllib32-linux-gnueabi/arm-oemllib32-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER ${OE_ROOT}/${OE_HOST_SYSROOT}/usr/bin/arm-oemllib32-linux-gnueabi/arm-oemllib32-linux-gnueabi-g++)
set(BIT_FLAGS "-mcpu=cortex-a15 -mfloat-abi=softfp -mfpu=neon")
else()
set(OE_TARGET_SYSROOT "$ENV{OE_ROOT}/poky/build/tmp-glibc/sysroots/${TARGET}")
set(CMAKE_C_COMPILER ${OE_ROOT}/${OE_HOST_SYSROOT}/usr/bin/aarch64-oe-linux/aarch64-oe-linux-gcc)
set(CMAKE_CXX_COMPILER ${OE_ROOT}/${OE_HOST_SYSROOT}/usr/bin/aarch64-oe-linux/aarch64-oe-linux-g++)
set(BIT_FLAGS "")
endif()

set(CMAKE_SYSROOT ${OE_TARGET_SYSROOT})
set(SYSROOT "--sysroot=${OE_TARGET_SYSROOT}")
set(INC_DIR "-I${OE_TARGET_SYSROOT}/usr/include")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SYSROOT} ${INC_DIR} ${BIT_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYSROOT} ${INC_DIR} ${BIT_FLAGS}" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH ${OE_ROOT}/${OE_HOST_SYSROOT}/usr/bin)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
