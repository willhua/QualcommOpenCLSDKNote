LOCAL_PATH := $(call my-dir)

CHECK_VERSION_GE = $(shell if [ $(1) -ge $(2) ] ; then echo true ; else echo false ; fi)

# Tries to determine whether to use libion or ion kernel uapi
KERNEL_VERSION = $(shell ls kernel | sed -n 's/msm-\([0-9]\+\)\.\([0-9]\+\)/-v x0=\1 -v x1=\2/p')
USE_LIBION = $(shell awk $(KERNEL_VERSION) -v y0="4" -v y1="12" 'BEGIN {printf (x0>=y0 && x1>=y1?"true":"false") "\n"}')
ifeq ($(USE_LIBION), true)
    $(info OpenCL SDK: Using libion)

    OPENCL_SDK_CPPFLAGS := -Wno-missing-braces -DUSES_LIBION

    OPENCL_SDK_SHARED_LIBS := libion libOpenCL

    OPENCL_SDK_COMMON_INCLUDES := \
        $(LOCAL_PATH)/src \
        kernel/msm-4.14/ \
        $(TARGET_OUT_INTERMEDIATES)/include/adreno

else
    $(info OpenCL SDK: Using ion uapi)

    OPENCL_SDK_CPPFLAGS := -Wno-missing-braces

    OPENCL_SDK_SHARED_LIBS := libOpenCL

    OPENCL_SDK_COMMON_INCLUDES := \
        $(LOCAL_PATH)/src \
        $(TARGET_OUT_INTERMEDIATES)/KERNEL_OBJ/usr/include \
        $(TARGET_OUT_INTERMEDIATES)/include/adreno

endif

OPENCL_SDK_SRC_FILES := \
    src/util/cl_wrapper.cpp \
    src/util/half_float.cpp \
    src/util/util.cpp

#########################
# compressed_image_nv12 #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := compressed_image_nv12

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/compressed_image_nv12.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

########################
# qcom_block_match_sad #
########################
include $(CLEAR_VARS)
LOCAL_MODULE := qcom_block_match_sad

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/qcom_block_match_sad.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

########################
# qcom_block_match_ssd #
########################
include $(CLEAR_VARS)
LOCAL_MODULE := qcom_block_match_ssd

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/qcom_block_match_ssd.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#########################
# qcom_box_filter_image #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := qcom_box_filter_image

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/qcom_box_filter_image.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#######################
# qcom_convolve_image #
#######################
include $(CLEAR_VARS)
LOCAL_MODULE := qcom_convolve_image

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/qcom_convolve_image.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###########################
# accelerated_convolution #
###########################
include $(CLEAR_VARS)
LOCAL_MODULE := accelerated_convolution

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/convolutions/accelerated_convolution.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###############
# convolution #
###############
include $(CLEAR_VARS)
LOCAL_MODULE := convolution

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/convolutions/convolution.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

####################################
# compressed_nv12_vector_image_ops #
####################################
include $(CLEAR_VARS)
LOCAL_MODULE := compressed_nv12_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/compressed_nv12_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

####################################
# compressed_p010_vector_image_ops #
####################################
include $(CLEAR_VARS)
LOCAL_MODULE := compressed_p010_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/compressed_p010_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

####################################
# compressed_tp10_vector_image_ops #
####################################
include $(CLEAR_VARS)
LOCAL_MODULE := compressed_tp10_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/compressed_tp10_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#########################
# nv12_vector_image_ops #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := nv12_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/nv12_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#########################
# p010_vector_image_ops #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := p010_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/p010_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#########################
# tp10_vector_image_ops #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := tp10_vector_image_ops

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/vector_image_ops/tp10_vector_image_ops.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###############
# hello_world #
###############
include $(CLEAR_VARS)
LOCAL_MODULE := hello_world

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/hello_world.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###########################
# p010_to_compressed_tp10 #
###########################
include $(CLEAR_VARS)
LOCAL_MODULE := p010_to_compressed_tp10

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/conversions/p010_to_compressed_tp10.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###########################
# nv12_to_rgba #
###########################
include $(CLEAR_VARS)
LOCAL_MODULE := nv12_to_rgba

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/conversions/nv12_to_rgba.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###################
# matrix_addition #
###################
include $(CLEAR_VARS)
LOCAL_MODULE := matrix_addition

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/matrix_addition.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###############################
# image_matrix_multiplication #
###############################
include $(CLEAR_VARS)
LOCAL_MODULE := image_matrix_multiplication

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/image_matrix_multiplication.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

################################
# buffer_matrix_multiplication #
################################
include $(CLEAR_VARS)
LOCAL_MODULE := buffer_matrix_multiplication

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/buffer_matrix_multiplication.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###########################
# buffer_matrix_transpose #
###########################
include $(CLEAR_VARS)
LOCAL_MODULE := buffer_matrix_transpose

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/buffer_matrix_transpose.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

##########################
# image_matrix_transpose #
##########################
include $(CLEAR_VARS)
LOCAL_MODULE := image_matrix_transpose

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/image_matrix_transpose.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

########################
# bayer_mipi10_to_rgba #
########################
include $(CLEAR_VARS)
LOCAL_MODULE := bayer_mipi10_to_rgba

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/bayer_mipi/bayer_mipi10_to_rgba.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

######################
# mipi10_to_unpacked #
######################
include $(CLEAR_VARS)
LOCAL_MODULE := mipi10_to_unpacked

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/bayer_mipi/mipi10_to_unpacked.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#################################
# unpacked_bayer_to_rgba #
#################################
include $(CLEAR_VARS)
LOCAL_MODULE := unpacked_bayer_to_rgba

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/bayer_mipi/unpacked_bayer_to_rgba.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

######################
# unpacked_to_mipi10 #
######################
include $(CLEAR_VARS)
LOCAL_MODULE := unpacked_to_mipi10

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/bayer_mipi/unpacked_to_mipi10.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

####################################
# image_matrix_multiplication_half #
####################################
include $(CLEAR_VARS)
LOCAL_MODULE := image_matrix_multiplication_half

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/image_matrix_multiplication_half.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#####################################
# buffer_matrix_multiplication_half #
#####################################
include $(CLEAR_VARS)
LOCAL_MODULE := buffer_matrix_multiplication_half

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/linear_algebra/buffer_matrix_multiplication_half.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#############
# fft_image #
#############
include $(CLEAR_VARS)
LOCAL_MODULE := fft_image

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/fft/fft_image.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

##############
# fft_matrix #
##############
include $(CLEAR_VARS)
LOCAL_MODULE := fft_matrix

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/fft/fft_matrix.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

###########################
# io_coherent_ion_buffers #
###########################
include $(CLEAR_VARS)
LOCAL_MODULE := io_coherent_ion_buffers

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/io_coherent_ion/io_coherent_ion_buffers.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

##########################
# io_coherent_ion_images #
##########################
include $(CLEAR_VARS)
LOCAL_MODULE := io_coherent_ion_images

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/io_coherent_ion/io_coherent_ion_images.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

#########################
# compressed_image_rgba #
#########################
include $(CLEAR_VARS)
LOCAL_MODULE := compressed_image_rgba

LOCAL_SRC_FILES := \
    $(OPENCL_SDK_SRC_FILES) \
    src/examples/basic/compressed_image_rgba.cpp

LOCAL_CPPFLAGS         := $(OPENCL_SDK_CPPFLAGS)
LOCAL_SHARED_LIBRARIES := $(OPENCL_SDK_SHARED_LIBS)
LOCAL_C_INCLUDES       := $(OPENCL_SDK_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)