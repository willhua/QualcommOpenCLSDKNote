#--------------------------------------------------------------------------------------
# File: CMakeLists.txt
# Desc:
#
# Author:      QUALCOMM
#
#               Copyright (c) 2017 QUALCOMM Technologies, Inc.
#                         All Rights Reserved.
#                      QUALCOMM Proprietary/GTDR
#--------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(sdk_examples)

set(CMAKE_CXX_STANDARD 11) # CMAKE_CXX_STANDARD exists for cmake 3.1 and later
if (CMAKE_VERSION VERSION_LESS "3.1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x") # This is more or less equivalent to the above for older cmake
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

set(COMMON_SOURCE_FILES
        src/util/util.h
        src/util/util.cpp
        src/util/half_float.h
        src/util/half_float.cpp
        src/util/cl_wrapper.h
        src/util/cl_wrapper.cpp
        )

if(ANDROID)
if("${ION_INCLUDE_PATH}" STREQUAL "")
    set(ION_INCLUDE_PATH "$ENV{ION_INCLUDE_PATH}")
endif()

if("${ION_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "Please set CMake variable ION_INCLUDE_PATH")
endif()

message("Using ${ION_INCLUDE_PATH} as include path for ION headers")

add_definitions(-DANDROID -DUSES_ANDROID_CMAKE)
set(ION_INCLUDE_PATH ${ANDROID_STANDALONE_TOOLCHAIN}/sysroot/usr/include ${ION_INCLUDE_PATH})
endif() #ANDROID

include_directories(
        src
        inc
        ${ION_INCLUDE_PATH}
)

if("${OPEN_CL_LIB}" STREQUAL "")
    message(FATAL_ERROR "Can't find libOpenCL.so, please set the CMake variable OPEN_CL_LIB to /path/to/libOpenCL.so.")
endif()

add_executable(qcom_box_filter_image ${COMMON_SOURCE_FILES} src/examples/basic/qcom_box_filter_image.cpp)
add_executable(qcom_convolve_image   ${COMMON_SOURCE_FILES} src/examples/basic/qcom_convolve_image.cpp)
add_executable(qcom_block_match_sad ${COMMON_SOURCE_FILES} src/examples/basic/qcom_block_match_sad.cpp)
add_executable(qcom_block_match_ssd ${COMMON_SOURCE_FILES} src/examples/basic/qcom_block_match_ssd.cpp)
add_executable(accelerated_convolution ${COMMON_SOURCE_FILES} src/examples/convolutions/accelerated_convolution.cpp)
add_executable(convolution ${COMMON_SOURCE_FILES} src/examples/convolutions/convolution.cpp)
add_executable(compressed_image_nv12 ${COMMON_SOURCE_FILES} src/examples/basic/compressed_image_nv12.cpp)
add_executable(nv12_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/nv12_vector_image_ops.cpp)
add_executable(tp10_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/tp10_vector_image_ops.cpp)
add_executable(p010_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/p010_vector_image_ops.cpp)
add_executable(compressed_nv12_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/compressed_nv12_vector_image_ops.cpp)
add_executable(compressed_p010_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/compressed_p010_vector_image_ops.cpp)
add_executable(compressed_tp10_vector_image_ops ${COMMON_SOURCE_FILES} src/examples/vector_image_ops/compressed_tp10_vector_image_ops.cpp)
add_executable(hello_world ${COMMON_SOURCE_FILES} src/examples/basic/hello_world.cpp)
add_executable(p010_to_compressed_tp10 ${COMMON_SOURCE_FILES} src/examples/conversions/p010_to_compressed_tp10.cpp)
add_executable(nv12_to_rgba ${COMMON_SOURCE_FILES} src/examples/conversions/nv12_to_rgba.cpp)
add_executable(matrix_addition ${COMMON_SOURCE_FILES} src/examples/linear_algebra/matrix_addition.cpp)
add_executable(image_matrix_multiplication ${COMMON_SOURCE_FILES} src/examples/linear_algebra/image_matrix_multiplication.cpp)
add_executable(buffer_matrix_multiplication ${COMMON_SOURCE_FILES} src/examples/linear_algebra/buffer_matrix_multiplication.cpp)
add_executable(buffer_matrix_transpose ${COMMON_SOURCE_FILES} src/examples/linear_algebra/buffer_matrix_transpose.cpp)
add_executable(image_matrix_transpose ${COMMON_SOURCE_FILES} src/examples/linear_algebra/image_matrix_transpose.cpp)
add_executable(bayer_mipi10_to_rgba ${COMMON_SOURCE_FILES} src/examples/bayer_mipi/bayer_mipi10_to_rgba.cpp)
add_executable(mipi10_to_unpacked ${COMMON_SOURCE_FILES} src/examples/bayer_mipi/mipi10_to_unpacked.cpp)
add_executable(unpacked_bayer_to_rgba ${COMMON_SOURCE_FILES} src/examples/bayer_mipi/unpacked_bayer_to_rgba.cpp)
add_executable(unpacked_to_mipi10 ${COMMON_SOURCE_FILES} src/examples/bayer_mipi/unpacked_to_mipi10.cpp)
add_executable(fft_image ${COMMON_SOURCE_FILES} src/examples/fft/fft_image.cpp)
add_executable(fft_matrix ${COMMON_SOURCE_FILES} src/examples/fft/fft_matrix.cpp)
add_executable(image_matrix_multiplication_half ${COMMON_SOURCE_FILES} src/examples/linear_algebra/image_matrix_multiplication_half.cpp)
add_executable(buffer_matrix_multiplication_half ${COMMON_SOURCE_FILES} src/examples/linear_algebra/buffer_matrix_multiplication_half.cpp)
add_executable(io_coherent_ion_buffers ${COMMON_SOURCE_FILES} src/examples/io_coherent_ion/io_coherent_ion_buffers.cpp)
add_executable(io_coherent_ion_images ${COMMON_SOURCE_FILES} src/examples/io_coherent_ion/io_coherent_ion_images.cpp)
add_executable(compressed_image_rgba ${COMMON_SOURCE_FILES} src/examples/basic/compressed_image_rgba.cpp)

target_link_libraries(qcom_box_filter_image ${OPEN_CL_LIB})
target_link_libraries(qcom_convolve_image ${OPEN_CL_LIB})
target_link_libraries(qcom_block_match_sad ${OPEN_CL_LIB})
target_link_libraries(qcom_block_match_ssd ${OPEN_CL_LIB})
target_link_libraries(accelerated_convolution ${OPEN_CL_LIB})
target_link_libraries(convolution ${OPEN_CL_LIB})
target_link_libraries(compressed_image_nv12 ${OPEN_CL_LIB})
target_link_libraries(nv12_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(tp10_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(p010_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(compressed_nv12_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(compressed_p010_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(compressed_tp10_vector_image_ops ${OPEN_CL_LIB})
target_link_libraries(hello_world ${OPEN_CL_LIB})
target_link_libraries(p010_to_compressed_tp10 ${OPEN_CL_LIB})
target_link_libraries(nv12_to_rgba ${OPEN_CL_LIB})
target_link_libraries(matrix_addition ${OPEN_CL_LIB})
target_link_libraries(image_matrix_multiplication ${OPEN_CL_LIB})
target_link_libraries(buffer_matrix_multiplication ${OPEN_CL_LIB})
target_link_libraries(buffer_matrix_transpose ${OPEN_CL_LIB})
target_link_libraries(image_matrix_transpose ${OPEN_CL_LIB})
target_link_libraries(bayer_mipi10_to_rgba ${OPEN_CL_LIB})
target_link_libraries(mipi10_to_unpacked ${OPEN_CL_LIB})
target_link_libraries(unpacked_bayer_to_rgba ${OPEN_CL_LIB})
target_link_libraries(unpacked_to_mipi10 ${OPEN_CL_LIB})
target_link_libraries(fft_image ${OPEN_CL_LIB})
target_link_libraries(fft_matrix ${OPEN_CL_LIB})
target_link_libraries(image_matrix_multiplication_half ${OPEN_CL_LIB})
target_link_libraries(buffer_matrix_multiplication_half ${OPEN_CL_LIB})
target_link_libraries(io_coherent_ion_buffers ${OPEN_CL_LIB})
target_link_libraries(io_coherent_ion_images ${OPEN_CL_LIB})
target_link_libraries(compressed_image_rgba ${OPEN_CL_LIB})
