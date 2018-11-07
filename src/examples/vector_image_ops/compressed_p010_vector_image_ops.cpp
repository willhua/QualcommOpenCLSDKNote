//--------------------------------------------------------------------------------------
// File: compressed_p010_vector_image_ops.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//
// Brief:
//      This example is included to demonstrate some functions which work only for
//      compressed P010 format (as opposed to uncompressed P010 images).
//      Comparison of compressed images byte-for-byte is a tricky business. To
//      compare two such images one should first convert them to an uncompressed
//      format, then compare them byte-for-byte.
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <cstring>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>
#include <array>

static const char *HELP_MESSAGE = "\n"
"Usage: compressed_p010_vector_image_ops <source image data file>\n"
"\n"
"Demonstrates use of vector image ops using Qualcomm extensions to OpenCL.\n"
"These functions read and write several pixels at once.\n"
"This example has several kernels which simply copy an image, and verifies each\n"
"result at runtime, indicating on std err if there is an error.\n"
"It takes an uncompressed image input, writes to a compressed image using\n"
"vector ops, and then uncompresses the image for comparison with the original.\n"
"There is no output image.\n";

// These examples use floating point valued functions, but half-float equivalents
// may be used in a completely analogous way.
static const char *PROGRAM_SOURCE[] = {
// The image format determines how the 4th argument to vector read operations is interpreted.
// For YUV images:
"static const int YUV_Y_PLANE = 0;\n",
"static const int YUV_U_PLANE = 1;\n",
"static const int YUV_V_PLANE = 2;\n",
"\n",
// For Y-only images:
"static const int Y_Y_PLANE = 0;\n",
"\n",
// For UV-only images:
"static const int UV_U_PLANE = 0;\n",
"static const int UV_V_PLANE = 1;\n",
"\n",
// Reads 2x2 from a YUV image and writes 2x1 to a Y-only image
"__kernel void read_yuv_2x2_write_y_2x1(__read_only  image2d_t src_image,\n",
"                                       __write_only image2d_t dest_image_y_plane,\n",
"                                                    sampler_t sampler)\n",
"{\n",
"    const int    wid_x              = get_global_id(0);\n",
"    const int    wid_y              = get_global_id(1);\n",
"    const float2 read_coord         = (float2)(2 * wid_x, 2 * wid_y) + 0.5;\n",
"    const int2   write_coord        = (int2)(2 * wid_x, 2 * wid_y);\n",
"    const float4 y_pixels_in        = qcom_read_imagef_2x2(src_image, sampler, read_coord, YUV_Y_PLANE);\n",
"    float        y_pixels_out[2][2] = {\n",
"       {y_pixels_in.s3, y_pixels_in.s2},\n",
"       {y_pixels_in.s0, y_pixels_in.s1},\n",
"       };\n",
"    qcom_write_imagefv_2x1_n10p00(dest_image_y_plane, write_coord,                y_pixels_out[0]);\n",
"    qcom_write_imagefv_2x1_n10p00(dest_image_y_plane, write_coord + (int2)(0, 1), y_pixels_out[1]);\n",
"}\n",
"\n",
// Reads 2x2 from a YUV image and writes 2x1 to a UV-only image
"__kernel void read_yuv_2x2_write_uv_2x1(__read_only  image2d_t src_image,\n",
"                                        __write_only image2d_t dest_image_y_plane,\n",
"                                                     sampler_t sampler)\n",
"{\n",
"    const int    wid_x              = get_global_id(0);\n",
"    const int    wid_y              = get_global_id(1);\n",
"    const float2 read_coord         = 2. * ((float2)(2 * wid_x, 2 * wid_y) + 0.5);\n",
"    const int2   write_coord        = (int2)(2 * wid_x, 2 * wid_y);\n",
"    const float4 u_pixels_in        = qcom_read_imagef_2x2(src_image, sampler, read_coord, YUV_U_PLANE);\n",
"    const float4 v_pixels_in        = qcom_read_imagef_2x2(src_image, sampler, read_coord, YUV_V_PLANE);\n",
"    float2       uv_pixels_out[2][2] = {\n",
"       {{u_pixels_in.s3, v_pixels_in.s3}, {u_pixels_in.s2, v_pixels_in.s2}},\n",
"       {{u_pixels_in.s0, v_pixels_in.s0}, {u_pixels_in.s1, v_pixels_in.s1}},\n",
"       };\n",
"    qcom_write_imagefv_2x1_n10p01(dest_image_y_plane, write_coord,                uv_pixels_out[0]);\n",
"    qcom_write_imagefv_2x1_n10p01(dest_image_y_plane, write_coord + (int2)(0, 1), uv_pixels_out[1]);\n",
"}\n",
"\n",
// Reads 2x2 from a UV-only image and writes 2x1 to a UV-only image
"__kernel void read_uv_2x2_write_uv_2x1(__read_only  image2d_t src_image_uv_plane,\n",
"                                       __write_only image2d_t dest_image_uv_plane,\n",
"                                                    sampler_t sampler)\n",
"{\n",
"    const int    wid_x           = get_global_id(0);\n",
"    const int    wid_y           = get_global_id(1);\n",
"    const float2 read_coord      = (float2)(2 * wid_x, 2 * wid_y) + 0.5;\n",
"    const int2   write_coord     = (int2)(2 * wid_x, 2 * wid_y);\n",
"    const float4 u_pixels_in     = qcom_read_imagef_2x2(src_image_uv_plane, sampler, read_coord, UV_U_PLANE);\n",
"    const float4 v_pixels_in     = qcom_read_imagef_2x2(src_image_uv_plane, sampler, read_coord, UV_V_PLANE);\n",
"    float2       uv_pixels_out[2][2] = {\n",
"        {{u_pixels_in.s3, v_pixels_in.s3}, {u_pixels_in.s2, v_pixels_in.s2}},\n",
"        {{u_pixels_in.s0, v_pixels_in.s0}, {u_pixels_in.s1, v_pixels_in.s1}},\n",
"        };\n",
"    qcom_write_imagefv_2x1_n10p01(dest_image_uv_plane, write_coord,                uv_pixels_out[0]);\n",
"    qcom_write_imagefv_2x1_n10p01(dest_image_uv_plane, write_coord + (int2)(0, 1), uv_pixels_out[1]);\n",
"}\n",
"\n",
// Reads 2x2 from a Y-only image and writes 2x1 to a Y-only image
"__kernel void read_y_2x2_write_y_2x2(__read_only  image2d_t src_image_y_plane,\n",
"                                     __write_only image2d_t dest_image_y_plane,\n",
"                                                  sampler_t sampler)\n",
"{\n",
"    const int    wid_x          = get_global_id(0);\n",
"    const int    wid_y          = get_global_id(1);\n",
"    const float2 read_coord     = (float2)(2 * wid_x, 2 * wid_y) + 0.5;\n",
"    const int2   write_coord    = (int2)(2 * wid_x, 2 * wid_y);\n",
"    const float4 y_pixels_in    = qcom_read_imagef_2x2(src_image_y_plane, sampler, read_coord, Y_Y_PLANE);\n",
"    float        y_pixels_out[] = {\n",
"       y_pixels_in.s3, y_pixels_in.s2,\n",
"       y_pixels_in.s0, y_pixels_in.s1,\n",
"       };\n",
"    qcom_write_imagefv_2x2_n10p00(dest_image_y_plane, write_coord, y_pixels_out);\n",
"}\n",
"\n",
// Blit used for image format conversion
"__kernel void blit(__read_only  image2d_t src_image,\n",
"                   __write_only image2d_t dest_image,\n",
"                                sampler_t sampler)\n",
"{\n",
"    const int    wid_x = get_global_id(0);\n",
"    const int    wid_y = get_global_id(1);\n",
"    const int2   coord = (int2)(wid_x, wid_y);\n",
"    const float4 pixel = read_imagef(src_image, sampler, coord);\n",
"    write_imagef(dest_image, coord, pixel);\n",
"}\n"
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Please specify source image.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string src_image_filename(argv[1]);

    cl_wrapper wrapper;
    cl_program   program             = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel    copy_kernels[]      = {
            wrapper.make_kernel("read_yuv_2x2_write_y_2x1",  program),
            wrapper.make_kernel("read_yuv_2x2_write_uv_2x1",  program),
            wrapper.make_kernel("read_uv_2x2_write_uv_2x1",  program),
            wrapper.make_kernel("read_y_2x2_write_y_2x2",  program),
    };
    cl_kernel    conversion_kernel   = wrapper.make_kernel("blit", program);
    cl_context   context             = wrapper.get_context();
    p010_image_t src_p010_image_info = load_p010_image_data(src_image_filename);

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for P010 image format is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_compressed_image"))
    {
        std::cerr << "Extension cl_qcom_compressed_image needed for reading/writing compressed images is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_vector_image_ops"))
    {
        std::cerr << "Extension cl_qcom_vector_image_ops needed for vector image reads/writes is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for ION-backed images is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_ion_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ion_host_ptr needed for ION-backed images is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 1: Create suitable ion buffer-backed CL images. Note that planar formats (like P010) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */

    cl_image_format src_p010_format;
    src_p010_format.image_channel_order     = CL_QCOM_P010;
    src_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_p010_desc;
    std::memset(&src_p010_desc, 0, sizeof(src_p010_desc));
    src_p010_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_p010_desc.image_width  = src_p010_image_info.y_width;
    src_p010_desc.image_height = src_p010_image_info.y_height;

    cl_int err = 0;
    cl_mem_ion_host_ptr src_p010_ion_mem = wrapper.make_ion_buffer_for_yuv_image(src_p010_format, src_p010_desc);
    cl_mem src_p010_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_p010_format,
            &src_p010_desc,
            &src_p010_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        std::exit(err);
    }

    cl_image_format out_p010_format;
    out_p010_format.image_channel_order     = CL_QCOM_P010;
    out_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_p010_desc;
    std::memset(&out_p010_desc, 0, sizeof(out_p010_desc));
    out_p010_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_p010_desc.image_width  = src_p010_image_info.y_width;
    out_p010_desc.image_height = src_p010_image_info.y_height;

    cl_mem_ion_host_ptr out_p010_ion_mem = wrapper.make_ion_buffer_for_yuv_image(out_p010_format, out_p010_desc);
    cl_mem out_p010_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_p010_format,
            &out_p010_desc,
            &out_p010_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        std::exit(err);
    }

    cl_image_format compressed_p010_format;
    compressed_p010_format.image_channel_order     = CL_QCOM_COMPRESSED_P010;
    compressed_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_p010_desc;
    std::memset(&compressed_p010_desc, 0, sizeof(compressed_p010_desc));
    compressed_p010_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_p010_desc.image_width  = src_p010_image_info.y_width;
    compressed_p010_desc.image_height = src_p010_image_info.y_height;

    cl_mem_ion_host_ptr compressed_p010_ion_mem = wrapper.make_ion_buffer_for_compressed_image(compressed_p010_format, compressed_p010_desc);
    cl_mem compressed_p010_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_p010_format,
            &compressed_p010_desc,
            &compressed_p010_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Separate planar P010 images into their component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_P010_Y;
    src_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_p010_image_info.y_width;
    src_y_plane_desc.image_height = src_p010_image_info.y_height;
    src_y_plane_desc.mem_object   = src_p010_image;

    cl_mem src_y_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &src_y_plane_format,
            &src_y_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." << "\n";
        std::exit(err);
    }

    cl_image_format src_uv_plane_format;
    src_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    src_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_uv_plane_desc;
    std::memset(&src_uv_plane_desc, 0, sizeof(src_uv_plane_desc));
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    src_uv_plane_desc.image_width  = src_p010_image_info.y_width;
    src_uv_plane_desc.image_height = src_p010_image_info.y_height;
    src_uv_plane_desc.mem_object   = src_p010_image;

    cl_mem src_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &src_uv_plane_format,
            &src_uv_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." << "\n";
        std::exit(err);
    }

    cl_image_format out_y_plane_format;
    out_y_plane_format.image_channel_order     = CL_QCOM_P010_Y;
    out_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_y_plane_desc;
    std::memset(&out_y_plane_desc, 0, sizeof(out_y_plane_desc));
    out_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_y_plane_desc.image_width  = out_p010_desc.image_width;
    out_y_plane_desc.image_height = out_p010_desc.image_height;
    out_y_plane_desc.mem_object   = out_p010_image;

    cl_mem out_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &out_y_plane_format,
            &out_y_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for destination image y plane." << "\n";
        std::exit(err);
    }

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    out_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    out_uv_plane_desc.image_width  = out_p010_desc.image_width;
    out_uv_plane_desc.image_height = out_p010_desc.image_height;
    out_uv_plane_desc.mem_object   = out_p010_image;

    cl_mem out_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &out_uv_plane_format,
            &out_uv_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for destination image uv plane." << "\n";
        std::exit(err);
    }

    cl_image_format compressed_y_plane_format;
    compressed_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_P010_Y;
    compressed_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_y_plane_desc;
    std::memset(&compressed_y_plane_desc, 0, sizeof(compressed_y_plane_desc));
    compressed_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_y_plane_desc.image_width  = compressed_p010_desc.image_width;
    compressed_y_plane_desc.image_height = compressed_p010_desc.image_height;
    compressed_y_plane_desc.mem_object   = compressed_p010_image;

    cl_mem compressed_y_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &compressed_y_plane_format,
            &compressed_y_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image y plane." << "\n";
        std::exit(err);
    }

    cl_image_format compressed_uv_plane_format;
    compressed_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_P010_UV;
    compressed_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_uv_plane_desc;
    std::memset(&compressed_uv_plane_desc, 0, sizeof(compressed_uv_plane_desc));
    compressed_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    compressed_uv_plane_desc.image_width  = compressed_p010_desc.image_width;
    compressed_uv_plane_desc.image_height = compressed_p010_desc.image_height;
    compressed_uv_plane_desc.mem_object   = compressed_p010_image;

    cl_mem compressed_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &compressed_uv_plane_format,
            &compressed_uv_plane_desc,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image uv plane." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear P010 images you must observe row alignment
     * restrictions. (You may also write to the ion buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_y_plane,
            CL_TRUE,
            CL_MAP_WRITE,
            origin,
            src_y_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image y-plane buffer for writing." << "\n";
        std::exit(err);
    }

    // Copies image data to the ION buffer from the host
    for (uint32_t i = 0; i < src_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_p010_image_info.y_plane.data() + i * src_y_plane_desc.image_width * 2,
                src_y_plane_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        std::exit(err);
    }

    // Note the discrepancy between the child plane image descriptor and the size required by clEnqueueMapImage.
    const size_t src_uv_region[] = {src_uv_plane_desc.image_width / 2, src_uv_plane_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_uv_plane,
            CL_TRUE,
            CL_MAP_WRITE,
            origin,
            src_uv_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image uv-plane buffer for writing." << "\n";
        std::exit(err);
    }

    // Copies image data to the ION buffer from the host
    for (uint32_t i = 0; i < src_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                image_ptr                           + i * row_pitch,
                src_p010_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width * 2,
                src_uv_plane_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_uv_plane, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image uv-plane data buffer." << "\n";
        std::exit(err);
    }

    /*
     * Step 4: Set up other kernel arguments
     */

    cl_sampler sampler = clCreateSampler(
            context,
            CL_FALSE,
            CL_ADDRESS_CLAMP_TO_EDGE,
            CL_FILTER_NEAREST,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        std::exit(err);
    }

    /*
     * Step 5: Run the kernel separately for y- and uv-planes
     */

    enum class comp_t
    {
        Y_ONLY,
        UV_ONLY,
    };

    // Impromptu data structure to cut down on code duplication to enqueue kernels.
    struct kernel_exec_params_t
    {
        cl_kernel copy_kernel;
        cl_mem    src_plane;
        cl_mem    dst_plane;
        size_t    width;
        size_t    height;
        comp_t    comp_type;
    };

    std::array<kernel_exec_params_t, 4> kernel_execution_params{
            /*                   kernel,          source,         destination,         work_size[0],                             work_size[1] */
            kernel_exec_params_t{copy_kernels[0], src_p010_image, compressed_y_plane,  src_p010_desc.image_width / 2,            src_p010_desc.image_height / 2,            comp_t::Y_ONLY},
            kernel_exec_params_t{copy_kernels[1], src_p010_image, compressed_uv_plane, work_units(src_p010_desc.image_width, 4), work_units(src_p010_desc.image_height, 4), comp_t::UV_ONLY},
            kernel_exec_params_t{copy_kernels[2], src_uv_plane,   compressed_uv_plane, work_units(src_p010_desc.image_width, 4), work_units(src_p010_desc.image_height, 4), comp_t::UV_ONLY},
            kernel_exec_params_t{copy_kernels[3], src_y_plane,    compressed_y_plane,  src_p010_desc.image_width / 2,            src_p010_desc.image_height / 2,            comp_t::Y_ONLY},
    };

    for (size_t i = 0; i < kernel_execution_params.size(); ++i)
    {
        cl_kernel copy_kernel      = kernel_execution_params[i].copy_kernel;
        cl_mem    src_plane        = kernel_execution_params[i].src_plane;
        cl_mem    compressed_plane = kernel_execution_params[i].dst_plane;

        err = clSetKernelArg(copy_kernel, 0, sizeof(src_plane), &src_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for copy kernel." << "\n";
            std::exit(err);
        }

        err = clSetKernelArg(copy_kernel, 1, sizeof(compressed_plane), &compressed_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for copy kernel." << "\n";
            std::exit(err);
        }

        err = clSetKernelArg(copy_kernel, 2, sizeof(sampler), &sampler);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for copy kernel." << "\n";
            std::exit(err);
        }

        const size_t work_size[] = {kernel_execution_params[i].width, kernel_execution_params[i].height};
        err = clEnqueueNDRangeKernel(
                command_queue,
                copy_kernel,
                2,
                NULL,
                work_size,
                NULL,
                0,
                NULL,
                NULL
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for copy kernel." << "\n";
            std::exit(err);
        }

        size_t comparison_map_region[] = {0, 0, 1};
        cl_mem comparison_plane, dst_plane;
        size_t comparison_width  = 0;
        size_t comparison_height = 0;

        switch(kernel_execution_params[i].comp_type)
        {
            case comp_t::Y_ONLY:
            {
                comparison_map_region[0] = src_p010_desc.image_width;
                comparison_map_region[1] = src_p010_desc.image_height;
                comparison_width         = src_p010_desc.image_width * 2;
                comparison_height        = src_p010_desc.image_height;
                comparison_plane         = src_y_plane;
                dst_plane                = out_y_plane;
                break;
            }
            case comp_t::UV_ONLY:
            {
                comparison_map_region[0] = src_p010_desc.image_width / 2;
                comparison_map_region[1] = src_p010_desc.image_height / 2;
                comparison_width         = src_p010_desc.image_width * 2;
                comparison_height        = src_p010_desc.image_height / 2;
                comparison_plane         = src_uv_plane;
                dst_plane                = out_uv_plane;
                break;
            }
            default:
            {
                std::cerr << "On iteration " << i << ":\n";
                std::cerr << "\tUnknown comparison type\n";
                std::exit(EXIT_FAILURE);
            }
        }

        err = clSetKernelArg(conversion_kernel, 0, sizeof(compressed_plane), &compressed_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for conversion kernel." << "\n";
            std::exit(err);
        }

        err = clSetKernelArg(conversion_kernel, 1, sizeof(dst_plane), &dst_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for conversion kernel." << "\n";
            std::exit(err);
        }

        err = clSetKernelArg(conversion_kernel, 2, sizeof(sampler), &sampler);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for conversion kernel." << "\n";
            std::exit(err);
        }

        err = clEnqueueNDRangeKernel(
                command_queue,
                conversion_kernel,
                2,
                NULL,
                comparison_map_region,
                NULL,
                0,
                NULL,
                NULL
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for conversion kernel." << "\n";
            std::exit(err);
        }

        size_t         src_row_pitch = 0;
        unsigned char *src_image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
                command_queue,
                comparison_plane,
                CL_TRUE,
                CL_MAP_READ,
                origin,
                comparison_map_region,
                &src_row_pitch,
                NULL,
                0,
                NULL,
                NULL,
                &err
        ));
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " mapping src image plane for validation." << "\n";
            std::exit(err);
        }

        size_t         dst_row_pitch = 0;
        unsigned char *dst_image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
                command_queue,
                dst_plane,
                CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE,
                origin,
                comparison_map_region,
                &dst_row_pitch,
                NULL,
                0,
                NULL,
                NULL,
                &err
        ));
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " mapping dst image plane for validation." << "\n";
            std::exit(err);
        }

        if (src_row_pitch != dst_row_pitch)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tRow pitches do not match, so unable to compare images.\n";
            std::exit(EXIT_FAILURE);
        }

        // Compare images and overwrite the destination buffer for the next iteration
        for (size_t j = 0; j < comparison_height; ++j)
        {
            const int res = std::memcmp(src_image_ptr + j * src_row_pitch,
                                        dst_image_ptr + j * dst_row_pitch,
                                        comparison_width);
            if(res != 0)
            {
                std::cerr << "On iteration " << i << ", " << j << ":\n";
                std::cerr << "\tImages were not equal!\n";
                std::exit(EXIT_FAILURE);
            }

            std::memset(dst_image_ptr + j * dst_row_pitch, 0, comparison_width);
        }

        err = clEnqueueUnmapMemObject(command_queue, comparison_plane, src_image_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " unmapping src image plane." << "\n";
            std::exit(err);
        }

        err = clEnqueueUnmapMemObject(command_queue, dst_plane, dst_image_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " unmapping dst image plane." << "\n";
            std::exit(err);
        }
    }

    clFinish(command_queue); // Finish up the unmaps above.

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_p010_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_p010_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_p010_image);

    return 0;
}
