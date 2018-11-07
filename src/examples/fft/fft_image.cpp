//--------------------------------------------------------------------------------------
// File: fft_image.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *HELP_MESSAGE = "\n"
"Usage: fft_image <source> <real output> <imaginary output>\n"
"Runs a kernel that computes the 2D fast Fourier transform of the y-plane of an\n"
"NV12 image <source>, and writes the real and imaginary parts of the output to\n"
"<real output> and <imaginary output>, respectively. We use the well-known\n"
"Cooley-Tukey algorithm.\n"
"The image must have width = height = a power of 2.\n";

static const char *PROGRAM_SOURCE[] = {
// Forward declaration
"uint bit_reverse(uint n, int num_bits);\n",
// Each work group will find the FFT of one row.
// Writes the result of this pass into a buffer, already transposed so that
// it can be used optimally in the next pass.
"__kernel void fft_row_pass(__read_only  image2d_t  src_image,\n",
"                           __global     float2    *result,\n",
"                                        int        width,\n",
"                                        int        log_w,\n",
"                                        sampler_t  sampler,\n",
"                           __local      float2    *scratch)\n",
"{\n",
"    const int local_id   = get_local_id(0);\n",
"    const int local_size = get_local_size(0);\n",
"    const int y_coord    = get_group_id(1);\n",
"\n",
"    for (int i = local_id; i < (width / 2); i += local_size)\n",
"    {\n",
"        const int    idx0    = bit_reverse(2 * i,     log_w);\n",
"        const int    idx1    = bit_reverse(2 * i + 1, log_w);\n",
"        const int2   coords0 = (int2)(idx0, y_coord);\n",
"        const int2   coords1 = (int2)(idx1, y_coord);\n",
"        const float  x0      = 255.f * read_imagef(src_image, sampler, coords0).x;\n",
"        const float  x1      = 255.f * read_imagef(src_image, sampler, coords1).x;\n",
"        const float2 res0    = (float2)(x0 + x1, 0.f);\n",
"        const float2 res1    = (float2)(x0 - x1, 0.f);\n",
"        scratch[2 * i]       = res0;\n",
"        scratch[2 * i + 1]   = res1;\n",
"    }\n",
"    barrier(CLK_LOCAL_MEM_FENCE);\n",
"\n",
"    for (int working_size = 2; working_size < (width / 2); working_size *= 2)\n",
"    {\n",
"        const int offset = (local_id / working_size) * working_size;\n",
"        for (int i = local_id; i < (width / 2); i += local_size)\n",
"        {\n",
"            const int    idx            = offset + i;\n",
"            const float2 temp0          = scratch[idx];\n",
"            const float2 temp1          = scratch[idx + working_size];\n",
"            const float  coeff_r        = native_cos(-1.f * M_PI_F * (i % working_size) * native_recip(working_size));\n",
"            const float  coeff_i        = native_sin(-1.f * M_PI_F * (i % working_size) * native_recip(working_size));\n",
"            const float2 product        = (float2)(coeff_r * temp1.x - coeff_i * temp1.y,\n",
"                                                   coeff_r * temp1.y + coeff_i * temp1.x);\n",
"            scratch[idx]                = temp0 + product;\n",
"            scratch[idx + working_size] = temp0 - product;\n",
"        }\n",
"        barrier(CLK_LOCAL_MEM_FENCE);\n",
"    }\n",
"\n",
"    for (int i = local_id; i < (width / 2); i += local_size)\n",
"    {\n",
"        const int    idx0    = i               * width + y_coord;\n",
"        const int    idx1    = (i + width / 2) * width + y_coord;\n",
"        const float2 temp0   = scratch[i];\n",
"        const float2 temp1   = scratch[i + width / 2];\n",
"        const float  coeff_r = native_cos(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float  coeff_i = native_sin(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float2 product = (float2)(coeff_r * temp1.x - coeff_i * temp1.y,\n",
"                                        coeff_r * temp1.y + coeff_i * temp1.x);\n",
"        result[idx0]         = temp0 + product;\n",
"        result[idx1]         = temp0 - product;\n",
"    }\n",
"}\n",
"\n",
// Does the column pass on the transposed result of the row pass.
// Each work group will find the FFT of one column.
"__kernel void fft_col_pass(__global const float2    *data,\n",
"                           __write_only   image2d_t  real_out,\n",
"                           __write_only   image2d_t  imag_out,\n",
"                                          int        width,\n",
"                                          int        log_w,\n",
"                           __local        float2    *scratch)\n",
"{\n",
"    const int local_id   = get_local_id(0);\n",
"    const int local_size = get_local_size(0);\n",
"    const int y_coord    = get_group_id(1);\n",
"\n",
"    for (int i = local_id; i < (width / 2); i += local_size)\n",
"    {\n",
"        const int    idx0    = bit_reverse(2 * i,     log_w) + width * y_coord;\n",
"        const int    idx1    = bit_reverse(2 * i + 1, log_w) + width * y_coord;\n",
"        const int2   coords0 = (int2)(idx0, y_coord);\n",
"        const int2   coords1 = (int2)(idx1, y_coord);\n",
"        const float2 x0      = data[idx0];\n",
"        const float2 x1      = data[idx1];\n",
"        const float2 res0    = x0 + x1;\n",
"        const float2 res1    = x0 - x1;\n",
"        scratch[2 * i]       = res0;\n",
"        scratch[2 * i + 1]   = res1;\n",
"    }\n",
"    barrier(CLK_LOCAL_MEM_FENCE);\n",
"\n",
"    for (int working_size = 2; working_size < (width / 2); working_size *= 2)\n",
"    {\n",
"        const int offset = (local_id / working_size) * working_size;\n",
"        for (int i = local_id; i < (width / 2); i += local_size)\n",
"        {\n",
"            const int    idx            = offset + i;\n",
"            const float2 temp0          = scratch[idx];\n",
"            const float2 temp1          = scratch[idx + working_size];\n",
"            const float  coeff_r        = native_cos(-1.f * M_PI_F * (i % working_size) * native_recip(working_size));\n",
"            const float  coeff_i        = native_sin(-1.f * M_PI_F * (i % working_size) * native_recip(working_size));\n",
"            const float2 product        = (float2)(coeff_r * temp1.x - coeff_i * temp1.y,\n",
"                                                   coeff_r * temp1.y + coeff_i * temp1.x);\n",
"            scratch[idx]                = temp0 + product;\n",
"            scratch[idx + working_size] = temp0 - product;\n",
"        }\n",
"        barrier(CLK_LOCAL_MEM_FENCE);\n",
"    }\n",
"\n",
"    for (int i = local_id; i < (width / 2); i += local_size)\n",
"    {\n",
"        const int2 coords0   = (int2)(y_coord, i);\n",
"        const int2 coords1   = (int2)(y_coord, i + width / 2);\n",
"        const float2 temp0   = scratch[i];\n",
"        const float2 temp1   = scratch[i + width / 2];\n",
"        const float  coeff_r = native_cos(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float  coeff_i = native_sin(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float2 product = (float2)(coeff_r * temp1.x - coeff_i * temp1.y,\n",
"                                        coeff_r * temp1.y + coeff_i * temp1.x);\n",
"        const float2 res0    = temp0 + product;\n",
"        const float2 res1    = temp0 - product;\n",
"        write_imagef(real_out, coords0, (float4)(res0.x));\n",
"        write_imagef(real_out, coords1, (float4)(res1.x));\n",
"        write_imagef(imag_out, coords0, (float4)(res0.y));\n",
"        write_imagef(imag_out, coords1, (float4)(res1.y));\n",
"    }\n",
"}\n",
"\n",
"uint bit_reverse(uint n, int num_bits)\n",
"{\n",
"    uint res = 0;\n",
"    for (int i = 0; i < num_bits; ++i)\n",
"    {\n",
"        res |= (((1 << i) & n) >> i) << (num_bits - 1 - i);\n",
"    }\n",
"    return res;\n",
"}\n",
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

static bool is_power_of_2(size_t n);

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Please specify source and output images.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string src_image_filename(argv[1]);
    const std::string real_out_filename(argv[2]);
    const std::string imag_out_filename(argv[3]);

    cl_wrapper   wrapper;
    cl_program   program             = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel    kernel_row_pass     = wrapper.make_kernel("fft_row_pass", program);
    cl_kernel    kernel_col_pass     = wrapper.make_kernel("fft_col_pass", program);
    cl_context   context             = wrapper.get_context();
    nv12_image_t src_nv12_image_info = load_nv12_image_data(src_image_filename);

    if ((src_nv12_image_info.y_width != src_nv12_image_info.y_height)
        || !is_power_of_2(src_nv12_image_info.y_width))
    {
        std::cerr << "For this example, the width and height of the input image must be equal, and\n"
                  << "they must both be a power of 2. " << src_image_filename << " is " << src_nv12_image_info.y_width << "x"
                  << src_nv12_image_info.y_height << ".\n";
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_extract_image_plane"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
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
     * Step 1: Create suitable ion buffer-backed CL images.
     */

    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order     = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc;
    std::memset(&src_nv12_desc, 0, sizeof(src_nv12_desc));
    src_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width  = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_ion_host_ptr src_nv12_ion_mem = wrapper.make_ion_buffer_for_yuv_image(src_nv12_format, src_nv12_desc);
    cl_int err;
    cl_mem src_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_nv12_format,
            &src_nv12_desc,
            &src_nv12_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        std::exit(err);
    }

    const size_t        row_pass_result_buffer_size = src_nv12_desc.image_width * src_nv12_desc.image_height
                                                      * sizeof(cl_float2);
    cl_mem_ion_host_ptr row_pass_result_ion_mem     = wrapper.make_ion_buffer(row_pass_result_buffer_size);
    cl_mem row_pass_result                          = clCreateBuffer(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            row_pass_result_buffer_size,
            &row_pass_result_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer." << "\n";
        std::exit(err);
    }

    cl_image_format real_out_format;
    real_out_format.image_channel_order     = CL_R;
    real_out_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc real_out_desc;
    std::memset(&real_out_desc, 0, sizeof(real_out_desc));
    real_out_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    real_out_desc.image_width     = src_nv12_image_info.y_width;
    real_out_desc.image_height    = src_nv12_image_info.y_height;
    real_out_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(real_out_format, real_out_desc);

    cl_mem_ion_host_ptr real_out_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(real_out_format, real_out_desc);
    cl_mem real_out_image = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &real_out_format,
            &real_out_desc,
            &real_out_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image, real part." << "\n";
        std::exit(err);
    }

    cl_image_format imag_out_format;
    imag_out_format.image_channel_order     = CL_R;
    imag_out_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc imag_out_desc;
    std::memset(&imag_out_desc, 0, sizeof(imag_out_desc));
    imag_out_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    imag_out_desc.image_width     = src_nv12_image_info.y_width;
    imag_out_desc.image_height    = src_nv12_image_info.y_height;
    imag_out_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(imag_out_format, imag_out_desc);

    cl_mem_ion_host_ptr imag_out_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(imag_out_format, imag_out_desc);
    cl_mem imag_out_image = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &imag_out_format,
            &imag_out_desc,
            &imag_out_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image, imaginary part." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Separate planar NV12 images into their component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    src_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_nv12_image_info.y_width;
    src_y_plane_desc.image_height = src_nv12_image_info.y_height;
    src_y_plane_desc.mem_object   = src_nv12_image;

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

    /*
     * Step 3: Copy data to input image plane. Note that for linear NV12 images you must observe row alignment
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
                src_nv12_image_info.y_plane.data() + i * src_y_plane_desc.image_width,
                src_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        std::exit(err);
    }

    /*
     * Step 4: Set up other kernel arguments
     */

    cl_sampler sampler = clCreateSampler(
            context,
            CL_FALSE,
            CL_ADDRESS_NONE,
            CL_FILTER_NEAREST,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        std::exit(err);
    }

    /*
     * Step 5: Set up and run the row- and column-pass kernels.
     */

    err = clSetKernelArg(kernel_row_pass, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_row_pass, 1, sizeof(row_pass_result), &row_pass_result);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    const cl_int width = static_cast<cl_int>(src_nv12_desc.image_width);
    err = clSetKernelArg(kernel_row_pass, 2, sizeof(width), &width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    const cl_int log_w = static_cast<cl_int>(std::log2(width));
    err = clSetKernelArg(kernel_row_pass, 3, sizeof(log_w), &log_w);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_row_pass, 4, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_row_pass, 5, sizeof(cl_float2) * width, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 5." << "\n";
        std::exit(err);
    }

    const size_t row_pass_wg_size           = wrapper.get_max_workgroup_size(kernel_row_pass);
    const size_t global_work_size[]         = {src_nv12_desc.image_width / 2, src_nv12_desc.image_height};
    const size_t row_pass_local_work_size[] = {std::min(src_nv12_desc.image_width / 2, row_pass_wg_size), 1};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel_row_pass,
            2,
            NULL,
            global_work_size,
            row_pass_local_work_size,
            0,
            NULL,
            NULL
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 0, sizeof(row_pass_result), &row_pass_result);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 1, sizeof(real_out_image), &real_out_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 2, sizeof(imag_out_image), &imag_out_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 3, sizeof(width), &width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 4, sizeof(log_w), &log_w);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 5, sizeof(cl_float2) * width, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 5." << "\n";
        std::exit(err);
    }

    const size_t col_pass_wg_size           = wrapper.get_max_workgroup_size(kernel_col_pass);
    const size_t col_pass_local_work_size[] = {std::min(src_nv12_desc.image_width / 2, col_pass_wg_size), 1};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel_col_pass,
            2,
            NULL,
            global_work_size,
            col_pass_local_work_size,
            0,
            NULL,
            NULL
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    /*
     * Step 6: Copy the data out of the ion buffer for each plane.
     */

    single_channel_float_image_t real_out_info;
    real_out_info.width  = real_out_desc.image_width;
    real_out_info.height = real_out_desc.image_height;
    real_out_info.pixels.resize(real_out_info.width * real_out_info.height * sizeof(cl_float));

    const size_t out_region[] = {real_out_desc.image_width, real_out_desc.image_height, 1};
    row_pitch                 = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            real_out_image,
            CL_TRUE,
            CL_MAP_READ,
            origin,
            out_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping real part output image for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < real_out_desc.image_height; ++i)
    {
        std::memcpy(
                real_out_info.pixels.data() + i * real_out_desc.image_width * sizeof(cl_float),
                image_ptr                   + i * row_pitch,
                real_out_desc.image_width * sizeof(cl_float)
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, real_out_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping real part output image." << "\n";
        std::exit(err);
    }

    single_channel_float_image_t imag_out_info;
    imag_out_info.width  = imag_out_desc.image_width;
    imag_out_info.height = imag_out_desc.image_height;
    imag_out_info.pixels.resize(imag_out_info.width * imag_out_info.height * sizeof(cl_float));

    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            imag_out_image,
            CL_TRUE,
            CL_MAP_READ,
            origin,
            out_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping imaginary part output image for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < imag_out_desc.image_height; ++i)
    {
        std::memcpy(
                imag_out_info.pixels.data() + i * imag_out_desc.image_width * sizeof(cl_float),
                image_ptr                   + i * row_pitch,
                imag_out_desc.image_width * sizeof(cl_float)
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, imag_out_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping imaginary part output image." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    save_single_channel_image_data(real_out_filename, real_out_info);
    save_single_channel_image_data(imag_out_filename, imag_out_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(real_out_image);
    clReleaseMemObject(imag_out_image);
    clReleaseMemObject(row_pass_result);

    return 0;
}

bool is_power_of_2(size_t n)
{
    return n && !(n & (n - 1));
}
