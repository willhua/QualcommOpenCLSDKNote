//--------------------------------------------------------------------------------------
// File: fft_matrix.cpp
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
"Usage: fft_matrix <source> <real output> <imaginary output>\n"
"Runs a kernel that computes the 2D fast Fourier transform of the matrix"
"<source>, and writes the real and imaginary parts of the output to the matrices\n"
"<real output> and <imaginary output>, respectively. We use the well-known\n"
"Cooley-Tukey algorithm.\n"
"The matrix must have width = height = a power of 2.\n";

static const char *PROGRAM_SOURCE[] = {
// Forward declaration
"uint bit_reverse(uint n, int num_bits);\n",
// Each work group will find the FFT of one row.
// Writes the result of this pass into a buffer, already transposed so that
// it can be used optimally in the next pass.
"__kernel void fft_row_pass(__global const float *src_matrix,\n",
"                           __global float2      *result,\n",
"                                    int          width,\n",
"                                    int          log_w,\n",
"                           __local  float2      *scratch)\n",
"{\n",
"    const int local_id   = get_local_id(0);\n",
"    const int local_size = get_local_size(0);\n",
"    const int y_coord    = get_group_id(1);\n",
"\n",
"    for (int i = local_id; i < (width / 2); i += local_size)\n",
"    {\n",
"        const int    idx0    = bit_reverse(2 * i,     log_w);\n",
"        const int    idx1    = bit_reverse(2 * i + 1, log_w);\n",
"        const int    coords0 = idx0 + width * y_coord;\n",
"        const int    coords1 = idx1 + width * y_coord;\n",
"        const float  x0      = src_matrix[coords0];\n",
"        const float  x1      = src_matrix[coords1];\n",
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
"__kernel void fft_col_pass(__global const float2 *data,\n",
"                           __global float        *real_part,\n",
"                           __global float        *imag_part,\n",
"                                    int           width,\n",
"                                    int           log_w,\n",
"                           __local  float2       *scratch)\n",
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
"        const int idx0       = y_coord + width * i;\n",
"        const int idx1       = y_coord + width * (i + width / 2);\n",
"        const float2 temp0   = scratch[i];\n",
"        const float2 temp1   = scratch[i + width / 2];\n",
"        const float  coeff_r = native_cos(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float  coeff_i = native_sin(-2.f * M_PI_F * i * native_recip(width));\n",
"        const float2 product = (float2)(coeff_r * temp1.x - coeff_i * temp1.y,\n",
"                                        coeff_r * temp1.y + coeff_i * temp1.x);\n",
"        const float2 res0    = temp0 + product;\n",
"        const float2 res1    = temp0 - product;\n",
"        real_part[idx0]      = res0.x;\n",
"        real_part[idx1]      = res1.x;\n",
"        imag_part[idx0]      = res0.y;\n",
"        imag_part[idx1]      = res1.y;\n",
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
        std::cerr << "Please specify source and output matrices.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string src_matrix_filename(argv[1]);
    const std::string real_out_filename(argv[2]);
    const std::string imag_out_filename(argv[3]);

    cl_wrapper wrapper;
    cl_program program         = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel  kernel_row_pass = wrapper.make_kernel("fft_row_pass", program);
    cl_kernel  kernel_col_pass = wrapper.make_kernel("fft_col_pass", program);
    cl_context context         = wrapper.get_context();
    matrix_t   src_matrix      = load_matrix(src_matrix_filename);

    if ((src_matrix.width != src_matrix.height)
        || !is_power_of_2(src_matrix.width))
    {
        std::cerr << "For this example, the width and height of the input matrix must be equal, and\n"
                  << "they must both be a power of 2. " << src_matrix_filename << " is " << src_matrix.width << "x"
                  << src_matrix.height << ".\n";
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

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

    const size_t        src_matrix_bytes = src_matrix.width * src_matrix.height * sizeof(cl_float);
    cl_mem_ion_host_ptr src_ion_mem      = wrapper.make_ion_buffer(src_matrix_bytes);
    std::memcpy(src_ion_mem.ion_hostptr, src_matrix.elements.data(), src_matrix_bytes);
    cl_int err;
    cl_mem src_matrix_mem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            src_matrix_bytes,
            &src_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for source image." << "\n";
        std::exit(err);
    }

    const size_t        row_pass_result_buffer_size = src_matrix.width * src_matrix.height * sizeof(cl_float2);
    cl_mem_ion_host_ptr row_pass_result_ion_mem     = wrapper.make_ion_buffer(row_pass_result_buffer_size);
    cl_mem row_pass_result_mem                      = clCreateBuffer(
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

    cl_mem_ion_host_ptr real_out_ion_mem = wrapper.make_ion_buffer(src_matrix_bytes);
    cl_mem real_out_matrix_mem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            src_matrix_bytes,
            &real_out_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for real output matrix." << "\n";
        std::exit(err);
    }

    cl_mem_ion_host_ptr imag_out_ion_mem = wrapper.make_ion_buffer(src_matrix_bytes);
    cl_mem imag_out_matrix_mem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            src_matrix_bytes,
            &imag_out_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for imaginary output matrix." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Set up and run the row- and column-pass kernels.
     */

    err = clSetKernelArg(kernel_row_pass, 0, sizeof(src_matrix_mem), &src_matrix_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_row_pass, 1, sizeof(row_pass_result_mem), &row_pass_result_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    const cl_int width = static_cast<cl_int>(src_matrix.width);
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

    err = clSetKernelArg(kernel_row_pass, 4, sizeof(cl_float2) * width, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        std::exit(err);
    }


    cl_command_queue command_queue          = wrapper.get_command_queue();
    const size_t row_pass_wg_size           = wrapper.get_max_workgroup_size(kernel_row_pass);
    const size_t global_work_size[]         = {static_cast<size_t>(src_matrix.width / 2), static_cast<size_t>(src_matrix.height)};
    const size_t row_pass_local_work_size[] = {std::min(global_work_size[0], row_pass_wg_size), 1};
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

    err = clSetKernelArg(kernel_col_pass, 0, sizeof(row_pass_result_mem), &row_pass_result_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 1, sizeof(real_out_matrix_mem), &real_out_matrix_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_col_pass, 2, sizeof(imag_out_matrix_mem), &imag_out_matrix_mem);
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
    const size_t col_pass_local_work_size[] = {std::min<size_t>(src_matrix.width / 2, col_pass_wg_size), 1};
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

    /*
     * Step 3: Copy the data out of the ion buffer for each plane.
     */

    matrix_t real_out_info;
    real_out_info.width  = src_matrix.width;
    real_out_info.height = src_matrix.height;
    real_out_info.elements.resize(real_out_info.width * real_out_info.height);
    cl_float *mat_ptr    = NULL;
    mat_ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            real_out_matrix_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            src_matrix_bytes,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping real part output matrix for reading." << "\n";
        std::exit(err);
    }

    std::memcpy(real_out_info.elements.data(), mat_ptr, src_matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, real_out_matrix_mem, mat_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping real part output matrix." << "\n";
        std::exit(err);
    }

    matrix_t imag_out_info;
    imag_out_info.width  = src_matrix.width;
    imag_out_info.height = src_matrix.height;
    imag_out_info.elements.resize(imag_out_info.width * imag_out_info.height);
    mat_ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            imag_out_matrix_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            src_matrix_bytes,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping imaginary part output matrix for reading." << "\n";
        std::exit(err);
    }

    std::memcpy(imag_out_info.elements.data(), mat_ptr, src_matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, imag_out_matrix_mem, mat_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping imaginary part output matrix." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    save_matrix(real_out_filename, real_out_info);
    save_matrix(imag_out_filename, imag_out_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(src_matrix_mem);
    clReleaseMemObject(row_pass_result_mem);
    clReleaseMemObject(real_out_matrix_mem);
    clReleaseMemObject(imag_out_matrix_mem);

    return 0;
}

bool is_power_of_2(size_t n)
{
    return n && !(n & (n - 1));
}
