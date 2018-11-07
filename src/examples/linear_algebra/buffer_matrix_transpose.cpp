//--------------------------------------------------------------------------------------
// File: buffer_matrix_transpose.cpp
// Desc: Demonstrates transposing matrices with buffers
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
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

static const char *HELP_MESSAGE = "\n"
"Usage: buffer_matrix_transpose <input matrix> [<output file>]\n"
"Given a matrix, computes its transpose.\n"
"There is no size restriction for the matrix. To the extent possible it\n"
"calculates the result using an efficient tiled algorithm. For the portion of\n"
"the result matrix not covered by tiles it uses a less efficient naive\n"
"implementation.\n"
"If no file is specified for the output, then it is written to stdout.\n";

static const char *PROGRAM_SOURCE[] = {
"__kernel void transpose(__global const float *matrix,\n",
"                        __global       float *matrix_t,\n",
"                                       int    width,\n",
"                                       int    height)\n",
"{\n",
"    const int             wid_x  = get_global_id(0);\n",
"    const int             wid_y  = get_global_id(1);\n",
"    __global const float *offset = matrix + width * 4 * wid_y;\n",
"    const float4          rows[] = {\n",
"        vload4(wid_x, offset),\n",
"        vload4(wid_x, offset + width),\n",
"        vload4(wid_x, offset + 2 * width),\n",
"        vload4(wid_x, offset + 3 * width),\n",
"        };\n",
"    __global float *write_offset = matrix_t + height * 4 * wid_x;\n",
"    vstore4((float4)(rows[0].x, rows[1].x, rows[2].x, rows[3].x), wid_y, write_offset);\n",
"    vstore4((float4)(rows[0].y, rows[1].y, rows[2].y, rows[3].y), wid_y, write_offset + height);\n",
"    vstore4((float4)(rows[0].z, rows[1].z, rows[2].z, rows[3].z), wid_y, write_offset + 2 * height);\n",
"    vstore4((float4)(rows[0].w, rows[1].w, rows[2].w, rows[3].w), wid_y, write_offset + 3 * height);\n",
"}\n",
"\n",
"__kernel void transpose_rem(__global const float *matrix,\n",
"                            __global       float *matrix_t,\n",
"                                           int    x_rem_start,\n",
"                                           int    y_rem_start,\n",
"                                           int    width,\n",
"                                           int    height)\n",
"{\n",
"    const int wid_x = get_global_id(0) + x_rem_start;\n",
"    const int wid_y = get_global_id(1) + y_rem_start;\n",
"    const int idx   = width * wid_y + wid_x;\n",
"    const int idx_t = height * wid_x + wid_y;\n",
"    matrix_t[idx_t] = matrix[idx];\n",
"}\n"
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Please specify input file.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }

    const std::string matrix_a_filename(argv[1]);
    const bool        output_to_file = argc >= 3;
    const matrix_t    matrix_a       = load_matrix(matrix_a_filename);
    const size_t      matrix_size    = matrix_a.width * matrix_a.height;
    const size_t      matrix_bytes   = matrix_size * sizeof(cl_float);
    const std::string output_filename(output_to_file ? argv[2] : "");

    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel        kernel_tiled  = wrapper.make_kernel("transpose", program);
    cl_kernel        kernel_rem    = wrapper.make_kernel("transpose_rem", program);
    cl_context       context       = wrapper.get_context();
    cl_command_queue command_queue = wrapper.get_command_queue();

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for ION-backed buffers is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_ion_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ion_host_ptr needed for ION-backed buffers is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    cl_int err = CL_SUCCESS;

    /*
     * Step 1: Create suitable ION-backed buffers.
     */

    /*
     * Matrix A
     */

    cl_mem_ion_host_ptr matrix_a_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
    std::memcpy(matrix_a_ion_buf.ion_hostptr, matrix_a.elements.data(), matrix_bytes);
    cl_mem              matrix_a_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_a_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix A." << "\n";
        std::exit(err);
    }

    /*
     * Matrix B
     */

    cl_mem_ion_host_ptr matrix_b_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
    cl_mem              matrix_b_mem     = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_b_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix B." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Set up the kernel arguments
     */

    err = clSetKernelArg(kernel_tiled, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_tiled, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    const cl_int width  = matrix_a.width;
    const cl_int height = matrix_a.height;

    err = clSetKernelArg(kernel_tiled, 2, sizeof(width), &width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_tiled, 3, sizeof(height), &height);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Run the kernel.
     */

    const size_t tiled_global_work_size[] = {static_cast<size_t>(matrix_a.width / 4), static_cast<size_t>(matrix_a.height / 4)};
    if (tiled_global_work_size[0] != 0 && tiled_global_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_tiled,
                2,
                NULL,
                tiled_global_work_size,
                NULL,
                0,
                NULL,
                NULL
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
            std::exit(err);
        }
    }

    /*
     * Step 4: Set up and run less efficient kernels for the edges of the result
     *         matrix that weren't covered by the tiled version.
     */

    const cl_int x_rem_start = (matrix_a.width / 4) * 4;
    const cl_int y_rem_start = (matrix_a.height / 4) * 4;


    err = clSetKernelArg(kernel_rem, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_rem, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_rem, 2, sizeof(x_rem_start), &x_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    const cl_int right_y_rem_start = 0;
    err = clSetKernelArg(kernel_rem, 3, sizeof(right_y_rem_start), &right_y_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_rem, 4, sizeof(width), &width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_rem, 5, sizeof(height), &height);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 5." << "\n";
        std::exit(err);
    }

    /*
     * Covers the remaining right side for the full height of the matrix.
     */

    const size_t right_rem_work_size[] = {static_cast<size_t>(matrix_a.width - x_rem_start), static_cast<size_t>(matrix_a.height)};
    if (right_rem_work_size[0] != 0 && right_rem_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_rem,
                2,
                NULL,
                right_rem_work_size,
                NULL,
                0,
                NULL,
                NULL
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for right remainder." << "\n";
            std::exit(err);
        }
    }

    const cl_int bottom_x_rem_start = 0;
    err = clSetKernelArg(kernel_rem, 2, sizeof(bottom_x_rem_start), &bottom_x_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel_rem, 3, sizeof(y_rem_start), &y_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    /*
     * Covers the remaining bottom portion of the result matrix not covered above.
     */

    const size_t bottom_rem_work_size[] = {static_cast<size_t>(x_rem_start), static_cast<size_t>(matrix_a.height - y_rem_start)};
    if (bottom_rem_work_size[0] != 0 && bottom_rem_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_rem,
                2,
                NULL,
                bottom_rem_work_size,
                NULL,
                0,
                NULL,
                NULL
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for right remainder." << "\n";
            std::exit(err);
        }
    }

    /*
     * Step 5: Copy the data out of the ION buffer.
     */

    matrix_t matrix_b;
    matrix_b.width              = matrix_a.height;
    matrix_b.height             = matrix_a.width;
    const size_t matrix_b_size  = matrix_b.width * matrix_b.height;
    matrix_b.elements.resize(matrix_b_size);

    cl_float *ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            matrix_b_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            matrix_bytes,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapBuffer." << "\n";
        std::exit(err);
    }

    std::memcpy(matrix_b.elements.data(), ptr, matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, matrix_b_mem, ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    if (output_to_file)
    {
        save_matrix(output_filename, matrix_b);
    }
    else
    {
        save_matrix(std::cout, matrix_b);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(matrix_a_mem);
    clReleaseMemObject(matrix_b_mem);

    return 0;
}
