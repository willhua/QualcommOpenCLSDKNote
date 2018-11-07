//--------------------------------------------------------------------------------------
// File: matrix_addition.cpp
// Desc:
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
"Usage: matrix_addition <matrix A> <matrix B> [<output file>]\n"
"Computes the matrix sum C = A + B. See README.md for matrix input format.\n"
"If no file is specified for the output, then it is written to stdout.\n";

static const char *PROGRAM_SOURCE[] = {
"__kernel void buffer_addition(__global const float *matrix_a,\n",
"                              __global const float *matrix_b,\n",
"                              __global       float *matrix_c)\n",
"{\n",
"    const int wid_x = get_global_id(0);\n",
"    matrix_c[wid_x] = matrix_a[wid_x] + matrix_b[wid_x];\n",
"}\n",
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify input files.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string matrix_a_filename(argv[1]);
    const std::string matrix_b_filename(argv[2]);
    const bool        output_to_file = argc >= 4;
    const matrix_t    matrix_a       = load_matrix(matrix_a_filename);
    const matrix_t    matrix_b       = load_matrix(matrix_b_filename);
    const size_t      matrix_size    = matrix_a.width * matrix_a.height;
    const size_t      matrix_bytes   = matrix_size * sizeof(cl_float);
    const std::string output_filename(output_to_file ? argv[3] : "");

    if (matrix_a.width != matrix_b.width && matrix_a.height != matrix_b.height)
    {
        std::cerr << "Matrix A and B must have the same dimensions.\n";
        std::exit(EXIT_FAILURE);
    }

    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel        kernel        = wrapper.make_kernel("buffer_addition", program);
    cl_context       context       = wrapper.get_context();
    cl_command_queue command_queue = wrapper.get_command_queue();

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
     * Step 1: Create suitable ION-backed buffers.
     */

    cl_int err =  CL_SUCCESS;

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

    cl_mem_ion_host_ptr matrix_b_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
    std::memcpy(matrix_b_ion_buf.ion_hostptr, matrix_b.elements.data(), matrix_bytes);
    cl_mem              matrix_b_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_b_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix B." << "\n";
        std::exit(err);
    }

    cl_mem_ion_host_ptr matrix_c_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
    cl_mem              matrix_c_mem     = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_c_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix C." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Set up the kernel arguments
     */

    err = clSetKernelArg(kernel, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel, 2, sizeof(matrix_c_mem), &matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Run the kernel.
     */

    const size_t global_work_size = matrix_size;
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            1,
            NULL,
            &global_work_size,
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

    /*
     * Step 4: Copy the data out of the ION buffer.
     */

    cl_float *ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            matrix_c_mem,
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

    matrix_t matrix_c;
    matrix_c.width  = matrix_a.width;
    matrix_c.height = matrix_a.height;
    matrix_c.elements.resize(matrix_size);
    std::memcpy(matrix_c.elements.data(), ptr, matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, matrix_c_mem, ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    if (output_to_file)
    {
        save_matrix(output_filename, matrix_c);
    }
    else
    {
        save_matrix(std::cout, matrix_c);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(matrix_a_mem);
    clReleaseMemObject(matrix_b_mem);
    clReleaseMemObject(matrix_c_mem);

    return 0;
}
