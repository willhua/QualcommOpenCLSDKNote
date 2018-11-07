//--------------------------------------------------------------------------------------
// File: image_matrix_multiplication_half.cpp
// Desc: Demonstrates half-float matrix multiplication using images
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
#include "util/half_float.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *HELP_MESSAGE = "\n"
"Usage: image_matrix_multiplication_half <matrix A> <matrix B> [<output file>]\n"
"Computes the matrix product C = A * B. See README.md for matrix input format.\n"
"It calculates the results matrix using an efficient tiled algorithm.\n"
"There is no size restriction for the matrices, but they may be padded with\n"
"extra elements to meet the tile size.\n"
"If no file is specified for the output, then it is written to stdout.\n";

static const char *PROGRAM_SOURCE[] = {
// Each work item computes a 4-column by 8-row (8x4) section of the output matrix.
// The inner loops read in a 4x4 section of matrix B, an 8x4 section of matrix A,
// and accumulate the partial results for the corresponding 8x4 section of
// matrix C.
// The outer loop iterates over the width of matrix A and the height of matrix B
// to get the complete result.
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
"__kernel void matmul_8x4_blocks(__read_only  image2d_t matrix_a,\n",
"                                __read_only  image2d_t matrix_b,\n",
"                                __write_only image2d_t matrix_c,\n",
"                                             int       matrix_a_width)\n",
"{\n",
"    const int wid_x = get_global_id(0);\n",
"    const int wid_y = get_global_id(1);\n",
"\n",
"    half4 a[8];\n",
"    half4 b[4];\n",
"    half4 c[8];\n",
"\n",
"    for (int i = 0; i < 8; ++i)\n",
"    {\n",
"        c[i] = (half4)(0.0f);\n",
"    }\n",
"\n",
"    for (int j = 0; j < matrix_a_width; j += 4)\n",
"    {\n",
"#pragma unroll\n",
"        for (int i = 0; i < 4; ++i)\n",
"        {\n",
"            b[i] = read_imageh(matrix_b, (int2)(wid_x, i + j));\n",
"        }\n",
"\n",
"#pragma unroll\n",
"        for (int i = 0; i < 8; ++i)\n",
"        {\n",
"            a[i] = read_imageh(matrix_a, (int2)(j / 4, 8 * wid_y + i));\n",
"        }\n",
"\n",
"#pragma unroll\n",
"        for (int i = 0; i < 8; ++i)\n",
"        {\n",
"            c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];\n",
"        }\n",
"    }\n",
"\n",
"#pragma unroll\n",
"    for (int i = 0; i < 8; ++i)\n",
"    {\n",
"        write_imageh(matrix_c, (int2)(wid_x, 8 * wid_y + i), c[i]);\n",
"    }\n",
"}\n"
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

    const std::string   matrix_a_filename(argv[1]);
    const std::string   matrix_b_filename(argv[2]);
    const bool          output_to_file = argc >= 4;
    const half_matrix_t matrix_a       = load_half_matrix(matrix_a_filename);
    const half_matrix_t matrix_b       = load_half_matrix(matrix_b_filename);
    const std::string   output_filename(output_to_file ? argv[3] : "");

    if (matrix_a.width != matrix_b.height)
    {
        std::cerr << "Can't multiply a matrix of dimensions "
                  << matrix_a.width << "x" << matrix_a.height << " "
                  << "by a matrix of dimensions "
                  << matrix_b.width << "x" << matrix_b.height << "\n";
        std::exit(EXIT_FAILURE);
    }

    matrix_t matrix_c;
    matrix_c.width  = matrix_b.width;
    matrix_c.height = matrix_a.height;
    const size_t matrix_c_size  = matrix_c.width * matrix_c.height;
    matrix_c.elements.resize(matrix_c_size);

    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel        kernel        = wrapper.make_kernel("matmul_8x4_blocks", program);
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

    cl_int err = CL_SUCCESS;

    /*
     * Step 1: Create suitable ION-backed images.
     */

    /*
     * Matrix A
     */

    cl_image_format matrix_a_format;
    matrix_a_format.image_channel_order     = CL_RGBA;
    matrix_a_format.image_channel_data_type = CL_HALF_FLOAT;

    cl_image_desc matrix_a_desc;
    std::memset(&matrix_a_desc, 0, sizeof(matrix_a_desc));
    matrix_a_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_a_desc.image_width     = ((matrix_a.width + 3) / 4);
    matrix_a_desc.image_height    = ((matrix_a.height + 7) / 8) * 8;
    matrix_a_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(matrix_a_format, matrix_a_desc);

    cl_mem_ion_host_ptr matrix_a_ion_buf = wrapper.make_ion_buffer_for_nonplanar_image(matrix_a_format,
                                                                                       matrix_a_desc);
    cl_mem matrix_a_mem = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &matrix_a_format,
            &matrix_a_desc,
            &matrix_a_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for matrix A." << "\n";
        std::exit(err);
    }

    char         *image_ptr;
    const size_t  origin[]          = {0, 0, 0};
    const size_t  matrix_a_region[] = {matrix_a_desc.image_width, matrix_a_desc.image_height, 1};
    size_t        row_pitch         = 0;
    image_ptr = static_cast<char *>(clEnqueueMapImage(
            command_queue,
            matrix_a_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            matrix_a_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping matrix A image." << "\n";
        std::exit(err);
    }

    for (size_t i = 0; i < matrix_a_desc.image_height; ++i)
    {
        if (i < static_cast<size_t>(matrix_a.height))
        {
            const size_t unpadded_row_size = sizeof(cl_half) * matrix_a.width;
            std::memcpy(
                    image_ptr                + i * row_pitch,
                    matrix_a.elements.data() + i * matrix_a.width,
                    unpadded_row_size
            );
            const size_t remaining_bytes = row_pitch - unpadded_row_size;
            std::memset(image_ptr + (i * row_pitch) + unpadded_row_size, 0, remaining_bytes);
        }
        else
        {
            std::memset(image_ptr + i * row_pitch, 0, row_pitch);
        }
    }

    err = clEnqueueUnmapMemObject(command_queue, matrix_a_mem, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping matrix A image." << "\n";
        std::exit(err);
    }

    /*
     * Matrix B
     */

    cl_image_format matrix_b_format;
    matrix_b_format.image_channel_order     = CL_RGBA;
    matrix_b_format.image_channel_data_type = CL_HALF_FLOAT;

    cl_image_desc matrix_b_desc;
    std::memset(&matrix_b_desc, 0, sizeof(matrix_b_desc));
    matrix_b_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_b_desc.image_width     = ((matrix_b.width + 3) / 4);
    matrix_b_desc.image_height    = ((matrix_b.height + 7) / 8) * 8;
    matrix_b_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(matrix_b_format, matrix_b_desc);

    cl_mem_ion_host_ptr matrix_b_ion_buf = wrapper.make_ion_buffer_for_nonplanar_image(matrix_b_format,
                                                                                       matrix_b_desc);
    cl_mem matrix_b_mem = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &matrix_b_format,
            &matrix_b_desc,
            &matrix_b_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for matrix B." << "\n";
        std::exit(err);
    }

    const size_t  matrix_b_region[] = {matrix_b_desc.image_width, matrix_b_desc.image_height, 1};
    image_ptr = static_cast<char *>(clEnqueueMapImage(
            command_queue,
            matrix_b_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            matrix_b_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping matrix B image." << "\n";
        std::exit(err);
    }

    for (size_t i = 0; i < matrix_b_desc.image_height; ++i)
    {
        if (i < static_cast<size_t>(matrix_b.height))
        {
            const size_t unpadded_row_size = sizeof(cl_half) * matrix_b.width;
            std::memcpy(
                    image_ptr                + i * row_pitch,
                    matrix_b.elements.data() + i * matrix_b.width,
                    unpadded_row_size
            );
            const size_t remaining_bytes = row_pitch - unpadded_row_size;
            std::memset(image_ptr + (i * row_pitch) + unpadded_row_size, 0, remaining_bytes);
        }
        else
        {
            std::memset(image_ptr + i * row_pitch, 0, row_pitch);
        }
    }

    err = clEnqueueUnmapMemObject(command_queue, matrix_b_mem, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping matrix B image." << "\n";
        std::exit(err);
    }

    /*
     * Matrix C
     */

    cl_image_format matrix_c_format;
    matrix_c_format.image_channel_order     = CL_RGBA;
    matrix_c_format.image_channel_data_type = CL_HALF_FLOAT;

    cl_image_desc matrix_c_desc;
    std::memset(&matrix_c_desc, 0, sizeof(matrix_c_desc));
    matrix_c_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_c_desc.image_width     = ((matrix_c.width + 3) / 4);
    matrix_c_desc.image_height    = ((matrix_c.height + 7) / 8) * 8;
    matrix_c_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(matrix_c_format, matrix_c_desc);

    cl_mem_ion_host_ptr matrix_c_ion_buf = wrapper.make_ion_buffer_for_nonplanar_image(matrix_c_format,
                                                                                       matrix_c_desc);
    cl_mem matrix_c_mem = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &matrix_c_format,
            &matrix_c_desc,
            &matrix_c_ion_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for matrix C." << "\n";
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

    const cl_int matrix_a_width = matrix_a.width;
    err = clSetKernelArg(kernel, 3, sizeof(matrix_a_width), &matrix_a_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Run the kernel.
     */

    const size_t global_work_size[] = {matrix_b_desc.image_width, matrix_a_desc.image_height / 8};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2,
            NULL,
            global_work_size,
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

    const size_t  matrix_c_region[] = {matrix_c_desc.image_width, matrix_c_desc.image_height, 1};
    cl_half *out_image_ptr = static_cast<cl_half *>(clEnqueueMapImage(
            command_queue,
            matrix_c_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            matrix_c_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage for matrix C." << "\n";
        std::exit(err);
    }

    for (size_t i = 0; i < static_cast<size_t>(matrix_c.height); ++i)
    {

        for (size_t j = 0; j < static_cast<size_t>(matrix_c.width); ++j)
        {
            const size_t idx = i * matrix_c.width + j;
            matrix_c.elements[idx] = to_float(*(out_image_ptr + (i * row_pitch / sizeof(cl_half)) + j));
        }
    }

    err = clEnqueueUnmapMemObject(command_queue, matrix_c_mem, out_image_ptr, 0, NULL, NULL);
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
