//--------------------------------------------------------------------------------------
// File: hello_world.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <fstream>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"

// Library includes
#include <CL/cl.h>

static const char *HELP_MESSAGE = "\n"
"Usage: hello_world <input> <output>\n"
"\n"
"This example copies the input file to the output file.\n"
"Use it to test your build tools.\n";

static const char *PROGRAM_SOURCE[] = {
"__kernel void copy(__global char *src,\n",
"                   __global char *dst\n",
"                   )\n",
"{\n",
"    uint wid_x = get_global_id(0);\n",
"    dst[wid_x] = src[wid_x];\n",
"}\n"
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify source and destination files.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string src_filename(argv[1]);
    const std::string out_filename(argv[2]);

    cl_wrapper       wrapper;
    cl_program       program         = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel        kernel          = wrapper.make_kernel("copy", program);
    cl_context       context         = wrapper.get_context();
    cl_command_queue command_queue   = wrapper.get_command_queue();
    cl_int           err             = CL_SUCCESS;

    /*
     * Step 0: Create CL buffers.
     */

    std::ifstream fin(src_filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Couldn't open file " << src_filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    const auto        fin_begin = fin.tellg();

    fin.seekg(0, std::ios::end);
    const auto        fin_end   = fin.tellg();
    const size_t      buf_size  = static_cast<size_t>(fin_end - fin_begin);
    std::vector<char> buf(buf_size);

    fin.seekg(0, std::ios::beg);
    fin.read(buf.data(), buf_size);

    cl_mem src_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            buf_size,
            buf.data(),
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for source file." << "\n";
        std::exit(err);
    }

    cl_mem out_buffer = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            buf_size,
            NULL,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for output file." << "\n";
        std::exit(err);
    }

    /*
     * Step 1: Set up kernel arguments and run the kernel.
     */

    err = clSetKernelArg(kernel, 0, sizeof(src_buffer), &src_buffer);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel, 1, sizeof(out_buffer), &out_buffer);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            1,
            NULL,
            &buf_size,
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
     * Step 2: Copy the data out of the ion buffer for each plane.
     */

    char *mapped_ptr = static_cast<char *>(clEnqueueMapBuffer(
            command_queue,
            out_buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            buf_size,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping output buffer." << "\n";
        std::exit(err);
    }

    std::ofstream fout(out_filename, std::ios::binary);
    if (!fout)
    {
        std::cerr << "Couldn't open file " << out_filename << "\n";
        std::exit(EXIT_FAILURE);
    }
    fout.write(mapped_ptr, buf_size);
    fout.close();

    err = clEnqueueUnmapMemObject(command_queue, out_buffer, mapped_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping output buffer." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(src_buffer);
    clReleaseMemObject(out_buffer);

    return 0;
}
