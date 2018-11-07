//--------------------------------------------------------------------------------------
// File: io_coherent_ion_buffers.cpp
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
#include <fstream>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"

// Library includes
#include <CL/cl.h>

static const char *HELP_MESSAGE = "\n"
"Usage: io_coherent_ion_buffers <input> <output>\n"
"\n"
"This example copies the input file to the output file.\n"
"It uses io-coherent ION buffers.\n";

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

    const auto          fin_begin   = fin.tellg();

    fin.seekg(0, std::ios::end);
    const auto          fin_end     = fin.tellg();
    const size_t        buf_size    = static_cast<size_t>(fin_end - fin_begin);
    cl_mem_ion_host_ptr src_buf_ion = wrapper.make_iocoherent_ion_buffer(buf_size);

    cl_mem src_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            buf_size,
            &src_buf_ion,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for source file." << "\n";
        std::exit(err);
    }

    char *buf_ptr = static_cast<char *>(clEnqueueMapBuffer(
            command_queue,
            src_buffer,
            CL_BLOCKING,
            CL_MAP_WRITE,
            0,
            buf_size,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source buffer for writing." << "\n";
        std::exit(err);
    }

    fin.seekg(0, std::ios::beg);
    fin.read(buf_ptr, buf_size);
    fin.close();

    err = clEnqueueUnmapMemObject(command_queue, src_buffer, buf_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source buffer." << "\n";
        std::exit(err);
    }

    cl_mem_ion_host_ptr out_buf_ion = wrapper.make_iocoherent_ion_buffer(buf_size);
    cl_mem out_buffer = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            buf_size,
            &out_buf_ion,
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
     * Step 2: Copy the data out of the ion buffer.
     */

    std::ofstream fout(out_filename, std::ios::binary);
    if (!fout)
    {
        std::cerr << "Couldn't open file " << out_filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    buf_ptr = static_cast<char *>(clEnqueueMapBuffer(
            command_queue,
            out_buffer,
            CL_BLOCKING,
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
        std::cerr << "Error " << err << " mapping output buffer for writing." << "\n";
        std::exit(err);
    }

    fout.write(buf_ptr, buf_size);
    fout.close();

    err = clEnqueueUnmapMemObject(command_queue, out_buffer, buf_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping output buffer." << "\n";
        std::exit(err);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(src_buffer);
    clReleaseMemObject(out_buffer);

    return 0;
}
