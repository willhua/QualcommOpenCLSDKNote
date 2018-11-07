//--------------------------------------------------------------------------------------
// File: unpacked_to_mipi10.cpp
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
"Usage: unpacked_to_mipi10 <source image data file> <output image data file>\n"
"\n"
"Converts a single-channel unpacked 16-bit image to MIPI10 format.\n";

static const char *PROGRAM_SOURCE[] = {
"__kernel void pack(__read_only  image2d_t unpacked_image,\n",
"                   __write_only image2d_t packed_image,\n",
"                                sampler_t sampler)\n",
"{\n",
"    const int  wid_x      = get_global_id(0);\n",
"    const int  wid_y      = get_global_id(1);\n",
"    const int2 coord      = (int2)(4 * wid_x, wid_y);\n",
"    const float4 pixels[] = {\n",
"        read_imagef(unpacked_image, sampler, coord + (int2)(0, 0)),\n",
"        read_imagef(unpacked_image, sampler, coord + (int2)(1, 0)),\n",
"        read_imagef(unpacked_image, sampler, coord + (int2)(2, 0)),\n",
"        read_imagef(unpacked_image, sampler, coord + (int2)(3, 0)),\n",
"    };\n",
"    float      out_pixel[] = {pixels[0].x, pixels[1].x, pixels[2].x, pixels[3].x};\n",
"    const int2 write_coord = (int2)(wid_x, wid_y);\n",
"    qcom_write_imagefv_4x1_n10m00(packed_image, write_coord, out_pixel);\n",
"}\n"
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify source and output images.\n";
        std::cerr << HELP_MESSAGE;
        std::exit(EXIT_SUCCESS);
    }
    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper wrapper;
    cl_program                   program              = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel                    kernel               = wrapper.make_kernel("pack", program);
    cl_context                   context              = wrapper.get_context();
    cl_command_queue             command_queue        = wrapper.get_command_queue();
    single_channel_int16_image_t src_int16_image_info = load_single_channel_image_data(src_image_filename);

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for MIPI10 data type is not supported.\n";
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
     *         Note the source image has the same layout as with bayer_mipi10_to_rgba.cpp example.
     *         The difference is how such images are addressed on the GPU.
     */

    cl_image_format src_format;
    src_format.image_channel_order     = CL_R;
    src_format.image_channel_data_type = CL_UNORM_INT16;

    cl_image_desc src_desc;
    std::memset(&src_desc, 0, sizeof(src_desc));
    src_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    src_desc.image_width     = src_int16_image_info.width;
    src_desc.image_height    = src_int16_image_info.height;
    src_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(src_format, src_desc);

    cl_mem_ion_host_ptr src_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(src_format, src_desc);
    cl_int              err         = 0;
    cl_mem              src_image   = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_format,
            &src_desc,
            &src_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        std::exit(err);
    }

    const size_t   origin[]     = {0, 0, 0};
    size_t         row_pitch    = 0 ;
    const size_t   src_region[] = {src_desc.image_width, src_desc.image_height, 1};
    unsigned char *image_ptr    = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_image,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            src_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage for source image." << "\n";
        std::exit(err);
    }

    // Copies image data from the host to the ION buffer
    for (uint32_t i = 0; i < src_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * src_desc.image_row_pitch,
                src_int16_image_info.pixels.data() + i * src_desc.image_width * 2,
                src_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image." << "\n";
        std::exit(err);
    }

    cl_image_format out_format;
    out_format.image_channel_order     = CL_R;
    out_format.image_channel_data_type = CL_QCOM_UNORM_MIPI10;

    cl_image_desc out_desc;
    std::memset(&out_desc, 0, sizeof(out_desc));
    out_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    out_desc.image_width     = src_int16_image_info.width;
    out_desc.image_height    = src_int16_image_info.height;
    out_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(out_format, out_desc);

    cl_mem_ion_host_ptr out_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(out_format, out_desc);
    cl_mem out_image = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_format,
            &out_desc,
            &out_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Set up kernel arguments and run the kernel.
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

    err = clSetKernelArg(kernel, 0, sizeof(src_image), &src_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel, 1, sizeof(out_image), &out_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2." << "\n";
        std::exit(err);
    }

    const size_t global_work_size[] = {out_desc.image_width / 4, out_desc.image_height};
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Copy the data out of the ion buffer.
     */

    bayer_mipi10_image_t out_image_info;
    out_image_info.width  = out_desc.image_width;
    out_image_info.height = out_desc.image_height;
    out_image_info.pixels.resize((out_image_info.width / 4 * 5) * out_image_info.height);

    const size_t out_region[] = {out_desc.image_width, out_desc.image_height, 1};
    row_pitch                 = 0;
    image_ptr                 = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_image,
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
        std::cerr << "Error " << err << " mapping dest image buffer for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < out_desc.image_height; ++i)
    {
        std::memcpy(
                out_image_info.pixels.data() + i * out_desc.image_width / 4 * 5,
                image_ptr                    + i * row_pitch,
                out_desc.image_width / 4 * 5
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    save_bayer_mipi_10_image_data(out_image_filename, out_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_image);
    clReleaseMemObject(out_image);

    return 0;
}

