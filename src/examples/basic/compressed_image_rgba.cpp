//--------------------------------------------------------------------------------------
// File: compressed_image_rgba.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tuple>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *HELP_MESSAGE = "\n"
"Usage: compressed_image_rgba <source image data file> <output image data file>\n"
"\n"
"Demonstrates use of compressed images using Qualcomm extensions to OpenCL.\n"
"The input RGBA image is compressed and then decompressed, with the result written\n"
"to the specified output file for comparison. (The compression is not lossy so\n"
"they are identical.)\n"
"\n"
"Compressed image formats may be saved to disk, however be advised that the format\n"
"is specific to each GPU.\n";

static const char *PROGRAM_SOURCE[] = {
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

// Some macros for use with tuples below
#define SRC_IMG(x) std::get<0>((x))
#define DST_IMG(x) std::get<1>((x))
#define WIDTH(x)   std::get<2>((x))
#define HEIGHT(x)  std::get<3>((x))

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

    cl_wrapper   wrapper;
    cl_program   program             = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel    blit_kernel         = wrapper.make_kernel("blit", program);
    cl_context   context             = wrapper.get_context();
    rgba_image_t src_rgba_image_info = load_rgba_image_data(src_image_filename);
    cl_int       err                 = 0;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_compressed_image"))
    {
        std::cerr << "Extension cl_qcom_compressed_image needed for reading/writing compressed images is not supported.\n";
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

    const std::vector<cl_image_format> formats = get_image_formats(context, CL_MEM_READ_WRITE | CL_MEM_COMPRESSED_IMAGE_QCOM);
    if (!is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_RGBA, CL_UNORM_INT8}))
    {
        std::cerr << "For this example your device must support read-write CL_QCOM_COMPRESSED_RGBA with CL_UNORM_INT8 "
                     "image format, but it does not.\n";
        std::cerr << "Supported read-write compressed image formats include:\n";
        print_formats(formats);
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 1: Create suitable ion buffer-backed CL images.
     */

    cl_image_format src_rgba_format;
    src_rgba_format.image_channel_order     = CL_RGBA;
    src_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_rgba_desc;
    std::memset(&src_rgba_desc, 0, sizeof(src_rgba_desc));
    src_rgba_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    src_rgba_desc.image_width     = src_rgba_image_info.width;
    src_rgba_desc.image_height    = src_rgba_image_info.height;
    src_rgba_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(src_rgba_format, src_rgba_desc);

    cl_mem_ion_host_ptr src_rgba_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(src_rgba_format, src_rgba_desc);
    cl_mem src_rgba_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_rgba_format,
            &src_rgba_desc,
            &src_rgba_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        std::exit(err);
    }

    cl_image_format compressed_rgba_format;
    compressed_rgba_format.image_channel_order     = CL_QCOM_COMPRESSED_RGBA;
    compressed_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_rgba_desc;
    std::memset(&compressed_rgba_desc, 0, sizeof(compressed_rgba_desc));
    compressed_rgba_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_rgba_desc.image_width  = src_rgba_image_info.width;
    compressed_rgba_desc.image_height = src_rgba_image_info.height;

    cl_mem_ion_host_ptr compressed_rgba_ion_mem = wrapper.make_ion_buffer_for_compressed_image(compressed_rgba_format,
                                                                                               compressed_rgba_desc);
    cl_mem compressed_rgba_image = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_rgba_format,
            &compressed_rgba_desc,
            &compressed_rgba_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image." << "\n";
        std::exit(err);
    }

    cl_image_format out_rgba_format;
    out_rgba_format.image_channel_order     = CL_RGBA;
    out_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_rgba_desc;
    std::memset(&out_rgba_desc, 0, sizeof(out_rgba_desc));
    out_rgba_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    out_rgba_desc.image_width     = src_rgba_image_info.width;
    out_rgba_desc.image_height    = src_rgba_image_info.height;
    out_rgba_desc.image_row_pitch = wrapper.get_ion_image_row_pitch(out_rgba_format, out_rgba_desc);

    cl_mem_ion_host_ptr out_rgba_ion_mem = wrapper.make_ion_buffer_for_nonplanar_image(out_rgba_format, out_rgba_desc);
    cl_mem out_rgba_image = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_rgba_format,
            &out_rgba_desc,
            &out_rgba_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Copy data to input image.
     */

    cl_command_queue command_queue = wrapper.get_command_queue();
    const size_t     origin[]      = {0, 0, 0};
    const size_t     src_region[]  = {src_rgba_desc.image_width, src_rgba_desc.image_height, 1};
    size_t           row_pitch     = 0;
    unsigned char   *image_ptr     = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_rgba_image,
            CL_TRUE,
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
        std::cerr << "Error " << err << " mapping source image for writing." << "\n";
        std::exit(err);
    }

    // Copies image data to the ION buffer from the host
    for (uint32_t i = 0; i < src_rgba_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                         + i * row_pitch,
                src_rgba_image_info.pixels.data() + i * src_rgba_desc.image_width * 4,
                src_rgba_desc.image_width * 4
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_rgba_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Set up other kernel arguments
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
     * Step 4: Run the kernel separately for y- and uv-planes
     */

    // Impromptu data structure to cut down on code duplication to enqueue kernels.
    std::array<std::tuple<cl_mem, cl_mem, size_t, size_t>, 2> kernel_args{
        /*              source plane,          destination plane,     image width,               image height*/
        std::make_tuple(src_rgba_image,        compressed_rgba_image, src_rgba_desc.image_width, src_rgba_desc.image_height),
        std::make_tuple(compressed_rgba_image, out_rgba_image,        src_rgba_desc.image_width, src_rgba_desc.image_height),
    };

    for (size_t i = 0; i < kernel_args.size(); ++i)
    {
        cl_mem src_plane  = SRC_IMG(kernel_args[i]);
        cl_mem dest_plane = DST_IMG(kernel_args[i]);

        err = clSetKernelArg(blit_kernel, 0, sizeof(src_plane), &src_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for blit kernel." << "\n";
            std::exit(err);
        }

        err = clSetKernelArg(blit_kernel, 1, sizeof(dest_plane), &dest_plane);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for blit kernel." << "\n";
            std::exit(err);
        }


        err = clSetKernelArg(blit_kernel, 2, sizeof(sampler), &sampler);
        if (err != CL_SUCCESS)
        {
            std::cerr << "On iteration " << i << ":\n";
            std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for blit kernel." << "\n";
            std::exit(err);
        }

        const size_t work_size[] = {WIDTH(kernel_args[i]), HEIGHT(kernel_args[i])};
        err = clEnqueueNDRangeKernel(
                command_queue,
                blit_kernel,
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
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for blit kernel." << "\n";
            std::exit(err);
        }
    }

    /*
     * Step 5: Copy the data out of the ion buffer for each plane.
     */

    rgba_image_t out_rgba_image_info;
    out_rgba_image_info.width  = out_rgba_desc.image_width;
    out_rgba_image_info.height = out_rgba_desc.image_height;
    out_rgba_image_info.pixels.resize(out_rgba_image_info.width * out_rgba_image_info.height * 4);

    const size_t out_region[] = {out_rgba_desc.image_width, out_rgba_desc.image_height, 1};
    row_pitch                 = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_rgba_image,
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
        std::cerr << "Error " << err << " mapping dest image for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < out_rgba_desc.image_height; ++i)
    {
        std::memcpy(
                out_rgba_image_info.pixels.data() + i * out_rgba_desc.image_width * 4,
                image_ptr                         + i * row_pitch,
                out_rgba_desc.image_width * 4
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_rgba_image, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    save_rgba_image_data(out_image_filename, out_rgba_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_rgba_image);
    clReleaseMemObject(compressed_rgba_image);
    clReleaseMemObject(out_rgba_image);

    return 0;
}
