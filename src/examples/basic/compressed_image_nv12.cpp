//--------------------------------------------------------------------------------------
// File: compressed_image_nv12.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017-2018 QUALCOMM Technologies, Inc.
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
"Usage: compressed_image_nv12 <source image data file> <output image data file>\n"
"\n"
"Demonstrates use of compressed images using Qualcomm extensions to OpenCL.\n"
"The input NV12 image is compressed and then decompressed, with the result written\n"
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
    nv12_image_t src_nv12_image_info = load_nv12_image_data(src_image_filename);

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

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

    std::vector<cl_image_format> formats;

    formats = get_image_formats(context, CL_MEM_READ_ONLY | CL_MEM_OTHER_IMAGE_QCOM | CL_MEM_COMPRESSED_IMAGE_QCOM);
    if (!is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_NV12, CL_UNORM_INT8}))
    {
        std::cerr << "For this example your device must support read-only CL_QCOM_COMPRESSED_NV12 with CL_UNORM_INT8 "
                     "image format, but it does not.\n";
        std::cerr << "Supported read-only compressed image formats include:\n";
        print_formats(formats);
        std::exit(EXIT_FAILURE);
    }

    formats = get_image_formats(context, CL_MEM_READ_WRITE | CL_MEM_OTHER_IMAGE_QCOM | CL_MEM_COMPRESSED_IMAGE_QCOM);
    const bool rw_formats_supported =
            is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_NV12_Y,  CL_UNORM_INT8}) &&
            is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_NV12_UV, CL_UNORM_INT8});
    if (!rw_formats_supported)
    {
        std::cerr << "For this example your device must support read-write CL_QCOM_COMPRESSED_NV12_Y and "
                     "CL_QCOM_COMPRESSED_NV12_UV with CL_UNORM_INT8 image format, but it does not.\n";
        std::cerr << "Supported read-write compressed image formats include:\n";
        print_formats(formats);
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 1: Create suitable ion buffer-backed CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */

    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order     = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc;
    std::memset(&src_nv12_desc, 0, sizeof(src_nv12_desc));
    src_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width  = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_int err = 0;
    cl_mem_ion_host_ptr src_nv12_ion_mem = wrapper.make_ion_buffer_for_yuv_image(src_nv12_format, src_nv12_desc);
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

    cl_image_format compressed_nv12_format;
    compressed_nv12_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12;
    compressed_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_nv12_desc;
    std::memset(&compressed_nv12_desc, 0, sizeof(compressed_nv12_desc));
    compressed_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_nv12_desc.image_width  = src_nv12_image_info.y_width;
    compressed_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_ion_host_ptr compressed_nv12_ion_mem = wrapper.make_ion_buffer_for_compressed_image(compressed_nv12_format,
                                                                                               compressed_nv12_desc);
    cl_mem compressed_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_nv12_format,
            &compressed_nv12_desc,
            &compressed_nv12_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image." << "\n";
        std::exit(err);
    }

    cl_image_format out_nv12_format;
    out_nv12_format.image_channel_order     = CL_QCOM_NV12;
    out_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_nv12_desc;
    std::memset(&out_nv12_desc, 0, sizeof(out_nv12_desc));
    out_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_nv12_desc.image_width  = src_nv12_image_info.y_width;
    out_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_ion_host_ptr out_nv12_ion_mem = wrapper.make_ion_buffer_for_yuv_image(out_nv12_format, out_nv12_desc);
    cl_mem out_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_nv12_format,
            &out_nv12_desc,
            &out_nv12_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
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

    cl_image_format src_uv_plane_format;
    src_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    src_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_uv_plane_desc;
    std::memset(&src_uv_plane_desc, 0, sizeof(src_uv_plane_desc));
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    src_uv_plane_desc.image_width  = src_nv12_image_info.y_width;
    src_uv_plane_desc.image_height = src_nv12_image_info.y_height;
    src_uv_plane_desc.mem_object   = src_nv12_image;

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

    cl_image_format compressed_y_plane_format;
    compressed_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_Y;
    compressed_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_y_plane_desc;
    std::memset(&compressed_y_plane_desc, 0, sizeof(compressed_y_plane_desc));
    compressed_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_y_plane_desc.image_width  = compressed_nv12_desc.image_width;
    compressed_y_plane_desc.image_height = compressed_nv12_desc.image_height;
    compressed_y_plane_desc.mem_object   = compressed_nv12_image;

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
    compressed_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_UV;
    compressed_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_uv_plane_desc;
    std::memset(&compressed_uv_plane_desc, 0, sizeof(compressed_uv_plane_desc));
    compressed_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    compressed_uv_plane_desc.image_width  = compressed_nv12_desc.image_width;
    compressed_uv_plane_desc.image_height = compressed_nv12_desc.image_height;
    compressed_uv_plane_desc.mem_object   = compressed_nv12_image;

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

    cl_image_format out_y_plane_format;
    out_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    out_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_y_plane_desc;
    std::memset(&out_y_plane_desc, 0, sizeof(out_y_plane_desc));
    out_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_y_plane_desc.image_width  = out_nv12_desc.image_width;
    out_y_plane_desc.image_height = out_nv12_desc.image_height;
    out_y_plane_desc.mem_object   = out_nv12_image;

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
    out_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    out_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    out_uv_plane_desc.image_width  = out_nv12_desc.image_width;
    out_uv_plane_desc.image_height = out_nv12_desc.image_height;
    out_uv_plane_desc.mem_object   = out_nv12_image;

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

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
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
                src_nv12_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width,
                src_uv_plane_desc.image_width
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

    // Impromptu data structure to cut down on code duplication to enqueue kernels.
    std::array<std::tuple<cl_mem, cl_mem, size_t, size_t>, 4> kernel_args{
        /*              source plane,        destination plane,   image width,                      image height*/
        std::make_tuple(src_y_plane,         compressed_y_plane,  out_y_plane_desc.image_width,     out_y_plane_desc.image_height),
        std::make_tuple(src_uv_plane,        compressed_uv_plane, out_y_plane_desc.image_width / 2, out_y_plane_desc.image_height / 2),
        std::make_tuple(compressed_y_plane,  out_y_plane,         out_y_plane_desc.image_width,     out_y_plane_desc.image_height),
        std::make_tuple(compressed_uv_plane, out_uv_plane,        out_y_plane_desc.image_width / 2, out_y_plane_desc.image_height / 2)
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

    clFinish(command_queue); // Note the blocking read below will flush/finish the queue, but this is included for clarity.

    /*
     * Step 6: Copy the data out of the ion buffer for each plane.
     */

    nv12_image_t out_nv12_image_info;
    out_nv12_image_info.y_width  = out_nv12_desc.image_width;
    out_nv12_image_info.y_height = out_nv12_desc.image_height;
    out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
    out_nv12_image_info.uv_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height / 2);

    const size_t out_y_region[] = {out_y_plane_desc.image_width, out_y_plane_desc.image_height, 1};
    row_pitch                   = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_y_plane,
            CL_TRUE,
            CL_MAP_READ,
            origin,
            out_y_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < out_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                out_nv12_image_info.y_plane.data() + i * out_y_plane_desc.image_width,
                image_ptr                          + i * row_pitch,
                out_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
        std::exit(err);
    }

    const size_t out_uv_region[] = {out_uv_plane_desc.image_width / 2, out_uv_plane_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_uv_plane,
            CL_TRUE,
            CL_MAP_READ,
            origin,
            out_uv_region,
            &row_pitch,
            NULL,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the ION buffer to the host
    for (uint32_t i = 0; i < out_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                out_nv12_image_info.uv_plane.data() + i * out_uv_plane_desc.image_width,
                image_ptr                           + i * row_pitch,
                out_uv_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane buffer." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    save_nv12_image_data(out_image_filename, out_nv12_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_nv12_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_nv12_image);

    return 0;
}
