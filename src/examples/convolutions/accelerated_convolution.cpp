//--------------------------------------------------------------------------------------
// File: accelerated_convolution.cpp
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
"Usage: accelerated_convolution <source image data file> <output image data file>\n"
"Runs a kernel demonstrating Gaussian blur with Qualcomm intrinsic functions.\n"
"Additionally demonstrates that images and buffers can use the same underlying\n"
"ION memory, by writing to an OpenCL buffer and reading results from an image\n"
"that share the same ION buffer.\n";

static const char *PROGRAM_SOURCE[] = {
"__kernel void accelerated_convolution(__read_only image2d_t            src_image,\n",
"                                      __global    unsigned char       *dest_image,\n",
"                                                  int                  dest_row_pitch,\n",
"                                                  sampler_t            sampler,\n",
"                                      __read_only qcom_weight_image_t  weight_image)\n",
"{\n",
"    const int    wid_x           = get_global_id(0);\n",
"    const int    wid_y           = get_global_id(1);\n",
"    const float2 src_coord       = (float2)(wid_x * 2, wid_y * 2);\n",
"    const float4 convolved_vals  = \n",
"                   (qcom_convolve_imagef(src_image, sampler, src_coord,                    weight_image).x,\n",
"                    qcom_convolve_imagef(src_image, sampler, src_coord + (float2)(1., 0.), weight_image).x,\n",
"                    qcom_convolve_imagef(src_image, sampler, src_coord + (float2)(0., 1.), weight_image).x,\n",
"                    qcom_convolve_imagef(src_image, sampler, src_coord + (float2)(1., 1.), weight_image).x);\n",
"    const uchar2 res1            = convert_uchar2(convolved_vals.s01 * 255.f);\n",
"    const uchar2 res2            = convert_uchar2(convolved_vals.s23 * 255.f);\n",
"    vstore2(res1, wid_x, dest_image + (2 * wid_y       * dest_row_pitch));\n",
"    vstore2(res2, wid_x, dest_image + ((2 * wid_y + 1) * dest_row_pitch));\n",
"}\n"
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

static const cl_uint  CONV_FILTER_WIDTH  = 3;
static const cl_uint  CONV_FILTER_HEIGHT = 3;

static const cl_half sixteenth = to_half(1.f / 16.f);
static const cl_half eighth    = to_half(1.f / 8.f);
static const cl_half fourth    = to_half(1.f / 4.f);

// This filter corresponds to a 3x3 Gaussian blur.
static cl_half CONVOLUTION_FILTER[CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH] = {
    sixteenth, eighth, sixteenth,
    eighth,    fourth, eighth,
    sixteenth, eighth, sixteenth
};

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
    cl_program   program             = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
    cl_kernel    y_plane_kernel      = wrapper.make_kernel("accelerated_convolution", program);
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

    if (!wrapper.check_extension_support("cl_qcom_accelerated_image_ops"))
    {
        std::cerr << "Extension cl_qcom_accelerated_image_ops needed for qcom_convolve_imagef is not supported.\n";
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
     * Step 1: Create suitable ion buffer-backed CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 1 for deriving child images.)
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

    // Note: images and buffers can be backed by the same underlying ION memory.
    const int out_img_row_pitch = static_cast<int>(wrapper.get_ion_image_row_pitch(out_nv12_format, out_nv12_desc));
    cl_mem out_nv12_buffer = clCreateBuffer(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            out_img_row_pitch * out_nv12_desc.image_height,
            &out_nv12_ion_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer." << "\n";
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

    cl_image_format weight_image_format;
    weight_image_format.image_channel_order     = CL_R;
    weight_image_format.image_channel_data_type = CL_HALF_FLOAT;

    cl_weight_image_desc_qcom weight_image_desc;
    std::memset(&weight_image_desc, 0, sizeof(weight_image_desc));
    weight_image_desc.image_desc.image_type        = CL_MEM_OBJECT_WEIGHT_IMAGE_QCOM;
    weight_image_desc.image_desc.image_width       = CONV_FILTER_WIDTH;
    weight_image_desc.image_desc.image_height      = CONV_FILTER_HEIGHT;
    weight_image_desc.image_desc.image_array_size  = 1;
    weight_image_desc.image_desc.image_row_pitch   = 0;
    weight_image_desc.image_desc.image_slice_pitch = 0;

    weight_image_desc.weight_desc.center_coord_x = 1;
    weight_image_desc.weight_desc.center_coord_y = 1;
    weight_image_desc.weight_desc.flags          = 0;

    cl_mem weight_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &weight_image_format,
            reinterpret_cast<cl_image_desc *>(&weight_image_desc),
            static_cast<void *>(CONVOLUTION_FILTER),
            &err
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for weight image." << "\n";
        std::exit(err);
    }

    /*
     * Step 5: Run the kernel separately for y-plane only
     */

    err = clSetKernelArg(y_plane_kernel, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0 for y-plane kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(y_plane_kernel, 1, sizeof(out_nv12_buffer), &out_nv12_buffer);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1 for y-plane kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(y_plane_kernel, 2, sizeof(out_img_row_pitch), &out_img_row_pitch);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2 for y-plane kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(y_plane_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3 for y-plane kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(y_plane_kernel, 4, sizeof(weight_image), &weight_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4 for y-plane kernel." << "\n";
        std::exit(err);
    }

    const size_t y_plane_work_size[] = {out_y_plane_desc.image_width / 2, out_y_plane_desc.image_height / 2};
    err = clEnqueueNDRangeKernel(
            command_queue,
            y_plane_kernel,
            2,
            NULL,
            y_plane_work_size,
            NULL,
            0,
            NULL,
            NULL
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for y-plane kernel." << "\n";
        std::exit(err);
    }

    clFinish(command_queue);

    /*
     * Step 6: Copy the data out of the ion buffer for each plane.
     */

    nv12_image_t out_nv12_image_info;
    out_nv12_image_info.y_width  = out_nv12_desc.image_width;
    out_nv12_image_info.y_height = out_nv12_desc.image_height;
    out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
    out_nv12_image_info.uv_plane = src_nv12_image_info.uv_plane;

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

    clFinish(command_queue);

    save_nv12_image_data(out_image_filename, out_nv12_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(weight_image);
    clReleaseSampler(sampler);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(out_nv12_buffer);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_nv12_image);

    return 0;
}
