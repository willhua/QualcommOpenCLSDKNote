//--------------------------------------------------------------------------------------
// File: cl_wrapper.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------
#include "cl_wrapper.h"
#include "util.h"

#include <CL/cl.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <cstdlib>
#include <iostream>

cl_wrapper::cl_wrapper()
{
    cl_platform_id platform;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetPlatformIDs." << "\n";
        std::exit(err);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_device, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceIDs." << "\n";
        std::exit(err);
    }

    m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateContext." << "\n";
        std::exit(err);
    }

    m_cmd_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateCommandQueue." << "\n";
        std::exit(err);
    }

    // ION stuff
#if USES_LIBION
    m_ion_device_fd = ion_open();
    if (m_ion_device_fd < 0)
    {
        std::cerr << "Error with ion_open()\n";
        std::exit(EXIT_FAILURE);
    }
#else
    m_ion_device_fd = open("/dev/ion", O_RDONLY);
    if (m_ion_device_fd < 0)
    {
        std::cerr << "Error " << errno << " opening /dev/ion : " << strerror(errno) << "\n";
        std::exit(errno);
    }
#endif
}

cl_wrapper::~cl_wrapper()
{
    // ION stuff
    for (auto &pair: m_ion_host_ptrs)
    {
        if (munmap(pair.first, pair.second) < 0)
        {
            std::cerr << "Error " << errno << " munmap-ing ion alloc: " << strerror(errno) << "\n";
            std::exit(errno);
        }
        pair.first = nullptr;
    }

    for (const auto fd : m_file_descs)
    {
        if (close(fd) < 0)
        {
            std::cerr << "Error " << errno << " closing ion allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }

#if USES_LIBION
    if (ion_close(m_ion_device_fd) < 0)
    {
        std::cerr << "Error closing ion device fd.\n";
        std::exit(EXIT_FAILURE);
    }
#else
    for (const auto &handle_data : m_handle_data)
    {
        if (ioctl(m_ion_device_fd, ION_IOC_FREE, &handle_data) < 0)
        {
            std::cerr << "Error " << errno << " freeing ion alloc with ioctl: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }

    if (close(m_ion_device_fd) < 0)
    {
        std::cerr << "Error " << errno << " closing ion device fd: " << strerror(errno) << "\n";
        std::exit(errno);
    }
#endif

    // OpenCL stuff
    for (auto kernel : m_kernels)
    {
        clReleaseKernel(kernel);
    }
    clReleaseCommandQueue(m_cmd_queue);
    for (auto program : m_programs)
    {
        clReleaseProgram(program);
    }
    clReleaseContext(m_context);
}

cl_kernel cl_wrapper::make_kernel(const std::string &kernel_name, cl_program program)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateKernel." << "\n";
        std::exit(err);
    }
    m_kernels.push_back(kernel);
    return kernel;
}

cl_context cl_wrapper::get_context() const
{
    return m_context;
}

cl_command_queue cl_wrapper::get_command_queue() const
{
    return m_cmd_queue;
}

cl_program cl_wrapper::make_program(const char **program_source, cl_uint program_source_len)
{
    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(m_context, program_source_len, program_source, NULL, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateProgramWithSource." << "\n";
        std::exit(err);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clBuildProgram.\n";
        static const size_t LOG_SIZE = 2048;
        char log[LOG_SIZE];
        log[0] = 0;
        err = clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, NULL);
        if (err == CL_INVALID_VALUE)
        {
            std::cerr << "There was a build error, but there is insufficient space allocated to show the build logs.\n";
        }
        else
        {
            std::cerr << "Build error:\n" << log << "\n";
        }
        std::exit(EXIT_FAILURE);
    }

    m_programs.push_back(program);

    return program;
}

cl_mem_ion_host_ptr
cl_wrapper::make_ion_buffer_for_yuv_image(const cl_image_format &img_format, const cl_image_desc &img_desc)
{
    const size_t effective_img_height = ((img_desc.image_height + 31) / 32) * 32; // Round up to the nearest multiple of 32
    const size_t img_row_pitch        = get_ion_image_row_pitch(img_format, img_desc);

    cl_int err;
    size_t padding_in_bytes = 0;

    err = clGetDeviceInfo(m_device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for padding." << "\n";
        std::exit(err);
    }

    const size_t y_plane_bytes  = img_row_pitch * effective_img_height;
    const size_t uv_plane_bytes = img_row_pitch * effective_img_height / 2;
    const size_t total_bytes    = y_plane_bytes + uv_plane_bytes + padding_in_bytes;

    return make_ion_buffer(total_bytes);
}

static std::string init_extension_string(cl_device_id device)
{
    static const size_t BUF_SIZE = 1024;
    char                extensions_buf[BUF_SIZE];
    std::memset(extensions_buf, 0, sizeof(extensions_buf));
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions_buf), extensions_buf, NULL);
    return std::string(extensions_buf);
}

bool cl_wrapper::check_extension_support(const std::string &desired_extension) const
{
    static const std::string extensions = init_extension_string(m_device);
    if (extensions.size() == 0)
    {
        std::cerr << "Couldn't identify available OpenCL extensions\n";
        std::exit(EXIT_FAILURE);
    }

    return extensions.find(desired_extension) != std::string::npos;
}

size_t cl_wrapper::get_ion_image_row_pitch(const cl_image_format &img_format, const cl_image_desc &img_desc) const
{
    size_t img_row_pitch = 0;
    cl_int err = clGetDeviceImageInfoQCOM(m_device, img_desc.image_width, img_desc.image_height, &img_format,
                                          CL_IMAGE_ROW_PITCH, sizeof(img_row_pitch), &img_row_pitch, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clGetDeviceImageInfoQCOM for CL_IMAGE_ROW_PITCH." << "\n";
        std::exit(err);
    }
    return img_row_pitch;
}

cl_mem_ion_host_ptr
cl_wrapper::make_ion_buffer_for_compressed_image(cl_image_format img_format, const cl_image_desc &img_desc)
{
    const bool valid_compressed_nv12 = img_format.image_channel_order        == CL_QCOM_COMPRESSED_NV12
                                       && img_format.image_channel_data_type == CL_UNORM_INT8;
    const bool valid_compressed_p010 = img_format.image_channel_order        == CL_QCOM_COMPRESSED_P010
                                       && img_format.image_channel_data_type == CL_QCOM_UNORM_INT10;
    const bool valid_compressed_tp10 = img_format.image_channel_order        == CL_QCOM_COMPRESSED_TP10
                                       && img_format.image_channel_data_type == CL_QCOM_UNORM_INT10;
    const bool valid_compressed_rgba = img_format.image_channel_order        == CL_QCOM_COMPRESSED_RGBA
                                       && img_format.image_channel_data_type == CL_UNORM_INT8;
    if (!valid_compressed_nv12 && !valid_compressed_p010 && !valid_compressed_tp10 && !valid_compressed_rgba)
    {
        std::cerr << "Unsupported image format for compressed image.\n";
        std::exit(EXIT_FAILURE);
    }

    static const size_t max_dims = 2048;
    if (img_desc.image_height > max_dims || img_desc.image_width > max_dims)
    {
        std::cerr << "For this example, the image dimensions must be less than or equal to " << max_dims << "\n";
        std::exit(EXIT_FAILURE);
    }

    // The size of this ION buffer will be sufficient to hold an image where each dimension is <= 2048.
    // This is a loose upper bound only, however the general calculation is not within the scope of these examples.
    static const size_t total_bytes = 12681216;

    return make_ion_buffer(total_bytes);
}

cl_mem_ion_host_ptr cl_wrapper::make_ion_buffer(size_t size)
{
    return make_ion_buffer_internal(size, 0, CL_MEM_HOST_UNCACHED_QCOM);
}

cl_mem_ion_host_ptr
cl_wrapper::make_ion_buffer_for_nonplanar_image(const cl_image_format &img_format, const cl_image_desc &img_desc)
{
    cl_int err;
    size_t padding_in_bytes = 0;

    (void) img_format; // Unused, for now

    err = clGetDeviceInfo(m_device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for padding." << "\n";
        std::exit(err);
    }

    const size_t total_bytes = img_desc.image_row_pitch * img_desc.image_height + padding_in_bytes;
    return make_ion_buffer(total_bytes);
}

size_t cl_wrapper::get_max_workgroup_size(cl_kernel kernel) const
{
    size_t result = 0;

    cl_int err = clGetKernelWorkGroupInfo(kernel, m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(result), &result, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetKernelWorkGroupInfo for CL_KERNEL_WORK_GROUP_SIZE." << "\n";
        std::exit(err);
    }

    return result;
}

cl_mem_ion_host_ptr cl_wrapper::make_iocoherent_ion_buffer(size_t size)
{
    return make_ion_buffer_internal(size, ION_FLAG_CACHED, CL_MEM_HOST_IOCOHERENT_QCOM);
}

cl_mem_ion_host_ptr cl_wrapper::make_ion_buffer_internal(size_t size, unsigned int ion_allocation_flags, cl_uint host_cache_policy)
{
    cl_int  err;
    cl_uint device_page_size;

    err = clGetDeviceInfo(m_device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for page size." << "\n";
        std::exit(err);
    }

#if USES_LIBION
    int fd = 0;
    err = ion_alloc_fd(m_ion_device_fd, size, device_page_size, ION_HEAP(ION_SYSTEM_HEAP_ID), ion_allocation_flags, &fd);
    if (err == -1)
    {
        std::cerr << "Error allocating ion memory\n";
        std::exit(EXIT_FAILURE);
    }

    void *host_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == host_addr)
    {
        close(fd);
        std::cerr << "Error " << errno << " mmapping fd to pointer: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    cl_mem_ion_host_ptr ion_mem;
    ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = host_cache_policy;
    ion_mem.ion_filedesc                   = fd;
    ion_mem.ion_hostptr                    = host_addr;

    m_ion_host_ptrs.push_back(std::make_pair(ion_mem.ion_hostptr, size));
    m_file_descs.push_back(fd);
#else // USES_LIBION
    ion_allocation_data allocation_data;
    allocation_data.len          = size;
    allocation_data.align        = device_page_size;
    allocation_data.heap_id_mask = ION_HEAP(ION_IOMMU_HEAP_ID);
    allocation_data.flags        = ion_allocation_flags;
    if (ioctl(m_ion_device_fd, ION_IOC_ALLOC, &allocation_data))
    {
        std::cerr << "Error " << errno << " allocating ion memory: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    ion_handle_data handle_data;
    ion_fd_data fd_data;
    handle_data.handle = allocation_data.handle;
    fd_data.handle     = allocation_data.handle;
    if (ioctl(m_ion_device_fd, ION_IOC_MAP, &fd_data))
    {
        ioctl(m_ion_device_fd, ION_IOC_FREE, &handle_data);
        std::cerr << "Error " << errno << " mapping ion memory to cpu-addressable fd: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    void *host_addr = mmap(NULL, allocation_data.len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_data.fd, 0);
    if (MAP_FAILED == host_addr)
    {
        close(fd_data.fd);
        ioctl(m_ion_device_fd, ION_IOC_FREE, &handle_data);
        std::cerr << "Error " << errno << " mmapping fd to pointer: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    cl_mem_ion_host_ptr ion_mem;
    ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = host_cache_policy;
    ion_mem.ion_filedesc                   = fd_data.fd;
    ion_mem.ion_hostptr                    = host_addr;

    m_ion_host_ptrs.push_back(std::make_pair(ion_mem.ion_hostptr, allocation_data.len));
    m_file_descs.push_back(fd_data.fd);
    m_handle_data.push_back(handle_data);
#endif // USES_LIBION

    return ion_mem;
}

cl_mem_ion_host_ptr
cl_wrapper::make_iocoherent_ion_buffer_for_yuv_image(const cl_image_format &img_format, const cl_image_desc &img_desc) {
    const size_t effective_img_height = ((img_desc.image_height + 31) / 32) * 32; // Round up to the nearest multiple of 32
    const size_t img_row_pitch        = get_ion_image_row_pitch(img_format, img_desc);

    cl_int err;
    size_t padding_in_bytes = 0;

    err = clGetDeviceInfo(m_device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for padding." << "\n";
        std::exit(err);
    }

    const size_t y_plane_bytes  = img_row_pitch * effective_img_height;
    const size_t uv_plane_bytes = img_row_pitch * effective_img_height / 2;
    const size_t total_bytes    = y_plane_bytes + uv_plane_bytes + padding_in_bytes;

    return make_iocoherent_ion_buffer(total_bytes);
}

