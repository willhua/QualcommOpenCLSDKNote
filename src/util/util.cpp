//--------------------------------------------------------------------------------------
// File: util.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include "util/util.h"
#include "half_float.h"

#include "CL/cl.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <CL/cl_ext_qcom.h>

/************************
 * Forward declarations *
 ************************/

/**
 * \brief Writes a given value to the output stream in little-endian byte order
 *
 * @param UIntType - Template param. For correctness, must be an unsigned integral type.
 * @param out - The stream to write to.
 * @param val - The value to write.
 */
template <typename UIntType>
static void write_le(std::ostream &out, UIntType val);

/**
 * \brief Reads a value from a little-endian byte order input stream.
 *
 * @param UIntType - For correctness, must be an unsigned integral type
 * @param in - The stream to read from.
 * @return The value read.
 */
template <typename UIntType>
static UIntType read_le(std::istream &in);

template <typename UIntType>
static void write_le(std::ostream &out, UIntType val)
{
    unsigned char byte = 0;
    for (uint32_t i = 0; i < sizeof(UIntType); ++i)
    {
        byte = (val & (0xFF << (i * 8))) >> (i * 8);
        out.put(*reinterpret_cast<char *>(&byte));
    }
}

template <typename UIntType>
static UIntType read_le(std::istream &in)
{
    UIntType val = 0;
    unsigned char byte = 0;
    for (uint32_t i = 0; i < sizeof(UIntType); ++i)
    {
        in.get(*reinterpret_cast<char *>(&byte));
        val |= byte << (i * 8);
    }
    return val;
}

static void read_and_check_header(std::istream &in, uint32_t &width, uint32_t &height, uint32_t desired_data_type,
                                  uint32_t desired_order, const std::string &data_type_name,
                                  const std::string &order_name)
{
    width              = read_le<uint32_t>(in);
    height             = read_le<uint32_t>(in);
    uint32_t data_type = read_le<uint32_t>(in);
    uint32_t order     = read_le<uint32_t>(in);
    if (order != desired_order || data_type != desired_data_type)
    {
        std::cerr << "Expected " << order_name << " and " << data_type_name << " channel order and data types\n";
        std::exit(EXIT_FAILURE);
    }
}

// Macros are great for stringification
#define GET_HEADER(strm, w, h, desired_dt, desired_ord) \
    read_and_check_header(strm, w, h, desired_dt, desired_ord, #desired_dt, #desired_ord)

/**
 * Internal method for reading a plane of image data from a stream.
 * If no re-ordering of bytes is desired, e.g. for a packed format, just set
 * "channel_bytes" to 1.
 *
 * @param in - A stream to read from. Must have data in little-endian byte order.
 * @param channel_bytes - The number of consecutive bytes to read for each color channel
 * @param num_channels - The number of color channels per pixel
 * @param plane - An appropriately sized buffer to hold the output
 */
static void read_plane(std::istream &in, uint32_t channel_bytes, uint32_t num_channels, std::vector<unsigned char> &plane)
{
    for (size_t i = 0; i < plane.size() / (channel_bytes * num_channels); ++i)
    {
        for (size_t j = 0; j < num_channels; ++j)
        {
            switch (channel_bytes)
            {
                case 1:
                {
                    const uint8_t val = read_le<uint8_t>(in);
                    std::memcpy(plane.data() + i * num_channels * channel_bytes + j * channel_bytes, &val, sizeof(val));
                    break;
                }
                case 2:
                {
                    const uint16_t val = read_le<uint16_t>(in);
                    std::memcpy(plane.data() + i * num_channels * channel_bytes + j * channel_bytes, &val, sizeof(val));
                    break;
                }
                case 4:
                {
                    const uint32_t val = read_le<uint32_t>(in);
                    std::memcpy(plane.data() + i * num_channels * channel_bytes + j * channel_bytes, &val, sizeof(val));
                    break;
                }
                default:
                {
                    std::cerr << "Error, can't read " << channel_bytes << " bytes at a time.\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
}

/**
 * Internal method for writing a plane of image data to a stream.
 * If no re-ordering of bytes is desired, e.g. for a packed format, just set
 * "channel_bytes" to 1.
 *
 * @param out - A stream to write to.
 * @param channel_bytes - The number of consecutive bytes to read for each color channel
 * @param num_channels - The number of color channels per pixel
 * @param plane - An appropriately sized buffer to read the data from.
 */
static void write_plane(std::ostream &out, uint32_t channel_bytes, uint32_t num_channels, const std::vector<unsigned char> &plane)
{
    for (size_t i = 0; i < plane.size() / (channel_bytes * num_channels); ++i)
    {
        for (size_t j = 0; j < num_channels; ++j)
        {
            switch (channel_bytes)
            {
                case 1:
                {
                    write_le<uint8_t>(out, plane[i * num_channels * channel_bytes + j * channel_bytes]);
                    break;
                }
                case 2:
                {
                    const uint16_t val = *reinterpret_cast<const uint16_t *>(plane.data() + i * num_channels * channel_bytes + j * channel_bytes);
                    write_le<uint16_t>(out, val);
                    break;
                }
                case 4:
                {
                    const uint32_t val = *reinterpret_cast<const uint32_t *>(plane.data() + i * num_channels * channel_bytes + j * channel_bytes);
                    write_le<uint32_t>(out, val);
                    break;
                }
                default:
                {
                    std::cerr << "Error, can't write " << channel_bytes << " bytes at a time.\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
}

static void save_yuv_file_internal(const std::string &filename, const yuv_image_t &image, uint32_t data_type, uint32_t order,
                                   uint32_t channel_bytes)
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
    {
        std::cerr << "Can't open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    write_le<uint32_t>(fout, image.y_width);
    write_le<uint32_t>(fout, image.y_height);
    write_le<uint32_t>(fout, data_type);
    write_le<uint32_t>(fout, order);

    write_plane(fout, channel_bytes, 1, image.y_plane);
    write_plane(fout, channel_bytes, 2, image.uv_plane);
}

static void
save_nonplanar_internal(const std::string &filename, const nonplanar_image_t &image, uint32_t data_type, uint32_t order,
                        uint32_t channel_bytes, uint32_t num_channels)
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
    {
        std::cerr << "Can't open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    write_le<uint32_t>(fout, image.width);
    write_le<uint32_t>(fout, image.height);
    write_le<uint32_t>(fout, data_type);
    write_le<uint32_t>(fout, order);

    write_plane(fout, channel_bytes, num_channels, image.pixels);
}

nv12_image_t load_nv12_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    nv12_image_t result;
    GET_HEADER(fin, result.y_width, result.y_height, CL_UNORM_INT8, CL_QCOM_NV12);

    const size_t y_plane_len = result.y_width * result.y_height;
    result.y_plane.resize(y_plane_len);
    read_plane(fin, 1, 1, result.y_plane);

    result.uv_plane.resize(y_plane_len / 2);
    read_plane(fin, 1, 2, result.uv_plane);

    return result;
}

tp10_image_t load_tp10_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    tp10_image_t result;
    GET_HEADER(fin, result.y_width, result.y_height, CL_QCOM_UNORM_INT10, CL_QCOM_TP10);

    const size_t y_plane_len = result.y_width * result.y_height / 3 * 4;
    result.y_plane.resize(y_plane_len);
    read_plane(fin, 4, 1, result.y_plane);

    result.uv_plane.resize(y_plane_len / 2);
    read_plane(fin, 4, 1, result.uv_plane);

    return result;
}

p010_image_t load_p010_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    p010_image_t result;
    GET_HEADER(fin, result.y_width, result.y_height, CL_QCOM_UNORM_INT10, CL_QCOM_P010);

    const size_t y_plane_len = result.y_width * result.y_height * 2;
    result.y_plane.resize(y_plane_len);
    read_plane(fin, 2, 1, result.y_plane);

    result.uv_plane.resize(y_plane_len / 2);
    read_plane(fin, 2, 2, result.uv_plane);

    return result;
}

void save_nv12_image_data(const std::string &filename, const nv12_image_t &image)
{
    save_yuv_file_internal(filename, image, CL_UNORM_INT8, CL_QCOM_NV12, 1);
}

void save_tp10_image_data(const std::string &filename, const tp10_image_t &image)
{
    save_yuv_file_internal(filename, image, CL_QCOM_UNORM_INT10, CL_QCOM_TP10, 4);
}

void save_p010_image_data(const std::string &filename, const p010_image_t &image)
{
    save_yuv_file_internal(filename, image, CL_QCOM_UNORM_INT10, CL_QCOM_P010, 2);
}

size_t work_units(size_t x, size_t r)
{
    return (x + r - 1) / r;
}

matrix_t load_matrix(const std::string &filename)
{
    std::ifstream fin(filename);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }
    matrix_t res;
    fin >> res.width >> res.height;
    res.elements.reserve(res.width * res.height);
    for (int i = 0; i < res.width * res.height; ++i)
    {
        cl_float num;
        fin >> num;
        res.elements.push_back(num);
    }
    return res;
}

void save_matrix(const std::string &filename, const matrix_t &matrix)
{
    std::ofstream fout(filename);
    if (!fout)
    {
        std::cerr << "Can't open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }
    save_matrix(fout, matrix);
}

void save_matrix(std::ostream &out, const matrix_t &matrix)
{
    out << matrix.width << " " << matrix.height << "\n";
    for (auto i = 0; i < matrix.height; ++i)
    {
        for (auto j = 0; j < matrix.width; ++j)
        {
            const int idx = i * matrix.width + j;
            out << matrix.elements[idx] << " ";
        }
        out << "\n";
    }
}

bayer_mipi10_image_t load_bayer_mipi_10_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    bayer_mipi10_image_t result;
    GET_HEADER(fin, result.width, result.height, CL_QCOM_UNORM_MIPI10, CL_QCOM_BAYER);

    const size_t data_length = (result.width / 4 * 5) * (result.height);
    result.pixels.resize(data_length);
    read_plane(fin, 1, 1, result.pixels);

    return result;
}

void save_rgba_image_data(const std::string &filename, const rgba_image_t &image)
{
    save_nonplanar_internal(filename, image, CL_UNORM_INT8, CL_RGBA, 1, 4);
}

bayer_int10_image_t load_bayer_int_10_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    bayer_int10_image_t result;
    GET_HEADER(fin, result.width, result.height, CL_QCOM_UNORM_INT10, CL_QCOM_BAYER);

    const size_t data_length = (result.width * 2) * (result.height);
    result.pixels.resize(data_length);
    read_plane(fin, 2, 1, result.pixels);

    return result;
}

void print_formats(const std::vector<cl_image_format> &formats)
{
    for (const auto &format : formats)
    {
        std::string order, data_type;

        switch (format.image_channel_order)
        {
            case CL_QCOM_COMPRESSED_NV12:
                order = "CL_QCOM_COMPRESSED_NV12";
                break;
            case CL_QCOM_COMPRESSED_NV12_Y:
                order = "CL_QCOM_COMPRESSED_NV12_Y";
                break;
            case CL_QCOM_COMPRESSED_NV12_UV:
                order = "CL_QCOM_COMPRESSED_NV12_UV";
                break;
            case CL_QCOM_COMPRESSED_NV12_4R:
                order = "CL_QCOM_COMPRESSED_NV12_4R";
                break;
            case CL_QCOM_COMPRESSED_NV12_4R_Y:
                order = "CL_QCOM_COMPRESSED_NV12_4R_Y";
                break;
            case CL_QCOM_COMPRESSED_NV12_4R_UV:
                order = "CL_QCOM_COMPRESSED_NV12_4R_UV";
                break;
            case CL_QCOM_COMPRESSED_P010:
                order = "CL_QCOM_COMPRESSED_P010";
                break;
            case CL_QCOM_COMPRESSED_P010_Y:
                order = "CL_QCOM_COMPRESSED_P010_Y";
                break;
            case CL_QCOM_COMPRESSED_P010_UV:
                order = "CL_QCOM_COMPRESSED_P010_UV";
                break;
            case CL_QCOM_COMPRESSED_TP10:
                order = "CL_QCOM_COMPRESSED_TP10";
                break;
            case CL_QCOM_COMPRESSED_TP10_Y:
                order = "CL_QCOM_COMPRESSED_TP10_Y";
                break;
            case CL_QCOM_COMPRESSED_TP10_UV:
                order = "CL_QCOM_COMPRESSED_TP10_UV";
                break;
            default:
            {
                std::stringstream strm;
                strm << "Unknown order: 0x" << std::hex << format.image_channel_order;
                order = strm.str();
                break;
            }
        }

        switch (format.image_channel_data_type)
        {
            case CL_UNORM_INT8:
                data_type = "CL_UNORM_INT8";
                break;
            case CL_QCOM_UNORM_INT10:
                data_type = "CL_QCOM_UNORM_INT10";
                break;
            default:
            {
                std::stringstream strm;
                strm << "Unknown data type: 0x" << std::hex << format.image_channel_data_type;
                data_type = strm.str();
                break;
            }
        }

        std::cerr << "\t" << order << "\n";
        std::cerr << "\t" << data_type << "\n";
        std::cerr << "\t--------------------\n";
    }
    std::cerr << "\n";
}

std::vector<cl_image_format> get_image_formats(cl_context context, cl_mem_flags mem_flags)
{
    cl_int  err               = 0;
    cl_uint num_image_formats = 0;

    err = clGetSupportedImageFormats(
            context,
            mem_flags,
            CL_MEM_OBJECT_IMAGE2D,
            0,
            NULL,
            &num_image_formats
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetSupportedImageFormats." << "\n";
        std::exit(err);
    }

    std::vector<cl_image_format> formats(num_image_formats);

    err = clGetSupportedImageFormats(
            context,
            mem_flags,
            CL_MEM_OBJECT_IMAGE2D,
            formats.size(),
            formats.data(),
            NULL
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetSupportedImageFormats." << "\n";
        std::exit(err);
    }

    return formats;
}

bool is_format_supported(const std::vector<cl_image_format> &formats, const cl_image_format &format)
{
    const auto compare = [&format](const cl_image_format &element)
    {
        return element.image_channel_data_type == format.image_channel_data_type
               && element.image_channel_order == format.image_channel_order;
    };
    return std::find_if(formats.begin(), formats.end(), compare) != formats.end();
}

half_matrix_t load_half_matrix(const std::string &filename) {
    std::ifstream fin(filename);

    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    half_matrix_t res;
    fin >> res.width >> res.height;
    res.elements.reserve(res.width * res.height);
    for (int i = 0; i < res.width * res.height; ++i)
    {
        cl_float num;
        fin >> num;
        res.elements.push_back(to_half(num));
    }
    return res;
}

void save_single_channel_image_data(const std::string &filename, const single_channel_int16_image_t &image)
{
    save_nonplanar_internal(filename, image, CL_UNORM_INT16, CL_R, 2, 1);
}

void save_single_channel_image_data(const std::string &filename, const single_channel_float_image_t &image)
{
    save_nonplanar_internal(filename, image, CL_FLOAT, CL_R, 4, 1);
}

single_channel_int16_image_t load_single_channel_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    single_channel_int16_image_t result;
    GET_HEADER(fin, result.width, result.height, CL_UNORM_INT16, CL_R);

    const size_t data_length = (result.width * 2) * (result.height);
    result.pixels.resize(data_length);
    read_plane(fin, 2, 1, result.pixels);

    return result;
}

void save_bayer_mipi_10_image_data(const std::string &filename, const bayer_mipi10_image_t &image)
{
    save_nonplanar_internal(filename, image, CL_QCOM_UNORM_MIPI10, CL_QCOM_BAYER, 1, 1);
}

rgba_image_t load_rgba_image_data(const std::string &filename)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Can't open " << filename << " for reading\n";
        std::exit(EXIT_FAILURE);
    }

    rgba_image_t result;
    GET_HEADER(fin, result.width, result.height, CL_UNORM_INT8, CL_RGBA);

    const size_t data_length = result.width * result.height * 4;
    result.pixels.resize(data_length);
    read_plane(fin, 1, 4, result.pixels);

    return result;
}
