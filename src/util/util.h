//--------------------------------------------------------------------------------------
// File: util.h
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#ifndef SDK_EXAMPLES_UTIL_H
#define SDK_EXAMPLES_UTIL_H

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <CL/cl.h>

/**
 * \brief yuv_image_t represents the "raw bytes" + width and height of YUV image with two planes.
 *        this encompasses e.g. NV12, TP10, P010.
 */
struct yuv_image_t
{
    uint32_t y_width;
    uint32_t y_height;
    std::vector<unsigned char> y_plane;
    std::vector<unsigned char> uv_plane;
};

struct nv12_image_t : public yuv_image_t {};

struct tp10_image_t : public yuv_image_t {};

struct p010_image_t : public yuv_image_t {};

struct matrix_t
{
    int width, height;
    std::vector<cl_float> elements;
};

struct half_matrix_t
{
    int width, height;
    std::vector<cl_half> elements;
};

/**
 * \brief nonplanar_image_t represents an image type that in contrast to
 *        yuv_image_t does not separate its pixel data into different planes.
 */
struct nonplanar_image_t
{
    uint32_t width;
    uint32_t height;
    std::vector<unsigned char> pixels;
};

/**
 * \brief bayer_mipi10_image_t represents the "raw bytes" + width and height of
 *        a Bayer-ordered MIPI RAW10 data type image. In Bayer order, blue and
 *        red values are interleaved with green values in alternating rows:
 *
 *            BGBGBGBG...
 *            GRGRGRGR...
 *
 *        One "quad" of values here means two green values and one each of red
 *        and blue values that are in the same two columns and span two
 *        consecutive rows. The top left corner of a quad is always a blue
 *        value.
 *
 *        We consider the width of such an image as the total # of blue/green
 *        or green/red values per row, and the height is the number of rows.
 *        However these images are addressed in OpenCL kernels as though each
 *        quad were one pixel, effectively dividing the image dimensions by 2.
 *
 *        MIPI RAW10 is a packed 10-bit ber channel data type -- the 8 most
 *        significant bits of 4 consecutive values per row are followed by 1
 *        byte with the 2 least significant bits for the preceding values, in
 *        order. The MSBs of the fifth byte hold the LSBs for value 1. For
 *        example the top row of a Bayer-ordered image would start with this
 *        sequence of 5 bytes:
 *
 *        | byte 1  | byte 2  | byte 3  | byte 4  | byte 5 |
 *        | b1 MSBs | g1 MSBs | b2 MSBs | g2 MSBs | LSBs   |
 */
struct bayer_mipi10_image_t : public nonplanar_image_t {};

/**
 * \brief Unpacked Bayer image format. Pixels are Bayer-ordered as above, but
 *        each 10-bit channel is held in a 16-bit int with 6 unused bits.
 */
struct bayer_int10_image_t : public nonplanar_image_t {};

/**
 * \brief Represents an RGBA 8888 image.
 */
struct rgba_image_t : public nonplanar_image_t {};

/**
 * \brief Represents a single-channel CL_R image type with an unsigned 16-bit
 *        data type.
 */
struct single_channel_int16_image_t : public nonplanar_image_t {};

/**
 * \brief Represents a single-channel CL_R image type with 32-bit float data
 *        type.
 */
struct single_channel_float_image_t : public nonplanar_image_t {};

/**
 * \brief Loads an 8-bit NV12 image from image data at filename
 *
 * @param filename
 * @return
 */
nv12_image_t load_nv12_image_data(const std::string &filename);

/**
 * \brief Saves 8-bit NV12 image to the given filename
 *
 * @param filename
 * @param image
 */
void save_nv12_image_data(const std::string &filename, const nv12_image_t &image);

/**
 * \brief Loads a TP10 image from image data at filename
 *
 * @param filename
 * @return
 */
tp10_image_t load_tp10_image_data(const std::string &filename);

/**
 * \brief Saves TP10 image to the given filename
 *
 * @param filename
 * @param image
 */
void save_tp10_image_data(const std::string &filename, const tp10_image_t &image);

/**
 * \brief Loads a p010 image from image data at filename
 *
 * @param filename
 * @return
 */
p010_image_t load_p010_image_data(const std::string &filename);

/**
 * \brief Saves p010 image to the given filename
 *
 * @param filename
 * @param image
 */
void save_p010_image_data(const std::string &filename, const p010_image_t &image);

/**
 * \brief Loads a matrix from the given file according to the format
 *        described in README.md
 * @param filename
 */
matrix_t load_matrix(const std::string &filename);

/**
 * \brief Loads a matrix of half-floats from the given file according to the
 *        format described in README.md
 * @param filename
 */
half_matrix_t load_half_matrix(const std::string &filename);

/**
 * \brief Saves a matrix to the given filename.
 * @param filename
 * @param matrix
 */
void save_matrix(const std::string &filename, const matrix_t &matrix);

/**
 * \brief Serializes the matrix to the given output stream
 * @param filename
 * @param matrix
 */
void save_matrix(std::ostream &out, const matrix_t &matrix);

/**
 * \brief Loads a Bayer MIPI10 from image data at filename
 * @param filename
 * @return
 */
bayer_mipi10_image_t load_bayer_mipi_10_image_data(const std::string &filename);

/**
 * \brief Saves a Bayer MIPI10 image to the given filename
 * @param filename
 * @param image
 */
void save_bayer_mipi_10_image_data(const std::string &filename, const bayer_mipi10_image_t &image);

/**
 * \brief Loads a Bayer unpacked 10-bit image from image data at filename
 * @param filename
 * @return
 */
bayer_int10_image_t load_bayer_int_10_image_data(const std::string &filename);


/**
 * \brief Loads an 8-bit depth RGBA image from the given filename.
 * @param filename
 * @param image
 */
rgba_image_t load_rgba_image_data(const std::string &filename);

/**
 * \brief Saves an 8-bit depth RGBA image to the given filename.
 * @param filename
 * @param image
 */
void save_rgba_image_data(const std::string &filename, const rgba_image_t &image);

/**
 * \brief Saves a 16-bit depth single-channel image to the given filename.
 * @param filename
 * @param image
 */
void save_single_channel_image_data(const std::string &filename, const single_channel_int16_image_t &image);

/**
 * \brief Saves a 32-bit float single-channel image to the given filename.
 * @param filename
 * @param image
 */
void save_single_channel_image_data(const std::string &filename, const single_channel_float_image_t &image);

/**
 * \brief Loads a 16-bit depth single-channel image from image data at filename
 * @param filename
 * @return
 */
single_channel_int16_image_t load_single_channel_image_data(const std::string &filename);

/**
 * \brief Returns smallest y such that y % r == 0 and y >= x
 * @param x
 * @param r
 * @return
 */
size_t work_units(size_t x, size_t r);

/**
 * \brief get supported formats with specific mem flag
 * @param context
 * @param mem_flags
 */
std::vector<cl_image_format> get_image_formats(cl_context context, cl_mem_flags mem_flags);

/**
 * \brief print supported image formats
 * @param formats
 */
void print_formats(const std::vector<cl_image_format> &formats);

/**
 * \brief check if specific format in the supported formats list
 * @param formats
 * @param format
 */
bool is_format_supported(const std::vector<cl_image_format> &formats, const cl_image_format &format);

#endif //SDK_EXAMPLES_UTIL_H
