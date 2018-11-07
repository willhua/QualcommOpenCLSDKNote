# SDK Examples

## What is this?

Usage examples for Qualcomm's extensions to OpenCL.

## Building for Android

There's a few things you'll need:

* The Android Open Source Project (AOSP) tree set up to build for your target
  device.
* Appropriate kernel headers (`linux/ion.h` and `linux/msm_ion.h`)
* A `libOpenCL` module defined by an `Android.mk` file.

More on those below. Once everything is set up just run `mma` in this directory
to build all the examples.

### Where do I get kernel headers?

If your target device's kernel has the appropriate headers, they still need to
be in a location where the Android build system can discover them. One way to
ensure this is to build a bootimage, which will export the appropriate files:

```
> cd $ANDROID_BUILD_TOP
> make bootimage
```

### Where do I get the libOpenCL module?

At the time of this writing `libOpenCL` is not available as part of Google's
prebuilt graphics libraries releases for Qualcomm devices. If you are lucky
enough to have it anyway, then you shouldn't need to do anything. Running `mma`
in this directory will build all dependencies, including `libOpenCL`.

### I don't have the libOpenCL module, can I still use these examples?

Maybe, if you have the `libOpenCL.so` binary for your device to link against,
but it's not for the faint of heart. Provided here is a `CMakeLists.txt` file
and a script `build_android.sh` that can be used as a starting point, but
there's no guarantee it will work for your target device. You'll still need the
AOSP tree for the kernel headers, so go get it if you don't have it.

Find taka-no-me's `android-cmake` project online and clone it into the
`android-cmake` directory here.

All of these examples use ION buffers, so you'll still need appropriate ION
headers. Find where your target device's `msm_ion.h` and `ion.h` headers are.
For example you might see them at
`$ANDROID_BUILD_TOP/hardware/qcom/<target-device>/kernel-headers/linux` where
`<target-device>` should be replaced by your target device. You'll include
this directory in the header search path.

You'll also need the Android NDK, Revision 11c. The specific version is
important.

Then run the build script, substituting the paths specific to your build
environment:

```
ANDROID_NDK=/path/to/android-ndk-r11c \
OPEN_CL_LIB=/path/to/libOpenCL.so \
ION_INCLUDE_PATH=$ANDROID_BUILD_TOP/hardware/qcom/<target-device>/kernel-headers/linux \
./build_android.sh <BITNESS>
```

`<BITNESS>` should be `32` or `64` depending on your target architecture.

## Usage

Building will produce a set of binaries. Run each one without arguments to see
a help message and description of what it does. Most binaries take an input
image in the format described above -- several sample images are given in the
example_images directory, which contains arbitrary data (e.g. it is not
visually interesting).

## Descriptions

### src/examples/basic directory

#### hello_world.cpp

A very basic example to test out building. It simply copies one file to another.

#### qcom_block_match_sad.cpp, qcom_block_match_ssd.cpp, qcom_box_filter_image.cpp, qcom_convolve_image.cpp

These examples all demonstrate basic usage for the named built-in extension functions.
Look here for minimal examples of how to use the extensions.

#### compressed_image_nv12.cpp, compressed_image_rgba.cpp

Demonstrates use of compressed images using Qualcomm extensions to OpenCL.
The input image is compressed and then decompressed, with the result written
to the specified output file for comparison. (The compression is not lossy so
they are identical.)

Compressed image formats may be saved to disk, however be advised that the format
is specific to each GPU.

The two examples show compression for NV12 and RGBA images.

### src/examples/bayer_mipi

The examples in this directory show how to use Bayer-ordered images and packed
MIPI data formats.

Bayer-ordered images have one red, green or blue value per pixel, and the pixels
are interleaved in a mosaic pattern. In order to get an equivalent RGB image
one must "demosaic" the image by interpolating the missing red, green, and blue
values. Bayer-ordered images are addressed by 2x2 blocks of such pixels, where
each block has one red and blue value, and two green values. A Bayer-ordered
image may also be addressed as a single-channel (`CL_R`) image to get one color
channel at a time.

`bayer_mipi10_to_rgba.cpp` and `unpacked_bayer_to_rgba.cpp` both demonstrate one
scheme for demosaicing. The former uses the packed MIPI10 format, and the latter
uses an unpacked 10-bit format (held in a 16-bit int with 6 bits unused). Both
use Bayer-ordered images to exploit the GPU's interpolation capabilities without
mixing different color channels. The destination format has 8-bits per channel,
so some precision is lost.

`mipi10_to_unpacked.cpp` and `unpacked_to_mipi10.cpp` demonstrate using the
MIPI10 data format with a single-channel `CL_R` order. The former converts a
packed MIPI10 image into an unpacked 10-bit image. The latter shows the
unpacked-to-packed conversion.

### src/examples/conversions

The examples in this directory show conversions to and from various image formats.

### src/examples/convolutions

#### convolution.cpp

Demonstrates efficient convolution without the use of built-in extension functions.

#### accelerated_convolution.cpp

Demonstrates efficient convolution with the qcom_convolve_imagef built-in extension
function.

### src/examples/fft

These examples compute the 2-dimensional fast Fourier transform (2D FFT) of an
image or matrix using the in-place Cooley-Tukey algorithm. First in the
"row pass" each work group calculates the 1D FFT of a row, by reading initial
data from global memory into local memory, and calculating intermediate results
in-place using local memory. The final result is written to global memory in
transposed order. This procedure is then repeated in a "column pass" that acts
on the rows of the result of the first pass. Calculating the 1D FFTs
back-to-back in this way is equal to the 2D FFT.

For the image-based version, the input is an 8-bit per channel NV12 image, and
the outputs are two single-channel images with a 32-bit float data type. The
outputs contain the real and imaginary parts of the FFT. The example acts on
the Y-plane only.

The buffer-based version takes a real-valued matrix as input (specified as
below), and produces two matrices as the output holding the real and imaginary
parts of the FFT.

### src/examples/io_coherent_ion

These simple examples demonstrate using the IO-coherent host cache policy for
ION buffers. Both examples simply copy a specified file or image. Except for
the parameters used to create the ION buffers, there is no difference in the
host or kernel code compared to using uncached ION buffers.

### src/examples/linear_algebra

Demonstrates some basic linear algebra operations:

* Matrix addition
* Matrix multiplication
* Matrix transposition

The transposition and multiplication examples come in two flavors, one using
OpenCL buffers and another that packs the matrices into 2D images. It is not a
foregone conclusion that using an image or a buffer will enjoy better
performance in any given use case, so generally one must try and see what works
best.

The image versions of both examples pad irregularly sized matrices, both because
images have per-row alignment requirements and because this permits an efficient
tiled algorithm to be applied uniformly. This approach can use substantially
more memory than the buffer-based version.

In contrast, the buffer versions do not pad the input matrices. They use an
efficient tiled algorithm where possible, and a less efficient algorithm to
calculate the remaining portion of the output not covered by the tiled
algorithm.

The multiplication examples additionally have a "half" variant, that
demonstrates using the 16-bit half-float data type. The input, output and
arithmetic all use half-floats. This can be a significant performance advantage,
although it introduces more error. One may mix use of floats and half-floats to
achieve the desired performance/accuracy trade off.

### src/examples/vector_image_ops

All examples in this directory demonstrate a variety of kernels using vector
read and write operations for the given image formats.

## Image data format

Input and output images have the following format, where multi-byte data types are written with the least significant
byte first:

* 4 bytes: plane width in pixels (unsigned integer)
* 4 bytes: plane height (unsigned integer)
* 4 bytes: OpenCL channel data type.
* 4 bytes: OpenCL channel order.
* N bytes: pixel data, where N is dependent on the preceding four values.

## Matrix data format

Matrices used by the examples in the `linear_algebra` directory have the
following plain text format:

* Two integers separated by whitespace indicating the number of columns and rows
  of the matrix.
* A sequence of whitespace-separated floating point element values in row-major
  order.

For example, the following represents a 3x2 matrix:

```
2 3
1.0 2.0
3.1 4.1
6   0
```
