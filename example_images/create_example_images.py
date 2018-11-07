import struct
import random as rand

########
# NV12 #
########

prefix = (
        "\x18\x00\x00\x00" +
        "\x18\x00\x00\x00" +
        "\xD2\x10\x00\x00" +
        "\x33\x41\x00\x00"
)

const_data = prefix
incr_data = prefix
for i in range(0, 0x18 * 0x18 * 3 / 2):
    const_data += struct.pack("B", 128)
    if i < 0x18 * 0x18:
        incr_data += struct.pack("B", i % 256)

for i in range(0, 0x18 * 0x18 / 4):
    incr_data += struct.pack("BB", i % 256, i % 256)


rand_prefix = (
        "\xA0\x00\x00\x00" +
        "\x78\x00\x00\x00" +
        "\xD2\x10\x00\x00" +
        "\x33\x41\x00\x00"
)

rand.seed(42)

rand_data = rand_prefix
for i in range(0xA0 * 0x78 * 3 / 2):
    rand_data += struct.pack("B", rand.randint(0, 255))

with open("CL_QCOM_NV12__CL_UNORM_INT8__RANDOM.dat", "wb") as f:
    f.write(rand_data)

with open("CL_QCOM_NV12__CL_UNORM_INT8__CONSTANT.dat", "wb") as f:
    f.write(const_data)

with open("CL_QCOM_NV12__CL_UNORM_INT8__INCREASING.dat", "wb") as f:
    f.write(incr_data)

# Make some circles of various sizes
for i in range(6, 9):
    height = width = 2 ** i
    data = ""
    data += struct.pack("<I", width)
    data += struct.pack("<I", height)
    data += "\xD2\x10\x00\x00" + \
            "\x33\x41\x00\x00"
    center         = (width / 2, height / 2)
    radius_squared = (width / 4) ** 2
    for w in range(width):
        for h in range(height):
            dist = ((center[0] - w) ** 2) + ((center[1] - h) ** 2)
            value = 255 if dist <= radius_squared else 0
            data += struct.pack("B", value)
    data += "\x00" * (width * height / 2)
    filename = "CL_QCOM_NV12__CL_UNORM_INT8__{}x{}_CIRCLE.dat".format(width, height)
    with open(filename, "wb") as f:
        f.write(data)

########
# TP10 #
########

prefix = (
        "\x18\x00\x00\x00" +
        "\x18\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x45\x41\x00\x00"
)

const_data = prefix
incr_data = prefix
tp10_mask = 0x3FFFFFFF # Zeroes out unused bits
for i in range(0, 0x18 * 0x18 / 3):
    const_data += struct.pack("<I", 0xFFFFFFFF & tp10_mask)
    incr_data += struct.pack("<I", (i % (2 ** 32)) & tp10_mask)

for i in range(0, 0x18 * 0x18 / 6):
    const_data += struct.pack("<I", 0xEEEEEEEE & tp10_mask)
    incr_data += struct.pack("<I", (i % (2 ** 32)) & tp10_mask)

rand_prefix = (
        "\xA2\x00\x00\x00" +
        "\x78\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x45\x41\x00\x00"
)

rand_data = rand_prefix
for i in range(0xA2 * 0x78 / 2):
    rand_data += struct.pack("<I", rand.randint(0, 2 ** 32) & tp10_mask)

with open("CL_QCOM_TP10__CL_QCOM_UNORM_INT10__RANDOM.dat", "wb") as f:
    f.write(rand_data)

with open("CL_QCOM_TP10__CL_QCOM_UNORM_INT10__CONSTANT.dat", "wb") as f:
    f.write(const_data)

with open("CL_QCOM_TP10__CL_QCOM_UNORM_INT10__INCREASING.dat", "wb") as f:
    f.write(incr_data)

########
# P010 #
########

prefix = (
        "\x18\x00\x00\x00" +
        "\x18\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x3C\x41\x00\x00"
)

const_data = prefix
incr_data = prefix
p010_mask = 0xFFC0 # Zeroes out unused bits
for i in range(0, 0x18 * 0x18):
    const_data += struct.pack("<H", 0xFFFF & p010_mask)
    incr_data += struct.pack("<H", (i % (2 ** 16)) & p010_mask)

for i in range(0, 0x18 * 0x18 / 4):
    const_data += struct.pack("<HH", 0xEEEE & p010_mask, 0xDDDD & p010_mask)
    incr_data += struct.pack("<HH", (i % (2 ** 16)) & p010_mask, (i % (2 ** 16)) & p010_mask)

rand_prefix = (
        "\xA2\x00\x00\x00" +
        "\x78\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x3C\x41\x00\x00"
)

rand_data = rand_prefix
for i in range(0xA2 * 0x78 * 3 / 2):
    rand_data += struct.pack("<H", rand.randint(0, 2 ** 16) & p010_mask)

with open("CL_QCOM_P010__CL_QCOM_UNORM_INT10__RANDOM.dat", "wb") as f:
    f.write(rand_data)

with open("CL_QCOM_P010__CL_QCOM_UNORM_INT10__CONSTANT.dat", "wb") as f:
    f.write(const_data)

with open("CL_QCOM_P010__CL_QCOM_UNORM_INT10__INCREASING.dat", "wb") as f:
    f.write(incr_data)

neutral_face = \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

##################
# Bayer + MIPI10 #
##################

prefix = (
        "\x30\x00\x00\x00" +
        "\x30\x00\x00\x00" +
        "\x59\x41\x00\x00" +
        "\x4E\x41\x00\x00"
)

bayer_data = prefix
for j in range(24):
    blue_row, red_row = "", ""
    for i in range(24 / 2):
        pixel_1, pixel_2 = neutral_face[2 * i + j * 24], neutral_face[2 * i + 1 + j * 24]
        green_val_1, green_val_2 = pixel_1 * 255, pixel_2 * 255
        blue_val = int(float(2 * i + j * 24) / (24 * 24) * 255)
        red_val = 255 - blue_val
        blue_row += struct.pack("BBBBB", blue_val, green_val_1, blue_val, green_val_2, 0)
        red_row += struct.pack("BBBBB", green_val_1, red_val, green_val_2, red_val, 0)
    bayer_data += blue_row
    bayer_data += red_row

with open("CL_QCOM_BAYER__CL_QCOM_UNORM_MIPI10__48x48_FACE.dat", "wb") as f:
    f.write(bayer_data)

prefix = (
        "\xC0\x00\x00\x00" +
        "\xC0\x00\x00\x00" +
        "\x59\x41\x00\x00" +
        "\x4E\x41\x00\x00"
)

bayer_data = prefix
for j in range(24):
    blue_row, red_row = "", ""
    for i in range(24):
        pixel_1, pixel_2 = neutral_face[i + j * 24], neutral_face[i + j * 24]
        green_val_1, green_val_2 = pixel_1 * 255, pixel_2 * 255
        blue_val = int(float(i + j * 24) / (24 * 24) * 255)
        red_val = 255 - blue_val
        blue_row += struct.pack("BBBBB", blue_val, green_val_1, blue_val, green_val_2, 0) * 2
        red_row += struct.pack("BBBBB", green_val_1, red_val, green_val_2, red_val, 0) * 2
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row

with open("CL_QCOM_BAYER__CL_QCOM_UNORM_MIPI10__192x192_FACE.dat", "wb") as f:
    f.write(bayer_data)

###########################
# Bayer + Unpacked 10-bit #
###########################

prefix = (
        "\x30\x00\x00\x00" +
        "\x30\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x4E\x41\x00\x00"
)

bayer_data = prefix
for j in range(24):
    blue_row, red_row = "", ""
    for i in range(24 / 2):
        pixel_1, pixel_2 = neutral_face[2 * i + j * 24], neutral_face[2 * i + 1 + j * 24]
        green_val_1, green_val_2 = pixel_1 * 255, pixel_2 * 255
        blue_val = int(float(2 * i + j * 24) / (24 * 24) * 255)
        red_val = 255 - blue_val
        blue_val    <<= 8
        green_val_1 <<= 8
        green_val_2 <<= 8
        red_val     <<= 8
        blue_row += struct.pack("<HHHH", blue_val, green_val_1, blue_val, green_val_2)
        red_row += struct.pack("<HHHH", green_val_1, red_val, green_val_2, red_val)
    bayer_data += blue_row
    bayer_data += red_row

with open("CL_QCOM_BAYER__CL_QCOM_UNORM_INT10__48x48_FACE.dat", "wb") as f:
    f.write(bayer_data)

prefix = (
        "\xC0\x00\x00\x00" +
        "\xC0\x00\x00\x00" +
        "\x5D\x41\x00\x00" +
        "\x4E\x41\x00\x00"
)

bayer_data = prefix
for j in range(24):
    blue_row, red_row = "", ""
    for i in range(24):
        pixel_1, pixel_2 = neutral_face[i + j * 24], neutral_face[i + j * 24]
        green_val_1, green_val_2 = pixel_1 * 255, pixel_2 * 255
        blue_val = int(float(i + j * 24) / (24 * 24) * 255)
        red_val = 255 - blue_val
        blue_val    <<= 8
        green_val_1 <<= 8
        green_val_2 <<= 8
        red_val     <<= 8
        blue_row += struct.pack("<HHHH", blue_val, green_val_1, blue_val, green_val_2) * 2
        red_row += struct.pack("<HHHH", green_val_1, red_val, green_val_2, red_val) * 2
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row
    bayer_data += blue_row
    bayer_data += red_row

with open("CL_QCOM_BAYER__CL_QCOM_UNORM_INT10__192x192_FACE.dat", "wb") as f:
    f.write(bayer_data)
