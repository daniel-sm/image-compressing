import cv2 as cv
import numpy as np

import truncate, huffman, color_table


def main():
    image = cv.imread("image.bmp")

    if image is None:
        print("Error: Could not read the image.")
        return
    
    image = np.array(image, dtype=np.uint8)
    print("Original:", image.dtype, image.shape, image.size)

    truncated = truncate.encoding(image)
    print("Truncated:", truncated.dtype, truncated.shape, truncated.size)

    color_mapped = color_table.encoding(truncated)
    print("Color Mapped:", color_mapped.dtype, color_mapped.shape, color_mapped.size)

    encoded = huffman.encoding(color_mapped)
    print("Encoded:", encoded.dtype, encoded.shape, encoded.size)

    decoded = huffman.decoding(encoded)
    print("Decoded:", decoded.dtype, decoded.shape, decoded.size)

    cv.imwrite("trun_huff.bmp", decoded)


if __name__ == "__main__":
    main()
