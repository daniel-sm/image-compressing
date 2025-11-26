import cv2 as cv
import numpy as np

import truncate
import huffman

def main():
    image = cv.imread("image.bmp")

    if image is None:
        print("Error: Could not read the image.")
        return
    
    image = np.array(image, dtype=np.uint8)
    print("Original:", image.dtype, image.shape, image.size)

    truncated = truncate.encode(image)
    print("Truncated:", truncated.dtype, truncated.shape, truncated.size)

    encoded = huffman.encode_huffman(truncated)
    print("Encoded:", encoded.dtype, encoded.shape, encoded.size)

    decoded = huffman.decode_huffman(encoded)
    print("Decoded:", decoded.dtype, decoded.shape, decoded.size)

    cv.imwrite("trun_huff.bmp", decoded)


if __name__ == "__main__":
    main()
