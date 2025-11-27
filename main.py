import sys
import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import truncate, huffman, color_table, files


IMAGE_BYTE = NDArray[np.uint8]


def open_image(filename: str) -> IMAGE_BYTE:
    image = cv.imread(filename)

    if image is None:
        raise ValueError("Could not open image file.")

    return np.array(image, dtype=np.uint8)


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py [compress|decompress] image.ext file.ext")
        sys.exit(1)

    command = sys.argv[1]
    imagename = sys.argv[2]
    filename = sys.argv[3]

    if not files.valid_filename(filename) or not files.valid_filename(imagename):
        print("Invalid filename.")
        sys.exit(1)    

    if command == "compress":
        image = open_image(imagename)
        print("Image size: ", image.size)
        data = truncate.truncate(image, bits=5)
        data = huffman.encoding(data)
        print("Compressed size:", data.size)

        files.write_bytes(filename, data)
        print(f"Image compressed and saved to `{filename}`.")

    elif command == "decompress":
        data = files.read_bytes(filename)
        print("Compressed size:", data.size)
        image = huffman.decoding(data)
        print("Image size:", image.size)

        cv.imwrite(imagename, image)
        print(f"Image decompressed and saved to `{imagename}`.")

    else:
        print("Invalid command. Use `compress` or `decompress`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
