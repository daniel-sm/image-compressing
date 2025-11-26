import numpy as np
import cv2 as cv
from numpy.typing import NDArray

IMAGE_BYTE = NDArray[np.uint8]
BITS = 4

def encode(img: IMAGE_BYTE) -> IMAGE_BYTE:
    mask = (~((1 << (8 - BITS)) - 1)) & 0xFF
    return (img & mask).astype(np.uint8)


def main():
    image = cv.imread("image.bmp")

    if image is None:
        print("Error: Could not read the image.")
        return
    
    image = np.array(image, dtype=np.uint8)

    truncated = encode(image)

    cv.imwrite("truncated.bmp", truncated)


if __name__ == "__main__":
    main()