import cv2 as cv
import numpy as np
from time import perf_counter
from numpy.typing import NDArray

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]
IMG_ENCODED = NDArray[np.int16]

def predictive_coding_encoding(image: IMG_BYTE) -> IMG_ENCODED:
    integer_image = image.astype(np.int16)
    encoded = np.zeros_like(integer_image, dtype=np.int16)
    for i in range(integer_image.shape[0]):
        for j in range(1, integer_image.shape[1]):
            if j == 0:
                encoded[i, j] = integer_image[i, j]
            else:
                encoded[i, j] = integer_image[i, j] - integer_image[i, j - 1]
    return encoded

def predictive_coding_decoding(encoded: IMG_ENCODED) -> IMG_BYTE:
    decoded = np.zeros_like(encoded, dtype=np.uint8)
    for i in range(encoded.shape[0]):
        for j in range(encoded.shape[1]):
            if j == 0:
                decoded[i, j] = encoded[i, j]
            else:
                decoded[i, j] = encoded[i, j] + decoded[i, j - 1]
    return decoded

def main() -> None:
    image = cv.imread("image.bmp", cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    image = np.array(image, dtype=np.uint8)
    # array = np.random.randint(0, 256, (1280, 720), dtype=np.uint8)
    # print(array)

    begin = perf_counter()
    encoded = predictive_coding_encoding(image)
    end = perf_counter()
    print(f"Encoding time: {end - begin: .6f} seconds")
    # print(encoded)

    begin = perf_counter()
    decoded = predictive_coding_decoding(encoded)
    end = perf_counter()
    print(f"Decoding time: {end - begin: .6f} seconds")
    # print(decoded)

if __name__ == "__main__":
    main()
