import cv2 as cv
import numpy as np
from time import perf_counter
from numpy.typing import NDArray

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]

def run_length_encoding(image: IMG_BYTE) -> IMG_BYTE:
    encoded = []
    # adicionando a largura da imagem como header
    width = np.uint16(image.shape[1])
    byte1 = width >> 8
    byte2 = width & 0b1111_1111
    encoded.extend([byte1, byte2])
    # computando algoritmo de run length
    size = image.size
    flattened = image.ravel()
    i = 0
    while i < size:
        count = 1
        while i + 1 < size and flattened[i] == flattened[i + 1] and count < 255:
            i += 1
            count += 1
        encoded.extend([count, flattened[i]])
        i += 1
    # convertendo para array numpy
    encoded = np.array(encoded, dtype=np.uint8)
    return encoded

def run_length_decoding(encoded: IMG_BYTE) -> IMG_BYTE:
    # recuperando a largura da imagem do header
    byte1 = np.uint16(encoded[0])
    byte2 = np.uint16(encoded[1])
    width = (byte1 << 8) | byte2
    # reconstruindo a imagem
    decoded = []
    i = 2
    while i < encoded.size:
        count = encoded[i]
        value = encoded[i + 1]
        decoded.extend([value] * count)
        i += 2
    # convertendo para array numpy e redimensionando
    decoded = np.array(decoded, dtype=np.uint8)
    image = decoded.reshape(-1, width)
    return image

def main() -> None:
    IMAGE = "capivara.jpg"
    image = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    image = np.array(image, dtype=np.uint8)
    print(image.size)

    begin = perf_counter()
    encoded = run_length_encoding(image)
    end = perf_counter()
    print(f"Encoding time: {end - begin: .6f} seconds")
    print(encoded.size)

    begin = perf_counter()
    decoded = run_length_decoding(encoded)
    end = perf_counter()
    print(f"Decoding time: {end - begin: .6f} seconds")
    print(decoded.size)

    print(np.all(image == decoded))

if __name__ == "__main__":
    main()
