import numpy as np
import cv2 as cv
from numpy.typing import NDArray

IMAGE_BYTE = NDArray[np.uint8]


def truncate(img: IMAGE_BYTE, bits: int = 4) -> IMAGE_BYTE:
    if bits < 1 or bits > 8:
        raise ValueError("Bits must be between 1 and 8")
    mask = (~((1 << (8 - bits)) - 1)) & 0xFF
    return (img & mask).astype(np.uint8)


def encoding(image: IMAGE_BYTE) -> IMAGE_BYTE:
    height, width, _ = image.shape
    pixels = image.reshape(-1, 3)

    if pixels.shape[0] % 2 != 0:
        pixels = np.vstack([pixels, np.zeros((1,3), dtype=np.uint8)])

    high = pixels >> 4
    high_1 = high[0::2]
    high_2 = high[1::2]

    compressed = ((high_1 << 4) | high_2).astype(np.uint8)
    header = np.zeros(4, dtype=np.uint8)
    header[0] = np.uint8(height >> 8 & 0xFF)
    header[1] = np.uint8(height & 0xFF)
    header[2] = np.uint8(width >> 8 & 0xFF)
    header[3] = np.uint8(width & 0xFF)

    # concatenando header com os dados comprimidos
    result = np.concatenate([header, compressed.flatten()])
    return result


def decoding(encoded: IMAGE_BYTE) -> IMAGE_BYTE:
    # lendo a altura da imagem  
    byte1 = np.uint16(encoded[0])
    byte2 = np.uint16(encoded[1])
    height = int((byte1 << 8) | byte2)
    # lendo a largura da imagem
    byte1 = np.uint16(encoded[2])
    byte2 = np.uint16(encoded[3])
    width = int((byte1 << 8) | byte2)

    # lendo os dados compactados
    packed = (encoded[4:]).reshape(-1, 3)

    high_1 = (packed >> 4)
    high_2 = (packed & 0b1111)

    shape = (packed.shape[0] * 2, 3)
    pixels = np.zeros(shape, dtype=np.uint8)

    pixels[0::2] = high_1
    pixels[1::2] = high_2
    pixels = (pixels << 4).astype(np.uint8)

    result = pixels[0:height * width].reshape((height, width, 3))

    return result


def main():
    image = cv.imread("image.bmp")

    if image is None:
        print("Error: Could not read the image.")
        return
    
    image = np.array(image, dtype=np.uint8)
    print("Original:", image.shape)

    truncated = truncate(image)

    compressed = encoding(image)
    print("Compressed size:", compressed.shape)
    
    decompressed = decoding(compressed)
    print("Decompressed size:", decompressed.shape)

    cv.imwrite("decompressed.bmp", decompressed)


if __name__ == "__main__":
    main()
