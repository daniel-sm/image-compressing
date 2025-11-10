import cv2 as cv
import numpy as np
from time import perf_counter
from numpy.typing import NDArray

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]
BYTE = np.uint8

def encoding(image: IMG_BYTE) -> IMG_BYTE:
    # criando uma tabela de cores
    color_table: dict[BYTE, BYTE] = {}
    distinct = np.unique(image)
    color: BYTE
    for color in distinct:
        color_table[color] = np.uint8(len(color_table))

    # gerando imagem codificada
    encoded: list[BYTE] = []
    pixel: BYTE
    for pixel in image.ravel():
        encoded.append(color_table[pixel])

    # calculando tamanho em bytes
    bits_per_pixel: int = np.ceil(np.log2(distinct.size)).astype(int)
    total_bytes: int = np.ceil(len(encoded) * bits_per_pixel / 8).astype(int)
    table_size: BYTE = np.uint8(distinct.size if distinct.size < 256 else 0)

    # montando array de bytes compactados
    compressed = np.zeros(total_bytes, dtype=np.uint8)
    shift = 7
    index = 0
    code: BYTE
    for code in encoded:
        for i in range(bits_per_pixel - 1, -1, -1):
            if (shift < 0):
                shift = 7
                index += 1
            bit = (code >> i) & 1
            compressed[index] |= (bit << shift)
            shift -= 1

    # criando a lista de bytes do header
    header_bytes: list[BYTE] = []

    # adicionando largura da imagem
    width = np.uint16(image.shape[1])
    byte1 = width >> 8
    byte2 = width & 0b1111_1111
    header_bytes.append(np.uint8(byte1))
    header_bytes.append(np.uint8(byte2))

    # adicionando quantos bits por codigo e tamanho da tabela
    header_bytes.append(np.uint8(bits_per_pixel))
    # adicionando padding do ultimo byte
    header_bytes.append(np.uint8(shift + 1))
    # adicionando tamanho da tabela de cores
    header_bytes.append(table_size)

    # adicionando tabela de cores
    color: BYTE
    for color in distinct:
        header_bytes.append(color)
    header = np.array(header_bytes, dtype=BYTE)

    # concatenando header e dados compactados
    result = np.concatenate((header, compressed))
    return result

def decoding(encoded: IMG_BYTE) -> IMG_BYTE:
    # lendo a largura da imagem
    byte1 = np.uint16(encoded[0])
    byte2 = np.uint16(encoded[1])
    width = (byte1 << 8) | byte2
    # lendo o numero de bits por codigo
    bits_per_pixel: int = int(encoded[2])
    # lendo o padding do ultimo byte
    padding: int = int(encoded[3])
    # lendo o tamanho da tabela de cores
    table_size: int = int(encoded[4]) or 256 # tabela inteira se for 0

    # reconstruindo a tabela de cores
    color_table: dict[BYTE, BYTE] = {}
    index = 5
    for i in range(table_size):
        color_table[np.uint8(i)] = encoded[index]
        index += 1

    # obtendo os dados compactados
    compressed = encoded[index:]
    # total de pixels
    total_pixels = (compressed.size * 8 - padding) // bits_per_pixel
    # array para imagem decodificada
    decoded = np.zeros(shape=total_pixels, dtype=np.uint8)

    # decodificando os pixels
    index = 0
    shift = 7
    for i in range(total_pixels):
        code = np.uint8(0)
        for j in range(bits_per_pixel - 1, -1, -1):
            if shift < 0:
                shift = 7
                index += 1
            bit = (compressed[index] >> shift) & 1
            code |= (bit << j)
            shift -= 1
        decoded[i] = color_table[code]
    # retornando a imagem redimensionada
    return decoded.reshape((-1, width))

def main():
    image = cv.imread("capivara.bmp", cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    image = np.array(image, dtype=np.uint8)
    print("Original:", image.size)

    compressed = encoding(image)
    print("Compressed:", compressed.size)

    decoded = decoding(compressed)
    print("Decoded:", decoded.size)

    """ array = np.array([
        [0, 50],
        [100, 200],
    ], dtype=np.uint8)
    print(array, array.size)

    compressed = encoding(array)
    print(compressed, compressed.size)

    decoded = decoding(compressed)
    print(decoded, decoded.size) """

if "__main__" == __name__:
    main()
