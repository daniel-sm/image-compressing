import cv2 as cv
import numpy as np
from numpy.typing import NDArray

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]
BYTE = np.uint8


def encoding(image: IMG_BYTE) -> IMG_BYTE:
    # redimensionando a imagem para uma lista de pixels
    pixels = image.reshape(-1, 3)

    # obtendo cores distintas
    distinct = np.unique(pixels, axis=0)
    distinct_size = distinct.shape[0]

    # criando uma tabela de cores
    color_table = { tuple(distinct[i]): np.uint8(i) for i in range(distinct_size) }

    # gerando imagem codificada
    encoded = np.array([color_table[tuple(c)] for c in pixels], dtype=np.uint32)

    # calculando os tamanhos em bytes
    bits_per_pixel = int(np.ceil(np.log2(distinct_size)))
    total_bytes = int(np.ceil(encoded.size * bits_per_pixel / 8))
    table_size = distinct_size

    # montando array de bytes compactados
    compressed = np.zeros(total_bytes, dtype=np.uint8)
    shift = 7
    index = 0
    code: np.uint32
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

    # adicionando altura da imagem
    height = image.shape[0]
    header_bytes.append(np.uint8(height >> 8 & 0xFF))
    header_bytes.append(np.uint8(height & 0xFF))
    # adicionando largura da imagem
    width = image.shape[1]
    header_bytes.append(np.uint8(width >> 8 & 0xFF))
    header_bytes.append(np.uint8(width & 0xFF))

    # adicionando quantos bits por codigo
    header_bytes.append(np.uint8(bits_per_pixel))
    # adicionando padding do ultimo byte
    header_bytes.append(np.uint8(shift + 1))
    # adicionando tamanho da tabela de cores
    header_bytes.append(np.uint8(table_size >> 16 & 0xFF))
    header_bytes.append(np.uint8(table_size >> 8 & 0xFF))
    header_bytes.append(np.uint8(table_size & 0xFF))

    # adicionando tabela de cores
    color: tuple[BYTE, BYTE, BYTE]
    for color in distinct:
        r = color[0]
        g = color[1]
        b = color[2]
        header_bytes.append(r)
        header_bytes.append(g)
        header_bytes.append(b)

    # convertendo header para array de bytes
    header = np.array(header_bytes, dtype=BYTE)
    # concatenando header e dados compactados
    result = np.concatenate((header, compressed))
    # retornando o resultado
    return result


def decoding(encoded: IMG_BYTE) -> IMG_BYTE:
    # lendo a altura da imagem
    byte1 = np.uint16(encoded[0])
    byte2 = np.uint16(encoded[1])
    height = int((byte1 << 8) | byte2)
    # lendo a largura da imagem
    byte1 = np.uint16(encoded[2])
    byte2 = np.uint16(encoded[3])
    width = int((byte1 << 8) | byte2)
    # lendo o numero de bits por codigo
    bits_per_pixel = int(encoded[4])
    # lendo o padding do ultimo byte
    padding = int(encoded[5])
    # lendo o tamanho da tabela de cores
    byte1 = np.uint32(encoded[6])
    byte2 = np.uint32(encoded[7])
    byte3 = np.uint32(encoded[8])
    table_size = int((byte1 << 16) | (byte2 << 8) | byte3)

    # reconstruindo a tabela de cores
    color_table: dict[BYTE, tuple[BYTE, BYTE, BYTE]] = {}
    index = 9
    for i in range(table_size):
        r = np.uint8(encoded[index])
        g = np.uint8(encoded[index + 1])
        b = np.uint8(encoded[index + 2])
        color_table[np.uint8(i)] = (r, g, b)
        index += 3

    # obtendo os dados compactados
    compressed = encoded[index:]
    # total de pixels
    total_pixels = (compressed.size * 8 - padding) // bits_per_pixel

    # array para imagem decodificada
    decoded = np.zeros(shape=(total_pixels, 3), dtype=np.uint8)

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
        decoded[i][0] = color_table[code][0]
        decoded[i][1] = color_table[code][1]
        decoded[i][2] = color_table[code][2]
    # redimensionando a imagem para suas dimensoes originais
    image = decoded.reshape((height, width, 3))
    # retornando a imagem redimensionada
    return image


def main():
    image = cv.imread("capivara.bmp")
    if image is None:
        raise FileNotFoundError("Image not found")

    image = np.array(image, dtype=np.uint8)
    print("Original:", image.shape)

    compressed = encoding(image)
    print("Compressed:", compressed.shape)

    decoded = decoding(compressed)
    print("Decoded:", decoded.size)

    # array = np.array([
    #     [[255,   0,   0], [255, 128,   0], [255, 255,   0], [128, 255,   0]],
    #     [[  0, 255,   0], [  0, 255, 128], [  0, 255, 255], [  0, 128, 255]],
    #     [[  0,   0, 255], [128,   0, 255], [255,   0, 255], [255,   0, 128]],
    #     [[128, 128, 128], [ 64,  64,  64], [192, 192, 192], [  0,   0,   0]]
    # ], dtype=np.uint8)

    # compressed = encoding(array)
    # print(compressed, compressed.size)

    # decoded = decoding(compressed)
    # print(decoded, decoded.size)

if "__main__" == __name__:
    main()
