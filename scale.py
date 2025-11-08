import cv2 as cv
import numpy as np
from time import perf_counter
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]
IMG_DOUBLE = NDArray[np.float64]
REDUCE_FACTOR = 0.8
EXPAND_FACTOR = 1 / REDUCE_FACTOR

def to_double(img: IMG_BYTE) -> IMG_DOUBLE:
    return img.astype(np.float64) / 255

def to_byte(img: IMG_DOUBLE) -> IMG_BYTE:
    return (img * 255).astype(np.uint8)

def scale(img: IMG_BYTE, k: float) -> IMG_BYTE:
    # convertendo a imagem para double
    image = to_double(img)
    # obtendo as dimensoes da imagem
    height, width = image.shape

    # calculando os valores de mapeamento
    x_space = np.linspace(0, width - 1, int(k * width))
    y_space = np.linspace(0, height - 1, int(k * height))
    x, y = np.meshgrid(x_space, y_space)

    # coordenadas correspondentes na imagem original
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    # partes fracionarias
    dx = x - x0
    dy = y - y0

    # obtendo os valores dos pixels vizinhos da imagem original
    p1 = image[y0, x0]
    p2 = image[y0, x1]
    p3 = image[y1, x0]
    p4 = image[y1, x1]

    # interpolacao bilinear dos valores
    above = (p1 * (1 - dx)) + (p2 * dx)
    below = (p3 * (1 - dx)) + (p4 * dx)
    interpoled = (above * (1 - dy)) + (below * dy)

    # convertendo de volta para bytes
    result = to_byte(interpoled)
    # retornando a imagem redimensionada
    return result

def reduce(image: IMG_BYTE) -> IMG_BYTE:
    reduced = scale(image, REDUCE_FACTOR)

    width = reduced.shape[1]

    width_byte1 = np.uint8(width >> 8)
    width_byte2 = np.uint8(width & 0b1111_1111)

    header = np.array([width_byte1, width_byte2], dtype=np.uint8)

    return np.concatenate((header, reduced.ravel()))

def expand(image: IMG_BYTE) -> IMG_BYTE:
    width_byte1 = np.uint16(image[0])
    width_byte2 = np.uint16(image[1])
    width = (width_byte1 << 8) | width_byte2

    reduced = image[2:].reshape((-1, width))
    expanded = scale(reduced, EXPAND_FACTOR)

    return expanded

def main() -> None:
    IMAGE = "image.bmp"
    image = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    image = np.array(image, dtype=np.uint8)
    print("Original size:", image.size)

    reduced = reduce(image)
    print("Reduced size:", reduced.size)

    expanded = expand(reduced)
    print("Expanded size:", expanded.size)

    print(image)
    print(reduced)
    print(expanded)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Expanded")
    plt.imshow(expanded, cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
