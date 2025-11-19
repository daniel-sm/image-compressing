import cv2 as cv
import numpy as np
from time import perf_counter
from numpy.typing import NDArray

# criando tipo para imagem de bytes
IMG_BYTE = NDArray[np.uint8]

def predictive_coding_encoding(image: IMG_BYTE) -> IMG_BYTE:
    # convertendo a imagem para int16 para evitar overflow
    signed_image = image.astype(np.int16)

    # calculando os erros preditivos dos pixels
    predictive = np.zeros_like(signed_image, dtype=np.int16)
    predictive[:, 1:] = signed_image[:, 1:] - signed_image[:, :-1]

    # criando os intervalos para mapeamento
    bins = np.linspace(-255, 255, 256)
    # mapeando valores de -255:255 para 0:255
    encoded = (np.digitize(predictive, bins) - 1).astype(np.uint8)

    # mantendo o primeiro pixel de cada linha
    encoded[:, 0] = signed_image[:, 0]

    # retornando a imagem codificada
    return encoded

def predictive_coding_decoding(encoded: IMG_BYTE) -> IMG_BYTE:
    # transformando de 0:255 para -255:255
    predictive = 2 * encoded.astype(np.int16) - 255
    predictive[:, 0] = encoded[:, 0].astype(np.int16)

    # criando array para imagem decodificada
    decoded = np.zeros_like(encoded, dtype=np.uint8)
    # copiando o primeiro pixel de cada linha
    decoded[:, 0] = encoded[:, 0]
 
    # calculando os valores originais dos pixels
    for col in range(1, encoded.shape[1]):
        predictive[:, col] += decoded[:, col - 1].astype(np.int16)
        decoded[:, col] = np.clip(predictive[:, col], 0, 255).astype(np.uint8)

    # retornando a imagem decodificada
    return decoded

def main() -> None:
    image = cv.imread("image.bmp", cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")
    image = np.array(image, dtype=np.uint8)

    # array = np.array([[ 78,  35, 189, 177],
    #                   [240, 184, 136, 234],
    #                   [216,  49,  46, 104],
    #                   [140, 182, 204, 151],
    #                   [ 78, 138, 119,  25]], dtype=np.uint8)
    # print(array)

    begin = perf_counter()
    encoded = predictive_coding_encoding(image)
    end = perf_counter()
    print(f"Encoding time: {end - begin: .6f} seconds")
    print(encoded)

    begin = perf_counter()
    decoded = predictive_coding_decoding(encoded)
    end = perf_counter()
    print(f"Decoding time: {end - begin: .6f} seconds")
    print(decoded)

    cv.imshow("Original Image", image)
    cv.waitKey(0)

    cv.imshow("Decoded Image", decoded)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
