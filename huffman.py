import cv2 as cv
import numpy as np
import heapq
from typing import Dict, Tuple
from numpy.typing import NDArray

IMAGE_BYTE = NDArray[np.uint8]
FREQUENCIES_TYPE = Dict[Tuple[int, int, int], int]
CODES_TYPE = Dict[Tuple[int, int, int], str]

def compute_frequencies(image: np.ndarray) -> FREQUENCIES_TYPE:
    flat = image.reshape(-1, 3)
    freqs: FREQUENCIES_TYPE = {}

    for r, g, b in flat:
        key = (int(r), int(g), int(b))
        freqs[key] = freqs.get(key, 0) + 1

    return freqs


def build_huffman_tree(freqs: FREQUENCIES_TYPE) -> CODES_TYPE:
    heap = []
    unique_id = 0

    for symbol in sorted(freqs.keys()):
        heapq.heappush(heap, (freqs[symbol], unique_id, symbol))
        unique_id += 1

    while len(heap) > 1:
        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)
        new_node = (left, right)
        heapq.heappush(heap, (freq1 + freq2, unique_id, new_node))
        unique_id += 1

    _, _, tree = heap[0]

    codes: CODES_TYPE = {}

    def traverse(node, prefix):
        if isinstance(node, tuple) and isinstance(node[0], int):
            codes[node] = prefix
            return
        left, right = node
        traverse(left, prefix + "0")
        traverse(right, prefix + "1")

    traverse(tree, "")
    return codes


def encode_huffman(image: IMAGE_BYTE) -> IMAGE_BYTE:
    freqs = compute_frequencies(image)
    codes = build_huffman_tree(freqs)

    entries = sorted(freqs.items())
    table = np.empty(len(entries) * 7, dtype=np.uint8)

    idx = 0
    for (r, g, b), freq in entries:
        table[idx] = np.uint8(r)
        table[idx + 1] = np.uint8(g)
        table[idx + 2] = np.uint8(b)
        table[idx + 3: idx + 7] = np.frombuffer(np.uint32(freq).tobytes(), dtype=np.uint8)
        idx += 7

    flat = image.reshape(-1, 3)
    parts = []
    for r, g, b in flat:
        parts.append(codes[(int(r), int(g), int(b))])
    bitstream = "".join(parts)

    padding = (8 - (len(bitstream) % 8)) % 8
    if padding:
        bitstream += "0" * padding

    bits = [int(b) for b in bitstream]
    encoded = np.packbits(bits)

    table_size = table.size
    table_size_bytes = np.array([
        (table_size >> 16) & 0xFF,
        (table_size >> 8)  & 0xFF,
        table_size & 0xFF
    ], dtype=np.uint8)

    width = image.shape[1]

    header = np.zeros(6, dtype=np.uint8)
    header[0] = np.uint8(padding)
    header[1:3] = np.frombuffer(np.uint16(width).tobytes(), dtype=np.uint8)
    header[3:6] = table_size_bytes

    return np.concatenate([header, table, encoded]).astype(np.uint8)


def decode_huffman(encoded: IMAGE_BYTE) -> IMAGE_BYTE:
    padding = int(encoded[0])
    width = int(np.frombuffer(encoded[1:3].tobytes(), dtype=np.uint16)[0])
    table_size = (int(encoded[3]) << 16) | (int(encoded[4]) << 8) | int(encoded[5])

    table_start = 6
    table_end = table_start + table_size
    table = encoded[table_start:table_end]

    freqs = {}
    i = 0
    while i < table.size:
        r = int(table[i])
        g = int(table[i + 1])
        b = int(table[i + 2])
        freq = int(np.frombuffer(table[i + 3: i + 7].tobytes(), dtype=np.uint32)[0])
        freqs[(r, g, b)] = freq
        i += 7

    codes = build_huffman_tree(freqs)
    codes_reverse = { v: k for k, v in codes.items() }

    encoded_bytes = encoded[table_end:]
    
    bits = np.unpackbits(encoded_bytes)

    if padding:
        bits = bits[:-padding]

    decoded = []
    current = []

    for b in bits:
        current.append('1' if b else '0')
        key = ''.join(current)
        if key in codes_reverse:
            decoded.append(codes_reverse[key])
            current = []

    image = np.array(decoded, dtype=np.uint8).reshape((-1, width, 3))

    return image


def main():
    image = cv.imread("capivara.bmp")

    if image is None:
        print("Error: Could not read the image.")
        return
    
    image = np.array(image, dtype=np.uint8)
    print("Original:", image.dtype, image.shape, image.size)

    encoded = encode_huffman(image)
    print("Encoded:", encoded.dtype, encoded.shape, encoded.size)

    decoded = decode_huffman(encoded)
    print("Decoded:", decoded.dtype, decoded.shape, decoded.size)

    cv.imwrite("decoded_image.png", decoded)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
