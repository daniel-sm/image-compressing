import sys
import re
import numpy as np

def valid_filename(name):
    pattern = r'^[\w,\-\.]+\.[A-Za-z0-9]+$'
    return re.match(pattern, name) is not None

def write_bytes(filename, data):
    with open(filename, "wb") as f:
        f.write(data.tobytes())

def read_bytes(filename):
    with open(filename, "rb") as f:
        content = f.read()
    return np.frombuffer(content, dtype=np.uint8)

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py [read|write] filename.ext")
        sys.exit(1)

    command = sys.argv[1]
    filename = sys.argv[2]

    if not valid_filename(filename):
        print("Invalid filename.")
        sys.exit(1)    

    if command == "write":
        array = np.array([48, 65, 122], dtype=np.uint8)
        write_bytes(filename, array)
        print("File written.")

    elif command == "read":
        try:
            array = read_bytes(filename)
            print(array)
        except FileNotFoundError:
            print("File not found.")
    
    else:
        print("Invalid command. Use read or write.")
        sys.exit(1)

if __name__ == "__main__":
    main()
