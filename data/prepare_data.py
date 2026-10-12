import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "input.txt")

def download_tiny_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(FILE_PATH):
        print(f"Dataset already exists at: {FILE_PATH} ({os.path.getsize(FILE_PATH):,} bytes)")
        return FILE_PATH
    
    print(f"Downloading Tiny Shakespeare from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, FILE_PATH)
    file_size = os.path.getsize(FILE_PATH)
    print(f"Successfully downloaded to {FILE_PATH} ({file_size:,} bytes)")
    return FILE_PATH

if __name__ == "__main__":
    download_tiny_shakespeare()
