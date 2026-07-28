import os
import urllib.request

# High-speed alternative mirrors hosted on Hugging Face
MODEL_URL = "https://huggingface.co"
CONFIG_URL = "https://huggingface.co.json"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully cached {filename}")
        except Exception as e:
            print(f"Download failed for {filename}: {e}")
            raise e

if __name__ == "__main__":
    download_file(MODEL_URL, "model.onnx")
    download_file(CONFIG_URL, "model.onnx.json")
    print("All backend Piper engine dependencies loaded!")
