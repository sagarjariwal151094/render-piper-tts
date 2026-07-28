import os
import urllib.request

# Low-memory, high-quality English voice files
MODEL_URL = "https://github.com"
CONFIG_URL = "https://github.com.json"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, filename)
        print(f"Finished downloading {filename}")

if __name__ == "__main__":
    download_file(MODEL_URL, "model.onnx")
    download_file(CONFIG_URL, "model.onnx.json")
    print("Piper TTS assets successfully cached!")
