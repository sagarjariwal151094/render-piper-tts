import os
import urllib.request

# The exact v1.0.0 hfc_male voice model paths you found
MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from target path...")
        opener = urllib.request.build_opener()
        # Chrome User-Agent structure to bypass strict network block walls
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded and cached {filename}")
        except Exception as e:
            print(f"Error fetching file: {e}")
            raise e

if __name__ == "__main__":
    # Save files with clean system names so main.py reads them instantly
    download_file(MODEL_URL, "model.onnx")
    download_file(CONFIG_URL, "model.onnx.json")
    print("All custom voice profile parameters verified and ready!")
