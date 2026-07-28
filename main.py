import os
import gc
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
from piper_onnx import Piper

app = FastAPI(title="Free Low-RAM Piper TTS API")

MODEL_PATH = "model.onnx"
CONFIG_PATH = "model.onnx.json"
piper_engine = None

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Piper Tuple Unpack Active"}

@app.post("/v1/predict")
async def generate_speech(request: TTSRequest):
    global piper_engine
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Initialize Piper engine on first WordPress request
        if piper_engine is None:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
                raise HTTPException(status_code=500, detail="Piper model or configuration files are missing.")
            piper_engine = Piper(MODEL_PATH, CONFIG_PATH)
            
        # FIX: Unpack the tuple directly into samples array and its native sample rate
        samples, sample_rate = piper_engine.create(request.text)
        
        output_filename = "piper_output.wav"
        if os.path.exists(output_filename):
            os.remove(output_filename)
            
        # Write the numpy samples directly using the model's native sample rate
        sf.write(output_filename, samples, sample_rate)
        
        # Clean up memory buffers immediately
        del samples
        gc.collect()
        
        return FileResponse(output_filename, media_type="audio/wav", filename="audio.wav")
        
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")
