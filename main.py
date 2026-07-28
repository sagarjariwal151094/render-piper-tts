import os
import gc
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
from piper_onnx import Piper

app = FastAPI(title="Free Low-RAM Piper TTS API")

MODEL_PATH = "model.onnx"
CONFIG_PATH = "model.onnx.json"
piper_engine = None

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Piper Engine Fix Active"}

@app.post("/v1/predict")
async def generate_speech(request: TTSRequest):
    global piper_engine
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Initialize Piper only on the first request to save RAM
        if piper_engine is None:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
                raise HTTPException(status_code=500, detail="Piper model or configuration files are missing.")
            piper_engine = Piper(MODEL_PATH, CONFIG_PATH)
            
        # Synthesize audio (Piper returns raw PCM 16-bit audio)
        audio_bytes = piper_engine.synthesize(request.text)
        
        # FIX: Explicitly convert the output into a numpy array 
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        output_filename = "piper_output.wav"
        if os.path.exists(output_filename):
            os.remove(output_filename)
            
        # Write out raw PCM frames to a standard playable WAV file
        sf.write(output_filename, audio_array, 22050)
        
        # Free up unused memory instantly
        del audio_bytes
        del audio_array
        gc.collect()
        
        return FileResponse(output_filename, media_type="audio/wav", filename="audio.wav")
        
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")
