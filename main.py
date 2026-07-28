import os
import gc
import re
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

def split_into_sentences(text):
    """Splits large text blocks into short, safe sentence chunks"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

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
            
        # Break text down into small pieces
        sentences = split_into_sentences(request.text)
        all_audio_chunks = []
        sample_rate = 22050 # Fallback default samplerate
        
        # Process each piece separately to prevent the 30-second gateway timeout
        for sentence in sentences:
            samples, current_rate = piper_engine.create(sentence)
            if samples is not None and len(samples) > 0:
                all_audio_chunks.append(samples)
                sample_rate = current_rate # Map actual rate from model
        
        if not all_audio_chunks:
            raise HTTPException(status_code=500, detail="Audio generation yielded no data")
            
        # Combine all sentence audio chunks together smoothly
        final_audio = np.concatenate(all_audio_chunks)
        
        output_filename = "piper_output.wav"
        if os.path.exists(output_filename):
            os.remove(output_filename)
            
        # Save the combined audio file
        sf.write(output_filename, final_audio, sample_rate)
        
        # Clean up memory buffers immediately
        del all_audio_chunks
        del final_audio
        gc.collect()
        
        return FileResponse(output_filename, media_type="audio/wav", filename="audio.wav")
        
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")
