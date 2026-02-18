from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import shutil
import os
import time

app = FastAPI(title="CPU Whisper API")

# CONFIGURATION
# 'tiny', 'base', 'small', 'medium', 'large-v2'
# 'small' is the sweet spot for CPU accuracy vs speed.
MODEL_SIZE = "small" 
COMPUTE_TYPE = "int8" # Optimized for CPU

print(f"Loading {MODEL_SIZE} model on CPU...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
print("Model loaded successfully!")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Save uploaded file temporarily
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Transcribe
        # beam_size=5 provides better accuracy for a small speed cost
        segments, info = model.transcribe(temp_filename, beam_size=5)
        
        # 3. Collect results
        transcript = []
        for segment in segments:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # 4. Cleanup
        os.remove(temp_filename)

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": transcript,
            "full_text": " ".join([s['text'] for s in transcript])
        }

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "model": MODEL_SIZE}
