import time
import io
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

app = FastAPI()

# -----------------------------
# Model & Pipeline Initialization
# -----------------------------

# Check for GPU or CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Initialize the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,    
)

# -----------------------------
# API Endpoints
# -----------------------------
class TranscriptionResponse(BaseModel):
    transcription: str
    duration_seconds: float

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
 
    contents = await file.read()
    # Transcription
    start = time.time()
    result = pipe(contents, generate_kwargs={
        "max_new_tokens": 200,
        "return_timestamps": True,
        "language": "danish"
    })
    duration = time.time() - start

    return TranscriptionResponse(
        transcription=result["text"],
        duration_seconds=round(duration, 2)
    )

# -----------------------------
# Run with Uvicorn (if run directly)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
