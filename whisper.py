import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

start = time.time()
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
result = pipe("test_data/danish_news_60_sec.mp4", generate_kwargs={"language": "danish"})
print(result["text"])

# Print time it took to run the code in minutes
print("Time it took to run the code in minutes rounded to 2 decimals")
print(round((time.time() - start) / 60, 2))

# Print time it took to run the code in seconds
print("Time it took to run the code in seconds rounded to 2 decimals")
print(round(time.time() - start, 2))