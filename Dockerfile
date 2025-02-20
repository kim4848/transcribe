# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Define build argument for model_id with a default value
ARG MODEL_ID="openai/whisper-large-v3-turbo"

# Update package list, install ffmpeg, transformers, torch, and pre-download Hugging Face model
RUN apt update && apt install ffmpeg -y && \
    python -m pip install transformers torch && \
    python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; model_id = '${MODEL_ID}'; AutoModelForSpeechSeq2Seq.from_pretrained(model_id); AutoProcessor.from_pretrained(model_id)"

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
