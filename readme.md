# Setup

## Virtual environment

Create:

`python -m venv myenv`

Activate
Linux:

`source myenv/bin/activate`

Powershell:

`.\myenv\Scripts\Activate.ps1`

## Requirements

### Python packages

`pip install -r requirement.txt`

### Install ffmpeg

Linix:
`apt update && apt install ffmpeg -y`

Windows:
`winget install ffmpeg`

## Start

### Single worker

`uvicorn api:app --host localhost --port 8000`

### Multiple workeres

`uvicorn api:app --host localhost --port 8000 --workers 4`

### Swagger

`http://localhost:8000/docs`

#### Vs code connection issue

setx TEMP "C:\Users\$env:USERNAME\AppData\Local\Temp"

setx TMP "C:\Users\$env:USERNAME\AppData\Local\Temp"
