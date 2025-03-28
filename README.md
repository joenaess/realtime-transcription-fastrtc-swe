# Real-Time Swedish Speech Transcription with FastRTC & Whisper

**Note:** This repository is a fork of [sofi444/realtime-transcription-fastrtc](https://github.com/sofi444/realtime-transcription-fastrtc). This version has been modified to specifically target Swedish transcription using the `KBLab/kb-whisper-large` model and includes Docker support.

## Changes in this Fork

* **Default Model:** Configured by default to use the Swedish Whisper model `KBLab/kb-whisper-large`.
* **Language Configuration:** Added a `TRANSCRIPTION_LANGUAGE` environment variable (set to `"swedish"` by default in `.env` and `Dockerfile`) to control the target transcription language.
* **Docker Support:** Includes a `Dockerfile` for building and running the application in a containerized environment. See "Running with Docker" section below.

## Forked repo readme from here Real Time Speech Transcription with FastRTC ⚡️and Local Whisper 🤗

This project uses FastRTC to handle the live audio streaming and open-source Automatic Speech Recognition models via Transformers.

Check the [FastRTC documentation](https://fastrtc.org/) to see what parameters you can tweak with respect to the audio stream, Voice Activity Detection (VAD), etc.

**System Requirements**

* python >= 3.10
* ffmpeg

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/joenaess/realtime-transcription-fastrtc-swe.git
cd realtime-transcription-fastrtc

```

### Step 2: Set up environment

Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

</details>

<summary>🐍 Using pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

### Step 3: Install ffmpeg

<details>

<summary>🍎 macOS</summary>

```bash
brew install ffmpeg
```

</details>

<details>
<summary>🐧 Linux (Ubuntu/Debian)</summary>

```bash
sudo apt update
sudo apt install ffmpeg
```

</details>

### Step 4: Configure environment

Create a `.env` file in the project root with your desired settings:

```env
UI_MODE=fastapi       # Or gradio
APP_MODE=local        # Use 'deployed' if using a TURN server
MODEL_ID="KBLab/kb-whisper-large" # Default Swedish model
TRANSCRIPTION_LANGUAGE="swedish" # Target transcription language
SERVER_NAME=localhost # Optional: Host to bind to
PORT=7860             # Optional: Port number
# LOG_LEVEL=INFO        # Optional: Set to DEBUG for more verbose logs
```

* **UI_MODE**: controls the interface to use. If set to `gradio`, you will launch the app via Gradio and use their default UI. If set to anything else (eg. `fastapi`) it will use the `index.html` file in the root directory to create the UI (you can customise it as you want) (default `fastapi`).
* **APP_MODE**: ignore this if running only locally. If you're deploying eg. in Spaces, you need to configure a Turn Server. In that case, set it to `deployed`, follow the instructions [here](https://fastrtc.org/deployment/) (default `local`).
* **MODEL_ID**: HF model identifier for the ASR model you want to use (see [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)) (default `openai/whisper-large-v3-turbo`)
* **SERVER_NAME**: Host to bind to (default `localhost`)
* **PORT**: Port number (default `7860`)

### Step 5: Launch the application

```bash
python main.py
```

click on the url that pops up (eg. <https://localhost:7860>) to start using the app!

## Running with Docker

This repository includes a `Dockerfile` for easy setup and deployment.

1. **Build the Image:**

    ```bash
    docker build -t realtime-transcription-swe .
    ```

2. **Run the Container:**

    ```bash
    # Basic run, mapping port 7860
    docker run -p 7860:7860 --rm --name whisper-swe realtime-transcription-swe

    # Optional: Override log level (e.g., for debugging)
    # docker run -p 7860:7860 -e LOG_LEVEL="DEBUG" --rm --name whisper-swe realtime-transcription-swe

    # Optional: For NVIDIA GPU usage (requires nvidia-docker & compatible setup)
    # docker run --gpus all -p 7860:7860 --rm --name whisper-swe realtime-transcription-swe
    ```

Access the application at `http://localhost:7860`.

### Whisper

Choose the Whisper model version you want to use. See all [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending&search=whisper) - you can of course also use a non-Whisper ASR model.

On MPS, I can run `whisper-large-v3-turbo` without problems. This is my current favourite as it’s lightweight, performant and multi-lingual!

Adjust the parameters as you like, but remember that for real-time, we want the batch size to be 1 (i.e. start transcribing as soon as a chunk is available).

If you want to transcribe different languages, set the language parameter to the target language, otherwise Whisper defaults to translating to English (even if you set `transcribe` as the task).
