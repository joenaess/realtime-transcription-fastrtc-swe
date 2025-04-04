import os
import logging
import json

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
    SileroVadOptions,
    audio_to_bytes,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.logger_config import setup_logging
from utils.device import get_device, get_torch_and_np_dtypes
from utils.turn_server import get_rtc_credentials


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


UI_MODE = os.getenv("UI_MODE", "fastapi")
APP_MODE = os.getenv("APP_MODE", "local")
MODEL_ID = os.getenv("MODEL_ID", "KBLab/kb-whisper-tiny")


device = get_device(force_cpu=False)

torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}")


attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
logger.info(f"Using attention: {attention}")

logger.info(f"Loading Whisper model: {MODEL_ID}")

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        attn_implementation=attention
    )
    model.to(device)
except Exception as e:
    logger.error(f"Error loading ASR model: {e}")
    logger.error(f"Are you providing a valid model ID? {MODEL_ID}")
    raise

processor = AutoProcessor.from_pretrained(MODEL_ID)

transcribe_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Warm up the model with empty audio
logger.info("Warming up Whisper model with dummy input")
warmup_audio = np.zeros((16000,), dtype=np_dtype)  # 1s of silence
transcribe_pipeline({"sampling_rate": 16000, "raw": warmup_audio})
logger.info("Model warmup complete")


async def transcribe(audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    language = os.getenv("TRANSCRIPTION_LANGUAGE", "swedish") # Get language from .env, default to swedish
    logger.info(f"Using language: {language}") # Optional: log the language

    outputs = transcribe_pipeline(
        audio_to_bytes(audio),
        chunk_length_s=3,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': language,
        },
        #return_timestamps="word"
    )
    yield AdditionalOutputs(outputs["text"].strip())


logger.info("Initializing FastRTC stream")
stream = Stream(
    handler=ReplyOnPause(
        transcribe, input_sample_rate=16000, output_sample_rate=16000, # Added sample_rate as handled for whisper models
        algo_options=AlgoOptions(
            # Duration in seconds of audio chunks (default 0.6) - change to 0.8 for more coherence
            audio_chunk_duration=0.8,
            # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
            started_talking_threshold=0.2,
            # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
            speech_threshold=0.1,
        ),
        model_options=SileroVadOptions(
            # Threshold for what is considered speech (default 0.5)
            threshold=0.5,
            # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
            min_speech_duration_ms=250,
            # Max duration of speech chunks, longer will be split (default float('inf'))
            max_speech_duration_s=30,
            # Wait for ms at the end of each speech chunk before separating it (default 2000) - change to 1500
            min_silence_duration_ms=1500,
            # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
            window_size_samples=512, # Changed to 512 to match 16kHz
            # Final speech chunks are padded by speech_pad_ms each side (default 400)
            speech_pad_ms=400,
        ),
    ),
    # send-receive: bidirectional streaming (default)
    # send: client to server only
    # receive: server to client only
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Transcript"),
    ],
    additional_outputs_handler=lambda current, new: current + " " + new,
    rtc_configuration=get_rtc_credentials(provider="hf") if APP_MODE == "deployed" else None
)

app = FastAPI()
stream.mount(app)

@app.get("/")
async def index():
    html_content = open("index.html").read()
    rtc_config = get_rtc_credentials(provider="hf") if APP_MODE == "deployed" else None
    return HTMLResponse(content=html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config)))

@app.get("/transcript")
def _(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                logger.debug(f"Sending transcript for {webrtc_id}: {transcript[:50]}...")
                yield f"event: output\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
            raise

    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":

    server_name = os.getenv("SERVER_NAME", "localhost")
    port = int(os.getenv("PORT", 7860)) # added int to avoid TypeError
    
    if UI_MODE == "gradio":
        logger.info("Launching Gradio UI")
        stream.ui.launch(
            server_port=port, 
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        import uvicorn
        logger.info("Launching FastAPI server")
        uvicorn.run(app, host=server_name, port=port)
