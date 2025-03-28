import os
import logging

import gradio as gr
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.logger_config import setup_logging
from utils.device import get_device, get_torch_and_np_dtypes

from dotenv import load_dotenv
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


MODEL_ID = os.getenv("MODEL_ID", "KBLab/kb-whisper-large")


device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(
    f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}"
)

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
transcribe_pipeline(warmup_audio)
logger.info("Model warmup complete")

async def transcribe(stream, audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    language = os.getenv("TRANSCRIPTION_LANGUAGE", "english") # Get language from .env
    logger.info(f"Using language: {language}") # Optional: log the language

    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    audio_array = audio_array.astype(np_dtype)
    audio_array /= np.max(np.abs(audio_array))
    
    if stream is not None:
        stream = np.concatenate((stream, audio_array))
    else:
        stream = audio_array

    outputs = transcribe_pipeline(
        {"sampling_rate": sample_rate, "raw": audio_array},
        chunk_length_s=4,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': language,
        },
        #return_timestamps="word"
    )
    return stream, outputs["text"].strip()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Audio Stream", streaming=True)
        with gr.Column():
            transcript = gr.Textbox(label="Transcript", value="")
        
        state = gr.State()
        audio_input.stream(
            transcribe, 
            inputs=[state, audio_input], 
            outputs=[state, transcript],
            stream_every=2
        )

        clear_button = gr.Button("Clear")
        clear_button.click(
            lambda: None, # clear the state
            outputs=[state]
        ).then(
            lambda: "", # clear the transcript
            outputs=[transcript]
        )

if __name__ == "__main__":
    
    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)
    
    demo.launch(
        server_name=server_name,
        server_port=port,
        debug=True
    )
