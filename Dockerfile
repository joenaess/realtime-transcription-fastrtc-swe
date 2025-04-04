# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg (needed for audio processing by Whisper) and git (for potential dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# Consider using UV if preferred: RUN pip install uv && uv pip install --system --no-cache -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Make port 7860 available to the world outside this container (adjust if using a different port)
EXPOSE 7860

# Define environment variables (can be overridden at runtime) - change tiny to large for prod
ENV UI_MODE="fastapi"
ENV APP_MODE="local"
ENV MODEL_ID="KBLab/kb-whisper-large"
ENV TRANSCRIPTION_LANGUAGE="swedish"
ENV SERVER_NAME="0.0.0.0"
ENV PORT="7860"
ENV LOG_LEVEL="INFO"

# Run main.py when the container launches
CMD ["python", "main.py"]