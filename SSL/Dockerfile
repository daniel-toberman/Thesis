# IPDnet Training Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    pytorch-lightning \
    soundfile \
    scipy \
    jsonargparse[signatures]

# Copy the entire SSL directory
COPY . .

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Default command (can be overridden in RunAI)
CMD ["python3", "run_IPDnet_6cm.py", "fit", "--trainer.max_epochs=150", "--trainer.accelerator=gpu", "--trainer.devices=1"]
