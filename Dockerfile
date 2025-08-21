# ==== Base image: PyTorch + CUDA + cuDNN (GPU ready) ====
FROM nvcr.io/nvidia/pytorch:23.08-py3

# ==== Environment ====
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

# ==== System dependencies ====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ==== Python dependencies ====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==== Project files ====
COPY . .

# Default command (you can override this with runai submit)
ENV PYTHONPATH=/workspace:/workspace/SSL
CMD ["python", "-m", "SSL.run_CRNN"]


