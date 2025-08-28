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

# strip any --hash=sha256:... pins so pip can choose the correct Linux/CUDA wheels
RUN python - <<'PY'
import re, pathlib
txt = pathlib.Path('requirements.txt').read_text()
txt = re.sub(r'\s*--hash=sha256:[a-f0-9]{64}', '', txt)   # remove all hash pins
txt = re.sub(r'\n{2,}', '\n', txt).strip() + '\n'        # tidy up
pathlib.Path('requirements.nohash.txt').write_text(txt)
PY

# (optional but recommended) upgrade pip, then install
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.nohash.txt


# ==== Project files ====
COPY . .

# Default command (you can override this with runai submit)
ENV PYTHONPATH=/workspace:/workspace/SSL
CMD ["python", "-m", "SSL.run_CRNN"]


