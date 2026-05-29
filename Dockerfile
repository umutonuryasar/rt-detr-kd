FROM python:3.12-slim

WORKDIR /app

# Runtime libs required by PyTorch and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in two layers so model-code changes don't bust the
# expensive torch install layer.
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-serve.txt

# Project source
COPY src/      ./src/
COPY serve/    ./serve/
COPY configs/  ./configs/

ENV PYTHONPATH=/app

EXPOSE 8000

# MODEL_PATH must be supplied at runtime, e.g.:
#   docker run -e MODEL_PATH=/weights/checkpoint_best.pth \
#              -v $(pwd)/weights:/weights ...
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
