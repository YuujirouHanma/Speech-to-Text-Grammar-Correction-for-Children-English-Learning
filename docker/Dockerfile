# syntax=docker/dockerfile:1.7
# CPU image: CTranslate2 (int8) + faster-whisper. No PyTorch -> small image, fast start.
# Build:  docker build -f docker/Dockerfile -t speech-grammar:cpu .
# Run:    docker run -p 8000:8000 -v ./models:/models:ro speech-grammar:cpu

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---------------------------------------------------------------- builder stage
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
# Optional features. ERRANT (error types in the learner feedback) is on by default;
# --build-arg EXTRAS= builds a slimmer image without it, EXTRAS=errant,cloud adds cloud STT.
ARG EXTRAS="errant"
RUN pip install --upgrade pip wheel \
 && if [ -n "${EXTRAS}" ]; then pip install -c constraints.txt ".[${EXTRAS}]"; else pip install -c constraints.txt .; fi \
 && case ",${EXTRAS}," in *,errant,*) python -m spacy download en_core_web_sm ;; esac \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- runtime stage
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="speech-grammar" \
      org.opencontainers.image.description="Speech transcription + grammar correction API (Whisper + T5)"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --create-home --home-dir /home/app app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    SPEECH_GRAMMAR_ENVIRONMENT=prod \
    SPEECH_GRAMMAR_MODELS_DIR=/models \
    SPEECH_GRAMMAR_DEVICE=cpu \
    SPEECH_GRAMMAR_CORRECTOR_BACKEND=ctranslate2 \
    SPEECH_GRAMMAR_CORRECTOR_FALLBACK_BACKEND=none \
    SPEECH_GRAMMAR_TEMP_DIR=/tmp/speech_grammar

COPY --from=builder /opt/venv /opt/venv
WORKDIR /home/app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["speech-grammar"]
