# Slim image that can run the CLI end-to-end. Build:
#   docker build -t legal-rag-benchmark .
# Run (with a volume so data + indices persist):
#   docker run --rm -it -v $(pwd)/data:/work/data -v $(pwd)/results:/work/results \
#       legal-rag-benchmark lrb eval run --corpus contractnli

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# faiss + numpy need libgomp for OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

# default command shows help; override with eg `docker run ... lrb eval run ...`
ENTRYPOINT ["lrb"]
CMD ["--help"]
