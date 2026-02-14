FROM python:3.11-slim

LABEL maintainer="NLP Pipeline - Support Ticket Classifier"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy project files
COPY preprocess_advanced.py .
COPY train_classifier.py .
COPY generate_embeddings.py .
COPY verify_chroma.py .
COPY monitoring/evidently_report.py monitoring/

# Copy data (will be mounted or baked-in for batch runs)
COPY data/ data/

# Entry point: run the full pipeline
COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
