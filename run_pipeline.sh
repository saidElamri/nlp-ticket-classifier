#!/bin/bash
set -e

echo "========================================="
echo " NLP Pipeline — Full Batch Run"
echo "========================================="

echo ""
echo "[1/4] Preprocessing..."
python preprocess_advanced.py

echo ""
echo "[2/4] Training classifier..."
python train_classifier.py

echo ""
echo "[3/4] Generating embeddings & storing in ChromaDB..."
python generate_embeddings.py

echo ""
echo "[4/4] Generating Evidently AI drift report..."
python monitoring/evidently_report.py

echo ""
echo "========================================="
echo " Pipeline complete!"
echo "========================================="
