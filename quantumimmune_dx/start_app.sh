#!/bin/bash
# Startup script for QuantumImmune Dx Streamlit app

echo "Starting QuantumImmune Dx..."
echo "Checking if dataset exists..."

if [ ! -f "data/dataset.csv" ]; then
    echo "Dataset not found. Generating synthetic dataset..."
    python3 src/generate_dataset_simple.py
fi

echo "Checking if models exist..."
if [ ! -f "models/qsvm.joblib" ] || [ ! -f "models/rf.joblib" ] || [ ! -f "models/svm_classical.joblib" ]; then
    echo "Models not found. Training models..."
    python3 src/train_models_simple.py
fi

echo "Launching Streamlit app..."
streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
