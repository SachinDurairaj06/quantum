"""
Simplified model training for QuantumImmune Dx
Uses classical ML with simulated quantum kernels for demonstration
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(dataset_path: str = "data/dataset.csv"):
    """
    Load and preprocess the dataset
    
    Returns:
        X: Feature matrix
        y: Encoded target vector
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
    """
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = df.drop('disease', axis=1).values
    y = df['disease'].values
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, scaler, label_encoder

def simulate_quantum_kernel(X: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    Simulate a quantum kernel using a classical RBF kernel with quantum-inspired properties
    This is a placeholder for actual quantum kernel computation
    
    Args:
        X: Feature matrix (n_samples, n_features)
        gamma: Kernel coefficient
        
    Returns:
        Kernel matrix (n_samples, n_samples)
    """
    from sklearn.metrics.pairwise import rbf_kernel
    # Use RBF kernel as a stand-in for quantum kernel
    # In a real implementation, this would be replaced with actual quantum state fidelity
    K = rbf_kernel(X, gamma=gamma)
    return K

def train_simulated_qsvm(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train a simulated Quantum Support Vector Machine using precomputed kernel
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    try:
        # Compute kernel matrices
        print("Computing training kernel matrix (simulated quantum)...")
        K_train = simulate_quantum_kernel(X_train, gamma=0.1)
        
        print("Computing test kernel matrix (simulated quantum)...")
        K_test = simulate_quantum_kernel(X_test, gamma=0.1)
        # For simplicity, we compute kernel between test and train separately
        # In practice, we'd need the cross-kernel
        from sklearn.metrics.pairwise import rbf_kernel
        K_test = rbf_kernel(X_test, X_train, gamma=0.1)
        
        # Train SVM with precomputed kernel
        model = SVC(kernel='precomputed', probability=True, random_state=42)
        model.fit(K_train, y_train)
        
        # Predict
        y_pred = model.predict(K_test)
        y_pred_proba = model.predict_proba(K_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        return {
            "model": model,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "accuracy": accuracy,
            "f1_score": f1,
            "kernel_train": K_train,
            "kernel_test": K_test,
            "model_type": "Simulated QSVM"
        }
    except Exception as e:
        return {"error": f"Simulated QSVM training failed: {str(e)}"}

def train_classical_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train classical baselines: Random Forest and SVM
    
    Returns:
        Dictionary with models, predictions, and metrics
    """
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    
    results["random_forest"] = {
        "model": rf_model,
        "predictions": rf_pred,
        "probabilities": rf_pred_proba,
        "accuracy": accuracy_score(y_test, rf_pred),
        "f1_score": f1_score(y_test, rf_pred, average='macro'),
        "model_type": "Random Forest"
    }
    
    # Classical SVM (RBF kernel)
    print("Training Classical SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_pred_proba = svm_model.predict_proba(X_test)
    
    results["svm_classical"] = {
        "model": svm_model,
        "predictions": svm_pred,
        "probabilities": svm_pred_proba,
        "accuracy": accuracy_score(y_test, svm_pred),
        "f1_score": f1_score(y_test, svm_pred, average='macro'),
        "model_type": "Classical SVM (RBF)"
    }
    
    return results

def main():
    """Main training function"""
    print("Starting QuantumImmune Dx model training (simplified version)...")
    
    # Check if dataset exists
    dataset_path = "data/dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please generate dataset first.")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, scaler, label_encoder = load_and_preprocess_data(dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Train models
    results = {}
    
    # Train Simulated QSVM
    print("\n=== Training Simulated Quantum SVM ===")
    qsvm_results = train_simulated_qsvm(X_train, y_train, X_test, y_test)
    results["qsvm"] = qsvm_results
    
    if "error" not in qsvm_results:
        print(f"QSVM Accuracy: {qsvm_results['accuracy']:.4f}")
        print(f"QSVM F1-Score: {qsvm_results['f1_score']:.4f}")
    else:
        print(f"QSVM Error: {qsvm_results.get('error', 'Unknown error')}")
    
    # Train classical baselines
    print("\n=== Training Classical Baselines ===")
    classical_results = train_classical_models(X_train, y_train, X_test, y_test)
    results.update(classical_results)
    
    for model_name, model_result in classical_results.items():
        if "error" not in model_result:
            print(f"{model_result['model_type']} Accuracy: {model_result['accuracy']:.4f}")
            print(f"{model_result['model_type']} F1-Score: {model_result['f1_score']:.4f}")
        else:
            print(f"{model_name.replace('_', ' ').title()} Error: {model_result.get('error', 'Unknown error')}")
    
    # Save models and preprocessors
    print("\n=== Saving Models ===")
    os.makedirs("models", exist_ok=True)
    
    # Save scaler and label encoder
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    
    # Save models if training succeeded
    if "error" not in qsvm_results:
        joblib.dump(qsvm_results["model"], "models/qsvm.joblib")
        # Save kernel matrices for inspection
        joblib.dump(qsvm_results["kernel_train"], "models/qsvm_kernel_train.joblib")
        joblib.dump(qsvm_results["kernel_test"], "models/qsvm_kernel_test.joblib")
    
    if "error" not in classical_results.get("random_forest", {}):
        joblib.dump(classical_results["random_forest"]["model"], "models/rf.joblib")
    
    if "error" not in classical_results.get("svm_classical", {}):
        joblib.dump(classical_results["svm_classical"]["model"], "models/svm_classical.joblib")
    
    # Save evaluation results
    eval_results = {}
    for model_key in ["qsvm", "random_forest", "svm_classical"]:
        if model_key in results and "error" not in results[model_key]:
            eval_results[model_key] = {
                "accuracy": results[model_key]["accuracy"],
                "f1_score": results[model_key]["f1_score"],
                "model_type": results[model_key].get("model_type", model_key)
            }
    
    with open("models/evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print("Models saved to models/ directory")
    print("Training complete!")

if __name__ == "__main__":
    main()
