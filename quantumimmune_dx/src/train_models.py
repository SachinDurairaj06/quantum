"""
Model training for QuantumImmune Dx
Trains Quantum SVM and classical baselines (Random Forest, SVM)
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries - handle gracefully if not available
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    import pennylane as qml
    from pennylane import numpy as pnp  # PennyLane's wrapped numpy
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available")

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available")

def load_and_preprocess_data(dataset_path: str = "data/dataset.csv") -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
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

def get_quantum_feature_map(n_features: int, reps: int = 2):
    """
    Create quantum feature map for quantum kernel computation
    
    Args:
        n_features: Number of features
        reps: Number of repetitions for the feature map
        
    Returns:
        Quantum feature map circuit
    """
    if PENNYLANE_AVAILABLE:
        # PennyLane approach
        dev = qml.device('default.qubit', wires=n_features)
        
        @qml.qnode(dev)
        def circuit(inputs, weights=None):
            # Angle encoding
            for i in range(n_features):
                qml.RX(inputs[i], wires=i)
            
            # Entangling layers
            for rep in range(reps):
                # Circular entangling
                for i in range(n_features):
                    qml.CNOT(wires=[i, (i+1) % n_features])
                # Optional: add some rotation gates for more expressibility
                for i in range(n_features):
                    qml.RZ(inputs[i] * 0.1, wires=i)  # Small feature-dependent rotation
            
            return qml.state()
        
        return circuit
    elif QISKIT_AVAILABLE:
        # Qiskit approach
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
        return feature_map
    else:
        raise RuntimeError("No quantum library available")

def compute_quantum_kernel_matrix(X: np.ndarray, feature_map, method: str = "pennylane") -> np.ndarray:
    """
    Compute quantum kernel matrix (state fidelity) between all pairs of samples
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_map: Quantum feature map function/circuit
        method: "pennylane" or "qiskit"
        
    Returns:
        Kernel matrix (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    
    if method == "pennylane" and PENNYLANE_AVAILABLE:
        # Compute fidelity between quantum states
        for i in range(n_samples):
            for j in range(i, n_samples):
                state_i = feature_map(X[i])
                state_j = feature_map(X[j])
                
                # Fidelity = |<ψ_i|ψ_j>|^2
                fidelity = np.abs(np.vdot(state_i, state_j))**2
                K[i, j] = fidelity
                K[j, i] = fidelity  # Symmetric
                
    elif method == "qiskit" and QISKIT_AVAILABLE:
        # Use Qiskit's QuantumKernel
        quantum_kernel = QuantumKernel(feature_map=feature_map)
        K = quantum_kernel.evaluate(X_vec=X)
        
    else:
        raise RuntimeError(f"Method {method} not available or library not installed")
    
    return K

def train_qsvm(X_train: np.ndarray, y_train: np.ndarray, 
               X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Train Quantum Support Vector Machine
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not (PENNYLANE_AVAILABLE or QISKIT_AVAILABLE):
        return {"error": "No quantum library available"}
    
    try:
        n_features = X_train.shape[1]
        
        # Create feature map
        feature_map = get_quantum_feature_map(n_features, reps=2)
        
        # Compute kernel matrices
        print("Computing training kernel matrix...")
        K_train = compute_quantum_kernel_matrix(X_train, feature_map, 
                                              method="pennylane" if PENNYLANE_AVAILABLE else "qiskit")
        
        print("Computing test kernel matrix...")
        # For test kernel, we need kernel between test and training samples
        K_test = compute_test_kernel_matrix(X_train, X_test, feature_map,
                                          method="pennylane" if PENNYLANE_AVAILABLE else "qiskit")
        
        # Train SVM with precomputed kernel
        if SKLEARN_AVAILABLE:
            # Use one-vs-rest for multi-class
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
                "kernel_test": K_test
            }
        else:
            return {"error": "scikit-learn not available for SVM"}
            
    except Exception as e:
        return {"error": f"QSVM training failed: {str(e)}"}

def compute_test_kernel_matrix(X_train: np.ndarray, X_test: np.ndarray, 
                              feature_map, method: str = "pennylane") -> np.ndarray:
    """
    Compute kernel matrix between test samples and training samples
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        feature_map: Quantum feature map
        method: "pennylane" or "qiskit"
        
    Returns:
        Kernel matrix (n_test_samples, n_train_samples)
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    K = np.zeros((n_test, n_train))
    
    if method == "pennylane" and PENNYLANE_AVAILABLE:
        for i in range(n_test):
            for j in range(n_train):
                state_i = feature_map(X_test[i])
                state_j = feature_map(X_train[j])
                fidelity = np.abs(np.vdot(state_i, state_j))**2
                K[i, j] = fidelity
                
    elif method == "qiskit" and QISKIT_AVAILABLE:
        quantum_kernel = QuantumKernel(feature_map=feature_map)
        K = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
    else:
        raise RuntimeError(f"Method {method} not available")
    
    return K

def train_classical_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Train classical baselines: Random Forest and SVM
    
    Returns:
        Dictionary with models, predictions, and metrics
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
    
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
        "f1_score": f1_score(y_test, rf_pred, average='macro')
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
        "f1_score": f1_score(y_test, svm_pred, average='macro')
    }
    
    return results

def main():
    """Main training function"""
    print("Starting QuantumImmune Dx model training...")
    
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
    
    # Train QSVM
    print("\n=== Training Quantum SVM ===")
    qsvm_results = train_qsvm(X_train, y_train, X_test, y_test)
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
            print(f"{model_name.replace('_', ' ').title()} Accuracy: {model_result['accuracy']:.4f}")
            print(f"{model_name.replace('_', ' ').title()} F1-Score: {model_result['f1_score']:.4f}")
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
        # Also save kernel matrices for inspection
        joblib.dump(qsvm_results["kernel_train"], "models/qsvm_kernel_train.joblib")
        joblib.dump(qsvm_results["kernel_test"], "models/qsvm_kernel_test.joblib")
    
    if "error" not in classical_results.get("random_forest", {}):
        joblib.dump(classical_results["random_forest"]["model"], "models/rf.joblib")
    
    if "error" not in classical_results.get("svm_classical", {}):
        joblib.dump(classical_results["svm_classical"]["model"], "models/svm_classical.joblib")
    
    # Save evaluation results
    eval_results = {}
    for model_name in ["qsvm", "random_forest", "svm_classical"]:
        if model_name in results and "error" not in results[model_name]:
            eval_results[model_name] = {
                "accuracy": results[model_name]["accuracy"],
                "f1_score": results[model_name]["f1_score"]
            }
    
    import json
    with open("models/evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print("Models saved to models/ directory")
    print("Training complete!")

if __name__ == "__main__":
    main()
