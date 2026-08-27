import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's numpy wrapper
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os
from functools import partial
from joblib import Parallel, delayed

# Load and preprocess data
def load_data(filepath):
    import csv
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        labels = []
        for row in reader:
            # Features: columns 1 to 18 (index 1 to 18) - all numeric
            features = [float(x) for x in row[1:19]]  # Age to Skin_lesion
            label = row[0]  # disease
            data.append(features)
            labels.append(label)
        return np.array(data), np.array(labels)

X, y_raw = load_data('data/dataset.csv')
print(f"Loaded data: {X.shape}, labels: {y_raw.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print(f"Unique classes: {label_encoder.classes_}")

# Normalize to [0, π] for quantum encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)
print(f"Data scaled to [{X_scaled.min()}, {X_scaled.max()}]")

# Save the scaler and label encoder for inference
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
print("Saved scaler and label encoder to models/")

# Quantum feature map: angle encoding with linear entanglement, depth=2
n_qubits = X_scaled.shape[1]  # 18 features
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def circuit(x):
    # Angle encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    # Linear entanglement, depth=2
    for _ in range(2):
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        for i in range(n_qubits - 1, 0, -1):
            qml.CNOT(wires=[i, i-1])
    return qml.state()

# Compute quantum kernel matrix (state fidelity) in parallel
def kernel_fidelity(x1, x2):
    state1 = circuit(x1)
    state2 = circuit(x2)
    fidelity = np.abs(np.dot(np.conj(state1), state2))**2
    return fidelity

def compute_kernel_matrix(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    
    # Compute diagonal (each vector with itself) -> should be 1
    for i in range(n_samples):
        K[i, i] = kernel_fidelity(X[i], X[i])
    
    # Compute off-diagonal in parallel
    def compute_pair(i, j):
        return i, j, kernel_fidelity(X[i], X[j])
    
    # We'll compute only the upper triangle and then mirror
    parallel = Parallel(n_jobs=-1, prefer="processes")
    results = parallel(delayed(compute_pair)(i, j) for i in range(n_samples) for j in range(i+1, n_samples))
    
    for i, j, val in results:
        K[i, j] = val
        K[j, i] = val
    
    return K

print("Computing quantum kernel matrix...")
K = compute_kernel_matrix(X_scaled)
print(f"Kernel matrix shape: {K.shape}")
print(f"Kernel matrix diagonal (should be 1): {np.diag(K)[:5]}")  # first 5

# Train QSVM with precomputed kernel
print("Training QSVM...")
qsvm = SVC(kernel='precomputed')
qsvm.fit(K, y)
print("QSVM training complete.")

# Save QSVM model
joblib.dump(qsvm, 'models/qsvm.joblib')
print("Saved QSVM to models/qsvm.joblib")

# Train Random Forest classifier
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)  # Note: RF uses the scaled data (same as quantum)
print("Random Forest training complete.")
joblib.dump(rf, 'models/rf.joblib')
print("Saved Random Forest to models/rf.joblib")

# Train classical SVM with RBF kernel
print("Training classical SVM (RBF)...")
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_scaled, y)
print("Classical SVM training complete.")
joblib.dump(svm_rbf, 'models/svm_classical.joblib')
print("Saved classical SVM to models/svm_classical.joblib")

print("All models saved.")
