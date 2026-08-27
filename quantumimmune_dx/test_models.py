import joblib
import numpy as np
import pandas as pd

# Load preprocessing objects
scaler = joblib.load('models/scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# Load and preprocess data (same as training)
import csv
def load_data(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        labels = []
        for row in reader:
            features = [float(x) for x in row[1:19]]  # Age to Skin_lesion
            label = row[0]  # disease
            data.append(features)
            labels.append(label)
        return np.array(data), np.array(labels)

X, y_raw = load_data('data/dataset.csv')
y = label_encoder.transform(y_raw)

# Scale features
X_scaled = scaler.transform(X)

# Load models
qsvm = joblib.load('models/qsvm.joblib')
rf = joblib.load('models/rf.joblib')
svm_rbf = joblib.load('models/svm_classical.joblib')

# Test
print("Testing models...")
print(f"QSVM accuracy: {qsvm.score(X_scaled, y):.4f}")
print(f"RF accuracy: {rf.score(X_scaled, y):.4f}")
print(f"SVM RBF accuracy: {svm_rbf.score(X_scaled, y):.4f}")

# Predictions
qsvm_pred = qsvm.predict(X_scaled[:5])
rf_pred = rf.predict(X_scaled[:5])
svm_pred = svm_rbf.predict(X_scaled[:5])

print("\nFirst 5 predictions:")
print("True labels:", label_encoder.inverse_transform(y[:5]))
print("QSVM:", label_encoder.inverse_transform(qsvm_pred))
print("RF:", label_encoder.inverse_transform(rf_pred))
print("SVM RBF:", label_encoder.inverse_transform(svm_pred))
