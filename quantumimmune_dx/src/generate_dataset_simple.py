"""
Simple synthetic dataset generation for QuantumImmune Dx
Uses only built-in Python modules for initial implementation
"""

import csv
import random
import json
import os
from typing import Dict, List, Tuple

# Define disease configurations with realistic biomarker ranges
DISEASE_CONFIGS = {
    "Healthy": {
        "Age": (20, 80),
        "Sex": [0, 1],  # 0=Female, 1=Male
        "CRP": (0, 5),          # mg/L
        "ESR": (0, 20),         # mm/hr
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0],       # Negative
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # mg/dL
        "TSH": (0.4, 4.0),      # mIU/L
        "Anti_TPO": [0],        # Negative
        "Fasting_Glucose": (70, 99), # mg/dL
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 2),   # 0-10 scale
        "Fatigue": (0, 3),      # 0-10 scale
        "GI_symptom": (0, 2),   # 0-10 scale
        "Skin_lesion": [0],     # No
    },
    "Rheumatoid_Arthritis": {
        "Age": (30, 70),
        "Sex": [0, 1],  # Slightly more female
        "CRP": (5, 50),         # Elevated
        "ESR": (20, 60),        # Elevated
        "RF": [0, 1],           # 70-80% positive
        "Anti_CCP": [0, 1],     # 60-70% positive
        "ANA_titer": [0, 1, 2], # Sometimes positive
        "Anti_dsDNA": [0],      # Usually negative
        "Complement_C3": (70, 120), # Normal/slightly low
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (4, 9),   # Significant joint pain
        "Fatigue": (4, 8),      # Significant fatigue
        "GI_symptom": (0, 3),   # Mild GI symptoms
        "Skin_lesion": [0],     # No skin lesions
    },
    "Systemic_Lupus_Erythematosus": {
        "Age": (15, 50),
        "Sex": [0, 1],  # Predominantly female
        "CRP": (0, 10),         # Usually normal/mildly elevated
        "ESR": (20, 80),        # Often elevated
        "RF": [0],              # Usually negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [1, 2, 3, 4], # Almost always positive
        "Anti_dsDNA": [0, 1],   # 70% positive
        "Complement_C3": (10, 90), # Often low
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0, 1],     # Sometimes positive
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (3, 8),   # Joint pain common
        "Fatigue": (5, 9),      # Severe fatigue
        "GI_symptom": (0, 4),   # GI involvement possible
        "Skin_lesion": [0, 1],  # Skin lesions common
    },
    "Type_1_Diabetes": {
        "Age": (0, 30),         # Usually younger onset
        "Sex": [0, 1],
        "CRP": (0, 5),          # Normal unless complications
        "ESR": (0, 20),         # Normal
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0, 1],    # Sometimes positive
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0, 1],     # Sometimes positive (associated autoimmunity)
        "Fasting_Glucose": (126, 250), # Elevated (diagnostic threshold)
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 3),   # Usually absent
        "Fatigue": (3, 7),      # Common symptom
        "GI_symptom": (0, 3),   # Variable
        "Skin_lesion": [0],     # No specific skin lesions
    },
    "Hashimoto_Thyroiditis": {
        "Age": (30, 60),
        "Sex": [0, 1],  # Predominantly female
        "CRP": (0, 5),          # Normal
        "ESR": (0, 30),         # Normal/mildly elevated
        "RF": [0],              # Usually negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0, 1],    # Sometimes positive
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (4.0, 20.0),     # Elevated (hypothyroid)
        "Anti_TPO": [0, 1],     # >90% positive
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 3),   # Usually absent
        "Fatigue": (3, 8),      # Common symptom
        "GI_symptom": (0, 3),   # Variable
        "Skin_lesion": [0],     # No specific skin lesions
    },
    "Graves_Disease": {
        "Age": (20, 50),
        "Sex": [0, 1],  # Predominantly female
        "CRP": (0, 5),          # Normal
        "ESR": (0, 20),         # Normal
        "RF": [0],              # Usually negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0, 1],    # Sometimes positive
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.01, 0.4),     # Suppressed (hyperthyroid)
        "Anti_TPO": [0, 1],     # Often positive
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 3),   # Usually absent
        "Fatigue": (2, 6),      # Variable
        "GI_symptom": (0, 3),   # Variable
        "Skin_lesion": [0],     # No specific skin lesions (though pretibial myxedema possible)
    }
}

def generate_synthetic_data(n_samples_per_class: int = 30, random_state: int = 42) -> List[Dict]:
    """
    Generate synthetic dataset for autoimmune disease classification
    
    Args:
        n_samples_per_class: Number of samples to generate per disease class
        random_state: Random seed for reproducibility
        
    Returns:
        List of dictionaries with synthetic patient data
    """
    random.seed(random_state)
    
    data_rows = []
    
    for disease_name, config in DISEASE_CONFIGS.items():
        for _ in range(n_samples_per_class):
            row = {"disease": disease_name}
            
            for feature, range_or_values in config.items():
                if isinstance(range_or_values, list):
                    # Categorical/binary feature
                    if len(range_or_values) == 2 and all(isinstance(x, (int, float)) for x in range_or_values):
                        # Binary feature with probability (e.g., [0, 1] means 50% chance of 1)
                        if sum(range_or_values) == 1 and range_or_values[0] == 0:
                            # Special case: [0, 1] means Bernoulli with p=0.5
                            row[feature] = random.choice([0, 1])
                        else:
                            # General case: choose from list with equal probability
                            row[feature] = random.choice(range_or_values)
                    else:
                        # Choose from list with equal probability
                        row[feature] = random.choice(range_or_values)
                else:
                    # Continuous feature with range (min, max)
                    low, high = range_or_values
                    row[feature] = random.uniform(low, high)
            
            data_rows.append(row)
    
    return data_rows

def save_dataset_csv(data: List[Dict], filepath: str = "data/dataset.csv"):
    """Save dataset to CSV file"""
    if not data:
        return
    
    # Get fieldnames from first row
    fieldnames = list(data[0].keys())
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Dataset saved to {filepath}")
    print(f"Total samples: {len(data)}")

def save_dataset_info(data: List[Dict], filepath: str = "data/dataset_info.json"):
    """Save dataset information and statistics"""
    if not data:
        return
    
    # Calculate statistics
    total_samples = len(data)
    disease_counts = {}
    feature_names = [key for key in data[0].keys() if key != 'disease']
    
    for row in data:
        disease = row['disease']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    info = {
        "total_samples": total_samples,
        "n_features": len(feature_names),
        "disease_distribution": disease_counts,
        "feature_names": feature_names,
        "disease_list": sorted(disease_counts.keys())
    }
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to {filepath}")
    return info

def print_dataset_summary(data: List[Dict]):
    """Print summary of generated dataset"""
    if not data:
        print("No data to summarize")
        return
    
    print("\nDataset Summary:")
    print(f"Total samples: {len(data)}")
    print(f"Number of features: {len(data[0].keys()) - 1}")  # Excluding disease
    
    # Disease distribution
    disease_counts = {}
    for row in data:
        disease = row['disease']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    print("\nDisease distribution:")
    for disease, count in sorted(disease_counts.items()):
        print(f"  {disease}: {count}")
    
    print("\nFirst few rows:")
    for i in range(min(3, len(data))):
        print(f"  {data[i]}")

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic autoimmune disease dataset...")
    dataset = generate_synthetic_data(n_samples_per_class=30)
    
    # Save dataset
    save_dataset_csv(dataset)
    
    # Save dataset info
    info = save_dataset_info(dataset)
    
    # Print summary
    print_dataset_summary(dataset)
