"""
Synthetic dataset generation for QuantumImmune Dx
Generates realistic lab/clinical data for 12 autoimmune diseases + healthy class
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json

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
    },
    "Multiple_Sclerosis": {
        "Age": (20, 50),
        "Sex": [0, 1],  # More common in females
        "CRP": (0, 5),          # Normal
        "ESR": (0, 20),         # Normal
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0],       # Usually negative
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 3),   # Usually absent
        "Fatigue": (6, 9),      # Severe fatigue (very common)
        "GI_symptom": (0, 3),   # Variable
        "Skin_lesion": [0],     # No specific skin lesions
        # Note: Would normally have oligoclonal bands, MRI findings, etc.
    },
    "Psoriatic_Arthritis": {
        "Age": (30, 60),
        "Sex": [0, 1],
        "CRP": (2, 30),         # Often elevated
        "ESR": (10, 40),        # Often elevated
        "RF": [0],              # Usually negative (seronegative)
        "Anti_CCP": [0],        # Usually negative
        "ANA_titer": [0, 1],    # Sometimes positive
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0, 1],      # ~20% positive
        "Joint_pain": (4, 9),   # Significant joint pain
        "Fatigue": (3, 7),      # Moderate fatigue
        "GI_symptom": (0, 3),   # Variable
        "Skin_lesion": [0, 1],  # Psoriasis skin lesions
    },
    "Celiac_Disease": {
        "Age": (0, 80),         # Can occur at any age
        "Sex": [0, 1],
        "CRP": (0, 10),         # Normal/mildly elevated
        "ESR": (0, 30),         # Normal/mildly elevated
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0],       # Usually negative
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0, 1],     # >95% positive (key diagnostic)
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 4),   # Possible arthralgia
        "Fatigue": (2, 6),      # Common symptom
        "GI_symptom": (4, 9),   # Significant GI symptoms
        "Skin_lesion": [0, 1],  # Dermatitis herpetiformis possible
    },
    "Inflammatory_Bowel_Disease": {
        "Age": (15, 40),        # Often younger onset
        "Sex": [0, 1],
        "CRP": (2, 50),         # Often elevated during flares
        "ESR": (10, 60),        # Often elevated
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0],       # Usually negative
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 5),   # Possible arthralgia/arthritis
        "Fatigue": (3, 8),      # Common symptom
        "GI_symptom": (5, 9),   # Significant GI symptoms
        "Skin_lesion": [0, 1],  # Skin manifestations possible
    },
    "Sjogrens_Syndrome": {
        "Age": (40, 60),
        "Sex": [0, 1],  # Overwhelmingly female
        "CRP": (0, 10),         # Normal/mildly elevated
        "ESR": (0, 40),         # Normal/mildly elevated
        "RF": [0, 1],           # Often positive
        "Anti_CCP": [0],        # Usually negative
        "ANA_titer": [0, 1, 2], # Often positive
        "Anti_dsDNA": [0],      # Usually negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 5),   # Possible arthralgia
        "Fatigue": (5, 9),      # Severe fatigue
        "GI_symptom": (0, 4),   # Variable GI symptoms
        "Skin_lesion": [0],     # No specific skin lesions (though dryness important)
        # Note: Would also have dry eyes/mouth symptoms
    },
    "Ankylosing_Spondylitis": {
        "Age": (17, 45),        # Usually young adult onset
        "Sex": [0, 1],  # More common in males
        "CRP": (0, 30),         # Often elevated
        "ESR": (0, 50),         # Often elevated
        "RF": [0],              # Negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0],       # Usually negative
        "Anti_dsDNA": [0],      # Negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0, 1],      # >90% positive (strong association)
        "Joint_pain": (3, 8),   # Back pain/joint pain
        "Fatigue": (2, 6),      # Variable fatigue
        "GI_symptom": (0, 4),   # Possible GI involvement (IBD association)
        "Skin_lesion": [0],     # No specific skin lesions (though psoriasis possible)
    },
    "Autoimmune_Hepatitis": {
        "Age": (15, 70),
        "Sex": [0, 1],  # More common in females
        "CRP": (0, 10),         # Normal/mildly elevated
        "ESR": (0, 40),         # Normal/mildly elevated
        "RF": [0],              # Usually negative
        "Anti_CCP": [0],        # Negative
        "ANA_titer": [0, 1, 2], # Often positive
        "Anti_dsDNA": [0],      # Usually negative
        "Complement_C3": (90, 180), # Normal
        "TSH": (0.4, 4.0),      # Normal
        "Anti_TPO": [0],        # Usually negative
        "Fasting_Glucose": (70, 110), # Normal
        "Anti_tTG": [0],        # Negative
        "HLA_B27": [0],         # Negative
        "Joint_pain": (0, 4),   # Possible arthralgia
        "Fatigue": (3, 8),      # Common symptom
        "GI_symptom": (1, 6),   # Variable GI symptoms
        "Skin_lesion": [0, 1],  # Skin manifestations possible
        # Note: Would have elevated liver enzymes (ALT/AST), IgG
    }
}

def generate_synthetic_data(n_samples_per_class: int = 50, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset for autoimmune disease classification
    
    Args:
        n_samples_per_class: Number of samples to generate per disease class
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic patient data
    """
    np.random.seed(random_state)
    
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
                            row[feature] = np.random.choice([0, 1], p=[0.5, 0.5])
                        else:
                            # General case: choose from list with equal probability
                            row[feature] = np.random.choice(range_or_values)
                    else:
                        # Choose from list with equal probability
                        row[feature] = np.random.choice(range_or_values)
                else:
                    # Continuous feature with range (min, max)
                    low, high = range_or_values
                    row[feature] = np.random.uniform(low, high)
            
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Reorder columns to put disease last
    cols = [col for col in df.columns if col != 'disease'] + ['disease']
    df = df[cols]
    
    return df

def save_dataset_info(df: pd.DataFrame, filepath: str = "data/dataset_info.json"):
    """Save dataset information and statistics"""
    info = {
        "total_samples": len(df),
        "n_features": len(df.columns) - 1,  # Excluding disease column
        "disease_distribution": df['disease'].value_counts().to_dict(),
        "feature_names": [col for col in df.columns if col != 'disease'],
        "disease_list": sorted(df['disease'].unique().tolist())
    }
    
    # Save to JSON
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    
    return info

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic autoimmune disease dataset...")
    dataset = generate_synthetic_data(n_samples_per_class=50)
    
    # Save dataset
    import os
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
    print(f"Shape: {dataset.shape}")
    
    # Save dataset info
    info = save_dataset_info(dataset)
    print(f"Dataset info saved to data/dataset_info.json")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of features: {len(dataset.columns) - 1}")
    print("\nDisease distribution:")
    for disease, count in dataset['disease'].value_counts().items():
        print(f"  {disease}: {count}")
    
    print("\nFirst few rows:")
    print(dataset.head())
