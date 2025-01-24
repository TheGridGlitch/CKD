
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows for the dataset
num_rows = 5000

# Define ranges and categories for synthetic features
age = np.random.randint(18, 90, num_rows)
blood_pressure = np.random.randint(60, 180, num_rows)
specific_gravity = np.round(np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], num_rows), 3)
albumin = np.random.randint(0, 5, num_rows)
sugar = np.random.randint(0, 5, num_rows)
red_blood_cells = np.random.choice(['normal', 'abnormal'], num_rows, p=[0.8, 0.2])
pus_cell = np.random.choice(['normal', 'abnormal'], num_rows, p=[0.8, 0.2])
pus_cell_clumps = np.random.choice(['present', 'notpresent'], num_rows, p=[0.2, 0.8])
bacteria = np.random.choice(['present', 'notpresent'], num_rows, p=[0.2, 0.8])
blood_glucose_random = np.random.randint(70, 490, num_rows)
blood_urea = np.random.uniform(10, 120, num_rows).round(1)
serum_creatinine = np.random.uniform(0.5, 15, num_rows).round(1)
sodium = np.random.uniform(120, 150, num_rows).round(1)
potassium = np.random.uniform(3.5, 6.5, num_rows).round(1)
hemoglobin = np.random.uniform(7, 17, num_rows).round(1)
packed_cell_volume = np.random.randint(20, 55, num_rows)
white_blood_cell_count = np.random.randint(4000, 15000, num_rows)
red_blood_cell_count = np.random.uniform(3.0, 6.0, num_rows).round(1)
hypertension = np.random.choice(['yes', 'no'], num_rows, p=[0.4, 0.6])
diabetes_mellitus = np.random.choice(['yes', 'no'], num_rows, p=[0.3, 0.7])
coronary_artery_disease = np.random.choice(['yes', 'no'], num_rows, p=[0.1, 0.9])
appetite = np.random.choice(['good', 'poor'], num_rows, p=[0.85, 0.15])
pedal_edema = np.random.choice(['yes', 'no'], num_rows, p=[0.2, 0.8])
anemia = np.random.choice(['yes', 'no'], num_rows, p=[0.3, 0.7])
classification = np.random.choice(['ckd', 'notckd'], num_rows, p=[0.4, 0.6])

# Combine all the features into a DataFrame
columns = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
    "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
    "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium", "potassium",
    "hemoglobin", "packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count",
    "hypertension", "diabetes_mellitus", "coronary_artery_disease", "appetite",
    "pedal_edema", "anemia", "classification"
]

data = pd.DataFrame({
    "age": age,
    "blood_pressure": blood_pressure,
    "specific_gravity": specific_gravity,
    "albumin": albumin,
    "sugar": sugar,
    "red_blood_cells": red_blood_cells,
    "pus_cell": pus_cell,
    "pus_cell_clumps": pus_cell_clumps,
    "bacteria": bacteria,
    "blood_glucose_random": blood_glucose_random,
    "blood_urea": blood_urea,
    "serum_creatinine": serum_creatinine,
    "sodium": sodium,
    "potassium": potassium,
    "hemoglobin": hemoglobin,
    "packed_cell_volume": packed_cell_volume,
    "white_blood_cell_count": white_blood_cell_count,
    "red_blood_cell_count": red_blood_cell_count,
    "hypertension": hypertension,
    "diabetes_mellitus": diabetes_mellitus,
    "coronary_artery_disease": coronary_artery_disease,
    "appetite": appetite,
    "pedal_edema": pedal_edema,
    "anemia": anemia,
    "classification": classification
})

# Save the dataset to a CSV file
output_file = "synthetic_kidney_disease_data.csv"
data.to_csv(output_file, index=False)

print(f"Dataset created successfully with {num_rows} rows and saved to '{output_file}'.")

