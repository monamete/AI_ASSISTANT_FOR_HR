from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import joblib
import os

# Example data (same categories you expect at prediction time)
job_roles = ["Sales Executive", "Laboratory Technician", "Research Scientist", "HR Manager", "Manager"]
educations = ["Below College", "College", "Bachelor", "Master"]
departments = ["Sales", "Research & Development", "HR", "Marketing"]

# Create training-like dataset
training_categorical_data = np.array([
    [job_roles[0], educations[2], departments[0]],
    [job_roles[1], educations[1], departments[1]],
    [job_roles[2], educations[3], departments[2]],
    [job_roles[3], educations[0], departments[3]],
    [job_roles[4], educations[2], departments[1]],
])

# Create and fit OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(training_categorical_data)

# Save the encoder
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/ordinal_encoder.pkl")

print("✅ Ordinal Encoder trained and saved successfully!")
