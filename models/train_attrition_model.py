import pandas as pd
import numpy as np
import random

np.random.seed(42)
n_samples = 2000

departments = ['HR', 'Engineering', 'Sales', 'Finance', 'Support', 'IT']
roles = ['Manager', 'Engineer', 'Sales Executive', 'HR Specialist', 'Finance Analyst', 'Support Engineer']
education_levels = ['Bachelors', 'Masters', 'PhD']

data = {
    'Age': np.random.randint(22, 60, size=n_samples),
    'Department': np.random.choice(departments, n_samples),
    'Role': np.random.choice(roles, n_samples),
    'Salary': np.random.randint(30000, 150000, size=n_samples),
    'YearsAtCompany': np.random.randint(0, 30, size=n_samples),
    'Education': np.random.choice(education_levels, n_samples),
    'LastPerformanceRating': np.random.randint(1, 5, size=n_samples)
}

df = pd.DataFrame(data)

# Target: Attrition
def generate_attrition(row):
    if (row['Age'] < 30 and row['Salary'] < 60000) or (row['LastPerformanceRating'] <= 2 and row['YearsAtCompany'] < 3):
        return 'Yes' if random.random() < 0.7 else 'No'
    else:
        return 'No' if random.random() < 0.8 else 'Yes'

df['Attrition'] = df.apply(generate_attrition, axis=1)

# Target: Performance Score
def generate_performance(row):
    base = 70 + (row['LastPerformanceRating'] * 5) - (row['YearsAtCompany'] * 0.5)
    noise = np.random.normal(0, 5)
    return np.clip(base + noise, 0, 100)

df['PerformanceScore'] = df.apply(generate_performance, axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Label Encoding
le_department = LabelEncoder()
le_role = LabelEncoder()
le_education = LabelEncoder()
le_attrition = LabelEncoder()

df['Department_enc'] = le_department.fit_transform(df['Department'])
df['Role_enc'] = le_role.fit_transform(df['Role'])
df['Education_enc'] = le_education.fit_transform(df['Education'])
df['Attrition_enc'] = le_attrition.fit_transform(df['Attrition'])

features = ['Age', 'Salary', 'YearsAtCompany', 'LastPerformanceRating', 'Department_enc', 'Role_enc', 'Education_enc']

X_attr = df[features]
y_attr = df['Attrition_enc']

X_perf = df[features]
y_perf = df['PerformanceScore']

# Scaling
scaler = StandardScaler()
X_attr_scaled = scaler.fit_transform(X_attr)
X_perf_scaled = scaler.transform(X_perf)

# Train/Test Split
X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(X_attr_scaled, y_attr, test_size=0.2, random_state=42)
X_train_perf, X_test_perf, y_train_perf, y_test_perf = train_test_split(X_perf_scaled, y_perf, test_size=0.2, random_state=42)



from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Attrition Model
attr_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
attr_model.fit(X_train_attr, y_train_attr)
y_pred_attr = attr_model.predict(X_test_attr)

# Performance Model
perf_model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
perf_model.fit(X_train_perf, y_train_perf)
y_pred_perf = perf_model.predict(X_test_perf)

# Attrition Model
print("\nAttrition Model Classification Report:")
print(classification_report(y_test_attr, y_pred_attr))
print("Confusion Matrix:\n", confusion_matrix(y_test_attr, y_pred_attr))

# Performance Model
print("\nPerformance Model Metrics:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_perf, y_pred_perf):.2f}")
print(f"R² Score: {r2_score(y_test_perf, y_pred_perf):.2f}")


import joblib

# Save XGBoost Models
joblib.dump(attr_model, 'models/attrition_model_xgb.joblib')
joblib.dump(perf_model, 'models/performance_model_xgb.joblib')

# Save Scaler
joblib.dump(scaler, 'models/scaler.joblib')

# Save Encoders
joblib.dump(le_department, 'models/le_department.joblib')
joblib.dump(le_role, 'models/le_role.joblib')
joblib.dump(le_education, 'models/le_education.joblib')

print("✅ Models and preprocessing objects saved successfully!")






