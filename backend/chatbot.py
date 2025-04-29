import joblib
import numpy as np

# Load trained models and encoders
attrition_model = joblib.load("models/attrition_model.pkl")
performance_model = joblib.load("models/performance_model.pkl")
ordinal_encoder = joblib.load("models/ordinal_encoder.pkl")

user_context = {}

def handle_features_input(user_message):
    """Handles inputting features when requested."""
    if 'features' in user_context:
        return "Features already collected. Proceeding to prediction."

    try:
        features = [x.strip() for x in user_message.split(",")]
        user_context['features'] = features
        return "Features collected successfully. Now you can predict."
    except Exception as e:
        return f"Error parsing features: {e}"

def preprocess_features(features):
    """Split numerical and categorical, encode, then combine."""
    # Assuming features order: [age, salary, years_at_company, job_role, education, department]
    numerical_features = np.array(features[:3], dtype=float)  # first three: age, salary, years
    categorical_features = np.array(features[3:]).reshape(1, -1)  # last three: job role, education, dept

    # Encode categorical
    encoded_categorical = ordinal_encoder.transform(categorical_features)

    # Combine numerical + encoded categorical
    final_features = np.hstack((numerical_features, encoded_categorical.flatten()))
    return final_features.reshape(1, -1)  # Make it 2D for model input

def predict_attrition(features):
    try:
        final_features = preprocess_features(features)
        prediction = attrition_model.predict(final_features)
        return prediction[0]
    except Exception as e:
        return f"Error predicting attrition: {e}"
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load the OrdinalEncoder
ordinal_encoder = joblib.load("models/ordinal_encoder.pkl")

def predict_performance(features):
    try:
        # Transform the features using OrdinalEncoder, ensuring that unknown categories are handled
        features_encoded = ordinal_encoder.transform([features[3:]])  # Features starting from JobRole, Education, Department
        # Insert the encoded categorical features back into the original feature set
        features_encoded = features[:3] + list(features_encoded[0])
        
        # Now use the model to predict performance (as per your setup)
        model = joblib.load("models/performance_model.pkl")
        performance_prediction = model.predict([features_encoded])[0]

        return performance_prediction

    except Exception as e:
        return f"Error predicting performance: {str(e)}"


def chat_response(user_message):
    if user_message.lower() == "predict attrition":
        features = user_context.get('features')
        if features:
            return predict_attrition(features)
        else:
            return "Please enter employee features first."
    
    elif user_message.lower() == "predict performance":
        features = user_context.get('features')
        if features:
            return predict_performance(features)
        else:
            return "Please enter employee features first."
    
    else:
        return "I'm here to assist you with attrition and performance prediction!"
