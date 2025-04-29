
import streamlit as st
import numpy as np
import joblib
from search_faiss import search_document
import faiss
import pickle


index = faiss.read_index("faiss_index.idx")
with open("all_texts.pkl", "rb") as f:
    all_texts = pickle.load(f)


# Load trained models
attr_model = joblib.load('models/attrition_model_xgb.joblib')
perf_model = joblib.load('models/performance_model_xgb.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load label encoders
le_department = joblib.load('models/le_department.joblib')
le_role = joblib.load('models/le_role.joblib')
le_education = joblib.load('models/le_education.joblib')

# Streamlit UI
st.title("🤖 HR Analytics AI Assistant")

# Sidebar Menu
menu = ["Predict Attrition", "Predict Performance Score", "HR Policy Q&A"]
choice = st.sidebar.selectbox("Choose Action", menu)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit, XGBoost, FAISS, and Scikit-Learn")

#  ATTRITION PREDICTION 
if choice == "Predict Attrition":
    st.subheader("🔎 Attrition Prediction")

    # Input form
    with st.form(key='attrition_form'):
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        salary = st.number_input("Salary", min_value=20000, max_value=200000, value=60000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        last_rating = st.selectbox("Last Performance Rating", [1, 2, 3, 4])

        department = st.selectbox("Department", le_department.classes_)
        role = st.selectbox("Role", le_role.classes_)
        education = st.selectbox("Education", le_education.classes_)

        submit_button = st.form_submit_button(label="Predict Attrition")

    if submit_button:
    
        dept_encoded = le_department.transform([department])[0]
        role_encoded = le_role.transform([role])[0]
        edu_encoded = le_education.transform([education])[0]

   
        input_data = np.array([[age, salary, years_at_company, last_rating, dept_encoded, role_encoded, edu_encoded]])
        input_scaled = scaler.transform(input_data)

      
        pred = attr_model.predict(input_scaled)[0]
        pred_proba = attr_model.predict_proba(input_scaled)[0]

        if pred == 1:
            st.error(f"⚠️ High Attrition Risk! (Confidence: {pred_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ Low Attrition Risk (Confidence: {pred_proba[0]*100:.2f}%)")

# PERFORMANCE PREDICTION
elif choice == "Predict Performance Score":
    st.subheader("📈 Performance Prediction")

    # Input form
    with st.form(key='performance_form'):
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        salary = st.number_input("Salary", min_value=20000, max_value=200000, value=60000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        last_rating = st.selectbox("Last Performance Rating", [1, 2, 3, 4])

        department = st.selectbox("Department", le_department.classes_)
        role = st.selectbox("Role", le_role.classes_)
        education = st.selectbox("Education", le_education.classes_)

        submit_button = st.form_submit_button(label="Predict Performance")

    if submit_button:
   
        dept_encoded = le_department.transform([department])[0]
        role_encoded = le_role.transform([role])[0]
        edu_encoded = le_education.transform([education])[0]

        input_data = np.array([[age, salary, years_at_company, last_rating, dept_encoded, role_encoded, edu_encoded]])
        input_scaled = scaler.transform(input_data)

      
        pred = perf_model.predict(input_scaled)[0]

        st.success(f"🏆 Predicted Performance Score: {pred:.2f} / 100")

#HR POLICY Q&A 
elif choice == "HR Policy Q&A":
    st.subheader("📚 HR Policy Q&A")

    user_query = st.text_input("Ask your HR question:")

    if user_query:
        results = search_document(user_query,index, all_texts)

        if results:
            st.success("Top Matched Policies:")
            for res in results:
                st.write(f"- {res}")
        else:
            st.warning("❓ No relevant policy found. Please try rephrasing your question.")
