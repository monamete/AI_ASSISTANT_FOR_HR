from setuptools import setup, find_packages

setup(
    name="hr_ai_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "scikit-learn",
        "xgboost",
        "faiss-cpu",
        "joblib"
    ],
)


