import streamlit as st
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_resources():   
    path = 'workspace'
    le_brand = joblib.load(os.path.join(path, 'le_brand.pkl'))
    le_name = joblib.load(os.path.join(path, 'le_name.pkl'))
    le_scents = joblib.load(os.path.join(path, 'le_scents.pkl'))
    le_department = joblib.load(os.path.join(path, 'le_department.pkl'))
    le_concentration = joblib.load(os.path.join(path, 'le_concentration.pkl'))
    model = joblib.load(os.path.join(path, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
    return le_brand, le_scents, le_concentration, le_department, le_name, scaler, model

def main():
    st.title("Perfume Rating Prediction")

    le_brand, le_scents, le_concentration, le_department, le_name, scaler, model = load_resources()

    brand = st.selectbox("Brand", le_brand.classes_)
    name = st.selectbox("Perfume Name", le_name.classes_)
    scent = st.selectbox("Scent", le_scents.classes_)
    department = st.selectbox("Department", le_department.classes_)
    concentration = st.selectbox("Concentration", le_concentration.classes_)
    price = st.number_input("Price", min_value=0.0, step=1.0)

    if st.button("Predict Rating"):
        brand_encoded = le_brand.transform([brand])[0]
        name_encoded = le_name.transform([name])[0]
        scent_encoded = le_scents.transform([scent])[0]
        department_encoded = le_department.transform([department])[0]
        concentration_encoded = le_concentration.transform([concentration])[0]

        input_features = [[brand_encoded, name_encoded, scent_encoded, department_encoded, concentration_encoded, price]]
        scaled_input = scaler.transform(input_features)

        prediction = model.predict(scaled_input)[0]
        st.success(f"Predicted Rating: {round(prediction, 2)}")

if __name__ == '__main__':
    main()
