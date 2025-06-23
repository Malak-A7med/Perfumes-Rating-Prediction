import streamlit as st
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_resources():   
    le_brand = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\le_brand.pkl')
    le_name = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\le_name.pkl')
    le_scents = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\le_scents.pkl')
    le_department = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\le_department.pkl')
    le_concentration = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\le_concentration.pkl')
    model = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\rf_model.pkl')
    scaler = joblib.load(r'C:\Users\e\Desktop\Data Science and AI BootCamp\perfumes\workspace\scaler.pkl')
    return le_brand, le_scents, le_concentration, le_department, le_name, scaler, model
    
def main() :
        st.title("Perfumes Rating Prediction Here")
        le_brand, le_scents, le_concentration, le_department, le_name, scaler, model = load_resources()
        input_data = {}
        
        available_brands = list(le_brand.classes_)
        input_data['le_brand'] = st.selectbox("The brand of perfume", available_brands, index=10)
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    main()        
        