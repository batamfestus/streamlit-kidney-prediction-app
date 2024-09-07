import streamlit as st  
import pickle as pickle
import pandas as pd  
import plotly.graph_objects as go
import numpy as np


def get_data():
    data = pd.read_csv("assets/df_model.csv")
    return data


def add_sidebar():
    st.sidebar.header("Clinical Parameters")

    data = get_data()

    slider_labels = [
        ('Serum Creatinine (mg/dL)', 'SerumCreatinine'),
        ('Glomerular Filtration Rate (mL/min/1.73 m²)', 'GFR'),
        ('Protein In Urine (mg/dL)', 'ProteinInUrine'),
        ('Blood Urea Nitrogen Levels (mg/dL)', 'BUNLevels'),
        ('Systolic Blood Pressure (mm Hg)', 'SystolicBP'),
        ('Diastolic Blood Pressure (mm Hg)', 'DiastolicBP'),
        ('Fasting Blood Sugar (mg/dL)', 'FastingBloodSugar'),
        ('HbA1c (%)', 'HbA1c'),
        ('Family History Kidney Disease (Yes/No)', 'FamilyHistoryKidneyDisease'),
        ('Previous Acute Kidney Injury', 'PreviousAcuteKidneyInjury'),
        ('Urinary Tract Infections', 'UrinaryTractInfections'),
        ('Smoking', 'Smoking'),
        ('Alcohol Consumption', 'AlcoholConsumption'),
        ('Body Mass Index (kg/m²)', 'BMI'),
        ('Total Cholesterol (mg/dL)', 'CholesterolTotal')
    ]

    input_dict = {}

    for label, key in slider_labels:
        value = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

        if value is not None:  # Safeguard to ensure no `None` values are added
            input_dict[key] = value
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_data()

    X = data.drop("Diagnosis", axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min() 
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = [
        'Serum Creatinine', 'GFR', 'Protein in Urine', 'BUN Levels',
        'Systolic BP', 'Diastolic BP', 'Fasting Blood Sugar', 'HbA1c',
        'Family History of Kidney Disease', 'Previous Acute Kidney Injury',
        'Urinary Tract Infections', 'Smoking', 'Alcohol Consumption', 'BMI', 'Cholesterol Total'
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['SerumCreatinine'], input_data['GFR'], input_data['ProteinInUrine'],
            input_data['BUNLevels'], input_data['SystolicBP'], input_data['DiastolicBP'], input_data['FastingBloodSugar'],
            input_data['HbA1c'], input_data['FamilyHistoryKidneyDisease'], input_data['PreviousAcuteKidneyInjury'], input_data['UrinaryTractInfections'],
            input_data['Smoking'], input_data['AlcoholConsumption'], input_data['BMI'], input_data['CholesterolTotal']
        ],
        theta=categories,
        fill='toself',
        name='Our Clinical Input Values (Parameter)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):

    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.write("Predictions Area")

    if prediction[0] == 0:
        st.write("<span class='diagnosis nockd'>NO CKD</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis yesckd'>YES CKD</span>", unsafe_allow_html=True)

    
    st.write(f"NO CKD: {model.predict_proba(input_array_scaled)[0][0] * 100:.2f}%")
    st.write(f"YES CKD: {model.predict_proba(input_array_scaled)[0][1] * 100:.2f}%")

    st.write("This app is meant to assist professionals in making diagnosis and should not be used for personal purpose. It is not made to substitute medical professionals.")


def main():
    st.set_page_config(
        page_title="Chronic Kidney Disease Predictor App",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    # st.write(input_data)


    with st.container():
        st.title("Chronic Kidney Disease (CKD) Predictor APP")
        st.header("This app is meant to assist medical personnel in detecting patients with kidney disease when clinical parameters are inputed (Data source 'Kaggle.com').", divider="rainbow")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

    st.header("This app is meant to assist medical professionals in making diagnosis decision and should not be used for personal purposes. It is not made to substitute medical professionals.")


if __name__ == "__main__":
    main()