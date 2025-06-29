
import numpy as np
import pickle
import streamlit as st

# Load the saved model
with open("trained_model1.sav", "rb") as f:
    loaded_model = pickle.load(f)

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction Web App')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of Person')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numerical values only.'

    st.success(diagnosis)

if __name__ == '__main__':
    main()


