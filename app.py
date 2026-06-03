
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

BASE_PATH = "/content/drive/MyDrive/ANN_CLASSIFICATION/"

# Load Model
model = tf.keras.models.load_model(BASE_PATH + 'model.h5')

# Load Encoders and Scaler
with open(BASE_PATH + 'label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(BASE_PATH + 'onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(BASE_PATH + 'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
# Streamlit UI
st.title("Customer Churn Prediction")

# Inputs
credit_score = st.number_input('Credit Score')

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 100)

tenure = st.slider('Tenure', 0, 10)

balance = st.number_input('Balance')

num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0,1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0,1]
)

estimated_salary = st.number_input('Estimated Salary')

# Create input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One Hot Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine dataframes
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write(f'Prediction Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error("Customer is likely to churn")
else:
    st.success("Customer is likely to stay")
