import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  

# Load the saved Keras model
model = load_model('iris_model.h5')

# Title of the app
st.title("Iris Flower Prediction App")

# Create inputs for the 4 features of the Iris dataset
st.header("Input Features")
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)

# When the user clicks the 'Predict' button, do the prediction
if st.button("Predict"):
    # Create the input array with the features
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the class
    prediction = model.predict(new_data)
    predicted_class = np.argmax(prediction, axis=1)

    # Class labels
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    # Show the prediction result
    st.success(f"The predicted class is: {classes[predicted_class[0]]}")


