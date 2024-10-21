import streamlit as st
import numpy as np
import joblib
st.title('Iris prediction')
st.subheader("Personal details")

sepal_length = st.number_input("Enter the sepal length:", min_value=0.0, max_value=8.0,step=0.1)
sepal_width = st.number_input("Enter the sepal width:", min_value=0.0,max_value=5.0, step=0.1)
petal_length = st.number_input("Enter the petal length:", min_value=0.0,max_value=7.0, step=0.1)
petal_width = st.number_input("Enter the petal width:", min_value=0.0,max_value=3.0, step=0.1)
association_dict = {"0":'Iris-setosa' ,"1" : 'Iris-versicolor', "2":'Iris-virginica'}
input = np.array([sepal_length,sepal_width,petal_length,petal_width])
prediction = ""
model = joblib.load("model.joblib")
if st.button("Predict"):
    prediction = model.predict([input])
    print(prediction)
    print(association_dict[str(prediction[0])])
    if prediction!="" : 
        st.success(f"Your prediction is : {association_dict[str(prediction[0])]}") 
