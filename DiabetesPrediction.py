
import numpy as np
import pickle
import streamlit as st



loaded_model=pickle.load(open('C:/Users/mayugam info/Machine Learning/Diabetes Prediction/train_model.sav','rb'))

#def app():

def diabetes_prediction(ip_data):
    
    ip_arr=np.array(ip_data)
    res_arr=ip_arr.reshape(1,-1)

    pred=loaded_model.predict(res_arr)
    print(pred)
    if (pred[0]==0):
        return("The Person is Not Diabetic")
    else:
        return("The Person is Diabetic")
    
    
def main():
    #giving title
    st.title('DIABETES PREDICTION')
    
    #Getting the input data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies=st.text_input("Number of Pregnancies")
    Glucose=st.text_input("Glucose Level")
    BloodPressure=st.text_input("Blood Pressure Value")
    SkinThickness=st.text_input("Skin Thickness Value")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI Value")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function Value")
    Age=st.text_input("Age")
    
    
    #Code for Prediction
    
    diagnosis=''
    
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()

    
    
