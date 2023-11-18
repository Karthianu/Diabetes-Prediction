# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:12:10 2023

@author: Vignesan
"""

import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu



#st.set_page_config(page_title="Diabetes Prediction")


with st.sidebar:
    selected=option_menu("Main Menu", 
                         options=["Home","SignUp","DiabetesPrediction"])

                    

if selected == "Home":
    st.title(f"Haiiiiii")
    
    
    #st.title("DIABETES PREDICTION SYSTEM USING MACHINE LEARNING")
    st.markdown("<h1 style='text-align: center; color: black;'>DIABETES PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

    img=Image.open("C:/Users/Vignesan/Documents/ML Projects/Diabetes Prediction/1.jpg",)
    new_image = img.resize((800, 250))
    st.image(new_image)

    #buff, col, buff2 = st.beta_columns([1,3,1])

    uname1=st.text_input('User Name:')
    pswd=st.text_input("Password")

    #col1, col2 = st.columns([0.04, 0.3],gap="small")

    #with col1:
    st.button("SignIn")
    #with col2:
       # st.button("SignUp")
        
        

       # if uname=="" or pswd=="":
            
    

    