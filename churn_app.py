from re import A
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image



#image
img=Image.open("employeechurn.jpg")
#st.image(img, caption="catie", width=800)
st.image(img)
html_temp = """
<div style="background-color:white;padding:1 px">
<a style="color:black;text-align:left;">Source: https://www.aihr.com/wp-content/uploads/High-employee-turnover.jpg</a>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)


html_temp = """
<div style="background-color:purple;padding:1.5px">
<h1 style="color:black;text-align:center;">Predict your employees churn before any leave occurs!</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

html_temp1 = """
<div style="background-color:cyan;">
<text style="color:purple;text-align:center;font-size:140%;">All you need to give your employee's general information and get the employee churn prediction rate!</text>
</div><br>"""
st.markdown(html_temp1,unsafe_allow_html=True)

html_temp = """
<div style="background-color:green;">
<p style="color:white;text-align:center;font-size:160%;">Your Employee's Information</p>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

satisfaction_level=st.sidebar.slider("Select employee's satisfaction level", 1, 100, step=1)
last_evaluation=st.sidebar.slider("Select last evaluation level", 1, 100, step=1)
number_project=st.sidebar.select_slider("Select number of projects", [2,3,4,5,6,7])
average_montly_hours=st.sidebar.slider("Select monthly work hours", 96, 310, step=1)
time_spend_company=st.sidebar.slider("Select years of experiences", 2, 10, step=1)
work_accident=st.sidebar.select_slider("Does work accident occur?", [0,1])
promotion_last_5years=st.sidebar.select_slider("Has been promoted in last 5 years?", [0,1])
departments=st.sidebar.selectbox("Select employee's department", ['sales','accounting','hr','IT','management','marketing','product_mng','RandD','support','technical'])
salary=st.sidebar.selectbox("Select employee's department", ['low','medium','high'])
def your_car():
    my_dict = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_montly_hours": average_montly_hours,
        "time_spend_company":time_spend_company }
    df_sample = pd.DataFrame.from_dict([my_dict])
    return df_sample
df = your_car()
st.table(df)


def your_car():
    my_dict = {
        "time_spend_company":time_spend_company ,
        "work_accident": work_accident,
        "promotion_last_5years": promotion_last_5years ,
        "departments": departments,
        "salary": salary}
    df_sample = pd.DataFrame.from_dict([my_dict])
    return df_sample
df = your_car()
st.table(df)

#using trained models
import pickle

model_selection=st.selectbox('Select a Machine Learning Algorithm', ['Gradient Boosting Algorithm','K-Neighborhood Algorithm','Random Forest Algorithm','XGBoost Classifier Algorithm'])
if model_selection=='Gradient Boosting Algorithm':
    final_model='GBModel.pkl'
if model_selection=='K-Neighborhood Algorithm':
    final_model='KNNModel.pkl'
if model_selection=='Random Forest Algorithm':
    final_model='RFModel.pkl'
if model_selection=='XGBoost Classifier Algorithm':
    final_model='XGBModel.pkl'


final_scaler=pickle.load(open("scaler.pkl","rb"))
final_model=pickle.load(open(final_model,"rb"))
my_dict = {
    "satisfaction_level": satisfaction_level/100,
    "last_evaluation": last_evaluation/100,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company":time_spend_company ,
    "work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years ,
    "departments": departments,
    "salary": salary}


my_dict = pd.DataFrame([my_dict])

scale_mapper = {"low":1, "medium":2, "high":3}
my_dict["salary"] = my_dict["salary"].replace(scale_mapper)

my_dict = pd.get_dummies(my_dict)
Xcolumns=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'work_accident',
       'promotion_last_5years', 'salary', 'departments_IT',
       'departments_RandD', 'departments_accounting', 'departments_hr',
       'departments_management', 'departments_marketing',
       'departments_product_mng', 'departments_sales', 'departments_support',
       'departments_technical']

my_dict = my_dict.reindex(columns = Xcolumns, fill_value=0)

my_dict = final_scaler.transform(my_dict)


if st.button("To get your car's price, press this button"): 
    price=int(final_model.predict(my_dict)[0])
    if price==0:
        st.markdown(f"""
        #### <span style="background-color:yellow;color:red;font-size:32px;border-radius:2%;text-align:center"> You have a loyal employee! </span>
        # # """,unsafe_allow_html=True)
    if price==1:
        st.markdown(f"""
        #### <span style="background-color:yellow;color:red;font-size:32px;border-radius:2%;text-align:center"> Employee will leave you! Find new one!</span>
        # # """,unsafe_allow_html=True)
