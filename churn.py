#try:
    #from numpy.core.numeric import ComplexWarning
#except ImportError:
    #class ComplexWarning(Warning):
        #pass

import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#from numpy.core.numeric import ComplexWarning
#try:
    #from numpy.core.numeric import ComplexWarning
#except ImportError:
    #class ComplexWarning(Warning):
       # pass


scaler = pickle.load(open('scal.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))

st.title('*Churn Prediction Application*')
st.write("*This application predicts whether a customer will churn or not*")

st.header("Enter the customer details here")

def cust_details():
    
    c1,c2 = st.columns(2)
    Surname = st.text_input("What is your name?")
    with c1:
        CreditScore = st.number_input("Input your credit score",350,1000,430)
        Geography = st.selectbox("Select the country you are from",(['Spain','France','Germany']))
        Gender = st.selectbox("Select your gender below",(['Male','Female']))
        Age = st.number_input("Please enter your age",18,100,91)
        Tenure = st.number_input("Please enter how long you have been a customer",0,12,8)
    with c2:
        Balance = st.number_input("Enter your balance",0.0, 300000.0,462.3)
        NumOfProducts = st.number_input("Enter the number of products purchased",1,4,2)
        HasCrCard = st.selectbox("Do you own a credit card?",(['Yes','No']))
        IsActiveMember = st.selectbox("Are you an active member?",(['Yes','No']))
        EstimatedSalary = st.number_input("Enter your salary below",0.0,200000.0,1300.5)
        
        
    feat = np.array([CreditScore,
                     Geography,
                     Gender,
                     Age,
                     Tenure,
                     Balance,
                     NumOfProducts,
                     HasCrCard,
                     IsActiveMember,
                     EstimatedSalary]).reshape(1,-1)
    cols = ['CreditScore',
            'Geography',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary']
    
    df = pd.DataFrame(feat, columns = cols)
    return df
df = cust_details()

st.write(df)
            

df.replace({'Yes':1,
            'No':0},inplace = True)
#st.write(df)


#feature engineering
df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce')
df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce')
df['CreditScore'] = pd.to_numeric(df['CreditScore'], errors = 'coerce')
df['NumOfProducts'] = pd.to_numeric(df['NumOfProducts'], errors = 'coerce')
            
        
df['salary_per_credit'] = df.EstimatedSalary/df.CreditScore
df['salary_per_age'] = df.EstimatedSalary/df.Age
df['salary_per_products'] = df.EstimatedSalary/df.NumOfProducts
df['credit_per_age'] = df.CreditScore/df.Age
df['products_per_age'] = df.NumOfProducts/df.Age
df['credit_per_product']= df.CreditScore/df.NumOfProducts
df['Balance_per_age'] = df.Balance/df.Age
df['balance_and_age'] = df.Balance*df.Age
df['salary_and_age'] = df.EstimatedSalary*df.Age
df['credit_and_age'] = df.CreditScore*df.Age
df['salary_and_products'] = df.EstimatedSalary*df.NumOfProducts
#st.write(df)

def preprocessing():
    df1 = df.copy()
    cat_cols = ['Geography','Gender']
    encoded_data = encoder.transform(df1[cat_cols])
    dense_data = encoded_data.todense()
    df1_encoded = pd.DataFrame(dense_data, columns = encoder.get_feature_names_out())

    df1 = pd.concat([df1,df1_encoded],
                    axis = 1)
    df1.drop(cat_cols,
             axis = 1,
             inplace = True)
    
    cols = df1.columns
    df1 = scaler.transform(df1)
    df1 = pd.DataFrame(df1,columns=cols)
    return df1
df1 = preprocessing()
st.write(df1)

prediction = model.predict(df1)

st.subheader('*Churn Prediction*')

import time

if st.button('*Click here to get your churn prediction*'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("This customer will not churn")
        else:
            st.success("This customer will churn")
    
        
        
        
        
        
        
        
        
