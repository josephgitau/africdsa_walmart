# import all the app dependencies
import streamlit as st
import pandas as pd
# fit a single model
from sklearn.ensemble import RandomForestRegressor
import joblib

# import walmart data
df = pd.read_csv("Walmart.csv")

df['Date'][1].split('-')
# Extract day, month and year from the Date column using split function
df['Day'] = df['Date'].apply(lambda x : x.split('-')[0])
df['Month'] = df['Date'].apply(lambda x : x.split('-')[1])
df['Year'] = df['Date'].apply(lambda x : x.split('-')[2])
# save the columns as integer type
df['Day'] = df['Day'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)

# drop the Date column
df.drop('Date', axis = 1, inplace = True)

# create X, y 

X = df.drop('Weekly_Sales', axis = 1)
y = df['Weekly_Sales']

# instantiate the model
random_forest_model = RandomForestRegressor()# add hyperparameters n_estimators - are the number of trees in the forest)

# fit the model to the data
random_forest_model.fit(X, y)

st.set_option('deprecation.showPyplotGlobalUse', False)

# 1: serious injury, 2: Slight injury, 0: Fatal Injury

st.set_page_config(page_title="Walmart Sales Prediction App",
        page_icon="🚧", layout="wide")

## Title
st.title("Walmart Sales Prediction App")

## Subheader
st.subheader("Predicting the sales of Walmart stores")

def predict_sales(user_data):
    user_df = pd.DataFrame(user_data, index = [0])
    pred = random_forest_model.predict(user_df)
    return pred

st.sidebar.header('User Input Parameters')
def user_input_features():
    store = st.sidebar.slider('Store', 1, 45, 1)
    isholiday = st.sidebar.slider('IsHoliday', 0, 1, 0)
    temperature = st.sidebar.slider('Temperature', -7.29, 101.95, 59.36)
    fuelprice = st.sidebar.slider('Fuel_Price', 2.47, 4.47, 3.41)
    cpi = st.sidebar.slider('CPI', 126.06, 227.23, 211.26)
    unemployment = st.sidebar.slider('Unemployment', 3.88, 14.31, 8.1)
    day = st.sidebar.slider('Day', 1, 31, 1)
    month = st.sidebar.slider('Month', 1, 12, 1)
    year = st.sidebar.slider('Year', 2010, 2012, 2010)

    data = {'Store': store, 'Holiday_Flag': isholiday, 'Temperature': temperature, 'Fuel_Price': fuelprice, 'CPI': cpi, 'Unemployment': unemployment, 'Day': day, 'Month': month, 'Year': year}
    features = pd.DataFrame(data, index = [0])
    return features

features = user_input_features()
# display the user input features
st.subheader('User Input features')
st.dataframe(features)


pred = predict_sales(features)
st.subheader('Predicted Sales')
st.write(pred)
