import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

from ModelBuilding import RF_Model

def Prediction(display=True):
    
    st.title("🚲 Bike-Sharing in Washington D.C")
        
    st.header('🔮 Prediction')
    
    def get_season(date):
        # Extract month and day from the date
        month = date.month
        day = date.day

        # Define season based on month and day
        if (month == 12 and day >= 21) or (month == 1) or (month == 2) or (month == 3 and day < 20):
            return 1
        elif (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day < 21):
            return 2
        elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 22):
            return 3
        elif (month == 9 and day >= 22) or (month == 10) or (month == 11) or (month == 12 and day < 21):
            return 4

    col1, col2, col3 = st.columns(3)

    with col1:
        date_str = '2013-01-01'
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        date = st.date_input(
            'Date',
            value=date_obj,
            min_value=date_obj,
            max_value=date_obj+relativedelta(years=1),
        )
    with col2:

        hour = st.selectbox(
            'Hour',
            options = range(1,25)
        )

    with col3:

        st.write(' ')
        st.write(' ')
        
        is_holiday = st.checkbox('Is Holiday')

        if is_holiday:
            is_holiday = 1
        else:
            is_holiday = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        weather_condition = st.slider(
            'Weather Condition',
            min_value = 1,
            max_value = 3,
            value = 1,
            step=1
        )
        
    with col2:

        temperature = st.slider(
            'Temperature',
            min_value = -25,
            max_value = 45,
            value = 15,
            step=1
        )

    with col3:

        humidity = st.slider(
            'Humidity',
            min_value = 0,
            max_value = 100,
            value = 50,
            step=1
        )

    with col4:
        wind_speed = st.slider(
            'Wind Speed',
            min_value = 0,
            max_value = 100,
            value = 25,
            step=1
        )


    day = date.day

    week = pd.to_numeric(datetime.strftime(date, '%U'))

    month = date.month

    year = date.year

    weekday = date.weekday()+1
    
    if weekday in [6,7] or is_holiday == 1:
        is_working_day = 0
    else:
        is_working_day = 1

    season = get_season(date)

    # user_input = {
    #     'Date' : date,
    #     'Season' : season,
    #     'Year' : year,
    #     'Month' : month,
    #     'Hour' : hour,
    #     'Is_Holiday' : is_holiday,
    #     'Is_Working_Day' : is_working_day,
    #     'Weekday' : weekday,
    #     'Weather_Condition' : weather_condition,
    #     'Temperature' : temperature,
    #     'Humidity' : humidity,
    #     'Wind_Speed' : wind_speed
    # }
    
    user_input = {
        'Season': 4.0,
         'Year': 2012.0,
         'Day_Period': 3.0,
         'Is_Holiday': 0.0,
         'Is_Working_Day': 1.0,
         'Weather_Condition': 1.0,
         'Temperature': 18.86,
         'Humidity': 41.0,
         'Wind_Speed': 23.9994,
         'Cos_Hour': -0.5000000000000004,
         'Sin_Hour': -0.8660254037844385,
         'Cos_Day': -0.8090169943749473,
         'Sin_Day': 0.5877852522924732,
         'Cos_Weekday': -0.2225209339563146,
         'Sin_Weekday': -0.9749279121818236,
         'Cos_Week': 0.23931566428755738,
         'Sin_Week': -0.9709418174260521,
         'Cos_Month': 0.5000000000000001,
         'Sin_Month': -0.8660254037844386,
         'Temperature/Feel_Factor': 1.204931071049841,
         'Weather_Factor': 18557.776044000002,
         'Average_Hourly_Users': 311.9835616438356,
         'Min_Hourly_Users': 11.0,
         'Max_Hourly_Users': 783.0,
         'Average_Hourly_Temperature_Season': 19.907005649717515,
         'Min_Hourly_Temperature_Season': 9.02,
         'Max_Hourly_Temperature_Season': 30.34,
         'Average_Hourly_Humidity_Season': 53.49152542372882,
         'Min_Hourly_Humidity_Season': 16.0,
         'Max_Hourly_Humidity_Season': 100.0,
         'Average_Hourly_Wind_Speed_Season': 13.989032203389831,
         'Min_Hourly_Wind_Speed_Season': 0.0,
         'Max_Hourly_Wind_Speed_Season': 35.0008,
         'Average_Daily_Users': 204.11166666666668,
         'Min_Daily_Users': 1.0,
         'Max_Daily_Users': 900.0,
         'Average_Daily_Users_Season': 17.157133333333334,
         'Min_Daily_Users_Season': 6.5600000000000005,
         'Max_Daily_Users_Season': 30.34,
         'Average_Daily_Humidity_Season': 63.266666666666666,
         'Min_Daily_Humidity_Season': 28.999999999999996,
         'Max_Daily_Humidity_Season': 100.0,
         'Average_Daily_Wind_Speed_Season': 11.183751666666668,
         'Min_Daily_Wind_Speed_Season': 0.0,
         'Max_Daily_Wind_Speed_Season': 36.9974
    }
    
    data = pd.DataFrame(user_input, index=[0])
    
    st.dataframe(data)
    
    # Define the button
    if st.button('Predict'):
        # Predict the target value using the pre-trained model
        y_pred = RF_Model(data)
        
    # Load the pre-trained regression model
    # with open('bike_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
        
    # model = 0
    # Define the slider

        # Display the predicted target value
        st.write('Predicted target value:', y_pred[0])
        
    # with open('bike_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    
    pipeline = load_model('bike_model')
    prediction = predict_model(pipeline, data_unseen)
    
    st.dataframe(prediction)
        