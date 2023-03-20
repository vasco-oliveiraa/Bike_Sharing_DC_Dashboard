import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from pycaret.regression import *

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

    # Load the pre-trained regression model
    # with open('bike_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # model = 0
    # Define the slider

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

    user_input = {
        'Date' : date,
        'Season' : season,
        'Year' : year,
        'Month' : month,
        'Hour' : hour,
        'Is_Holiday' : is_holiday,
        'Is_Working_Day' : is_working_day,
        'Weekday' : weekday,
        'Weather_Condition' : weather_condition,
        'Temperature' : temperature,
        'Humidity' : humidity,
        'Wind_Speed' : wind_speed
    }
    
    data = pd.DataFrame(user_input, index=[0])
    
    st.dataframe(data)
    
    # with open('bike_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    
    # # Define the button
    # if st.button('Predict'):
    #     # Predict the target value using the pre-trained model
    #     y_pred = model.predict([[x]])

    #     # Display the predicted target value
    #     st.write('Predicted target value:', y_pred[0])
    