import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ModelBuilding import Model

def Prediction(display=True):
    
    st.title("🚲 Bike-Sharing in Washington D.C")
        
    st.header('🔮 Prediction')

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
            options = range(0,24)
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
        
    season = get_season(date)
    
    year = date.year

    day = date.day

    week = pd.to_numeric(datetime.strftime(date, '%U'))

    month = date.month

    weekday = date.weekday()+1
    
    # Working Day
    if weekday in [6,7] or is_holiday == 1:
        is_working_day = 0
    else:
        is_working_day = 1
        
    # Day Period
    if 0 <= hour < 6:
        day_period = 1 # Twilight
    elif 6 <= hour < 12:
        day_period = 2 # Morning
    elif 12 <= hour < 18:
        day_period = 3 # Afternoon
    elif 18 <= hour:
        day_period = 4 # Night
        
    # Cos & Sin
    cos_hour = np.cos(2*np.pi*hour/24)
    
    sin_hour = np.sin(2*np.pi*hour/24)
    
    cos_day = np.cos(2*np.pi*day/30)
    
    sin_day = np.sin(2*np.pi*day/30)
    
    cos_weekday = np.cos(2*np.pi*weekday/7)
    
    sin_weekday = np.sin(2*np.pi*weekday/7)
    
    cos_week = np.cos(2*np.pi*week/52)
    
    sin_week = np.sin(2*np.pi*week/52)
    
    cos_month = np.cos(2*np.pi*month/12)
    
    sin_month = np.sin(2*np.pi*month/12)
    
    # Weather Factor
    weather_factor = temperature * humidity * wind_speed * 1/weather_condition
    
    reference_data = pd.read_csv('reference_data.csv')
    
    # Hourly Aggregates
    
    average_hourly_users = reference_data.loc[reference_data['Hour'] == hour, 'Average_Hourly_Users'].iloc[0]
    min_hourly_users = reference_data.loc[reference_data['Hour'] == hour, 'Min_Hourly_Users'].iloc[0]
    max_hourly_users = reference_data.loc[reference_data['Hour'] == hour, 'Max_Hourly_Users'].iloc[0]
    
    average_hourly_temperature_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Average_Hourly_Temperature_Season'].iloc[0]
    min_hourly_temperature_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Min_Hourly_Temperature_Season'].iloc[0]
    max_hourly_temperature_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Max_Hourly_Temperature_Season'].iloc[0]
    
    average_hourly_humidity_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Average_Hourly_Humidity_Season'].iloc[0]
    min_hourly_humidity_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Min_Hourly_Humidity_Season'].iloc[0]
    max_hourly_humidity_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Max_Hourly_Humidity_Season'].iloc[0]
    
    average_hourly_wind_speed_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Average_Hourly_Wind_Speed_Season'].iloc[0]
    min_hourly_wind_speed_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Min_Hourly_Wind_Speed_Season'].iloc[0]
    max_hourly_wind_speed_season = reference_data.loc[(reference_data['Hour'] == hour) & (reference_data['Season'] == season), 'Max_Hourly_Wind_Speed_Season'].iloc[0]
    
     # Daily Aggregates
        
    average_daily_users = reference_data.loc[reference_data['Day'] == day, 'Average_Daily_Users'].iloc[0]
    min_daily_users = reference_data.loc[reference_data['Day'] == day, 'Min_Daily_Users'].iloc[0]
    max_daily_users = reference_data.loc[reference_data['Day'] == day, 'Max_Daily_Users'].iloc[0]

    average_daily_temperature_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Average_Daily_Temperature_Season'].iloc[0]
    min_daily_temperature_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Min_Daily_Temperature_Season'].iloc[0]
    max_daily_temperature_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Max_Daily_Temperature_Season'].iloc[0]
    
    average_daily_humidity_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Average_Daily_Humidity_Season'].iloc[0]
    min_daily_humidity_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Min_Daily_Humidity_Season'].iloc[0]
    max_daily_humidity_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Max_Daily_Humidity_Season'].iloc[0]
    
    average_daily_wind_speed_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Average_Daily_Wind_Speed_Season'].iloc[0]
    min_daily_wind_speed_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Min_Daily_Wind_Speed_Season'].iloc[0]
    max_daily_wind_speed_season = reference_data.loc[(reference_data['Day'] == day) & (reference_data['Season'] == season), 'Max_Daily_Wind_Speed_Season'].iloc[0]
    
    
    user_input = {
        'Season': season,
        'Year': year,
        'Day_Period': day_period,
        'Is_Holiday': is_holiday,
        'Is_Working_Day': is_working_day,
        'Weather_Condition': weather_condition,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        
         'Cos_Hour': cos_hour,
         'Sin_Hour': sin_hour,
         'Cos_Day': cos_day,
         'Sin_Day': sin_day,
         'Cos_Weekday': cos_weekday,
         'Sin_Weekday': sin_weekday,
         'Cos_Week': cos_week,
         'Sin_Week': sin_week,
         'Cos_Month': cos_month,
         'Sin_Month': sin_month,
        
         'Weather_Factor': weather_factor,
        
         'Average_Hourly_Users': average_hourly_users,
         'Min_Hourly_Users': min_hourly_users,
         'Max_Hourly_Users': max_hourly_users,
        
         'Average_Hourly_Temperature_Season': average_hourly_temperature_season,
         'Min_Hourly_Temperature_Season': min_hourly_temperature_season,
         'Max_Hourly_Temperature_Season': max_hourly_temperature_season,
        
         'Average_Hourly_Humidity_Season': average_hourly_humidity_season,
         'Min_Hourly_Humidity_Season': min_hourly_humidity_season,
         'Max_Hourly_Humidity_Season': max_hourly_humidity_season,
        
         'Average_Hourly_Wind_Speed_Season': average_hourly_wind_speed_season,
         'Min_Hourly_Wind_Speed_Season': min_hourly_wind_speed_season,
         'Max_Hourly_Wind_Speed_Season': max_hourly_wind_speed_season,
        
         'Average_Daily_Users': average_daily_users,
         'Min_Daily_Users': min_daily_users,
         'Max_Daily_Users': max_daily_users,
        
         'Average_Daily_Temperature_Season': average_daily_temperature_season,
         'Min_Daily_Temperature_Season': min_daily_temperature_season,
         'Max_Daily_Temperature_Season': max_daily_temperature_season,
        
         'Average_Daily_Humidity_Season': average_daily_humidity_season,
         'Min_Daily_Humidity_Season': min_daily_humidity_season,
         'Max_Daily_Humidity_Season': max_daily_humidity_season,
        
         'Average_Daily_Wind_Speed_Season': average_daily_wind_speed_season,
         'Min_Daily_Wind_Speed_Season': min_daily_wind_speed_season,
         'Max_Daily_Wind_Speed_Season': max_daily_wind_speed_season
    }
    
    data = pd.DataFrame(user_input, index=[0])
    
    st.dataframe(data)
    
    # Define the button
    if st.button('Predict'):
        # Predict the target value using the pre-trained model
        y_pred = Model(data)

        # Display the predicted target value
        st.write('Predicted target value:', y_pred[0])
    
        