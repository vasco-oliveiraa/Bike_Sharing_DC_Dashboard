import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

def FeatureEngineering(display=False):
    
    def BasicFeatureEngineering():

        data = pd.read_csv('bike-sharing_hourly.csv', index_col='instant').reset_index(drop=True)

        data.rename(columns={
            'dteday' : 'Date',
            'season' : 'Season',
            'yr' : 'Year',
            'mnth' : 'Month',
            'hr' : 'Hour',
            'holiday' : 'Is_Holiday',
            'weekday' : 'Weekday',
            'workingday' : 'Is_Working_Day',
            'weathersit' : 'Weather_Condition',
            'temp' : 'Temperature',
            'atemp' : 'Temperature_Feel',
            'hum' : 'Humidity',
            'windspeed' : 'Wind_Speed',
            'casual' : 'Casual_Users',
            'registered' : 'Registered_Users',
            'cnt' : 'Total_Users'
        }, inplace=True)
        
        # Denormalizing as to be repreducible with predictions that might fall outside normalization range

        data['Temperature'] *= 41

        data['Temperature_Feel'] *= 50

        data['Humidity'] *= 100

        data['Wind_Speed'] *= 67

        # Simple & Visually Useful Feature Engineering

        data['Date'] = pd.to_datetime(data['Date'])

        data['Day'] = data['Date'].apply(lambda x: x.day)

        data['Week'] = pd.to_numeric(data['Date'].apply(lambda x: datetime.strftime(x, '%U')))

        data['Year'].replace({0:2011,1:2012},inplace=True)

        data['Hour'].replace({0:24},inplace=True)

        data['Weekday'].replace({0:7},inplace=True)

        data['Weather_Condition'].replace(4,3,inplace=True)

        def day_period(hour):
            if 0 <= hour < 6:
                return 1 # Twilight
            elif 6 <= hour < 12:
                return 2 # Morning
            elif 12 <= hour < 18:
                return 3 # Afternoon
            elif 18 <= hour:
                return 4 # Night

        data['Day_Period'] = data.apply(lambda x: day_period(x['Hour']), axis=1)

        return data

    def AdvancedFeatureEngineering():

        data = BasicFeatureEngineering()

        data['Cos_Hour'] = data['Hour'].map(lambda x: np.cos(2*np.pi*x/24))

        data['Sin_Hour'] = data['Hour'].map(lambda x: np.sin(2*np.pi*x/24))

        data['Cos_Day'] = data['Day'].map(
            lambda x: np.cos(2*np.pi*x/30)
        )

        data['Sin_Day'] = data['Day'].map(
            lambda x: np.sin(2*np.pi*x/30)
        )

        data['Cos_Weekday'] = data['Weekday'].map(
            lambda x: np.cos(2*np.pi*x/7)
        )

        data['Sin_Weekday'] = data['Weekday'].map(
            lambda x: np.sin(2*np.pi*x/7)
        )

        data['Cos_Week'] = data['Week'].map(
            lambda x: np.cos(2*np.pi*x/52)
        )

        data['Sin_Week'] = data['Week'].map(
            lambda x: np.sin(2*np.pi*x/52)
        )

        data['Cos_Month'] = data['Month'].map(
            lambda x: np.cos(2*np.pi*x/12)
        )

        data['Sin_Month'] = data['Month'].map(
            lambda x: np.sin(2*np.pi*x/12)
        )

        data ['Weather_Factor'] = data['Temperature'] * data['Humidity'] * data['Wind_Speed'] * 1/data['Weather_Condition']

        # Hourly
        data['Average_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('mean')
        data['Min_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('min')
        data['Max_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('max')

        data['Average_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('mean')
        data['Min_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('min')
        data['Max_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('max')

        data['Average_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('mean')
        data['Min_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('min')
        data['Max_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('max')

        data['Average_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('mean')
        data['Min_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('min')
        data['Max_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('max')

        #Daily
        data['Average_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('mean')
        data['Min_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('min')
        data['Max_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('max')

        data['Average_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('mean')
        data['Min_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('min')
        data['Max_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('max')

        data['Average_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('mean')
        data['Min_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('min')
        data['Max_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('max')

        data['Average_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('mean')
        data['Min_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('min')
        data['Max_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('max')

        return data
    
    def NewFeatures():
        
        # Define variable definitions
        var_def = [
            ('Instant', 'Record index', 'Dropped'),
            ('Date', 'Date in YY-MM-DD format', 'Dropped'),
            ('Season', 'Winter = 1, Spring = 2, Summer = 3, Fall = 4', 'Kept'),
            ('Year','Year of record', 'Transformed'),
            ('Month', 'Month of record', 'Transformed'),
            ('Week', 'Week of record', 'Transformed'),
            ('Day', 'Day of record', 'Transformed'),
            ('Hour', 'Hour of record', 'Transformed'),
            ('Weekday', 'Monday = 1 ... Sunday = 7', 'Transformed'),
            ('Is_Holiday', '1 if holiday, 0 otherwise', 'Kept'),
            ('Is_Working_Day', '1 if neither holiday nor weekend, 0 otherwise','Kept'),
            ('Weather_Condition', '1 = Clear/Few Clouds, 2 = Cloudy/Mist  3 = Rain/Thunderstorm','Kept'),
            ('Temperature', 'Temperature in Celsius','Transformed'),
            ('Temperature_Feel', 'Temperature Feel in Celsius','Dropped'),
            ('Humidity', 'Humidity %', 'Transformed'),
            ('Wind_Speed', 'Wind Speed in km/h', 'Transformed'),
            ('Casual_Users', 'Number of Casual Users', 'Dropped'),
            ('Registered_Useres', 'Number of Registered Users', 'Dropped'),
            ('Total_Users', 'Sum of Casual and Registered Users', 'Kept'),
            ('Temperature/Feel_Factor', 'Factor of Temperature_Feel/Temperature', 'New Feature'),
            ('Weather_Factor', 'Factor of Temperature*Humidity*Wind_Speed*1/Weather_Condition', 'New Feature'),
            ('Average_Hourly_Users', 'Average users by hour of day', 'New Feature'),
            ('Min_Hourly_Users', 'Minimum users by hour of day', 'New Feature'),
            ('Max_Hourly_Users', 'Maximum users by hour of day', 'New Feature'),
            ('Average_Hourly_Temperature_Season', 'Average temperature by hour of day by season', 'New Feature'),
            ('Min_Hourly_Temperature_Season', 'Minimum temperature by hour of day by season', 'New Feature'),
            ('Max_Hourly_Temperature_Season', 'Maximum temperature by hour of day by season', 'New Feature'),
            ('Average_Hourly_Humidity_Season', 'Average humidity by hour of day by season', 'New Feature'),
            ('Min_Hourly_Humidity_Season', 'Minimum humidity by hour of day by season', 'New Feature'),
            ('Max_Hourly_Humidity_Season', 'Maximum humidity by hour of day by season', 'New Feature'),
            ('Average_Hourly_Wind_Speed_Season', 'Average wind speed by hour of day by season', 'New Feature'),
            ('Min_Hourly_Wind_Speed_Season', 'Minimum wind speed by hour of day by season', 'New Feature'),
            ('Max_Hourly_Wind_Speed_Season', 'Maximum wind speed by hour of day by season', 'New Feature'),
            ('Average_Daily_Users', 'Average daily users by season and weekday', 'New Feature'),
            ('Min_Daily_Users', 'Minimum daily users by season and weekday', 'New Feature'),
            ('Max_Daily_Users', 'Maximum daily users by season and weekday', 'New Feature'),
            ('Average_Daily_Temperature_Season', 'Average temperature by season and weekday', 'New Feature'),
            ('Min_Daily_Temperature_Season', 'Minimum temperature by season and weekday', 'New Feature'),
            ('Max_Daily_Temperature_Season', 'Maximum temperature by season and weekday', 'New Feature'),
            ('Average_Daily_Humidity_Season', 'Average humidity by season and weekday', 'New Feature'),
            ('Min_Daily_Humidity_Season', 'Minimum humidity by season and weekday', 'New Feature'),
            ('Max_Daily_Humidity_Season', 'Maximum humidity by season and weekday', 'New Feature'),
            ('Average_Daily_Wind_Speed_Season', 'Average wind speed by season and weekday', 'New Feature'),
            ('Min_Daily_Wind_Speed_Season', 'Minimum wind speed by season and weekday', 'New Feature'),
            ('Max_Daily_Wind_Speed_Season', 'Maximum wind speed by season and weekday', 'New Feature')
        ]
        
        # Define a dictionary containing the variable name as key and its corresponding code as value
        var_code = {
            'None' : '',
            'Year' : "data['Year'].replace({0:2011,1:2012},inplace=True)",
            'Month' : "data['Cos_Month'] = data['Month'].map(lambda x: np.cos(2*np.pi*x/12))\n"
            "data['Sin_Month'] = data['Month'].map(lambda x: np.sin(2*np.pi*x/12))",
            'Week' : "data['Cos_Week'] = data['Week'].map(lambda x: np.cos(2*np.pi*x/52))\n"
            "data['Sin_Week'] = data['Week'].map(lambda x: np.sin(2*np.pi*x/52))",
            'Day' : "data['Cos_Day'] = data['Day'].map(lambda x: np.cos(2*np.pi*x/30))\n"
            "data['Sin_Day'] = data['Day'].map(lambda x: np.sin(2*np.pi*x/30))",
            'Hour' : "data['Hour'].replace({0:24},inplace=True)\n"
            "data['Cos_Hour'] = data['Hour'].map(lambda x: np.cos(2*np.pi*x/24))\n"
            "data['Sin_Hour'] = data['Hour'].map(lambda x: np.sin(2*np.pi*x/24))",
            'Weekday' : "data['Cos_Weekday'] = data['Weekday'].map(lambda x: np.cos(2*np.pi*x/7))\n"
            "data['Sin_Weekday'] = data['Weekday'].map(lambda x: np.sin(2*np.pi*x/7)",
            'Temperature' : "data['Temperature'] *= 41",
            'Humidity' : "data['Humidity'] *= 100",
            'Wind_Speed' : "data['Wind_Speed'] *= 67",
            'Temperature/Feel_Factor' : "data['Temperature_Feel'] / abs(data['Temperature']",
            'Weather_Factor' : "data ['Weather_Factor'] = data['Temperature'] * data['Humidity'] * data['Wind_Speed'] * 1/data['Weather_Condition']",
            'Average_Hourly_Users': "data['Average_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('mean')",
            'Min_Hourly_Users' : "data['Min_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('min')",
            'Max_Hourly_Users' : "data['Max_Hourly_Users'] = data.groupby('Hour')['Total_Users'].transform('max')",
            'Average_Hourly_Temperature_Season' : "data['Average_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('mean')",
            'Min_Hourly_Temperature_Season' : " data['Min_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('min')",
            'Max_Hourly_Temperature_Season' : "data['Max_Hourly_Temperature_Season'] = data.groupby(['Season','Hour'])['Temperature'].transform('max')",
            'Average_Hourly_Humidity_Season' : "data['Average_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('mean')",
            'Min_Hourly_Humidity_Season' : "data['Min_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('min')",
            'Max_Hourly_Humidity_Season' :" data['Max_Hourly_Humidity_Season'] = data.groupby(['Season','Hour'])['Humidity'].transform('max')",
            'Average_Hourly_Wind_Speed_Season' : "data['Average_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('mean')",
            'Min_Hourly_Wind_Speed_Season' : "data['Min_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('min')",
            'Max_Hourly_Wind_Speed_Season' : "data['Max_Hourly_Wind_Speed_Season'] = data.groupby(['Season','Hour'])['Wind_Speed'].transform('max')",
            'Average_Daily_Users' : "data['Average_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('mean')",
            'Min_Daily_Users' : "data['Min_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('min')",
            'Max_Daily_Users' : "data['Max_Daily_Users'] = data.groupby(['Season','Weekday'])['Total_Users'].transform('max')",
            'Average_Daily_Temperature_Season' : "data['Average_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('mean')",
            'Min_Daily_Temperature_Season' : "data['Min_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('min')",
        'Max_Daily_Temperature_Season' : "data['Max_Daily_Temperature_Season'] = data.groupby(['Season','Weekday'])['Temperature'].transform('max')",
            'Average_Daily_Humidity_Season' :"data['Average_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('mean')",
            'Min_Daily_Humidity_Season' : "data['Min_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('min')",
            'Max_Daily_Humidity_Season' : "data['Max_Daily_Humidity_Season'] = data.groupby(['Season','Weekday'])['Humidity'].transform('max')",
            'Average_Daily_Wind_Speed_Season' : "data['Average_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('mean')",
            'Min_Daily_Wind_Speed_Season' : "data['Min_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('min')",
            'Max_Daily_Wind_Speed_Season' : "data['Max_Daily_Wind_Speed_Season'] = data.groupby(['Season','Weekday'])['Wind_Speed'].transform('max')"
        }
        
        var_explain = {
            'None' : '',
            'Month' : 'Sine and cosine functions generate different but unique numerical values for each hour, which together provide a complete representation of the cyclical nature of time.',
            'Week' : 'Sine and cosine functions generate different but unique numerical values for each hour, which together provide a complete representation of the cyclical nature of time.',
            'Day' : 'Sine and cosine functions generate different but unique numerical values for each hour, which together provide a complete representation of the cyclical nature of time.',
            'Hour' : 'Sine and cosine functions generate different but unique numerical values for each hour, which together provide a complete representation of the cyclical nature of time.',
            'Weekday' : 'Sine and cosine functions generate different but unique numerical values for each hour, which together provide a complete representation of the cyclical nature of time.',
            'Temperature' : "Denormalized by multiplying all their values by the respective real maximums - as per indication of the original dataset. While normalization like this might benefit some model's understanding of the data, it critically hinders the model's predictability of values above that maximum, as values will no longer be contained between 0 and 1.",
            'Temperature_Feel' : "Denormalized by multiplying all their values by the respective real maximums - as per indication of the original dataset. While normalization like this might benefit some model's understanding of the data, it critically hinders the model's predictability of values above that maximum, as values will no longer be contained between 0 and 1.",
            'Humidity' : "Denormalized by multiplying all their values by the respective real maximums - as per indication of the original dataset. While normalization like this might benefit some model's understanding of the data, it critically hinders the model's predictability of values above that maximum, as values will no longer be contained between 0 and 1.",
            'Wind_Speed' : "Denormalized by multiplying all their values by the respective real maximums - as per indication of the original dataset. While normalization like this might benefit some model's understanding of the data, it critically hinders the model's predictability of values above that maximum, as values will no longer be contained between 0 and 1.",
        }
        
        # Create a DataFrame from the list of lists and set the index to the variable names
        df = pd.DataFrame(var_def, columns=['Variable', 'Definition', 'Status'])
        df.set_index('Variable', inplace=True)

        def color_status(val):
            if val == 'New Feature':
                color = 'color: green'
            elif val == 'Transformed':
                color = 'color: yellow'
            elif val == 'Dropped':
                color = 'color: red'
            else:
                color = ''
            return color

        styled_df = df.style.applymap(color_status, subset=['Status'])

        # Display the styled DataFrame in Streamlit
        st.dataframe(styled_df, use_container_width=True)
        
        selected_var = st.selectbox(label='Select a feature to see the transformation and the reasoning behind it', options=list(var_code.keys()))

        if selected_var == 'None':
            st.write('')

        elif selected_var != 'None' and selected_var not in var_explain.keys():
            st.code(var_code[selected_var], language='python')
            
        else:
            st.code(var_code[selected_var], language='python')
            st.caption(var_explain[selected_var])
        
    if display == True:
        
        st.title("🚲 Bike-Sharing in Washington D.C")
        
        st.header('🛠️ Feature Engineering')

        st.write("Feature engineering is an ***essential*** part of the predictive model-building process  that involves ***transforming raw data into meaningful features*** for model training. It helps to ***improve model accuracy, reduce overfitting, and incorporate domain knowledge*** into the model building process. Through techniques such as data cleaning, dimensionality reduction, and feature selection, feature engineering ***enables the extraction of relevant information*** from complex datasets.") 

        st.write("In this section, you will be able to see ***what transformations were made*** to the data in order to facilitate its understanding as well as to increase the model's predictive capability. In this sense, ***5 features were dropped, 5 features were kept, 9 features were transformed***, and ***26  new features were created***.  ***Not all the features*** that were created in this phase ***ended up being used*** in the final model, as through iterative testing, ***only the best performing features*** were selected.")
        
        st.write('In the table below, you can find a ***summary the new features*** with their ***definitions*** and indication of their ***status*** as per the first model run. If ***more detail*** is needed, the ***dropdown menu*** shows the ***code snippet*** for each transformation as well as the ***explanation for such transformation***.')
    
        NewFeatures()

        data = BasicFeatureEngineering()
        
    return {
        'BasicFE': BasicFeatureEngineering(),
        'AdvancedFE': AdvancedFeatureEngineering()
    }