import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from FeatureEngineering import FeatureEngineering
from VariableInfoInsights import VariableInfoInsights
from TimeDistributionInsights import TimeDistributionInsights

def DataExploration(display=True):
    
    data = FeatureEngineering(display=False)['BasicFE']
    
    # Categorical Variables
    cat_variables = [
        'Date',
        'Season',
        'Year',
        'Month',
        'Week',
        'Day_Period',
        'Hour',
        'Is_Holiday',
        'Weekday',
        'Is_Working_Day',
        'Weather_Condition'
    ]
    
    num_variables = [x for x in data.columns if x not in cat_variables]

    # Title & Header

    st.title("Bike-Sharing in Washington D.C")

    st.header("Data Exploration")

    st.subheader("Overview")

    st.markdown("Data exploration is a ***critical step*** in the predictive model-building process as it lays the foundation for the development of the ***best and most effective model***. The main objective of data exploration is to gain a ***better understanding of the data*** and its underlying patterns, relationships, and characteristics. Likewise, data exploration helps ***uncover important features and trends*** that improve model development and decision-making.")
    
    st.markdown("In this section, you will be able to see what data exploration was made and get an initial idea of the ***possible transformations*** that need to be performed for the modeling process. In this sense, the dataset is composed of ***17 columns*** and ***17379 rows***. Likewise, there are ***no null values*** in the dataset which makes it accurate and easier to deal with.")
    
    st.markdown("On the other hand, understanding the ***distribution of the variables and the relationship between them*** is essential now that it will make it easier to ***identify potential issues***. Based on this, some features have a ***categorical data type***, but instead, they should have a ***numerical*** one. Also, some features are ***right skewed*** like `Casual_Users`, `Registered_Users`, `Total_Users`, and `Wind Speed`.")
    
    st.markdown("In the table below, you can see dataset information where you will find a ***description of each of the features***. Also, you will find a ***descriptive summary*** of the dataset as a whole and of each feature independently. Finally, you will be able to see the ***users' distribution by categorical or time dimensions, or a mix of both***.")
                 
    tab1, tab2, tab3 = st.tabs(['Dataset Information','Dataset Analysis','Variable Analysis'])
    
    with tab1:
        
        def DatasetInformation():
            
            var_def = {
                'Instant' : 'Record index',
                'Date' : 'Date in YY-MM-DD format',
                'Season' : 'Winter = 1, Spring = 2, Summer = 3, Fall = 4',
                'Year' : 'Year of record',
                'Month' : 'Month of record',
                'Week' : 'Week of record',
                'Day' : 'Day of record',
                'Hour' : 'Hour of record',
                'Weekday' : 'Monday = 1 ... Sunday = 7',
                'Is_Holiday' : '1 if holiday, 0 otherwise',
                'Is_Working_Day' : '1 if neither holiday nor weekend, 0 otherwise',
                'Weather_Condition' : '1 = Clear/Few Clouds, 2 = Cloudy/Mist  3 = Rain/Thunderstorm',
                'Temperature' : 'Temperature in Celsius',
                'Temperature_Feel' : 'Temperature Feel in Celsius',
                'Humidity' : 'Humidity %',
                'Wind_Speed' : 'Wind Speed in km/h',
                'Casual_Users' : 'Number of Casual Users',
                'Registered_Useres' : 'Number of Registered Users',
                'Total_Users' : 'Sum of Casual and Registered Users'
            }
            var_def_list = list(var_def.items())

            # Pass the var_def_list to DataFrame constructor and specify column names
            df = pd.DataFrame(var_def_list, columns=['Variable', 'Definition'])
            df.index = df['Variable']
            df.drop(columns=['Variable'],inplace=True)
            
            st.dataframe(df, use_container_width=True)
        
        DatasetInformation()
            
    with tab2:
        # Dataset Analysis Section
        def DatasetAnalysis():

            # Missing cells
            missing_cells = data.isnull().sum().sum()

            # Missing cells (%)
            missing_cells_percent = (missing_cells / (data.shape[0] * data.shape[1])) * 100

            # Duplicate rows
            duplicate_rows = data.duplicated().sum()

            # Duplicate rows (%)
            duplicate_rows_percent = (duplicate_rows / data.shape[0]) * 100


            st.subheader('Dataset Analysis')

            st.write('')

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Number of Observations: {len(data)}")
                st.write(f"Missing Cells: {missing_cells}")
                st.write(f"% of Missing Cells: {missing_cells_percent}")
                st.write(f"Duplicate Rows: {duplicate_rows}")
                st.write(f"% of Missing Cells: {duplicate_rows_percent}")

            with col2:
                st.write(f"Number of Variables: {len(data.columns)}")
                st.write(f"Number of Categorical Variables: {len(cat_variables)}")
                st.write(f"Number of Numerical Variables: {len(num_variables)}")
                
            st.caption(f':bulb: Write Overall Dataset Insight here!!!')

        DatasetAnalysis()
    
    with tab3:
    
        def VariableAnalysis():

            st.subheader("Variable Analysis")

            # Select variable
            variable = st.selectbox("Select a variable", data.columns)

            if variable in cat_variables:
                variable_type = 'Categorical'
            else:
                variable_type = 'Numerical'

            chart = px.histogram(
                data,
                x=variable,
                color_discrete_sequence=['orange'],
                opacity=.6,
                barmode='group',
                height=300,
                title=f'Distribution of {variable}',
                text_auto=True,
                labels={'count': 'Count'}
            )

            chart.update_layout(
                bargap=0.2,
                bargroupgap=0.1
            )

            # Calculate number of distinct values and percentage of distinct values
            n_distinct = data[variable].nunique()

            # Calculate number of missing values and percentage of missing values
            n_missing = data[variable].isnull().sum()
            pct_missing = round(n_missing / len(data[variable]) * 100, 2)

            # Display information and graph side by side
            col1, col2 = st.columns([2, 3])
            with col1:
                st.write('')
                st.write('')
                st.write(f"Variable Type: {variable_type}")
                st.write(f"Number of Distinct Values: {n_distinct}")
                st.write(f"Missing Values: {n_missing}")
                st.write(f"% of Missing Values: {pct_missing}")
            with col2:
                st.plotly_chart(chart, use_container_width=True)

            st.caption(f':bulb: {VariableInfoInsights(variable)}')

        VariableAnalysis()
    
    def TimeDistribution():
        
        with st.expander('Users Distribution by Categorical/Time Dimensions', expanded=True):
        
            st.subheader('Users Distribution by Categorical/Time Dimensions')

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                    start_date = st.date_input(
                        'Start Date',
                        value=min(data['Date']),
                        min_value=min(data['Date']),
                        max_value=max(data['Date'])
                    )

            with col2:

                    end_date = st.date_input(
                        'End Date',
                        value=max(data['Date']),
                        min_value=min(data['Date']),
                        max_value=max(data['Date'])
                    )

            with col3:

                start_hour = st.selectbox(
                    'Start hour',
                    options = range(1,25)
                )

            with col4:

                end_hour = st.selectbox(
                    'End Hour',
                    options = range(1,25),
                    index = 23
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric('Total Users',
                          f"{data.loc[(data['Date'].dt.date>=start_date) & (data['Date'].dt.date<=end_date) & (data['Hour']>=start_hour) & (data['Hour']<=end_hour)]['Total_Users'].sum():,.0f}"
                         )

            with col2:
                st.metric('Registered Users',
                          f"{data.loc[(data['Date'].dt.date>=start_date) & (data['Date'].dt.date<=end_date) & (data['Hour']>=start_hour) & (data['Hour']<=end_hour)]['Registered_Users'].sum():,.0f}"
                         )

            with col3:
                st.metric('Casual Users',
                          f"{data.loc[(data['Date'].dt.date>=start_date) & (data['Date'].dt.date<=end_date) & (data['Hour']>=start_hour) & (data['Hour']<=end_hour)]['Casual_Users'].sum():,.0f}"
                         )

            col1, col2, col3 = st.columns(3)

            with col1:

                user_type = st.selectbox(
                    'Select the Type of Users',
                    [
                        'Total Users',
                        'Registered Users',
                        'Casual Users'
                    ]
                )

                user_type_str = user_type.replace(' ', '_')

            with col2:

                cat_dim = st.selectbox(
                    'Select Categorical Dimension',
                    [
                        'None',
                        'Weather Condition',
                        'Year',
                        'Season',
                        'Weekday',
                        'Is Holiday',
                        'Is Working Day'
                    ]
                )

                cat_dim_str = cat_dim.replace(' ', '_')

            with col3:

                time_dim_options = [
                    'Hour',
                    'Day',
                    'Week',
                    'Month',
                    'Year',
                    'Total'
                ]

                if cat_dim == 'None':

                    time_dim_options_without_total = [option for option in time_dim_options if option != "Total"]

                    time_dim = st.selectbox(
                        'Select the Time Dimension',
                        time_dim_options_without_total
                    )
                else:

                    time_dim = st.selectbox(
                        'Select the Time Dimension', 
                        time_dim_options
                    )


            if time_dim == 'Total' and cat_dim != 'None':

                fig = px.bar(
                    data.loc[
                        (data['Date'].dt.date>=start_date) & 
                        (data['Date'].dt.date<=end_date) &
                        (data['Hour']>=start_hour) &
                        (data['Hour']<=end_hour)
                    ].groupby(cat_dim_str)[user_type_str].sum(),
                    title=f"Distribution of {user_type} by {cat_dim}",
                    labels={
                        cat_dim: cat_dim_str,
                        user_type_str: user_type
                    }
                )

                # Update the axis titles
                fig.update_layout(
                    xaxis_title=cat_dim,
                    yaxis_title=user_type
                )

            elif time_dim != 'Total' and cat_dim == 'None':

                fig = px.line(
                    data.loc[
                        (data['Date'].dt.date>=start_date) & 
                        (data['Date'].dt.date<=end_date) &
                        (data['Hour']>=start_hour) &
                        (data['Hour']<=end_hour)
                    ].groupby(time_dim)[user_type_str].sum(),
                    title=f"Distribution of {user_type} by {time_dim}",
                    labels={
                        time_dim: time_dim,
                        user_type_str: user_type
                    }
                )

                # Update the axis titles
                fig.update_layout(
                    xaxis_title=time_dim,
                    yaxis_title=user_type
                )

            elif time_dim != 'Total' and cat_dim != 'None':

                fig = px.line(
                    data.loc[
                        (data['Date'].dt.date>=start_date) &
                        (data['Date'].dt.date<=end_date) &
                        (data['Hour']>=start_hour) &
                        (data['Hour']<=end_hour)
                    ].groupby([cat_dim_str,time_dim])[user_type_str].sum().unstack().T,
                    color = cat_dim_str,
                    title=f"Distribution of {user_type} by {cat_dim} by {time_dim}",
                    labels={
                        time_dim: time_dim,
                        user_type_str: user_type,
                        cat_dim_str : cat_dim
                    }
                )

                # Update the axis titles
                fig.update_layout(
                    xaxis_title=time_dim,
                    yaxis_title=user_type
                )

            st.plotly_chart(fig, use_container_width=True)
            
            if TimeDistributionInsights(time_dim,user_type,cat_dim) == None:
                
                st.write('')
                
            else:
                
                st.caption(f':bulb: {TimeDistributionInsights(time_dim,user_type,cat_dim)}')

    TimeDistribution()
    
    st.subheader("Analysis")

    st.markdown("After understanding how the dataset behaves and finding different relations between features, we can analyze how the bike-sharing service ***currently works***. To begin with, the ***overall use*** of the service is ***low***. With this in mind, the ***peak hours*** in which the service is used are ***8:00*** and ***between 17:00 and 18:00*** which could be related to high commuting. Also, the ***usage rate*** is ***impacted by the seasons*** but in general, it is ***higher on weekdays than on holidays or weekends***.")
    
    st.markdown("On the other hand, and as expected, the ***average usage*** of the bike-sharing service is ***higher*** when the ***weather conditions are sunny/misty/cloudy*** rather than when it includes rain or snow. In this sense, the ***outside temperature and the feeling temperature*** have a relation with the bike-sharing service whereas it ***decreases***, the ***usage decreases***, and vice versa. Finally, as ***humidity increases*** the ***usage*** of bikes ***decreases***. All previously mentioned can be related to the weather conditions meaning that when there is rain or snow there is more humidity and lower temperatures, hence the bike-sharing service lowers.")
    
    st.markdown("With this in mind, some ***recommendations*** based on the actual behavior could be that bike ***provisions*** should be ***higher during the weekdays***. Similarly, from June to September, hence ***summer***, could be the best months to also ***increase the bike provisions***. On the other hand, some ***promotions*** could be established for example ***during the weekends/holidays*** or when the ***weather conditions aren’t good***. The previous aimed to ***increase the usage*** of the bike-sharing service and ***lower the costs***.")
    
# DataExploration() # Delete when trying app navigation