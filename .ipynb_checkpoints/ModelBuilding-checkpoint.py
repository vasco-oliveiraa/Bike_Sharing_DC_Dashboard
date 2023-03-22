import streamlit as st
import pandas as pd
import plotly.express as px
from FeatureEngineering import FeatureEngineering
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def ModelBuilding(display=True):
    
    if display == True:
        
        st.title("🚲 Bike-Sharing in Washington D.C")
        
        st.header('🧪 Model Building')
            
        data = FeatureEngineering(display=False)['BasicFE']

        with st.expander('Correlation Matrix', expanded=True):

            st.subheader('Correlation Matrix')

            fig = px.imshow(data.corr())
            fig.update_layout(
            width=700,
            height=700
            )

            st.plotly_chart(fig, use_container_width=True)

def Model(user_input):

    data = FeatureEngineering(display=False)['BasicFE']
    data = FeatureEngineering(display=False)['AdvancedFE']

    target = ['Total_Users']

    cat_cols = [
        'Season',
        'Year',
        'Day_Period',
        'Is_Holiday',
        'Is_Working_Day',
        'Weather_Condition'
    ]

    ordinal_cols = {'Weather_Condition' : ['1','2','3']}

    date_cols = ['Date']

    ignore_cols = [
        'Hour',
        'Day',
        'Month',
        'Weekday',
        'Week',
        'Temperature_Feel',
        'Registered_Users',
        'Casual_Users'
    ]

    other_cols = cat_cols + date_cols + ignore_cols + target

    num_cols = [x for x in data.columns if x not in other_cols]
    
    data.drop(columns=ignore_cols,inplace=True)

    data_unseen = data.loc[data['Date']>='2012-09-01']
    data = data.loc[data['Date']<'2012-09-01']

    X_train = data[[*cat_cols,*num_cols]]
    y_train = data['Total_Users']

    X_test = data_unseen[[*cat_cols,*num_cols]]
    y_test = data_unseen['Total_Users']
    
    CBR = CatBoostRegressor(
    depth = 8,
    l2_leaf_reg = 30,
    loss_function = RMSE,
    border_count = 254,
    verbose = False,
    random_strength = 0.2,
    task_type = 'CPU',
    n_estimators = 180,
    random_state = 123,
    eta = 0.4
    )

    # TrainTest()

    CBR.fit(X_train,y_train)
    
    # y_train_Pred = RF.predict(X_train)

    user_output = CBR.predict(user_input)

    return user_output  # add metrics_rf to return metrics

    
    
    