import streamlit as st
import pandas as pd
import plotly.express as px
from FeatureEngineering import FeatureEngineering
from catboost import CatBoostRegressor
from PIL import Image


def ModelBuilding(display=True):
    
    if display == True:
        
        st.title("🚲 Bike-Sharing in Washington D.C")
        
        st.header('🧪 Model Building')
            
        data = FeatureEngineering(display=False)['BasicFE']
        
        st.subheader('The Target')
        
        st.markdown("Before starting any modelling, it is important to understand what the objective of the model is. In this case it is clear:\n > Predict the hourly users of the bike-sharing network of Washington D.C.")
        
        st.subheader('The Method')
        
        st.markdown("In building the best model to predict our target variable, this report uses the Python library `pycaret` to carry out its model building, fitting, tuning and predicting. The steps taken below are followed by the corresponding code as well as the relevant explanations of the decisions taken")
        
        st.subheader('1. Checking the Correlation Matrix')
        
        st.markdown("The first step when building the model is to understand what features are correlated, in order to avoid multicolinearity in the model. That is why it's important to visualize the correlation matrix.\n In this scenario, it only makes sense to look at the original features, as there are some of the new features that will be correlated, but will bring other relevance. Since the model used below is a tree-based model, correlation between features is not hindering the model's performance, due to the feature selection naturally performed.")

        with st.expander('Correlation Matrix', expanded=False):
            
            fig = px.imshow(data.corr())
            fig.update_layout(
            width=700,
            height=700
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(" - `Temperature` \ `Temperature_Feel` are highly correlated, as expected. In the case for this model `Temperature` is taken as the relevant variable because it is more likely to impact the actions of the users, as they are likely to check the daily temperature before leaving to ride a bike. \n - `Total_Users` \ `Registered_Users` \ `Casual_Users` are also very correlated, as the `Total_Users` is a sum of the other two features. In this case, as the information provided is limited in aspects that could distinguish registered from casual users, the model will focus in predicting the total users, making these two features irrelevant. \n - Even though `Season` \ `Month`, `Is_Working_Day` \ `Weekday` and `Day_Period` \ `Hour` are strongly correlated, it would be interesting to understand to each extend is the granualarity of the provided information relevant, or if the model uses more general information to predict the target variable.")
        
        st.subheader("2. Feature Engineering")
        
        st.markdown("The next step is to apply all the decided transformations in the dataset that will be used for modeling, including the creation of new features. Refer to the Feature Engineering section to understand what was done regarding this")
        
        st.subheader("3. Train\Test Split")
        
        st.markdown("After applying all the relevant transformations to the data (which in this case is sensible to do before the train/test split due to their nature), it is important to immediatly split the available data into what is going to be used for training the model, and what is going to be used for testing its performance at the end of the process. \n Given the origin of this problem, it is sensible to split the data based on time, instead of a random split. The predictions this model has to provide will always be farther in the future than any data used in the training process. \n The cuttof was made at `2011-09-01`, allowing for 85% of the data to be used for testing, while 15% is kept for testing")
        
        st.code("data_unseen = data.loc[data['Date']>='2012-09-01'] \ndata = data.loc[data['Date']<'2012-09-01']")
            
        st.subheader("4. Setup")

        st.write("After exploring the data and running the feature engineering, the team ran Pycaret to help select and tune the most optimal model for our solution.The final model selected and used was a CatBoostRegressor which scored a 90% R2 & a RMSE of 68 on our test data. With this level of performance we were satisfied and put the model into production in our streamlit build.  All the action that takes place under-the-hood of pycaret is explained below.")
        
        
        st.code("model = setup(\n\t# Basic options\n\tdata = data,\n\ttarget = target[0],\n\ttrain_size = 0.8,\n\tcategorical_features = cat_cols,\n\tnumeric_features = num_cols,\n\tpreprocess = False,\n\tfold_strategy = 'kfold',\n\tfold = 10,\n\tdata_split_shuffle = True,\n\t# Feature Selection \n\tfeature_selection = True,\n\tfeature_selection_threshold = 0.1,\n\tfeature_selection_method = 'classic',\n\t# Paralellization options\n\tn_jobs = -1,\n\tuse_gpu = True,\n\t# Randomness Control\n\tsession_id = 123")
        
        st.subheader("4. Optimization Target")    
        
        st.write("The model selection was optimized to minimize RMSE (root mean squared error) as this was identified as the metric best used to evaluate the performance of a model for this particular case; we want to know the exact number of error in the most understandable way. The error in this case would be how many bikes away from the actual that our model is predicting, this will help all audiences understand the performance of our model.")
        
        st.code("model = compare_models(\n\tsort='RMSE',\n\tfold=10,\n\tn_select = 1)")
                 
        st.subheader("5. Model Selection")   
        
        st.write("The team ran Pycaret to help select and tune the most optimal model for our solution. Pycaret runs and cross validates (in our case 10 folds) several different regression models ranging from Random Forest to Linear Regression to Catboost. In the end our three best models were Catboost, Extra Trees Regression and Random Forest Regression.")
        
        st.code("model = compare_models(\n\tsort='RMSE',\n\tfold=10,\n\tn_select = 1")
                    
        models = [
            (1, 'CatBoost Regressor', 21.6047, 1206.7916, 34.6447, 0.9591, 0.3962, 0.3782),
            (2, 'Extra Trees Regressor', 23.1078, 1450.3404, 37.9841, 0.9508, 0.3310, 0.2969),
            (3, 'Random Forest Regressor', 25.3069, 1669.5958, 40.7794, 0.9434, 0.3580, 0.3399),
            (4, 'Decision Tree Regressor', 34.5092, 3308.1558, 57.4242, 0.8878, 0.4598, 0.4064),
            (5, 'Gradient Boosting Regressor', 41.4490, 3568.0958, 59.6854, 0.8791, 0.6280, 0.8425),
            (6, 'Ridge Regression', 69.3209, 8791.6268, 93.7332, 0.7020, 0.9819, 2.4206),
            (7, 'Linear Regression', 69.3213, 8791.7281, 93.7337, 0.7020, 0.9822, 2.4203),
            (8, 'Bayesian Ridge', 69.3220, 8793.5071, 93.7431, 0.7019, 0.9816, 2.4255),
            (9, 'Lasso Regression', 70.1050, 8983.2452, 94.7521, 0.6955, 0.9943, 2.4915),
            (10, 'AdaBoost Regressor', 83.9256, 9675.5359, 98.3270, 0.6717, 1.2688, 4.4599),
            (11, 'Elastic Net', 71.2367, 9756.2784, 98.7414, 0.6693, 0.9473, 2.1682),
            (12, 'Orthogonal Matching Pursuit', 74.2535, 9792.5423, 98.9305, 0.6681, 1.0229, 2.6742),
            (13, 'Huber Regressor', 75.5203, 11799.5393, 108.5685, 0.6003, 0.8825, 1.6440),
            (14, 'K Neighbors Regressor', 76.2122, 12563.3223, 112.0297, 0.5740, 0.7815, 1.1802),
            (15, 'Lasso Least Angle Regression', 123.8996, 25140.3866, 158.4990, 0.1482, 1.4871, 6.6880),
            (16, 'Dummy Regressor', 134.9497, 29534.2622, 171.8017, -0.0009, 1.5630, 7.7376),
            (17, 'Passive Aggressive Regressor', 130.8272, 34466.7974, 175.6653, -0.1464, 1.2910, 4.4137),
            (18, 'Least Angle Regression', 926.3119, 5345802.9510, 1158.7873, -167.1976, 2.2208, 47.4698)
        ]
        
        columns = ['Rank','Model','MAE','MSE','RMSE','R2','RMSLE','MAPE']
        
        df = pd.DataFrame(models, columns=columns)
        df.index = df['Rank']
        
        options = [x[1] for x in models] 
        
        selected_models = st.multiselect(label = 'Compare Models', options=options, default = options[0])
        
        df_filtered = df.loc[df['Model'].isin(selected_models)]
        
        st.dataframe(df_filtered, use_container_width=True)
                 
        st.write("Catboost is a gradient boosting algorithm that uses a combination of techniques to build an ensemble of decision trees. The performance of the model is evaluated using a loss function and the errors made are used to update the model. Once updated, another decision tree is then built and the process is repeated until a certain number of trees have been built or until the performance of the model stops improving. ")
                 
        st.subheader("6. Hyperparameter Tuning")  
        
        st.write("Once we were satisfied with a particular model it was important to tune the hyperparameters to maximize the peformance of the model. Pycaret helps establish the most optimal model through a RandomSearchCV, which randomly goes through a list of hyperparameters, cross validates the results and then selects the optimal parameters. The parameters obtained here were then used in streamlit to replicate the Pycaret results. Let us run through some of the parameters selected:")
        
        st.code("tuned_model = tune_model(\n\tmodel,\n\toptimize = 'RMSE',\n\tfold = 10,\n\tsearch_algorithm = 'random',\n\tn_iter = 100)")
                 
        st.write("depth = 8: The maximum depth of each decision tree in the ensemble. Increasing the depth can improve model performance, but also increases the risk of overfitting.")               
        
        st.write("l2_leaf_reg = 30: The L2 regularization coefficient. This is used to control the amount of regularization applied to the weights of the model. Increasing this hyperparameter can reduce overfitting, increasing it too much however can reduce model performance") 

        st.write("loss_function = RMSE: The loss function used to optimize the model. In this case, the root mean squared error (RMSE) is being used.") 

        st.write("random_strength = 0.2: The amount of randomization to apply to the weights of the model. This can help to prevent overfitting and improve generalization.") 

        st.write("border_count = 254: The number of splits to use for numeric features. Increasing this hyperparameter can improve model performance, but may also increase training time and memory usage.")
                 
        st.write("n_estimators = 180: The number of decision trees in the ensemble. Increasing this hyperparameter can improve model performance, but also increases training time and memory usage.")
                 
        st.write("eta = 0.4: The learning rate used by the gradient boosting algorithm. This hyperparameter controls the step size used during each iteration of the algorithm. Increasing this hyperparameter can speed up training, but may also decrease model performance.")

        st.subheader("7. Evaluating on Test")  

        st.write("The last step of the model building process is to predict on our test data and evaluate the performance of the fitted model. In our case our test data was the data we had split at the start of the process (all data after September 1,2012). It is important that our results do not different too much from what we obtained in our training data to ensure we do not have a problem of overfitting. In our case, the model performed similarly with a R2 of 90% on test compared to the 95% on train it was getting. While there may be some very slight overfitting, we were still content with the predictive power of the model.")
        
        prediction = [('0','CatBoost Regressor', 64.7970, 8285.9270, 91.0271, 0.8251, 0.7909, 1.3820)]
        
        df_pred = pd.DataFrame(prediction, columns=columns)
        df_pred.index = df_pred['Rank']
        
        st.code("prediction = predict_model(model, data_unseen)")
        
        st.dataframe(df_pred, use_container_width=True)
        
        with st.expander('Feature Importance Plot', expanded=False):
            
            image = Image.open('feature_importance.png')

            st.image(image, caption='Feature Importance')
        
        st.subheader('8. Results')
        st.write('This is the prediction on the unseen data')
        
        pred = pd.read_csv('prediction.csv')
                
        freq = st.selectbox(
            label='Time Dimension',
            options=[
                'Hourly',
                'Daily',
                'Weekly',
                'Monthly'
            ]
        )
        
        if freq == 'Hourly':
            freq_str = 'TimeStamp'
        elif freq == 'Daily':
            freq_str = 'Date'
        else:
            freq_str = freq[:-2]
        
        by_freq = pred.groupby([freq_str])['Actual','Prediction'].sum()
        by_freq.reset_index(inplace=True)
        
        pred_fig = px.line(by_freq, x=freq_str, y=['Actual', 'Prediction'], color_discrete_sequence=['blue', 'green'],
              title=f'{freq} Number of Users (Actual vs Prediction)')
        
        st.plotly_chart(pred_fig, use_container_width=True)
        

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

    ignore_cols = [
        'Date',
        'Hour',
        'Day',
        'Month',
        'Weekday',
        'Week',
        'Temperature_Feel',
        'Registered_Users',
        'Casual_Users'
    ]

    other_cols = cat_cols + ignore_cols + target

    num_cols = [x for x in data.columns if x not in other_cols]

    data_unseen = data.loc[data['Date']>='2012-09-01']
    data = data.loc[data['Date']<'2012-09-01']
    
    data.drop(columns=ignore_cols,inplace=True)
    data_unseen.drop(columns=ignore_cols,inplace=True)

    X_train = data[[*cat_cols,*num_cols]]
    y_train = data['Total_Users']

    X_test = data_unseen[[*cat_cols,*num_cols]]
    y_test = data_unseen['Total_Users']
    
    CBR = CatBoostRegressor(
    depth = 8,
    l2_leaf_reg = 30,
    loss_function = 'RMSE',
    border_count = 254,
    verbose = False,
    random_strength = 0.2,
    task_type = 'CPU',
    n_estimators = 180,
    random_state = 123,
    eta = 0.4
    )

    CBR.fit(X_train,y_train)

    user_output = CBR.predict(user_input)

    return user_output  # add metrics_rf to return metrics

    
    
    