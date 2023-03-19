import streamlit as st
import pandas as pd
import plotly.express as px
from FeatureEngineering import FeatureEngineering


def ModelBuilding(display=True):
    
    if display == True:
        
        st.title('Model Building')
        st.write('This is Model Building Page')
            
        data = FeatureEngineering(display=False)['BasicFE']

        with st.expander('Correlation Matrix', expanded=True):

            st.subheader('Correlation Matrix')

            fig = px.imshow(data.corr())
            fig.update_layout(
            width=700,
            height=700
            )

            st.plotly_chart(fig, use_container_width=True)


        data = FeatureEngineering(display=False)['AdvancedFE']

        data_unseen = data.loc[data['Date']>='2012-09-01']
        data = data.loc[data['Date']<'2012-09-01']

# ModelBuilding() #Delete for nav later
    
    
    