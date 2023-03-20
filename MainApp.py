import streamlit as st
from ExecutiveSummary import ExecutiveSummary
from DataExploration import DataExploration
from FeatureEngineering import FeatureEngineering
from ModelBuilding import ModelBuilding
from Prediction import Prediction


# Create a dictionary with the different pages and their titles

st.set_page_config(
    page_title="Bike-Sharing Service Analysis",
    page_icon='bike')

pages = {
    '📊 Executive Summary': ExecutiveSummary,
    '🔎 Data Exploration': DataExploration,
    '🛠️ Feature Engineering' : FeatureEngineering,
    '🧪 Model Building': ModelBuilding,
    '🔮 Prediction' : Prediction,
}

st.sidebar.title("Navigation")
# Create a radio button in the sidebar to select a page
page = st.sidebar.radio("Go to", list(pages.keys()))
# Display the selected page with the corresponding function
pages[page](display=True)




