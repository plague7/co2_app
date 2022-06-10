# Manip data
import pandas as pd

# SKLEARN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import os
import os.path
import pickle

import streamlit as st

def load_data():
    csv = os.path.join(os.path.dirname('__file__'), '2016_Building_Energy_Benchmarking.csv')
    df = pd.read_csv(csv)
    return df

def load_model():
    lr = os.path.join(os.path.dirname('__file__'), 'baseline_lr_model.sav')
    with open(lr , 'rb') as handle:
        lr_baseline = pickle.load(handle)
    return lr_baseline

def preprocessing(df):
    # Drop colonnes useless à 1ere vue
    df_dummy = df.drop(["BuildingType","PrimaryPropertyType","LargestPropertyUseType","ListOfAllPropertyUseTypes","TaxParcelIdentificationNumber","Neighborhood","OSEBuildingID", "DataYear", "PropertyName", "Address", "City", "State", "ZipCode", "Outlier","ComplianceStatus", "Comments", "DefaultData", "YearsENERGYSTARCertified", 'ThirdLargestPropertyUseType', 'ThirdLargestPropertyUseTypeGFA', 'SecondLargestPropertyUseType', 'SecondLargestPropertyUseTypeGFA', "GHGEmissionsIntensity", "TotalGHGEmissions", 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'Electricity(kWh)', 'Electricity(kBtu)'], axis = 1 )

    # Remplacer par moyenne
    df_dummy["ENERGYSTARScore"] = round(df_dummy['ENERGYSTARScore'].fillna(df_dummy['ENERGYSTARScore'].mean()))

    # Dropna restant
    df_dummy.dropna(axis=0, inplace = True)
    df_dummy.reset_index(drop=True, inplace=True)

    return df_dummy

df = load_data()
df = preprocessing(df)
model = load_model()
y_pred = model.predict(df)

### Header ###
st.title('CO2 Prediction App')
st.write('Test')
st.write(y_pred)
