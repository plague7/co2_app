import streamlit as st
import numpy as np
import app.Utils as Utils
import pandas as pd
import numpy as np


def predict(model, data):
    # TO DO -> change type of data (float?)
    data = np.array(data, dtype='float64').reshape(1, -1)
    print('Data shape :', data.shape)
    return model.predict(data)


def init():
    st.title('Demo')

    df = Utils.load_data()
    model = Utils.loadModel()

    features_list = df.columns
    #n_features = len(features_list)

    with st.expander('Fill in your data :'):
        for feature in features_list:
            feature_var = '_feature_{}'.format(feature)
            globals()[feature_var] = st.text_input(label=feature)

    launch_prediction_btn = st.button(label='Predict')

    if launch_prediction_btn:
        features_values = [v for k, v in globals().items()
                           if k.startswith('_feature_')]

        if not any(v == '' for v in features_values):
            df_dtypes = dict(df.dtypes)

            for i, l in enumerate(features_list):
                v_dtype = df_dtypes[l]
                if v_dtype == 'float64':
                    v = features_values[i]
                    features_values[i] = np.float64(v)

            y_pred = predict(model, features_values)

            unit = 'kBtu'
            y_pred_str = str(y_pred) + ' ' + unit
            st.write('Prediction ' + y_pred_str)
        else:
            st.write('Please provide a value for all the features.')
