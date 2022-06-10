import pandas as pd
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')


def load_data():
    return pd.read_csv(DATA_PATH)


def loadModel():
    with open(MODEL_PATH, 'rb') as path:
        return pickle.load(path)
