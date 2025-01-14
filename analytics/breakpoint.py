import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import sys

class Breakpoint():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def load_data(self):
        #Reading CSVs
        data = pd.read_excel("data/breakpoint_players.xlsx", sheet_name=None)
        df_sheet1 = data["2025 Players"]
        print(df_sheet1)

    def init(self):
        print("data analysis 1 init")
        self.dbconnector()
        self.load_data()
