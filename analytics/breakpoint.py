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
        print("Loading data")
        #Reading CSVs
        data = pd.read_excel("data/breakpoint_data.xlsx", sheet_name=None)
        self.breakpoint_players = data["Players"]
        self.breakpoint_teams = data["Teams"]

        #Getting Data From Postgres Tables
        with self.engine.connect() as conn:
            result = conn.execute(text(f'select * from public."playerStatsDetails"'))
            self.playerStatsDetails_current = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f"select team_1_id, team_2_id, winner_id, team_1_score, team_2_score, spreadsheet_id from public.matches_matches where winner_id != 'nan'"))
            self.matches = pd.DataFrame(result.fetchall())
            self.matches['spreadsheet_id'] = self.matches['spreadsheet_id'].str[:4]
            
            result = conn.execute(text(f'select id, name_short, name from public.matches_teams'))
            self.standing_current = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select id, tag from public."matches_allPlayers"'))
            self.allPlayers = pd.DataFrame(result.fetchall())
            
        self.transform()

    def transform(self):
        merged_df = pd.merge(self.standing_current, self.breakpoint_teams, left_on='name_short', right_on='Team', how='right')
        self.team_standings = merged_df[~merged_df[['name_short', 'Year']].duplicated(keep='first')]
        self.team_standings.drop('name_short', axis=1, inplace=True)
        self.team_standings.reset_index()
        print(self.team_standings)
        
        player_merged_df = pd.merge(self.team_standings[['id', 'Team', 'Year']], self.breakpoint_players, on=['Team', 'Year'], how='right')
        print(player_merged_df)
        
        self.allPlayers = self.allPlayers.rename(columns={'tag': 'Player'})
        player_merged_df = pd.merge(player_merged_df, self.allPlayers, on=['Player'], how='left')
        self.player_merged_df = player_merged_df[~player_merged_df[['Player', 'Year']].duplicated(keep='first')]
        self.player_merged_df = self.player_merged_df.rename(columns={'id_x': 'team_id', 'id_y': 'id'})
        print(self.player_merged_df)
        
    def init(self):
        print("dBreakpoint init")
        self.dbconnector()
        self.load_data()
