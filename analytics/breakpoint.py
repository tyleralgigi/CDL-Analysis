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
        # print(self.team_standings)
        
        player_merged_df = pd.merge(self.team_standings[['id', 'Team', 'Year']], self.breakpoint_players, on=['Team', 'Year'], how='right')
        # print(player_merged_df)
        
        self.allPlayers = self.allPlayers.rename(columns={'tag': 'Player'})
        player_merged_df = pd.merge(player_merged_df, self.allPlayers, on=['Player'], how='left')
        self.player_merged_df = player_merged_df[~player_merged_df[['Player', 'Year']].duplicated(keep='first')]
        self.player_merged_df = self.player_merged_df.rename(columns={'id_x': 'team_id', 'id_y': 'id'})
        
        self.player_merged_df = self.player_merged_df.fillna(0)
        # print(self.player_merged_df)
        # print(self.player_merged_df[self.player_merged_df.isnull().any(axis=1)])
    
    def FeatureEngineering(self):
        #Calculate averages or rates 
        self.player_merged_df.replace(',','', regex=True, inplace=True)
        self.player_merged_df.drop(['Player', 'BP Rating'], axis=1, inplace=True)
        # print(self.player_merged_df.columns)
        aggregated_players = pd.DataFrame()
        
        # Define the conversion dictionary
        convert_dict = {'K/D': float,
                        'Slayer Rating': float,
                        'T.E.S': float,
                        'HP KD': float,
                        'HP K/10M': float,
                        'HP OBJ/10M': float,
                        'HP Eng/10M': float,
                        'HP Maps': float,
                        'SND KD': float,
                        'SND KPR': float,
                        'First Bloods': float,
                        'First Blood %': float,
                        'Plants': float,
                        'Defuses': float,
                        'SND Maps': float,
                        'CTL KD': float,
                        'CTL K/10M': float,
                        'CTL DMG/10M': float,
                        'CTL Eng/10M': float,
                        'Zone Tier Captures': float,
                        'CTL Maps': float,
                        'Non-Traded Kills': float,
                        'Maps': float,
                        }

        # Convert columns using the dictionary
        self.player_merged_df = self.player_merged_df.astype(convert_dict)
        
        aggregated_players[['id', 'team_id', 'Year', 'SlayerRating', 'TES']] = self.player_merged_df[['id', 'team_id', 'Year', 'Slayer Rating', 'T.E.S']]
        aggregated_players['KD'] = self.player_merged_df['K/D'] / self.player_merged_df['Maps']
        aggregated_players['NonTradedKills'] = self.player_merged_df['Non-Traded Kills'] / self.player_merged_df['Maps']
        aggregated_players['HP_KD'] = self.player_merged_df['HP KD'] / self.player_merged_df['HP Maps']
        aggregated_players['HP_K10M'] = self.player_merged_df['HP K/10M'] / self.player_merged_df['HP Maps']
        aggregated_players['HP_OBJ10M'] = self.player_merged_df['HP OBJ/10M'] / self.player_merged_df['HP Maps']
        aggregated_players['HP_Eng10M'] = self.player_merged_df['HP Eng/10M'] / self.player_merged_df['HP Maps']
        aggregated_players['SND_KD'] = self.player_merged_df['SND KD'] / self.player_merged_df['SND Maps']
        aggregated_players['SND_KPR'] = self.player_merged_df['SND KPR'] / self.player_merged_df['SND Maps']
        aggregated_players['SND_FB'] = self.player_merged_df['First Bloods'] / self.player_merged_df['SND Maps']
        aggregated_players['SND_FB_Percent'] = self.player_merged_df['First Blood %'] / self.player_merged_df['SND Maps']
        aggregated_players['SND_OBJ'] = (self.player_merged_df['Plants'] + self.player_merged_df['Defuses']) / self.player_merged_df['SND Maps']
        aggregated_players['CTL_KD'] = self.player_merged_df['CTL KD'] / self.player_merged_df['CTL Maps']
        aggregated_players['CTL_K10M'] = self.player_merged_df['CTL K/10M'] / self.player_merged_df['CTL Maps']
        aggregated_players['CTL_DMG10M'] = self.player_merged_df['CTL DMG/10M'] / self.player_merged_df['CTL Maps']
        aggregated_players['CTL_Eng10M'] = self.player_merged_df['CTL Eng/10M'] / self.player_merged_df['CTL Maps']
        aggregated_players['CTL_Zone_Captures'] = self.player_merged_df['Zone Tier Captures'] / self.player_merged_df['CTL Maps']
        aggregated_players = aggregated_players.rename(columns={'Year': 'year'})
        
        aggregated_team_averages = aggregated_players.groupby(['team_id', 'year']).agg({
            'SlayerRating': 'mean',
            'TES': 'mean',
            'KD': 'mean',
            'NonTradedKills': 'mean',
            'HP_KD': 'mean',
            'HP_K10M': 'mean',
            'HP_OBJ10M': 'mean',
            'HP_Eng10M': 'mean',
            'SND_KD': 'mean',
            'SND_KPR': 'mean',
            'SND_FB': 'mean',
            'SND_FB_Percent': 'mean',
            'SND_OBJ': 'mean',
            'CTL_KD': 'mean',
            'CTL_K10M': 'mean',
            'CTL_DMG10M': 'mean',
            'CTL_Eng10M': 'mean',
            'CTL_Zone_Captures': 'mean',
            # Add more columns as needed
        }).reset_index()
        
        # print(self.team_standings.columns)
        
        self.team_standings = self.team_standings.rename(columns={'id': 'team_id', 'Year': 'year'})
        self.aggregated_team_averages = pd.merge(self.team_standings[['team_id', 'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',  'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %', 'year', 'HP +/-'
                                                                    , 'HP Win %', 'S&D Win %', 'S&D +/-', 'CTL Win %', 'CTL Round +/-']], aggregated_team_averages, on=['team_id', 'year'], how='right')
        # print(self.aggregated_team_averages)
        del aggregated_team_averages
        del self.player_merged_df
        
        self.matches['spreadsheet_id'] = self.matches['spreadsheet_id'].astype(int)
        # Merge Team 1 data
        match_with_team1 = pd.merge(
            self.matches,
            self.aggregated_team_averages,
            left_on=['team_1_id', 'spreadsheet_id'],  # Match team ID and year
            right_on=['team_id', 'year'],            # Columns in aggregated averages
            suffixes=('', '_team1')                  # Add suffix for team 1 stats
        )
        
        # Merge Team 2 data
        match_with_teams = pd.merge(
            match_with_team1,
            self.aggregated_team_averages,
            left_on=['team_2_id', 'spreadsheet_id'],
            right_on=['team_id', 'year'],
            suffixes=('_team1', '_team2')  # Add suffix for Team 2
        )
        
        # Drop redundant columns to avoid confusion
        columns_to_drop = ['team_id_team1', 'year_team1', 'team_id_team2', 'year_team2']
        self.match_with_teams = match_with_teams.drop(columns=columns_to_drop, errors='ignore')
        del match_with_teams
        del match_with_team1
        # print(self.match_with_teams.columns)
        
        relative_metrics = [
            'Total_Series_Wins',
            'Total_Series_Losses', 'Total_Series %',
            'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %',
            'HP +/-', 'HP Win %', 'S&D Win %', 'S&D +/-',
            'CTL Win %', 'CTL Round +/-', 'SlayerRating',
            'TES', 'KD', 'NonTradedKills', 'HP_KD',
            'HP_K10M', 'HP_OBJ10M', 'HP_Eng10M', 'SND_KD',
            'SND_KPR', 'SND_FB', 'SND_FB_Percent',
            'SND_OBJ', 'CTL_KD', 'CTL_K10M', 'CTL_DMG10M',
            'CTL_Eng10M', 'CTL_Zone_Captures']
        
        self.match_with_teams = self.match_with_teams.apply(pd.to_numeric, errors='coerce')
        
        # Compute relative metrics
        for metric in relative_metrics:
            self.match_with_teams[f'{metric}_diff'] = (
                self.match_with_teams[f'{metric}_team1'] - self.match_with_teams[f'{metric}_team2']
            )
        columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
        self.match_with_teams = self.match_with_teams.drop(columns=columns_to_drop)
        print(self.match_with_teams)
        
    def init(self):
        print("dBreakpoint init")
        self.dbconnector()
        self.load_data()
        self.FeatureEngineering()
