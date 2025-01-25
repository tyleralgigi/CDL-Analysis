import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import sys
import requests
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

class Breakpoint():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def add_noise(self,X, noise_level=0.01):
        noise = np.random.normal(0, noise_level, size=X.shape)
        return X + noise
    
    def load_data(self):
        print("Loading data")
        #Reading CSVs
        # data = pd.read_excel("data/breakpoint_data.xlsx", sheet_name=None)
        # self.breakpoint_players = data["Players"]
        # self.breakpoint_teams = data["Teams"]
        advanced_json = requests.get('https://www.breakingpoint.gg/_next/data/93XP0w20UMsR0XR20IgVN/stats/advanced.json')
        advanced_json_data = advanced_json.json()
        self.advanced_all_players = pd.DataFrame(advanced_json_data['pageProps']['allPlayers'])
        self.advanced_all_teams = pd.DataFrame(advanced_json_data['pageProps']['allTeams'])
        self.advanced_all_teams.rename(columns={'id': 'team_id'}, inplace=True)
        # print(self.advanced_all_players[['tag','current_team_id']])
        
        #Getting Data From Postgres Tables
        with self.engine.connect() as conn:
            # result = conn.execute(text(f'select * from public."playerStatsDetails"'))
            # self.playerStatsDetails_current = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f"select team_1_id, team_2_id, winner_id, team_1_score, team_2_score, spreadsheet_id from public.matches_matches where winner_id != 'nan'"))
            self.matches = pd.DataFrame(result.fetchall())
            self.matches['spreadsheet_id'] = self.matches['spreadsheet_id'].str[:4]
            
            result = conn.execute(text(f'select id, name_short, name from public.matches_teams'))
            self.standing_current = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select id, tag from public."matches_allPlayers"'))
            self.allPlayers = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select tag, team_id from public.team_rosters'))
            self.team_rsoters = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select * from public.breakpoint_advanced_stats_standings'))
            self.breakpoint_standings = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select * from public.breakpoint_advanced_stats_teams'))
            self.breakpoint_teams = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select * from public.breakpoint_advanced_stats_players'))
            self.breakpoint_players = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select * from public."playerRollingAvg"'))
            self.rolling_averages = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f'select id, name_short from public.matches_teams'))
            self.all_teams = pd.DataFrame(result.fetchall())
            self.all_teams.rename(columns={'id': 'team_id'}, inplace=True)
        self.transform()

    def transform(self):
        # merged_df = pd.merge(self.standing_current, self.breakpoint_teams, left_on='name_short', right_on='Team', how='right')
        # self.team_standings = merged_df[~merged_df[['name_short', 'Year']].duplicated(keep='first')]
        # self.team_standings.drop('name_short', axis=1, inplace=True)
        # self.team_standings.reset_index()
        # print(self.team_standings)
        
        #Merging Breakpoint Advancded Team Stats and Team Standings
        standing_2025 = self.breakpoint_standings.loc[self.breakpoint_standings['year'] == 2025]
        standing_2024 = self.breakpoint_standings.loc[self.breakpoint_standings['year'] == 2024]
        standing_2025 = pd.merge(self.standing_current, standing_2025, left_on='name', right_on='Team', how='right')
        standing_2024 = pd.merge(self.standing_current, standing_2024, left_on='name_short', right_on='Team', how='right')
        self.team_standings = pd.concat([standing_2025, standing_2024], axis=0)
        self.team_standings.drop(['Team','Rank', 'Points', 'Team'], axis=1, inplace=True)
        del standing_2025
        del standing_2024
        del self.breakpoint_standings
        
        # self.team_standings.drop(['#'], axis=1, inplace=True)
        teams_2025 = self.breakpoint_teams.loc[self.breakpoint_teams['year'] == 2025]
        teams_2024 = self.breakpoint_teams.loc[self.breakpoint_teams['year'] == 2024]
        teams_2025 = pd.merge(self.advanced_all_teams[['name_short', 'team_id']], teams_2025, left_on='name_short', right_on='Team', how='right')
        self.breakpoint_teams = pd.concat([teams_2025, teams_2024], axis=0)
        self.breakpoint_teams['team_id'] = self.breakpoint_teams['team_id'].fillna( self.breakpoint_teams['team_id_x'])
        self.breakpoint_teams.drop(['name_short', 'team_id_x', 'team_id_y', '#'], axis=1, inplace=True)
        self.team_standings['id'] = self.team_standings['id'].astype(int)
        self.team_standings = pd.merge(self.team_standings, self.breakpoint_teams, left_on='id', right_on='team_id', how='right')
        # self.team_standings = pd.concat([standing_2025, self.breakpoint_teams], axis=0)
        self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']] = self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']].apply(lambda col: col.str.replace('%', ''))
        self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']] = self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']].astype(float)
        self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']] = self.team_standings[['MW%','GW%','HP Win %','S&D Win %','CTL Win %']].apply(lambda x: x / 100)
        self.team_standings.drop(['name_short','name','id', 'year_x'], axis=1, inplace=True)
        self.team_standings.rename(columns={'year_y': 'year'}, inplace=True)
        del teams_2024
        del teams_2025
        del self.breakpoint_teams
        
        #Merge Players with Current Teams and Team Standings
        players_2025 = pd.merge(self.team_rsoters, self.breakpoint_players.loc[self.breakpoint_players['year'] == 2025], left_on='tag', right_on='Player', how='right')
        missing_players = players_2025[players_2025['team_id'].isna()]
        missing_players = pd.merge(missing_players, self.advanced_all_players[['tag','current_team_id']], left_on='Player', right_on='tag', how='left')
        missing_players.drop(['team_id','tag_x','tag_y'], axis=1, inplace=True)
        missing_players.rename(columns={'current_team_id': 'team_id'}, inplace=True)
        missing_players = missing_players[missing_players['team_id'].notna()]
        players_2025 = players_2025[players_2025['team_id'].notna()]
        players_2025 = pd.concat([players_2025, missing_players], axis=0)
        
        players_2024 = pd.merge(self.all_teams, self.breakpoint_players.loc[self.breakpoint_players['year'] == 2024], left_on='name_short', right_on='Team', how='right')
        self.players = pd.concat([players_2025, players_2024], axis=0)
        self.players.drop(['tag','Team', 'name_short'], axis=1, inplace=True)
        # print(self.players)
        del self.breakpoint_players
        del missing_players
        del players_2025
        del players_2024
        
        #Adding rolling averages to players_2025
        rolling_averages_2025 = pd.merge(self.rolling_averages, self.players.loc[self.players['year'] == 2025], left_on='teamId', right_on='team_id', how='right')
        self.players = pd.concat([rolling_averages_2025, self.players.loc[self.players['year'] == 2024]], axis=0)
        self.players.drop(['teamId'], axis=1, inplace=True)
        print(self.players.columns)
        print(self.team_standings.columns)
        del rolling_averages_2025
        del self.rolling_averages
        # player_merged_df = pd.merge(self.team_standings[['id', 'Team', 'Year']], self.breakpoint_players, on=['Team', 'Year'], how='right')
        # print(player_merged_df)
        
        # self.allPlayers = self.allPlayers.rename(columns={'tag': 'Player'})
        # player_merged_df = pd.merge(player_merged_df, self.allPlayers, on=['Player'], how='left')
        # self.player_merged_df = player_merged_df[~player_merged_df[['Player', 'Year']].duplicated(keep='first')]
        # self.player_merged_df = self.player_merged_df.rename(columns={'id_x': 'team_id', 'id_y': 'id'})
        
        # self.player_merged_df = self.player_merged_df.fillna(0)
        # print(self.player_merged_df)
        # print(self.player_merged_df[self.player_merged_df.isnull().any(axis=1)])
    
    def FeatureEng(self):
        print()
        self.players.drop(['Player', 'Game Time (Min)'], axis=1, inplace=True)
        aggregated_players = pd.DataFrame()
        
        convert_dict = {'ctl_bp_rating_avg': float,
                        'hp_bp_rating_avg': float,
                        'snd_bp_rating_avg': float,
                        'BP Rating': float,
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
                        'First Deaths': float,
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
        
        self.players = self.players.astype(convert_dict)
        
        aggregated_players[['team_id', 'year', 'SlayerRating', 'TES', 'hp_bp_rating_avg', 'ctl_bp_rating_avg', 'snd_bp_rating_avg']] = self.players[['team_id', 'year', 'Slayer Rating', 'T.E.S', 'hp_bp_rating_avg', 'ctl_bp_rating_avg', 'snd_bp_rating_avg']]
        aggregated_players['KD'] = self.players['K/D'] / self.players['Maps']
        aggregated_players['NonTradedKills'] = self.players['Non-Traded Kills'] / self.players['Maps']
        aggregated_players['HP_KD'] = self.players['HP KD'] / self.players['HP Maps']
        aggregated_players['HP_K10M'] = self.players['HP K/10M']
        aggregated_players['HP_OBJ10M'] = self.players['HP OBJ/10M']
        aggregated_players['HP_Eng10M'] = self.players['HP Eng/10M']
        aggregated_players['SND_KD'] = self.players['SND KD'] / self.players['SND Maps']
        aggregated_players['SND_KPR'] = self.players['SND KPR'] / self.players['SND Maps']
        aggregated_players['SND_FB'] = self.players['First Bloods'] / self.players['SND Maps']
        aggregated_players['SND_FD'] = self.players['First Deaths'] / self.players['SND Maps']
        aggregated_players['SND_FB_Percent'] = self.players['First Blood %'] / self.players['SND Maps']
        aggregated_players['SND_OBJ'] = (self.players['Plants'] + self.players['Defuses']) / self.players['SND Maps']
        aggregated_players['CTL_KD'] = self.players['CTL KD'] / self.players['CTL Maps']
        aggregated_players['CTL_K10M'] = self.players['CTL K/10M']
        aggregated_players['CTL_DMG10M'] = self.players['CTL DMG/10M']
        aggregated_players['CTL_Eng10M'] = self.players['CTL Eng/10M']
        aggregated_players['CTL_Zone_Captures'] = self.players['Zone Tier Captures'] / self.players['CTL Maps']
        
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
            'SND_FD': 'mean',
            'SND_FB_Percent': 'mean',
            'SND_OBJ': 'mean',
            'CTL_KD': 'mean',
            'CTL_K10M': 'mean',
            'CTL_DMG10M': 'mean',
            'CTL_Eng10M': 'mean',
            'CTL_Zone_Captures': 'mean',
            'hp_bp_rating_avg': 'first',
            'ctl_bp_rating_avg': 'first',
            'snd_bp_rating_avg': 'first'
            # Add more columns as needed
        }).reset_index()
        
        self.team_standings['team_id'] = self.team_standings['team_id'].astype(float)
        aggregated_team_averages['team_id'] = aggregated_team_averages['team_id'].astype(float)
        
        self.aggregated_team_averages = pd.merge(self.team_standings[['team_id', 'year', 'MW', 'ML', 'MW%', 'GW', 'GL', 'GW%', 'Team', 'K/D', 'HP Win %', 'HP K/D', 'HP Score', 'HP +/-', 'S&D Win %', 'S&D K/D','S&D Round Wins', 'S&D +/-', 'CTL Win %', 'CTL K/D', 'CTL Round Wins','CTL Round +/-']], aggregated_team_averages, on=['team_id', 'year'], how='left')
        self.aggregated_team_averages = self.aggregated_team_averages.drop_duplicates( 
                                                    subset = ['team_id', 'year'], 
                                                    keep = 'last').reset_index(drop = True) 
        del aggregated_team_averages
        
        self.matches['spreadsheet_id'] = self.matches['spreadsheet_id'].astype(int)
        self.matches['team_1_id'] = self.matches['team_1_id'].astype(int)
        self.matches['team_2_id'] = self.matches['team_2_id'].astype(int)
        # Merge Team 1 datas
        
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
        print(self.match_with_teams.columns)
        
        relative_metrics = [
            'MW', 'ML', 'MW%', 'GW',
            'GL', 'GW%', 'K/D', 'HP Win %',
            'HP K/D', 'HP Score', 'HP +/-', 'S&D Win %',
            'S&D K/D', 'S&D Round Wins', 'S&D +/-',
            'CTL Win %', 'CTL K/D', 'CTL Round Wins',
            'CTL Round +/-', 'SlayerRating', 'TES', 'KD',
            'NonTradedKills', 'HP_KD', 'HP_K10M',
            'HP_OBJ10M', 'HP_Eng10M', 'SND_KD', 'SND_KPR',
            'SND_FB', 'SND_FD', 'SND_FB_Percent', 'SND_OBJ',
            'CTL_KD', 'CTL_K10M', 'CTL_DMG10M',
            'CTL_Eng10M', 'CTL_Zone_Captures', 'hp_bp_rating_avg',
            'ctl_bp_rating_avg', 'snd_bp_rating_avg']
        
        self.match_with_teams = self.match_with_teams.apply(pd.to_numeric, errors='coerce')
        
        # Compute relative metrics
        for metric in relative_metrics:
            self.match_with_teams[f'{metric}_diff'] = (
                self.match_with_teams[f'{metric}_team1'] - self.match_with_teams[f'{metric}_team2']
            )
        columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
        self.match_with_teams = self.match_with_teams.drop(columns=columns_to_drop)
        print(self.match_with_teams)
        
        # Columns to normalize
        columns_to_normalize = [
            'MW_diff', 'ML_diff', 'MW%_diff', 'GW_diff',
            'GL_diff', 'GW%_diff', 'K/D_diff', 'HP Win %_diff',
            'HP K/D_diff', 'HP Score_diff', 'HP +/-_diff', 'S&D Win %_diff',
            'S&D K/D_diff', 'S&D Round Wins_diff', 'S&D +/-_diff',
            'CTL Win %_diff', 'CTL K/D_diff', 'CTL Round Wins_diff',
            'CTL Round +/-_diff', 'SlayerRating_diff', 'TES_diff', 'KD_diff',
            'NonTradedKills_diff', 'HP_KD_diff', 'HP_K10M_diff',
            'HP_OBJ10M_diff', 'HP_Eng10M_diff', 'SND_KD_diff', 'SND_KPR_diff',
            'SND_FB_diff', 'SND_FD_diff', 'SND_FB_Percent_diff', 'SND_OBJ_diff',
            'CTL_KD_diff', 'CTL_K10M_diff', 'CTL_DMG10M_diff',
            'CTL_Eng10M_diff', 'CTL_Zone_Captures_diff', 'hp_bp_rating_avg_diff',
            'ctl_bp_rating_avg_diff', 'snd_bp_rating_avg_diff'
        ]
        
        # Initialize the scaler
        self.scaler = MinMaxScaler()
        
        # Apply MinMaxScaler to the selected columns
        self.match_with_teams[columns_to_normalize] = self.scaler.fit_transform(self.match_with_teams[columns_to_normalize])
        self.match_with_teams['winner_id'] = (self.match_with_teams['winner_id'] == self.match_with_teams['team_2_id']).astype(int)
        self.match_with_teams['spreadsheet_id'] = self.match_with_teams['spreadsheet_id'].astype('category')

        self.match_with_teams = self.match_with_teams.drop(columns=['team_1_id', 'team_2_id', 'Team_team1', 'Team_team2'])
        self.match_with_teams = self.match_with_teams.rename(columns={"spreadsheet_id": "year"})

        print(self.match_with_teams)
    
    def train_modal_1(self):
        print("training")
        # Prepare features and target
        X = self.match_with_teams.drop(columns=['team_1_score', 'team_2_score', 'winner_id'])
        y_winner = self.match_with_teams['winner_id']  # Classification target

        # Save training features (categorical features are part of this)
        self.training_features = X.columns.tolist()

        print('feautres')
        print(self.training_features)
        
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_winner, test_size=0.15, random_state=42
        )

        # Specify categorical features
        categorical_features = self.training_features

        # LightGBM parameter grid for hyperparameter tuning
        param_dist = {
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500],  # High value for early stopping
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [0.1, 1, 3],
            'min_gain_to_split': [0.1],  # Prevent insignificant splits
        }

        # Initialize LightGBM classifier
        lgbm = LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            boosting_type='gbdt',
            verbose=-1  # Suppress training noise
        )

        # Randomized Search with Stratified K-Fold
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_dist,
            n_iter=50,
            scoring='accuracy',
            cv=cv_strategy,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        # Fit the randomized search
        random_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],         # Validation set for monitoring
            eval_metric='accuracy',           # Metric to optimize
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(10)
            ]
        )

        # Best model
        best_lgbm = random_search.best_estimator_

        # Save the best model
        self.best_model = best_lgbm

        # Predictions and evaluation
        y_pred = best_lgbm.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        print("Accuracy with LightGBM:", accuracy)
        print("Classification Report:")
        print(classification_report(y_val, y_pred))

    def prediction1(self):
        # Load upcoming games and preprocess
        self.upcoming_games = pd.read_csv("data/upcoming_games.csv", header=0)
        self.upcoming_games['year'] = '2025'

        # Map team1_name and team2_name to their respective IDs
        team_id_map = dict(zip(self.standing_current['name_short'], self.standing_current['id']))
        self.upcoming_games['team1_id'] = self.upcoming_games['team_1_name'].map(team_id_map)
        self.upcoming_games['team2_id'] = self.upcoming_games['team_2_name'].map(team_id_map)
        self.upcoming_games['year'] = self.upcoming_games['year'].astype(int)

        self.upcoming_games['team1_id'] = self.upcoming_games['team1_id'].astype(int)
        self.upcoming_games['team2_id'] = self.upcoming_games['team2_id'].astype(int)
        
        # Prepare stats for upcoming games
        match_with_team1 = pd.merge(
            self.upcoming_games,
            self.aggregated_team_averages,
            left_on=['team1_id', 'year'],
            right_on=['team_id', 'year'],
            suffixes=('', '_team1'),
            how='left'
        )
        match_with_team2 = pd.merge(
            match_with_team1,
            self.aggregated_team_averages,
            left_on=['team2_id', 'year'],
            right_on=['team_id', 'year'],
            suffixes=('_team1', '_team2'),
            how='left'
        )

        # Concatenate and clean
        upcoming_matches_stats = match_with_team2.drop(columns=['team1_id', 'team2_id', 'team_id_team1', 'team_id_team2'], errors='ignore')

        # Compute relative metrics
        relative_metrics = [
            'MW', 'ML', 'MW%', 'GW',
            'GL', 'GW%', 'K/D', 'HP Win %',
            'HP K/D', 'HP Score', 'HP +/-', 'S&D Win %',
            'S&D K/D', 'S&D Round Wins', 'S&D +/-',
            'CTL Win %', 'CTL K/D', 'CTL Round Wins',
            'CTL Round +/-', 'SlayerRating', 'TES', 'KD',
            'NonTradedKills', 'HP_KD', 'HP_K10M',
            'HP_OBJ10M', 'HP_Eng10M', 'SND_KD', 'SND_KPR',
            'SND_FB', 'SND_FD', 'SND_FB_Percent', 'SND_OBJ',
            'CTL_KD', 'CTL_K10M', 'CTL_DMG10M',
            'CTL_Eng10M', 'CTL_Zone_Captures', 'hp_bp_rating_avg',
            'ctl_bp_rating_avg', 'snd_bp_rating_avg'
        ]
        for metric in relative_metrics:
            upcoming_matches_stats[f'{metric}_diff'] = (
                upcoming_matches_stats[f'{metric}_team1'] - upcoming_matches_stats[f'{metric}_team2']
            )
        columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
        upcoming_matches_stats = upcoming_matches_stats.drop(columns=columns_to_drop, errors='ignore')

        # Normalize numeric columns
        columns_to_normalize = [
            f'{metric}_diff' for metric in relative_metrics
        ]
        upcoming_matches_stats[columns_to_normalize] = self.scaler.transform(upcoming_matches_stats[columns_to_normalize])

        # Ensure categorical feature alignment
        categorical_features = [  # Add your categorical features here
            'year'
        ]
        for feature in categorical_features:
            if feature in upcoming_matches_stats.columns:
                upcoming_matches_stats[feature] = upcoming_matches_stats[feature].astype('category')

        # Align features with training
        upcoming_matches_stats = upcoming_matches_stats[self.training_features]

        # Simulate predictions
        n_simulations = 10000
        simulated_winner_predictions = []

        for i in range(n_simulations):
            sys.stdout.write(f"\rSimulation {i + 1}/{n_simulations}")
            sys.stdout.flush()

            # Add small random noise to numeric features
            numeric_features = upcoming_matches_stats.select_dtypes(include=['float64', 'int64']).columns
            noisy_numeric = upcoming_matches_stats[numeric_features].copy()
            noise = np.random.normal(0, 0.05, size=noisy_numeric.shape)
            noisy_numeric += noise

            # Combine numeric and categorical features
            noisy_features = pd.concat([noisy_numeric, upcoming_matches_stats[categorical_features]], axis=1)

            # Align with training features
            noisy_features = noisy_features[self.training_features]

            # Predict winners
            winner_predictions = self.best_model.predict(noisy_features)
            simulated_winner_predictions.append(winner_predictions)

        # Aggregate simulation results
        simulated_winner_predictions = np.array(simulated_winner_predictions)
        team_1_win_prob = (simulated_winner_predictions == 0).mean(axis=0)
        team_2_win_prob = (simulated_winner_predictions == 1).mean(axis=0)

        upcoming_matches_stats['team_1_win_prob'] = team_1_win_prob
        upcoming_matches_stats['team_2_win_prob'] = team_2_win_prob
        upcoming_matches_stats['winner_pred'] = np.where(team_1_win_prob > team_2_win_prob, 0, 1)

        print("\nPrediction Results:")
        print(upcoming_matches_stats[['team_1_win_prob', 'team_2_win_prob', 'winner_pred']])

    # def train_modal_1(self):
    #     X = self.match_with_teams.drop(columns=['team_1_score', 'team_2_score', 'winner_id'])
    #     y_winner = self.match_with_teams['winner_id']  # Classification target
        
    #     # Remove low-variance features
    #     # low_variance_features = X.columns[X.var() == 0].tolist()
    #     # if low_variance_features:
    #     #     print("Dropping low-variance features:", low_variance_features)
    #     #     X = X.drop(columns=low_variance_features)

    #     self.training_features = X.columns.tolist()

    #     # Train-test split
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y_winner, test_size=0.15, random_state=42
    #     )

    #     # Updated parameter grid
    #     param_dist = {
    #         'num_leaves': [31, 63, 127],
    #         'max_depth': [10, 15, 20],
    #         'learning_rate': [0.01, 0.05, 0.1],
    #         'n_estimators': [500],  # High value for early stopping
    #         'subsample': [0.7, 0.8, 1.0],
    #         'colsample_bytree': [0.7, 0.9, 1.0],
    #         'min_child_weight': [0.1, 1, 3],
    #         'min_gain_to_split': [0.1],  # Prevent insignificant splits
    #     }

    #     lgbm = LGBMClassifier(random_state=42, class_weight='balanced', boosting_type='gbdt',verbose=-1)

    #     cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    #     random_search = RandomizedSearchCV(
    #         estimator=lgbm,
    #         param_distributions=param_dist,
    #         n_iter=50,
    #         scoring='accuracy',
    #         cv=cv_strategy,
    #         verbose=2,
    #         random_state=42,
    #         n_jobs=-1
    #     )

    #     # Fit LightGBM
    #     random_search.fit(
    #         X_train, y_train,
    #         eval_set=[(X_test, y_test)],
    #         eval_metric='accuracy',
    #         callbacks=[
    #             early_stopping(stopping_rounds=50),
    #             log_evaluation(10)
    #         ],
    #         categorical_feature=self.training_features  # Add this parameter
    #     )
    #     best_lgbm = random_search.best_estimator_

    #     # Predictions and evaluation
    #     y_pred = best_lgbm.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)

    #     print("Accuracy with LightGBM:", accuracy)
    #     print("Classification Report:")
    #     print(classification_report(y_test, y_pred))

    #     self.best_model = best_lgbm
        
    #     # Feature importance analysis
    #     # lgb.plot_importance(best_lgbm, max_num_features=20)
    #     # plt.show()

    #     # # Calibrate with Platt scaling
    #     # self.calibrated_classifier = CalibratedClassifierCV(best_rf, method='sigmoid', cv=cv_strategy)
    #     # self.calibrated_classifier.fit(X_train, y_winner_train)

    #     # Evaluate on the test set
    #     # y_pred = self.calibrated_classifier.predict(X_test)
    #     # accuracy = accuracy_score(y_winner_test, y_pred)

    #     # print("Accuracy after tuning and calibration:", accuracy)
    #     # print("Classification Report:")
    #     # print(classification_report(y_winner_test, y_pred))

    # def prediction1(self):
    #     # Load upcoming games and preprocess
    #     self.upcoming_games = pd.read_csv("data/upcoming_games.csv", header=0)
    #     self.upcoming_games['year'] = '2025'

    #     # Map team1_name and team2_name to their respective IDs
    #     team_id_map = dict(zip(self.standing_current['name_short'], self.standing_current['id']))
    #     self.upcoming_games['team1_id'] = self.upcoming_games['team_1_name'].map(team_id_map)
    #     self.upcoming_games['team2_id'] = self.upcoming_games['team_2_name'].map(team_id_map)
    #     self.upcoming_games['year'] = self.upcoming_games['year'].astype(int)

    #     self.upcoming_games['team1_id'] = self.upcoming_games['team1_id'].astype(int)
    #     self.upcoming_games['team2_id'] = self.upcoming_games['team2_id'].astype(int)
        
    #     # Prepare stats for upcoming games
    #     match_with_team1 = pd.merge(
    #         self.upcoming_games,
    #         self.aggregated_team_averages,
    #         left_on=['team1_id', 'year'],
    #         right_on=['team_id', 'year'],
    #         suffixes=('', '_team1'),
    #         how='left'
    #     )
    #     match_with_team2 = pd.merge(
    #         match_with_team1,
    #         self.aggregated_team_averages,
    #         left_on=['team2_id', 'year'],
    #         right_on=['team_id', 'year'],
    #         suffixes=('_team1', '_team2'),
    #         how='left'
    #     )
        
    #     # Concatenate and clean
    #     upcoming_matches_stats = match_with_team2.drop(columns=['team1_id', 'team2_id'], errors='ignore')
        
    #     # Compute relative metrics
    #     relative_metrics = [
    #         'MW', 'ML', 'MW%', 'GW',
    #         'GL', 'GW%', 'K/D', 'HP Win %',
    #         'HP K/D', 'HP Score', 'HP +/-', 'S&D Win %',
    #         'S&D K/D', 'S&D Round Wins', 'S&D +/-',
    #         'CTL Win %', 'CTL K/D', 'CTL Round Wins',
    #         'CTL Round +/-', 'SlayerRating', 'TES', 'KD',
    #         'NonTradedKills', 'HP_KD', 'HP_K10M',
    #         'HP_OBJ10M', 'HP_Eng10M', 'SND_KD', 'SND_KPR',
    #         'SND_FB', 'SND_FD', 'SND_FB_Percent', 'SND_OBJ',
    #         'CTL_KD', 'CTL_K10M', 'CTL_DMG10M',
    #         'CTL_Eng10M', 'CTL_Zone_Captures', 'hp_bp_rating_avg',
    #         'ctl_bp_rating_avg', 'snd_bp_rating_avg']
    #     for metric in relative_metrics:
    #         upcoming_matches_stats[f'{metric}_diff'] = (
    #             upcoming_matches_stats[f'{metric}_team1'] - upcoming_matches_stats[f'{metric}_team2']
    #         )
    #     columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
    #     upcoming_matches_stats = upcoming_matches_stats.drop(columns=columns_to_drop, errors='ignore')
        
    #     # Normalize columns
    #     columns_to_normalize = [
    #         'MW_diff', 'ML_diff', 'MW%_diff', 'GW_diff',
    #         'GL_diff', 'GW%_diff', 'K/D_diff', 'HP Win %_diff',
    #         'HP K/D_diff', 'HP Score_diff', 'HP +/-_diff', 'S&D Win %_diff',
    #         'S&D K/D_diff', 'S&D Round Wins_diff', 'S&D +/-_diff',
    #         'CTL Win %_diff', 'CTL K/D_diff', 'CTL Round Wins_diff',
    #         'CTL Round +/-_diff', 'SlayerRating_diff', 'TES_diff', 'KD_diff',
    #         'NonTradedKills_diff', 'HP_KD_diff', 'HP_K10M_diff',
    #         'HP_OBJ10M_diff', 'HP_Eng10M_diff', 'SND_KD_diff', 'SND_KPR_diff',
    #         'SND_FB_diff', 'SND_FD_diff', 'SND_FB_Percent_diff', 'SND_OBJ_diff',
    #         'CTL_KD_diff', 'CTL_K10M_diff', 'CTL_DMG10M_diff',
    #         'CTL_Eng10M_diff', 'CTL_Zone_Captures_diff', 'hp_bp_rating_avg_diff',
    #         'ctl_bp_rating_avg_diff', 'snd_bp_rating_avg_diff'
    #     ]
    #     upcoming_matches_stats[columns_to_normalize] = self.scaler.transform(upcoming_matches_stats[columns_to_normalize])

    #     # Align with training features
    #     upcoming_matches_stats = upcoming_matches_stats[self.training_features]

    #     # Predict outcomes
    #     # winner_predictions = self.classifier.predict(upcoming_matches_stats)
    #     # upcoming_matches_stats['winner_pred'] = winner_predictions

    #     # print("\nPrediction Results:")
    #     # print(upcoming_matches_stats[['winner_pred']])
        
    #     # Number of simulations
    #     n_simulations = 10000
        
    #     simulated_winner_predictions = [] # Placeholder for predictions
    #     simulated_team_1_scores = []  # Store Team 1 scores
    #     simulated_team_2_scores = []  # Store Team 2 scores
        
    #     # Run predictions 10,000 times
    #     for i in range(n_simulations):
    #         # Print the current iteration on the same line
    #         sys.stdout.write(f"\rSimulation {i + 1}/{n_simulations}")
    #         sys.stdout.flush()

    #         # Add small random noise to features
    #         noisy_features = upcoming_matches_stats[self.training_features].copy()
    #         noise = np.random.normal(0, 0.05, size=noisy_features.shape)  # Adjust scale as needed
    #         noisy_features += noise

    #         # Predict winners
    #         winner_predictions = self.best_model.predict(noisy_features)
    #         simulated_winner_predictions.append(winner_predictions)
            
    #         # # Predict scores
    #         # score_predictions = self.regressor.predict(upcoming_matches_stats)
    #         # simulated_team_1_scores.append(score_predictions[:, 0])  # Team 1 scores
    #         # simulated_team_2_scores.append(score_predictions[:, 1])  # Team 2 scores

    #     # Convert predictions to NumPy arrays
    #     simulated_winner_predictions = np.array(simulated_winner_predictions)
    #     # simulated_team_1_scores = np.array(simulated_team_1_scores)
    #     # simulated_team_2_scores = np.array(simulated_team_2_scores)

    #     # Calculate probabilities for winner predictions
    #     team_1_win_prob = (simulated_winner_predictions == 0).mean(axis=0)
    #     team_2_win_prob = (simulated_winner_predictions == 1).mean(axis=0)
        
    #     upcoming_matches_stats['team_1_win_prob'] = team_1_win_prob
    #     upcoming_matches_stats['team_2_win_prob'] = team_2_win_prob
    #     upcoming_matches_stats['winner_pred'] = np.where(team_1_win_prob > team_2_win_prob, 0, 1)
    #     # upcoming_matches_stats['team_1_score_pred'] = team_1_score_avg
    #     # upcoming_matches_stats['team_2_score_pred'] = team_2_score_avg
    #     print("\n")
    #     print(upcoming_matches_stats[['team_1_win_prob', 'team_2_win_prob','winner_pred']])

    def init(self):
        print("dBreakpoint init")
        self.dbconnector()
        self.load_data()
        self.FeatureEng()
        # self.FeatureEngineering()
        self.train_modal_1()
        # self.prediction()
        self.prediction1()
