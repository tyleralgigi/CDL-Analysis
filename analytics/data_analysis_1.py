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

class Data_Analysis_1():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def load_data(self):
        #Reading CSVs
        self.team_data = pd.read_csv("data/team_data.csv", header=None)
        self.player_data = pd.read_csv("data/player_data.csv", header=None)
        
        #getting headers from row 1 and 2
        headers = self.team_data.iloc[0] + "_" + self.team_data.iloc[1]
        headers_1 = self.player_data.iloc[0] + "_" + self.player_data.iloc[1]
        
        #dropping rows 1 and 2
        self.team_data = self.team_data[2:].reset_index(drop=True)
        self.player_data = self.player_data[2:].reset_index(drop=True)
        
        #setting new headers
        self.team_data.columns = headers
        self.player_data.columns = headers_1
        
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
        merged_df = pd.merge(self.standing_current, self.team_data, left_on='name_short', right_on=' _Team', how='right')
        self.team_standings = merged_df[~merged_df[['name_short', 'year_YEAR']].duplicated(keep='first')]

        self.player_data.columns = self.player_data.columns[:-1].tolist() + ['year']
                
        new_column_names = ['rank', 'player_name', 'name_short']
        self.player_data.columns = new_column_names + self.player_data.columns[3:].tolist()
        self.team_standings = self.team_standings.rename(columns={'id': 'team_id'})
        self.team_standings = self.team_standings.rename(columns={'year_YEAR': 'year'})
        
        player_merged_df = pd.merge(self.team_standings[['team_id', 'name_short', 'year']], self.player_data, on=['name_short', 'year'], how='right')
        
        self.allPlayers = self.allPlayers.rename(columns={'tag': 'player_name'})
        player_merged_df = pd.merge(player_merged_df, self.allPlayers, on=['player_name'], how='left')
        self.player_merged_df = player_merged_df[~player_merged_df[['player_name', 'year']].duplicated(keep='first')]
        
    def FeatureEngineering(self):
        #Calculate averages or rates 
        self.player_merged_df.replace(',','', regex=True, inplace=True)
        aggregated_players = pd.DataFrame()
        
        # Define the conversion dictionary
        convert_dict = {'search_K': float,
                        'search_D': float,
                        'search_KD': float,
                        'search_+/-': float,
                        'search_K': float,
                        'search_FK': float,
                        'search_M': float,
                        'Hardpoint_K': float,
                        'Hardpoint_D': float,
                        'Hardpoint_KD': float,
                        'Hardpoint_+/-': float,
                        'Hardpoint_K/M': float,
                        'Hardpoint_TM': float,
                        'Hardpoint_M': float,
                        'Control_K': float,
                        'Control_D': float,
                        'Control_KD': float,
                        'Control_+/-': float,
                        'Control_K/M': float,
                        'Control_CP': float,
                        'Control_M': float}

        # Convert columns using the dictionary
        self.player_merged_df = self.player_merged_df.astype(convert_dict)
        
        aggregated_players[['id', 'team_id', 'year']] = self.player_merged_df[['id', 'team_id', 'year']]
        aggregated_players['search_K_pm'] = self.player_merged_df['search_K'] / self.player_merged_df['search_M']
        aggregated_players['search_D_pm'] = self.player_merged_df['search_D'] / self.player_merged_df['search_M']
        aggregated_players['search_KD_pm'] = self.player_merged_df['search_KD'] / self.player_merged_df['search_M']
        aggregated_players['search_+/-_pm'] = self.player_merged_df['search_+/-'] / self.player_merged_df['search_M']
        aggregated_players['hardpoint_K_pm'] = self.player_merged_df['Hardpoint_K'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['hardpoint_D_pm'] = self.player_merged_df['Hardpoint_D'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['hardpoint_KD_pm'] = self.player_merged_df['Hardpoint_KD'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['hardpoint_+/-_pm'] = self.player_merged_df['Hardpoint_+/-'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['hardpoint_K/M_pm'] = self.player_merged_df['Hardpoint_K/M'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['hardpoint_TM_pm'] = self.player_merged_df['Hardpoint_TM'] / self.player_merged_df['Hardpoint_M']
        aggregated_players['control_K_pm'] = self.player_merged_df['Control_K'] / self.player_merged_df['Control_M']
        aggregated_players['control_D_pm'] = self.player_merged_df['Control_D'] / self.player_merged_df['Control_M']
        aggregated_players['control_KD_pm'] = self.player_merged_df['Control_KD'] / self.player_merged_df['Control_M']
        aggregated_players['control_+/-_pm'] = self.player_merged_df['Control_+/-'] / self.player_merged_df['Control_M']
        aggregated_players['control_K/M_pm'] = self.player_merged_df['Control_K/M'] / self.player_merged_df['Control_M']
        aggregated_players['control_TM_pm'] = self.player_merged_df['Control_CP'] / self.player_merged_df['Control_M']

        aggregated_team_averages = aggregated_players.groupby(['team_id', 'year']).agg({
            'search_K_pm': 'mean',
            'search_D_pm': 'mean',
            'search_KD_pm': 'mean',
            'search_+/-_pm': 'mean',
            'hardpoint_K_pm': 'mean',
            'hardpoint_D_pm': 'mean',
            'hardpoint_KD_pm': 'mean',
            'hardpoint_+/-_pm': 'mean',
            'hardpoint_K/M_pm': 'mean',
            'hardpoint_TM_pm': 'mean',
            'control_K_pm': 'mean',
            'control_D_pm': 'mean',
            'control_KD_pm': 'mean',
            'control_+/-_pm': 'mean',
            'control_K/M_pm': 'mean',
            'control_TM_pm': 'mean',
            # Add more columns as needed
        }).reset_index()
        self.aggregated_team_averages = pd.merge(self.team_standings[['team_id', 'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',  'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %', 'year']], aggregated_team_averages, on=['team_id', 'year'], how='right')
        # print(self.aggregated_team_averages)
        del aggregated_team_averages
        del self.player_merged_df
        
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
        
        self.match_with_teams['search_KD_pm_diff'] = self.match_with_teams['search_KD_pm_team1'] - self.match_with_teams['search_KD_pm_team2']
        self.match_with_teams['control_KD_pm_diff'] = self.match_with_teams['control_KD_pm_team1'] - self.match_with_teams['control_KD_pm_team2']
        
        relative_metrics = [
            'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %', 'Total_Map_Wins',
            'Total_Map_Losses', 'Total_Maps %', 'search_K_pm', 'search_D_pm', 'search_KD_pm',
            'search_+/-_pm', 'hardpoint_K_pm', 'hardpoint_D_pm', 'hardpoint_KD_pm',
            'hardpoint_+/-_pm', 'hardpoint_K/M_pm', 'hardpoint_TM_pm', 'control_K_pm',
            'control_D_pm', 'control_KD_pm', 'control_+/-_pm', 'control_K/M_pm', 'control_TM_pm'
        ]
        self.match_with_teams = self.match_with_teams.apply(pd.to_numeric, errors='coerce')

        # Compute relative metrics
        for metric in relative_metrics:
            self.match_with_teams[f'{metric}_diff'] = (
                self.match_with_teams[f'{metric}_team1'] - self.match_with_teams[f'{metric}_team2']
            )
        columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
        self.match_with_teams = self.match_with_teams.drop(columns=columns_to_drop)

        # Columns to normalize
        columns_to_normalize = [
            'search_KD_pm_diff', 'control_KD_pm_diff', 'Total_Series_Wins_diff',
            'Total_Series_Losses_diff', 'Total_Series %_diff', 'Total_Map_Wins_diff',
            'Total_Map_Losses_diff', 'Total_Maps %_diff', 'search_K_pm_diff',
            'search_D_pm_diff', 'search_+/-_pm_diff', 'hardpoint_K_pm_diff',
            'hardpoint_D_pm_diff', 'hardpoint_KD_pm_diff', 'hardpoint_+/-_pm_diff',
            'hardpoint_K/M_pm_diff', 'hardpoint_TM_pm_diff', 'control_K_pm_diff',
            'control_D_pm_diff', 'control_+/-_pm_diff', 'control_K/M_pm_diff',
            'control_TM_pm_diff'
        ]
        
        # Initialize the scaler
        self.scaler = MinMaxScaler()

        # Apply MinMaxScaler to the selected columns
        self.match_with_teams[columns_to_normalize] = self.scaler.fit_transform(self.match_with_teams[columns_to_normalize])
        self.match_with_teams['winner_id'] = (self.match_with_teams['winner_id'] == self.match_with_teams['team_2_id']).astype(int)
        self.match_with_teams['spreadsheet_id'] = self.match_with_teams['spreadsheet_id'].astype('category')

        self.match_with_teams = self.match_with_teams.drop(columns=['team_1_id', 'team_2_id'])
        
        print(self.match_with_teams.columns)
        print(self.match_with_teams)
        self.match_with_teams = self.match_with_teams.rename(columns={"spreadsheet_id": "year"})

    def train_modal(self):
        # Define features and target (excluding team scores)
        X = self.match_with_teams.drop(columns=['team_1_score', 'team_2_score', 'winner_id'])
        y_winner = self.match_with_teams['winner_id']  # Classification target
        
        # Train-test split
        X_train, X_test, y_winner_train, y_winner_test = train_test_split(
            X, y_winner, test_size=0.2, random_state=42
        )

        # Train classifier
        self.classifier = RandomForestClassifier(random_state=42)
        self.classifier.fit(X_train, y_winner_train)

        # Predict winner
        y_winner_pred = self.classifier.predict(X_test)

        # Evaluate classification
        accuracy = accuracy_score(y_winner_test, y_winner_pred)
        classification_rep = classification_report(y_winner_test, y_winner_pred)
        print(f"Classification Accuracy (No Scores): {accuracy}")
        print(classification_rep)

        # Feature importance analysis
        classifier_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.classifier.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Compute correlation matrix by temporarily adding 'winner_id'
        X_with_target = self.match_with_teams[['winner_id']].join(X)
        corr_matrix = X_with_target.corr()

        # Print correlation with 'winner_id'
        print("Correlation with 'winner_id':")
        print(corr_matrix['winner_id'].sort_values(ascending=False))

        # Plotting classification feature importance
        # plt.figure(figsize=(10, 6))
        # classifier_features = classifier_importance.head(10)  # Top 10 features
        # plt.barh(classifier_features['Feature'], classifier_features['Importance'], align='center', color='orange')
        # plt.xlabel('Importance')
        # plt.ylabel('Feature')
        # plt.title('Top 10 Features for Winner Prediction (No Scores)')
        # plt.gca().invert_yaxis()
        # plt.show()

    def train_modal_1(self):
        # Define features and target
        X = self.match_with_teams.drop(columns=['team_1_score', 'team_2_score', 'winner_id'])
        
        # Targets
        y_scores = self.match_with_teams[['team_1_score', 'team_2_score']]  # Regression targets
        y_winner = self.match_with_teams['winner_id']  # Classification target

        # Train-test split
        X_train, X_test, y_scores_train, y_scores_test, y_winner_train, y_winner_test = train_test_split(
            X, y_scores, y_winner, test_size=0.2, random_state=42
        )
        
        # Multi-output regression
        self.regressor = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        self.regressor.fit(X_train, y_scores_train)

        # Predict scores
        y_scores_pred = self.regressor.predict(X_test)

        # Evaluate regression
        mse_team_1 = mean_squared_error(y_scores_test['team_1_score'], y_scores_pred[:, 0])
        mse_team_2 = mean_squared_error(y_scores_test['team_2_score'], y_scores_pred[:, 1])
        
        # Train classifier
        self.classifier = RandomForestClassifier(random_state=42)
        self.classifier.fit(X_train, y_winner_train)

        # Predict winner
        y_winner_pred = self.classifier.predict(X_test)

        # Evaluate classification
        accuracy = accuracy_score(y_winner_test, y_winner_pred)
        classification_rep = classification_report(y_winner_test, y_winner_pred)
        
        print(accuracy)
        print(classification_rep)
        
        # Combine feature importance from regressor and classifier
        regressor_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.regressor.estimators_[0].feature_importances_
        }).sort_values(by='Importance', ascending=False)

        classifier_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.classifier.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Correlation matrix for winner_id
        corr_matrix = self.match_with_teams.corr()
        print("Correlation with 'winner_id':")
        print(corr_matrix['winner_id'].sort_values(ascending=False))
        
        # # Plotting regression feature importance
        # plt.figure(figsize=(10, 6))
        # regressor_features = regressor_importance.head(10)  # Top 10 features
        # plt.barh(regressor_features['Feature'], regressor_features['Importance'], align='center')
        # plt.xlabel('Importance')
        # plt.ylabel('Features')
        # plt.title('Top 10 Features for Team Score Prediction (Regression)')
        # plt.gca().invert_yaxis()
        # plt.show()

        # # Plotting classification feature importance
        plt.figure(figsize=(10, 6))
        classifier_features = classifier_importance.head(10)  # Top 10 features
        plt.barh(classifier_features['Feature'], classifier_features['Importance'], align='center', color='orange')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Top 10 Features for Winner Prediction (Classification)')
        plt.gca().invert_yaxis()
        plt.show()

    def prediction(self):
        self.upcoming_games = pd.read_csv("data/upcoming_games.csv", header=0)
        self.upcoming_games['year'] = '2025'
        
        # Map team1_name and team2_name to their respective IDs using the df_teams dataframe
        team_id_map = dict(zip(self.standing_current['name_short'], self.standing_current['id']))

        self.upcoming_games['team1_id'] = self.upcoming_games['team_1_name'].map(team_id_map)
        self.upcoming_games['team2_id'] = self.upcoming_games['team_2_name'].map(team_id_map)
        
        # Drop the original name columns if no longer needed
        self.upcoming_games_team1 = self.upcoming_games[['team1_id', 'year']]
        self.upcoming_games_team2 = self.upcoming_games[['team2_id', 'year']]
        
        upcoming_games_team1_stats = pd.merge(self.team_standings[['team_id', 'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',  'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %', 'year']], 
                                                self.upcoming_games_team1,
                                                left_on=['team_id', 'year'],
                                                right_on=['team1_id', 'year'],
                                                suffixes=('', '_team1'),
                                                how='right')

        upcoming_games_team2_stats = pd.merge(self.team_standings[['team_id', 'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',  'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %', 'year']], 
                                                self.upcoming_games_team2,
                                                left_on=['team_id', 'year'],
                                                right_on=['team2_id', 'year'],
                                                suffixes=('', '_team2'),
                                                how='right')
        
        upcoming_games_team_stats = pd.concat([upcoming_games_team1_stats, upcoming_games_team2_stats], axis=1)
        columns_to_drop = ['team1_id', 'team2_id', 'team_id_team1', 'team_id_team2']
        upcoming_games_team_stats = upcoming_games_team_stats.drop(columns=columns_to_drop, errors='ignore')
        
        match_with_team1 = pd.merge(
            self.upcoming_games,
            self.aggregated_team_averages,
            left_on=['team1_id', 'year'],  # Match team ID and year
            right_on=['team_id', 'year'],  # Columns in aggregated averages
            suffixes=('', '_team1'),  # Add suffix for team 1 stats
            how='left'
        )

        # Merge Team 2 data
        match_with_team2 = pd.merge(
            match_with_team1,
            self.aggregated_team_averages,
            left_on=['team2_id', 'year'],  # Match team ID and year
            right_on=['team_id', 'year'],  # Columns in aggregated averages
            suffixes=('_team1', '_team2'),  # Add suffix for Team 2
            how='left'
        )
        columns_to_drop = ['team1_id', 'team2_id']
        match_with_teams = match_with_team2.drop(columns=columns_to_drop, errors='ignore')
        
        upcoming_matches_stats = pd.concat([match_with_teams, upcoming_games_team_stats], axis=1)
        upcoming_matches_stats = upcoming_matches_stats.loc[:, ~upcoming_matches_stats.columns.duplicated(keep='first')]
        columns_to_drop = ['team_1_name', 'team_2_name', 'Total_Maps %', 'Total_Map_Losses', 'Total_Map_Wins',
                        'Total_Series', 'Total_Series_Losses', 'Total_Series_Wins', 'Total_Series %', 'team_id']
        upcoming_matches_stats = upcoming_matches_stats.drop(columns=columns_to_drop, errors='ignore')

        upcoming_matches_stats['search_KD_pm_diff'] = upcoming_matches_stats['search_KD_pm_team1'] - upcoming_matches_stats['search_KD_pm_team2']
        upcoming_matches_stats['control_KD_pm_diff'] = upcoming_matches_stats['control_KD_pm_team1'] - upcoming_matches_stats['control_KD_pm_team2']
        
        relative_metrics = [
            'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %', 'Total_Map_Wins',
            'Total_Map_Losses', 'Total_Maps %', 'search_K_pm', 'search_D_pm', 'search_KD_pm',
            'search_+/-_pm', 'hardpoint_K_pm', 'hardpoint_D_pm', 'hardpoint_KD_pm',
            'hardpoint_+/-_pm', 'hardpoint_K/M_pm', 'hardpoint_TM_pm', 'control_K_pm',
            'control_D_pm', 'control_KD_pm', 'control_+/-_pm', 'control_K/M_pm', 'control_TM_pm'
        ]
        upcoming_matches_stats = upcoming_matches_stats.apply(pd.to_numeric, errors='coerce')

        # Compute relative metrics
        for metric in relative_metrics:
            upcoming_matches_stats[f'{metric}_diff'] = (
                upcoming_matches_stats[f'{metric}_team1'] - upcoming_matches_stats[f'{metric}_team2']
            )
        columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
        upcoming_matches_stats = upcoming_matches_stats.drop(columns=columns_to_drop)
        
        # Columns to normalize
        columns_to_normalize = [
            'search_KD_pm_diff', 'control_KD_pm_diff', 'Total_Series_Wins_diff',
            'Total_Series_Losses_diff', 'Total_Series %_diff', 'Total_Map_Wins_diff',
            'Total_Map_Losses_diff', 'Total_Maps %_diff', 'search_K_pm_diff',
            'search_D_pm_diff', 'search_+/-_pm_diff', 'hardpoint_K_pm_diff',
            'hardpoint_D_pm_diff', 'hardpoint_KD_pm_diff', 'hardpoint_+/-_pm_diff',
            'hardpoint_K/M_pm_diff', 'hardpoint_TM_pm_diff', 'control_K_pm_diff',
            'control_D_pm_diff', 'control_+/-_pm_diff', 'control_K/M_pm_diff',
            'control_TM_pm_diff'
        ]
        
        # Apply MinMaxScaler to the selected columns
        upcoming_matches_stats[columns_to_normalize] = self.scaler.transform(upcoming_matches_stats[columns_to_normalize])
        upcoming_matches_stats['year'] = upcoming_matches_stats['year'].astype('category')

        upcoming_matches_stats = upcoming_matches_stats.drop(columns=['team_id_team1', 'team_id_team2'])
        
        print(upcoming_matches_stats.columns)
        print(upcoming_matches_stats)

        # Number of simulations
        n_simulations = 10000
        
        simulated_winner_predictions = [] # Placeholder for predictions
        simulated_team_1_scores = []  # Store Team 1 scores
        simulated_team_2_scores = []  # Store Team 2 scores
        
        # Run predictions 10,000 times
        for i in range(n_simulations):
            # Print the current iteration on the same line
            sys.stdout.write(f"\rSimulation {i + 1}/{n_simulations}")
            sys.stdout.flush()
    
            winner_predictions = self.classifier.predict(upcoming_matches_stats)
            simulated_winner_predictions.append(winner_predictions)
            
            # # Predict scores
            # score_predictions = self.regressor.predict(upcoming_matches_stats)
            # simulated_team_1_scores.append(score_predictions[:, 0])  # Team 1 scores
            # simulated_team_2_scores.append(score_predictions[:, 1])  # Team 2 scores

        # Convert predictions to NumPy arrays
        simulated_winner_predictions = np.array(simulated_winner_predictions)
        # simulated_team_1_scores = np.array(simulated_team_1_scores)
        # simulated_team_2_scores = np.array(simulated_team_2_scores)

        # Calculate probabilities for winner predictions
        team_1_win_prob = (simulated_winner_predictions == 0).mean(axis=0)
        team_2_win_prob = (simulated_winner_predictions == 1).mean(axis=0)

        # Average scores for each team
        # team_1_score_avg = simulated_team_1_scores.mean(axis=0)
        # team_2_score_avg = simulated_team_2_scores.mean(axis=0)

        # Add results to the DataFrame
        upcoming_matches_stats['team_1_win_prob'] = team_1_win_prob
        upcoming_matches_stats['team_2_win_prob'] = team_2_win_prob
        upcoming_matches_stats['winner_pred'] = np.where(team_1_win_prob > team_2_win_prob, 0, 1)
        # upcoming_matches_stats['team_1_score_pred'] = team_1_score_avg
        # upcoming_matches_stats['team_2_score_pred'] = team_2_score_avg
        print("\n")
        print(upcoming_matches_stats[['team_1_win_prob', 'team_2_win_prob','winner_pred']])
        # # Predict winners and scores for upcoming games
        # winner_predictions = self.classifier.predict(upcoming_matches_stats)
        # score_predictions = self.regressor.predict(upcoming_matches_stats)
        
        # # # Add predictions to the DataFrame
        # upcoming_matches_stats['team_1_score_pred'] = score_predictions[:, 0]
        # upcoming_matches_stats['team_2_score_pred'] = score_predictions[:, 1]
        # upcoming_matches_stats['winner_pred'] = winner_predictions
        # print(upcoming_matches_stats[['team_1_score_pred', 'team_2_score_pred','winner_pred' ]])
        
    def init(self):
        print("data analysis 1 init")
        self.dbconnector()
        self.load_data()
        self.FeatureEngineering()
        self.train_modal()
        self.prediction()