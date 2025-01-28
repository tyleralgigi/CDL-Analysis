
    # def FeatureEngineering(self):
    #     #Calculate averages or rates 
    #     # self.players.replace(',','', regex=True, inplace=True)
    #     self.player_merged_df.drop(['Player', 'BP Rating'], axis=1, inplace=True)
    #     # print(self.player_merged_df.columns)
    #     aggregated_players = pd.DataFrame()
        
    #     # Define the conversion dictionary
    #     convert_dict = {'K/D': float,
    #                     'Slayer Rating': float,
    #                     'T.E.S': float,
    #                     'HP KD': float,
    #                     'HP K/10M': float,
    #                     'HP OBJ/10M': float,
    #                     'HP Eng/10M': float,
    #                     'HP Maps': float,
    #                     'SND KD': float,
    #                     'SND KPR': float,
    #                     'First Bloods': float,
    #                     'First Blood %': float,
    #                     'Plants': float,
    #                     'Defuses': float,
    #                     'SND Maps': float,
    #                     'CTL KD': float,
    #                     'CTL K/10M': float,
    #                     'CTL DMG/10M': float,
    #                     'CTL Eng/10M': float,
    #                     'Zone Tier Captures': float,
    #                     'CTL Maps': float,
    #                     'Non-Traded Kills': float,
    #                     'Maps': float,
    #                     }

    #     # Convert columns using the dictionary
    #     self.player_merged_df = self.player_merged_df.astype(convert_dict)
        
    #     aggregated_players[['id', 'team_id', 'Year', 'SlayerRating', 'TES']] = self.player_merged_df[['id', 'team_id', 'Year', 'Slayer Rating', 'T.E.S']]
    #     aggregated_players['KD'] = self.player_merged_df['K/D'] / self.player_merged_df['Maps']
    #     aggregated_players['NonTradedKills'] = self.player_merged_df['Non-Traded Kills'] / self.player_merged_df['Maps']
    #     aggregated_players['HP_KD'] = self.player_merged_df['HP KD'] / self.player_merged_df['HP Maps']
    #     aggregated_players['HP_K10M'] = self.player_merged_df['HP K/10M'] / self.player_merged_df['HP Maps']
    #     aggregated_players['HP_OBJ10M'] = self.player_merged_df['HP OBJ/10M'] / self.player_merged_df['HP Maps']
    #     aggregated_players['HP_Eng10M'] = self.player_merged_df['HP Eng/10M'] / self.player_merged_df['HP Maps']
    #     aggregated_players['SND_KD'] = self.player_merged_df['SND KD'] / self.player_merged_df['SND Maps']
    #     aggregated_players['SND_KPR'] = self.player_merged_df['SND KPR'] / self.player_merged_df['SND Maps']
    #     aggregated_players['SND_FB'] = self.player_merged_df['First Bloods'] / self.player_merged_df['SND Maps']
    #     aggregated_players['SND_FB_Percent'] = self.player_merged_df['First Blood %'] / self.player_merged_df['SND Maps']
    #     aggregated_players['SND_OBJ'] = (self.player_merged_df['Plants'] + self.player_merged_df['Defuses']) / self.player_merged_df['SND Maps']
    #     aggregated_players['CTL_KD'] = self.player_merged_df['CTL KD'] / self.player_merged_df['CTL Maps']
    #     aggregated_players['CTL_K10M'] = self.player_merged_df['CTL K/10M'] / self.player_merged_df['CTL Maps']
    #     aggregated_players['CTL_DMG10M'] = self.player_merged_df['CTL DMG/10M'] / self.player_merged_df['CTL Maps']
    #     aggregated_players['CTL_Eng10M'] = self.player_merged_df['CTL Eng/10M'] / self.player_merged_df['CTL Maps']
    #     aggregated_players['CTL_Zone_Captures'] = self.player_merged_df['Zone Tier Captures'] / self.player_merged_df['CTL Maps']
    #     aggregated_players = aggregated_players.rename(columns={'Year': 'year'})
        
    #     aggregated_team_averages = aggregated_players.groupby(['team_id', 'year']).agg({
    #         'SlayerRating': 'mean',
    #         'TES': 'mean',
    #         'KD': 'mean',
    #         'NonTradedKills': 'mean',
    #         'HP_KD': 'mean',
    #         'HP_K10M': 'mean',
    #         'HP_OBJ10M': 'mean',
    #         'HP_Eng10M': 'mean',
    #         'SND_KD': 'mean',
    #         'SND_KPR': 'mean',
    #         'SND_FB': 'mean',
    #         'SND_FB_Percent': 'mean',
    #         'SND_OBJ': 'mean',
    #         'CTL_KD': 'mean',
    #         'CTL_K10M': 'mean',
    #         'CTL_DMG10M': 'mean',
    #         'CTL_Eng10M': 'mean',
    #         'CTL_Zone_Captures': 'mean',
    #         # Add more columns as needed
    #     }).reset_index()
        
    #     # print(self.team_standings.columns)
        
    #     self.team_standings = self.team_standings.rename(columns={'id': 'team_id', 'Year': 'year'})
    #     self.aggregated_team_averages = pd.merge(self.team_standings[['team_id', 'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',  'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %', 'year', 'HP +/-'
    #                                                                 , 'HP Win %', 'S&D Win %', 'S&D +/-', 'CTL Win %', 'CTL Round +/-']], aggregated_team_averages, on=['team_id', 'year'], how='right')
    #     # print(self.aggregated_team_averages)
    #     del aggregated_team_averages
    #     del self.player_merged_df
        
    #     self.matches['spreadsheet_id'] = self.matches['spreadsheet_id'].astype(int)
    #     # Merge Team 1 data
    #     match_with_team1 = pd.merge(
    #         self.matches,
    #         self.aggregated_team_averages,
    #         left_on=['team_1_id', 'spreadsheet_id'],  # Match team ID and year
    #         right_on=['team_id', 'year'],            # Columns in aggregated averages
    #         suffixes=('', '_team1')                  # Add suffix for team 1 stats
    #     )
        
    #     # Merge Team 2 data
    #     match_with_teams = pd.merge(
    #         match_with_team1,
    #         self.aggregated_team_averages,
    #         left_on=['team_2_id', 'spreadsheet_id'],
    #         right_on=['team_id', 'year'],
    #         suffixes=('_team1', '_team2')  # Add suffix for Team 2
    #     )
        
    #     # Drop redundant columns to avoid confusion
    #     columns_to_drop = ['team_id_team1', 'year_team1', 'team_id_team2', 'year_team2']
    #     self.match_with_teams = match_with_teams.drop(columns=columns_to_drop, errors='ignore')
    #     del match_with_teams
    #     del match_with_team1
    #     # print(self.match_with_teams.columns)
        
    #     relative_metrics = [
    #         'Total_Series_Wins',
    #         'Total_Series_Losses', 'Total_Series %',
    #         'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %',
    #         'HP +/-', 'HP Win %', 'S&D Win %', 'S&D +/-',
    #         'CTL Win %', 'CTL Round +/-', 'SlayerRating',
    #         'TES', 'KD', 'NonTradedKills', 'HP_KD',
    #         'HP_K10M', 'HP_OBJ10M', 'HP_Eng10M', 'SND_KD',
    #         'SND_KPR', 'SND_FB', 'SND_FB_Percent',
    #         'SND_OBJ', 'CTL_KD', 'CTL_K10M', 'CTL_DMG10M',
    #         'CTL_Eng10M', 'CTL_Zone_Captures']
    #     self.match_with_teams = self.match_with_teams.apply(pd.to_numeric, errors='coerce')
        
    #     # Compute relative metrics
    #     for metric in relative_metrics:
    #         self.match_with_teams[f'{metric}_diff'] = (
    #             self.match_with_teams[f'{metric}_team1'] - self.match_with_teams[f'{metric}_team2']
    #         )
    #     columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
    #     self.match_with_teams = self.match_with_teams.drop(columns=columns_to_drop)
    #     print(self.match_with_teams)
        
    #     # Columns to normalize
    #     columns_to_normalize = [
    #         'Total_Series_Wins_diff',
    #         'Total_Series_Losses_diff', 'Total_Series %_diff',
    #         'Total_Map_Wins_diff', 'Total_Map_Losses_diff', 'Total_Maps %_diff',
    #         'HP +/-_diff', 'HP Win %_diff', 'S&D Win %_diff', 'S&D +/-_diff',
    #         'CTL Win %_diff', 'CTL Round +/-_diff', 'SlayerRating_diff',
    #         'TES_diff', 'KD_diff', 'NonTradedKills_diff', 'HP_KD_diff',
    #         'HP_K10M_diff', 'HP_OBJ10M_diff', 'HP_Eng10M_diff', 'SND_KD_diff',
    #         'SND_KPR_diff', 'SND_FB_diff', 'SND_FB_Percent_diff',
    #         'SND_OBJ_diff', 'CTL_KD_diff', 'CTL_K10M_diff', 'CTL_DMG10M_diff',
    #         'CTL_Eng10M_diff', 'CTL_Zone_Captures_diff'
    #     ]
        
    #     # Initialize the scaler
    #     self.scaler = MinMaxScaler()
        
    #     # Apply MinMaxScaler to the selected columns
    #     self.match_with_teams[columns_to_normalize] = self.scaler.fit_transform(self.match_with_teams[columns_to_normalize])
    #     self.match_with_teams['winner_id'] = (self.match_with_teams['winner_id'] == self.match_with_teams['team_2_id']).astype(int)
    #     self.match_with_teams['spreadsheet_id'] = self.match_with_teams['spreadsheet_id'].astype('category')

    #     self.match_with_teams = self.match_with_teams.drop(columns=['team_1_id', 'team_2_id'])
        
    #     print(self.match_with_teams.columns)
    #     print(self.match_with_teams)
    #     self.match_with_teams = self.match_with_teams.rename(columns={"spreadsheet_id": "year"})


    # def train_modal(self):
    #     # Define features and target (excluding team scores)
    #     X = self.match_with_teams.drop(columns=['team_1_score', 'team_2_score', 'winner_id'])
    #     y_winner = self.match_with_teams['winner_id']  # Classification target
        
    #     # Compute correlation matrix by temporarily adding 'winner_id'
    #     if 'winner_id' in self.match_with_teams.columns:
    #         X_with_target = self.match_with_teams[['winner_id']].join(X)
    #     else:
    #         raise ValueError("The 'winner_id' column is missing from self.match_with_teams")

    #     # Compute the correlation matrix
    #     corr_matrix = X_with_target.corr()

    #     # Identify low-correlation features
    #     self.low_corr_features = corr_matrix['winner_id'][corr_matrix['winner_id'].abs() < 0.15].index
    #     print(f"Features with low correlation (abs < 0.2): {list(self.low_corr_features)}")

    #     # Drop low-correlation features from X
    #     X = X.drop(columns=self.low_corr_features, errors='ignore')
    #     print(f"Remaining features after dropping low-correlation ones: {X.columns.tolist()}")

    #     # Save the training feature set
    #     self.training_features = X.columns.tolist()
        
    #     # Train-test split
    #     X_train, X_test, y_winner_train, y_winner_test = train_test_split(
    #         X, y_winner, test_size=0.2, random_state=42
    #     )

    #     # Train classifier
    #     self.classifier = RandomForestClassifier(random_state=42)
    #     # Add noise to training features
    #     X_train_noisy = self.add_noise(X_train, noise_level=0.12)
        
    #     self.classifier.fit(X_train_noisy, y_winner_train)

    #     # Predict winner
    #     y_winner_pred = self.classifier.predict(X_test)

    #     # Evaluate classification
    #     accuracy = accuracy_score(y_winner_test, y_winner_pred)
    #     classification_rep = classification_report(y_winner_test, y_winner_pred)
    #     print(f"Classification Accuracy (No Scores): {accuracy}")
    #     print(classification_rep)

    #     # Feature importance analysis
    #     classifier_importance = pd.DataFrame({
    #         'Feature': X.columns,
    #         'Importance': self.classifier.feature_importances_
    #     }).sort_values(by='Importance', ascending=False)

    #     # Print feature importance
    #     print("Feature Importance:")
    #     print(classifier_importance)

    #     # Plotting
    #     fig, ax = plt.subplots(figsize=(10, 6))

    #     # Plot feature importance
    #     ax.bar(classifier_importance['Feature'], classifier_importance['Importance'], alpha=0.6)
    #     ax.set_ylabel('Importance')
    #     ax.set_xlabel('Features')
    #     ax.set_xticklabels(classifier_importance['Feature'], rotation=45, ha='right')

    #     plt.title('Feature Importance After Removing Low-Correlation Features')
    #     plt.tight_layout()
    #     plt.show()

    # def prediction(self):
    #     # Load upcoming games and preprocess
    #     self.upcoming_games = pd.read_csv("data/upcoming_games.csv", header=0)
    #     self.upcoming_games['year'] = '2025'

    #     # Map team1_name and team2_name to their respective IDs
    #     team_id_map = dict(zip(self.standing_current['name_short'], self.standing_current['id']))
    #     self.upcoming_games['team1_id'] = self.upcoming_games['team_1_name'].map(team_id_map)
    #     self.upcoming_games['team2_id'] = self.upcoming_games['team_2_name'].map(team_id_map)
    #     self.upcoming_games['year'] = self.upcoming_games['year'].astype(int)


        
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
    #         'Total_Series_Wins', 'Total_Series_Losses', 'Total_Series %',
    #         'Total_Map_Wins', 'Total_Map_Losses', 'Total_Maps %',
    #         'HP +/-', 'HP Win %', 'S&D Win %', 'S&D +/-',
    #         'CTL Win %', 'CTL Round +/-', 'SlayerRating', 'TES', 'KD',
    #         'NonTradedKills', 'HP_KD', 'HP_K10M', 'HP_OBJ10M', 'HP_Eng10M',
    #         'SND_KD', 'SND_KPR', 'SND_FB', 'SND_FB_Percent', 'SND_OBJ',
    #         'CTL_KD', 'CTL_K10M', 'CTL_DMG10M', 'CTL_Eng10M', 'CTL_Zone_Captures'
    #     ]
    #     for metric in relative_metrics:
    #         upcoming_matches_stats[f'{metric}_diff'] = (
    #             upcoming_matches_stats[f'{metric}_team1'] - upcoming_matches_stats[f'{metric}_team2']
    #         )
    #     columns_to_drop = [f'{col}_team1' for col in relative_metrics] + [f'{col}_team2' for col in relative_metrics]
    #     upcoming_matches_stats = upcoming_matches_stats.drop(columns=columns_to_drop, errors='ignore')

    #     # Normalize columns
    #     columns_to_normalize = [
    #         'Total_Series_Wins_diff', 'Total_Series_Losses_diff', 'Total_Series %_diff',
    #         'Total_Map_Wins_diff', 'Total_Map_Losses_diff', 'Total_Maps %_diff',
    #         'HP +/-_diff', 'HP Win %_diff', 'S&D Win %_diff', 'S&D +/-_diff',
    #         'CTL Win %_diff', 'CTL Round +/-_diff', 'SlayerRating_diff', 'TES_diff',
    #         'KD_diff', 'NonTradedKills_diff', 'HP_KD_diff', 'HP_K10M_diff',
    #         'HP_OBJ10M_diff', 'HP_Eng10M_diff', 'SND_KD_diff', 'SND_KPR_diff',
    #         'SND_FB_diff', 'SND_FB_Percent_diff', 'SND_OBJ_diff', 'CTL_KD_diff',
    #         'CTL_K10M_diff', 'CTL_DMG10M_diff', 'CTL_Eng10M_diff', 'CTL_Zone_Captures_diff'
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
    #         noise = np.random.normal(0, 0.10, size=noisy_features.shape)  # Adjust scale as needed
    #         noisy_features += noise

    #         # Predict winners
    #         winner_predictions = self.classifier.predict(noisy_features)
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
