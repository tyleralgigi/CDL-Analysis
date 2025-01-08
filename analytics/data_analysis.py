import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.stats import f_oneway
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json

rf_model = None

class Data_Analysis():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def calculate_role_weights(self, df):
        self.role_weights = {}
        
        # Features and target
        X = df[['average_kills', 'average_assists', 'average_objective_time', 'average_death'
                    ,'average_damage', 'assigned_role']]
        y = df[['game_win_percent', 'match_win_percent']]
        
        for role in df['assigned_role'].unique():
            role_df = df[df['assigned_role'] == role]
            print(role)
            
            role_df = df[['average_kills', 'average_assists', 'average_objective_time', 'average_death'
                    ,'average_damage']]
            role_df = df[['game_win_percent', 'match_win_percent']]
            
            # Train Random Forest
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)

            # self.role_weights[role] = dict(zip(X.columns, model.coef_))
            
            # Get feature importances
            importances = model.feature_importances_
            weights = importances / importances.sum()
            self.role_weights[role] = dict(zip(X.columns, weights))

    def calculate_weighted_score(self, row):
        weights = self.role_weights[row['assigned_role']]
        score = (
            row['average_kills'] * weights['average_kills'] +
            row['average_assists'] * weights['average_assists'] +
            row['average_objective_time'] * weights['average_objective_time'] +
            row['average_death'] * weights['average_death']+
            row['average_damage'] * weights['average_damage']
        )
        return score
    
    def calculate_player_rank(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT team_rosters.*, player_roles.* FROM public.team_rosters AS team_rosters JOIN public.player_roles AS player_roles ON player_roles.player_id = team_rosters.id;"))
            df = pd.DataFrame(result.fetchall())
            
            self.calculate_role_weights(df)
            df['weighted_score'] = df.apply(self.calculate_weighted_score, axis=1)
            df['rank'] = df['weighted_score'].rank(ascending=False).astype(int)
            
            scaler = MinMaxScaler()
            self.player_stats = df
            self.player_stats['normalized_score'] = scaler.fit_transform(self.player_stats[['weighted_score']])
            print(self.player_stats)

    def get_player_data(self):
        print("get_player_data")
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT team_rosters.*, player_roles.* FROM public.team_rosters AS team_rosters JOIN public.player_roles AS player_roles ON player_roles.player_id = team_rosters.id;"))
            df = pd.DataFrame(result.fetchall())

            # Features and target
            X = df[['average_kills', 'average_assists', 'average_objective_time', 'average_death'
                        ,'average_damage', 'assigned_role']]
            y = df[['game_win_percent', 'match_win_percent']]

            # Train Random Forest
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)

            # Get feature importances
            importances = model.feature_importances_
            weights = importances / importances.sum()
            print("Feature Importances as Weights:", weights)
            #kills,      assists', ' obj,     ' death',   damage',    'assigned_role'
            # 0.23621846 0.18929841 0.13319174 0.31498264 0.09520257 0.03110618

    def role_classification(self):
        print("role_classification")
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.df)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['role_cluster'] = kmeans.fit_predict(normalized_data)
        
        # Add cluster centers to the DataFrame for analysis
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        role_centers = pd.DataFrame(cluster_centers, columns=['player_id', 'kills_normalized', 'deaths_normalized', 
                                                            'damage_normalized','assists_normalized',
                                                            'objective_normalized'])
        print(role_centers)
        # Group by 'player_id' and calculate the average for each category
        
        self.df = (
            self.df.groupby('player_id')
            .agg(
                assigned_role=('role_cluster', lambda x: x.value_counts().idxmax()),
                average_kills=('kills_normalized', 'mean'),
                average_assists=('assists_normalized', 'mean'),
                average_objective_time=('objective_normalized', 'mean'),
                average_death=('deaths_normalized', 'mean'),
                average_damage=('damage_normalized', 'mean')
            )
            .reset_index()
        )
        
        print(self.df)
        # Replace cluster numbers with roles based on analysis
        role_mapping = {
            0: 'Slayer',  
            1: 'Support',
            2: 'Objective Player'
        }
        self.df['role'] = self.df['assigned_role'].map(role_mapping)
        
        #shotzzy
        print(self.df[self.df['player_id'] == 27])
        
        #dashy
        print(self.df[self.df['player_id'] == 11])
        
        #beans
        print(self.df[self.df['player_id'] == 42])
        
        self.df.to_sql("player_roles", self.engine, if_exists="replace", index=False)
    
    def plot_role_classification(self):
        print("plot_role_classification")
        # Scatter plot
        with self.engine.connect() as conn:
            result = conn.execute(text(f"select id, tag from public.\"matches_allPlayers\";"))
            players = pd.DataFrame(result.fetchall())
            
            scatter = plt.scatter(self.df['average_kills'], self.df['average_objective_time'], c=self.df['assigned_role'], cmap='viridis', s=100)
            plt.xlabel('Kills')
            plt.ylabel('Objective Time')
            plt.title('Player Roles by Clustering')
            plt.legend(handles=scatter.legend_elements()[0], labels=['Slayer', "Support", "Objective Player"])
            # Add labels to each point
            
            texts = []
            for i in range(len(self.df)):
                player = players.loc[self.df['player_id'][i] == players["id"].astype(int)]
                texts.append(plt.text(self.df['average_kills'][i], self.df['average_objective_time'][i], player.iloc[0]['tag']))
            
            adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=1))
            plt.show()
    
    def team_comparison(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT team_rosters.*, player_roles.* FROM public.team_rosters AS team_rosters JOIN public.player_roles AS player_roles ON player_roles.player_id = team_rosters.id;"))
            teams = pd.DataFrame(result.fetchall())
            
            # Step 1: Count the role composition per team
            role_composition = teams.groupby(['team_id', 'role']).size().unstack(fill_value=0).reset_index()
            # print(role_composition)
            
            # Step 2: Add the team name and win rate
            team_stats = (
                teams.groupby('team_id')
                .agg(
                    team_name=('name', 'first'),
                    match_win_percent=('match_win_percent', 'first'),  # Assuming win rates are the same for all players in a team
                    total_game_wins=('game_win_percent', 'first')
                )
                .reset_index()
            )
            # print(team_stats)
            
            # Step 3: Merge role composition with team stats
            result = pd.merge(role_composition, team_stats, on='team_id')
            # print(result)
            
            # Step 1: Create a role composition column
            result['role_composition'] = (
                result['Objective Player'].astype(str) + '-' + 
                result['Slayer'].astype(str) + '-' + 
                result['Support'].astype(str)
            )

            # Step 2: Group by role composition and extract win rates
            composition_groups = [
                group['match_win_percent'].values for _, group in result.groupby('role_composition')
            ]

            # Step 3: Perform ANOVA
            anova_result = f_oneway(*composition_groups)

            # Step 4: Display the results
            print("ANOVA F-statistic:", anova_result.statistic)
            print("ANOVA p-value:", anova_result.pvalue)

            # Step 5: (Optional) Display the grouped data for reference
            print(result[['role_composition', 'match_win_percent']].groupby('role_composition').mean())
            
            # Prepare the independent variables (role counts) and the dependent variable (win_rate)
            X = result[['Slayer', 'Support', 'Objective Player']]
            X = sm.add_constant(X)  # Add a constant for the regression model
            y = result['match_win_percent']

            # Fit the regression model
            model = sm.OLS(y, X).fit()

            # Display the summary
            print(model.summary())
            
            # sns.boxplot(x='Slayer', y='match_win_percent', data=result)
            # plt.title('Win Rates by Slayer Count')
            # plt.show()
            
            # sns.pairplot(result, x_vars=['Slayer', 'Support', 'Objective Player'], y_vars='match_win_percent', kind='reg')
            # plt.show()

    def probabilistic_model(self):
        
        self.calculate_player_rank()
        
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT team_rosters.*, player_roles.* FROM public.team_rosters AS team_rosters JOIN public.player_roles AS player_roles ON player_roles.player_id = team_rosters.id;"))
            rosters = pd.DataFrame(result.fetchall())
            
            result = conn.execute(text(f"select id, team_1_id, team_2_id, team_1_score, team_2_score, winner_id, datetime from public.matches_matches where status = 'complete'"))
            matches = pd.DataFrame(result.fetchall())
            # print(matches)
            
            # Step 1: Count the role composition per team
            print("role comps")
            role_composition = rosters.groupby(['team_id', 'role']).size().unstack(fill_value=0).reset_index()
            print(role_composition)
            # print(role_composition)
            
            # Step 2: Add the team name and win rate
            team_stats = (
                rosters.groupby('team_id')
                .agg(
                    team_name=('name', 'first'),
                    match_win_percent=('match_win_percent', 'first'),  # Assuming win rates are the same for all players in a team
                    total_game_wins=('game_win_percent', 'first')
                )
                .reset_index()
            )
            # print(team_stats)
            
            # Step 3: Merge role composition with team stats
            teams = pd.merge(role_composition, team_stats, on='team_id')
            # print(result)
            
                        # Aggregate ranks to team-level metrics
            team_ranks = self.player_stats.groupby('team_id')['rank'].mean().reset_index()
            team_ranks.rename(columns={'rank': 'avg_team_rank'}, inplace=True)
            
            print(team_ranks)
            
            # Ensure consistent data types
            matches['team_1_id'] = matches['team_1_id'].astype(str)
            matches['team_2_id'] = matches['team_2_id'].astype(str)
            teams['team_id'] = teams['team_id'].astype(str)
            team_ranks['team_id'] = team_ranks['team_id'].astype(str)
            
            teams = teams.merge(team_ranks[['avg_team_rank','team_id']], left_on='team_id', right_on='team_id', how='left')
            
            teams.to_sql("teams_with_ranks", self.engine, if_exists="replace", index=False)
            
            # Merge to add team_1 composition
            matches = matches.merge(teams, left_on='team_1_id', right_on='team_id', how='left')
            matches.drop(columns=['team_id' ], inplace=True)

            # Merge to add team_2 composition
            matches = matches.merge(teams, left_on='team_2_id', right_on='team_id', how='left')
            
            matches.drop(columns=['team_id', 'team_name_x', 'team_name_y' , 'team_1_id', 'team_2_id', 'id', 'datetime'], inplace=True)
            # print(matches.columns)
            
            print(matches)
            
            matches.rename(columns={'Objective Player_y': 'Objective Player_2',
                                    'Slayer_y': 'Slayer_2',
                                    'Support_y': 'Support_2',
                                    "match_win_percent_y": 'match_win_percent_2',
                                    "total_game_wins_y": 'total_game_wins_2',
                                    "avg_team_rank_y": "avg_team_rank_2",
                                    'Objective Player_x': 'Objective Player_1',
                                    'Slayer_x': 'Slayer_1',
                                    'Support_x': 'Support_1',
                                    "match_win_percent_x": 'match_win_percent_1',
                                    "total_game_wins_x": 'total_game_wins_1',
                                    "avg_team_rank_x": "avg_team_rank_1",}, inplace=True)
            
            matches = matches.dropna()
            # print(matches)
            # print(matches.columns)
            
            # Feature engineering
            matches['slayers_diff'] = matches['Slayer_1'].astype(float) - matches['Slayer_2'].astype(float)
            matches['support_diff'] = matches['Support_1'].astype(float) - matches['Support_2'].astype(float)
            matches['objective_diff'] = matches['Objective Player_1'].astype(float) - matches['Objective Player_1'].astype(float)
            matches['win_rate_diff'] = matches['match_win_percent_1'].astype(float) - matches['total_game_wins_2'].astype(float)
            matches['maps_won_diff'] = matches['team_1_score'].astype(float) - matches['team_2_score'].astype(float)
            matches['winner_id'] = matches['winner_id'].astype(float)
            matches['team_rank_diff'] = matches['avg_team_rank_1'].astype(float) - matches['avg_team_rank_2'].astype(float)
            
            # Final feature set
            features = ['slayers_diff', 'support_diff', 'objective_diff', 'win_rate_diff', 'maps_won_diff', 'team_rank_diff']
            target = 'winner_id'
            
            # Train-test split
            X = matches[features]
            y = matches[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and train the Random Forest Classifier
            rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, class_weight='balanced')
            rf_model.fit(X_train, y_train)
            # Predict on the test set
            # y_pred = rf_model.predict(X_test)
            # y_pred_proba = rf_model.predict_proba(X_test) # Probability for Team 1 winning

            # Evaluate on the training set
            y_train_pred = rf_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # Evaluate on the test set
            y_test_pred = rf_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            print("Training Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)

            cv_scores = cross_val_score(rf_model, X, y, cv=5)
            print("Cross-Validation Accuracy:", cv_scores.mean())
            
            # Stratified k-fold cross-validation
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
            print("Cross-Validation Scores:", cv_scores)
            print("Mean Cross-Validation Accuracy:", cv_scores.mean())
            
            # Feature importance
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            # print(feature_importances)

            # Plot feature importance
            plt.barh(feature_importances['Feature'], feature_importances['Importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.show()
            # Evaluate
            # print("Accuracy:", accuracy_score(y_test, y_pred))
            # print("Classification Report:\n", classification_report(y_test, y_pred))

            # Calculate ROC AUC (for binary classification)
            if len(y.unique()) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                print("ROC AUC Score:", roc_auc)
            # Train the logistic regression model
            # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            # model.fit(X_train, y_train)

            # Predict probabilities for test set
            # y_pred_proba = model.predict_proba(X_test)  # This should return a 2D array

            # y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of team_id_1 winningq

            # Evaluate the model
            # print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
            # print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba, multi_class='ovo') )
            # print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

    def model_prediction(self):
        print("running model predicitions")
        team1_ids = None
        team2_ids = None
        with open('upcoming_game.json', 'r') as f:
            data = pd.DataFrame(json.load(f))
            data = data.T 
            team1_ids = data['team1_id'].to_list()
            team2_ids = data['team2_id'].to_list()
            print(data)

        with self.engine.connect() as conn:
            result = conn.execute(text(f"select * from public.team_rosters where team_id = "))
            rosters = pd.DataFrame(result.fetchall())
            # print(rosters)
            team1 = pd.DataFrame()
            team2 = pd.DataFrame()
            for i in range(len(team1_ids)):
                team1_sql = conn.execute(text(f"select * from public.team_rosters where team_id = " + team1_ids[i]))
                team2_sql = conn.execute(text(f"select * from public.team_rosters where team_id = " + team2_ids[i]))
                
                team1 = pd.concat([team1, pd.DataFrame(team1_sql.fetchall())], ignore_index=True)
                team2 = pd.concat([team2, pd.DataFrame(team2_sql.fetchall())], ignore_index=True)


            # merged_df = pd.merge(data, rosters, on='employee_id')
            
            # upcoming_matches = pd.DataFrame({
            #     'slayers_diff': [1.2, -0.5, 0.7],
            #     'support_diff': [0.3, -1.0, 0.6],
            #     'rank_diff': [0.8, -0.4, 0.5],
            #     'win_rate_diff': [0.1, -0.2, 0.15],
            #     'maps_won_diff': [1, -1, 0],
            #     'objective_diff': [0, 0.5, -0.3]
            # })

            
    def player_ranked(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT team_rosters.*, player_roles.* FROM public.team_rosters AS team_rosters JOIN public.player_roles AS player_roles ON player_roles.player_id = team_rosters.id;"))
            rosters = pd.DataFrame(result.fetchall())
            
            # print(rosters)

    def test_get_more_teams(self):
        # Provide the URL or HTML file path containing the table
        url = 'https://liquipedia.net/callofduty/Call_of_Duty_League'

        # Use read_html to scrape all tables from the HTML
        tables = pd.read_html(url)
        # print(tables)
    
    def init(self):
        print("data analysis init")
        self.df = pd.DataFrame()
        self.dbconnector()
        # self.calculate_player_rank()
        # self.get_player_data()
        # self.role_classification()
        # self.plot_role_classification()
        # self.team_comparison()
        # self.probabilistic_model()
        # self.test_get_more_teams()
        # self.player_ranked()
        # self.model_prediction()