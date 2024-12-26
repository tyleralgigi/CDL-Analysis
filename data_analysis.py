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
import seaborn as sns

class Data_Analysis():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def get_player_data(self):
        print("get_player_data")
        with self.engine.connect() as conn:
            result = conn.execute(text(f"select player_id, player_tag, match_id, kills, deaths, damage, assists, hill_time, plant_count, defuse_count, zone_tier_capture_count, mode_id, map_id from public.\"playerStatsDetails\";"))
            df = pd.DataFrame(result.fetchall())

            df = df.fillna(0)
            print(df['hill_time'])
            if not df.isnull().values.any():  # Shows the count of NaN values in each column
                #Normailzing data
                grouped_stats = df.groupby(['map_id', 'mode_id']).agg({
                    'kills': ['mean', 'std'],
                    'damage': ['mean', 'std'],
                    'assists': ['mean', 'std'],
                    'deaths': ['mean', 'std'],
                    'hill_time': ['mean', 'std'],
                    'zone_tier_capture_count': ['mean', 'std']
                }).reset_index()
                
                grouped_stats.columns = ['map_id', 'mode_id', 
                            'kills_mean', 'kills_std', 
                            'damage_mean', 'damage_std',
                            'assists_mean', 'assists_std',  
                            'deaths_mean', 'deaths_std', 
                            'hill_time_mean', 'hill_time_std', 
                            'zone_tier_capture_count_mean', 'zone_tier_capture_count_std']
                
                grouped_stats.fillna(0, inplace=True)
                
                df = df.merge(grouped_stats, on=['map_id', 'mode_id'])
                df = df.fillna(0)
                df['kills_normalized'] = (df['kills'] - df['kills_mean']) / df['kills_std']
                df['deaths_normalized'] = (df['deaths'] - df['deaths_mean']) / df['deaths_std']
                df['damage_normalized'] = (df['damage'] - df['damage_mean']) / df['damage_std']
                df['assists_normalized'] = (df['assists'] - df['assists_mean']) / df['assists_std']
                df['hill_time_normalized'] = (df['hill_time'] - df['hill_time_mean']) / df['hill_time_std']
                df['zone_tier_capture_count_normalized'] = (df['zone_tier_capture_count'] - df['zone_tier_capture_count_mean']) / df['zone_tier_capture_count_std']
                df['objective_normalized'] = df[['hill_time_normalized', 'zone_tier_capture_count_normalized']].mean(axis=1)
                
                # Drop the original columns that were combined
                df.drop(['hill_time_normalized', 'zone_tier_capture_count_normalized'], axis=1, inplace=True)

                # df['zone_tier_capture_count_normalized'] = (df['zone_tier_capture_count'] - df['zone_tier_capture_count_mean']) / df['zone_tier_capture_count_std']
                df = df.fillna(0)
                self.df = df[['player_id', 'kills_normalized','deaths_normalized',
                            'damage_normalized','assists_normalized','objective_normalized']]
                del df
            else:
                print(df['hill_time'])
                
                
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
            print(role_composition)
            
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
            print(team_stats)
            
            # Step 3: Merge role composition with team stats
            result = pd.merge(role_composition, team_stats, on='team_id')
            print(result)
            
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
                
    def init(self):
        print("data analysis init")
        self.df = pd.DataFrame()
        self.dbconnector()
        # self.get_player_data()
        # self.role_classification()
        # self.plot_role_classification()
        self.team_comparison()
        