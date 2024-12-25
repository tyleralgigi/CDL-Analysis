import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

class Data_Analysis():
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def get_player_data(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"select player_id, player_tag, match_id, kills, deaths, damage, assists, hill_time, plant_count, defuse_count, zone_tier_capture_count, mode_id, map_id from public.\"playerStatsDetails\";"))
            df = pd.DataFrame(result.fetchall())
            
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
                        'assists_mean', 'dassists_std',  
                        'deaths_mean', 'deaths_std', 
                        'hill_time_mean', 'hill_time_std', 
                        'zone_tier_capture_count_mean', 'zone_tier_capture_count_std']
            
            grouped_stats.fillna(0, inplace=True)
            
            df = df.merge(grouped_stats, on=['map_id', 'mode_id'])
            df['kills_normalized'] = (df['kills'] - df['kills_mean']) / df['kills_std']
            df['deaths_normalized'] = (df['deaths'] - df['deaths_mean']) / df['deaths_std']
            df['hill_time_normalized'] = (df['hill_time'] - df['hill_time_mean']) / df['hill_time_std']
            df['zone_tier_capture_count_normalized'] = (df['zone_tier_capture_count'] - df['zone_tier_capture_count_mean']) / df['zone_tier_capture_count_std']
            df.fillna(0, inplace=True)
            print(df)

    def init(self):
        self.dbconnector()
        self.get_player_data()
        