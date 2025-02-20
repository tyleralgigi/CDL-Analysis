import json
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import pandas as pd
import requests
from datetime import datetime
import re
import numpy as np
from io import StringIO
from time import sleep
from selenium import webdriver as webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from seleniumwire import webdriver as webdriverwire

class CDL_Worker:
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def iterator(self):
        print("Iterating")
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            self.endpoints = self.config['API']['endpoints']
            self.apikey = self.config['API']['apikey'] #this is the key used in breakpoint api calls (is subject to be removed at any point)
        
        self.getUrlHash()
        
        for endpoint in self.endpoints:
            print(self.config[endpoint]['url'])
            for url in self.config[endpoint]['url']:
                if "REPLACE" in url:
                    self.url = str(url).replace("REPLACE", self.urlHash)
                    print(f"replaced: {self.url}")
                else:
                    self.url = url
                self.tableName = self.config[endpoint]['table_name']
                
                eval(self.config[endpoint]['method'])

    def breakpoint_players(self):
        print("Running breakpoint_players")
        response = requests.post(self.url)
        if response.status_code == 200:
            self.df = pd.DataFrame(response.json())
            # print(self.df)
            self.transform()
            self.loader('replace')
        else:
            print(f"Failed to retrieve webpage: {response.status_code}")
    
    def playerStatsDetials(self):
        print("playerStatsDetials")
        #get matches id, complete date
        with self.engine.connect() as conn:
                print("Getting match dates and ids from breakpoint table")
                
                result = conn.execute(text(f"select id from public.\"matches_matches\" WHERE completed_at != 'None';"))
                df = pd.DataFrame(result.fetchall())
                
                match_ids = df['id'].tolist()
                del df
                
                for id in match_ids:

                    header = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "apikey": self.apikey,
                        "Authorization": "Bearer " + self.apikey
                    }
                    
                    response = requests.get(self.url+str(f"({id})"), headers=header)
                    self.df = pd.json_normalize(response.json())
                    
                    self.check_ids() #Comment this line out on first run to not get table not found error
                    self.transform()
                    self.loader('append')

    def breakpoint_matches(self):
        print("Running breakpoint_matches")
        self.df = pd.DataFrame()
        response = requests.get(self.url)
        if response.status_code == 200:
            print(response.status_code)
            response_df = response.json()
            # Get the keys
            keys = response_df['pageProps'].keys()
            base_table_name = self.tableName
            for key in keys:
                if key not in ['initialColorScheme', 'eventPlacements', 'mainEventId', 'event']:
                    self.tableName = base_table_name + "_" + key
                    try:
                        query = f'select * from public."{self.tableName}"'
                        with self.engine.connect() as conn:
                            print(f"running {query}")
                            result = conn.execute(text(query))
                            self.df = pd.concat([pd.DataFrame(result.fetchall()), pd.DataFrame(response_df['pageProps'][key]).astype(str)], ignore_index=True, axis=0)
                            self.df = self.df.drop_duplicates(subset=['id'], keep='last')
                        self.transform()
                        # print(self.df.size)
                        self.loader('replace')
                    except Exception as e:
                        print("Unable to make data frame for " + key + " : "+ str(e))

            # self.check_ids() #Comment this line out on first run to not get table not found error
        else:
            print(f"Failed to retrieve webpage: {response.status_code}")

    def breakpoint_playerStats(self):
        #Creating rolling averages for players the past 2 games
        #get team ids
        query1 = "select distinct team_id from public.team_rosters"
        
        matchIds = []
        rolling_team_data = []
        #get match id for past 2 games
        with self.engine.connect() as conn:
            result = conn.execute(text(query1)).fetchall()
            
            # Using list comprehension
            team_ids = [item[0] for item in result]
            
            for team_id in team_ids:
                print(team_id)
                query2 = f"""
                    SELECT id, team_1_id, team_2_id
                    FROM public.matches_matches
                    WHERE team_1_id::integer = {team_id} OR team_2_id::integer = {team_id}
                    ORDER BY datetime DESC
                    LIMIT 3;
                """
                result = conn.execute(text(query2)).fetchall()
                matches = np.array(result).tolist()
                
                #3 game rolling average
                matchIds.append(int(matches[0][0]))
                matchIds.append(int(matches[1][0]))
                matchIds.append(int(matches[2][0]))
                
                response = requests.post(self.url, 
                                json = {
                                    "teamId": team_id, 
                                    "matchId": matchIds
                                },
                                headers = {
                                    "Content-Type": "application/json",
                                    "Accept": "application/json",
                                    "apikey": self.apikey,
                                    "Authorization": "Bearer " + self.apikey
                                })
                data = pd.DataFrame(response.json())
                data_averages = pd.DataFrame()
                column_names = ['ctl_bp_rating', 'hp_bp_rating', 'snd_bp_rating', 'ctl_game_count', 'hp_game_count', 'snd_game_count']
                print(data)
                data = data[column_names]
                
                data_averages['ctl_bp_rating_avg'] = data['ctl_bp_rating'] / data['ctl_game_count']
                data_averages['hp_bp_rating_avg'] = data['hp_bp_rating'] / data['hp_game_count']
                data_averages['snd_bp_rating_avg'] = data['snd_bp_rating'] / data['snd_game_count']
                
                data_averages = data_averages[['ctl_bp_rating_avg','hp_bp_rating_avg','snd_bp_rating_avg']].sum()
                rolling_team_data.append([team_id, float(data_averages['ctl_bp_rating_avg'])/3, 
                                    float(data_averages['hp_bp_rating_avg'])/3, float(data_averages['snd_bp_rating_avg'])/3])
            
            rolling_team_data_df = pd.DataFrame(rolling_team_data)
            rolling_team_data_df = rolling_team_data_df.rename(columns={
                    0: 'teamId',
                    1: 'ctl_bp_rating_avg',
                    2: 'hp_bp_rating_avg',
                    3: 'snd_bp_rating_avg'
                })

            self.df = rolling_team_data_df[['ctl_bp_rating_avg','hp_bp_rating_avg','snd_bp_rating_avg']].map(lambda x: np.trunc(x * 1000) / 1000)
            self.df['teamId'] = rolling_team_data_df['teamId']
            self.loader("replace")

    def current_rosters(self):
        self.getUrlHash()
        advanced_json = requests.get(f'https://www.breakingpoint.gg/_next/data/{self.urlHash}/stats/advanced.json')
        advanced_json_data = advanced_json.json()
        self.advanced_all_players = pd.DataFrame(advanced_json_data['pageProps']['allPlayers'])
        # self.advanced_all_teams = pd.DataFrame(advanced_json_data['pageProps']['allTeams'])
        # self.advanced_all_teams.rename(columns={'id': 'team_id'}, inplace=True)
        self.advanced_all_players = self.advanced_all_players[~self.advanced_all_players['tag'].str.contains("KingAbody|Khhx")]
        print(self.advanced_all_players)
        self.advanced_all_players.dropna()
        print(self.advanced_all_players)
        self.advanced_all_players[['id', 'tag', 'current_team_id']].to_sql(self.tableName, self.engine, if_exists="replace", index=False)
    
    def breakpoint_team_stats(self):
        print('')
        self.df = pd.DataFrame()
        
    
    def check_ids(self):
        # Check if the table exists in the database
        with self.engine.connect() as conn:
            print(self.tableName)
            result = conn.execute(
                text(f"SELECT EXISTS (SELECT table_name FROM information_schema.tables WHERE table_name = '{self.tableName}');")
            )
            table_exists = result.scalar()  # Returns True if the table exists, otherwise False

        if not table_exists:
            print(f"Table '{self.tableName}' does not exist in the database. All rows in `self.df` are considered new.")
        else:
            
            with self.engine.connect() as conn:
                # Load the existing table from PostgreSQL into a DataFrame
                db_df = pd.read_sql_table(self.tableName, self.engine)
                
                new_ids = self.df['id'].values.tolist()
                for id in new_ids:
                    if id in db_df['id'].values.tolist():
                            # Delete the row if it's not identical
                            delete_query = text(f'DELETE FROM public."{self.tableName}" WHERE id = :id')
                            conn.execute(delete_query, {"id": id})

    def getRosters(self):
        print("getRosters()")
        self.df = pd.DataFrame()
        with self.engine.connect() as conn:
            result = conn.execute(text(f"select name from public.matches_teams;"))
            df = pd.DataFrame(result.fetchall())
            teams = df['name'].to_list()
            i = 1
            for team in teams:
                print(self.url + str(i) + "/" + str(team).replace(' ', '-')+".json")
                response = requests.get((self.url + str(i) + "/" + str(team).replace(' ', '-')+".json"), headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "apikey": self.apikey,
                            "Authorization": "Bearer " + self.apikey
                })
                
                response_js = response.json()
                if "notFound" not in response_js:
                    response_df = pd.DataFrame(response_js['pageProps']['team']['players'])
                    if len(response_df) != 0:
                        response_df = response_df[['id', 'tag']]
                        response_df['team_id'] = response_js['pageProps']['team']['id']
                        response_df['name'] = response_js['pageProps']['team']['name']
                        response_df['match_wins'] = response_js['pageProps']['standings']['match_wins']
                        response_df['match_losses'] = response_js['pageProps']['standings']['match_losses']
                        response_df['match_win_percent'] = response_js['pageProps']['standings']['match_wins'] / (response_js['pageProps']['standings']['match_wins']+response_js['pageProps']['standings']['match_losses'])
                        response_df['game_wins'] = response_js['pageProps']['standings']['game_wins']
                        response_df['game_losses'] = response_js['pageProps']['standings']['game_losses']
                        response_df['game_win_percent'] = response_js['pageProps']['standings']['game_wins'] / (response_js['pageProps']['standings']['game_wins']+response_js['pageProps']['standings']['game_losses'])
                        # print(response_df)
                        self.df = pd.concat([self.df, response_df])
                else:
                    print("notFound")
                i += 1
        print(self.df)
        self.transform()
        self.loader("replace")
    
    def fandom_current_stats(self):
        tables = pd.read_html(self.url)
        df = tables[4]
        df.columns = df.columns.map('_'.join)
        df.drop('Overall_Unnamed: 0_level_1', axis=1, inplace=True)
        df.columns = df.columns.str.replace('/', '', regex=True)
        df.columns = df.columns.str.replace(' ', '', regex=True)
        df.columns = df.columns.str.replace('&', '_', regex=True)
        df.iloc[:, 0] = df.iloc[:, 0].str.replace(' ', '')
        
        regex = re.compile(r'\d{4}')
        year = regex.findall(self.url)
        print(year)
        df['year'] = year[0]
        
        df.to_sql(self.tableName, self.engine, if_exists="replace", index=False)

    def fandom_past_player_stats(self):
        self.df = pd.DataFrame()

        for url in self.url:
            print(url)
            tables = pd.read_html(url)
            df = tables[4]
            df.columns = df.columns.map('_'.join)
            df.drop('Overall_Unnamed: 0_level_1', axis=1, inplace=True)
            df.columns = df.columns.str.replace('/', '', regex=True)
            df.columns = df.columns.str.replace(' ', '', regex=True)
            df.columns = df.columns.str.replace('&', '_', regex=True)
            
            col_0 = df.columns[0]
            df.iloc[:, 0] = df.iloc[:, 0].str.replace("\xa0", " ")
            
            regex = re.compile(r'\d{4}')
            year = regex.findall(url)
            df['year'] = year[0]
            
            self.df = pd.concat([self.df, df] ,axis=0, ignore_index=True)
        
        
        with self.engine.connect() as conn:
            result = conn.execute(text(f'select distinct id, tag from public."matches_allPlayers"'))
            df1 = pd.DataFrame(result.fetchall())
            # df1['temp'] = df1['tag'].str.slice(1)
            
            # print(df1)
            self.df = pd.merge(self.df, df1, left_on=col_0, right_on='tag', how='left')
            # self.df = self.df.drop(['Search_Destroy_FD', 'Search_Destroy_P', 'Search_Destroy_D.1'], axis=1, inplace=True)
            # df = df[df['id'].notna()]
            
        self.loader("replace")

    def fandom_past_team_stats(self):
        self.df = pd.DataFrame()

        for url in self.url:
            print(url)
            tables = pd.read_html(url)
            df = tables[4]
            df.columns = df.columns.map('_'.join)
            df.drop('Overall_Unnamed: 0_level_1', axis=1, inplace=True)
            df.columns = df.columns.str.replace('/', '', regex=True)
            df.columns = df.columns.str.replace(' ', '', regex=True)
            df.columns = df.columns.str.replace('&', '_', regex=True)
            
            print(df)
            col_0 = df.columns[0]
            df.iloc[:, 0] = df.iloc[:, 0].str.replace("\xa0", " ")
            
            regex = re.compile(r'\d{4}')
            year = regex.findall(url)
            df['year'] = year[0]
            
            self.df = pd.concat([self.df, df] ,axis=0, ignore_index=True)
        
        
        with self.engine.connect() as conn:
            result = conn.execute(text(f'select distinct id, name_short from public.matches_teams'))
            df1 = pd.DataFrame(result.fetchall())
            # df1['temp'] = df1['tag'].str.slice(1)
            
            # print(df1)
            self.df = pd.merge(self.df, df1, left_on=col_0, right_on='name_short', how='left')
            # self.df = self.df.drop(['Search_Destroy_FD', 'Search_Destroy_P', 'Search_Destroy_D.1'], axis=1, inplace=True)
            # df = df[df['id'].notna()]
            
        self.loader("replace")

    def breakpoint_advanced_stats(self):
        
        # Configure WebDriver options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        subset = self.config["breakpoint_advanced_stats"]['subset']
        
        # Initialize WebDriver without Service
        driver = webdriver.Chrome(options=chrome_options)
        for i in range(len(self.url)):
            url = self.url[i]
            driver.get(url)
            
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "mantine-datatable-table"))
            )
            table=pd.read_html(StringIO(element.get_attribute("outerHTML")))
            # print(table[0])
            table[0]['year'] = 2025
            self.df = table[0]
            self.tableName = self.config["breakpoint_advanced_stats"]['table_name'] + "_" + subset[i]
            
            #2024 data has missing teams on breakingpoint site, adding from static CSV file
            if i == 0:
                players_2024 = pd.read_csv('data/breakpoint_data_players.csv')  
                self.df = pd.concat([self.df, players_2024], axis=0)
                self.df.drop(columns=['#'], inplace=True)
            elif i == 1:
                team_2024 = pd.read_csv('data/breakpoint_data_teams.csv')  
                self.df = pd.concat([self.df, team_2024], axis=0)
            else:
                standings_2024 = pd.read_csv('data/breakpoint_data_standings.csv')  
                self.df = pd.concat([self.df, standings_2024], axis=0)
            
            print(self.tableName)
            self.loader("replace")

    def getUrlHash(self):
        self.urlHash = None
        # Configure Selenium WebDriver
        options = webdriverwire.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")  # Disable GPU (for older systems)
        options.add_argument("--no-sandbox")  # Bypass OS security model
        options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        driver = webdriverwire.Chrome(options=options)

        # Navigate to the website
        driver.get("https://www.breakingpoint.gg/stats/teams/advanced")

        print("sleepy time")
        sleep(5)

        # Find the hash in network requests
        for request in driver.requests:
            if "_next/data" in request.url:
                print(request.url)
                self.urlHash = request.url.split("/_next/data/")[1].split("/")[0]
                print(self.urlHash)
                # Extract the hash between "data/" and "/matches.json"
                # hash_value = request.url.split("/_next/data/")[1].split("/matches.json")[0]
                break

        if self.urlHash:
            print(f"Hash Value: {self.urlHash}")
        else:
            print("Hash not found in network requests.")

        # Quit the driver
        driver.quit()
    
    def transform(self):
        print("Transforming - Adding rundate")
        now = datetime.now()
        self.df['rundate'] = now.strftime("%Y%m%d")
    
    def loader(self,method):
        print("Loading into DB")
        # Insert the DataFrame into a table
        # print(self.df)
        if not self.df.empty:
            print(f"adding rows to {self.tableName}")
            self.df.to_sql(self.tableName, self.engine, if_exists=method, index=False)
        else:
            print("empty df, not adding anything")

    def init(self):
        self.urlHash = None
        self.config = None
        self.endpoints = None
        self.apikey = None
        self.dbconnector()
        self.iterator()
        
        