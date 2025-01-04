import json
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import pandas as pd
import requests
from datetime import datetime

class CDL_Worker:
    def dbconnector(self):
        print("Connecting to DB")
        load_dotenv() # you can specify a location to your .env file as an argument if it's not at your project root
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.engine = create_engine(f'postgresql://{"postgres"}:{self.POSTGRES_PASSWORD}@localhost:{"5432"}/{"Data"}')

    def iterator(self):
        print("Iterating")
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.endpoints = config['API']['endpoints']
            self.apikey = config['API']['apikey'] #this is the key used in breakpoint api calls (is subject to be removed at any point)
                    
        for endpoint in self.endpoints:
            print(config[endpoint]['url'])
            for url in config[endpoint]['url']:
                self.url = url
                self.tableName = config[endpoint]['table_name']
                
                eval(config[endpoint]['method'])

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
                    
                    self.check_ids()
                    self.transform()
                    self.loader('append')

    def breakpoint_matches(self):
        print("Running breakpoint_matches")
        response = requests.get(self.url)
        if response.status_code == 200:
            response_df = response.json()
            # Get the keys
            keys = response_df['pageProps'].keys()
            base_table_name = self.tableName
            for key in keys:
                if key not in ['initialColorScheme', 'eventPlacements', 'mainEventId']:
                    self.tableName = base_table_name + "_" + key
                    try:
                        self.df = pd.DataFrame(response_df['pageProps'][key]).astype(str)
                        # print(self.df)
                        self.check_ids()
                        self.transform()
                        self.loader('append')
                    except Exception as e:
                        print("Unable to make data frame for " + key + " : "+ str(e))
                    
        else:
            print(f"Failed to retrieve webpage: {response.status_code}")

    def check_ids(self):
        # query = f'select * from public."{self.tableName}"'
        # print(query)
        df = pd.read_sql_table(self.tableName, self.engine)
        # print(df)
        if not df.empty:
            with self.engine.connect() as conn:

                filtered_db_df = self.df[~self.df['id'].isin(df['id'])]
                print("compared and len is: " + str(len(self.df)))
                result = conn.execute(text(f'DELETE FROM public."{self.tableName}" WHERE id = ANY (ARRAY[{', '.join(f"'{id}'" for id in filtered_db_df)}]);'))
                print(result)

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
        
        df.to_sql(self.tableName, self.engine, if_exists="replace", index=False)

    def fandom_past_stats(self):
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
            self.df = pd.concat([self.df, df] ,axis=0, ignore_index=True)

        self.loader("replace")

    def transform(self):
        print("Transforming - Adding rundate")
        now = datetime.now()
        self.df['rundate'] = now.strftime("%Y%m%d")
    
    def loader(self,method):
        print("Loading into DB")
        # Insert the DataFrame into a table
        self.df.to_sql(self.tableName, self.engine, if_exists=method, index=False)

    def init(self):
        self.dbconnector()
        self.iterator()
        
        