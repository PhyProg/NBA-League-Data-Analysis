import numpy as np
import pandas as pd

players1 = pd.read_csv("../../data/players/player_data.csv")
players2 = pd.read_csv("../../data/players/Players.csv")
draft = pd.read_csv("../../data/draft/draft78.csv")
stat = pd.read_csv("../../data/players/Seasons_Stats.csv")
colleges = []
teams = []
positions = ['G', 'G-F', 'F', 'F-C', 'C']
pos_players = [[] for i in range(5)]

def feed_colleges():
    col = []
    ind = players1.index
    for ii in ind:
        if players1['college'][ii] not in col:
            col.append(players1['college'][ii])
    return col

def feed_teams():
    tm = []
    ind = stat.index
    for ii in ind:
        if stat['Tm'][ii] not in tm:
            tm.append(stat['Tm'][ii])
    return tm

def feed_position(pos):
    pls = []
    ind = players1.index
    for ii in ind:
        if players1['position'][ii] == pos:
            pls.append(players1['name'][ii])
    return pls

colleges = feed_colleges()
teams = feed_teams()

#print(teams, len(teams))

for i in range(5):
    pos_players[i] = feed_position(positions[i])

color = {
    'ATL': ['red','black'],
    'BOS': ['green','black'],
    'BRN': ['white','black'],
    'CHI': ['red', 'black'],
    'CLE': ['orange', 'black'],
    'DET': ['blue', 'red'],
    'DAL': ['blue', 'white'],
    'LAL': ['yellow', 'darkviolet'],
    'LAC': ['red', 'blue'],
    'NYK': ['orange', 'blue'],
    'OKC': ['blue', 'orange'],
    'MPH': ['royalblue', 'lightsteelblue'],
    'DNN': ['deepskyblue', 'yellow']
}
