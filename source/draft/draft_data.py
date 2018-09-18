import pandas as pd
import numpy as np
import tensorflow as tf

dr = pd.read_csv('../../data/draft/draft78.csv')
st = pd.read_csv('../../data/draft/season78.csv')
draft_class = [[] for i in range(38)]


ind = st.index

for i in ind:
    l = len(st['Player'][i])
    if st['Player'][i][l-1] == '*':
        new = st['Player'][i][:l-1]
        new = pd.DataFrame({'Player': new}, index = [i])
        st.update(new)

"""
dr: csv file with draft class info - position, player name, years in a league, draft year;
st: csv file with players ratings each season.

dr.keys:
Pick, Player, Yrs, Draft

sr.keys:
Season, Player, WS
data: to concat all data
"""

def feed_data():
    dict = {
        'Player' : dr['Player'],
        'WS': [None for i in range(len(dr))]
    }

    data = dr

    season = [{'Player': dr['Player'],'WS': [None for i in range(len(dr))]} for i in range(39)]
    for pp in range(len(dict['Player'])):
        player = dict['Player'][pp]
        p_data = st.loc[st['Player'] == player]
        #print(p_data)
        ind = p_data.index
        if len(ind) == 0:
            continue
        for i in range(len(p_data)):
            season[p_data['Season'][ind[i]]-1979]['WS'][pp] = p_data['WS'][ind[i]]

    for i in range(39):
        season[i] = pd.DataFrame(season[i])
        key = str(i+1979)
        season[i].rename(columns = {'WS': key}, inplace = True)
        #print(season[i])
        data = pd.concat([data, season[i][key]], axis = 1, join_axes = [data.index])

    return data


def feed_draft_class():
    j = 0
    for i in range(38):
        yr = 1978+i
        while dr['Draft'][j] == yr and j < 3641:
            if dr['Yrs'][j] > 0:
                draft_class[i].append(dr['Player'][j])
            j+=1

#data = feed_data()

feed_draft_class()
