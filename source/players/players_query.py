import numpy as np
import pandas as pd

from players_data import *

def draft_class(player):
    pl = draft.loc[draft['Player'] == player]
    ind = pl.index
    pick = int(pl['Pick'][ind])
    year = int(pl['Draft'][ind])
    return pick, year

def players_from_college(college, position = None):
    pfc = []
    if college not in colleges:
        return pfc
    if position == None:
        pl = players1.loc[players1['college'] == college]
    elif position in positions:
        pl = players1.loc[(players1['college'] == college) & (players1['position'] == position)]
    else:
        return pfc

    ind = pl.index
    for ii in ind:
        pfc.append(pl['name'][ii])

    return pfc
