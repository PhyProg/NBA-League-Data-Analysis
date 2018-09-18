import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from draft_data import *

"""
Functions for obtaining player's stat's.
"""

def locate(player):
    return st.loc[st['Player'] == player]

def seasons(player):
    pl = locate(player)
    s = np.array(pl['Season'])
    return s

def stats(player):
    pl = locate(player)
    s = np.array(pl['WS'])
    return s

def seasons_and_stats(player):
    return seasons(player), stats(player)

def total_ws(player):
    tw = 0
    stat = stats(player)
    for w in stat:
        tw += w
    return tw

def total_avg_ws(player):
    stat = stats(player)
    return total_ws/len(stat)

"""
Modeling of player's season.
"""

def LinearRegression(player):
    seas = seasons(player)
    stat = stats(player)

    pl = data.loc[data['Player'] == player]
    yr = int(pl['Yrs'])
    years = np.arange(1, yr+1, 1)

    w = (stat[yr-1]-stat[0])/(yr)
    b = stat[0]

    X = tf.placeholder(tf.float64, name = 'X')
    Y = tf.placeholder(tf.float64, name = 'Y')
    W = tf.Variable(w, name = 'W')
    B = tf.Variable(b, name = 'B')

    Y_p = tf.multiply(X, W) + B
    cost = tf.square(Y - Y_p)
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        for i in range(50):
            for j in range(yr):
                n, l = sess.run([train, cost], feed_dict = {X: stat[j], Y: years[j]})

        w, b = sess.run([W, B])

    return w, b, stat, seas

def WeightedLinearRegression(player,weight = None):
    seas = seasons(player)
    stat = stats(player)

    if weight == None:
        weight = generate_weight(seas, stats)

    pl = data.loc[data['Player'] == player]
    yr = int(pl['Yrs'])
    years = np.arange(1, yr+1, 1)

    w = (stat[yr-1]-stat[0])/(yr)
    b = stat[0]

    X = tf.placeholder(tf.float32, name = 'X')
    Y = tf.placeholder(tf.float32, name = 'Y')
    wgh = tf.placeholder(tf.float32, name = 'w')
    W = tf.Variable(w, name = 'W')
    B = tf.Variable(b, name = 'B')

    Y_p = tf.multiply(X, W) + B
    cost = tf.multiply(tf.square(Y - Y_p), wgh)
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        for i in range(50):
            for j in range(yr):
                n, l = sess.run([train, cost], feed_dict = {X: stat[j], Y: years[j], w: weight[j]})

            w, b = sess.run([W, B])

    return w, b, stats, seas

def generate_weight(st, se):
    w = np.linspace(0,0,len(st))
    return w

def QuadraticRegression(player):
    seas = seasons(player)
    stat = stats(player)

    pl = data.loc[data['Player'] == player]
    yr = int(pl['Yrs'])
    years = np.arange(1, yr+1, 1)

    a = stat[0]
    c = (stat[yr-1]/(years[yr-1]-2*years[int(yr/2)]))
    b = -2*c*years[int(yr/2)]

    X = tf.placeholder(tf.float32, name = 'X')
    Y = tf.placeholder(tf.float32, name = 'Y')
    A = tf.Variable([a,b,c], name = 'A')
    #B = tf.Variable(, name = 'B')
    #C = tf.Variable(stat.mean(), name = 'C')

    Y_p = tf.slice(A, [0], [1])
    for i in range(1,3):
        Y_p = tf.add(Y_p, tf.multiply(tf.pow(X, i), tf.slice(A, [i], [1])))
    cost = tf.reduce_mean(tf.square(Y - Y_p), name = 'cost')
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.00001).minimize(cost)
    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        for i in range(500):
            loss = 0
            for j in range(yr):
                n, l = sess.run([train, cost], feed_dict = {X: stat[j], Y: years[j]})
                loss+=l
            print(loss/yr)

            a, b, c = sess.run(A)

    return a, b, c, stat, years

"""
Functions for analysis of draft class:
"""

def draft_total_years(year):
    i = year-1978
    yrs = 0
    for name in draft_class[i]:
        pl = dr.loc[dr['Player'] == name]
        #print(pl)
        ind = pl.index
        yr = pl['Yrs'][ind[0]]
        #print(yr, yrs)
        if yr > 0:
            yrs += yr
    return yrs

def draft_avg_years(year):
    yrs = draft_total_years(year)
    avg = yrs/len(draft_class[year-1978])
    return avg

def draft_total_WS(year):
    i = year-1978
    if i > 38:
        raise IndexError('List index out of range')
    total_WS = 0
    for name in draft_class[i]:
        pl = st.loc[st['Player'] == name]
        index = pl.index
        for x in index:
            total_WS += pl['WS'][x]
    return total_WS

def draft_yearly_WS(draft_year):
    size = 2016-draft_year
    i = draft_year-1978
    ws = np.linspace(0, 0, size)
    yr = np.linspace(draft_year+1, 2016, size)

    for name in draft_class[i]:
        pl = st.loc[st['Player'] == name]
        index = pl.index
        for x in index:
            s = pl['Season'][x]
            if s <= draft_year:
                continue
            ws[s-draft_year-1] += pl['WS'][x]

    while i < len(ws):
        if ws[i] == 0.0:
            ws = np.delete(ws, i)
            yr = np.delete(yr, i)
            i-=1
        i+=1
        if i<len(ws) and ws[i] == 0:
            while i<len(ws):
                ws = np.delete(ws,i)
                yr = np.delete(yr,i)

    return ws, yr

#print(draft_total_years(1986))
