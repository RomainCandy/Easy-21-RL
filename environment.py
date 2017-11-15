#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:20:35 2017

@author: romain
"""


import random as rd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def dealer_strategy(sum_cards):
    """hit while his sum is inferior to 17"""
    while sum_cards<17:
        count = 2*(rd.random()<2/3)-1
        value_card = rd.randint(1,10)
        sum_cards += count*value_card
        if sum_cards<1:
            return 0
    if sum_cards > 21:
        return 0
    return sum_cards
    

def step(state,action):
    """state-> tuple (D,Sum) where D is the dealer first card and Sum is the player Sum or Terminal if terminal
    action -> stick:0 or hit:1"""
    if action == 'stick':
        dealer_card,sum_player = state 
        dealer_count = dealer_strategy(dealer_card)
        if dealer_count == sum_player:
            reward = 0
        elif dealer_count > sum_player:
            reward = -1
        elif dealer_count < sum_player:
            reward = 1
        return 'Terminal',reward
    elif action == 'hit':
        count = 2*(rd.random()<2/3)-1
        value_card = rd.randint(1,10)
        dealer_card,sum_player = state
        new_value = sum_player+count*value_card 
        if new_value<=0 or new_value >=22:
            return 'Terminal',-1
        else:
            return (dealer_card,new_value),0
        
def is_terminal(state):
    return 'T' in state
        

def eps_greedy(state,Q,epsilon):
    """state : Dealers cards, sum_player or Terminal 
    Q: dict(state->list(V_stick,V_hit))
    epsilon controls the exploration"""
    if rd.random()<epsilon:
        return rd.choice(('hit','stick'))
    return ['stick','hit'][np.argmax(Q[state])]


def monte_carlo_control(number_step = 1000000,discount = 1,N0 = 100):
    """initialize Q(state,action) to 0 for all state and action"""
    Q = {(i,j):[0,0] for i in range(1,11) for j in range(1,22)}
    Q['Terminal'] = [0,0]
    counter = {(i,j):[1,1] for i in range(1,11) for j in range(1,22)}
    counter['Terminal'] = [0,0]
    for _ in range(number_step):
        state = rd.randint(1,10),rd.randint(1,10)
        reward_final = 0
        time_step = 0
        epsilon = N0/(N0+sum(counter[state]))
        action = eps_greedy(state,Q,epsilon)
        list_state = [state]
        list_action = [action]
        while True:
            state,reward = step(state,action)
            reward_final += reward*discount**time_step
            if is_terminal(state):
                break
            epsilon = N0/(N0+sum(counter[state]))
            action = eps_greedy(state,Q,epsilon)
            list_state.append(state)
            list_action.append(action)
            time_step += 1
        for state,action in reversed(list(zip(list_state,list_action))):
            if action == 'hit':
                index = 1
            elif action == 'stick':
                index = 0
            counter[state][index] += 1
            Q[state][index] += (reward_final-Q[state][index])/(counter[state][index])

    return Q


def sarsa(lambd, num_episodes = 10000, discount = 1, N0=100 ):
    """initialize Q(state,action) to 0 for all state and action"""
    Q = {(i,j):[0,0] for i in range(1,11) for j in range(1,22)}
    Q['Terminal'] = [0,0]
    counter = {(i,j):[1,1] for i in range(1,11) for j in range(1,22)}
    counter['Terminal'] = [1,1]
    for _ in range(num_episodes):
        """ initialise eligibilty traces"""
        E = {(i,j):[0,0] for i in range(1,11) for j in range(1,22)}
        E['Terminal'] = [0,0]
        state = rd.randint(1,10),rd.randint(1,10)
        epsilon = N0/(N0+sum(counter[state]))
        action = eps_greedy(state,Q,epsilon)
        while not is_terminal(state):
            new_state,reward = step(state,action)
            epsilon = N0/(N0+sum(counter[new_state]))
            new_action = eps_greedy(state,Q,epsilon)
            delta = reward+discount*Q[new_state][new_action == 'hit']-Q[state][action=='hit']
            if action == 'hit':
                index = 1
            elif action == 'stick':
                index = 0
            E[state][index] += 1
            counter[state][index]+=1
            for state in Q:
                Q[state][0] += discount*delta*E[state][0]/counter[state][0]
                Q[state][1] += discount*delta*E[state][1]/counter[state][1]
                E[state][0]*=discount*lambd
                E[state][1]*=discount*lambd
            state = new_state
            action = new_action
    return Q


def convert_Q_to_policy(Q):
    policy = dict()
    for state in Q:
        if not is_terminal(state):
            if np.argmax(Q[state]):
                policy[state] = 'hit'
            else:
                policy[state] = 'stick'
    return policy

def convert_Q_to_value_fonction(Q):
    value_fonction = dict()
    for state in Q:
        value_fonction[state]=np.max(Q[state])
    return value_fonction


def convert_dict_value_to_array_value(dv):
    array_value = np.zeros((10,21))
    for item,value in dv.items():
        if item != 'Terminal':
            D,sum_player = item
            array_value[D-1,sum_player-1]=value
    return array_value

def MSE(X,Y):
    return np.sum((X-Y)**2)/(X.shape[0]*X.shape[1])


def compare():
    best_value = convert_dict_value_to_array_value(convert_Q_to_value_fonction(monte_carlo_control(1000000)))
    diff = list()
    print("best_value computed")
    for lambd in np.arange(0,1.1,0.1):
        print("lambda = ",lambd)
        res = convert_dict_value_to_array_value(convert_Q_to_value_fonction(sarsa(lambd,1000)))
        diff.append(MSE(best_value,res))
    
    x = np.arange(0,1.1,0.1)
    plt.plot(x,diff)
    plt.show()


def prettyPrint(data, tile, zlabel='reward'):
    fig = plt.figure()
    fig.suptitle(tile)
    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []
    for i in range(1, 11):
        for j in range(1, 22):
            axisX.append(i)
            axisY.append(j)
            axisZ.append(data[i, j])
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_zlabel(zlabel)
    
    
def onPolicy():
    MCC = convert_Q_to_value_fonction(monte_carlo_control(100000))
    prettyPrint(MCC, 'MCC', '100000 Episodes')
    plt.show()

def plot_frame(V, ax):
        def get_stat_val(x, y):
            return V[x, y]

        X = np.arange(0,10 , 1)
        Y = np.arange(0,21, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
        return surf
    
def affiche(title):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_zlabel('Reward')
    plot_frame(convert_dict_value_to_array_value(convert_Q_to_value_fonction(monte_carlo_control(100000))),ax)
    plt.show()



def simulation_stick(state):
    D,sum_player = state
    while D < 17:
        if D < 1:
            return 1
        ajout = 2*(rd.random()<2/3)-1
        card = rd.randint(1,10)
        D += ajout*card
    if D > 21 or D < sum_player:
        return 1
    elif D > sum_player:
        return -1
    elif D == sum_player:
        return 0


        
            
        
        
