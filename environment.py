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

rd.seed(9001)


class State:
    def __init__(self):
        self.dealer_card = rd.randint(1,10)
        self.sum_player = rd.randint(1,10)

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
    res = State()
    res.dealer_card = state.dealer_card
    if action == 0:
        dealer_count = dealer_strategy(state.dealer_card)
        if dealer_count == state.sum_player:
            reward = 0
        elif dealer_count > state.sum_player:
            reward = -1
        elif dealer_count < state.sum_player:
            reward = 1
        return 'Terminal',reward
    elif action == 1:
        count = 2*(rd.random()<2/3)-1
        value_card = rd.randint(1,10)
        res.sum_player = state.sum_player +count*value_card 
        if res.sum_player<=0 or res.sum_player >=22:
            return 'Terminal',-1
        else:
            return res,0
        
def is_terminal(state):
    return state == 'Terminal'
        

def eps_greedy(state,Q,epsilon):
    """state : Dealers cards, sum_player 
    Q: triple array D,C,V
    epsilon controls the exploration"""
    dealer_id = state.dealer_card - 1
    player_id = state.sum_player - 1
    if rd.random()<epsilon:
        return rd.randint(0,1)
    return np.argmax(Q[dealer_id,player_id,:])


def monte_carlo_control(number_episode = 1000000,discount = 1,N0 = 100):
    """initialize Q(state,action) to 0 for all state and action"""
    Q = np.zeros((10,21,2),dtype=float)
    counter = np.ones((10,21,2),dtype=int)
    for _ in range(number_episode):
        state = State()
        state.dealercard = rd.randint(1,10)
        state.playersum = rd.randint(1,10)
        list_state_action = []
        list_reward = []
        while not is_terminal(state):
            epsilon = N0/(N0+np.sum(counter[
                    state.dealer_card-1,state.sum_player-1]))
            action = eps_greedy(state,Q,epsilon)
            list_state_action.append(
                    (state.dealer_card,state.sum_player,action))
            state,reward = step(state,action)
            list_reward.append(reward)
        assert len(list_reward) == len(list_state_action)
        sum_reward = 0
        for (stateD,stateP,action),reward in zip(reversed(
                list_state_action),reversed(list_reward)):
            counter[stateD-1,stateP-1,action] += 1
            sum_reward = discount*sum_reward + reward
            Q[stateD-1,stateP-1,action] += (
                    sum_reward-Q[stateD-1,stateP-1,action])/(
                    counter[stateD-1,stateP-1,action])
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return np.max(Q,axis=2), np.argmax(Q,axis=2)

def sarsa(lambd, num_episodes = 1000000, discount = 1, N0=100 ):
    """initialize Q(state,action) to 0 for all state and action"""
    Q = np.zeros((10,21,2),dtype=float)
    counter = np.ones((10,21,2),dtype=int)
    for _ in range(num_episodes):
        """ initialise eligibilty traces"""
        elligibility = np.zeros((10,21,2),dtype=float)
        state = State()
        state.dealer_card = rd.randint(1,10)
        state.sum_player = rd.randint(1,10)
        epsilon = N0/(N0+np.sum(counter[
                state.dealer_card-1,state.sum_player-1]))
        action = eps_greedy(state,Q,epsilon)
        new_action = action
        while not is_terminal(state):
            counter[state.dealer_card-1,
                    state.sum_player-1,action] += 1
            new_state,reward = step(state,action)
            q = Q[state.dealer_card-1,
                  state.sum_player-1,action]
            if not is_terminal(new_state):
                epsilon = N0/(N0+np.sum(counter[
                        new_state.dealer_card-1,
                        new_state.sum_player-1]))
                new_action = eps_greedy(new_state,Q,epsilon)
                new_q = Q[new_state.dealer_card-1,
                          new_state.sum_player-1,new_action]
                delta = reward + discount*new_q - q
            else:
                delta = reward - q
            elligibility[state.dealer_card-1,
                         state.sum_player-1,action] += 1
            alpha = 1/counter[state.dealer_card-1,
                              state.sum_player-1,action]
            update = alpha*delta*elligibility
            Q += update
            elligibility *= lambd
            state = new_state
            action = new_action
            
            
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    return np.max(Q,axis=2), np.argmax(Q,axis=2)


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
    best_value = monte_carlo_control(1000000)[0]
    diff = list()
    print("best_value computed")
    for lambd in np.arange(0,1.1,0.1):
        print("lambda = ",lambd)
        res = sarsa(lambd,50000)[0]
        diff.append(MSE(best_value,res))
    
    x = np.arange(0,1.1,0.1)
    plt.plot(x,diff)
    plt.show()

compare()

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


            
        
        
