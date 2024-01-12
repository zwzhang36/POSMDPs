#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:18:22 2023

@author: zhangz31
"""


import copy
import numpy as np
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# from matplotlib import cm
from matplotlib.colors import ListedColormap
# from scipy.stats import expon, norm
# from numpy.random import rand, choice
# from itertools import chain

path = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/code'

# In[]: normalization    
    
def normalize(matrix, baseline=None):
    # keep the sum of each row is equal to 1
    if baseline == None:
        if matrix.ndim == 1:
            matrix = matrix / matrix.sum()
            return matrix
        for i in range(matrix.shape[0]):
            matrix[i,] = matrix[i,] / matrix[i,].sum()
    else:
        if matrix.ndim == 1:
            num_smaller = (matrix <= baseline).sum()
            matrix[matrix < baseline] = baseline
            matrix[matrix > baseline] = matrix[matrix > baseline] / matrix[matrix > baseline].sum() * (
                        1 - num_smaller * baseline)
            return matrix
        for i in range(matrix.shape[0]):
            num_smaller = (matrix[i, :] <= baseline).sum()
            matrix[i, matrix[i, :] > baseline] = matrix[i, matrix[i, :] > baseline] / matrix[
                i, matrix[i, :] > baseline].sum() * (1 - num_smaller * baseline)
            matrix[i, matrix[i, :] < baseline] = baseline
    return matrix

# In[]: functions for plotting transition and observation matrix 
def plot_transition_matrix(*args):
    if len(args) == 1:
        return plot_transition_matrix_single(args[0])
    if len(args) == 2:
        return plot_transition_matrix_double(args[0], args[1])
    
    
def plot_transition_matrix_single(T):
    m, n = T.shape[0], T.shape[1]

    fig = plt.figure()
    plt.imshow(T, cmap='Greys', aspect='equal')
    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, m, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, n + 1, 1))
    ax.set_yticklabels(np.arange(1, m + 1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, m, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)

    return fig


def plot_transition_matrix_double(T1, T2):
    
    m, n = T1.shape[0], T1.shape[1]
    max_ = np.max([T1.max(), T2.max()])
    min_ = np.min([T1.min(), T2.min()])
    assert T1.shape == T2.shape
    assert max_ == 1
    assert min_ == 0
    
    T1, T2  = np.flipud(T1).T, np.flipud(T2).T
    fig, ax = plt.subplots(1, 1)
    fig.title = 'asd'
    for i in range(m):
        for j in range(n):
            v1, v2 = T1[i, j], T2[i, j]
            plt.scatter([i,i,i+1,i+1], [j, j+1, j, j+1], s = 0)
            t1 = plt.Polygon(np.array([[i+1, j], [i, j], [i, j+1]]), 
                             facecolor=[1-v1, 1-0.3*v1, 1-0.3*v1])
            plt.gca().add_patch(t1)
    
            t2 = plt.Polygon([[i+1, j], [i+1, j+1], [i, j+1]], 
                             facecolor=[1-0.4*v2, 1-v2, 1-v2])
            plt.gca().add_patch(t2)
    for i in range(m+1):
        plt.plot([i, i], [0, 10], linewidth=0.2, color = 'grey')
    for i in range(n+1):
        plt.plot([0, 10], [i, i], linewidth=0.2, color = 'grey')
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    plt.xlim([0, m])
    plt.ylim([0, n])

    # Major ticks
    ax.set_xticks(np.arange(0.5, n+0.5, 1))
    ax.set_yticks(np.arange(0.5, m+0.5, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, n + 1, 1))
    ax.set_yticklabels(np.arange(1, m + 1, 1))

    # # Gridlines based on minor ticks
    # ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)

    plt.show()
    return fig
# In[]: function for saving all figures
def plot_save(figs, prefix=[], i = 0):
    # save figs in folder path
    for fig in figs:
        i += 1
        if type(fig) == list:
            plot_save(fig, prefix, i = i)
        else:
            name = prefix + str(i) + '.png' # png
            fig.savefig(path + name)
            # name = prefix + fig._label + '.eps' # png
            # fig.savefig(self.path + name, format = 'eps')
            fig.clf()

# In[]  poz, transition matrix, and colormap
# trnaistion matrix
T = np.zeros((10, 10))
T[0, 1:3] = .5  # trial start - cue left/right
T[1, 3] = 1  # cue left  - L fluid well
T[2, 4] = 1  # cue right - R fluid well
T[3, 5] = 1  # L fluid well - L-reward 1
T[4, 6] = 1  # R fluid well - R-reward 1
T[7:9, 9] = 1  # L/R reward 2 - trial end
T[9, 0] = 1  # ITI - trial start
T[5, 7] = .5  # L-reward 1 - L-reward 2
T[5, 9] = .5  # L-reward 1 - trial end
T[6, 8] = .5  # R-reward 1 - R-reward 2
T[6, 9] = .5  # R-reward 1 - trial end


T1, T2 = copy.deepcopy(T), copy.deepcopy(T)
T2[1, 3:5] = 0.5
T2[2, 3:5] = 0.5

figs = []
figs.append(plot_transition_matrix(T1, T2))


v1 = np.linspace(0,1,256)
v2 = np.linspace(0,1,256)

newcolors_1 = np.array([1-v1, 1-0.3*v1, 1-0.3*v1, 1+v1*0]).T
newcmp_1 = ListedColormap(newcolors_1)
newcolors_2 = np.array([1-0.4*v2, 1-v2, 1-v2, 1+v1*0]).T
newcmp_2 = ListedColormap(newcolors_2)

norm = mpl.colors.Normalize(vmin=0, vmax=1)

for newcmp in [newcmp_1, newcmp_2]:
    fig2, ax = plt.subplots(figsize=(6, 1))
    fig2.subplots_adjust(bottom=0.5)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=newcmp,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Some Units')
    figs.append(fig2)


# In[]  poz, transition matrix, and colormap
state = {'start': 0, 'cue_L': 1, 'cue_R': 2, 'well_L': [3, 4, 5, 6],
        'well_R': [7, 8, 9, 10], 'rwd_L_1': [11, 12, 13, 14],
        'rwd_R_1': [15, 16, 17, 18], 'rwd_L_2': 19,
        'rwd_R_2': 20, 'end': 21}
n_states, n_observations = 22, 7            
T = np.zeros((n_states, n_states))
# trial start - cue left/right
for s in [state['cue_L'], state['cue_R']]:
    T[state['start'], s] = .5
# well - reward 1
r1 = state['rwd_L_1'] + state['rwd_R_1']
well = state['well_L'] + state['well_R']
for i, j in zip(well, r1):
    T[i, j] = 1
# reward 1-short/long/small - trial end
for s in state['rwd_L_1'][:-1] + state['rwd_R_1'][:-1]:
    T[s, state['end']] = 1
# reward 1-big - reward 2
T[state['rwd_L_1'][-1], state['rwd_L_2']] = 1
T[state['rwd_R_1'][-1], state['rwd_R_2']] = 1
# reward 2-big - trial end
T[state['rwd_L_2'], state['end']] = 1
T[state['rwd_R_2'], state['end']] = 1
# trial end - trial start
T[state['end'], state['start']] = 1

for s in state['well_L']:
    T[state['cue_L'], s] = 1 / len(state['well_L'])
for s in state['well_R']:
    T[state['cue_R'], s] = 1 / len(state['well_R'])
    
    
O_null, baseline_O = 0.1, 1e-4
O = np.zeros((n_states, n_observations))
# null observation for all states
O[:, 0] = O_null
# trial start - light on
O[state['start'], 1] = 1 - O_null
# cue left    - cue left
O[state['cue_L'], 2] = 1 - O_null
# cue right   - cue right
O[state['cue_R'], 3] = 1 - O_null
# L/R fluid wl  - well
for s in state['well_L'] + state['well_R']:
    O[s, 4] = 1 - O_null
# L/R-reward 1  - reward
for s in state['rwd_L_1'] + state['rwd_R_1']:
    O[s, 5] = 1 - O_null
# L/R-reward 2  - reward
for s in [state['rwd_L_2'], state['rwd_R_2']]:
    # O[s,   6] = high_p
    O[s, 5] = 1 - O_null
# trial end   - light off
O[state['end'], 6] = 1 - O_null
O = normalize(O, baseline_O)

figs.append(plot_transition_matrix(T))
figs.append(plot_transition_matrix(O))

# In[]  poz, transition matrix, and colormap
state = {'start': 0, 'cue_L': [1, 2, 3, 4], 'cue_R': [5, 6, 7, 8], 
         'well_L': [9, 10, 11, 12], 'well_R': [13, 14, 15, 16],
         'rwd_L_1': [17, 18, 19 ,20], 'rwd_R_1': [21, 22, 23, 24],
         'rwd_L_2': 25, 'rwd_R_2': 26, 'end': 27}

n_states, n_observations = 28, 7            
T = np.zeros((n_states, n_states))
cue = state['cue_L'] + state['cue_R']
well = state['well_L'] + state['well_R']
for i, j in zip(cue, well):
    T[i, j] = 1
# well - reward 1
r1 = state['rwd_L_1'] + state['rwd_R_1']
for i, j in zip(well, r1):
    T[i, j] = 1
# reward 1-short/long/small - trial end
for s in state['rwd_L_1'] + state['rwd_R_1']:
    T[s, state['end']] = 1
T[state['rwd_L_1'][1], state['end']] = 0
T[state['rwd_R_1'][1], state['end']] = 0

# reward 1-big - reward 2
T[state['rwd_L_1'][1], state['rwd_L_2']] = 1
T[state['rwd_R_1'][1], state['rwd_R_2']] = 1
# reward 2-big - trial end
T[state['rwd_L_2'], state['end']] = 1
T[state['rwd_R_2'], state['end']] = 1
# trial end - trial start
T[state['end'], state['start']] = 1
# trial start - cue left/right
for s in cue:
    T[state['start'], s] = 1/len(cue)

    
O_null, baseline_O = 0.1, 1e-4
O = np.zeros((n_states, n_observations))
# null observation for all states
O[:, 0] = O_null
# trial start - light on
O[state['start'], 1] = 1 - O_null
# cue left    - cue left
O[state['cue_L'], 2] = 1 - O_null
# cue right   - cue right
O[state['cue_R'], 3] = 1 - O_null
# L/R fluid wl  - well
for s in state['well_L'] + state['well_R']:
    O[s, 4] = 1 - O_null
# L/R-reward 1  - reward
for s in state['rwd_L_1'] + state['rwd_R_1']:
    O[s, 5] = 1 - O_null
# L/R-reward 2  - reward
for s in [state['rwd_L_2'], state['rwd_R_2']]:
    # O[s,   6] = high_p
    O[s, 5] = 1 - O_null
# trial end   - light off
O[state['end'], 6] = 1 - O_null
O = normalize(O, baseline_O)

figs.append(plot_transition_matrix(T))
figs.append(plot_transition_matrix(O))

plot_save(figs, prefix = 'test')
