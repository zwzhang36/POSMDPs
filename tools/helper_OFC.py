#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhewei Zhang
    
Created on Sun Jan 29 16:06:08 2023

"""

from typing import Any

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
from numpy.random import rand, choice
from itertools import chain

# parameters for dwell time
dt = 0.1  # unit: second
t_range = 7  # unit: second


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            if type(my_dict[key]) == dict:
                my_dict[key] = Dict2Class(my_dict[key])
            setattr(self, key, my_dict[key])


def assign(t_event):
    temp = np.zeros(int(t_range / dt))
    if type(t_event) in [float, int]:
        temp[round(t_event / dt)] = 1
    elif type(t_event) == list:
        for i in t_event:
            temp[round(i / dt)] = 1 / len(t_event)
    return temp


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


def plot_transition_matrix(T):
    m, n = T.shape[0], T.shape[1]

    fig = plt.figure()
    im = plt.imshow(T, cmap='Greys', aspect='equal')

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


class odorTask(object):
    """
    it is consistent with the task setting, used for generate the task sequence and control animals
    condition:
        1: long:  one drop with long delay
        2: short: one drop with short delay
        3: big:   two drops with short delay
        4: small: one drop with short delay
        'agent', kind of like the average of all conditions used in task
    Add two additional state for entering the fluid well
    """

    def __init__(self, timing, params=[], task_space='poz'):
        """
        :param timing:
        :param task_space: 'poz': left / right well
                            'poz_rwd': left/right + long/big/common
        """
        self.task_space = params.state_map if params != [] else task_space
        self.params = params
        self.timing = timing
        if self.task_space == 'poz':
            '''assume the agent learn the dwell distribution and value of each state'''
            self.n_states = 10
            self.r_obs = [5, 6]
            self.n_observations = 8
            self.state = {'start': [0], 'cue_L': [1], 'cue_R': [2], 'well_L': [3], 'well_R': [4],
                          'rwd_L_1': [5], 'rwd_R_1': [6], 'rwd_L_2': [7], 'rwd_R_2': [8], 'end': [9]}
        elif self.task_space == 'poz_rwd':
            '''assume the dwell distribution is known, and agent leans the transition among states'''
            self.n_states = 37
            self.r_obs = [5, 6]
            self.n_observations = 8
            self.state = {'start': [0, 1, 2, 3],
                          'cue_L': [4, 5, 6, 7], 'cue_R': [8, 9, 10, 11], 
                          'well_L': [12, 13, 14, 15], 'well_R': [16, 17, 18, 19],
                          'rwd_L_1': [20, 21, 22, 23], 'rwd_R_1': [24, 25, 26, 27],
                          'rwd_L_2': [28, 29, 30, 31], 'rwd_R_2': [32, 33, 34, 35], 'end': [36, 36, 36, 36]}

    def get_dwell_times(self, condition='agent'):
        if self.task_space == 'poz':
            D = self.get_dwell_times_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            D = self.get_dwell_times_PozRwd(condition=condition)
        return D

    def get_transition_matrix(self, condition='agent', baseline_big_T=1e-3):
        if self.task_space == 'poz':
            T = self.get_transition_matrix_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            T = self.get_transition_matrix_PozRwd(condition=condition, baseline=baseline_big_T)
        return T

    def get_observation_matrix(self, condition='agent', baseline_cue=1):
        if self.task_space == 'poz':
            O = self.get_observation_matrix_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            O = self.get_observation_matrix_PozRwd(condition=condition, baseline_cue=baseline_cue)
        return O

    def get_info(self, condition, left_right=None, baseline_cue=1, baseline_big_T=1e-3):
        """
        :param left_right: 0 for left trials, 1 for right trials
        :param block: 1/2/3/4: long/short/big/small
        :return:
        """
        if self.task_space == 'poz':
            T = self.get_transition_matrix(condition=condition)
            if left_right == 0:
                T[0, 1] = 1  # trial start - cue left
                T[0, 2] = 0  # trial start - cue right
            elif left_right == 1:
                T[0, 1] = 0  # trial start - cue left
                T[0, 2] = 1  # trial start - cue right
        elif self.task_space == 'poz_rwd':
            T = self.get_transition_matrix(condition=condition,
                                           baseline_big_T=baseline_big_T)
        return T, self.get_observation_matrix(condition=condition, baseline_cue=baseline_cue), self.get_dwell_times(condition=condition)

    def get_transition_matrix_Poz(self, condition='agent'):
        """
        Transition matrix  (from row to column), 10 states
        States: 0:trial starts, 1:cue L, 2:cue R, 3:left well , 4:right well,
            5:L-reward 1, 6: R-reward 1, 7: L-reward 2, 8: R-reward 2, 9:trial end
        """
        T = np.zeros((self.n_states, self.n_states))
        T[0, 1:3] = .5  # trial start - cue left/right
        T[1, 3] = 1  # cue left  - L fluid well
        T[2, 4] = 1  # cue right - R fluid well
        T[3, 5] = 1  # L fluid well - L-reward 1
        T[4, 6] = 1  # R fluid well - R-reward 1
        T[7:9, 9] = 1  # L/R reward 2 - trial end
        T[9, 0] = 1  # ITI - trial start
        if condition == 'agent':
            T[5, 7] = .5  # L-reward 1 - L-reward 2
            T[5, 9] = .5  # L-reward 1 - trial end
            T[6, 8] = .5  # R-reward 1 - R-reward 2
            T[6, 9] = .5  # R-reward 1 - trial end
        elif condition == 3:
            T[5, 7] = 1  # L-reward 1 - L-reward 2
            T[6, 8] = 1  # R-reward 1 - R-reward 2
        elif condition in [1, 2, 4]:
            T[5, 9] = 1  # L-reward 1 - trial end
            T[6, 9] = 1  # R-reward 1 - trial end
        else:
            raise
        return T

    def get_transition_matrix_PozRwd(self, condition='agent', baseline = 1e-3):
        """
        Transition matrix  (from row to column)
        States: 0:trial starts, 1:cue L, 2:cue R,
            3:left well-common, 4:left well-big, 5:left well-long,
            6:right well-common, 7:right well-big, 8:right well-long,
            9:L-reward-common 1, 10:L-reward 1-big, 11:L-reward 1-long,
            12:R-reward-common 1, 13:R-reward 1-big, 14:R-reward 1-long,
            15:L-reward-big 2, 16: R-reward-big 2, 17:trial end
        """
        T = np.zeros((self.n_states, self.n_states))
        # block 1: left short / right long; block 2: left long /right short
        # block 3: left big / right small; block 4 left small/ right big
        # trial start - cue left/right
        for i in range(len(self.state['start'])):
            T[self.state['start'][i], self.state['cue_L'][i]] = 0.5 
            T[self.state['start'][i], self.state['cue_R'][i]] = 0.5  
        # cue - well
        cue = self.state['cue_L'] + self.state['cue_R']
        well = self.state['well_L'] + self.state['well_R']
        for i, j in zip(cue, well):
            T[i, j] = 1
        # well - reward 1
        r1 = self.state['rwd_L_1'] + self.state['rwd_R_1']
        for i, j in zip(well, r1):
            T[i, j] = 1
        # trial end - trial start
        for s1, s2 in zip(self.state['end'], self.state['start']):
            if len(set(self.state['end'])) == len(self.state['end']):
                T[s1, s2] = 1
            elif len(set(self.state['end'])) == 1:
                T[s1, s2] = 1/len(self.state['end'])
            else:
                print('*'*40)
                print('unknown situation, check it')
                print('*'*40)
                
        if condition == 'agent':
            # reward 1-short/long/small - trial end
            for i in range(len(self.state['rwd_L_1'])):
                T[self.state['rwd_L_1'][i], self.state['end'][i]] = 1 - baseline
                T[self.state['rwd_R_1'][i], self.state['end'][i]] = 1 - baseline
                T[self.state['rwd_L_2'][i], self.state['end'][i]] = baseline
                T[self.state['rwd_R_2'][i], self.state['end'][i]] = baseline
                T[self.state['rwd_L_1'][i], self.state['rwd_L_2'][i]] = baseline
                T[self.state['rwd_R_1'][i], self.state['rwd_R_2'][i]] = baseline
            
            T[self.state['rwd_L_1'][2], self.state['end'][2]] = baseline
            T[self.state['rwd_R_1'][3], self.state['end'][3]] = baseline
            # reward 1-big - reward 2
            T[self.state['rwd_L_1'][2], self.state['rwd_L_2'][2]] = 1 - baseline
            T[self.state['rwd_R_1'][3], self.state['rwd_R_2'][3]] = 1 - baseline
            # reward 2-big - trial end
            T[self.state['rwd_L_2'][2], self.state['end'][2]] = 1 - baseline
            T[self.state['rwd_R_2'][3], self.state['end'][3]] = 1 - baseline
            
        else:
            # reward 1-short/long/small - trial end
            for i in range(len(self.state['rwd_L_1'])):
                T[self.state['rwd_L_1'][i], self.state['end'][i]] = 1
                T[self.state['rwd_R_1'][i], self.state['end'][i]] = 1
                
            T[self.state['rwd_L_1'][2], self.state['end'][2]] = 0
            T[self.state['rwd_R_1'][3], self.state['end'][3]] = 0
            # reward 1-big - reward 2
            T[self.state['rwd_L_1'][2], self.state['rwd_L_2'][2]] = 1
            T[self.state['rwd_R_1'][3], self.state['rwd_R_2'][3]] = 1
            # reward 2-big - trial end
            T[self.state['rwd_L_2'][2], self.state['end'][2]] = 1
            T[self.state['rwd_R_2'][3], self.state['end'][3]] = 1
        return T

    def get_observation_matrix_Poz(self, condition='agent'):
        # observation_matrix (the probability of observing dim3 when transiting from dim 1 to dim 2)
        # observations: 0: a null observation; 1:light on (trial start); 2/3 two odor cues;
        #               4: for both fluid well; 5/6:first/second rewards; 7:light off (trial end);
        if condition == 'task' or condition in [1, 2, 3, 4]:
            O = np.zeros((self.n_states, self.n_states, self.n_observations))
            O[9, 0, 1] = 1  # trial end - trial start - light on
            O[0, 1, 2] = 1  # trial start - cue left  - cue left
            O[0, 2, 3] = 1  # trial start - cue right - cue right
            O[1, 3, 4] = 1  # cue left  - L fluid wl  - well
            O[2, 4, 4] = 1  # cue right - R fluid wl  - well
            O[3, 5, 5] = 1  # L fluid wl - L-reward 1 - reward
            O[4, 6, 5] = 1  # R fluid wl - R-reward 1 - reward
            O[5, 7, 5] = 1  # L-reward 1 - L-reward 2 - reward
            O[6, 8, 5] = 1  # R-reward 1 - R-reward 2 - reward
            O[5:9, 9, 7] = 1  # L/R-reward 1/2 - trial end  - light off
            O[np.sum(O[:, :, 1:], axis=2) == 0, 0] = 1  # null
        elif condition == 'agent':
            high_p = 1 - self.params.baseline.O_null
            null_p = self.params.baseline.O_null
            O = np.zeros((self.n_states, self.n_observations))

            O[:, 0] = null_p  # null observation for all states
            O[0, 1] = high_p  # trial start - light on
            O[1, 2] = high_p  # cue left    - cue left
            O[2, 3] = high_p  # cue right   - cue right
            O[3:5, 4] = high_p  # L/R fluid wl  - well
            O[5:7, 5] = high_p  # L/R-reward 1  - reward
            O[7:9, 5] = high_p  # L/R-reward 2  - reward
            O[9, 7] = high_p  # trial end   - light off
            O = normalize(O, self.params.baseline.O)

            O[1, :] = 0  # cue left    - cue left
            O[1, 0] = null_p  # cue left    - cue left
            O[1, 2] = high_p  # cue left    - cue left
            O[2, :] = 0  # cue right   - cue right
            O[2, 0] = null_p  # cue right   - cue right
            O[2, 3] = high_p  # cue right   - cue right
            O[1, :] = normalize(O[1, :], self.params.baseline.O/1e10)
            O[2, :] = normalize(O[2, :], self.params.baseline.O/1e10)

        return O

    def get_observation_matrix_PozRwd(self, condition='agent', baseline_cue=1):
        # observation_matrix (the probability of observing dim3 when transiting from dim 1 to dim 2)
        # observations: 0: a null observation; 1:light on (trial start); 2/3 two odor cues;
        #               4: for both fluid well; 5/6:first/second rewards; 7:light off (trial end);
        # block 1: left short / right long; block 2: left long /right short
        # block 3: left big / right small; block 4 left small/ right big

        if condition == 'task' or condition in [1, 2, 3, 4]:
            O = np.zeros((self.n_states, self.n_states, self.n_observations))
            # trial end - trial start - light on
            for s1, s2 in zip(self.state['end'], self.state['start']):
                O[s1, s2] = 1
            # trial start - cue left  - cue left
            for s1, s2 in zip(self.state['start'], self.state['cue_L']):
                O[s1, s2, 2] = 1
            # trial start - cue right - cue right
            for s1, s2 in zip(self.state['start'], self.state['cue_R']):
                O[s1, s2, 3] = 1
            # cue left  - L fluid wl  - well
            for s1, s2 in zip(self.state['cue_L'], self.state['well_L']):
                O[s1, s2, 4] = 1
            # cue right - R fluid wl  - well
            for s1, s2 in zip(self.state['cue_R'], self.state['well_R']):
                O[s1, s2, 4] = 1
            # L fluid wl - L-reward 1 - reward
            for i, j in zip(self.state['well_L'], self.state['rwd_L_1']):
                O[i, j, 5] = 1
            # R fluid wl - R-reward 1 - reward
            for i, j in zip(self.state['well_R'], self.state['rwd_R_1']):
                O[i, j, 5] = 1

            # O[self.state['rwd_L_1'][1], self.state['rwd_L_2'],  6] = 1  # L-reward - L-reward 2 - reward
            # O[self.state['rwd_R_1'][1], self.state['rwd_R_2'],  6] = 1  # R-reward - R-reward 2 - reward
            # L/R-reward 1/2 - trial end  - light off
            for i in range(len(self.state['end'])):
                O[self.state['rwd_L_1'][i], self.state['end'][i], 7] = 1      # L-reward 1- trial end - light off
                O[self.state['rwd_R_1'][i], self.state['end'][i], 7] = 1      # L-reward 1- trial end - light off
                O[self.state['rwd_L_2'][i], self.state['end'][i], 7] = 1      # L-reward 2- trial end - light off
                O[self.state['rwd_R_2'][i], self.state['end'][i], 7] = 1      # L-reward 2- trial end - light off
                O[self.state['rwd_L_1'][i], self.state['rwd_L_2'][i], 5] = 1  # L-reward - L-reward 2 - reward
                O[self.state['rwd_R_1'][i], self.state['rwd_R_2'][i], 5] = 1  # R-reward - R-reward 2 - reward

            O[np.sum(O[:, :, 1:], axis=2) == 0, 0] = 1  # null
            
        elif condition == 'agent':
            high_p = 1 - self.params.baseline.O_null
            null_p = self.params.baseline.O_null
            base_p = self.params.baseline.O
            O = np.zeros((self.n_states, self.n_observations))
            # null observation for all states
            O[:, 0] = null_p
            # trial start - light on
            for s in self.state['start']:
                O[s, 1] = high_p
            # cue left    - cue left
            for s in self.state['cue_L']:
                O[s, 2] = high_p
            # cue right   - cue right
            for s in self.state['cue_R']:
                O[s, 3] = high_p
            # L/R fluid wl  - well
            for s in self.state['well_L'] + self.state['well_R']:
                O[s, 4] = high_p
            # L/R-reward 1  - reward
            for s in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                O[s, 5] = high_p
            # L/R-reward 2  - reward
            for s in self.state['rwd_L_2'] + self.state['rwd_R_2']:
                # O[s,   6] = high_p
                O[s, 5] = high_p
            # trial end   - light off
            for s in self.state['end']:
                O[s, 7] = high_p
            O = normalize(O, base_p)
            
            # added by zzw, 2023/01/15
            # cue left    - cue left
            for s in self.state['cue_L']:
                O[s, :] = 0
                O[s, 2] = high_p
                O[s, 0] = null_p
                O[s, :] = normalize(O[s, :], base_p/baseline_cue)
            # cue right   - cue right
            for s in self.state['cue_R']:
                O[s, :] = 0
                O[s, 3] = high_p
                O[s, 0] = null_p
                O[s, :] = normalize(O[s, :], base_p/baseline_cue)
            # for o in [1, 2]:
            #     O[:, o] = normalize(O[:, o], self.params.baseline.O/1e5)

        return O

    def get_dwell_times_Poz(self, condition='agent'):
        # dwell time (for agents)
        short, long = self.timing.short, self.timing.long
        average = (short + long) / 2

        if condition == 'agent':
            dwell_std = self.params.dwell_std
            baseline = expon.pdf(np.arange(dt, t_range + dt, dt), 0, t_range)
            rwd_expection = norm.pdf(np.arange(dt, t_range + dt, dt), average, average * dwell_std)
            D = {i: baseline for i in range(self.n_states)}
            D[3] = rwd_expection
            D[4] = rwd_expection
            for key in D.keys():
                D[key] = normalize(D[key], self.params.baseline.D)
            return D

        D = {0: assign(0.5), 1: assign(0.5), 2: assign(0.5),
             5: assign(1), 6: assign(1), 9: assign(0.5)}
        if condition == 1:  # 1:long
            D[3] = assign(long)
            D[4] = assign(long)
        elif condition in [2, 4]:  # 2:short; 4: small
            D[3] = assign(short)
            D[4] = assign(short)
        elif condition == 3:  # 3:big
            D[3] = assign(short)
            D[4] = assign(short)
            D[5] = assign(0.5)
            D[6] = assign(0.5)
            D[7] = assign(0.5)
            D[8] = assign(0.5)
        else:
            raise
        return D

    def get_dwell_times_PozRwd(self, condition='agent'):
        D = {}  # dwell time
        short, long = self.timing.short, self.timing.long

        def dwell_func(x):
            return norm.pdf(np.arange(dt, t_range + dt, dt), x, x * dwell_std)
        # block 1: left short / right long; block 2: left long /right short
        # block 3: left big / right small; block 4 left small/ right big
        # for agents
        if condition == 'agent':
            dwell_std = self.params.dwell_std
            # trial start and end
            for i in self.state['start']+self.state['end']:
                D[i] = dwell_func(short)
            # cue
            for i in self.state['cue_L'] + self.state['cue_R']:
                D[i] = dwell_func(short)
            # wells
            for i in self.state['well_L'] + self.state['well_R']:
                D[i] = dwell_func(short)
            # well indicating long delay
            for i in [self.state['well_L'][1], self.state['well_R'][0]]:
                D[i] = dwell_func(long)
            # reward 1
            for i in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                D[i] = dwell_func(short*2)
            # reward 1 - big reward
            for i in [self.state['rwd_L_1'][2], self.state['rwd_R_1'][3]]:
                D[i] = dwell_func(short)
            # reward 2
            for i in self.state['rwd_L_2'] + self.state['rwd_R_2']:
                D[i] = dwell_func(short*2)
            # reward 2
            for i in self.state['rwd_L_2'] + self.state['rwd_R_2']:
                D[i] = dwell_func(short*2)

            for key in D.keys():
                D[key] = normalize(D[key], self.params.baseline.D)
        else:
            # for task, to generate the task event sequence
            # trial start and end
            for i in self.state['start']+self.state['end']:
                D[i] = assign(short)
            # cue
            for i in self.state['cue_L'] + self.state['cue_R']:
                D[i] = assign(short)
            # wells
            for i in self.state['well_L'] + self.state['well_R']:
                D[i] = assign(short)
            # well indicating long delay
            for i in [self.state['well_L'][1], self.state['well_R'][0]]:
                D[i] = assign(long)
            # reward 1
            for i in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                D[i] = assign(short*2)
            # reward 1 - big reward
            for i in [self.state['rwd_L_1'][2], self.state['rwd_R_1'][3]]:
                D[i] = assign(short)
            # reward 2
            for i in self.state['rwd_L_2'] + self.state['rwd_R_2']:
                D[i] = assign(short*2)
        return D


class odorTaskLesion(odorTask):
    def __init__(self, timing, params=[]):
        super(odorTaskLesion, self).__init__(timing, params=params)
        self.largerP = params.largerP
        # overwrite the function definition in the parent class
        self.get_dwell_times = self.get_dwell_times
        self.get_transition_matrix = self.get_transition_matrix
        self.get_observation_matrix = self.get_observation_matrix

    def get_info(self, condition, left_right=None, baseline_cue=1, baseline_big_T=1e-3):
        """
        :param left_right: 0 for left trials, 1 for right trials
        :param block: 1/2/3/4: long/short/big/small
        :return:
        """
        T = self.get_transition_matrix(condition=condition, baseline_big_T=baseline_big_T)
        if self.task_space == 'poz':
            if left_right == 0:
                T[0, 1] = 1  # trial start - cue left
                T[0, 2] = 0  # trial start - cue right
            elif left_right == 1:
                T[0, 1] = 0  # trial start - cue left
                T[0, 2] = 1  # trial start - cue right
        elif self.task_space == 'poz_rwd':
            # trial start - cue
            if left_right == 0:
                for s1, s2 in zip(self.state['start'], self.state['cue_L']):
                    T[s1, s2] = 1
                for s1, s2 in zip(self.state['start'], self.state['cue_R']):
                    T[s1, s2] = 0 
            elif left_right == 1:
                for s1, s2 in zip(self.state['start'], self.state['cue_L']):
                    T[s1, s2] = 0
                for s1, s2 in zip(self.state['start'], self.state['cue_R']):
                    T[s1, s2] = 1 
        return T, self.get_observation_matrix(condition=condition, baseline_cue=baseline_cue), self.get_dwell_times(condition=condition)

    def get_transition_matrix(self, condition='agent', baseline_big_T=1e-3):
        # print('return lesion matrix')
        if self.task_space == 'poz_rwd':
            T = super().get_transition_matrix(condition=condition, baseline_big_T=baseline_big_T)
            # OFC lesion, equal transition to left and right well
            cue_L,  cue_R  = self.state['cue_L'], self.state['cue_R']
            well_L, well_R = self.state['well_L'], self.state['well_R']
            for cl, cr, wl, wr in zip(cue_L, cue_R, well_L, well_R):
                tmp = (T[cl, wl] + T[cl, wr])/2
                T[cl, wl] = tmp
                T[cl, wr] = tmp
                
                tmp2 = (T[cr, wl] + T[cr, wr])/2
                T[cr, wl] = tmp2
                T[cr, wr] = tmp2
                
            # # for OFC lesion
            # reward 1-short/long/small - trial end
            for i in range(2, 4):
                T[self.state['rwd_L_1'][i], self.state['end'][i]] = 0.5
                T[self.state['rwd_R_1'][i], self.state['end'][i]] = 0.5
                T[self.state['rwd_L_2'][i], self.state['end'][i]] = 1
                T[self.state['rwd_R_2'][i], self.state['end'][i]] = 1
                T[self.state['rwd_L_1'][i], self.state['rwd_L_2'][i]] = 0.5
                T[self.state['rwd_R_1'][i], self.state['rwd_R_2'][i]] = 0.5
            return T
        
        if self.task_space == 'poz':
            T = super().get_transition_matrix(condition=condition)
            # # OFC lesion - HPC lesion method 2
            p = self.largerP  
            if condition == 'agent':
                T[1, 3] = p      # cue left  - L fluid well
                T[1, 4] = 1 - p  # cue left  - R fluid well
                T[2, 3] = 1 - p  # cue right - L fluid well
                T[2, 4] = p      # cue right - R fluid well
            return T

    def get_observation_matrix(self, condition='agent', baseline_cue=1):
        O = super().get_observation_matrix(condition=condition, baseline_cue=baseline_cue)
        return O

    def get_dwell_times(self, condition='agent'):
        if self.task_space == 'poz':
            D = self.get_dwell_times_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            D = self.get_dwell_times_PozRwd(condition=condition)
            # OFC lesion, equal transition to left and right well
            # well_L, well_R = self.state['well_L'], self.state['well_R']
            # for wl, wr in zip(well_L, well_R):
            #     tmp = (D[wl] + D[wr])/2
            #     D[wl] = tmp
            #     D[wr] = tmp
            # r1_L, r1_R = self.state['rwd_L_1'], self.state['rwd_R_1']
            # for wl, wr in zip(r1_L, r1_R):
            #     tmp = (D[wl] + D[wr])/2
            #     D[wl] = tmp
            #     D[wr] = tmp
            # r2_L, r2_R = self.state['rwd_L_2'], self.state['rwd_R_2']
            # for wl, wr in zip(r2_L, r2_R):
            #     tmp = (D[wl] + D[wr])/2
            #     D[wl] = tmp
            #     D[wr] = tmp
        return D

def sample(n_trials, conditions, timing, task_space='poz'):
    # generate the observation in each block
    task = odorTask(timing, task_space=task_space)
    
    # condition: 1/2/3/4: long_delay/short delay/big/small
    if task_space == 'poz_rwd':
        if conditions == [2, 1]:
            block = 0
        elif conditions == [1, 2]:
            block = 1
        elif conditions == [3, 4]:
            block = 2
        elif conditions == [4, 3]:
            block = 3
            
    observations = []
    T_left, O, D_left = task.get_info(conditions[0], left_right=0)
    T_right, O, D_right = task.get_info(conditions[1], left_right=1)
    # left trials, the number of left and right trials are not random
    trial_type = choice(n_trials, int(n_trials / 2), replace=False)
    for n in range(n_trials):
        # left or right trial
        if n in trial_type:
            T, D = T_left, D_left
        else:
            T, D = T_right, D_right
        # the observation in current trial
        trial = [[0]]
        # sample the state and observation, it is kind of necessary for this task, but more flexible
        if task_space == 'poz':
            state, observation = [0], [1]  # trial start and light on
        elif task_space == 'poz_rwd':
            state, observation = [block], [1]  # trial start and light on
        while True:
            # get the next state
            prob_s = T[state[-1], :].reshape(-1, )
            state.append(choice(range(T.shape[0]), 1, p=prob_s)[0])
            # get the observation in the next state
            prob_o = O[state[-2], state[-1], :].reshape(-1, )
            observation.append(choice(range(O.shape[2]), 1, p=prob_o)[0])
            # if it the last state, break
            if state[-1] in task.state['end']:
                break
        # get the duration
        for s, o in zip(state, observation):
            duration = choice(range(D[s].shape[0]), 1, p=D[s])[0]
            trial.append([o] + [0] * (duration - 1))
        trial = list(chain(*trial))
        observations.append(trial)
    return observations


class Agent(object):

    def __init__(self, parameters):
        # the task object
        if not parameters.lesion:
            task = odorTask(parameters.timing, params=parameters)
        else:
            task = odorTaskLesion(parameters.timing, params=parameters)
        self.task = task
        self.lesion = parameters.lesion
        # init
        self.init_values = []
        self.parameters = parameters
        self.offline = {'beta': [], 'alpha': [], 'belief': []}
        self.baseline = parameters.baseline
        self.init()

        self.n_states = task.n_states
        self.RPE = np.zeros(task.n_states, )  # reward prediction error
        self.state_value = self.init_values.V  # the expected value for each state

        self.observations = []  # record all observations
        self.o_since_latest_O = []  # record observations from latest non-empty observation
        self.observations_prob_chain = []  # record p(o_t+1 | o_1, ... ,o_t)

        self.e_trace = np.zeros(task.n_states, )  # eligibility trace
        self.mean_dwell = []  # the expected duration staying each state
        self.belief = [self.init_values.belief]  # p(S_t=s | O_1,...,O_t)
        # å: the probability of transition from state s at time t given all observations o_1, ..., O_t
        self.alpha = [self.init_values.alpha]
        # ß: the probability of transition from state s at time t given all observations o_1, ..., O_t+1
        self.beta = [self.init_values.belief]

        # get prior information from Task
        self.T_matrix = task.get_transition_matrix(condition='agent', baseline_big_T=self.baseline.T_big)
        self.D_matrix = task.get_dwell_times(condition='agent')
        self.O_matrix = task.get_observation_matrix(condition='agent', baseline_cue=self.baseline.O_cue)

        self.state_value_list = []
        self.discount = np.zeros(task.n_states, )

        self.replay_flag = 0
        self.nth_trial = 0
        # for two layer task state space; 20230126
        self.prior = np.ones(len(self.task.state['start']), )/len(self.task.state['start'])
        assert self.parameters.dt == dt
        assert self.parameters.t_whole == t_range

    def init(self):
        n_states = self.task.n_states
        # initial value
        alpha, belief = 0.01 + np.zeros(n_states, ), np.zeros(n_states, )
        vinit = np.zeros(n_states, )
        var = 1e-5
        v = vinit # + var * np.random.randn(n_states, )
        self.init_values = Dict2Class({'alpha': alpha, 'belief': belief, 'likelihood': [0.02, 0.02],
                                       'var': var, 'Vinit': vinit, 'V': v})
        self.state_value_list = []
        self.discount = []
        self.nth_trial = 0

    def reset(self, first=False):
        """ clear the variables at the beginning of each trial """
        # =====================================================================
        # before 02/09/2023
        if first:
            self.RPE[:] = 0
            self.discount[:] = 0
            self.mean_dwell = []  # the expected duration staying each state'
            self.observations = []  # record all observations
            self.o_since_latest_O = []  # record observations from latest non-empty observation
            self.observations_prob_chain = []  # record p(o_t+1 | o_1, ... ,o_t)

        else:
        # =====================================================================
            ind_lastO = np.where(np.array(self.observations)!=0)[0][-1]
            self.observations = self.observations[ind_lastO:]  # record all observations
            self.observations_prob_chain = self.observations_prob_chain[ind_lastO:]  # record p(o_t+1 | o_1, ... ,o_t)
    
        self.belief = [self.belief[-1]]  # p(S_t=s | O_1,...,O_t)
        self.alpha = [self.alpha[-1]]
        self.beta = [self.beta[-1]]
        
        # self.beta = [self.init_values.belief]
        # self.alpha = [self.init_values.alpha]]
        # self.belief = [self.init_values.belief]
        
        self.state_value_list = []
        self.offline = {'beta': [], 'alpha': [], 'belief': []}
        self.nth_trial += 1
                
    def clean(self):
        """ clear the variables at the beginning of each trial """

        self.observations = []  # record all observations

        n = len(self.o_since_latest_O)
        self.observations_prob_chain = self.observations_prob_chain[-n - 2:]

        self.beta = self.beta[-n - 2:]
        self.alpha = self.alpha[-n - 2:]
        self.belief = self.belief[-n - 2:]

    def T_reset(self):
        self.T_matrix = self.task.get_transition_matrix(condition='agent')
                    
    def prior_reset(self):
        self.prior[:] = 1/len(self.task.state['start'])
        
    def T_baseline_reset(self, p):
        self.baseline.T = p

    def update_observation(self, o):
        self.observations.append(o)
        if o == 0:  # null observation
            self.o_since_latest_O.append(o)
        else:
            self.o_since_latest_O = [o]

    def state_and_transition_estimation(self, o_tp1):
        """
        return beta, indicates the probability of transition from state s at time t give all observations o_1,...,O_t+1
            beta = P(S_t = s, Phi_t = 1 | O_1, ..., O_t+1)
                 = P(O_t+1 | S_t = s, Phi_t = 1) * P(S_t = s, Phi_t = 1 |O_1, ..., O_t) / P(O_t+1 | O_1, ..., O_t)
            P(O_t+1 | S_t = s, Phi_t = 1), i.e. likelihood = sum over s' (T_s_s', O_s'_t+1)
            P(S_t = s, Phi_t = 1 |O_1, ..., O_t), i.e.  piror = sum over d [(O_s_o__t-d-1)*D_s_d*...
                        P(S_t-d+1=s, Phi_t-d=1|O_1, ..., O_t) / P(O_t-d+1=s, ..., O_t|O_1, ..., O_t-d)


            ƒ(d, t) = P(O_t−d+1,..., O_t | O_1...O_t−d)
            f(1, t+1) = P(O_t+1 | O_1, ..., O_t)
            f(d, t) = f(1, t)*f(1, t-1)*...*f(1, t-d+1)
        """
        D = copy.deepcopy(self.D_matrix)
        T = copy.deepcopy(self.T_matrix)
        n_states = self.n_states
        dwell_time_max = len(self.o_since_latest_O)
        # ß: the probability of transition from state s at time t given all observations o_1, ..., O_t + 1
        # alpha: is a prior, the probability of transition from state s at time t\
        #   given all observations o_1,...,O_t
        alpha = np.zeros((n_states, dwell_time_max))
        belief = np.zeros(n_states, )
        # d should be equal to the number of observation s since latest observation
        for d in range(1, dwell_time_max + 1):
            o = self.o_since_latest_O[-d]  # observation at time step t-d+1
            c_temp = np.matmul(T.T, self.alpha[-d])  # alpha at time step t-d
            # TODO: check here, it is D_matrix[s][d-1:] or D_matrix[s][d:]
            future_dwell = [D[s][d - 1:].sum() for s in range(n_states)]
            # future_dwell = [D[s][d:].sum() for s in range(n_states)]
            if dwell_time_max == 1:
                denominater = (self.O_matrix[:, o] * future_dwell * c_temp).sum()
                if len(self.observations_prob_chain) == 0:
                    self.observations_prob_chain.append(denominater)
                else:
                    self.observations_prob_chain[-1] = denominater
            else:
                denominater = np.prod(self.observations_prob_chain[-d:])
            # P(S_t = s | O_1, ..., O_t)
            belief += self.O_matrix[:, o] * future_dwell * c_temp / denominater
            # TODO: check here, it is D_matrix[s][d-1:] or D_matrix[s][d:]; The former is correct
            dwell = [D[s][d - 1] for s in range(n_states)]
            # dwell = [D[s][d] for s in range(n_states)]
            # P(S_t = s, Phi_t = 1 | O_1, ..., O_t)
            alpha[:, d - 1] = self.O_matrix[:, o] * dwell * c_temp / denominater

        # P(O_t+1 |S_t = s, Phi_t = 1)
        likelihood = np.matmul(T, self.O_matrix[:, o_tp1])
        # P(O_t+1 | O_1, ..., O_t)
        if o_tp1 != 0:
            observations_prob = (likelihood * alpha.sum(axis=1)).sum()
        else:
            observations_prob = (likelihood * alpha.sum(axis=1)).sum() + (belief - alpha.sum(axis=1)).sum()

        # P(S_t = s, Phi_t = 1 | O_1, ..., O_t+1)
        beta = likelihood * alpha.sum(axis=1) / observations_prob

        # the expected duration stay in state s
        mean_dwell = np.zeros(n_states, )
        for s in range(n_states):
            mean_dwell[s] = np.matmul(alpha[s, :], np.arange(1, dwell_time_max + 1)) / np.sum(alpha[s, :])
        # print(belief.sum())
        # save the observation probability
        self.observations_prob_chain.append(copy.deepcopy(observations_prob))
        self.beta.append(copy.deepcopy(beta))
        self.alpha.append(copy.deepcopy(alpha.sum(axis=1)))
        self.belief.append(copy.deepcopy(belief))
        self.mean_dwell = mean_dwell
        # used for retrospective evaluation
        self.offline['beta'].append(copy.deepcopy(beta))
        self.offline['alpha'].append(copy.deepcopy(alpha))
        self.offline['belief'].append(copy.deepcopy(belief))

    def e_trace_update(self):
        o_tp1 = self.o_since_latest_O[-1]
        # reset eligibility if trial start
        if o_tp1 == 1:  # null state
            self.e_trace[:] = 0
        # update eligibility based on self.beta
        if o_tp1 == 0:  # null state
            self.e_trace = np.maximum(self.parameters.e_decay * self.e_trace, self.beta[-1])
        # else:
        # # this is correct, but might not be necessary
        #     self.e_trace[:] = 0
        #     self.e_trace[o_tp1] = 1
        
    def dwell_update(self, o_tp1):
        """
        dwell-time distribution in each state is updated at the time of each non-empty observation
        :param o_tp1:
        :return:
        """
        if o_tp1 == 0:
            return
        lr, dwell_std = self.parameters.dwell_lr, self.parameters.dwell_std  # 0.1, 0.1
        dt, t_whole = self.parameters.dt, self.parameters.t_whole  # 0.1, 10
        dwell = len(self.o_since_latest_O)  # + 1
        for s in range(self.n_states):
            k_distribution = norm.pdf(np.arange(dt, t_whole + dt, dt), dt * dwell, dt * dwell * dwell_std)
            k_distribution = normalize(k_distribution, self.baseline.D)

            # Angela gated the update with self.beta in the code, not in the paper;
            #   The former is more reasonable, since the dwell distribution
            #   of unlikey states should not be updated
            # self.D_matrix[s] += lr * (k_distribution - self.D_matrix[s])
            self.D_matrix[s] += lr * (k_distribution - self.D_matrix[s]) * self.beta[-1][s]
            self.D_matrix[s] = copy.deepcopy(normalize(self.D_matrix[s], self.baseline.D))

    def value_update(self, o_tp1, r_tp1):
        """
        update the state value based on the eligibility trace (t+1) and RPE (t+1)

        input: o_tp1, observation at time t+1
               r_tp1, reward at time t+1
        get the expected value at time t+1, value_tp1
                reward prediction error at time t+1. RPE
        :return
        """

        if r_tp1 == 1:
            a = 1
        params = self.parameters
        T, O = self.T_matrix, self.O_matrix
        n_states, value = self.n_states, self.state_value

        expected_value_tp1 = np.matmul(T, (value * O[:, o_tp1])) / np.matmul(T, O[:, o_tp1])

        # expected_value_tp1_2 = np.zeros(expected_value_tp1.shape)
        # for i in range(T.shape[0]):
        #     for ii in range(T.shape[0]):
        #         d = sum(T[i, :]*O[:, o_tp1])
        #         expected_value_tp1_2[i] += value[ii]*(T[i,ii]*O[ii,o_tp1]/d)

        temporal_discount = np.exp(-params.tau * self.mean_dwell)
        RPE = self.beta[-1] * (temporal_discount * (r_tp1 + expected_value_tp1) - value)
        # dopamine neurons might code a vector error signal, but absent data from experiments, we illustrate the
        # dopamine response as a scalar, cumulative error over all the states
        # self.state_value += params.value_lr * RPE
        self.state_value += params.value_lr * self.e_trace * RPE.sum()
        # self.state_value -= self.state_value * (1-self.e_trace) * 1e-5
        # self.state_value += params.value_lr * self.e_trace * RPE
        self.RPE = copy.deepcopy(RPE)

        # TODO: lock the state of trial state to zero, Angela did this in her paper
        # self.state_value[0] = 0  # self.state_value[1:3].mean()
        self.state_value[self.task.state['end']] = 0  # self.state_value[1:3].mean()
        self.discount = copy.deepcopy(temporal_discount)
        self.state_value_list.append(copy.deepcopy(self.state_value))

        return expected_value_tp1

    def transition_update(self):
        """
        update the transition matrix, when T converges, the following equation should be true
            P(S_t, Phi_t = 0|o_1,...,o_t) + P(S_t, Phi_t = 1|o_1,...,o_t) * T =  P(S_t+1|o_1,...,o_t+1)
        """
        """
        temporal difference learning rule, not work well, transition matrix changed dramticly,
        so the prediction error disappear in the second trial after block switch

        self.state = {'start': 0, 'cue_L': 1, 'cue_R': 2, 'well_L': [3, 4, 5], 'well_R': [6, 7, 8],
                      'rwd_L_1': [9, 10, 11], 'rwd_R_1': [12, 13, 14],
                      'rwd_L_2': 15, 'rwd_R_2': 16, 'end': 17}
        # observations: 0: a null observation; 1:light on (trial start); 2/3 two odor cues;
        #               4: for both fluid well; 5/6:first/second rewards; 7:light off (trial end);
        """

        state_space = self.task.state
        observations = np.array(self.observations)
        # common/big/long
        lr = self.parameters.trans_lr
        
        cue = 1 if 2 in observations else 2
        prior = self.prior
        start_state, end_state = state_space['start'] , state_space['end'] 
        if cue == 1:
            cue_state = state_space['cue_L']
        else:
            cue_state = state_space['cue_R']
        waitR_state = state_space['well_L'] +state_space['well_R'] 
        R1_state = state_space['rwd_L_1'] + state_space['rwd_R_1'] 
        R2_state = state_space['rwd_L_2'] + state_space['rwd_R_2']
            
        # time step
        t_well = np.where(observations == 4)[0][0]
        t_end = np.where(observations == 7)[0][0]
        t_r1 = np.where(observations == self.task.r_obs[0])[0][0]
        if np.where(observations == self.task.r_obs[1])[0]:
            t_r2 = np.where(observations == self.task.r_obs[1])[0][0]
        else:
            t_r2 = np.where(observations == self.task.r_obs[0])[0][-1]

        # dwell duration
        dwell_wait = t_r1 - t_well
        if t_r2 > t_r1:
            dwell_r1, dwell_r2 = t_r2 - t_r1, t_end - t_r2
        else:
            dwell_r1, dwell_r2 = t_end - t_r1, np.nan
            
        # likelihood*prior
        T, D = self.T_matrix, self.D_matrix
        # prior
        likelihood = np.zeros(prior.shape)
        for i in range(len(start_state)):
            for j in [0, 4]:
                wstate, r1, r2 = waitR_state[i+j], R1_state[i+j], R2_state[i+j]
                tmp = D[wstate][dwell_wait - 1]
                if t_r2 > t_r1:
                    tmp *= T[r1, r2] * D[r2][dwell_r1 - 1] * T[r2, end_state[i]] * \
                                 D[r2][dwell_r2 - 1]
                else:
                    tmp *= T[r1, end_state[i]] * D[r1][dwell_r1 - 1]
                likelihood[i] += tmp*T[cue_state[i], wstate]

        post = likelihood * prior
        post = post/post.sum()
        
        # learning rate
        # lr = lr / sum(abs(pe)) if sum(abs(pe)) > 0 else 1
        # lr = lr if lr < 1 else 1
        # print(lr)
        # print(self.T_matrix[0, 1:9])
        # learning
        self.prior = normalize((1-lr)*prior+ lr*post, self.baseline.T)
        self.T_matrix[end_state[0], start_state] = copy.deepcopy(self.prior)

        # normalization
        # self.T_matrix = copy.deepcopy(normalize(self.T_matrix, self.baseline.T))
        """
        if len(self.belief) < 2:
        # belief states at least two time point is required
        return

        b_t, b_tp1, alpha = self.belief[-2], self.belief[-1], self.alpha[-1]
        # b_t - alpha + np.matmul(alpha, T_target) = b_tp1
        T_target = np.matmul(b_tp1 - b_t + alpha, np.linalg.inv(alpha.reshape(-1, 1)))
        params = self.parameters
        self.T_matrix += params.lr_T*(T_target-self.T_matrix)
        """

    def state_prior(self):
        prob = []
        for key in self.D_matrix.keys():
            prob.append(np.matmul(np.arange(dt, t_range + dt, dt), self.D_matrix[key]))
        prob = np.array(prob)
        return prob / prob.sum()

    def observation_prior(self):
        s_prior = self.state_prior()
        o_prior = np.matmul(s_prior, self.O_matrix)
        return o_prior


class Recorder:
    def __init__(self, path='D:/Zhewei/coorperate_with_yuji/figs'):
        self.path = path
        # self.df = pd.DataFrame(columns=['observation', 'belief', 'RPE'])
        self.data = Dict2Class({'o': [], 'r': [], 'v': [], 'RPE': [], 'dwell': [],
                                'trans': [], 'belief': [], 'alpha': [], 'beta': [],
                                'discount': [], 'replay_flag': []})

    def create(self):
        """
        create and open a .csv file
        """
        pass

    def write(self):
        """
        write all data in a .csv file
        """
        pass

    def reset(self):
        """"""
        self.data.o.append([])
        # self.data.r.append([])
        self.data.v.append([])
        self.data.RPE.append([])
        self.data.dwell.append([])
        self.data.trans.append([])
        self.data.belief.append([])
        # self.data.alpha.append([])
        self.data.beta.append([])
        # self.data.discount.append([])

    def record(self, o, r, v, RPE, belief, alpha, beta, discount):
        self.data.o[-1].append(o)
        # self.data.r[-1].append(r)
        self.data.v[-1].append(copy.deepcopy(v.tolist()))
        self.data.RPE[-1].append(copy.deepcopy(RPE.tolist()))
        self.data.belief[-1].append(copy.deepcopy(belief))
        # self.data.alpha[-1].append(copy.deepcopy(alpha))
        self.data.beta[-1].append(copy.deepcopy(beta))
        # self.data.discount[-1].append(copy.deepcopy(discount))

    def record_dwell(self, dwell):
        self.data.dwell[-1].append(copy.deepcopy(np.array([dwell[key] for key in np.sort(list(dwell.keys()))])))

    def record_trans(self, trans):
        self.data.trans[-1].append(copy.deepcopy(trans))

