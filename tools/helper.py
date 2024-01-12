# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 2021

@author: Zhewei Zhang

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
            self.state = {'start': 0, 'cue_L': 1, 'cue_R': 2, 'well_L': 3, 'well_R': 4,
                          'rwd_L_1': 5, 'rwd_R_1': 6, 'rwd_L_2': 7, 'rwd_R_2': 8, 'end': 9}
        elif self.task_space == 'poz_rwd':
            '''assume the dwell distribution is known, and agent leans the transition among states'''
            self.n_states = 18
            self.r_obs = [5, 6]
            self.n_observations = 8
            self.state = {'start': 0, 'cue_L': 1, 'cue_R': 2, 'well_L': [3, 4, 5], 'well_R': [6, 7, 8],
                          'rwd_L_1': [9, 10, 11], 'rwd_R_1': [12, 13, 14],
                          'rwd_L_2': 15, 'rwd_R_2': 16, 'end': 17}

    def get_dwell_times(self, condition='agent'):
        if self.task_space == 'poz':
            D = self.get_dwell_times_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            D = self.get_dwell_times_PozRwd(condition=condition)
        return D

    def get_transition_matrix(self, condition='agent'):
        if self.task_space == 'poz':
            T = self.get_transition_matrix_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            T = self.get_transition_matrix_PozRwd(condition=condition)
        return T

    def get_observation_matrix(self, condition='agent'):
        if self.task_space == 'poz':
            O = self.get_observation_matrix_Poz(condition=condition)
        elif self.task_space == 'poz_rwd':
            O = self.get_observation_matrix_PozRwd(condition=condition)
        return O

    def get_info(self, block, left_right=None):
        """
        :param left_right: 0 for left trials, 1 for right trials
        :param block: 1/2/3/4: long/short/big/small
        :return:
        """
        T = self.get_transition_matrix(condition=block)
        if left_right == 0:
            T[0, 1] = 1  # trial start - cue left
            T[0, 2] = 0  # trial start - cue right
        elif left_right == 1:
            T[0, 1] = 0  # trial start - cue left
            T[0, 2] = 1  # trial start - cue right
        return T, self.get_observation_matrix(condition=block), self.get_dwell_times(condition=block)

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

    def get_transition_matrix_PozRwd(self, condition='agent'):
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
        # trial start - cue left/right
        for s in [self.state['cue_L'], self.state['cue_R']]:
            T[self.state['start'], s] = .5
        # well - reward 1
        r1 = self.state['rwd_L_1'] + self.state['rwd_R_1']
        well = self.state['well_L'] + self.state['well_R']
        for i, j in zip(well, r1):
            T[i, j] = 1
        # reward 1-short/long/small - trial end
        for s in self.state['rwd_L_1'][::2] + self.state['rwd_R_1'][::2]:
            T[s, self.state['end']] = 1
        # reward 1-big - reward 2
        T[self.state['rwd_L_1'][1], self.state['rwd_L_2']] = 1
        T[self.state['rwd_R_1'][1], self.state['rwd_R_2']] = 1
        # reward 2-big - trial end
        T[self.state['rwd_L_2'], self.state['end']] = 1
        T[self.state['rwd_R_2'], self.state['end']] = 1
        # trial end - trial start
        T[self.state['end'], self.state['start']] = 1

        # cue - well
        if condition == 1:  # long
            T[1, self.state['well_L'][2]] = 1  # cue L - L-well-long
            T[2, self.state['well_R'][2]] = 1  # cue R - R-well-long
        elif condition in [2, 4]:  # common
            T[1, self.state['well_L'][0]] = 1  # cue L - L-well-common
            T[2, self.state['well_R'][0]] = 1  # cue R - R-well-common
        elif condition == 3:  # big
            T[1, self.state['well_L'][1]] = 1  # cue L - L-well-big
            T[2, self.state['well_R'][1]] = 1  # cue R - R-well-big
        elif condition == 'agent':
            T[1, 3] = 1 / 2  # cue left  - L fluid well common
            T[2, 6] = 1 / 2  # cue right - R fluid well common
            T[1, 4:6] = 1 / 4  # cue left  - L fluid well big/small
            T[2, 7:9] = 1 / 4  # cue right - R fluid well big/small
            # T = normalize(T, self.params.baseline.T)
        else:
            raise

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
            baseline_O = self.params.baseline.O_null
            O = np.zeros((self.n_states, self.n_observations))

            O[:, 0] = baseline_O  # null observation for all states
            O[0, 1] = high_p  # trial start - light on
            O[1, 2] = high_p  # cue left    - cue left
            O[2, 3] = high_p  # cue right   - cue right
            O[3:5, 4] = high_p  # L/R fluid wl  - well
            O[5:7, 5] = high_p  # L/R-reward 1  - reward
            O[7:9, 5] = high_p  # L/R-reward 2  - reward
            O[9, 7] = high_p  # trial end   - light off
            O = normalize(O, self.params.baseline.O)

        return O

    def get_observation_matrix_PozRwd(self, condition='agent'):
        # observation_matrix (the probability of observing dim3 when transiting from dim 1 to dim 2)
        # observations: 0: a null observation; 1:light on (trial start); 2/3 two odor cues;
        #               4: for both fluid well; 5/6:first/second rewards; 7:light off (trial end);
        if condition == 'task' or condition in [1, 2, 3, 4]:
            O = np.zeros((self.n_states, self.n_states, self.n_observations))
            # trial end - trial start - light on
            O[self.state['end'], self.state['start'], 1] = 1
            # trial start - cue left  - cue left
            O[self.state['start'], self.state['cue_L'], 2] = 1
            # trial start - cue right - cue right
            O[self.state['start'], self.state['cue_R'], 3] = 1
            # cue left  - L fluid wl  - well
            for s in self.state['well_L']:
                O[self.state['cue_L'], s, 4] = 1
            # cue right - R fluid wl  - well
            for s in self.state['well_R']:
                O[self.state['cue_R'], s, 4] = 1
            # L fluid wl - L-reward 1 - reward
            for i, j in zip(self.state['well_L'], self.state['rwd_L_1']):
                O[i, j, 5] = 1
            # R fluid wl - R-reward 1 - reward
            for i, j in zip(self.state['well_R'], self.state['rwd_R_1']):
                O[i, j, 5] = 1

            # O[self.state['rwd_L_1'][1], self.state['rwd_L_2'],  6] = 1  # L-reward - L-reward 2 - reward
            # O[self.state['rwd_R_1'][1], self.state['rwd_R_2'],  6] = 1  # R-reward - R-reward 2 - reward
            # reward 1 big - L-reward 2 - reward
            O[self.state['rwd_L_1'][1], self.state['rwd_L_2'], 5] = 1  # L-reward - L-reward 2 - reward
            O[self.state['rwd_R_1'][1], self.state['rwd_R_2'], 5] = 1  # R-reward - R-reward 2 - reward
            # L/R-reward 1/2 - trial end  - light off
            for s in self.state['rwd_L_1'][::2] + self.state['rwd_R_1'][::2]:
                O[s, self.state['end'], 7] = 1
            for s in [self.state['rwd_L_2']] + [self.state['rwd_R_2']]:
                O[s, self.state['end'], 7] = 1

            O[np.sum(O[:, :, 1:], axis=2) == 0, 0] = 1  # null
        elif condition == 'agent':
            high_p = 1 - self.params.baseline.O_null
            baseline_O = self.params.baseline.O_null
            O = np.zeros((self.n_states, self.n_observations))
            # null observation for all states
            O[:, 0] = self.params.baseline.O_null
            # trial start - light on
            O[self.state['start'], 1] = high_p
            # cue left    - cue left
            O[self.state['cue_L'], 2] = high_p
            # cue right   - cue right
            O[self.state['cue_R'], 3] = high_p
            # L/R fluid wl  - well
            for s in self.state['well_L'] + self.state['well_R']:
                O[s, 4] = high_p
            # L/R-reward 1  - reward
            for s in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                O[s, 5] = high_p
            # L/R-reward 2  - reward
            for s in [self.state['rwd_L_2'], self.state['rwd_R_2']]:
                # O[s,   6] = high_p
                O[s, 5] = high_p
            # trial end   - light off
            O[self.state['end'], 7] = high_p
            O = normalize(O, self.params.baseline.O)
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

        # for agents
        if condition == 'agent':
            dwell_std = self.params.dwell_std
            for i in [self.state['start'], self.state['end'],
                      self.state['cue_L'], self.state['cue_R']]:
                D[i] = dwell_func(0.5)
            # wells
            for i in self.state['well_L'] + self.state['well_R']:
                D[i] = dwell_func(short)
            # well indicating long delay
            for i in [self.state['well_L'][-1], self.state['well_R'][-1]]:
                D[i] = dwell_func(long)
            # reward 1
            for i in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                D[i] = dwell_func(short)
            # reward 2
            for i in [self.state['rwd_L_2'], self.state['rwd_R_2']]:
                D[i] = dwell_func(short)
            # trial end
            D[self.state['end']] = dwell_func(short)
            for key in D.keys():
                D[key] = normalize(D[key], self.params.baseline.D)
        else:
            # for task, to generate the task event sequence
            for i in [self.state['start'], self.state['end'],
                      self.state['cue_L'], self.state['cue_R']]:
                D[i] = assign(0.5)
            # wells
            for i in self.state['well_L'] + self.state['well_R']:
                D[i] = assign(short)
            # well indicating long delay
            for i in [self.state['well_L'][-1], self.state['well_R'][-1]]:
                D[i] = assign(long)
            # reward 1
            for i in self.state['rwd_L_1'] + self.state['rwd_R_1']:
                D[i] = assign(short)
            # reward 2
            for i in [self.state['rwd_L_2'], self.state['rwd_R_2']]:
                D[i] = assign(short)
            # trial end
            D[self.state['end']] = assign(short)
        return D


class odorTaskLesion(odorTask):
    def __init__(self, timing, params=[]):
        super(odorTaskLesion, self).__init__(timing, params=params)
        self.largerP = params.largerP
        # overwrite the function definition in the parent class
        self.get_dwell_times = self.get_dwell_times
        self.get_transition_matrix = self.get_transition_matrix
        self.get_observation_matrix = self.get_observation_matrix

    def get_info(self, block, left_right=None):
        """
        :param left_right: 0 for left trials, 1 for right trials
        :param block: 1/2/3/4: long/short/big/small
        :return:
        """
        T = self.get_transition_matrix(condition=block)
        if left_right == 0:
            T[0, 1] = 1  # trial start - cue left
            T[0, 2] = 0  # trial start - cue right
        elif left_right == 1:
            T[0, 1] = 0  # trial start - cue left
            T[0, 2] = 1  # trial start - cue right
        return T, self.get_observation_matrix(condition=block), self.get_dwell_times(condition=block)

    def get_transition_matrix(self, condition='agent'):
        # print('return lesion matrix')
        T = super().get_transition_matrix(condition=condition)
        if self.task_space == 'poz_rwd':
            # no changes in T and O
            return T
        # # OFC lesion
        # p = 0.5
        # HPC lesion method 2
        if self.task_space == 'poz':
            p = self.largerP  # 0.65
            if condition == 'agent':
                T[1, 3] = p  # cue left  - L fluid well
                T[1, 4] = 1 - p  # cue left  - R fluid well
                T[2, 3] = 1 - p  # cue right - L fluid well
                T[2, 4] = p  # cue right - R fluid well
                # T[3, 5] = p    # L fluid well - L-reward 1
                # T[3, 6] = 1-p  # L fluid well - R-reward 1
                # T[4, 5] = 1-p  # R fluid well - L-reward 1
                # T[4, 6] = p    # R fluid well - R-reward 1
            return T

    def get_observation_matrix(self, condition='agent'):
        O = super().get_observation_matrix(condition=condition)
        if self.task_space == 'poz_rwd':
            # no changes in T and O
            return O
        # if condition == 'agent':
        #     p = self.largerP
        #     # additional HPC lesion model
        #     O[1, 3] = O[1, 2] + base_p - p  # cue left  - cue right
        #     O[2, 2] = O[2, 3] + base_p - p  # cue right - cue left
        #     O[1, 2] = p  # cue left  - cue left
        #     O[2, 3] = p  # cue right - cue right
        return O


def sample(n_trials, blocks, timing, task_space='poz'):
    # generate the observation in each block
    task = odorTask(timing, task_space=task_space)

    observations = []
    T_left, O, D_left = task.get_info(blocks[0], left_right=0)
    T_right, O, D_right = task.get_info(blocks[1], left_right=1)
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
        state, observation = [0], [1]  # trial start and light on
        while True:
            # get the next state
            prob_s = T[state[-1], :].reshape(-1, )
            state.append(choice(range(T.shape[0]), 1, p=prob_s)[0])
            # get the observation in the next state
            prob_o = O[state[-2], state[-1], :].reshape(-1, )
            observation.append(choice(range(O.shape[2]), 1, p=prob_o)[0])
            # if it the last state, break
            if state[-1] == task.state['end']:
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
        self.T_matrix = task.get_transition_matrix(condition='agent')
        self.D_matrix = task.get_dwell_times(condition='agent')
        self.O_matrix = task.get_observation_matrix(condition='agent')

        self.state_value_list = []
        self.discount = np.zeros(task.n_states, )

        self.replay_flag = 0

        assert self.parameters.dt == dt
        assert self.parameters.t_whole == t_range

    def init(self):
        n_states = self.task.n_states
        # initial value
        alpha, belief = 0.01 + np.zeros(n_states, ), 0.01 + np.zeros(n_states, )
        belief[0] = 1 - belief[1:].sum()
        vinit = np.zeros(n_states, )
        vinit[:3, ] = .7
        var = 1e-5
        v = vinit + var * np.random.randn(n_states, )
        self.init_values = Dict2Class({'alpha': alpha, 'belief': belief, 'likelihood': [0.02, 0.02],
                                       'var': var, 'Vinit': vinit, 'V': v})
        self.state_value_list = []
        self.discount = []

    def reset(self):
        """ clear the variables at the beginning of each trial """

        self.observations = []  # record all observations
        self.o_since_latest_O = []  # record observations from latest non-empty observation
        self.observations_prob_chain = []  # record p(o_t+1 | o_1, ... ,o_t)

        self.mean_dwell = []  # the expected duration staying each state
        self.belief = [self.init_values.belief]  # p(S_t=s | O_1,...,O_t)
        self.alpha = [self.init_values.alpha]
        self.beta = [self.init_values.belief]
        self.RPE[:] = 0

        self.discount[:] = 0
        self.state_value_list = []
        self.offline = {'beta': [], 'alpha': [], 'belief': []}

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

    def lr_reset(self):
        self.parameters.value_lr = 0

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
            c_temp = np.matmul(self.T_matrix.T, self.alpha[-d])  # alpha at time step t-d
            # TODO: check here, it is D_matrix[s][d-1:] or D_matrix[s][d:]
            future_dwell = [self.D_matrix[s][d - 1:].sum() for s in range(n_states)]
            # future_dwell = [self.D_matrix[s][d:].sum() for s in range(n_states)]
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
            dwell = [self.D_matrix[s][d - 1] for s in range(n_states)]
            # dwell = [self.D_matrix[s][d] for s in range(n_states)]
            # P(S_t = s, Phi_t = 1 | O_1, ..., O_t)
            alpha[:, d - 1] = self.O_matrix[:, o] * dwell * c_temp / denominater

        # P(O_t+1 |S_t = s, Phi_t = 1)
        likelihood = np.matmul(self.T_matrix, self.O_matrix[:, o_tp1])
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
        #         self.state_value += params.value_lr * self.e_trace * RPE
        self.RPE = copy.deepcopy(RPE)

        # TODO: lock the state of trial state to zero, Angela did this in her paper
        # self.state_value[0] = 0  # self.state_value[1:3].mean()
        self.state_value[-1] = 0  # self.state_value[1:3].mean()
        self.discount = copy.deepcopy(temporal_discount)
        self.state_value_list.append(copy.deepcopy(self.state_value))

        return expected_value_tp1

    def replay_state(self):
        # get the sequence of states in last trial for learning during ITI
        bs_winner = [0]
        # TODO: current version is hard-coded for poz_rwd task space
        # non_null_obv = np.where(np.array(self.observations)!=0)[0].tolist()
        bs_winner += [0] * 5
        if 2 in self.observations:
            bs_winner += [1] * 5
            left_right = 1
        elif 3 in self.observations:
            bs_winner += [2] * 5
            left_right = 2

        if 6 in self.observations or (np.array(self.observations) == 5).sum() == 2:
            if left_right == 1:
                bs_winner += [4] * 5
                bs_winner += [10] * 5
                bs_winner += [15] * 5
            elif left_right == 2:
                bs_winner += [7] * 5
                bs_winner += [13] * 5
                bs_winner += [16] * 5
        elif np.where(np.array(self.observations) == 5)[0] < 30:
            if left_right == 1:
                bs_winner += [3] * 5
                bs_winner += [9] * 5
            elif left_right == 2:
                bs_winner += [6] * 5
                bs_winner += [12] * 5
        else:
            if left_right == 1:
                bs_winner += [5] * 40
                bs_winner += [11] * 5
            elif left_right == 2:
                bs_winner += [8] * 40
                bs_winner += [14] * 5
        bs_winner += [17] * 4
        """
        if self.task.task_space == 'poz' or self.task.task_space == 'poz_rwd':
            bs_winner_1 = [0]
            non_null_obv = np.where(np.array(self.observations)!=0)[0].tolist()
            for nth, i in enumerate(non_null_obv[1:]):
                state = np.argmax(self.beta[i-1])
                bs_winner_1 += [state] * (non_null_obv[nth+1] - non_null_obv[nth])
            bs_winner_1 += [self.task.state['end']]*(len(self.beta) - len(bs_winner_1))

        if bs_winner != bs_winner_1:
            a=1
            T, D, O = self.T_matrix, self.D_matrix, self.O_matrix
            beta, alpha, belief = self.offline['beta'], self.offline['alpha'], self.offline['belief']

            s_list = [np.argmax(belief[-1])]
            for beta_, alpha_, belief_ in zip(beta[-2::-1], alpha[-2::-1], belief[-2::-1]):
                likelihood = np.zeros(belief_.size, )
                for i in range(belief_.size):
                    likelihood[i] = T[i, s_list[-1]]*alpha_.sum(axis=1)[i]
                    if i == s_list[-1]:
                        likelihood[i] += belief_[i] - alpha_.sum(axis=1)[i]
                s_list.append(np.argmax(likelihood))
            s_list.append([0])
            s_list = s_list[::-1]
        """
        # T, D, O = self.T_matrix, self.D_matrix, self.O_matrix
        # beta, alpha, belief = self.offline['beta'], self.offline['alpha'], self.offline['belief']
        # s_list = np.zeros(len(belief), dtype = int)
        # s_list[-1] = np.argmax(belief[-1])
        # likelihood_all = []
        # for t in range(len(belief)-2, -1, -1):
        #     beta_, alpha_, belief_  = beta[t], alpha[t], belief[t]
        #     likelihood = np.zeros(belief_.size, )
        #     for i in range(belief_.size):
        #         likelihood[i] = T[i, s_list[t+1]]*alpha_.sum(axis=1)[i]
        #         if i == s_list[t+1]:
        #             likelihood[i] += belief_[i] - alpha_.sum(axis=1)[i]
        #     likelihood_all.append(likelihood)
        #     s_list[t] = np.argmax(likelihood)

        T = self.T_matrix
        beta, alpha, belief = self.offline['beta'], self.offline['alpha'], self.offline['belief']
        s_list = [np.argmax(belief[-1])]
        for beta_, alpha_, belief_ in zip(beta[-2::-1], alpha[-2::-1], belief[-2::-1]):
            likelihood = np.zeros(belief_.size, )
            for i in range(belief_.size):
                likelihood[i] = T[i, s_list[-1]] * alpha_.sum(axis=1)[i]
                if i == s_list[-1]:
                    likelihood[i] += belief_[i] - alpha_.sum(axis=1)[i]
            s_list.append(np.argmax(likelihood))
        s_list.append(0)
        s_list = s_list[::-1]
        if bs_winner != s_list:
            return s_list, 0
            # print('diffferent replay states')
            # print('observations:  ', self.observations)
            # print('s_list:        ', s_list)
            # print('bs_winner:     ', bs_winner)
            # print('--------------------------')
        else:
            return s_list, 1

    def replay_value(self, rewards=[5]):
        """
        learning state value during the inter-trial interval; replay
        """
        bs_winner, flag = self.replay_state()
        rs = [1 if o in rewards else 0 for o in self.observations]
        rs, bs_winner = rs[1:], bs_winner[1:]

        # # remove information about duration
        # index_new_state_start = np.hstack((1, np.diff(bs_winner)!=0)) != 0
        # rs_concatenate = np.array(rs)[index_new_state_start]
        # bs_concatenate = np.array(bs_winner)[index_new_state_start]

        # replay
        num = 1
        lr = self.parameters.value_lr
        e_trace = np.zeros(self.state_value.shape)
        for i in range(num):
            d = 0
            for t in range(len(bs_winner) - 1):
                d += 1
                if bs_winner[t] == bs_winner[t + 1]:
                    continue
                # print("d:", d)
                # learning
                temporal_discount = np.exp(-self.parameters.tau * d)
                s_t, s_tp1, r_tp1 = bs_winner[t], bs_winner[t + 1], rs[t + 1]
                v_t, v_tp1 = self.state_value[s_t], self.state_value[s_tp1]
                rpe = temporal_discount * (r_tp1 + v_tp1) - v_t
                e_trace = e_trace * self.parameters.e_decay
                e_trace[s_t] = 1
                # print(rpe)
                # print(e_trace)
                # self.state_value[s_t] += lr * rpe  # * e_trace
                self.state_value += lr * rpe * e_trace
                # self.state_value[0] = 0
                self.state_value[-1] = 0
                d = 0  # reset
        self.replay_flag = flag

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

    def replay_dwell(self):
        """
        learning dwell distribution during the inter-trial interval; replay
        """
        bs_winner, flag = self.replay_state()
        bs_winner = bs_winner[1:]

        num = 1
        lr, dwell_std = self.parameters.dwell_lr / num / 2, self.parameters.dwell_std  # 0.1, 0.1
        dt, t_whole = self.parameters.dt, self.parameters.t_whole  # 0.1, 10

        # replay
        for i in range(num):
            dwell = 0
            for t in range(len(bs_winner) - 1):
                dwell += 1
                if bs_winner[t] == bs_winner[t + 1]:
                    continue
                s_t = bs_winner[t]
                # print("dwell:", dwell)
                # learning
                k_distribution = norm.pdf(np.arange(dt, t_whole + dt, dt), dt * dwell, dt * dwell * dwell_std)
                k_distribution = normalize(k_distribution, self.baseline.D)
                self.D_matrix[s_t] += lr * (k_distribution - self.D_matrix[s_t])
                self.D_matrix[s_t] = normalize(self.D_matrix[s_t], self.baseline.D)
                # self.D_matrix[s_t] = k_distribution

                dwell = 0  # reset
        self.replay_flag = flag

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
        waitR_state = state_space['well_L'] if cue == 1 else state_space['well_R']
        R1_state = state_space['rwd_L_1'] if cue == 1 else state_space['rwd_R_1']
        R2_state = state_space['rwd_L_2'] if cue == 1 else state_space['rwd_R_2']
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
        prob_s = np.array([0., 0., 0.])
        for i, (wstate, r1) in enumerate(zip(waitR_state, R1_state)):
            prob_s[i] = T[wstate, r1] * D[wstate][dwell_wait - 1]
            if t_r2 > t_r1:
                prob_s[i] *= T[r1, R2_state] * D[R2_state][dwell_r1 - 1] * T[R2_state, state_space['end']] * \
                             D[R2_state][dwell_r2 - 1]
            else:
                prob_s[i] = prob_s[i] * T[r1, state_space['end']] * D[r1][dwell_r1 - 1]

        prob_s = prob_s / prob_s.sum()

        if cue == 1:
            self.T_matrix[1, 3:6] = normalize(prob_s, self.baseline.T)
        else:
            self.T_matrix[2, 6:9] = normalize(prob_s, self.baseline.T)
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
        return

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
        self.data.r.append([])
        self.data.v.append([])
        self.data.RPE.append([])
        self.data.dwell.append([])
        self.data.trans.append([])
        self.data.belief.append([])
        self.data.alpha.append([])
        self.data.beta.append([])
        self.data.discount.append([])
        self.data.replay_flag.append([])

    def record(self, o, r, v, RPE, belief, alpha, beta, discount):
        self.data.o[-1].append(o)
        self.data.r[-1].append(r)
        self.data.v[-1].append(copy.deepcopy(v.tolist()))
        self.data.RPE[-1].append(copy.deepcopy(RPE.tolist()))
        self.data.belief[-1].append(copy.deepcopy(belief))
        self.data.alpha[-1].append(copy.deepcopy(alpha))
        self.data.beta[-1].append(copy.deepcopy(beta))
        self.data.discount[-1].append(copy.deepcopy(discount))

    def record_dwell(self, dwell):
        self.data.dwell[-1].append(copy.deepcopy(np.array([dwell[key] for key in dwell.keys()])))

    def record_trans(self, trans):
        self.data.trans[-1].append(copy.deepcopy(trans))

    def record_replayflag(self, replay_flag):
        self.data.replay_flag[-1].append(copy.deepcopy(replay_flag))
