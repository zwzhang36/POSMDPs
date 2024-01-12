# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 2021

@author: Zhewei Zhang

reproduce Takahashi-2011-Expectancy-related changes in firing of dopamine neurons depend on orbitofrontal cortex

parameter:  control:  decay value: 0.99; decay eligibility: 0.9; lr value 0.2; lr preference: 0.1
            lesion:   decay value: 0.99; decay eligibility: 0.9; lr value 0.1; lr preference: 0.2

"""
import copy
import random
import numpy as np
from numpy import random
from helper import Dict2Class
# import pandas as pd
import matplotlib.pyplot as plt


def RPE2firing(RPE):
    firing = np.zeros(RPE.shape)
    firing[RPE >= 0] = 5 + 4*RPE[RPE >= 0]
    firing[RPE < 0] = 5 + 0.8*RPE[RPE < 0]
    return firing


class StateTransitionModel:
    def __init__(self, nstates):
        self.nstates = nstates
        self.tran_mtx = []

    def step(self, state, action):
        """
        :param state:   current state
        :param action:  executed action
        :return: whether the episode is terminal, next state
        """
        tran_mtx = self.tran_mtx[action][state]
        if tran_mtx.sum() == 0:
            return 1, []
        else:
            assert tran_mtx.sum() == 1
            return 0, np.random.choice(self.nstates, p=tran_mtx)


class ShamTransitionModel(StateTransitionModel):
    """
    16 states (1 background (start/terminal state) state)
                      - right                                            | <- |
                              - enter right port - R rwd 1 - R rwd 2 - R wait - R rwd 3 _
    light - odor port - free                                                               end
                              - enter left port  - L rwd 1 - L rwd 2 - L wait - L rwd 3 -
                      - left                                            | <- |
    """

    def __init__(self, nstates=16):
        """
        :param nstates:
        0: choose left, 1: choose right
        """
        super().__init__(nstates)
        tran_mtx = np.zeros((self.nstates, self.nstates))  # from row to column
        tran_mtx[0, 1] = 1
        tran_mtx[1, 2:5] = 1/3
        for i in [5, 6, 7]:
            tran_mtx[i, i+1] = 1
            tran_mtx[i+5, i+5+1] = 1
        # wait right - R rwd 3 & R rwd 3 - end
        tran_mtx[8, 8] = 0.5
        tran_mtx[8, 9] = 0.5
        tran_mtx[9, -1] = 1
        # wait left - L rwd 3 & L rwd 3 - end
        tran_mtx[13, 13] = 0.5
        tran_mtx[13, 14] = 0.5
        tran_mtx[14, -1] = 1
        # choose left/right well
        tran_mtx_left, tran_mtx_right = copy.deepcopy(tran_mtx), copy.deepcopy(tran_mtx)
        tran_mtx_left[2:5, 5] = 1
        tran_mtx_right[2:5, 10] = 1

        self.tran_mtx = [tran_mtx_left, tran_mtx_right]


class LesionTransitionModel(StateTransitionModel):
    """
    11 states
                      - right
                              -                               | <- |
    light - odor port - free    enter port - rwd 1 - rwd 2 - wait - rwd 3  - end
                              -
                      - left
    """
    def __init__(self, nstates=11):
        super().__init__(nstates)
        tran_mtx = np.zeros((self.nstates, self.nstates))  # from row to column
        tran_mtx[1, 2:5] = 1/3
        tran_mtx[2:5, 5] = 1
        for i in [0, 5, 6, 7, 9]:
            tran_mtx[i, i+1] = 1
        tran_mtx[8, 8] = 0.5
        tran_mtx[8, 9] = 0.5
        self.tran_mtx = [tran_mtx, tran_mtx]


class OdorTask:
    """
    not flexible, but enough, we won't use this for other tasks
    """
    def __init__(self, setting):
        self.state = 0    # time step in each trial
        self.ntrial = 0   # nth trials in one session
        self.action = 0   # choice in one trial
        self.correct = 0  # correct or not
        self.setting = setting
        self.transition = LesionTransitionModel() if setting.lesion else ShamTransitionModel()
        self.block = [self.setting.block.left[0], self.setting.block.right[0]]
        self.odor = []
        nblock, odor = len(self.setting.block.left), [0, 2]*70 + [1]*60
        for i in range(nblock):
            random.shuffle(odor)
            self.odor.extend(odor)

    def re_init(self):
        """reset whole task, from the first block"""
        self.state = 0    # time step in each trial
        self.ntrial = 0   # nth trials in one session
        self.action = 0   # choice in one trial
        self.correct = 0  # correct or not
        self.block = [self.setting.block.left[0], self.setting.block.right[0]]
        # select the odor before the task, to equalize the presentation of odors
        self.odor = []
        nblock, odor = len(self.setting.block.left), [0, 2]*70 + [1]*60
        for i in range(nblock):
            random.shuffle(odor)
            self.odor.extend(odor)

    def reset(self):
        """call this function at the beginning of each trial"""
        self.state = 0
        self.ntrial += 1
        self.action = 0
        self.correct = 0     # odor present in each trial
        # check block change
        nblock = int(np.ceil(self.ntrial / self.setting.block.size) - 1)
        self.block = [self.setting.block.left[nblock], self.setting.block.right[nblock]]

    def step(self, action):
        """
        :param action: left/right choice, 0/1
        :return: next state and reward in the next state
        """
        # retrieval or restore the choice
        if self.state in [2, 3, 4]:  # check it
            self.action = action

        # sample the next state
        if self.state == 1:
            # cannot be totally random, have to balance the number of each types
            terminal, next_state = 0, self.odor.pop(0)+2
        else:
            terminal, next_state = self.transition.step(self.state,  self.action)

        # check and restore the performance
        if self.state in [2, 3, 4]:
            odor = self.state - 2  # 0/1/2: left/free/right
            self.correct = (odor == 0 and self.action == 0) or odor == 1 or (odor == 2 and self.action == 1)

        # check reward
        reward = -0  # default
        block = self.block[self.action]  # check current block
        # block: 1/2/3/4: long_delay/short delay/big/small
        if self.correct:
            if self.setting.lesion:
                if block in [2, 3, 4] and next_state in [6]:
                    reward += 1
                if block in [3] and next_state in [7]:
                    reward += 1
                if block in [1] and next_state in [9]:
                    reward += 1
            else:
                if block in [2, 3, 4] and next_state in [6, 11]:
                    reward += 1
                if block in [3] and next_state in [7, 12]:
                    reward += 1
                if block in [1] and next_state in [9, 14]:
                    reward += 1

        # restore the state
        self.state = next_state

        return terminal, next_state, reward


class Agent_AC:
    def __init__(self, setting):
        nstates = 11 if setting.lesion else 16
        self.nstates = nstates
        self.lr = setting.lr
        self.decay = setting.decay
        self.preference = np.zeros((3, 2))  # action is made only state [2, 3, 4]
        self.preference[0, 0] = 1
        self.preference[2, 1] = 1
        self.exptd_value = np.zeros(nstates, )
        self.eligibility = np.zeros(nstates, )
        self.eligibility[0] = 1

        # for restoration
        self.state_prv = 0

    def re_init(self):
        self.preference[:, :] = 0  # action is made only state [2, 3, 4]
        self.preference[0, 0] = 1
        self.preference[2, 1] = 1
        self.exptd_value[:, ] = 0
        self.eligibility[:, ] = 0
        self.eligibility[0] = 1

        # for restoration
        self.state_prv = 0

    def action(self, state):
        """Action is required only if it is state 1"""
        if state in [2, 3, 4]:
            preference = self.preference[state-2, :]
            return random.choice([0, 1], p=np.exp(preference)/np.exp(preference).sum())
        return None

    def update(self, action, state, reward):
        if state != 0:
            # expected value
            value = self.exptd_value[self.state_prv]
            value_tp1 = self.exptd_value[state]
            # reward prediction error
            RPE = reward + self.decay.value * value_tp1 - value
            # update expected value
            self.exptd_value += self.lr.value * self.eligibility * RPE
        # for background state is fixed at 0
        if self.state_prv in [2, 3, 4]:
            self.preference[self.state_prv-2, action] += RPE * self.lr.preference  # * self.eligibility[self.state_prv]
        # eligibility update
        self.eligibility = self.eligibility * self.decay.eligibility
        self.eligibility[state] = 1
        # restore the state
        self.state_prv = state
        return RPE

    def reset(self):
        """reset eligibility trace, call this function at the beginning of each trial """
        self.state_prv = 0
        self.eligibility = np.zeros(self.nstates, )
        self.eligibility[0] = 1


class Record:
    def __init__(self, path):
        self.path = path
        # self.df = pd.DataFrame(columns=['observation', 'belief', 'RPE'])
        self.data = Dict2Class({'odor': [], 'action': [], 'reward': [], 'RPE': []})

    def creat(self):
        pass

    def reset(self):
        """"""
        self.data.RPE.append([])
        self.data.odor.append([])
        self.data.action.append([])
        # self.data.reward.append([])

    def write(self, state, action, RPE):
        self.data.RPE[-1].append(RPE)
        if state in [2, 3, 4]:
            self.data.odor[-1].append(state-2)
            self.data.action[-1].append(action)
        # self.data.reward[-1].append(reward)


def plot_RPE(RPE_rwd, RPE_odor):
    """
    plot for each individual block, group by the odor and choice
    """
    fig = plt.figure(num='RPE')
    h = []
    for key in ['pos_fst', 'pos_lst', 'neg_fst', 'neg_lst']:  # RPE_rwd[0].keys():
        value = np.vstack([i[key] for i in RPE_rwd])
        x = range(10) if 'fst' in key else range(13, 23)
        color = [.2, .2, .2] if 'pos' in key else [.7, .7, .7]
        h.append(plt.errorbar(x, value.mean(axis=0),
                              yerr=value.std(axis=0)/np.sqrt(value.shape[0]),
                              color=color))
    plt.xticks([5, 18], ['first 10', 'last 10'], fontsize=14)
    plt.legend([h[0], h[2]], ['unexpected reward', 'unexpected omission'], fontsize=14)
    plt.ylabel('firing to odor (spk/s)', fontsize=14)

    fig2 = plt.figure(num='free-forced')
    i, h = 0, []
    for key in ['forced_high', 'forced_low', 'free_high', 'free_low']:
        value = np.concatenate([i[key] for i in RPE_odor])
        color = [.2, .2, .8] if 'high' in key else [.8, .2, .2]
        h.append(plt.bar(i, value.mean(), color=color))
        plt.errorbar(i, value.mean(), yerr=value.std()/np.sqrt(value.size))
        i += 1
    plt.xticks([.5, 2.5], ['forced', 'free'], fontsize=14)
    plt.legend([h[0], h[1]], ['high', 'low'], fontsize=14)
    plt.ylabel('firing to odor (spk/s)', fontsize=14)
    plt.show()

    return [fig, fig2]


def data_extract(RPE, odor, action, block_size):
    """
    plot for each individual block, group by the odor and choice
    """
    RPE = np.array(RPE, dtype=object).reshape(4, block_size)
    odor = np.array(odor, dtype=int).reshape(4, block_size)
    action = np.array(action).reshape(4, block_size)
    # TODO: plot
    RPE_odor = {a+b: [] for a in ['free', 'forced'] for b in ['_high', '_low']}
    RPE_rwd = {a+b:  [] for a in ['pos', 'neg'] for b in ['_fst', '_lst']}

    for nB, (RPE_, odor_, action_) in enumerate(zip(RPE, odor, action)):
        # 2nd block, reward omission at left well (state 5), unexpected delivery at the right well (state 10)
        # 3rd block, unexpected delivery at left well (state 5 6)
        # 4th block, reward omission at left well (state 6), unexpected delivery at the right well (state 11)
        # prediction error in response to the reward
        RPE_neg, RPE_pos = [], []
        correct = np.logical_and(odor_ == 0, action_ == 0)
        correct = np.logical_or(correct, odor_ == 1)
        correct = np.logical_or(correct, np.logical_and(odor_ == 2, action_ == 1))
        if nB == 0:  #  or nB == 2
            continue
        if nB == 1:
            RPE_pos = [i[3] for i in RPE_[np.logical_and(odor_ == 2, correct)]]
            RPE_neg = [i[3] for i in RPE_[np.logical_and(odor_ == 0, correct)]]
        if nB == 2:
            RPE_pos = [i[3:5] for i in RPE_[np.logical_and(odor_ == 0, correct)]]
        if nB == 3:
            RPE_pos = [i[4] for i in RPE_[np.logical_and(odor_ == 2, correct)]]
            RPE_neg = [i[4] for i in RPE_[np.logical_and(odor_ == 0, correct)]]

        # print(nB, len(RPE_pos), len(RPE_neg))
        if RPE_pos:
            RPE_rwd['pos_fst'].append(RPE_pos[:10])
            RPE_rwd['pos_lst'].append(RPE_pos[-10:])

        if RPE_neg:
            RPE_rwd['neg_fst'].append(RPE_neg[:10])
            RPE_rwd['neg_lst'].append(RPE_neg[-10:])

    for nB, (RPE_, odor_, action_) in enumerate(zip(RPE, odor, action)):
        # reward prediction error in response to odor
        if nB == 0: # or nB == 2
            continue
        high_odor = 0 if nB == 2 else 2
        high_action = 0 if nB == 2 else 1
        RPE_odor['free_low'].append([i[1] for i in RPE_[np.logical_and(odor_ == 1, action_ != high_action)]])
        RPE_odor['free_high'].append([i[1] for i in RPE_[np.logical_and(odor_ == 1, action_ == high_action)]])
        RPE_odor['forced_low'].append([i[1] for i in RPE_[odor_ != high_odor]])
        RPE_odor['forced_high'].append([i[1] for i in RPE_[odor_ == high_odor]])

    # 
    for key in RPE_odor.keys():
        RPE_odor[key] = RPE2firing(np.hstack(RPE_odor[key]))

    for key in RPE_rwd.keys():
        if 'pos' in key:
            RPE_rwd[key] = RPE2firing(np.vstack([np.array(i).T for i in RPE_rwd[key]]))
        else:
            RPE_rwd[key] = RPE2firing(np.array(RPE_rwd[key]))
    return RPE_rwd, RPE_odor


def interact(task, agent, record, numTrials):
    for n in range(numTrials):
        # start a new trial, reset task and agent's belief
        task.reset()
        agent.reset()
        record.reset()
        # initial observation
        state = 0
        while True:
            action = agent.action(state)
            terminal, next_state, reward = task.step(action)
            if terminal:
                break
            RPE = agent.update(action, next_state, reward)
            # record the information
            record.write(state, action, RPE)
            # update state
            state = next_state
    return record


def plot_save(path, figs):
    for fig in figs:
        if type(fig) == list:
            plot_save(fig)
        else:
            name = fig._label+'_'+str(num_fig)+'.png'
            fig.savefig(path+name)
            fig.clf()


def main():
    """
    :return:
    """
    task = OdorTask(setting)
    agent = Agent_AC(setting)
    numTrials = len(setting.block.left) * setting.block.size

    records = []
    for i in range(setting.nRuns):
        # print('* '*10 + 'nRun: {}'.format(i))
        task.re_init()
        agent.re_init()
        record = Record(path)
        records.append(interact(task, agent, record, numTrials))
    
    RPE_rwd, RPE_odor = [], []
    for r in records:
        RPE_r, RPE_o = data_extract(r.data.RPE,
                                    r.data.odor,
                                    r.data.action,
                                    setting.block.size)
        RPE_rwd.append(RPE_r)
        RPE_odor.append(RPE_o)

    figs = plot_RPE(RPE_rwd, RPE_odor)
    plot_save(path, figs)


if __name__ == '__main__':
    # block: 1/2/3/4: long_delay/short delay/big/small
    path = '/Users/zhangz31/Dropbox (SchoenbaumLab)/project/coorperate_with_yuji/code/figs/'  # mac
    num_fig = 0

    setting = Dict2Class({'lesion': True, 'nRuns': 10,
                          # 'block': {'left': [1, 2, 1, 2], 'right': [2, 1, 2, 1], 'size': 50},
                          # 'block': {'left': [3, 4, 3, 4], 'right': [4, 3, 4, 3], 'size': 50},
                          'block': {'left': [2, 1, 3, 4], 'right': [1, 2, 4, 3], 'size': 200},  # used in the task
                          'decay': {'value': 0.99, 'eligibility': 0.9},
                          'lr':    {'value': 0.1, 'preference': 0.2}}
                         )
    main()
