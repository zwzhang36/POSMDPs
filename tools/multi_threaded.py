# -*- coding: utf-8 -*-
"""
Created on Wed Nov 3 2021

@author: Zhewei Zhang


"""
import copy
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class Agent():
    def __init__(self, nstates, transition, observation, parameters):
        self.nstates = nstates
        self.tran_mtx = transition.tran_mtx
        self.obz_mtx = observation.obz_mtx
        self.parameters = parameters
        # init variables
        self.RPE = [] # reward prediction error
        self.belief = np.zeros(nstates, )
        self.belief_prv = np.zeros(nstates, )
        self.weight = np.zeros(nstates, )
        self.exptd_value = np.zeros(nstates, )
        self.eligibility = np.zeros(nstates, )

    def update(self, action, observation):
        # belief update
        self.belief_update(action, observation)
        # value update
        self.weight_update(observation)
        # eligibility update
        self.eligibility_update()

    def belief_update(self, action, observation):
        self.belief_prv = copy.deepcopy(self.belief)
        tran_mtx = self.tran_mtx
        obz_mtx = self.obz_mtx[:, :, observation]

        # TODO: double check, better to be safe
        belief = np.matmul(np.multiply(obz_mtx, tran_mtx).T, self.belief)
        if belief.sum() == 0:
            a=1
        else:
            self.belief = belief/belief.sum()

    def eligibility_update(self):
        decay = self.parameters.eligibility.decay * self.parameters.value.decay
        self.eligibility = self.eligibility * decay + self.belief
        self.eligibility[self.eligibility < 0] = 0
        self.eligibility[self.eligibility > 1] = 1

    def weight_update(self, observation):
        # parameters for learning
        parameters = self.parameters
        decay_v = parameters.value.decay
        lr, decay_w = parameters.weight.lr, parameters.weight.decay
        # expected value = sum(weight*belief)
        value = np.dot(self.weight, self.belief_prv)
        value_tp1 = np.dot(self.weight, self.belief)
        # reward prediction error
        reward = 0 if observation < 2 else 1
        self.RPE = reward + decay_v * value_tp1 - value
        self.weight = self.weight * decay_w + self.RPE * lr * self.eligibility
        # for background state is fixed at 0
        self.weight[0] = 0

    def act(self, state):
        action = None
        return action

    def reset(self):
        self.belief[:, ] = 0
        self.belief[0, ] = 1
        self.eligibility[:, ] = 0


class ObservationModel:
    """
    five possible observation:
        empty
        cue
        chocolate reward
        vanilla reward
        terminal reward
    """

    def __init__(self, nstates):
        self.nstates = nstates
        self.nobservations = []
        self.obz_mtx = []

    def step(self, state, next_state):
        prob = self.obz_mtx[state, next_state, :]
        return np.random.choice(self.nobservations, p=prob)


class classicObservationModel(ObservationModel):
    def __init__(self, nstates=10):
        super().__init__(nstates)
        self.nobservations = 5
        self.obz_mtx = np.zeros((nstates, nstates, self.nobservations))
        self.obz_mtx[0, 0, 0] = 1
        self.obz_mtx[0, 1, 1] = 1
        for i in range(1, self.nstates - 1):
            self.obz_mtx[i, i + 1, 0] = 0.1
            self.obz_mtx[i, i + 1, 2:] = 0.1


class resetObservationModel(ObservationModel):
    def __init__(self, nstates=10):
        super().__init__(nstates)
        self.nobservations = 5
        self.obz_mtx = np.array((nstates, nstates, self.nobservations))
        self.obz_mtx[0, 0, 0] = 1
        self.obz_mtx[0, 1, 1] = 1
        for i in range(1, nstates - 1):
            self.obz_mtx[i, 0, 2:] = 1 / 3  # TODO: check it
            self.obz_mtx[i, i + 1, 0] = 1


class StateTransitionModel:
    def __init__(self, nstates):
        self.nstates = nstates

    def step(self, state, action=None):
        """
        :param state:   current state
        :param action:  executed action
        :return: whether the episode is terminal, next state
        """
        if self.tran_mtx.sum() == 0:
            return 1, []
        else:
            return 0, np.random.choice(self.nstates, p=self.tran_mtx[state, :])


class classicTransitionModel(StateTransitionModel):
    """
    10 states (1 background (start/terminal state) state)
    """

    def __init__(self, nstates=10):
        super().__init__(nstates)
        self.tran_mtx = np.zeros((self.nstates, self.nstates))  # from row to column
        self.tran_mtx[0, 0:2] = 0.5
        for i in range(1, self.nstates - 1):
            # deterministic
            # self.tran_mtx[i, i + 1] = 1
            # probabilistic
            if i == self.nstates-2:
                self.tran_mtx[i, i] = 0.15
                self.tran_mtx[i, i + 1] = 0.85
            else:
                self.tran_mtx[i, i] = 0.125
                self.tran_mtx[i, i + 1] = 0.75
                self.tran_mtx[i, i + 2] = 0.125
        self.tran_mtx[-1, 0] = 1


class resetTransitionModel(StateTransitionModel):
    def __init__(self, nstates=10):
        super().__init__(nstates)
        self.tran_mtx = np.zeros((self.nstates, self.nstates))  # from row to column
        self.tran_mtx[:-1, 0] = 0.5
        self.tran_mtx[0, 0:2] = 0.5
        for i in range(1, self.nstates - 1):
            # deterministic
            # self.tran_mtx[i, i + 1] = 1
            # probabilistic
            if i == 1:
                self.tran_mtx[i, i] = 0.85
                self.tran_mtx[i, i + 2] = 0.15
            elif i == self.nstates-2:
                self.tran_mtx[i, i] = 0.15
                self.tran_mtx[i, i + 1] = 0.75
            else:
                self.tran_mtx[i, i] = 0.125
                self.tran_mtx[i, i + 1] = 0.75
                self.tran_mtx[i, i + 2] = 0.125
        self.tran_mtx

def setting_checking(setting):
    """ check the setting format """
    assert type(setting) == dict
    assert 'block' in setting.keys()
    assert 'block_size' in setting.keys()

    assert len(setting['block']) == 2
    assert setting['block'][0][0] in ['choc', 'van']
    assert setting['block'][1][0] in ['choc', 'van']
    assert setting['block'][0][1] in ['long', 'short']
    assert setting['block'][1][1] in ['long', 'short']

    assert len(setting['block_size']) == 2
    assert type(setting['block_size'][0]) == int
    assert type(setting['block_size'][1]) == int


class Task():
    """
    five possible outcome: chocolate, vanilla, terminal reward, empty and cue
    reward is delivered after a cue
    two interval: short/long interval
    """

    def __init__(self, setting):
        """
        setting: possible block and trial number in each block
        """
        setting_checking(setting)
        self.timer = setting['nstates']
        self.ntrial = 0
        self.block = setting['block'][0]
        self.setting = setting
        self.outcome = []
        self.delay = {'short_reward': 10, 'long_reward': 20, 'cue': 2}

    def step(self, action=None):
        """
        block: ['long', 'choc'],
            first element indicates the reward interval; 'short'/'long': short/long
            second element indicates the reward type; 'choc'/'van': chocolate/vanilla
        """
        self.timer -= 1
        nstates = self.setting['nstates']
        # check the cue
        terminal = False
        if self.timer == (nstates - self.delay['cue']):
            return terminal, 1
        # check the terminal reward
        if self.timer == 0:
            terminal = True
            return terminal, 4
        # check the inter reward
        inter_reward = self.timer == (nstates - self.delay[self.block[1]+'_reward'])
        if inter_reward:
            return terminal, 2 if self.block[0] == 'choc' else 3
        # nothing at all
        return terminal, 0

    def set_block_setting(self, setting):
        setting_checking(setting)
        self.ntrial = 0
        self.block = setting['block'][0]
        self.setting = setting

    def block_switch(self):
        """
        assuming there are only two different blocks
        :return:
        """
        if self.ntrial == (self.setting['block_size'][0] + 1):
            self.block = self.setting['block'][1]

    def reset(self):
        self.ntrial += 1
        self.timer = self.setting['nstates']
        self.outcome = 0
        self.block_switch()

    # def step(self, state, action):
    #     terminal, next_state = self.transition.step(state, action)
    #     observation = self.observation.step(state, next_state)
    #     return terminal, next_state, observation


class Record():
    def __init__(self, path):
        self.path = path
        # self.df = pd.DataFrame(columns=['observation', 'belief', 'RPE'])
        self.data = Dict2Class({'action': [], 'belief': [], 'observation': [], 'RPE': []})

    def creat(self):
        pass

    def reset(self):
        """"""
        self.data.RPE.append([])
        self.data.action.append([])
        # self.data.belief.append([])
        self.data.observation.append([])

    def write(self, action, observation, agent):
        self.data.RPE[-1].append(agent.RPE)
        self.data.action[-1].append(action)
        # self.data.belief[-1].append(agent.belief)
        self.data.observation[-1].append(observation)


def interact():
    def __init__(self, agent, task):
        self.task = task
        self.agent = agent

    def step(self, state, action):
        next_state = []
        observation = []
        return next_state, observation


def plot_RPE(RPE, block_size):
    RPE_block1 = np.array(RPE[:block_size[0]])
    RPE_block2 = np.array(RPE[block_size[0]:])

    fig = plt.figure(num='RPE')
    plt.subplot(1, 2, 1)
    plt.imshow(RPE_block1, aspect='auto', vmin=-1, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(RPE_block2, aspect='auto', vmin=-1, vmax=1)
    plt.show()
    return fig


def main():
    """
    :return:
    """
    nstates = 30
    setting['nstates'] = nstates
    transitionModel = classicTransitionModel(nstates)
    observationModel = classicObservationModel(nstates)

    task = Task(setting)
    agent = Agent(nstates, transitionModel, observationModel, parameters)

    record = Record(path)
    for n in range(sum(setting['block_size'])):
        # start a new trial, reset task and agent's belief
        task.reset()
        agent.reset()
        record.reset()
        # initial observation
        observation = 0
        terminal = 0
        while not terminal:
            action = None  # action = agent.act(observation)
            terminal, observation = task.step(action)
            agent.update(action, observation)
            # print(agent.eligibility.round(1))
            # print(agent.weight.round(1))
            record.write(action, observation, agent)

    plot_RPE(record.data.RPE, setting['block_size'])


if __name__ == '__main__':
    # absolute path where the program is
    path = 'D:/Dropbox (SchoenbaumLab)/data/multi_threaded'  # lab PC
    # path = '/Users/zhangz31/Dropbox (SchoenbaumLab)/data/multi_threaded/'  # mac

    # setting for task
    setting = {'block': [['choc', 'short'], ['choc', 'long']],
               'block_size': [75, 75],
               'nstates': 30}
    # hyper-parameters for learning
    parameters = Dict2Class({'weight': Dict2Class({'decay': 1-1e-3, 'lr': 0.1}),  #
                             'value': Dict2Class({'decay': 0.99}),
                             'eligibility': Dict2Class({'decay': .9})})  #.5
    main()

