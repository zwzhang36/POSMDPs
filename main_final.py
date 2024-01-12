#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:57:41 2023

@author: Zhewei Zhang

reproduce the results in Takahashi-2016-Temporal specificity of reward prediction errors
            signaled by putative dopamine neurons in rat VTA depends on the ventral striatum
"""
import copy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from os.path import exists
from scipy.io import savemat, loadmat
from tools.plot_kits import PlotTools


from tools.helper_final import Dict2Class, sample
from tools.helper_final import Agent, Recorder

# In[]
# absolute path where the program is
path = '/Users/zhangz31/Dropbox (SchoenbaumLab)/project/coorperate_with_yuji/results/'  # mac
# path = 'E:/Zhewei/full_combine/'
def interact(path, parameters, setting, dwell_learning, trans_learning, prefix, Agent, Recorder, placeholder=1):
    # initial
    agent = Agent(copy.deepcopy(parameters))
    recorder = Recorder(path)
    reward_obvs = agent.task.r_obs
    observations = []
    indicators = [[], []]
    # generate data for training
    numBlock = len(setting.block.left)
    for n_block in range(numBlock):
        observations.extend(sample(n_trials=setting.block.size,
                                   conditions=[setting.block.left[n_block], setting.block.right[n_block]],
                                   timing=parameters.timing,
                                   task_space=setting.task_space))
        indicators[0].extend([n_block]*setting.block.size)
        indicators[1].extend(list(range(setting.block.size)))

    # training
    for n_trial, (n_block, i, o_trial) in enumerate(zip(indicators[0],  indicators[1], observations)):
        # reset beta and other variables
        if n_trial%(setting.block.size * 4) == 0:
            agent.prior_reset()
            agent.reset(first=True)
        else:
            agent.reset()
        recorder.reset()
        
        # if n_trial%(setting.block.size * 4) == 0:
        #     agent.T_baseline_reset(0.15)
        
        # TODO: double check the first and last step
        # skip the first and last time step
        index_start, index_end = 1, len(o_trial) - 1
        # index_start = 0
        # index_end = len(o_trial)-1 if n_trial == len(observations)-1 else len(o_trial)
        if True:  # n_trial == 0:
            index_start = 1
            r_2 = 1 if o_trial[1] in reward_obvs else 0  # -0.01
            agent.update_observation(o_trial[0])
            recorder.record(o_trial[1], r_2, agent.state_value, agent.RPE,
                            agent.belief[-1], agent.alpha[-1], agent.beta[-1],
                            agent.discount)

        # skip the first timestep
        for n_obv in range(index_start, index_end):
            # observations
            o_t = o_trial[n_obv]
            if n_obv != len(o_trial)-1:
                o_tp1 = o_trial[n_obv + 1]
            else:
                observations[n_trial+1][0]
            # reward
            r_tp1 = 1 if o_tp1 in reward_obvs else 0  # -0.01
            # feed observation 
            agent.update_observation(o_t)
            # the probability of transition from state s at time t give all observations o_1,...,O_t+1
            agent.state_and_transition_estimation(o_tp1)
            # update the eligibility trace based on the above probability
            agent.e_trace_update()
            # update the state value based on the eligibility trace (t+1) and RPE (t+1)
            agent.value_update(o_tp1, r_tp1)
            # update the dwell distribution at the time of each non-empty observations
            if dwell_learning:
                agent.dwell_update(o_tp1)
            # save the state value and RPE for each state at each time point
            recorder.record(o_tp1, r_tp1, agent.state_value, agent.RPE,
                            agent.belief[-1], agent.alpha[-1], agent.beta[-1],
                            agent.discount)

        # if n_trial == len(observations)-1:
        #     agent.update_observation(o_trial[-1])
        #     r_last = 1 if o_trial[-1] in reward_obvs else 0
        #     recorder.record(o_trial[-1], r_last, agent.state_value, agent.RPE,
        #                     agent.belief[-1], agent.alpha[-1], agent.beta[-1])
        
        # update transition matrix
        if trans_learning: # and not parameters.lesion:
            agent.transition_update()
        
        # record
        if dwell_learning:
            recorder.record_dwell(agent.D_matrix)

        recorder.record_trans([agent.prior])

    return recorder


def main_run(N, task_space = 'poz_rwd'):
    i = -1
    for task_space in ['poz_rwd', 'poz']:
        for lesion in [True, False]: # False, 
            i += 1
            if i != N:
                continue
                
            # setting for task
            # block: 1/2/3/4: long_delay/short delay/big/small
            print("*" * 40)
            print(task_space, lesion)
            nBlocks = 20
            setting = Dict2Class({'task_space': task_space, 'lesion': lesion,
                                  'replay': False, 'nRuns': 20, 'nBlocks': nBlocks,
                                  # 'condition ': {'left': [2], 'right': [2], 'size': 200}
                                  'block': {'left': [2, 1, 3, 4]*nBlocks,
                                            'right': [1, 2, 4, 3]*nBlocks, 'size': 50}  # used in the task
                                  })
            
            # # parameters that are changed with lesion
            # hyper-parameters  for learning
            parameters = Dict2Class({'state_map': task_space,
                                     'lesion': lesion,
                                     'timing': {'short': 0.5, 'long': 4.0},
                                     'dwell_std': 0.05,
                                     'dwell_lr':  0.3,  # better than 0.1
                                     'value_lr':  0.5,
                                     'trans_lr':  0.5,
                                     't_whole': 7,
                                     'tau': 0.05, # 0.002 in Angela's paper, 0.04 wokrs well for 'poz'
                                     'dt': 0.1,
                                     'largerP': 0.55, # for lesion in 'poz' state space
                                     'e_decay': 0.99,
                                     'baseline': Dict2Class({'T': 0 if task_space == 'poz' else 1e-20 if not lesion else 0.15,
                                                             'T_big': 1e-10, 
                                                             'D': 1e-4,
                                                             'O': 1e-4,
                                                             'O_cue': 1e10,
                                                             'O_null': 0.05})
                                     })
            dwell_learning = task_space == 'poz' 
            trans_learning = task_space == 'poz_rwd'
            
            # label/file name
            prefix = 'sham_HPC_' if not setting.lesion else 'lesion_HPC_'  # +str(parameters.largerP) + '_'
            prefix += task_space + '_transitionProb_'
    # run
    if i < N:
        return 
    # if exists(path+prefix[:-1]+'.mat'):
    #     return
    recorders = []
    for n in range(setting.nRuns):
        print(n)
        recorders.append(interact(path, parameters, setting, 
                                  dwell_learning, trans_learning, prefix,
                                  Agent, Recorder))        
    savemat(path+prefix[:-1]+'.mat', {'recorders':recorders, 'setting':setting, 
                                      'parameters':parameters, 'prefex':prefix,
                                      'dwell_learning': dwell_learning, 
                                      'trans_learning': trans_learning})
    
# In[]
if __name__ == '__main__':
    import time
    for i in range(2,3):
        main_run(i)
    # pass



