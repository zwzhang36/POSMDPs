#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:50:42 2023

@author: zhangz31
"""

params1_list = [0]          # dwell_std if 'poz_rwd'
params2_list = [0]          # baseline T
params3_list = [0.001]      # baseline D
params4_list = [1e-6]       # baseline O
params5_list = [0.05]       # tau 
params6_list = [0.05]       # dwell_std if 'poz'
params7_list = [0.3]        # dwell_lr
params8_list = [0.55]       # largerP, for lesion if 'poz' 

# In[]
# setting for task
# block: 1/2/3/4: long_delay/short delay/big/small
task_space = 'poz'  # 'poz' or 'poz_rwd'
nBlocks = 4 if task_space == 'poz' else 4
setting = Dict2Class({'task_space': task_space, 'lesion': lesion,
                      'replay': False, 'nRuns': 20, 'nBlocks': nBlocks,
                      # 'block': {'left': [2], 'right': [2], 'size': 200}
                      'block': {'left': [2, 1, 3, 4]*nBlocks,
                                'right': [1, 2, 4, 3]*nBlocks, 'size': 50}  # used in the task
                      })

# # parameters that are changed with lesion
# dwell_std = 0.05 if setting.lesion else 0.05
# hyper-parameters  for learning
parameters = Dict2Class({'state_map': task_space,
                         'lesion': lesion,
                         'timing': {'short': 0.5, 'long': 4.0},
                         'dwell_std': 0.05 if task_space == 'poz',
                         'dwell_lr': 0.05,  # better than 0.1
                         'value_lr': 0.3 if task_space == 'poz'
                         'trans_lr': 0 if task_space == 'poz' ,
                         't_whole': 7,
                         'tau': [0.05] if task_space == 'poz', # 0.002 in Angela's paper, 0.04 wokrs well for 'poz'
                         'dt': 0.1,
                         'largerP': 0.55 if task_space == 'poz' else 0, # for lesion in 'poz' state space
                         'e_decay': 0.999 if task_space == 'poz' else 0.999,
                         'baseline': Dict2Class({'T': 0, 
                                                 'D': 0.001,
                                                 'O': 1e-6,
                                                 'O_null': 0.1})
                         })
dwell_learning = True if task_space == 'poz' else False
trans_learning = True if (task_space == 'poz_rwd' and not setting.lesion) else False

parameters.dwell_lr = parameters.dwell_lr if dwell_learning else 0
# label
prefix = 'sham_HPC_' if not setting.lesion else 'lesion_HPC_'  # +str(parameters.largerP) + '_'
prefix += str(params1) + '_'
prefix += str(params2) + '_'
prefix += str(params3) + '_'
prefix += str(params4) + '_'
prefix += str(params5) + '_'
prefix += str(params6) + '_'
prefix += str(params7) + '_'
prefix += str(params8) + '_0.1_noReplay_'
