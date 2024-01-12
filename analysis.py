# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:47:44 2022

@author: Plexon
"""

import os
import copy
import time
import matplotlib.pyplot as plt
# from time import sleep
from tools.helper_final import Dict2Class
from tools.plot_kits import PlotTools
from scipy.io import savemat, loadmat
import numpy as np
from os.path import exists

datapath = 'C:/Users/Plexon/Desktop/Yuji_ws_4cue/results_full/'
figspath = 'C:/Users/Plexon/Desktop/Yuji_ws_4cue/results_full/'

datapath = 'E:/Zhewei/full/'
figspath = 'E:/Zhewei/full/figs/'

# datapath = '/Users/zhangz31/Yuji_ws_4cue/results/'  # mac
# figspath = '/Users/zhangz31/Yuji_ws_4cue/results/'  # mac
# datapath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz_rwd/'  # mac
# figspath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz_rwd/'  # mac
# datapath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz/same_parameters_to_full_model/'  # mac
# figspath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz/same_parameters_to_full_model/'  # mac
datapath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/'  # mac
figspath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/'  # mac
# datapath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz/same_parameters_to_full_model/'  # mac
# figspath = '/Users/zhangz31/SchoenbaumLab Dropbox/哲伟 张/project/coorperate_with_yuji/results/poz/same_parameters_to_full_model/'  # mac
# datapath = 'D:/Yuji/poz_rwd_4_4cue/temp/'

block_size = 50
ntrials_sess = 4*block_size

task_space = 'poz_rwd'
block ='last' # 'first'

# time.sleep(7200)

def plot_func(file_name_sham, file_name_lesion, fileName=None):
    if fileName == None:
        name = file_name_sham[5:-4]
    else:
        name = fileName
    # if exists(figspath+name+'.txt'):
        # return 
    print(name)

    # print(file_name_sham, file_name_lesion)
    plot = PlotTools(figspath, task_space = task_space)
    # In[]
    data = loadmat(datapath +  file_name_lesion)['recorders'][0]
    # the data from last session are included
    ntrials = ntrials_sess
    if block == 'last':
        o_lesion     = [r[0][0][1]['o'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess) for r in data]
        v_lesion     = [r[0][0][1]['v'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        RPE_lesion   = [r[0][0][1]['RPE'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        beta_lesion  = [r[0][0][1]['beta'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        belief_lesion= [r[0][0][1]['belief'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        # trans_lesion = [r[0][0][1]['trans'][0, 0][-ntrials:,0,:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        trans_lesion = [r[0][0][1]['trans'][0, 0][-ntrials:,0,:,:] for r in data]
        # dwell_lesion = [r[0][0][1]['dwell'][0, 0][-ntrials:,0,:,:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
    elif block == 'first':
        o_lesion     = [r[0][0][1]['o'][0, 0][0][:ntrials] for r in data]
        v_lesion     = [r[0][0][1]['v'][0, 0][0][:ntrials] for r in data]
        RPE_lesion   = [r[0][0][1]['RPE'][0, 0][0][:ntrials] for r in data]
        beta_lesion  = [r[0][0][1]['beta'][0, 0][0][:ntrials] for r in data]
        belief_lesion= [r[0][0][1]['belief'][0, 0][0][:ntrials] for r in data]
        # trans_lesion = [r[0][0][1]['trans'][0, 0][:ntrials,0,:,:] for r in data]
        # trans_lesion = [r[0][0][1]['trans'][0, 0][:ntrials,1,:] for r in data]
        # dwell_lesion = [r[0][0][1]['dwell'][0, 0][:ntrials,0,:,:] for r in data]
    else:
        o_lesion     = [r[0][0][1]['o'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        v_lesion     = [r[0][0][1]['v'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        RPE_lesion   = [r[0][0][1]['RPE'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        beta_lesion  = [r[0][0][1]['beta'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        belief_lesion= [r[0][0][1]['belief'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        # trans_lesion = [r[0][0][1]['trans'][0, 0][:ntrials,0,:,:] for r in data]
        # trans_lesion = [r[0][0][1]['trans'][0, 0][block*ntrials:(block+1)*ntrials,0,:,:] for r in data]
        # dwell_lesion = [r[0][0][1]['dwell'][0, 0][block*ntrials:(block+1)*ntrials,0,:,:] for r in data]

    data = loadmat(datapath  + file_name_sham)['recorders'][0]
    if block == 'last':
        o_sham     = [r[0][0][1]['o'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        v_sham     = [r[0][0][1]['v'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        RPE_sham   = [r[0][0][1]['RPE'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        beta_sham  = [r[0][0][1]['beta'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        belief_sham= [r[0][0][1]['belief'][0, 0][0][-ntrials:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        # trans_sham = [r[0][0][1]['trans'][0, 0][-ntrials:,1,:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
        trans_sham = [r[0][0][1]['trans'][0, 0][-ntrials:,0,:,:] for r in data]
        # dwell_sham = [r[0][0][1]['dwell'][0, 0][-ntrials:,0,:,:][n:n+ntrials_sess] for n in range(0, ntrials, ntrials_sess)  for r in data]
    elif block == 'first':
        o_sham     = [r[0][0][1]['o'][0, 0][0][:ntrials:] for r in data]
        v_sham     = [r[0][0][1]['v'][0, 0][0][:ntrials:] for r in data]
        RPE_sham   = [r[0][0][1]['RPE'][0, 0][0][:ntrials:] for r in data]
        beta_sham  = [r[0][0][1]['beta'][0, 0][0][:ntrials:] for r in data]
        belief_sham= [r[0][0][1]['belief'][0, 0][0][:ntrials:] for r in data]
        # trans_sham = [r[0][0][1]['trans'][0, 0][:ntrials,1,:] for r in data]
        # trans_sham = [r[0][0][1]['trans'][0, 0][:ntrials,0,:,:] for r in data]
        # dwell_sham = [r[0][0][1]['dwell'][0, 0][:ntrials,0,:,:] for r in data]
    else:
        o_sham     = [r[0][0][1]['o'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        v_sham     = [r[0][0][1]['v'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        RPE_sham   = [r[0][0][1]['RPE'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        beta_sham  = [r[0][0][1]['beta'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        belief_sham= [r[0][0][1]['belief'][0, 0][0][block*ntrials:(block+1)*ntrials] for r in data]
        # trans_sham = [r[0][0][1]['trans'][0, 0][:ntrials,0,:,:] for r in data]
        # trans_sham = [r[0][0][1]['trans'][0, 0][block*ntrials:(block+1)*ntrials,1,:] for r in data]
        # dwell_sham = [r[0][0][1]['dwell'][0, 0][block*ntrials:(block+1)*ntrials,0,:,:] for r in data]

    # In[]
    # plot
    # plot.trans_learning(trans, o, block_size=block_size)
    
    # value_cue, value_rwd = [], []
    # for b in ['change', 'big', 'short']:
    #     cue, rwd = plot.get_rpe_sorted([RPE_sham, RPE_lesion], 
    #                                                 [o_sham, o_lesion], 
    #                                                 Dict2Class({'short': 0.5, 'long': 4.0}), 
    #                                                 block_size=block_size, 
    #                                                 block=b,
    #                                                 num_trials=int(5),
    #                                                 raw=False)
    #     value_cue.append(np.array(cue))
    #     value_rwd.append(np.array(rwd))
    # value_cue = np.concatenate(value_cue)
    # value_rwd = np.concatenate(value_rwd)
    
    for type in ['big','short','all','change']:  # [] 
        plot.plot_RPE([RPE_sham, RPE_lesion], [o_sham, o_lesion], Dict2Class({'short': 0.5, 'long': 4.0}), block_size=block_size, 
                      block=type, num_trials=int(10), raw=False)
        plt.title(fileName)
        plt.show()
    # plot.save_all_figs(name+'_')
    # for n in range(6):
    #     for type in ['change']:  # [] 'all','short','change',
    #         plot.plot_RPE([[RPE_sham[n]], [RPE_lesion[n]]], [[o_sham[n]], [o_lesion[n]]],
    #                       Dict2Class({'short': 0.5, 'long': 4.0}), block_size=block_size, 
    #                       block=type, num_trials=int(10), raw=False)
    #         # plt.ylims([-1, 5])
    #         plt.show()
    
    # plot.belief_given_O(beta_sham,     o_sham,   o=4, block_size=block_size)
    # plt.show()
    # plot.belief_given_O(beta_lesion,   o_lesion, o=4, block_size=block_size)
    # plt.show()
    # plot.belief_given_O(belief_sham,   o_sham,   o=4, block_size=block_size)
    # plt.show()
    # plot.belief_given_O(belief_lesion, o_lesion, o=4, block_size=block_size)
    # plt.show()
    
    # plot.plot_value(v_sham, block_size=block_size)
    # plt.show()
    # plot.plot_value(v_lesion, block_size=block_size)
    # plt.show()

    # for 'poz'
    # if task_space == 'poz':
    # plot.dwell_example_2nd(dwell_lesion)
    # plot.dwell_example_2nd(dwell_sham)
    # plot.dwell_example_4th(dwell_lesion)
    # plot.dwell_example_4th(dwell_sham)
    #     plot.blief_beta_example_left_2nd(beta_sham, belief_sham, o_sham, task_space=task_space, suffix = 'sham')
    #     plot.blief_beta_example_left_2nd(beta_lesion, belief_lesion, o_lesion, task_space=task_space, suffix = 'lesion')
    # plt.show()

    # for 'poz_rwd'
    # if task_space == 'poz_rwd':
    #     plot.trans_left_2nd(trans_lesion) # 
    #     plot.blief_beta_example_left_2nd(beta_sham, belief_sham, o_sham, task_space=task_space, suffix = 'sham')
    #     plot.blief_beta_example_left_2nd(beta_lesion, belief_lesion, o_lesion, task_space=task_space, suffix = 'lesion')
    # plt.show()

    # save figs
    if fileName == None:
        name = file_name_sham[5:-4]
    else:
        name = fileName
    print(name)
    
    # plot.save_all_figs(name+'_')
    # np.savetxt(figspath+name+'.txt', np.concatenate((value_cue, value_rwd)).round(5), fmt='%.4f')
    
    """
    plot.trans_learning(trans_sham, o_sham, block_size=50)
    plot.state_estimation([beta_sham, beta_lesion], [belief_sham, belief_lesion],
        [o_sham, o_lesion], block_size=block_size, state_map='poz_rwd')    
    plot.dwell_change(dwell, states=states, block_size=setting.block.size)
    # visualize belief when a specific observation comes out
    plot.belief_given_O(belief, o, o=4, block_size=setting.block.size,
    plot.plot_value(v_sham, block_size=block_size)
    plot.plot_value(v_lesion, block_size=block_size)
    
    plot.state_estimation(beta, belief, o, delivery=['r1'], omission=['l1'],
        block_size=block_size, state_map='poz_rwd')
    plot.state_estimation(beta, belief, o, delivery=['r3'], omission=['l3'],
        block_size=block_size, state_map='poz_rwd')
      
    sham = {'belief': belief_sham, 'beta': beta_sham, 'rpe': RPE_sham}
    lesion = {'belief': belief_lesion, 'beta': beta_lesion, 'rpe': RPE_lesion}
    for type in ['belief','beta','rpe']: # 'belief','beta','rpe'
    plot.belief_given_O(sham[type], o_sham, o=4, block_size=block_size,
      prefix='belief')
    plt.suptitle(type)
    plt.show()
    plot.belief_given_O(lesion[type], o_lesion, o=4, block_size=block_size,
       prefix='belief')
    plt.suptitle(type)
    plt.show()
   """ 
   
files = sorted(os.listdir(datapath))
params7_list = [0.05, 0.1, 0.15, 0.2]
params7_list = [0.15, 0.18]

def plot_func_helper(n):
    file = files[n]
    if file.split('_')[0] == 'lesion' or file[-4:] != '.mat':
        return
    if file.split('_')[0] == 'sham':
        lesion_file = file.split('_')
        lesion_file[0] = 'lesion'
        lesion_file = '_'.join(lesion_file)
        plot_func(file, lesion_file, fileName='_'.join(lesion_file[1:-2]))

def plot_func_helper_match(n):
    file = files[n]
    if file[-4:] != '.mat' or file.split('_')[0] == 'lesion':
        return
    
    if file.split('_')[0] == 'sham':
        for param7 in params7_list:
            lesion_file = file.split('_')
            lesion_file[8] = str(param7)
            lesion_file[0] = 'lesion'

            fileName = copy.deepcopy(lesion_file[1:-2])
            lesion_file[3] = '0.01'
            
            plot_func(file, '_'.join(lesion_file), fileName='_'.join(fileName))

# sham = 'sham_HPC_0.01_1e-06_0.005_0.0001_0.05_0.1_0.1_0.3_four_well_four_cue_.mat'
# lesion = 'lesion_HPC_0.01_0.01_0.005_0.0001_0.05_0.1_0.1_0.3_four_well_four_cue_.mat'
# sham = 'sham_HPC_0.01_1e-06_0.005_0.0001_0.05_0.075_0.1_0.3_four_well_four_cue_.mat'
# lesion = 'lesion_HPC_0.01_0.01_0.005_0.0001_0.05_0.075_0.1_0.3_four_well_four_cue_.mat'
# fileName = 'HPC_0.01_1e-06_0.005_0.0001_0.05_0.75_0.1_0.3'
# plot_func(sham, lesion, fileName)

if __name__ == '__main__':
    files = sorted(os.listdir(datapath))
    #files = [i[len(datapath):] for i in files]
    #files = sorted([datapath+i for i in os.listdir(datapath)], key=os.path.getmtime)
    for file in files[::-1]:
        if 'VS' in file:
            continue
        if file[-4:] != '.mat' or file.split('_')[0] == 'lesion':
            continue
        # if file[-4:] == '.mat' and file.split('_')[0] == 'lesion':
        #     plot_func('sham_VS_amplitude_0_poz_rwd_transitionProb.mat',file,fileName=file)
        # else:
        #     continue
        # if file.split('_')[0] == 'sham':
        #     lesion_file = file.split('_')
        #     lesion_file[0] = 'lesion'                
        #     plot_func(file, '_'.join(lesion_file), fileName='_'.join(lesion_file[1:-1]))
                
        # continue
        if file.split('_')[0] == 'sham':
            for param7 in [0]:
                lesion_file = file.split('_')
                # lesion_file[8] = str(param7)
                # if lesion_file[9] != '0.5':
                #     continue
                # if lesion_file[5] != '0.001':
                #     continue
                lesion_file[0] = 'lesion'
    
                fileName = copy.deepcopy(lesion_file[1:])
                # lesion_file[3] = '0.01'
    
                # print(file)
                # print('_'.join(lesion_file))
                # print('_'.join(fileName))
                try:
                    plot_func(file, '_'.join(lesion_file), fileName='_'.join(fileName))
                except:
                    print('unsuccessful: ',  file, '_'.join(lesion_file))
                    