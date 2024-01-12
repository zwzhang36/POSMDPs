import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# parameters for dwell time
dt = 0.1  # unit: second
t_range = 10  # unit: second


def RPE2firing(RPE):
    firing = np.zeros(RPE.shape)
    firing[RPE >= 0] = 3 + 4 * RPE[RPE >= 0]
    firing[RPE < 0] = 3 + 2 * RPE[RPE < 0]
    return firing


def data_extract(data, observations, timing, block_size=50, block='all', num_trials=10, lesion=False):
    """
    plot for each individual block, group by the odor and choice
    """
    n_Blocks = len(observations)/(block_size*4)
    if block == 'all':
        target_blocks = [int(1+4*(n_Blocks-1)), int(3+4*(n_Blocks-1))]
    elif block == 'short':
        target_blocks = [int(1+4*(n_Blocks-1))]  # [1]
    elif block == 'big':
        target_blocks = [int(3+4*(n_Blocks-1))]  # [3]
    elif block == 'change':
        target_blocks = [int(2+4*(n_Blocks-1))]  # [2]
    else:
        print('not a valid block type')
        return

    # parameters.state_map == 'wells'
    idx_cue, idx_1st_rwd, idx_2nd_rwd = int(.5/dt), int(1/dt+timing.short/dt), int(1.5/dt+timing.short/dt)

    data = np.array(data, dtype=object).reshape(-1, block_size)
    observations = np.array(observations, dtype=object).reshape(-1, block_size)
    #
    data_cue = {a + b: [] for a in ['pos', 'neg'] for b in ['_fst', '_lst']}
    data_rwd = {a + b: [] for a in ['pos', 'neg'] for b in ['_fst', '_lst']}

    def func(vector, ind):
        return sum(vector[ind])#  + sum(vector[ind-1]) + sum(vector[ind+1]))/3   # + sum(vector[ind-2]) + sum(vector[ind+2])
    
    for nB, (rpe, o) in enumerate(zip(data, observations)):
        # prediction error in response to the reward
        cue = np.array([1 if 2 in i else 2 for i in o])
        cue_gain, cue_loss = [], []
        rwd_gain, rwd_loss = [], []
        if nB not in target_blocks:
            continue
        if nB in list(range(1, 100, 4)):
            # print('short')
            cue_gain = [[func(i, idx_cue) for i in rpe[cue == 2]]]
            cue_loss = [[func(i, idx_cue) for i in rpe[cue == 1]]]
            rwd_gain = [[func(i, idx_1st_rwd) for i in rpe[cue == 2]]]
            rwd_loss = [[func(i, idx_1st_rwd) for i in rpe[cue == 1]]]
        if nB in list(range(2, 100, 4)):
            # if block == 'all':
            #     continue
            # print('change')
            if block == 'change':
                cue_gain = [[func(i, idx_cue) for i in rpe[cue == 1]]]
                rwd_gain = [[func(i, idx_2nd_rwd) for i in rpe[cue == 1]]]
                cue_loss = [[func(i, idx_cue) for i in rpe[cue == 1]]]
                rwd_loss = [[func(i, idx_1st_rwd) for i in rpe[cue == 1]]]
            else:
                cue_gain = [[func(i, idx_cue) for i in rpe[cue == 1]]]
                # if not lesion:
                #     rwd_gain = [[func(i, idx_1st_rwd) for i in rpe[cue == 1]],
                #                 [func(i, idx_2nd_rwd) for i in rpe[cue == 1]]]

        if nB in list(range(3, 100, 4)):
#            print('big')
            cue_gain = [[func(i, idx_cue) for i in rpe[cue == 2]]]
            cue_loss = [[func(i, idx_cue) for i in rpe[cue == 1]]]
            rwd_gain = [[func(i, idx_2nd_rwd) for i in rpe[cue == 2]]]
            rwd_loss = [[func(i, idx_2nd_rwd) for i in rpe[cue == 1]]]

        # print(nB, len(RPE_pos), len(RPE_neg))
        if rwd_gain:
            for idata in cue_gain:
                data_cue['pos_fst'].append(idata[:num_trials])
                data_cue['pos_lst'].append(idata[-num_trials:])
            for idata in rwd_gain:
                data_rwd['pos_fst'].append(idata[:num_trials])
                data_rwd['pos_lst'].append(idata[-num_trials:])

        if rwd_loss:
            for idata in cue_loss:
                data_cue['neg_fst'].append(idata[:num_trials])
                data_cue['neg_lst'].append(idata[-num_trials:])
            for idata in rwd_loss:
                data_rwd['neg_fst'].append(idata[:num_trials])
                data_rwd['neg_lst'].append(idata[-num_trials:])

    # convert reward prediction error into firing rate
    for key in data_rwd.keys():
        if 'pos' in key:
            data_cue[key] = np.vstack([np.array(i).T for i in data_cue[key]])
            data_rwd[key] = np.vstack([np.array(i).T for i in data_rwd[key]])
        else:
            data_cue[key] = np.hstack(data_cue[key]).T.reshape(-1, num_trials)
            data_rwd[key] = np.hstack(data_rwd[key]).T.reshape(-1, num_trials)

    return data_cue, data_rwd


def give_me_a_name(beliefs, observations, block_size, o):
    num_blocks = int(len(beliefs) / block_size)
    b_left, b_right = {i: [] for i in range(num_blocks)}, {i: [] for i in range(num_blocks)}
    for i, (b_trial, o_trial) in enumerate(zip(beliefs, observations)):
        # index = int(np.where(np.array(o_trial) == o)[0])
        n_block = int(i / block_size + 1e-9)
        if 2 in o_trial:  # left odor
            # b_left[n_block].append(b_trial[index])
            b_left[n_block].append(np.array(b_trial))
        else:  # right odor
            # b_right[n_block].append(b_trial[index])
            b_right[n_block].append(np.array(b_trial))
    return b_left, b_right


class PlotTools(object):
    def __init__(self, path, task_space):
        self.figs = []
        self.path = path
        self.task_space = task_space
        
    def save_all_figs(self, prefix):
        self.plot_save(self.figs, prefix)

    def plot_save(self, figs, prefix=[]):
        # save figs in folder path
        for fig in figs:
            if type(fig) == list:
                self.plot_save(fig, prefix)
            else:
                name = prefix + fig._label + '.png' # png
                fig.savefig(self.path + name)
                # name = prefix + fig._label + '.eps' # png
                # fig.savefig(self.path + name, format = 'eps')
                fig.clf()

    def plot_value(self, value, block_size=50):
        num_blocks = int(len(value[0]) / block_size)
        value_blocks = [np.array_split(v, num_blocks) for v in value]

        fig = plt.figure(num='value changes', figsize=(12, 8))
        for i, n_block in enumerate(range(num_blocks-4, num_blocks)):
            # v_average = np.array([np.array(i).mean(axis=0) for i in v])
            data = [v[n_block] for v in value_blocks]
            v_average = np.array([[ii[0] for ii in i] for i in data]).mean(axis=0)

            plt.subplot(1, 4, i+1)
#            plt.subplot(math.ceil(num_blocks/4), 4, n_block+1)
            plt.title('block' + str(n_block+1))
            if v_average.shape[-1] == 7:
                plt.xticks(range(7), ['start', 'cue L', 'cue R', 'rw L1', 'rw R1', 'rw 2', 'trial'])
            else:
                plt.xticks(range(10), ['trial', 'cue L', 'cue R', 'well L', 'well R',
                                      'rw L1', 'rw R1', 'rw L2', 'rw R2', 'trial'])

            plt.imshow(v_average, aspect='auto')
            plt.colorbar()
        self.figs.append(fig)


    def get_rpe_sorted(self, data, o, timing, block='all', block_size=50, num_trials=10, raw=True):
        def get_data_cue_rwd(data, o, block=block, lesion=False):
            data_cue, data_rwd = [], []
            for data_this, o_this in zip(data, o):
                temp = data_extract(data_this, o_this, timing, block_size, block=block, num_trials=num_trials, lesion=lesion)
                data_cue.append(temp[0])
                data_rwd.append(temp[1])
            return data_cue, data_rwd
        data_cue_sham, data_rwd_sham =  get_data_cue_rwd(data[0], o[0], block=block, lesion=False)
        data_cue_lesion, data_rwd_lesion =  get_data_cue_rwd(data[1], o[1], block=block, lesion=True)
        
        if block == 'all':
            _, data_rwd_3rd_sham =  get_data_cue_rwd(data[0], o[0], block='change', lesion=False)
            _, data_rwd_3rd_lesion =  get_data_cue_rwd(data[1], o[1], block='change', lesion=True)
        
            value_cue, value_rwd = [], []
            for data_cue, data_rwd, data_3rd in zip([data_cue_sham, data_cue_lesion],  [data_rwd_sham, data_rwd_lesion], [data_rwd_3rd_sham, data_rwd_3rd_lesion]):
                for key in ['pos_fst', 'pos_lst', 'neg_fst', 'neg_lst']: 
                    if raw: # RPE_rwd[0].keys():
                        value_cue.append(np.array([i[key] for i in data_cue]).mean(axis=1).mean(axis=0))
                        value_rwd.append(np.array([i[key] for i in data_rwd]).mean(axis=1).mean(axis=0))
                    else:
                        value_cue.append(np.array([RPE2firing(i[key]) for i in data_cue]).mean(axis=1).mean(axis=0))
                        value_rwd.append(np.array([RPE2firing(i[key]) for i in data_rwd]).mean(axis=1).mean(axis=0))
                
                for key in ['pos_fst', 'pos_lst']: 
                    if raw: # RPE_rwd[0].keys():
                        value_rwd.append(np.array([i[key] for i in data_3rd]).mean(axis=1).mean(axis=0))
                    else:
                        value_rwd.append(np.array([RPE2firing(i[key]) for i in data_3rd]).mean(axis=1).mean(axis=0))
        else:
            value_cue, value_rwd = [], []
            for data_cue, data_rwd in zip([data_cue_sham, data_cue_lesion],  [data_rwd_sham, data_rwd_lesion]):
                for key in ['pos_fst', 'pos_lst', 'neg_fst', 'neg_lst']: 
                    if raw: # RPE_rwd[0].keys():
                        value_cue.append(np.array([i[key] for i in data_cue]).mean(axis=1).mean(axis=0))
                        value_rwd.append(np.array([i[key] for i in data_rwd]).mean(axis=1).mean(axis=0))
                    else:
                        value_cue.append(np.array([RPE2firing(i[key]) for i in data_cue]).mean(axis=1).mean(axis=0))
                        value_rwd.append(np.array([RPE2firing(i[key]) for i in data_rwd]).mean(axis=1).mean(axis=0))
                

        return value_cue, value_rwd


    def plot_RPE(self, data, o, timing, block='all', block_size=50, num_trials=10, raw=True):
        """
        plot for each individual block, group by the odor and choice
        block: 'short' (from first to second block), 'big' (from third to forth block), and 'all'(both)
        data:[data_sham, data_lesion]
        o:[o_sham, data_lesion]
        """
        def get_data_cue_rwd(data, o, block=block, lesion=False):
            data_cue, data_rwd = [], []
            for data_this, o_this in zip(data, o):
                temp = data_extract(data_this, o_this, timing, block_size, block=block, num_trials=num_trials, lesion=lesion)
                data_cue.append(temp[0])
                data_rwd.append(temp[1])
            return data_cue, data_rwd
        data_cue_sham, data_rwd_sham =  get_data_cue_rwd(data[0], o[0], lesion=False)
        data_cue_lesion, data_rwd_lesion =  get_data_cue_rwd(data[1], o[1], lesion=True)
        
        if block == 'all':
            _, data_rwd_3rd_sham =  get_data_cue_rwd(data[0], o[0], block='change', lesion=False)
            _, data_rwd_3rd_lesion =  get_data_cue_rwd(data[1], o[1], block='change', lesion=True)
        
        for poz in ['cue', 'rwd']:
            # if raw:
            #     # ylim = [-.4, .4] if poz=='cue' else [-.6, .6]
            #     if self.task_space == 'poz_rwd':
            #         ylim = [-2 , 2] if poz=='cue' else [-1, 1] # 
            #     elif self.task_space == 'poz':
            #         ylim = [-.4, .4] if poz=='cue' else [-.6, .6]
            # else:
            #     if self.task_space == 'poz_rwd':
            #         ylim = [2.2, 4.3] if poz=='cue' else [1.9, 4.9] # 
            #     elif self.task_space == 'poz':
            #         ylim = [2.3, 4.7] if poz=='cue' else [1.3, 6.7] # 

            if poz == 'cue':
                data_value = [data_cue_sham, data_cue_lesion] 
            else:
                data_value = [data_rwd_sham, data_rwd_lesion] 
            title = 'RPE_' + block + '_' + poz
            self.figs.append(plt.figure(num=title, figsize = (8, 6)))
            plt.title(title)
            h = []
            if poz == 'cue':
                legend = ['Cue paired with high reward', 'Cue paired with low reward']
            else:
                legend = ['reward delivery', 'reward omission']
            for i in range(2):
                # plt.subplot(1, 2, int(i+1))
                data_value_i = data_value[i]
                for key in ['pos_fst', 'pos_lst', 'neg_fst', 'neg_lst']:  # RPE_rwd[0].keys():
                    value = [i[key] for i in data_value_i] if raw else [RPE2firing(i[key]) for i in data_value_i]
                    value = np.array(value)
                    value = value.mean(axis=1) # averaged across short/big/change 
                    value = value.reshape(-1, num_trials)  # check this when average across multiple blocks and runs
                    # x = range(1) if 'fst' in key else range(3, 4)
                    x = range(num_trials) if 'fst' in key else range(num_trials+3, 2*num_trials+3)
                    if i ==0:
                        color = [.125, .512, .5] if 'pos' in key else [.473, .773, .75]
                    if i ==1:
                        color = [0.809, 0.430, 0.426] if 'pos' in key else [0.852, 0.672, 0.672]
                    baseline = 0
                    # if key in ['neg_fst', 'neg_lst'] and i == 1:
                    #     baseline = 0.2
                    # if key in ['pos_fst', 'pos_lst'] and i == 1:
                    #     baseline = -0.1
#                    if poz == 'cue':
#                        if 'fst' in key:
#                            baseline = value.mean(axis=0)[0]
#                        else:
#                            baseline = value_fst.mean(axis=0)[0]
                    h.append(plt.errorbar(x, value.mean(axis=0) - baseline,
                                          yerr=value.std(axis=0) / np.sqrt(value.shape[0]),
                                          color=color, fmt='o-', linewidth=4))
                    if 'fst' in key:
                        value_fst = value
                # plt.ylim(ylim)
                plt.xticks([num_trials/2, num_trials*1.5+3],
                           ['first '+str(num_trials) + ' trials', 'last '+ str(num_trials) + ' trials'], fontsize=30)
                plt.yticks(fontsize=20)
                # plt.legend([h[0], h[2]], legend, fontsize=25)
                if raw:
                    plt.ylabel('reward prediction error', fontsize=30)
                else:
                    plt.ylabel('normalizaed firing rate', fontsize=30)
        if block == 'all':
            poz = 'rwd'
            data_value = [data_rwd_3rd_sham, data_rwd_3rd_lesion] 
            h = []
            for i in range(2):
                data_value_i = data_value[i]
                for key in ['pos_fst', 'pos_lst']:  # RPE_rwd[0].keys():
                    value = [i[key] for i in data_value_i] if raw else [RPE2firing(i[key]) for i in data_value_i]
                    value = np.array(value)
                    value = value.mean(axis=1) # averaged across short/big/change 
                    value = value.reshape(-1, num_trials)  # check this when average across multiple blocks and runs
                    # x = range(1) if 'fst' in key else range(3, 4)
                    x = range(num_trials) if 'fst' in key else range(num_trials+3, 2*num_trials+3)
                    if i ==0:
                        color = [.227, .480, .746]
                    if i ==1:
                        color = [.648, .465, .695]
                    baseline = 0
                    # if i == 1:
                    #     baseline = -0.1

                    h.append(plt.errorbar(x, value.mean(axis=0) - baseline,
                                          yerr=value.std(axis=0) / np.sqrt(value.shape[0]),
                                          color=color, fmt='o-', linewidth=4))
                    if 'fst' in key:
                        value_fst = value

    def dwell_change(self, dwell, states='rewards', block_size=50):
        # only for the last four blocks 
        num_blocks = int(len(dwell[0]) / block_size)
        fig = plt.figure(num='dwell learning', figsize=(12, 8))
        outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        dwell_blocks = [np.array_split(d, num_blocks) for d in dwell]
        for n_block in range(num_blocks):
            inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                     subplot_spec=outer[n_block], wspace=0.1, hspace=0.1)
            i = 0
            d = [d[n_block] for d in dwell_blocks]
            # plt.suptitle('nblock: ' + str(n_block + 1))
            if n_block in [0, 1]:
                if states=='rewards':
                    states_new = [3, 4]
                else:
                    states_new = states
            elif n_block in [2, 3]:
                if states=='rewards':
                    states_new = [5, 6]
                else:
                    states_new = states
            else:
                print('unknown block number')
            color = {states_new[0]: {0: [0.4, 0, 0], -1: [0.8, 0, 0]},
                     states_new[1]: {0: [0, 0.4, 0], -1: [0, 0.8, 0]}}
            label = {states_new[0]: {0: 'left start', -1: 'left end'},
                     states_new[1]: {0: 'right start', -1: 'right end'}}

            for state in states_new:
                for index in [0, -1]:
                    ax = plt.Subplot(fig, inner[i])
                    data = np.array([i[index][index][state] for i in d])
                    # ind = int(6/dt) if state in [3, 4] else int(2/dt)
                    # ind = int(6/dt) if state in [5, 8] else int(2/dt)
                    ind = int(7/dt)
                    ax.plot(data.mean(axis=0)[0:ind],
                            color=color[state][index],
                            label=label[state][index])
                    ax.legend()
                    fig.add_subplot(ax)
                    i += 1
        self.figs.append(fig)

    def dwell_example_2nd(self, dwell, block_size=50):
        # only for the last four blocks 
        fig = plt.figure(num='dwell learning', figsize=(12, 8))
        D_left_wellt, D_right_well = [], []
        for nSess in range(len(dwell)):
            D_left_wellt.append(dwell[nSess][block_size][3,:])
            D_right_well.append(dwell[nSess][block_size][4,:])
        fig = plt.figure(num='dwell_2nd', figsize=(8,7))
        data = np.array(D_left_wellt)
        plt.errorbar(range(data.shape[1]), data.mean(axis=0), data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
        data = np.array(D_right_well)
        plt.errorbar(range(data.shape[1]), data.mean(axis=0), data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.legend(['D left wellt', 'D right well'])
        # plt.ylim([-0.08, 1.08])
        # plt.xlim([0.5, 3.5])
        self.figs.append(fig)

        
    def belief_given_O(self, beliefs, observations, o=[], block_size=50,
                       vmin=0, vmax=1, nthTrial=0, prefix='belief'):
        """
        plot the belief when o is observed in each trial,
        trials are sorted by the odor cue and block
        observations: 1:light on (trial start); 2/3 two odor cues; 4/5:first/second rewards;
                      6:light off (trial end); 0: a null observation; 7: add one observation for both fluid well
        States: 0:trial starts, 1:cue L, 2:cue R, 3: left well, 4: right well, 5:L-reward 1, 6:R-reward 1,
                      7:L-reward 2, 8:R-reward 2, 9:trial end

        """
        # four blocks
        assert int(len(beliefs[0]) / block_size) == 4
        # 
        belief_rearrange = [give_me_a_name(bs, os, block_size, o) for bs, os in zip(beliefs, observations)]
        b_left = [b[0] for b in belief_rearrange]
        b_right = [b[1] for b in belief_rearrange]

        color = {0: [0.4, 0, 0],  1: [0.8, 0, 0], 2: [0, 0.4, 0], 3: [0, 0.8, 0]}
        label = {0: 'left start', 1: 'left end', 2: 'right start', 3: 'right end'}

        fig = plt.figure(num=prefix, figsize=(24, 18))
        outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        for n_block in range(4):
            inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                     subplot_spec=outer[n_block], wspace=0.2, hspace=0.2)
            i = 0
            for bs in [b_left, b_right]:
                for index in [0+nthTrial, -1-nthTrial]:
                    ax = plt.Subplot(fig, inner[i])
                    # ax.bar(range(b_left[0][0].size), b[n_block][index], color=color[i], label=label[i])
                    data = np.array([b[n_block][index] for b in bs])
                    average_data = data.mean(axis=0)
                    # if self.task_space == 'poz_rwd':
                    #     if average_data.shape[0] > 60:
                    #         average_data = average_data[0:60, [1, 3, 4, 5]]
                    #     else:
                    #         temp = np.zeros((60-average_data.shape[0] ,4))
                    #         average_data = np.vstack((average_data[:, [1, 3, 4, 5]], temp))
                    # elif self.task_space == 'poz':
                    #     if average_data.shape[0] > 60:
                    #         average_data = average_data[0:60, [1, 3, 4, 5]]
                    #     else:
                    #         temp = np.zeros((60-average_data.shape[0] ,4))
                    #         average_data = np.vstack((average_data[:, [1, 3, 4, 5]], temp))
                        
                        
                    if data.shape[-1] == 7:
                        ax.set_xticklabels(['start', 'cue L',  'cue R',
                                            'rw L1', 'rw R1', 'rw 2', 'end'])
                    else:
                        ax.set_xticklabels(['start', 'cue L',  'cue R', 'well L', 'well R',
                                            'rw L1', 'rw R1', 'rw L2', 'rw R2', 'end'], fontsize=8)
                    ax.set_yticklabels([''])
                    ax.imshow(average_data, aspect='auto', vmin=vmin, vmax=vmax)
                    ax.set_title(label[i])
                    ax.set_xticks(range(data.shape[-1]))
                    plt.axis('off')
                    # plt.colorbar(cax=cax)
                    # ax.legend()
                    fig.add_subplot(ax)
                    i += 1
        self.figs.append(fig)

    def trans_learning(self, trans, observations, block_size=50):
        # plot how the transition probabilities changes with block proceeding
        left_marker, right_marker = 2, 3
        states = {'L': [0, 2, 1, 0], 'R': [2, 0, 0, 1]}
        old, new, third = [], [], []
        for n, (T, O) in enumerate(zip(trans, observations)):
            for LR, marker in enumerate([left_marker, right_marker]):
                trials_all = np.where(np.array([marker in i for i in O]))[0]
                for i in range(1, 4):
                    if i == 2 and LR==1:
                        continue
                    old.append([]);   new.append([]);   third.append([])

                    new_ind = states['L'][i] if LR == 0 else states['R'][i]
                    old_ind = states['L'][i - 1] if LR == 0 else states['R'][i - 1]
                    other_ind = int(3 - old_ind - new_ind)

                    trials = np.concatenate([trials_all[trials_all<50*i][-10:], trials_all[trials_all>=50*i][:20]])
                    for trial in trials:
                        old[-1].append(T[trial][LR][new_ind])
                        new[-1].append(T[trial][LR][old_ind])
                        third[-1].append(T[trial][LR][other_ind])
        fig = plt.figure(figsize=(4, 4))
        for data in [np.array(old), np.array(new), np.array(third)]:
            plt.errorbar(range(data.shape[1]), data.mean(axis=0), data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.legend(['old', 'new', 'the other'])
        self.figs.append(fig)

        return fig
    
        
    def trans_left_2nd(self, trans, block_size=50):
        # plot how the transition probabilities changes with block proceeding
        T_left_first, T_left_last = [], []
        for nSess in range(len(trans)):
            T_left_first.append(trans[nSess][block_size][0,:])
            T_left_last.append(trans[nSess][block_size*2][0,:])
        fig = plt.figure(num='trans_left_2nd', figsize=(8,2.5))
        ind = [[0.95, 1.95, 2.95], [1.05, 2.05, 3.05]]
        i=0
        for data in [np.array(T_left_first), np.array(T_left_last)]:
            plt.errorbar(ind[i], data.mean(axis=0), data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
            i += 1
        plt.legend(['first', 'last'])
        plt.ylim([-0.08, 1.08])
        plt.xlim([0.5, 3.5])
        self.figs.append(fig)

        return fig

    
    def blief_beta_example_left_2nd(self, beta, belief, observations, block_size=50, task_space='poz', suffix=''):
        
        # plot how the transition probabilities changes with block proceeding
        left_marker = 2
        if task_space == 'poz_rwd':
            stay_in_common, stay_in_long, transite_from_common_to_r = [[], []], [[] ,[]], [[], []]
        if task_space == 'poz':
            stay_in_left, stay_in_right, transite_from_left_to_r = [[], []], [[] ,[]], [[], []]
        for nSess in range(len(beta)):
            trials =  np.where(np.array([left_marker in i for i in observations[nSess]]))[0] 
            trials = trials[trials >= block_size]
            trials = trials[trials < block_size*2]
            start, end = trials[0], trials[-1]
            if task_space == 'poz':
                stay_in_left[0].append(belief[nSess][start][16, 3])
                stay_in_left[1].append(belief[nSess][end][16, 3])
                
                stay_in_right[0].append(belief[nSess][start][16, 4])
                stay_in_right[1].append(belief[nSess][end][16, 4])
                
                transite_from_left_to_r[0].append(beta[nSess][start][15,3])
                transite_from_left_to_r[1].append(beta[nSess][end][15,3])
            if task_space == 'poz_rwd':
                stay_in_common[0].append(belief[nSess][start][16, 3])
                stay_in_common[1].append(belief[nSess][end][16, 3])
                
                stay_in_long[0].append(belief[nSess][start][16, 5])
                stay_in_long[1].append(belief[nSess][end][16, 5])
                
                transite_from_common_to_r[0].append(beta[nSess][start][15,3])
                transite_from_common_to_r[1].append(beta[nSess][end][15,3])
            
            
        fig = plt.figure(num='example_left_2nd'+suffix, figsize=(8, 4))
        ind = [[1, 4, 7], [2, 5, 8]]
        for i in range(2):
            if task_space == 'poz':
                data = np.vstack((stay_in_left[i], transite_from_left_to_r[i], stay_in_right[i]))
            if task_space == 'poz_rwd':
                data = np.vstack((stay_in_common[i], transite_from_common_to_r[i], stay_in_long[i]))
            plt.bar(ind[i], data.mean(axis=1), width=0.8)
            plt.errorbar(ind[i], data.mean(axis=1), data.std(axis=1)/np.sqrt(data.shape[1]),
                         fmt='o')
        plt.ylim([0, 1.05])
        plt.xticks([1.5, 4.5, 7.5])
        plt.xlim([0, 9])
            
        self.figs.append(fig)

        return fig

    def state_estimation(self, beta, belief, observation, 
                         delivery = ['r1', 'r3'], omission = ['l1', 'l3'], 
                         block_size=50, state_map='poz_rwd'):
        # one for reward e
        # delivery = ['r1', 'l2', 'r3']
        states = {'l': [0, 2, 1, 0], 'r': [2, 0, 0, 1]}
        delivery_leave_old, delivery_stay_old, delivery_leave_new, delivery_stay_new = [], [], [], []
        omission_leave_old, omission_stay_old, omission_leave_new, omission_stay_new = [], [], [], []

        for nSess, (b_sess, B_sess, O_sess) in enumerate(zip(beta, belief, observation)):
            for d in delivery:
                if state_map == 'poz' and d[1] == '2':
                    continue
                # trials with the cue indicating left or right
                nBlock = int(d[1])
                marker = 2 if d[0] == 'l' else 3
                trials = np.where(np.array([marker in i for i in O_sess]))[0]
                trials = np.concatenate([trials[trials < 50*nBlock][-10:], trials[trials >= 50*nBlock][:20]])
                #
                fst_snd = 15 if nBlock != 3 else 20
                delivery_leave_old.append([]);         delivery_stay_old.append([])
                delivery_leave_new.append([]);         delivery_stay_new.append([])
                if state_map == 'poz_rwd':
                    index = 3 if d[0] == 'l' else 6
                    index = index+6 if int(d[1])==3 else index
                    new_ind, old_ind = states[d[0]][int(d[1])], states[d[0]][int(d[1]) - 1]
                elif state_map=='poz':
                    index = 5 if int(d[1])==3 else 3
                    new_ind = 0 if d[0] == 'l' else 1
                    old_ind = 1 if d[0] == 'l' else 0

                for trial in trials:
                    delivery_stay_old[-1].append(B_sess[trial][fst_snd+1][index + old_ind])
                    delivery_stay_new[-1].append(B_sess[trial][fst_snd+1][index + new_ind])
                    delivery_leave_new[-1].append(b_sess[trial][fst_snd][index + new_ind])
                    delivery_leave_old[-1].append(b_sess[trial][fst_snd][index + old_ind])

            for d in omission:
                # trials with the cue indicating left or right
                nBlock = int(d[1])
                marker = 2 if d[0] == 'l' else 3
                trials = np.where(np.array([marker in i for i in O_sess]))[0]
                trials = np.concatenate([trials[trials < 50*nBlock][-10:], trials[trials >= 50*nBlock][:20]])
                #
                fst_snd = 15 if nBlock != 3 else 20
                omission_leave_old.append([]);         omission_stay_old.append([])
                omission_leave_new.append([]);         omission_stay_new.append([])
                if state_map=='poz_rwd':
                    index = 3 if d[0] == 'l' else 6
                    index = index+6 if int(d[1])==3 else index
                    new_ind, old_ind = states[d[0]][int(d[1])], states[d[0]][int(d[1]) - 1]
                elif state_map=='poz':
                    index = 5 if int(d[1])==3 else 3
                    new_ind = 0 if d[0] == 'l' else 1
                    old_ind = 1 if d[0] == 'l' else 0
                
                for trial in trials:
                    omission_stay_old[-1].append(B_sess[trial][fst_snd+1][index + old_ind])
                    omission_stay_new[-1].append(B_sess[trial][fst_snd+1][index + new_ind])
                    omission_leave_old[-1].append(b_sess[trial][fst_snd][index + old_ind])
                    omission_leave_new[-1].append(b_sess[trial][fst_snd][index + new_ind])

        fig = plt.figure(num='state estimation - delivery', figsize=(4, 4))
        for data in [delivery_leave_old, delivery_stay_old, delivery_leave_new, delivery_stay_new]:
            data = np.array(data)
            plt.errorbar(range(data.shape[1]), data.mean(axis=0), 
                         data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.title('unexpected delivery')
        plt.legend(['leave the old state', 'stay the old state', 'leave the new state', 'stay the new state'])
        plt.ylim([0, 1])
        self.figs.append(fig)

        fig2 = plt.figure(num='state estimation - omission', figsize=(4,4))
        for data in [omission_leave_old, omission_stay_old, omission_leave_new, omission_stay_new]:
            data = np.array(data)
            plt.errorbar(range(data.shape[1]), data.mean(axis=0), 
                         data.std(axis=0)/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.title('reward omission')
        plt.ylim([0, 1])
#        plt.legend(['leave the old state', 'stay the old state', 'leave the new state', 'stay the new state'])
        self.figs.append(fig2)

        return [fig, fig2]
    
    
    def state_estimation_simpfy(self, beta, belief, observation, 
                         delivery = ['r1', 'l2', 'r3'], omission = ['l1', 'l3'], 
                         block_size=50, state_map='poz_rwd'):
        # one for reward e
        # delivery = ['r1', 'l2', 'r3']
        states = {'l': [0, 2, 1, 0], 'r': [2, 0, 0, 1]}
        # reward delivery
        delivery_stay_in_low, delivery_tran_from_high = [], []
        omisssion_stay_in_low, omisssion_tran_to_high = [], []
        for nSess, (b_sess, B_sess, O_sess) in enumerate(zip(beta, belief, observation)):
            delivery_stay_in_low.append([]), delivery_tran_from_high.append([])
            omisssion_stay_in_low.append([]), omisssion_tran_to_high.append([])
            for d in delivery:
                if state_map == 'poz' :
                    continue
                # trials with the cue indicating left or right
                nBlock = int(d[1])
                marker = 2 if d[0] == 'l' else 3
                trials = np.where(np.array([marker in i for i in O_sess]))[0]
                trials_ind = np.logical_and(trials >= block_size*nBlock, trials < block_size*(nBlock+1))
                trials = np.concatenate((trials[trials_ind][:10], trials[trials_ind][-10:]))
                #
                fst_snd = [15, 20] if nBlock==2 else [15] if nBlock != 3 else [20]
                delivery_stay_in_low[-1].append([])
                delivery_tran_from_high[-1].append([])
                if state_map == 'poz_rwd':
                    index = 3 if d[0] == 'l' else 6
                    index = index+6 if int(d[1])==3 else index
                    if nBlock==1:
                        tran_from_ind, stay_in_ind = [[0, 1]], [[2]]
                    elif nBlock==2:
                        tran_from_ind, stay_in_ind = [np.array([0, 1]), np.array([7])], [[2], [0]] 
                    elif nBlock==3:
                        tran_from_ind, stay_in_ind = [[1]], [[0]] 
                    tran_from_ind, stay_in_ind = np.array(tran_from_ind), np.array(stay_in_ind)
                elif state_map=='poz':
                    index = 5 if int(d[1])==3 else 3
                    new_ind = 0 if d[0] == 'l' else 1
                    old_ind = 1 if d[0] == 'l' else 0
                for trial in trials:
                    temp1, temp2 = [], []
                    for i, rwd_ind in enumerate(fst_snd):
                        temp1.append(B_sess[trial][rwd_ind+1][index + stay_in_ind[i]].sum())
                        temp2.append(b_sess[trial][rwd_ind][index + tran_from_ind[i]].sum())
                    delivery_stay_in_low[-1][-1].append(np.mean(temp1))
                    delivery_tran_from_high[-1][-1].append(np.mean(temp2))
#                    delivery_stay_in_low[-1][-1].append(temp1)
#                    delivery_tran_from_high[-1][-1].append(temp2)


#        omisssion_stay_in_low, omisssion_tran_from_high = [], []
#        for nSess, (b_sess, B_sess, O_sess) in enumerate(zip(beta, belief, observation)):
#            omisssion_stay_in_low.append([]), omisssion_tran_from_high.append([])
            # reward omisssion
            for d in omission:
                # trials with the cue indicating left or right
                nBlock = int(d[1])
                marker = 2 if d[0] == 'l' else 3
                trials = np.where(np.array([marker in i for i in O_sess]))[0]
                trials_ind = np.logical_and(trials >= block_size*nBlock, trials < block_size*(nBlock+1))
                trials = np.concatenate((trials[trials_ind][:10], trials[trials_ind][-10:]))
                #
                fst_snd = 15 if nBlock != 3 else 20
                omisssion_stay_in_low[-1].append([])
                omisssion_tran_to_high[-1].append([])
                if state_map=='poz_rwd':
                    index = 3 if d[0] == 'l' else 6
                    index = index+6 if int(d[1])==3 else index
                    if nBlock==1:
                        tran_to_ind, stay_in_ind = [6, 7], [0, 1, 2]
                    elif nBlock==3:
                        tran_to_ind, stay_in_ind = [6], [8]
                elif state_map=='poz':
                    index = 5 if int(d[1])==3 else 3
                    new_ind = 0 if d[0] == 'l' else 1
                    old_ind = 1 if d[0] == 'l' else 0
                
                for trial in trials:
                    omisssion_stay_in_low[-1][-1].append(B_sess[trial][fst_snd+1][index + np.array(stay_in_ind)].sum())
                    omisssion_tran_to_high[-1][-1].append(B_sess[trial][fst_snd+1][index + np.array(tran_to_ind)].sum())

        fig = plt.figure(num='state estimation - delivery', figsize=(6, 5))
        h = []
        for data in [delivery_stay_in_low, delivery_tran_from_high]:
            data = np.array(data).mean(axis=1)
            h.append(plt.errorbar(range(10), data.mean(axis=0)[:10], 
                         data.std(axis=0)[:10]/np.sqrt(data.shape[0]),
                         fmt='o-'))
            plt.errorbar(range(13,23), data.mean(axis=0)[10:], 
                         data.std(axis=0)[10:]/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.title('unexpected delivery')
        plt.legend(h, ['stay in states with low value', 'transit to states with high value'])
        plt.ylim([-.1, 1.1])
        self.figs.append(fig)

        h = []
        fig2 = plt.figure(num='state estimation - omission', figsize=(6, 5))
        for data in [omisssion_stay_in_low, omisssion_tran_to_high]:
            data = np.array(data).mean(axis=1)
            h.append(plt.errorbar(range(10), data.mean(axis=0)[:10], 
                         data.std(axis=0)[:10]/np.sqrt(data.shape[0]),
                         fmt='o-'))
            plt.errorbar(range(13,23), data.mean(axis=0)[10:], 
                         data.std(axis=0)[10:]/np.sqrt(data.shape[0]),
                         fmt='o-')
        plt.title('reward omission')
        plt.ylim([-.1, 1.1])
        plt.legend(h, ['stay in states with low value', 'transit to states with immediate reward'])
        self.figs.append(fig2)

        return [fig, fig2]