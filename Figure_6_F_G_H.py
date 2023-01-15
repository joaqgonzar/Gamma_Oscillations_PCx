#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:05:23 2021

@author: joaquin
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.io 
import pandas as pd
import os
import h5py
import mat73
from scipy import signal
from scipy import stats
from sklearn.decomposition import FastICA, PCA
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr


def spike_rate(spike_times):
    recording_time = np.max(spike_times[0])+1
    srate_spikes = 30000
    
    neuron_number = len(spike_times)
    multi_units_1 = np.zeros([neuron_number-1,int(recording_time*srate_spikes)],dtype = np.bool_)
        
        
    for x in range(neuron_number-1):
        spike_range = np.where(spike_times[int(x+1)]<recording_time)[0]
        spikes = spike_times[int(x+1)][0:spike_range.shape[0]]
        s_unit = np.rint(spikes[0]*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1
           
        
    srate_resample = 2000
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)
        
    del multi_units_1,multi_units_re
        
    kernel = signal.gaussian(int(0.1*srate_resample),20)
        
    conv_neurons_session = []
    for x in range(sum_units_1.shape[0]):   
        conv = signal.convolve(np.squeeze(sum_units_1[x,:]), kernel, mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_session.append(conv)
        
    conv_neurons_session = np.array(conv_neurons_session)
    
    return(conv_neurons_session,sum_units_1)

#%%

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos'

os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx')

exp_data = pd.read_csv('ExperimentCatalog_TeLC-PCX.txt', sep=" ", header=None)

#%%

names = ['150220','150221','150223','150312','150327','150403','150406','150812']
#names = ['150221','150312','150327','150403','150406','150812']

ic_one_animals_TeLC = []
winners_losers_animals_TeLC = []
all_animals_TeLC = []

loading = np.array(exp_data[3][7:15])

for index,name in enumerate(names):
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
        num_odors = 6
        
    elif  loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        num_odors = 4
        
    print(name)
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')   
    lfp = np.load('PFx_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
        
    srate = 2000
        
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name)   
    
    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
    inh_start = inh_start[inh_start<lfp.shape[1]]
        
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']
    recording_time = np.max(spike_times[0])+1
       
    [exc_neurons_session,units_exc] = spike_rate(spike_times)
    exc_neurons_session = np.array(exc_neurons_session)[:,0:lfp.shape[1]]
    units_exc = np.array(units_exc)[:,0:lfp.shape[1]]
        
    avg_rates = np.sum(units_exc,axis = 1)/recording_time
    putative_pyramidals = avg_rates<100
        #conv_spikes_exc = (conv_spikes_exc-np.mean(conv_spikes_exc,axis = 1)[:,np.newaxis])/np.std(conv_spikes_exc,axis = 1)[:,np.newaxis]
        
    conv_spikes_norm = stats.zscore(exc_neurons_session[putative_pyramidals,:],axis = 1)
        
    # get all inhalations with odor
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    # get odor identities for all inhalations
    odorants = []
    for index_odors, x in enumerate(odor_onset_times):
        odor_repeats = x.shape[1]
        odorants.append(np.repeat(index_odors,odor_repeats))
        
    odorants = np.concatenate(odorants,axis = 0)
    
    inh_odor = []
    odorants_repeat = []
    
    for index_odor,x in enumerate(odor_onset_srate):
        
        if x+2000 < conv_spikes_norm.shape[1]:
            index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
            
            inh_odor.append(inh_start[index_smells])
            if len(inh_start[index_smells]) > 0:
                num_breaths = len(inh_start[index_smells])
                odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    # get spiking activity for all inhalations
    
    smells_trig = []
    for x in inh_smell:
        if conv_spikes_norm[:,int(x):int(x+2000)].shape[1] == 2000:
            smells_trig.append(conv_spikes_norm[:,int(x):int(x+2000)])
            
    
    smells_trig = np.array(smells_trig)
    
    # do ICA analysis
    
    ic_one_weights_TeLC = []
    
    winners_time = []
    losers_time = []
    all_time = []
    
    # define threshold for winners and losers
    
    #winners_threshold = 0.16472863212658467
    #losers_threshold = 0.03670100857202258
    
    winners_threshold = 0.0887137180248916
    losers_threshold = 0.013753072267353993
     
    for odor in np.arange(0,num_odors):
        
        # get individual odors
        
        odor_index = odorants == odor
        spikes_odor = smells_trig[odor_index,:,:]
        
        test_data = np.concatenate(spikes_odor,axis =1)
        test_data = test_data[~np.isnan(np.sum(test_data,axis = 1)),:]
        
        # run fast ica
        
        ica = FastICA(n_components=1,random_state=0, whiten='unit-variance')
        S_ica_ = ica.fit(test_data.T).transform(test_data.T)
        S_ica_ = S_ica_/np.std(S_ica_)
        
        # get ica weights
        
        ica_weights = ica.fit(test_data.T).components_.T
        
        # get winners/losers
        
        sign_winners = np.sign(ica_weights[np.argmax(np.abs(ica_weights)),0])
        winners = ica_weights*sign_winners > winners_threshold
        losers = ica_weights*sign_winners <= losers_threshold
        
        # get all ics with the same sign
        
        ica_weights = ica_weights*sign_winners
        
        # get time-courses
        
        time_course_winners = test_data[np.squeeze(winners),:]
        time_course_losers = test_data[np.squeeze(losers),:]
        
        # average winners and losers
        
        mean_winners = np.mean(spikes_odor[:,np.squeeze(winners),:],axis = 0)
        mean_losers = np.mean(spikes_odor[:,np.squeeze(losers),:],axis = 0)
        mean_all = np.mean(spikes_odor[:,:,:],axis = 0)

        
        # save results from odor
        winners_time.append(mean_winners)    
        losers_time.append(mean_losers)    
        ic_one_weights_TeLC.append(ica_weights)
        all_time.append(mean_all)
        

    # get all winners and losers
    
    # save results
    winners_losers_animals_TeLC.append([np.concatenate(winners_time,axis = 0),np.concatenate(losers_time,axis = 0)])
    all_animals_TeLC.append(np.concatenate(all_time,axis = 0))
    ic_one_animals_TeLC.append(ic_one_weights_TeLC)

            
# check contralateral hemishpere

names = ['150221','150312','150327','150403','150406','150812']

ic_one_animals_TeLC_contra = []
winners_losers_animals_TeLC_contra = []
all_animals_TeLC_contra = [] 
loading = np.array(exp_data[3][1:7])

for index,name in enumerate(names):
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
        num_odors = 6
        
    elif loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        num_odors = 4
        
    print(name)
    
    print(num_odors)
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')   
    lfp = np.load('PFx_contra_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
        
    srate = 2000
        
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name)   
    
    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
    inh_start = inh_start[inh_start<lfp.shape[1]]
        
    spike_times = mat73.loadmat(name+'_bank2_st.mat')['SpikeTimes']['tsec']
    recording_time = np.max(spike_times[0])+1
       
    [exc_neurons_session,units_exc] = spike_rate(spike_times)
    exc_neurons_session = np.array(exc_neurons_session)[:,0:lfp.shape[1]]
    units_exc = np.array(units_exc)[:,0:lfp.shape[1]]
        
    avg_rates = np.sum(units_exc,axis = 1)/recording_time
    putative_pyramidals = avg_rates<100
        #conv_spikes_exc = (conv_spikes_exc-np.mean(conv_spikes_exc,axis = 1)[:,np.newaxis])/np.std(conv_spikes_exc,axis = 1)[:,np.newaxis]
        
    conv_spikes_norm = stats.zscore(exc_neurons_session[putative_pyramidals,:],axis = 1)
        
    # get all inhalations with odor
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    # get odor identities for all inhalations
    odorants = []
    for index_odors, x in enumerate(odor_onset_times):
        odor_repeats = x.shape[1]
        odorants.append(np.repeat(index_odors,odor_repeats))
        
    odorants = np.concatenate(odorants,axis = 0)
    
    inh_odor = []
    odorants_repeat = []
    
    for index_odor,x in enumerate(odor_onset_srate):
        
        if x+2000 < conv_spikes_norm.shape[1]:
            index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
            inh_odor.append(inh_start[index_smells])
            if len(inh_start[index_smells]) > 0:
                num_breaths = len(inh_start[index_smells])
                odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    # get spiking activity for all inhalations
    
    smells_trig = []
    for x in inh_smell:
        if conv_spikes_norm[:,int(x):int(x+2000)].shape[1] == 2000:
            smells_trig.append(conv_spikes_norm[:,int(x):int(x+2000)])
            
    
    smells_trig = np.array(smells_trig)
    
    # do ICA analysis
    
    ic_one_weights_TeLC_contra = []
    
    winners_time = []
    losers_time = []
    
    # define threshold for winners and losers
    
    #winners_threshold = 0.1215556731865974
    #losers_threshold = 0.0013207588045774166
    
    winners_threshold = 0.0887137180248916
    losers_threshold = 0.013753072267353993
    
    for odor in np.arange(0,num_odors):
        
        # get individual odors
        
        odor_index = odorants == odor
        spikes_odor = smells_trig[odor_index,:,:]
        
        test_data = np.concatenate(spikes_odor,axis =1)
        test_data = test_data[~np.isnan(np.sum(test_data,axis = 1)),:]
        
        # run fast ica
        
        ica = FastICA(n_components=1,random_state=0, whiten='unit-variance')
        S_ica_ = ica.fit(test_data.T).transform(test_data.T)
        S_ica_ = S_ica_/np.std(S_ica_)
        
        # get ica weights
        
        ica_weights = ica.fit(test_data.T).components_.T
        
        # get winners/losers
        
        sign_winners = np.sign(ica_weights[np.argmax(np.abs(ica_weights)),0])
        winners = ica_weights*sign_winners > winners_threshold
        losers = ica_weights*sign_winners <= losers_threshold
        
        # get all ics with the same sign
        
        ica_weights = ica_weights*sign_winners
        
        # get time-courses
        
        time_course_winners = test_data[np.squeeze(winners),:]
        time_course_losers = test_data[np.squeeze(losers),:]
        
        # average winners and losers
        
        mean_winners = np.mean(spikes_odor[:,np.squeeze(winners),:],axis = 0)
        mean_losers = np.mean(spikes_odor[:,np.squeeze(losers),:],axis = 0)
        mean_all =  np.mean(spikes_odor[:,:,:],axis = 0)
        
        # save results from odor
        winners_time.append(mean_winners)    
        losers_time.append(mean_losers)    
        all_time.append(mean_all)
        ic_one_weights_TeLC_contra.append(ica_weights)
        

    # get all winners and losers
    
    # save results
    winners_losers_animals_TeLC_contra.append([np.concatenate(winners_time,axis = 0),np.concatenate(losers_time,axis = 0)])
    ic_one_animals_TeLC_contra.append(ic_one_weights_TeLC_contra)
    all_animals_TeLC_contra.append(np.concatenate(all_time,axis = 0))

#%%

os.chdir(directory)

np.savez('Figure_6_2.npz', winners_losers_animals_TeLC = winners_losers_animals_TeLC, all_animals_TeLC = all_animals_TeLC, ic_one_animals_TeLC = ic_one_animals_TeLC, winners_losers_animals_TeLC_contra = winners_losers_animals_TeLC_contra, ic_one_animals_TeLC_contra = ic_one_animals_TeLC_contra, all_animals_TeLC_contra = all_animals_TeLC_contra)

#%%

os.chdir(directory)

winners_losers_animals_TeLC = np.load('Figure_6_2.npz',allow_pickle=(True))['winners_losers_animals_TeLC']
all_animals_TeLC = np.load('Figure_6_2.npz',allow_pickle=(True))['all_animals_TeLC']
ic_one_animals_TeLC = np.load('Figure_6_2.npz',allow_pickle=(True))['ic_one_animals_TeLC']
winners_losers_animals_TeLC_contra = np.load('Figure_6_2.npz',allow_pickle=(True))['winners_losers_animals_TeLC_contra']
ic_one_animals_TeLC_contra = np.load('Figure_6_2.npz',allow_pickle=(True))['ic_one_animals_TeLC_contra']
all_animals_TeLC_contra = np.load('Figure_6_2.npz',allow_pickle=(True))['all_animals_TeLC_contra']

#%%
         
winners_threshold_TeLC = []
losers_threshold_TeLC = []


for x in range(8):
    winners_threshold_TeLC.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals_TeLC[x])),0.95))
    losers_threshold_TeLC.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals_TeLC[x])),0.5))


winners_threshold_TeLC_contra = []
losers_threshold_TeLC_contra = []

for x in range(6):
    winners_threshold_TeLC_contra.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals_TeLC_contra[x])),0.95))
    losers_threshold_TeLC_contra.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals_TeLC_contra[x])),0.5))



#%% get number of winners and losers 

losers_animals_TeLC = []
for animals in range(len(ic_one_animals_TeLC)):
    
    losers = []
    for x in ic_one_animals_TeLC[animals]:
        losers.append(np.sum(x<=0.013753072267353993)/x.shape[0])
        
        
    losers_animals_TeLC.append(np.mean(losers))    
losers_animals_TeLC = np.array(losers_animals_TeLC)    
        
losers_animals_TeLC_contra = []
for animals in range(len(ic_one_animals_TeLC_contra)):
    
    losers = []
    for x in ic_one_animals_TeLC_contra[animals]:
        losers.append(np.sum(x<=0.013753072267353993)/x.shape[0])
        
    losers_animals_TeLC_contra.append(np.mean(losers))            
losers_animals_TeLC_contra = np.array(losers_animals_TeLC_contra)    


ic_odor_TeLC = []
sk_TeLC = []

for x in range(8):
    ic_odor_TeLC.append(np.concatenate(ic_one_animals_TeLC[x]))
    sk_TeLC.append(stats.kurtosis(np.concatenate(ic_one_animals_TeLC[x])))

ic_odor_TeLC_contra = []
sk_TeLC_contra = []

for x in range(6):
    ic_odor_TeLC_contra.append(np.concatenate(ic_one_animals_TeLC_contra[x]))
    sk_TeLC_contra.append(stats.kurtosis(np.concatenate(ic_one_animals_TeLC_contra[x])))
    
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from matplotlib import gridspec
    
plt.figure(dpi = 300, figsize = (20,3))
gs = gridspec.GridSpec(1, 6, width_ratios= (2,1,1,2,2,1),wspace = 0.5)
    
plt.subplot(gs[0])
data_dist = np.squeeze(np.concatenate(ic_odor_TeLC))
mean_TeLC = np.mean(data_dist)
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'TeLC Ipsi', color = 'tab:red')

data_dist = np.squeeze(np.concatenate(ic_odor_TeLC_contra))
mean_TeLC_contra = np.mean(data_dist)
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'TeLC Contra')
plt.legend()

plt.xlim([-0.1,0.2])
#plt.ylim([0,1.5])
#plt.yscale('symlog')
plt.xlabel('1st IC Weights')
plt.vlines(mean_TeLC, 0,28,color = 'tab:red', linestyles='dashed')
plt.vlines(mean_TeLC_contra, 0,28,color = 'tab:blue', linestyles='dashed')


plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


common_animals = [1,3,4,5,6,7]

s_loser_num,p_loser_num = stats.ttest_rel(losers_animals_TeLC[common_animals], losers_animals_TeLC_contra, alternative = 'less')


loser_TeLC_percentage  = losers_animals_TeLC[common_animals]*100
loser_TeLC_contra_percentage  = losers_animals_TeLC_contra*100



plt.subplot(gs[1])
#plt.hlines(0, 0.8, 2.2, color = 'grey', linestyle = 'dashed')

plt.boxplot([loser_TeLC_percentage,loser_TeLC_contra_percentage],widths = 0.2, showfliers=False)

for x in range(6):
    plt.plot([1.2,1.8],[loser_TeLC_percentage[x],loser_TeLC_contra_percentage[x]], color = 'grey')
    
plt.ylabel('% of losers')
plt.xticks(ticks = [1,2], labels = ['TeLC ipsi','TeLC contra'])

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

winners_losers_animals_TeLC = np.array(winners_losers_animals_TeLC)

winners = winners_losers_animals_TeLC[:,0]
losers = winners_losers_animals_TeLC[:,1]

winners_TeLC = []
losers_TeLC = []

for x in range(6):
    winners_TeLC.append(np.mean(winners[x],axis = 0))
    losers_TeLC.append(np.mean(losers[x],axis = 0))
    
winners_TeLC = np.array(winners_TeLC)
losers_TeLC = np.array(losers_TeLC)

winners_losers_animals_TeLC_contra = np.array(winners_losers_animals_TeLC_contra)

winners = winners_losers_animals_TeLC_contra[:,0]
losers = winners_losers_animals_TeLC_contra[:,1]

winners_TeLC_contra = []
losers_TeLC_contra = []

for x in range(6):
    winners_TeLC_contra.append(np.mean(winners[x],axis = 0))
    losers_TeLC_contra.append(np.mean(losers[x],axis = 0))

winners_TeLC_contra = np.array(winners_TeLC_contra)
losers_TeLC_contra = np.array(losers_TeLC_contra)

mean_winners_TeLC = np.mean(winners_TeLC,axis = 0)
error_winners_TeLC = np.std(winners_TeLC,axis = 0)/np.sqrt(winners_TeLC.shape[0])
mean_winners_TeLC_contra = np.mean(winners_TeLC_contra,axis = 0)
error_winners_TeLC_contra = np.std(winners_TeLC_contra,axis = 0)/np.sqrt(winners_TeLC_contra.shape[0])

mean_losers_TeLC = np.nanmean(losers_TeLC,axis = 0)
error_losers_TeLC = np.nanstd(losers_TeLC,axis = 0)/np.sqrt(losers_TeLC.shape[0])
mean_losers_TeLC_contra = np.mean(losers_TeLC_contra,axis = 0)
error_losers_TeLC_contra = np.std(losers_TeLC_contra,axis = 0)/np.sqrt(losers_TeLC_contra.shape[0])

plt.subplot(gs[2])

telc = np.mean(losers_TeLC[:,200:1000],axis = 1)
telc_contra = np.mean(losers_TeLC_contra[:,200:1000],axis = 1)

plt.boxplot([telc,telc_contra],widths = 0.2, showfliers=False)

s_loser_act,p_loser_act = stats.ttest_rel(telc,telc_contra, alternative = 'greater')

for x in range(6):
    plt.plot([1.2,1.8],[telc[x],telc_contra[x]], color = 'grey')
    
plt.ylabel('% of losers')
plt.xticks(ticks = [1,2], labels = ['TeLC ipsi','TeLC contra'])
plt.ylabel('<Losers Z-score>')

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


ic_vector_corr_TeLC = []
ic_vector_p_TeLC = []

for animals in [0,1,2,3,4,7]:
    vector_corr_matrix = np.zeros([6,6])
    vector_p_matrix = np.zeros([6,6])
    for x in np.arange(0,5):
        for y in np.arange(x+1,6):
            vector_corr_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals_TeLC[animals][x]),np.squeeze(ic_one_animals_TeLC[animals][y]))[0]
            vector_p_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals_TeLC[animals][x]),np.squeeze(ic_one_animals_TeLC[animals][y]))[1]
                    
    vector_corr_matrix = vector_corr_matrix+vector_corr_matrix.T
    vector_corr_matrix = vector_corr_matrix + np.identity(6)
    
    vector_p_matrix = vector_p_matrix+vector_p_matrix.T
    
    ic_vector_corr_TeLC.append(vector_corr_matrix)
    ic_vector_p_TeLC.append(vector_p_matrix)
     
    
ic_vector_corr_TeLC_contra = []
ic_vector_p_TeLC_contra = []

for animals in [0,1,2,5]:
    vector_corr_matrix = np.zeros([6,6])
    vector_p_matrix = np.zeros([6,6])
    for x in np.arange(0,5):
        for y in np.arange(x+1,6):
            vector_corr_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals_TeLC_contra[animals][x]),np.squeeze(ic_one_animals_TeLC_contra[animals][y]))[0]
            vector_p_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals_TeLC_contra[animals][x]),np.squeeze(ic_one_animals_TeLC_contra[animals][y]))[1]
                    
    vector_corr_matrix = vector_corr_matrix+vector_corr_matrix.T
    vector_corr_matrix = vector_corr_matrix + np.identity(6)
    
    vector_p_matrix = vector_p_matrix+vector_p_matrix.T
    
    ic_vector_corr_TeLC_contra.append(vector_corr_matrix)
    ic_vector_p_TeLC_contra.append(vector_p_matrix)
    
# plot IC correlation matrices

plt.subplot(gs[3])
population_correlation = np.nanmean(ic_vector_corr_TeLC,axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 1)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'], rotation = 90)
plt.yticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
#plt.title('TeLC Ipsi')

plt.subplot(gs[4])
population_correlation = np.mean(ic_vector_corr_TeLC_contra,axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 1)
plt.colorbar()
plt.yticks(ticks = np.arange(0,6),labels = [])
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'], rotation = 90)

plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
#plt.title('TeLC Contra')

#plt.savefig('Correlation_matrix_TeLC.pdf')

common_animals = [1,3,4,5,6,7]

mean_corr_telc = []

for animals in range(8):
    
    num_odors = len(ic_one_animals_TeLC[animals])
    vector_corr_telc = []
    for x in np.arange(0,num_odors-1):
        for y in np.arange(x+1,num_odors):
            vector_corr_telc.append(pearsonr(np.squeeze(ic_one_animals_TeLC[animals][x]),np.squeeze(ic_one_animals_TeLC[animals][y]))[0])
            #vector_corr_telc.append(np.inner(np.squeeze(ic_one_animals_TeLC[animals][x]),np.squeeze(ic_one_animals_TeLC[animals][y])))
            
    mean_corr_telc.append(np.mean(np.arctanh(vector_corr_telc)))
    

mean_corr_telc = np.array(mean_corr_telc)

mean_corr_telc_contra = []    

for animals in range(6):
    
    num_odors = len(ic_one_animals_TeLC_contra[animals])
    vector_corr_telc_contra = []
    
    for x in np.arange(0,num_odors-1):
        for y in np.arange(x+1,num_odors):
            vector_corr_telc_contra.append(pearsonr(np.squeeze(ic_one_animals_TeLC_contra[animals][x]),np.squeeze(ic_one_animals_TeLC_contra[animals][y]))[0])
            #vector_corr_telc_contra.append(np.inner(np.squeeze(ic_one_animals_TeLC_contra[animals][x]),np.squeeze(ic_one_animals_TeLC_contra[animals][y])))
                    
    mean_corr_telc_contra.append(np.mean(np.arctanh(vector_corr_telc_contra)))
mean_corr_telc_contra = np.array(mean_corr_telc_contra)


s_corr,p_corr = stats.ttest_rel(mean_corr_telc[common_animals],mean_corr_telc_contra, alternative = 'greater')

plt.subplot(gs[5])
#plt.hlines(0, 0.8, 2.2, color = 'grey', linestyle = 'dashed')

plt.boxplot([mean_corr_telc[common_animals],mean_corr_telc_contra],widths = 0.2, showfliers=False)

for x in range(6):
    plt.plot([1.2,1.8],[mean_corr_telc[common_animals][x],mean_corr_telc_contra[x]], color = 'grey')
    
plt.ylabel('<IC Correlation>')
plt.xticks(ticks = [1,2], labels = ['TeLC ipsi','TeLC contra'])

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

#plt.savefig('TeLC_ICA.pdf')

#%%

ic_odor_TeLC = []
sk_TeLC = []

for x in range(8):
    ic_odor_TeLC.append(np.concatenate(ic_one_animals_TeLC[x]))
    sk_TeLC.append(stats.skew(np.concatenate(ic_one_animals_TeLC[x])))

sk_TeLC = np.squeeze(sk_TeLC)[common_animals]

ic_odor_TeLC_contra = []
sk_TeLC_contra = []

for x in range(6):
    ic_odor_TeLC_contra.append(np.concatenate(ic_one_animals_TeLC_contra[x]))
    sk_TeLC_contra.append(stats.skew(np.concatenate(ic_one_animals_TeLC_contra[x])))
    
sk_TeLC_contra = np.squeeze(sk_TeLC_contra)  
    
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize=(2,4))


plt.boxplot([sk_TeLC,sk_TeLC_contra],widths = 0.2, showfliers=False)

for x in range(6):
    plt.plot([1.2,1.8],[sk_TeLC[x],sk_TeLC_contra[x]], color = 'grey')
    
plt.ylabel('<IC distribution skewness>')
plt.xticks(ticks = [1,2], labels = ['TeLC ipsi','TeLC contra'])

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s_sk,p_sk = stats.ttest_rel(sk_TeLC,sk_TeLC_contra, alternative = 'less')
df_sk = sk_TeLC.shape[0]-1
#plt.savefig('SK_Telc.pdf')

#%%

winners_losers_animals_TeLC = np.array(winners_losers_animals_TeLC)

winners = winners_losers_animals_TeLC[:,0]
losers = winners_losers_animals_TeLC[:,1]
winners_TeLC = np.concatenate(winners)
losers_TeLC = np.concatenate(losers)

winners_TeLC = []
losers_TeLC = []

for x in range(6):
    winners_TeLC.append(np.mean(winners[x],axis = 0))
    losers_TeLC.append(np.mean(losers[x],axis = 0))
    
winners_TeLC = np.array(winners_TeLC)
losers_TeLC = np.array(losers_TeLC)

winners_losers_animals_TeLC_contra = np.array(winners_losers_animals_TeLC_contra)

winners = winners_losers_animals_TeLC_contra[:,0]
losers = winners_losers_animals_TeLC_contra[:,1]
winners_TeLC_contra = np.concatenate(winners)
losers_TeLC_contra = np.concatenate(losers)

winners_TeLC_contra = []
losers_TeLC_contra = []

for x in range(6):
    winners_TeLC_contra.append(np.mean(winners[x],axis = 0))
    losers_TeLC_contra.append(np.mean(losers[x],axis = 0))

winners_TeLC_contra = np.array(winners_TeLC_contra)
losers_TeLC_contra = np.array(losers_TeLC_contra)

mean_winners_TeLC = np.mean(winners_TeLC,axis = 0)
error_winners_TeLC = np.std(winners_TeLC,axis = 0)/np.sqrt(winners_TeLC.shape[0])
mean_winners_TeLC_contra = np.mean(winners_TeLC_contra,axis = 0)
error_winners_TeLC_contra = np.std(winners_TeLC_contra,axis = 0)/np.sqrt(winners_TeLC_contra.shape[0])

mean_losers_TeLC = np.nanmean(losers_TeLC,axis = 0)
error_losers_TeLC = np.nanstd(losers_TeLC,axis = 0)/np.sqrt(losers_TeLC.shape[0])
mean_losers_TeLC_contra = np.mean(losers_TeLC_contra,axis = 0)
error_losers_TeLC_contra = np.std(losers_TeLC_contra,axis = 0)/np.sqrt(losers_TeLC_contra.shape[0])



plt.figure(dpi = 300, figsize = (3,5))

plt.subplot(212)
#plt.subplot(211)
plt.plot(np.linspace(0,1000,2000),mean_losers_TeLC, label = 'losers TeLC')
plt.fill_between(np.linspace(0,1000,2000), mean_losers_TeLC-error_losers_TeLC, mean_losers_TeLC+error_losers_TeLC,alpha = 0.2)
#plt.subplot(212)
plt.plot(np.linspace(0,1000,2000),mean_losers_TeLC_contra, label = 'losers TeLC contra')
plt.fill_between(np.linspace(0,1000,2000), mean_losers_TeLC_contra-error_losers_TeLC_contra, mean_losers_TeLC_contra+error_losers_TeLC_contra,alpha = 0.2)
#plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel(' Activity (z-score)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel 
plt.xlim([0,1000])
plt.ylim([-0.15,0.4])
plt.hlines(0,0,1000, color = 'grey',linestyles = 'dashed')
plt.title('Losers')

plt.subplot(211)

plt.plot(np.linspace(0,1000,2000),mean_winners_TeLC, label = 'TeLC')
plt.fill_between(np.linspace(0,1000,2000), mean_winners_TeLC-error_winners_TeLC, mean_winners_TeLC+error_winners_TeLC,alpha = 0.2)
#plt.subplot(212)
plt.plot(np.linspace(0,1000,2000),mean_winners_TeLC_contra, label = 'TeLC contra')
plt.fill_between(np.linspace(0,1000,2000), mean_winners_TeLC_contra-error_winners_TeLC_contra, mean_winners_TeLC_contra+error_winners_TeLC_contra,alpha = 0.2)
plt.legend()
#plt.xlabel('Time (ms)')
plt.ylabel('Activity (z-score)')
plt.title('Winners')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel 
plt.xlim([0,1000])
plt.ylim([-0.15,2.2])
plt.hlines(0,0,1000, color = 'grey',linestyles = 'dashed')

plt.tight_layout()

plt.ylabel(' Activity (z-score)')

#%%

winners_losers_animals_TeLC = np.array(winners_losers_animals_TeLC)

winners = winners_losers_animals_TeLC[:,0]
losers = winners_losers_animals_TeLC[:,1]
winners_TeLC = np.concatenate(winners)
losers_TeLC = np.concatenate(losers)

winners_losers_animals_TeLC_contra = np.array(winners_losers_animals_TeLC_contra)

winners = winners_losers_animals_TeLC_contra[:,0]
losers = winners_losers_animals_TeLC_contra[:,1]
winners_TeLC_contra = np.concatenate(winners)
losers_TeLC_contra = np.concatenate(losers)


mean_winners_TeLC = np.mean(winners_TeLC,axis = 0)
error_winners_TeLC = np.std(winners_TeLC,axis = 0)/np.sqrt(winners_TeLC.shape[0])
mean_winners_TeLC_contra = np.mean(winners_TeLC_contra,axis = 0)
error_winners_TeLC_contra = np.std(winners_TeLC_contra,axis = 0)/np.sqrt(winners_TeLC_contra.shape[0])

mean_losers_TeLC = np.nanmean(losers_TeLC,axis = 0)
error_losers_TeLC = np.nanstd(losers_TeLC,axis = 0)/np.sqrt(losers_TeLC.shape[0])
mean_losers_TeLC_contra = np.mean(losers_TeLC_contra,axis = 0)
error_losers_TeLC_contra = np.std(losers_TeLC_contra,axis = 0)/np.sqrt(losers_TeLC_contra.shape[0])



plt.figure(dpi = 300, figsize = (3,5))

plt.subplot(212)
#plt.subplot(211)
plt.plot(np.linspace(0,1000,2000),mean_losers_TeLC, label = 'losers TeLC')
plt.fill_between(np.linspace(0,1000,2000), mean_losers_TeLC-error_losers_TeLC, mean_losers_TeLC+error_losers_TeLC,alpha = 0.2)
#plt.subplot(212)
plt.plot(np.linspace(0,1000,2000),mean_losers_TeLC_contra, label = 'losers TeLC contra')
plt.fill_between(np.linspace(0,1000,2000), mean_losers_TeLC_contra-error_losers_TeLC_contra, mean_losers_TeLC_contra+error_losers_TeLC_contra,alpha = 0.2)
#plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel(' Activity (z-score)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel 
plt.xlim([0,1000])
plt.ylim([-0.15,0.4])
plt.hlines(0,0,1000, color = 'grey',linestyles = 'dashed')
plt.title('Losers')

plt.subplot(211)

plt.plot(np.linspace(0,1000,2000),mean_winners_TeLC, label = 'TeLC')
plt.fill_between(np.linspace(0,1000,2000), mean_winners_TeLC-error_winners_TeLC, mean_winners_TeLC+error_winners_TeLC,alpha = 0.2)
#plt.subplot(212)
plt.plot(np.linspace(0,1000,2000),mean_winners_TeLC_contra, label = 'TeLC contra')
plt.fill_between(np.linspace(0,1000,2000), mean_winners_TeLC_contra-error_winners_TeLC_contra, mean_winners_TeLC_contra+error_winners_TeLC_contra,alpha = 0.2)
plt.legend()
plt.xlabel('Time from inhalation start (ms)')
plt.ylabel('Activity (z-score)')
plt.title('Winners')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel 
plt.xlim([0,1000])
plt.ylim([-0.15,2.2])
plt.hlines(0,0,1000, color = 'grey',linestyles = 'dashed')

plt.tight_layout()

plt.ylabel(' Activity (z-score)')

#%%

all_telc = np.array(all_animals_TeLC)[common_animals]
all_telc_contra = np.array(all_animals_TeLC_contra)

all_telc = np.concatenate(all_telc)
all_telc_contra = np.concatenate(all_telc_contra)

mean_all_telc = np.mean(all_telc,axis = 0)
error_all_telc = np.std(all_telc,axis = 0)/np.sqrt(all_telc.shape[0])

mean_all_telc_contra = np.mean(all_telc_contra,axis = 0)
error_all_telc_contra = np.std(all_telc_contra,axis = 0)/np.sqrt(all_telc_contra.shape[0])



plt.plot(np.linspace(0,1000,2000),mean_all_telc, label = 'TeLC')
plt.fill_between(np.linspace(0,1000,2000), mean_all_telc-error_all_telc, mean_all_telc+error_all_telc,alpha = 0.2)
#plt.subplot(212)
plt.plot(np.linspace(0,1000,2000),mean_all_telc_contra, label = 'TeLC contra')
plt.fill_between(np.linspace(0,1000,2000), mean_all_telc_contra-error_all_telc_contra, mean_all_telc_contra+error_all_telc_contra,alpha = 0.2)
plt.legend()
