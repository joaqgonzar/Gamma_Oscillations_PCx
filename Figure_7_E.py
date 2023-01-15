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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#%%

names = ['150220','150221','150223','150312','150327','150403','150406','150812']

os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/TeLC_PFx')

exp_data = pd.read_csv('ExperimentCatalog_TeLC-PCX.txt', sep=" ", header=None)

#%%
accuracies_animals_TeLC = [] 

names = ['150220','150221','150223','150312','150327','150403','150406','150812']

loading = np.array(exp_data[3][7:15])

for index,name in enumerate(names):
    
    
    print(name)
    
    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    
    lfp = np.load('PFx_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
    srate = 2000
    
    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name)
    
    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
    
    conc_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    #smell_series = list(np.array([11,7,8,6,12,10])-1) 
    
    if loading[index] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
    elif loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)  

    
    inh_start = inh_start[inh_start<lfp.shape[1]]
    
    
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']

    #
    recording_time = np.max(spike_times[0])+1
    srate_spikes = 30000
    
    neuron_number = len(spike_times)
    multi_units_1 = np.zeros([neuron_number,int(recording_time*srate_spikes)],dtype = np.bool_)
    
    
    for x in range(neuron_number):
        spike_range = np.where(spike_times[int(x)]<recording_time)[0]
        spikes = spike_times[int(x)][0:spike_range.shape[0]]
        s_unit = np.rint(spikes[0]*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1
        #indice = indice+1
    
    srate_resample = 2000
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)
    
    del multi_units_1,multi_units_re
    
    avg_rates = np.sum(sum_units_1,axis = 1)/recording_time
    putative_pyramidals = avg_rates<100
    
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    odorants = []
    for index_odors, x in enumerate(odor_onset_times):
        odor_repeats = x.shape[1]
        odorants.append(np.repeat(index_odors,odor_repeats))
        
    odorants = np.concatenate(odorants,axis = 0)
    
    inh_odor = []
    odorants_repeat = []
    
    for index_odor,x in enumerate(odor_onset_srate):
        index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
        inh_odor.append(inh_start[index_smells])
        if len(inh_start[index_smells]) > 0:
            num_breaths = len(inh_start[index_smells])
            odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    smells_trig = []
    gamma_envelope_odor = [] 
    gamma_phase_odor = []
    resp_odor = []
    
    for index_odor, x in enumerate(inh_smell):
        smells_trig.append(sum_units_1[:,int(x):int(x+1400)])
        
    
    window = 200
    
    accuracies_bin = []
    mean_gamma_bin = []   
    mean_resp_bin = []
    
    for time_bin in np.arange(100,1225,75):
        
        spike_averages = []
        
        for x in smells_trig:
            
            x = np.array(x)
            
            spike_averages.append(np.mean(x[:,int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 1))
            
        
        spike_averages = np.vstack(spike_averages)
        
        # average across different train-test spilts
        
        accuracy_sgd = []
        
        for random_splits in range(100):
            
            spikes_train, spikes_test, odors_train, odors_test = train_test_split(spike_averages, odorants, test_size=0.33, random_state=random_splits)
                
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=2000000, tol=1e-8,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(spikes_train, odors_train)
            accuracy_sgd.append(sgd_clf.score(spikes_test, odors_test))
        
        accuracies_bin.append(np.mean(accuracy_sgd))
        
        
    accuracies_animals_TeLC.append(accuracies_bin)   
    
#

accuracies_animals_TeLC_contra = [] 

# check contralateral hemishpere

names = ['150221','150312','150327','150403','150406','150812']

loading = np.array(exp_data[3][1:7])

for index,name in enumerate(names):
    
    
    print(name)
    
    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    
    lfp = np.load('PFx_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
    srate = 2000
    
    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name)
    
    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
    
    conc_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    #smell_series = list(np.array([11,7,8,6,12,10])-1) 
    
    if loading[index] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
    elif loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)  

    
    inh_start = inh_start[inh_start<lfp.shape[1]]
    
    
    spike_times = mat73.loadmat(name+'_bank2_st.mat')['SpikeTimes']['tsec']

    #
    recording_time = np.max(spike_times[0])+1
    srate_spikes = 30000
    
    neuron_number = len(spike_times)
    multi_units_1 = np.zeros([neuron_number,int(recording_time*srate_spikes)],dtype = np.bool_)
    
    
    for x in range(neuron_number):
        spike_range = np.where(spike_times[int(x)]<recording_time)[0]
        spikes = spike_times[int(x)][0:spike_range.shape[0]]
        s_unit = np.rint(spikes[0]*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1
        #indice = indice+1
    
    srate_resample = 2000
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)
    
    del multi_units_1,multi_units_re
    
    avg_rates = np.sum(sum_units_1,axis = 1)/recording_time
    putative_pyramidals = avg_rates<100
    
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    odorants = []
    for index_odors, x in enumerate(odor_onset_times):
        odor_repeats = x.shape[1]
        odorants.append(np.repeat(index_odors,odor_repeats))
        
    odorants = np.concatenate(odorants,axis = 0)
    
    inh_odor = []
    odorants_repeat = []
    
    for index_odor,x in enumerate(odor_onset_srate):
        index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
        inh_odor.append(inh_start[index_smells])
        if len(inh_start[index_smells]) > 0:
            num_breaths = len(inh_start[index_smells])
            odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    smells_trig = []
    
    for index_odor, x in enumerate(inh_smell):
        smells_trig.append(sum_units_1[:,int(x):int(x+1400)])
        
    
    window = 200
    
    accuracies_bin = []
    
    for time_bin in np.arange(100,1225,75):
        
        spike_averages = []
        
        for x in smells_trig:
            
            x = np.array(x)
            
            spike_averages.append(np.mean(x[:,int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 1))
            
        
        spike_averages = np.vstack(spike_averages)
        
        # average across different train-test spilts
        
        accuracy_sgd = []
        
        for random_splits in range(100):
            
            spikes_train, spikes_test, odors_train, odors_test = train_test_split(spike_averages, odorants, test_size=0.33, random_state=random_splits)
                
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=2000000, tol=1e-8,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(spikes_train, odors_train)
            accuracy_sgd.append(sgd_clf.score(spikes_test, odors_test))
        
        accuracies_bin.append(np.mean(accuracy_sgd))
        
    accuracies_animals_TeLC_contra.append(accuracies_bin)
    
#%% save results


directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

np.savez('Figure_7_2.npz', accuracies_animals_TeLC = accuracies_animals_TeLC, accuracies_animals_TeLC_contra = accuracies_animals_TeLC_contra)

 
#%%
directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

accuracies_animals_TeLC = np.load('Figure_7_2.npz')['accuracies_animals_TeLC']
accuracies_animals_TeLC_contra = np.load('Figure_7_2.npz')['accuracies_animals_TeLC_contra']

#%%
    
times = np.arange(100,1225,75)/2


accuracies_animals_TeLC = np.array(accuracies_animals_TeLC)
mean_acc_TeLC = np.mean(accuracies_animals_TeLC,axis = 0)
error_acc_TeLC = np.std(accuracies_animals_TeLC,axis = 0)/np.sqrt(8)

accuracies_animals_TeLC_contra = np.array(accuracies_animals_TeLC_contra)
mean_acc_TeLC_contra = np.mean(accuracies_animals_TeLC_contra,axis = 0)
error_acc_TeLC_contra = np.std(accuracies_animals_TeLC_contra,axis = 0)/np.sqrt(6)


plt.figure(dpi = 300, figsize = (3,5))

#plt.boxplot(accuracies_animals[:,:,0],showfliers=False,positions = times, widths = 5)
plt.fill_between(times, mean_acc_TeLC-error_acc_TeLC,mean_acc_TeLC+error_acc_TeLC , alpha = 0.2, color = 'tab:blue')
plt.plot(times, mean_acc_TeLC,color = 'tab:blue', label = 'SGD Accuracy', linewidth = 2)

plt.fill_between(times, mean_acc_TeLC_contra-error_acc_TeLC_contra,mean_acc_TeLC_contra+error_acc_TeLC_contra , alpha = 0.2, color = 'tab:red')
plt.plot(times, mean_acc_TeLC_contra,color = 'tab:red', label = 'SGD Accuracy', linewidth = 2)


plt.xticks(ticks = np.arange(50,600,100),labels = np.arange(50,600,100), rotation = 30)
plt.xlim([0,600])
#plt.ylim([16,60])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Decoding Accuracy (%)')
plt.xlabel('Time (ms)')

#%%

common_animals = [1,3,4,5,6,7]

time_bin = 4
plt.boxplot([accuracies_animals_TeLC[common_animals,time_bin],accuracies_animals_TeLC_contra[:,time_bin]], widths = 0.2)

accuracies_common_animals_TeLC = accuracies_animals_TeLC[common_animals,:]
for x in range(6):
    plt.plot([1.2,1.8],[accuracies_common_animals_TeLC[x,time_bin],accuracies_animals_TeLC_contra[x,time_bin]], color = 'grey')
    
plt.xticks(ticks = [1,2], labels = ['TeLC ipsi','TeLC contra'])

plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

s,p = stats.ttest_rel(accuracies_animals_TeLC_contra[:,time_bin],accuracies_common_animals_TeLC[:,time_bin],alternative = 'greater')
   
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize=(3,2))

#plt.subplot(121)

common_animals = [1,3,4,5,6,7]

time_bin = 4

norm_TeLC = accuracies_animals_TeLC[common_animals,time_bin]-accuracies_animals_TeLC_contra[:,time_bin]
norm_TeLC_contra = accuracies_animals_TeLC_contra[:,time_bin]-accuracies_animals_TeLC_contra[:,time_bin]

s,p = stats.ttest_1samp(norm_TeLC,popmean = 0,alternative = 'less')   
df = norm_TeLC.shape[0]-1

plt.boxplot([norm_TeLC_contra*100,norm_TeLC*100], widths = 0.4, positions = [0.8,2.2])

for x in range(6):
    plt.plot([1.2,1.8],[norm_TeLC_contra[x]*100,norm_TeLC[x]*100], color = 'grey')
    
plt.ylabel('$\Delta$ Decoding Accuracy (%)')    
    
plt.xticks(ticks = [0.8,2.2], labels = ['TeLC contra','TeLC ipsi'])

plt.xlim([0.4,2.6])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both') 

#plt.savefig('decoding_telc.pdf')
#%%
norm_TeLC = accuracies_animals_TeLC[common_animals,:]#/accuracies_animals_TeLC_contra
norm_TeLC_contra = accuracies_animals_TeLC_contra#/accuracies_animals_TeLC_contra


s_time = []
p_time = []
diff_time = []

for x in range(15):
    s,p = stats.ttest_rel(norm_TeLC[:,x],norm_TeLC_contra[:,x],alternative = 'less')    
    diff_time.append(norm_TeLC[:,x]-norm_TeLC_contra[:,x])
    s_time.append(s)
    p_time.append(p)
    
    
#

mean_diff = np.mean(diff_time,axis = 1)
error_diff = np.std(diff_time,axis = 1)/np.sqrt(6)
    
#plt.figure(dpi = 300, figsize = (3,5))
plt.subplot(122)
plt.plot(times, p_time)    
#plt.plot(times, mean_diff,'-o')    
#plt.fill_between(times, mean_diff-error_diff,mean_diff+error_diff , alpha = 0.2, color = 'tab:blue')
plt.plot(times,np.ones(15)*0.05)

plt.xticks(ticks = np.arange(50,600,100),labels = np.arange(50,600,100), rotation = 30)
plt.xlim([0,600])
#plt.ylim([16,60])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('P Value')
plt.xlabel('Time (ms)')
plt.title('TeLC Contra - TeLC ipsi')
plt.tight_layout()