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

def eegfilt(data,srate,flow,fhigh):
    
    # fir LS
    trans = 0.15
    nyq = srate*0.5
    f=[0, (1-trans)*flow/nyq, flow/nyq, fhigh/nyq, (1+trans)*fhigh/nyq, 1]
    m=[0,0,1,1,0,0]
    filt_order = 3*np.fix(srate/flow)
    if filt_order % 2 == 0:
        filt_order = filt_order + 1
      
    filtwts = signal.firls(filt_order,f,np.double(m))
    data_filt = signal.filtfilt(filtwts,1, data) 
    
    return(data_filt)

#%% experiment data 

names = ['160403-1','160403-2','160406-2','160409-1','160409-2','160422-1','160422-2','160423-1','160423-2','160425-1','160428-1','160428-2','160613-1','160623-1','160623-2']


directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory+'/VGAT')
exp_data = pd.read_csv('ExperimentCatalog_VGAT.txt', sep=" ", header=None)

loading = np.array(exp_data[3][1:16])

#%% loop through animals

accuracies_animals = []
accuracies_animals_gamma_phase = []
gamma_envelope_animals = []
vector_correlation_aniamls = []
accuracies_surrogate_animals = []

for index, name in enumerate(names):
    
    print(name)

    os.chdir(directory+'/VGAT/Decimated_LFPs')

    lfp = np.load('PFx_VGAT_lfp_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
    srate = 2000
    
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp[28,:],srate, 30,60)))
    gamma_phase = np.angle(signal.hilbert(eegfilt(lfp[28,:],srate, 30,60)))
    

    os.chdir(directory+'/VGAT/processed/'+name)
    
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate
    inh_start = np.squeeze(inh_start)
    positions = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['Wave']
    conc_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]

    odor_series = list(np.array([4,7,8,12,15,16])-1)
    odor_identity = [0,1,2,3,4,5]
    
    smell_times = conc_data[odor_series]
    smell_times_srate = smell_times*srate   
    
    # check VGAT+ neurons
    laser_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesDuringLaser'][0][0]
    before_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesBeforeLaser'][0][0]
    inh_laser = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserTimes'][0][0][0]['PREXIndex'][0][0]
    inh_nonlaser = np.delete(inh_start, inh_laser[0][:,0:20][0])
    
    inh_start = inh_start[inh_start<lfp.shape[1]]
    

    vgat = []
    vgat_neg = []
    y_position = []
    y_position_neg = []
    
    for x in np.arange(1,laser_spikes.shape[0]):
        spikes_laser = laser_spikes[x][0][0:20]
        spikes_before = before_spikes[x][0][0:20]
    
        s,p = stats.ranksums(spikes_laser,spikes_before,alternative = 'greater')
        if p < 0.001:
            vgat.append(x)
            y_position.append(positions[x]['Position'][1])
        else:
            y_position_neg.append(positions[x]['Position'][1])
            vgat_neg.append(x)
        
    vgat_neg_pos = 115

    vgats = np.vstack([vgat,y_position])    
    
    
    exc_neurons = vgat_neg
    conv_exc = []
    exc_spikes = []
    for x in exc_neurons:
        exc_spikes.append(spike_times[int(x)][0])
        
    
    recording_time = np.max(np.concatenate(exc_spikes))
    srate_spikes = 30000
    
    neuron_number = len(exc_spikes)
    multi_units_1 = np.zeros([neuron_number,int(recording_time*srate_spikes)],dtype = np.bool_)
    
    for x in range(neuron_number):
        spike_range = np.where(exc_spikes[x]<recording_time)[0]
        spikes = exc_spikes[x][spike_range]
        s_unit = np.rint(spikes*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1

    
    srate_resample = 2000
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)
    
    del multi_units_1,multi_units_re
    

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
        gamma_envelope_odor.append(gamma_envelope[int(x):int(int(x)+1400)])
        gamma_phase_odor.append(gamma_phase[int(x):int(int(x)+1400)])
        
   
    mean_envelope_odor = np.mean(gamma_envelope_odor,axis = 0)
    
    window = 200
    
    accuracies_bin = []
    mean_gamma_bin = []   
    mean_resp_bin = []
    
    for time_bin in np.arange(100,1225,75):
        
        gamma_bin = np.mean(mean_envelope_odor[int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 0)
        mean_gamma_bin.append(np.mean(gamma_bin))
        
        spike_averages = []
        
        for x in smells_trig:
            
            x = np.array(x)
            
            spike_averages.append(np.mean(x[:,int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 1))
            
        
        spike_averages = np.vstack(spike_averages)
        
        # average across different train-test spilts
        
        accuracy_sgd = []
        
        for random_splits in range(100):
            
            spikes_train, spikes_test, odors_train, odors_test = train_test_split(spike_averages, odorants, test_size=0.33, random_state=random_splits)
                
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=100000, tol=1e-8,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(spikes_train, odors_train)
            accuracy_sgd.append(sgd_clf.score(spikes_test, odors_test))
        
        accuracies_bin.append(np.mean(accuracy_sgd))
        
        
    accuracies_animals.append(accuracies_bin)        
    gamma_envelope_animals.append(mean_gamma_bin)
    
    # check pop vector correlations

    smells_trig = np.array(smells_trig)
    
    window = 200
    vector_corr_time = []
    
    for time_bin in np.arange(100,1225,75):
        
        spikes_gamma = []
        for x in np.arange(0,6):
            
            odors_indexes = odorants == x
            spikes_gamma.append(np.nanmean(np.nanmean(np.array(smells_trig[odors_indexes,:,:])[:,:,int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 2),axis = 0))
            
        vector_corr_matrix = np.zeros([6,6])  
        
        for x in np.arange(0,5):
            for y in np.arange(x+1,6):
                vector_corr_matrix[x,y] = pearsonr(spikes_gamma[x],spikes_gamma[y])[0]
                        
        vector_corr_matrix = vector_corr_matrix+vector_corr_matrix.T
        vector_corr_matrix = vector_corr_matrix + np.identity(6)

        vector_corr_time.append(vector_corr_matrix)

    vector_correlation_aniamls.append(vector_corr_time)        
        
    
    # test decoding as a function of the gamma cycle 

    numbin = 4
    
    position=np.zeros(numbin) # this variable will get the beginning (not the center) of each phase bin (in rads)
    winsize = 2*np.pi/numbin # bin de fase
    
    position = []
    for j in np.arange(1,numbin+1):
        position.append(-np.pi+(j-1)*winsize)
        

    time_bin = 500
    window = 1000
    
    
    spikes_gamma = []
    gamma_phase_trig = []
    
    for x in range(len(smells_trig)):
        
        spikes = np.array(smells_trig[x])
        phase = np.array(gamma_phase_odor[x])
        
        spikes_gamma.append(spikes[:,int(time_bin-(window/2)):int(time_bin+(window/2))])
        
        gamma_phase_trig.append(phase[int(time_bin-(window/2)):int(time_bin+(window/2))])
        
    
    spikes_gamma = np.array(spikes_gamma)
    gamma_phase_trig = np.vstack(gamma_phase_trig)
    #

    spike_phase_times_smells = []
    for x in range(gamma_phase_trig.shape[0]):
        
        gamma_phase_trig_smell = gamma_phase_trig[x,:]
        spike_phase_times = []
        
        for j in np.arange(0,numbin):  
            boolean_array = np.logical_and(gamma_phase_trig_smell >=  position[j], gamma_phase_trig_smell < position[j]+winsize)
            I = np.where(boolean_array)[0]
            spike_phase_times.append(np.mean(spikes_gamma[x,:,I],axis = 0))
            
            
        spike_phase_times_smells.append(spike_phase_times)
    
    spike_phase_times_smells = np.array(spike_phase_times_smells)
    
 
    accuracies_bin_loo = []
    
    for random_splits in range(100):    
        
        spikes_train, spikes_test, odors_train, odors_test = train_test_split(spike_phase_times_smells, odorants, test_size=0.33, random_state=random_splits)
        accuracies_bin = []
        for bins in range(numbin):
            sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=100000, tol=1e-8,alpha = 0.2, random_state = 1234))
            sgd_clf.fit(spikes_train[:,bins], odors_train)
            
            accuracy_sgd = sgd_clf.score(spikes_test[:,bins], odors_test)
            
            accuracies_bin.append(accuracy_sgd)
        accuracies_bin_loo.append(accuracies_bin) 
       
    accuracies_animals_gamma_phase.append(np.mean(accuracies_bin_loo,axis = 0))  




    # check surrogates
    
    accuracies_surrogate_gamma_phase = []
    
    for surrogates in range(100):
        
        # circular shift gamma phase
        shift = np.random.randint(gamma_phase.shape[0])
        gamma_phase_surr = np.roll(gamma_phase,shift)
        
        gamma_phase_odor = []
        
        for index_odor, x in enumerate(inh_smell):
            gamma_phase_odor.append(gamma_phase_surr[int(x):int(int(x)+1400)])
            
        spikes_gamma = []
        gamma_phase_trig = []
        
        for x in range(len(smells_trig)):
            
            spikes = np.array(smells_trig[x])
            phase = np.array(gamma_phase_odor[x])
            
            spikes_gamma.append(spikes[:,int(time_bin-(window/2)):int(time_bin+(window/2))])
            
            gamma_phase_trig.append(phase[int(time_bin-(window/2)):int(time_bin+(window/2))])
            
        
        spikes_gamma = np.array(spikes_gamma)
        gamma_phase_trig = np.vstack(gamma_phase_trig)
        #
    
        spike_phase_times_smells = []
        for x in range(gamma_phase_trig.shape[0]):
            
            gamma_phase_trig_smell = gamma_phase_trig[x,:]
            spike_phase_times = []
            
            for j in np.arange(0,numbin):  
                boolean_array = np.logical_and(gamma_phase_trig_smell >=  position[j], gamma_phase_trig_smell < position[j]+winsize)
                I = np.where(boolean_array)[0]
                spike_phase_times.append(np.mean(spikes_gamma[x,:,I],axis = 0))
                
                
            spike_phase_times_smells.append(spike_phase_times)
        
        spike_phase_times_smells = np.array(spike_phase_times_smells)
    
        accuracies_bin_loo = []
        
        for random_splits in range(100):    
        
            spikes_train, spikes_test, odors_train, odors_test = train_test_split(spike_phase_times_smells, odorants, test_size=0.33, random_state=random_splits)
            accuracies_bin = []
            for bins in range(numbin):
                sgd_clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=100000, tol=1e-8,alpha = 0.2, random_state = 1234))
                sgd_clf.fit(spikes_train[:,bins], odors_train)
                
                accuracy_sgd = sgd_clf.score(spikes_test[:,bins], odors_test)
                
                accuracies_bin.append(accuracy_sgd)
            accuracies_bin_loo.append(accuracies_bin) 
           
        accuracies_surrogate_gamma_phase.append(np.mean(accuracies_bin_loo,axis = 0))    
    
    accuracies_surrogate_animals.append(accuracies_surrogate_gamma_phase)
    
accuracies_animals = np.array(accuracies_animals)*100
accuracies_animals_gamma_phase = np.array(accuracies_animals_gamma_phase)*100




#%% save results

os.chdir(directory)

np.savez('Figure_7.npz', accuracies_animals = accuracies_animals, accuracies_animals_gamma_phase = accuracies_animals_gamma_phase, vector_correlation_aniamls = vector_correlation_aniamls, gamma_envelope_animals = gamma_envelope_animals)

#%%
os.chdir(directory)

accuracies_animals = np.load('Figure_7.npz') ['accuracies_animals'] 
gamma_envelope_animals = np.load('Figure_7.npz') ['gamma_envelope_animals'] 
vector_correlation_aniamls = np.load('Figure_7.npz') ['vector_correlation_aniamls'] 


#%%

gamma_envelope_animals = np.array(gamma_envelope_animals)

gamma_norm = []
for x in range(15):
    gamma_norm.append(gamma_envelope_animals[x,:]/np.mean(gamma_envelope_animals[x,:]))
    
gamma_norm = np.array(gamma_norm)

mean_envelope_odor = np.mean(gamma_norm,axis = 0)


times = np.linspace(75,600,accuracies_animals.shape[1])
times = np.arange(100,1225,75)/2
times = times

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

fig = plt.figure(dpi = 300, figsize = (4,6))

ax1 = plt.subplot(111)

mean_acc = np.mean(accuracies_animals[:,:],axis = 0)
error_acc = np.std(accuracies_animals[:,:],axis = 0)/np.sqrt(15)

plt.fill_between(times, mean_acc-error_acc,mean_acc+error_acc , alpha = 0.2, color = 'tab:blue')
plt.plot(times, mean_acc,color = 'tab:blue', linewidth = 2)

plt.plot(times,(mean_envelope_odor*100)-59, color = 'tab:orange', label = 'Rescaled $\gamma$ envelope', linewidth = 4)

plt.legend()

plt.xticks(ticks = np.arange(50,600,100),labels = np.arange(50,600,100), rotation = 30)
plt.ylim([25,56])

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Decoding Accuracy (%)')
plt.xlabel('Time (ms)')

ax2 = fig.add_axes([0.38,0.22,0.25,0.2])

acc_z = []
gamma_z = []

for x in range(15):
    
    acc_z.append(stats.zscore(accuracies_animals[x,:]))    
    gamma_z.append(stats.zscore(gamma_envelope_animals[x,:]))

acc_z = np.concatenate(acc_z)
gamma_z = np.concatenate(gamma_z)
    
sns.regplot(gamma_z,acc_z,robust = True, scatter_kws={'s':3, 'alpha':0.2}, color = 'black')
#sns.regplot(np.concatenate(gamma_norm[:,:]),np.concatenate(accuracies_animals[:,:]),robust = False, scatter_kws={'s':3, 'alpha':0.2}, color = 'black')
plt.ylabel('Z-score Accuracy (%)')
plt.xlabel('Z-score $\gamma$ power')

#r,p = stats.pearsonr(np.concatenate(gamma_norm[:,:]),np.concatenate(accuracies_animals[:,:]))
r,p = stats.pearsonr(acc_z,gamma_z)

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


s_pre,p_pre = stats.ttest_rel(accuracies_animals[:,4],accuracies_animals[:,0],alternative = 'greater')
s_post,p_post = stats.ttest_rel(accuracies_animals[:,4],accuracies_animals[:,-1],alternative = 'greater')
df_acc = accuracies_animals.shape[0]-1

plt.savefig('accuracy_gamma.pdf')
#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (8,2))


time = times.shape[0]
indice = 0
datos = []

plt.hlines(2,indice,16)
for x in range(15):
    
    plt.fill_between(np.arange(indice,time+indice), (np.ones(time)*-4)-0.3,(np.ones(time)*-4)+0.3,alpha = 0.4)
    
    acc_plot = stats.zscore(accuracies_animals[x,0:time])
    
    plt.plot(np.arange(indice,time+indice),acc_plot, color = 'tab:blue', linewidth = 1)
    
    gamma_plot = stats.zscore(gamma_envelope_animals[x,0:time])
    
    
    
    plt.plot(np.arange(indice,time+indice),gamma_plot, color = 'tab:orange', linewidth = 2)
    datos.append(np.arange(indice,50+indice))
    indice = indice+20
    
    
plt.box(False)
plt.xticks([])
plt.yticks([])

#plt.savefig('accuracies_animals_gamma.pdf')

#%%

accuracies_animals_gamma_phase = np.array(accuracies_animals_gamma_phase)

mean_accuracy = np.mean(accuracies_animals_gamma_phase,axis = 0)
error_accuracy = np.std(accuracies_animals_gamma_phase,axis = 0)/np.sqrt(15)

mean_accuracy = np.hstack([mean_accuracy,mean_accuracy])
error_accuracy = np.hstack([error_accuracy,error_accuracy])


phase = np.linspace(0,360*2,numbin*2)

#phase = np.hstack([phase,phase])

plt.figure(dpi = 300, figsize = (3,4))

plt.plot(phase,mean_accuracy)

plt.fill_between(phase, mean_accuracy-error_accuracy,mean_accuracy+error_accuracy, alpha = 0.2)
plt.ylabel('Odor decoding accuracy (%)')
plt.xlabel('Gamma phase (deg)')
plt.ylim([30,50])

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


#%%

from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


vector_correlation_aniamls = np.array(vector_correlation_aniamls)

plt.figure(dpi = 300, figsize = (15,5))

gs = GridSpec(2, 15, height_ratios=(4,1))

plt.subplot(gs[0,0:5])
population_correlation = np.mean(vector_correlation_aniamls[:,0,:,:],axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0.6, vmax = 0.8)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'], rotation = 35)
plt.yticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
plt.title('Pre gamma')

plt.subplot(gs[0,5:10])
population_correlation = np.mean(vector_correlation_aniamls[:,4,:,:],axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0.6, vmax = 0.8)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'],rotation = 35)
plt.yticks(ticks = np.arange(0,6),labels = [])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
plt.title('Gamma peak')

plt.subplot(gs[0,10:15])
population_correlation = np.mean(vector_correlation_aniamls[:,14,:,:],axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0.6, vmax = 0.8)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'], rotation = 35)
plt.yticks(ticks = np.arange(0,6),labels = [])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
plt.title('Post gamma')

plt.tight_layout()


#plt.figure(dpi = 300, figsize = (30,2))
for x in range(15):
    
    plt.subplot(gs[1,x])
    population_correlation = np.mean(vector_correlation_aniamls[:,x,:,:],axis = 0) 
    
    plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0.6, vmax = 0.8)
    plt.xticks(ticks = np.arange(0,6),labels = [])
    plt.yticks(ticks = np.arange(0,6),labels = [])
    plt.xlim([-0.5,5.5])
    plt.ylim([-0.5,5.5])
 
#plt.savefig('pop_vector_corr.pdf')    

#%% stats

mean_correlations = np.mean(np.mean(vector_correlation_aniamls[:,:,:,:],axis = 3),axis = 2)

s_pre_corr,p_pre_corr = stats.ttest_rel(mean_correlations[:,4],mean_correlations[:,0],alternative = 'less')
s_post_corr,p_post_corr = stats.ttest_rel(mean_correlations[:,4],mean_correlations[:,-1],alternative = 'less')
df_acc_corr = mean_correlations.shape[0]-1

#%%

accuracies_surrogate_animals = np.array(accuracies_surrogate_animals)

surrogate_decoding_dist = []
for x in range(10000):
    
    random_surr = np.random.randint(100, size = 15)
    surrogate_decoding_animals = []
    
    for y in range(15):    
        surrogate_decoding_animals.append(accuracies_surrogate_animals[y,random_surr[y],:])
    
    
    surrogate_decoding_dist.append(np.mean(surrogate_decoding_animals, axis = 0))
        
    
mean_accuracy_animal = np.mean(accuracies_animals_gamma_phase,axis = 0)

# construct MI surrogates from average decoding across 15 surrogates

accuracies_norm_surr = np.array(surrogate_decoding_dist)/np.sum(surrogate_decoding_dist,axis = 1)[:,np.newaxis]
entrop_surr = -1*np.sum(accuracies_norm_surr*np.log(accuracies_norm_surr),axis = 1)
mi_surr = (np.log(numbin)-entrop_surr)/np.log(numbin) 

# construct MI from average decoding

accuracies_norm_real = np.array(mean_accuracy_animal)/np.sum(mean_accuracy_animal,axis = 0)
entrop_real = -1*np.sum(accuracies_norm_real*np.log(accuracies_norm_real),axis = 0)
mi_real = (np.log(numbin)-entrop_real)/np.log(numbin) 
  
#%%

norm_acc_phase = accuracies_animals_gamma_phase-np.mean(accuracies_animals_gamma_phase,axis = 1)[:,np.newaxis]

mean_accuracy = np.mean(norm_acc_phase,axis = 0)
error_accuracy = np.std(norm_acc_phase,axis = 0)/np.sqrt(15)

mean_accuracy = np.hstack([mean_accuracy,mean_accuracy])
error_accuracy = np.hstack([error_accuracy,error_accuracy])


phase = np.linspace(0,360*2,numbin*2)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (6,3))

plt.subplot(121)
plt.plot(phase,mean_accuracy)

plt.fill_between(phase, mean_accuracy-error_accuracy,mean_accuracy+error_accuracy, alpha = 0.2)
plt.ylabel('$\Delta$ acurracy (%)')
plt.xlabel('Gamma phase (deg)')
plt.ylim([-5,5])

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.subplot(122)
sns.histplot(mi_surr,kde = True, color = 'black', element = 'bars')
plt.vlines(mi_real,0,600, color = 'tab:red',linewidth = 3, label = 'Real MI')


plt.ylabel('Counts')
plt.xlabel('Surrogate decoding MI')

p = np.sum(mi_surr>mi_real)/10000
#plt.text(0.0055,632,'p='+str(p))

plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.tight_layout()

plt.savefig('accuracy_gamma_phase.pdf')
