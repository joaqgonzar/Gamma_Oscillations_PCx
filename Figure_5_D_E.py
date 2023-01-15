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
from scipy.stats import pearsonr
from scipy.stats import spearmanr

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


def spike_rate(spike_times):
    
    srate_spikes = 30000
    srate_resample = 2000
    
    recording_time = np.max(np.concatenate(spike_times))

    neuron_number = len(spike_times)
    multi_units_1 = np.zeros([neuron_number,int(recording_time*srate_spikes)],dtype = np.bool_)
    
    indice = 1
    for x in range(neuron_number):
        spike_range = np.where(spike_times[x]<recording_time)[0]
        spikes = spike_times[x][spike_range]
        s_unit = np.rint(spikes*srate_spikes) 
        s_unit = s_unit.astype(np.int64)
        multi_units_1[x,s_unit]=1
        indice = indice+1
    
    
    new_bin = int(srate_spikes/srate_resample)
    max_length = int(multi_units_1.shape[1]/new_bin)*new_bin
    multi_units_re = np.reshape(multi_units_1[:,0:max_length],[multi_units_1.shape[0],int(multi_units_1.shape[1]/new_bin),new_bin])
    sum_units_1 = np.sum(multi_units_re,axis = 2)


    kernel = signal.gaussian(int(0.1*srate_resample),20)
   
    # convolve units with gaussian kernel
    
    conv_neurons_session = []
    for x in range(sum_units_1.shape[0]):
        conv = signal.convolve(np.squeeze(sum_units_1[x,:]), kernel,mode = 'same') 
        conv = conv*2000/np.sum(kernel)
        conv_neurons_session.append(conv)
    
    return(conv_neurons_session,sum_units_1)

#%% experiment data 

names = ['160403-1','160403-2','160406-2','160409-1','160409-2','160422-1','160422-2','160423-1','160423-2','160425-1','160428-1','160428-2','160613-1','160623-1','160623-2']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory+'/VGAT')

exp_data = pd.read_csv('ExperimentCatalog_VGAT.txt', sep=" ", header=None)

loading = np.array(exp_data[3][1:16])

#%% loop through animals

gamma_envelope_animals = []
spec_index_animals = []

for index, name in enumerate(names):
    
    print(name)

    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/VGAT/Decimated_LFPs')

    lfp = np.load('PFx_VGAT_lfp_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    srate = 2000
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp[28,:], srate, 30, 60)))
    
    
    os.chdir('/run/user/1000/gvfs/smb-share:server=nas_sueno.local,share=datos/Frank_Boldings_Dataset/VGAT/processed/'+name)
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate
    inh_start = np.squeeze(inh_start)
    positions = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['Wave']
    conc_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]

    odor_series = list(np.array([4,7,8,12,15,16])-1) 
    
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
    
        #if np.sum(spikes_laser-spikes_before)>0:
        s,p = stats.ranksums(spikes_laser,spikes_before,alternative = 'greater')
        if p < 0.001:
            vgat.append(x)
            y_position.append(positions[x]['Position'][1])
        else:
            y_position_neg.append(positions[x]['Position'][1])
            vgat_neg.append(x)
        
    vgat_neg_pos = 115

    vgats = np.vstack([vgat,y_position])    
    
    # get exc neurons 
    
    exc_neurons = vgat_neg
    exc_spikes = []
    for x in exc_neurons:
        exc_spikes.append(spike_times[int(x)][0])
        
    if len(exc_spikes)>0:    
        [exc_neurons_session,units_exc] = spike_rate(exc_spikes)
        exc_neurons_session = np.array(exc_neurons_session)[:,0:lfp.shape[1]]
        units_exc = np.array(units_exc)[:,0:lfp.shape[1]]
    
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
        index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
        inh_odor.append(inh_start[index_smells])
        if len(inh_start[index_smells]) > 0:
            num_breaths = len(inh_start[index_smells])
            odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    # get spiking activity and gamma envelopes for all inhalations
    smells_trig = []
    gamma_envelope_odor = []  
    
    for x in inh_smell:
        smells_trig.append(units_exc[:,int(x):int(x+1800)])
        gamma_envelope_odor.append(gamma_envelope[int(x):int(int(x)+1800)])
        
    mean_envelope_odor = np.mean(gamma_envelope_odor,axis = 0)
    smells_trig = np.array(smells_trig)
    
    
    # calculate specificity index as a funcion of time following the inhalation start
    
    window = 200
    spec_index_bins = []
    mean_gamma_bins = []
    
    for time_bin in np.arange(100,1500,75):
        
        # bin gamma envelope
        
        gamma_bin = np.mean(mean_envelope_odor[int(time_bin-(window/2)):int(time_bin+(window/2))],axis = 0)
        mean_gamma_bins.append(np.mean(gamma_bin))
        
        # loop though odors
        
        mean_spikes_odor = []
        for x in range(6):
            odor = odorants == x
            spikes_odor = smells_trig[odor,:,int(time_bin-(window/2)):int(time_bin+(window/2))]
            mean_spikes_odor.append(np.nanmean(np.nanmean(spikes_odor,axis = 2),axis = 0))
         
        # get specificty index for each time bin
        
        spike_sum_odors = np.array(mean_spikes_odor[0]+mean_spikes_odor[1]+mean_spikes_odor[2]+mean_spikes_odor[3]+mean_spikes_odor[4]+mean_spikes_odor[5])
        mean_spikes_odor = np.array(mean_spikes_odor)
        max_spikes = np.max(mean_spikes_odor,axis = 0)
        non_zero_spikes = spike_sum_odors > 0
        spec_index = np.nanmean(max_spikes[non_zero_spikes]/spike_sum_odors[non_zero_spikes])
        
        spec_index_bins.append(spec_index)
        
    
    gamma_envelope_animals.append(np.array(mean_gamma_bins))
    spec_index_animals.append(np.array(spec_index_bins))

#%% save results

os.chdir(directory)

np.savez('Figure_5_2.npz', gamma_envelope_animals = gamma_envelope_animals, spec_index_animals = spec_index_animals)
    
#%% plot specificity index and gamma amplitude envelope


spec_index_animals = np.array(spec_index_animals)


mean_odor = np.mean(spec_index_animals,axis = 0)
error_odor = np.std(spec_index_animals,axis = 0)/np.sqrt(15)

# rescale gamma envelope
envelope_odor_norm = (gamma_envelope_animals/np.sum(gamma_envelope_animals,axis = 1)[:,np.newaxis]*6.2)+0.1
mean_envelope_odor = np.mean(envelope_odor_norm,axis = 0)
error_envelope_odor = np.std(envelope_odor_norm,axis = 0)/np.sqrt(15)


times = np.arange(100,1500,75)/2

#
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

fig = plt.figure(dpi = 300, figsize=(4,3.5))

plt.plot(times,mean_odor, label = 'Specificity index')
plt.fill_between(times, mean_odor-error_odor,mean_odor+error_odor, alpha = 0.2)

plt.plot(times,mean_envelope_odor, label = 'Norm $\gamma$ amp')
plt.fill_between(times, mean_envelope_odor-error_envelope_odor,mean_envelope_odor+error_envelope_odor, alpha = 0.2)
plt.legend()

plt.ylabel('Odor specificity index')
plt.xlabel('Time from inh start (ms)')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(100,800,100))

fig.align_ylabels()
plt.tight_layout()

plt.savefig('specificity_index.pdf')
    