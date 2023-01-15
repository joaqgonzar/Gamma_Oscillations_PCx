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
from statsmodels.tsa.stattools import grangercausalitytests

# define functions

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

names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)

awake_times = np.array([names,np.array(exp_data[2][1:14])])

# define electrode positions in the central array of the probe 

positions_2 = np.array([8,28,1,30,11,20,10,21,15,16,14,17])
positions_1 = np.array([9,29,0,31,10,21,11,20,14,17,15,16])

loading = np.array(exp_data[3][1:14])
 
#%% loop through animals

power_animals_odorless = []
power_animals_odor = []
induced_gamma_animals = []
induced_beta_animals = []
evoked_gamma_animals = []
evoked_beta_animals = []
itc_gamma_animals = []
itc_beta_animals = []

gamma_envelope_no_odor_animals = []
gamma_envelope_odor_animals = []
beta_envelope_no_odor_animals = []
beta_envelope_odor_animals = []

beta_odor_identity_animals = []
gamma_odor_identity_animals = []
beta_odor_concentration_animals = []
gamma_odor_concentration_animals = []

for index, name in enumerate(names):
    
    print(name)
    
    # load data 

    # load lfp
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')
    lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    srate = 2000
    
    # get same channel for all recordings
    if (name == '141208-1') or (name =='141208-2') or (name == '141209'):
        channel = 17
        positions = positions_1
    else: 
        channel = 16 
        positions = positions_2
    
    
    # load respiration and get awake recording
    os.chdir(directory+'/Simul/processed/'+name)
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    resp = np.squeeze(resp[0:int(lfp.shape[1])]) # get only awake times
    
    # get odor inhalations
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
    elif loading[index] == 'C':
        odor_series = list(np.array([11,7,8,6,12,10])-1)
    
    
    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate   
    odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate,axis = 1))
    
    # get inh starts
    resp_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    resp_start = resp_start[0]
    
    # exclude odor deliveries and non-awake times
    inh_no_odor = np.setxor1d(odor_times_srate,resp_start)
    inh_no_odor = inh_no_odor[inh_no_odor<resp.shape[0]]
  
    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[1]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[1]))
    
    lfp_odorless = lfp[channel,lfp_mask]
    lfp_odor = lfp[channel,odor_times]
    
    
    #smell_times = odor_data[odor_series]
    #smell_times_srate = smell_times*srate
    
    #inh_smell = odor_times_awake
    
    #
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    
    inh_odor = []
    for x in odor_onset_srate:
        index_smells = np.logical_and((resp_start>x),(resp_start<x+2000))
        inh_odor.append(resp_start[index_smells])
            
    inh_odor = np.concatenate(inh_odor,axis = 0)
    inh_no_odor = np.setxor1d(inh_odor,resp_start)
    
    
    # compute power spectrum 
    
    f,pxx_odorless = signal.welch(lfp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,pxx_odor = signal.welch(lfp_odor,fs = srate,nperseg = srate,nfft = 10*srate)
    
    
    # get triggered induced, evoked responses, and itc
    
    gamma_filtered = eegfilt(lfp[channel,:],srate, 30,60)
    gamma_envelope = np.abs(signal.hilbert(gamma_filtered))
    gamma_phase = np.angle(signal.hilbert(gamma_filtered))
    gamma_complex = np.exp(1j*gamma_phase)
    
    beta_filtered = eegfilt(lfp[channel,:],srate, 10,20)
    beta_envelope = np.abs(signal.hilbert(beta_filtered))
    beta_phase = np.angle(signal.hilbert(beta_filtered))
    beta_complex = np.exp(1j*beta_phase)         
        
    gamma_filtered_inh = []
    gamma_envelope_inh = []
    gamma_complex_inh = []
    
    for x in inh_odor:
        if gamma_filtered[int(x):int(int(x)+2000)].shape[0] == 2000:
            gamma_filtered_inh.append(gamma_filtered[int(x):int(int(x)+2000)]) 
            gamma_envelope_inh.append(gamma_envelope[int(x):int(int(x)+2000)])
            gamma_complex_inh.append(gamma_complex[int(x):int(int(x)+2000)]) 
        
    mean_gamma_inh = np.mean(gamma_filtered_inh, axis = 0)
    evoked_gamma_inh = np.abs(signal.hilbert(mean_gamma_inh))
    induced_gamma_inh = np.mean(gamma_envelope_inh, axis = 0)
    itc_gamma = np.abs(np.mean(gamma_complex_inh,axis = 0)) 
    
    beta_filtered_inh = []
    beta_envelope_inh = []
    beta_complex_inh = []
    
    for x in inh_odor:
        if beta_filtered[int(x):int(int(x)+2000)].shape[0] == 2000:
            beta_filtered_inh.append(beta_filtered[int(x):int(int(x)+2000)]) 
            beta_envelope_inh.append(beta_envelope[int(x):int(int(x)+2000)])
            beta_complex_inh.append(beta_complex[int(x):int(int(x)+2000)]) 
        
    mean_beta_inh = np.mean(beta_filtered_inh, axis = 0)
    evoked_beta_inh = np.abs(signal.hilbert(mean_beta_inh))
    induced_beta_inh = np.mean(beta_envelope_inh, axis = 0)
    itc_beta = np.abs(np.mean(beta_complex_inh,axis = 0)) 
    
    power_animals_odorless.append(pxx_odorless)
    power_animals_odor.append(pxx_odor)
    
    induced_gamma_animals.append(induced_gamma_inh)
    induced_beta_animals.append(induced_beta_inh)
    evoked_gamma_animals.append(evoked_gamma_inh)
    evoked_beta_animals.append(evoked_beta_inh)
    itc_gamma_animals.append(itc_gamma)
    itc_beta_animals.append(itc_beta)
    
    
    # compare odor and odorless cycles 
    
    gamma_envelope_inh_no_odor = []
    beta_envelope_inh_no_odor = []
    
    for x in inh_no_odor:
        if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            gamma_envelope_inh_no_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
            beta_envelope_inh_no_odor.append(beta_envelope[int(x):int(int(x)+2000)])
     
    gamma_envelope_inh_no_odor = np.mean(gamma_envelope_inh_no_odor,axis = 0)        
    beta_envelope_inh_no_odor = np.mean(beta_envelope_inh_no_odor,axis = 0)
    
    
    gamma_envelope_inh_odor = []
    beta_envelope_inh_odor = []
    
    for x in inh_odor:
        if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            gamma_envelope_inh_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
            beta_envelope_inh_odor.append(beta_envelope[int(x):int(int(x)+2000)])
            
    gamma_envelope_inh_odor = np.mean(gamma_envelope_inh_odor,axis = 0)        
    beta_envelope_inh_odor = np.mean(beta_envelope_inh_odor,axis = 0)
        
    gamma_envelope_no_odor_animals.append(gamma_envelope_inh_no_odor)
    gamma_envelope_odor_animals.append(gamma_envelope_inh_odor)
    beta_envelope_no_odor_animals.append(beta_envelope_inh_no_odor)
    beta_envelope_odor_animals.append(beta_envelope_inh_odor)
    
    
    # get beta and gamma responses to odor identity 
    
    if loading[index] == 'A':
        odor_series = list(np.array([4,7,8,12,15,16])-1) 
        #conc_series1 = list(np.array([2,3,4,5])-1)
        #conc_series2 = list(np.array([10,11,12,13])-1)
        
    elif loading[index] == 'C':
        odor_series = list(np.array([11,7,8,6,12,10])-1)
    
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    
    inh_odor_identity = []
    for y in range(odor_onset_srate.shape[0]):
        onsets_odor = odor_onset_srate[y][0]
        inh_odor = []
        for x in onsets_odor:
            index_smells = np.logical_and((resp_start>x),(resp_start<x+2000))
            inh_odor.append(resp_start[index_smells])
        inh_odor_identity.append(np.concatenate(inh_odor))    
    
    
    gamma_envelope_odor_identity = []
    beta_envelope_odor_identity = []
    
    for y in range(odor_onset_srate.shape[0]):
        
        inh_odor = inh_odor_identity[y]
        
        gamma_envelope_inh_odor = []
        beta_envelope_inh_odor = []
        
        for x in inh_odor:
            if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
                gamma_envelope_inh_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
                beta_envelope_inh_odor.append(beta_envelope[int(x):int(int(x)+2000)])
                
        gamma_envelope_odor_identity.append(np.mean(gamma_envelope_inh_odor,axis = 0))
        beta_envelope_odor_identity.append(np.mean(beta_envelope_inh_odor,axis = 0))
    
    beta_odor_identity_animals.append(beta_envelope_odor_identity)   
    gamma_odor_identity_animals.append(gamma_envelope_odor_identity)   
    
    # get beta and gamma responses to odor concentration 
    
    if loading[index] == 'A':
        
        conc_series1 = list(np.array([2,3,4,5])-1)
        conc_series2 = list(np.array([10,11,12,13])-1)
    
        odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
        odor_onset_times_1 = odor_onset[conc_series1]
        odor_onset_srate_1 = odor_onset_times_1*srate   
        odor_onset_times_2 = odor_onset[conc_series2]
        odor_onset_srate_2 = odor_onset_times_2*srate  
        
        inh_odor_concentration = []
        for y in range(odor_onset_srate_1.shape[0]):
            
            onsets_odor_1 = odor_onset_srate_1[y][0]
            onsets_odor_2 = odor_onset_srate_2[y][0]
            onsets_odor = np.concatenate([onsets_odor_1,onsets_odor_2])
            
            inh_odor = []
            for x in onsets_odor:
                index_smells = np.logical_and((resp_start>x),(resp_start<x+2000))
                inh_odor.append(resp_start[index_smells])
            inh_odor_concentration.append(np.concatenate(inh_odor))
            
        gamma_envelope_odor_concentration = []
        beta_envelope_odor_concentration = []
        
        for y in range(odor_onset_srate_1.shape[0]):
            
            inh_odor = inh_odor_concentration[y]
            
            gamma_envelope_inh_odor = []
            beta_envelope_inh_odor = []
            
            for x in inh_odor:
                if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
                    gamma_envelope_inh_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
                    beta_envelope_inh_odor.append(beta_envelope[int(x):int(int(x)+2000)])
                    
            gamma_envelope_odor_concentration.append(np.mean(gamma_envelope_inh_odor,axis = 0))
            beta_envelope_odor_concentration.append(np.mean(beta_envelope_inh_odor,axis = 0))
        
        beta_odor_concentration_animals.append(beta_envelope_odor_concentration)   
        gamma_odor_concentration_animals.append(gamma_envelope_odor_concentration)   
       
# get telc ipsi odor responses 
    
# experiment data 

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos'

os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx')

exp_data = pd.read_csv('ExperimentCatalog_TeLC-PCX.txt', sep=" ", header=None)

names = ['150220','150221','150223','150312','150327','150403','150406','150812']

loading = np.array(exp_data[3][7:15])

beta_envelope_telc_animals = []
gamma_envelope_telc_animals = []

for index,name in enumerate(names):
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
        
    elif  loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        
    print(name)
    
    # load data 
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    lfp = np.load('PFx_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    channel = 17
    lfp = lfp[channel,:]
    srate = 2000
    
    # get envelopes
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp,srate, 30, 60)))
    beta_envelope = np.abs(signal.hilbert(eegfilt(lfp,srate, 10, 20)))
    
    # get resp starts
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name) 
    resp_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    resp_start = resp_start[0]
    
    # get odor onsets
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    inh_odor = []
    for x in odor_onset_srate:
        index_smells = np.logical_and((resp_start>x),(resp_start<x+2000))
        inh_odor.append(resp_start[index_smells])
            
    inh_odor = np.concatenate(inh_odor,axis = 0)
    inh_no_odor = np.setxor1d(inh_odor,resp_start)
        
    beta_envelope_odor = []
    for x in inh_odor:
        if beta_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            beta_envelope_odor.append(beta_envelope[int(x):int(int(x)+2000)])        
    
    gamma_envelope_odor = []
    for x in inh_odor:
        if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            gamma_envelope_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
    
    mean_envelope_beta = np.mean(beta_envelope_odor, axis = 0)
    mean_envelope_gamma = np.mean(gamma_envelope_odor, axis = 0)
        
    beta_envelope_telc_animals.append(mean_envelope_beta)
    gamma_envelope_telc_animals.append(mean_envelope_gamma)
    
# get telc contra odor responses 
    
# experiment data 

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos'

os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx')

exp_data = pd.read_csv('ExperimentCatalog_TeLC-PCX.txt', sep=" ", header=None)

names_contra = ['150221','150312','150327','150403','150406','150812']

loading = np.array(exp_data[3][1:7])

beta_envelope_telc_contra_animals = []
gamma_envelope_telc_contra_animals = []

for index,name in enumerate(names_contra):
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
        
    elif  loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        
    print(name)
    
    # load data 
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    lfp = np.load('PFx_contra_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    channel = 17
    lfp = lfp[channel,:]
    srate = 2000
    
    # get envelopes
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp,srate, 30, 60)))
    beta_envelope = np.abs(signal.hilbert(eegfilt(lfp,srate, 10, 20)))
    
    # get resp starts
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name) 
    resp_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    resp_start = resp_start[0]
    
    # get odor onsets
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    inh_odor = []
    for x in odor_onset_srate:
        index_smells = np.logical_and((resp_start>x),(resp_start<x+2000))
        inh_odor.append(resp_start[index_smells])
            
    inh_odor = np.concatenate(inh_odor,axis = 0)
    inh_no_odor = np.setxor1d(inh_odor,resp_start)
        
    beta_envelope_odor = []
    for x in inh_odor:
        if beta_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            beta_envelope_odor.append(beta_envelope[int(x):int(int(x)+2000)])        
    
    gamma_envelope_odor = []
    for x in inh_odor:
        if gamma_envelope[int(x):int(int(x)+2000)].shape[0] == 2000:
            gamma_envelope_odor.append(gamma_envelope[int(x):int(int(x)+2000)])
    
    mean_envelope_beta = np.mean(beta_envelope_odor, axis = 0)
    mean_envelope_gamma = np.mean(gamma_envelope_odor, axis = 0)
        
    beta_envelope_telc_contra_animals.append(mean_envelope_beta)
    gamma_envelope_telc_contra_animals.append(mean_envelope_gamma)    
    
#%% save results 

os.chdir(directory)

np.savez('Figure_4.npz', power_animals_odorless = power_animals_odorless, power_animals_odor = power_animals_odor, induced_gamma_animals = induced_gamma_animals, induced_beta_animals = induced_beta_animals,evoked_gamma_animals = evoked_gamma_animals,  evoked_beta_animals = evoked_beta_animals,itc_gamma_animals = itc_gamma_animals,  itc_beta_animals = itc_beta_animals,gamma_envelope_no_odor_animals = gamma_envelope_no_odor_animals, gamma_envelope_odor_animals = gamma_envelope_odor_animals,beta_envelope_no_odor_animals = beta_envelope_no_odor_animals, beta_envelope_odor_animals = beta_envelope_odor_animals,beta_odor_identity_animals = beta_odor_identity_animals, gamma_odor_identity_animals = gamma_odor_identity_animals,beta_odor_concentration_animals = beta_odor_concentration_animals, gamma_odor_concentration_animals = gamma_odor_concentration_animals, beta_envelope_telc_contra_animals = beta_envelope_telc_contra_animals, gamma_envelope_telc_contra_animals = gamma_envelope_telc_contra_animals, beta_envelope_telc_animals = beta_envelope_telc_animals, gamma_envelope_telc_animals = gamma_envelope_telc_animals)

#%%
power_animals_odorless = np.array(power_animals_odorless)[:,5:3000]
power_animals_odor = np.array(power_animals_odor)[:,5:3000]
    
#%% plot power spectrum

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,3))

f = np.arange(0,1000.1,0.1)[5:3000]
#f = f[5:]


mean_power = np.mean(power_animals_odorless,axis = 0)*f
error_power = (np.std(power_animals_odorless,axis = 0)*f)/np.sqrt(13)


# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power[line_freq-20:line_freq+20] = (mean_power[line_freq-60:line_freq-20]+mean_power[line_freq+20:line_freq+60])/2
error_power[line_freq-20:line_freq+20] = (error_power[line_freq-60:line_freq-20]+error_power[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power,linewidth = 1.5, color = 'tab:blue', label = 'Odorless cycles')
plt.fill_between(f,mean_power-2*error_power,mean_power+2*error_power,color = 'tab:blue',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3)
plt.ylabel('Whitened Power')
plt.xlabel('Frequency(Hz)')

mean_power = np.mean(power_animals_odor,axis = 0)*f
error_power = (np.std(power_animals_odor,axis = 0)*f)/np.sqrt(13)


# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power[line_freq-20:line_freq+20] = (mean_power[line_freq-60:line_freq-20]+mean_power[line_freq+20:line_freq+60])/2
error_power[line_freq-20:line_freq+20] = (error_power[line_freq-60:line_freq-20]+error_power[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power,linewidth = 1.5, color = 'tab:orange', label = 'Odor cycles')
plt.fill_between(f,mean_power-2*error_power,mean_power+2*error_power,color = 'tab:orange',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'x', which = 'both')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.5,300])
plt.ylim([2e3,3e5])
plt.ylabel('Whitened Power')
plt.xlabel('Frequency(Hz)')

plt.legend()

plt.tight_layout()

plt.savefig('power_odor_odorless.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

gamma_envelope_no_odor_animals = np.array(gamma_envelope_no_odor_animals)
gamma_envelope_odor_animals = np.array(gamma_envelope_odor_animals)
beta_envelope_no_odor_animals = np.array(beta_envelope_no_odor_animals)
beta_envelope_odor_animals = np.array(beta_envelope_odor_animals)


mean_gamma_no_odor = np.mean(gamma_envelope_no_odor_animals,axis = 0)
error_gamma_no_odor = np.std(gamma_envelope_no_odor_animals,axis = 0)/np.sqrt(13)

mean_gamma_odor = np.mean(gamma_envelope_odor_animals,axis = 0)
error_gamma_odor = np.std(gamma_envelope_odor_animals,axis = 0)/np.sqrt(13)

plt.figure(dpi = 300, figsize = (8,4))

plt.subplot(122)

plt.plot(np.linspace(0,1000,2000),mean_gamma_no_odor, label = 'No odor', color = 'tab:blue')
plt.fill_between(np.linspace(0,1000,2000), mean_gamma_no_odor-error_gamma_no_odor, mean_gamma_no_odor+error_gamma_no_odor, alpha = 0.2, color = 'tab:blue')

plt.plot(np.linspace(0,1000,2000),mean_gamma_odor, label = 'Odor', color = 'tab:orange')
plt.fill_between(np.linspace(0,1000,2000), mean_gamma_odor-error_gamma_odor, mean_gamma_odor+error_gamma_odor, alpha = 0.2, color = 'tab:orange')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,550])
plt.title('Gamma')
plt.legend()

mean_beta_no_odor = np.mean(beta_envelope_no_odor_animals,axis = 0)
error_beta_no_odor = np.std(beta_envelope_no_odor_animals,axis = 0)/np.sqrt(13)

mean_beta_odor = np.mean(beta_envelope_odor_animals,axis = 0)
error_beta_odor = np.std(beta_envelope_odor_animals,axis = 0)/np.sqrt(13)


plt.subplot(121)

plt.plot(np.linspace(0,1000,2000),mean_beta_no_odor, label = 'beta', color = 'tab:blue')
plt.fill_between(np.linspace(0,1000,2000), mean_beta_no_odor-error_beta_no_odor, mean_beta_no_odor+error_beta_no_odor, alpha = 0.2, color = 'tab:blue')

plt.plot(np.linspace(0,1000,2000),mean_beta_odor, label = 'gamma', color = 'tab:orange')
plt.fill_between(np.linspace(0,1000,2000), mean_beta_odor-error_beta_odor, mean_beta_odor+error_beta_odor, alpha = 0.2, color = 'tab:orange')

plt.title('Beta')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,550])
plt.ylabel('Envelope amplitude (a.u.)')
    
plt.savefig('envelopes_odor_odorless.pdf')

#%%

evoked_gamma = np.array(evoked_gamma_animals)[:,0:1000]
induced_gamma = np.array(induced_gamma_animals)[:,0:1000]

mean_evoked_gamma = np.mean(evoked_gamma-evoked_gamma[:,0][:,np.newaxis],axis = 0)
error_evoked_gamma = np.std(evoked_gamma-evoked_gamma[:,0][:,np.newaxis],axis = 0)/np.sqrt(13)

mean_induced_gamma = np.mean(induced_gamma-induced_gamma[:,0][:,np.newaxis],axis = 0)
error_induced_gamma = np.std(induced_gamma-induced_gamma[:,0][:,np.newaxis],axis = 0)/np.sqrt(13)

evoked_beta = np.array(evoked_beta_animals)[:,0:1000]
induced_beta = np.array(induced_beta_animals)[:,0:1000]

mean_evoked_beta = np.mean(evoked_beta-evoked_beta[:,0][:,np.newaxis],axis = 0)
error_evoked_beta = np.std(evoked_beta-evoked_beta[:,0][:,np.newaxis],axis = 0)/np.sqrt(13)

mean_induced_beta = np.mean(induced_beta-induced_beta[:,0][:,np.newaxis],axis = 0)
error_induced_beta = np.std(induced_beta-induced_beta[:,0][:,np.newaxis],axis = 0)/np.sqrt(13)


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,8))

plt.subplot(313)

plt.plot(np.linspace(0,500,1000),mean_evoked_gamma, label = 'Evoked', color = 'tab:purple')
plt.fill_between(np.linspace(0,500,1000), mean_evoked_gamma-error_evoked_gamma, mean_evoked_gamma+error_evoked_gamma, alpha = 0.2, color = 'tab:purple')

plt.plot(np.linspace(0,500,1000),mean_induced_gamma, label = 'Induced', color = 'tab:green')
plt.fill_between(np.linspace(0,500,1000), mean_induced_gamma-error_induced_gamma, mean_induced_gamma+error_induced_gamma, alpha = 0.2, color = 'tab:green')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Norm Ganna Power')
plt.xlabel('Time from inh start (ms)')

plt.subplot(312)

plt.plot(np.linspace(0,500,1000),mean_evoked_beta, label = 'Evoked', color = 'tab:purple')
plt.fill_between(np.linspace(0,500,1000), mean_evoked_beta-error_evoked_beta, mean_evoked_beta+error_evoked_beta, alpha = 0.2, color = 'tab:purple')

plt.plot(np.linspace(0,500,1000),mean_induced_beta, label = 'Induced', color = 'tab:green')
plt.fill_between(np.linspace(0,500,1000), mean_induced_beta-error_induced_beta, mean_induced_beta+error_induced_beta, alpha = 0.2, color = 'tab:green')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Norm Beta Power')
plt.legend()


itc_beta_animals = np.array(itc_beta_animals)
itc_gamma_animals = np.array(itc_gamma_animals)

mean_itc_beta = np.mean(itc_beta_animals[:,0:1000],axis = 0)
error_itc_beta = np.std(itc_beta_animals[:,0:1000],axis = 0)/np.sqrt(13)

mean_itc_gamma = np.mean(itc_gamma_animals[:,0:1000],axis = 0)
error_itc_gamma = np.std(itc_gamma_animals[:,0:1000],axis = 0)/np.sqrt(13)


plt.subplot(311)

plt.plot(np.linspace(0,500,1000),mean_itc_beta, label = 'Beta', color = 'tab:green')
plt.fill_between(np.linspace(0,500,1000), mean_itc_beta-error_itc_beta, mean_itc_beta+error_itc_beta, alpha = 0.2, color = 'tab:green')
plt.plot(np.linspace(0,500,1000),mean_itc_gamma, label = 'Gamma', color = 'tab:purple')
plt.fill_between(np.linspace(0,500,1000), mean_itc_gamma-error_itc_gamma, mean_itc_gamma+error_itc_gamma, alpha = 0.2, color = 'tab:purple')

plt.ylabel('Phase-resetting index')
plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.savefig('evoked_induced_itc_odor.pdf')

#%% plot envelopes and odor identities and concentrations

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

beta_odor_concentration_animals = np.array(beta_odor_concentration_animals)
gamma_odor_concentration_animals = np.array(gamma_odor_concentration_animals)

plt.figure(dpi = 300, figsize = (5,3))
plt.subplot(121)

plt.boxplot(np.mean(beta_odor_concentration_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.plot([1,2,3,4],np.median(np.mean(beta_odor_concentration_animals[:,:,0:1000],axis = 2),axis = 0), '-o',color = 'black', linewidth = 2)
plt.xticks(ticks = np.arange(1,5), labels = [0.03,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Envelope Amplitude (a.u.)')
plt.title('Beta')
plt.ylim([80,420])

plt.subplot(122)

plt.boxplot(np.mean(gamma_odor_concentration_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.plot([1,2,3,4],np.median(np.mean(gamma_odor_concentration_animals[:,:,0:1000],axis = 2),axis = 0), '-o',color = 'black', linewidth = 2)
plt.xticks(ticks = np.arange(1,5), labels = [0.03,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.title('Gamma')
plt.ylim([80,420])

plt.tight_layout()

plt.savefig('odor_concentration_beta_gamma.pdf')

#%%

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

beta_odor_identity_animals = np.array(beta_odor_identity_animals)
gamma_odor_identity_animals = np.array(gamma_odor_identity_animals)

plt.figure(dpi = 300, figsize = (5,3))
plt.subplot(121)

plt.boxplot(np.mean(beta_odor_identity_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.xticks(ticks = np.arange(1,7), labels = ['eth but','2-hex','iso','hex','eth tig','eth ace'], rotation = 45)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Envelope Amplitude (a.u.)')
plt.title('Beta')
plt.ylim([80,800])

plt.subplot(122)

plt.boxplot(np.mean(gamma_odor_identity_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.xticks(ticks = np.arange(1,7), labels = ['eth but','2-hex','iso','hex','eth tig','eth ace'], rotation = 45)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.title('Gamma')
plt.ylim([80,350])

plt.tight_layout()

plt.savefig('odor_identity_beta_gamma.pdf')

#%% plot telc results

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

beta_envelope_telc_animals = np.array(beta_envelope_telc_animals)
gamma_envelope_telc_animals = np.array(gamma_envelope_telc_animals)

beta_envelope_telc_contra_animals = np.array(beta_envelope_telc_contra_animals)
gamma_envelope_telc_contra_animals = np.array(gamma_envelope_telc_contra_animals)

mean_gamma_telc_odor = np.mean(gamma_envelope_telc_animals,axis = 0)
error_gamma_telc_odor = np.std(gamma_envelope_telc_animals,axis = 0)/np.sqrt(6)

mean_gamma_telc_contra_odor = np.mean(gamma_envelope_telc_contra_animals,axis = 0)
error_gamma_telc_contra_odor = np.std(gamma_envelope_telc_contra_animals,axis = 0)/np.sqrt(6)

mean_beta_telc_odor = np.mean(beta_envelope_telc_animals,axis = 0)
error_beta_telc_odor = np.std(beta_envelope_telc_animals,axis = 0)/np.sqrt(6)

mean_beta_telc_contra_odor = np.mean(beta_envelope_telc_contra_animals,axis = 0)
error_beta_telc_contra_odor = np.std(beta_envelope_telc_contra_animals,axis = 0)/np.sqrt(6)


plt.figure(dpi = 300, figsize = (6,3))

plt.subplot(122)

plt.plot(np.linspace(0,1000,2000),mean_gamma_telc_odor, label = 'TeLC Ipsi', color = 'tab:red')
plt.fill_between(np.linspace(0,1000,2000), mean_gamma_telc_odor-error_gamma_telc_odor, mean_gamma_telc_odor+error_gamma_telc_odor, alpha = 0.2, color = 'tab:red')

plt.plot(np.linspace(0,1000,2000),mean_gamma_telc_contra_odor, label = 'TeLC Contra', color = 'tab:blue')
plt.fill_between(np.linspace(0,1000,2000), mean_gamma_telc_contra_odor-error_gamma_telc_contra_odor, mean_gamma_telc_contra_odor+error_gamma_telc_contra_odor, alpha = 0.2, color = 'tab:blue')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,500])
plt.title('Gamma')
plt.xlabel('Time from inh start (ms)')

plt.subplot(121)

plt.plot(np.linspace(0,1000,2000),mean_beta_telc_odor, label = 'TeLC Ipsi', color = 'tab:red')
plt.fill_between(np.linspace(0,1000,2000), mean_beta_telc_odor-error_beta_telc_odor, mean_beta_telc_odor+error_beta_telc_odor, alpha = 0.2, color = 'tab:red')

plt.plot(np.linspace(0,1000,2000),mean_beta_telc_contra_odor, label = 'TeLC Contra', color = 'tab:blue')
plt.fill_between(np.linspace(0,1000,2000), mean_beta_telc_contra_odor-error_beta_telc_contra_odor, mean_beta_telc_contra_odor+error_beta_telc_contra_odor, alpha = 0.2, color = 'tab:blue')

plt.title('Beta')
plt.legend()
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlim([0,500])
#plt.ylim([50,400])
plt.ylabel('Envelope amplitude (a.u.)')
plt.xlabel('Time from inh start (ms)')

plt.savefig('TeLC_odor.pdf')

#%%

os.chdir(directory)

beta_conc = np.load('Figure_4.npz')['beta_odor_concentration_animals']
gamma_conc = np.load('Figure_4.npz')['gamma_odor_concentration_animals']

beta_odor_concentration_animals = np.array(beta_conc)
gamma_odor_concentration_animals = np.array(gamma_conc)

plt.figure(dpi = 300, figsize = (5,3))
plt.subplot(121)

plt.boxplot(np.mean(beta_odor_concentration_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.plot([1,2,3,4],np.median(np.mean(beta_odor_concentration_animals[:,:,0:1000],axis = 2),axis = 0), '-o',color = 'black', linewidth = 2)
plt.xticks(ticks = np.arange(1,5), labels = [0.03,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.ylabel('Envelope Amplitude (a.u.)')
plt.title('Beta')
plt.ylim([80,420])

plt.subplot(122)

plt.boxplot(np.mean(gamma_odor_concentration_animals[:,:,0:1000],axis = 2), showfliers = False)
plt.plot([1,2,3,4],np.median(np.mean(gamma_odor_concentration_animals[:,:,0:1000],axis = 2),axis = 0), '-o',color = 'black', linewidth = 2)
plt.xticks(ticks = np.arange(1,5), labels = [0.03,0.1,0.3,1])
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.title('Gamma')
plt.ylim([80,420])

plt.tight_layout()



#%%

from statsmodels.stats.anova import AnovaRM

mean_beta_conc = np.mean(beta_odor_concentration_animals[:,:,0:1000],axis = 2)
mean_gamma_conc = np.mean(gamma_odor_concentration_animals[:,:,0:1000],axis = 2)

amp_data_beta = np.hstack([mean_beta_conc[:,0],mean_beta_conc[:,1],mean_beta_conc[:,2],mean_beta_conc[:,3]])
amp_data_gamma = np.hstack([mean_gamma_conc[:,0],mean_gamma_conc[:,1],mean_gamma_conc[:,2],mean_gamma_conc[:,3]])

number = np.tile([0,1,2,3,4],4)
conc = np.repeat([0,1,2,3],5)

dataframe_beta = pd.DataFrame({'mice':number,'conc':conc,'amp':amp_data_beta})
res_beta = AnovaRM(dataframe_beta, depvar='amp',subject='mice', within=['conc']).fit()

dataframe_gamma = pd.DataFrame({'mice':number,'conc':conc,'amp':amp_data_gamma})
res_gamma = AnovaRM(dataframe_gamma, depvar='amp',subject='mice', within=['conc']).fit()


#%% plot gamma beta example 

names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)

loading = exp_data[3][1:14]

    
name = names[1]
os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')

lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
srate = 2000

os.chdir(directory+'/Simul/processed/'+name)
inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
inh_start = np.squeeze(inh_start)
    
odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 

odor_times = odor_data[odor_series]
odor_times_srate = odor_times*srate   
odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate,axis = 1))
    
gamma_filtered = eegfilt(lfp[17,:],srate, 30,60)
beta_filtered = eegfilt(lfp[17,:],srate, 10,20)


gamma_filtered_inh = []
beta_filtered_inh = []
for x in odor_times_srate:
    if gamma_filtered[int(x):int(int(x)+2000)].shape[0] == 2000:
        gamma_filtered_inh.append(gamma_filtered[int(x):int(int(x)+2000)]) 
        beta_filtered_inh.append(beta_filtered[int(x):int(int(x)+2000)]) 
            

window = 5

plt.figure(dpi = 300, figsize = (3,4))
plt.plot(np.linspace(0,1000,2000),gamma_filtered_inh[window], color = 'tab:purple')
plt.plot(np.linspace(0,1000,2000),beta_filtered_inh[window]+1200, color = 'tab:green')
plt.box()
plt.yticks([])
plt.xticks([])
plt.xlim([0,500])

plt.hlines(-1000,0,200, color = 'black')

plt.savefig('beta_gamma_example_filtered.pdf')
