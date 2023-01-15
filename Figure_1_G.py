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
import mat73
from scipy import signal
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

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


def spike_triggered_avg(units, gamma_envelope):
    
    gamma_trig_units = []
    
    for x in range(units.shape[0]):
        
        spikes = np.where(units[x,:]>0)[0] 
        
        gamma_trig = []
        for spike in spikes:
            if len(gamma_envelope[int(spike-1000):int(spike+2000)]) == 3000:
                gamma_trig.append(gamma_envelope[int(spike-1000):int(spike+2000)])
        
        if len(gamma_trig)>0:
            gamma_trig_units.append(np.mean(gamma_trig,axis = 0))
     
    return(gamma_trig_units)      
 
#%% experiment data 

names = ['160403-1','160403-2','160406-2','160409-1','160409-2','160422-1','160422-2','160423-1','160423-2','160425-1','160428-1','160428-2','160613-1','160623-1','160623-2']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory+'/VGAT')

exp_data = pd.read_csv('ExperimentCatalog_VGAT.txt', sep=" ", header=None)

loading = np.array(exp_data[3][1:16])

#%% loop through animals


exc_spikes_aniamls_no_odor = []
ff_spikes_aniamls_no_odor = []
fb_spikes_aniamls_no_odor = []

exc_spikes_aniamls_odor = []
ff_spikes_aniamls_odor = []
fb_spikes_aniamls_odor = []

spec_aniamls_no_odor = []
spec_aniamls_odor = []

gc_exc_gamma = []
gc_fb_gamma = []
gc_ff_gamma = []

gamma_trig_fb = []
gamma_trig_ff = []
gamma_trig_exc = []

for index, name in enumerate(names):
    
    print(name)

    os.chdir(directory+'/VGAT/Decimated_LFPs')

    lfp = np.load('PFx_VGAT_lfp_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
    srate = 2000
    
    os.chdir(directory+'/VGAT/processed/'+name)
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate
    inh_start = np.squeeze(inh_start)
    inh_start = inh_start[inh_start<lfp.shape[1]]
    positions = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['Wave']
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
    elif loading[index] == 'B':
        odor_series = list(np.array([4,7,8,12,15,16])-1)
        
    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate   
    odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate))
    
    
    # check VGAT+ neurons
    laser_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesDuringLaser'][0][0]
    before_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesBeforeLaser'][0][0]
    inh_laser = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserTimes'][0][0][0]['PREXIndex'][0][0]
    inh_nonlaser = np.delete(inh_start, inh_laser[0][:,0:20][0])
    
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
        
    vgat_neg_pos = np.mean(y_position_neg)
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
        
    
    # get ff interneurons    
    
    ff_interneurons = vgats[:,vgats[1,:]<vgat_neg_pos-70]
    ff_spikes = []
    for x in ff_interneurons[0]:
        ff_spikes.append(spike_times[int(x)][0])
        
    ff_neurons_session = [] 
    if len(ff_spikes)>0:    
       [ff_neurons_session,units_ff] = spike_rate(ff_spikes)
       ff_neurons_session = np.array(ff_neurons_session)[:,0:lfp.shape[1]]
       units_ff = np.array(units_ff)[:,0:lfp.shape[1]]
            
    
    # get fb interneurons    
    
    fb_interneurons = vgats[:,vgats[1,:]>vgat_neg_pos]
    fb_spikes = []
    for x in fb_interneurons[0]:
        fb_spikes.append(spike_times[int(x)][0])
        
    fb_neurons_session = []    
    if len(fb_spikes)>0:    
        [fb_neurons_session,units_fb] = spike_rate(fb_spikes)
        fb_neurons_session = np.array(fb_neurons_session)[:,0:lfp.shape[1]]
        units_fb = np.array(units_fb)[:,0:lfp.shape[1]]    

    
    
    # get inhalation triggered responses
    
    # get all inhalations with odor
    
    odor_onset = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].FVSwitchTimesOn[0]
    odor_onset_times = odor_onset[odor_series]
    odor_onset_srate = odor_onset_times*srate   
    odor_onset_srate = np.matrix.flatten(np.concatenate(odor_onset_srate,axis = 1))
    
    inh_odor = []
    odorants_repeat = []
    
    for index_odor,x in enumerate(odor_onset_srate):
        index_smells = np.logical_and((inh_start>=x),(inh_start<x+2000))
        inh_odor.append(inh_start[index_smells])
            
    inh_smell = np.concatenate(inh_odor,axis = 0)

    # choose inhalation without smell
    
    inh_no_odor = np.setxor1d(inh_smell,inh_nonlaser)
    inh_start = inh_no_odor
    
    exc_spikes_inh = []
    ff_spikes_inh = []
    fb_spikes_inh = []
    
    # in case there are no ff or fb neurons
    if len(ff_neurons_session) == 0:
        ff_neurons_session = np.empty((1,1))    
    if len(fb_neurons_session) == 0:
        fb_neurons_session = np.empty((1,1))        
        
    # get inhalation triggered firing rates
    
    for x in inh_start:
        if len(exc_neurons_session[0,int(x):int(x+2000)]) == 2000:
            exc_spikes_inh.append(exc_neurons_session[:,int(x):int(x+2000)])
        if len(fb_neurons_session[0,int(x):int(x+2000)]) == 2000:
            fb_spikes_inh.append(fb_neurons_session[:,int(x):int(x+2000)])
        if len(ff_neurons_session[0,int(x):int(x+2000)]) == 2000:
            ff_spikes_inh.append(ff_neurons_session[:,int(x):int(x+2000)])
    
    # average the inhalation triggered rates
    
    trig_spikes_exc = np.mean(exc_spikes_inh,axis = 0)
    exc_spikes_aniamls_no_odor.append(trig_spikes_exc)
    
    if len(fb_spikes_inh)>0:
        trig_spikes_fb = np.mean(fb_spikes_inh,axis = 0)
        fb_spikes_aniamls_no_odor.append(trig_spikes_fb)
         
    if len(ff_spikes_inh)>0:
        trig_spikes_ff = np.mean(ff_spikes_inh,axis = 0)
        ff_spikes_aniamls_no_odor.append(trig_spikes_ff)

    

    # get inhalation triggered spectrogram
    
    lfp_inh_no_odor = []
    for x in inh_no_odor:
        lfp_inh_no_odor.append(lfp[28,int(x):int(x+1000)])
       
    
    
    trig_spec = []
    gc_exc_gamma = []
    for x in lfp_inh_no_odor[:-1]:
        if len(x) == 1000:
            
            f,t,p = signal.spectrogram(x,fs = srate,window = ('hamming'),nperseg = int(srate/25), noverlap = int(srate/30),nfft = 10*srate)
            trig_spec.append(np.abs(p))
            
            
    
    mean_spec_no_odor = np.mean(trig_spec,axis = 0)
    spec_aniamls_no_odor.append(mean_spec_no_odor)

    

    # get spike triggered gamma averages
    
    # get gamma envelope signal

    gamma = eegfilt(lfp[28,:],srate,30,50)
    gamma_envelope = np.abs(signal.hilbert(gamma))

    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[1]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[0]))
    
    gamma_envelope_odorless = gamma_envelope[lfp_mask]
    
    # get spike triggered averages for each subtype 
    
    if len(exc_spikes)>0:
        units_exc_odorless = units_exc[:,lfp_mask]
        gamma_trig_exc.append(spike_triggered_avg(units_exc_odorless, gamma_envelope_odorless))
    
    if len(fb_spikes)>0:
        units_fb_odorless = units_fb[:,lfp_mask]
        gamma_trig_fb.append(spike_triggered_avg(units_fb_odorless, gamma_envelope_odorless)) 
        
    if len(ff_spikes)>0:
        units_ff_odorless = units_ff[:,lfp_mask]
        gamma_trig_ff.append(spike_triggered_avg(units_ff_odorless, gamma_envelope_odorless)) 
    

#%% save

os.chdir(directory)
np.savez('Figure_1_g.npz', exc_spikes_aniamls_no_odor= exc_spikes_aniamls_no_odor, ff_spikes_aniamls_no_odor=ff_spikes_aniamls_no_odor,fb_spikes_aniamls_no_odor = fb_spikes_aniamls_no_odor, gamma_trig_exc = gamma_trig_exc, gamma_trig_ff = gamma_trig_ff, gamma_trig_fb= gamma_trig_fb, spec_aniamls_no_odor = spec_aniamls_no_odor)


#%%
exc_spikes_inh = exc_spikes_aniamls_no_odor
ff_spikes_inh = ff_spikes_aniamls_no_odor
fb_spikes_inh = fb_spikes_aniamls_no_odor



exc_spikes_aniamls_conc = np.concatenate(exc_spikes_inh)

fb_spikes_true = []

for x in range(len(fb_spikes_inh)):
    if np.sum(fb_spikes_inh[x]) > 0:
       fb_spikes_true.append(fb_spikes_inh[x])
       
fb_spikes_true = np.concatenate(fb_spikes_true)
fb_spikes_true = fb_spikes_true[np.sum(fb_spikes_true,axis = 1) > 0,:]

ff_spikes_true = []

for x in range(len(ff_spikes_inh)):
    if np.sum(ff_spikes_inh[x]) > 0:
       ff_spikes_true.append(ff_spikes_inh[x])    

ff_spikes_true = np.concatenate(ff_spikes_true)
ff_spikes_true = ff_spikes_true[np.sum(ff_spikes_true,axis = 1) > 0,:]

exc_spikes_aniamls_norm = exc_spikes_aniamls_conc/np.mean(exc_spikes_aniamls_conc[:,0:2000],axis = 1)[:,np.newaxis]
fb_spikes_true_norm = fb_spikes_true/np.mean(fb_spikes_true[:,0:2000],axis = 1)[:,np.newaxis]
ff_spikes_true_norm = ff_spikes_true/np.mean(ff_spikes_true[:,0:2000],axis = 1)[:,np.newaxis]
mask_ff = np.mean(ff_spikes_true[:,0:2000],axis = 1)>0.5
ff_spikes_true_norm = ff_spikes_true_norm[mask_ff,:]

mean_exc_smell = np.nanmean(exc_spikes_aniamls_norm,axis = 0)
mean_fb_inh = np.nanmean(fb_spikes_true_norm,axis = 0)
mean_ff_inh = np.nanmean(ff_spikes_true_norm,axis = 0)

error_exc_smell = np.nanstd(exc_spikes_aniamls_norm,axis = 0)/np.sqrt(exc_spikes_aniamls_norm.shape[0])
error_fb_inh = np.nanstd(fb_spikes_true_norm,axis = 0)/np.sqrt(len(fb_spikes_true))
error_ff_inh = np.nanstd(ff_spikes_true_norm,axis = 0)/np.sqrt(len(ff_spikes_true))

gamma_trig_exc = np.concatenate(gamma_trig_exc)
gamma_trig_fb = np.concatenate(gamma_trig_fb)
gamma_trig_ff = np.concatenate(gamma_trig_ff)

gamma_trig_exc_norm = gamma_trig_exc/np.mean(gamma_trig_exc[:,0:1000],axis = 1)[:,np.newaxis]
gamma_trig_fb_norm = gamma_trig_fb/np.mean(gamma_trig_fb[:,0:1000],axis = 1)[:,np.newaxis]
gamma_trig_ff_norm = gamma_trig_ff/np.mean(gamma_trig_ff[:,0:1000],axis = 1)[:,np.newaxis]

mean_trig_fb = np.mean(gamma_trig_fb_norm,axis =0)
error_trig_fb = np.std(gamma_trig_fb_norm,axis =0)/np.sqrt(gamma_trig_fb.shape[0])

mean_trig_exc = np.mean(gamma_trig_exc_norm,axis =0)
error_trig_exc = np.std(gamma_trig_exc_norm,axis =0)/np.sqrt(gamma_trig_exc.shape[0])

mean_trig_ff = np.mean(np.array(gamma_trig_ff_norm[mask_ff,:]),axis =0)
error_trig_ff = np.std(gamma_trig_ff_norm[mask_ff,:],axis =0)/np.sqrt(gamma_trig_ff[mask_ff,:].shape[0])

#%% plot spectrogram

frequency = np.arange(0,1000.1,0.1)


mean_control = np.mean(spec_aniamls_no_odor,axis = 0)
white_spec = mean_control*frequency[:,np.newaxis]


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.gridspec as gridspec


fig = plt.figure(dpi = 300,figsize = (4,7))
gs = gridspec.GridSpec(2, 1,height_ratios = [1,2])

plt.subplot(gs[0])

max_gamma = 70000
min_gamma = 35000

max_gamma = 235000
min_gamma = 120000

extent = [0,500,1000,0] 
plt.imshow(white_spec,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = min_gamma, vmax = max_gamma,aspect = 'auto')
#plt.xlabel('Time from inh start (ms)')    
#plt.ylim([0,1300])
plt.yticks(ticks = np.arange(0,130,10),labels = np.arange(0,130,10))
#plt.xticks(ticks = np.arange(0,white_spec.shape[1]+5,white_spec.shape[1]/5),labels = np.arange(0,550,100))

plt.xlim([0,500])
plt.ylim([0,130])

plt.ylabel('Frequency (Hz)')


ax1 = plt.subplot(gs[1])

plt.plot(mean_ff_inh, label = 'FFI (VGAT+)', color = 'deeppink')
plt.fill_between(np.arange(0,2000), mean_ff_inh-error_ff_inh, mean_ff_inh+error_ff_inh, alpha = 0.2, color = 'deeppink')
plt.plot(mean_exc_smell, label = 'EXC (VGAT-)', color = 'black')
plt.fill_between(np.arange(0,2000), mean_exc_smell-error_exc_smell, mean_exc_smell+error_exc_smell, alpha = 0.2, color = 'black')
plt.plot(mean_fb_inh, label = 'FBI (VGAT+)', color = 'lightseagreen')
plt.fill_between(np.arange(0,2000), mean_fb_inh-error_fb_inh, mean_fb_inh+error_fb_inh, alpha = 0.2, color = 'lightseagreen')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from inh start (ms)')
plt.xlim([0,1000])    
#plt.ylim([0.75,1.6])
plt.legend(loc = 'lower left',fontsize = 7)
plt.xticks(ticks = np.arange(0,1050,200), labels = np.arange(0,550,100))
#plt.ylim([0.7,1.6])
plt.ylabel('Normalized Firing Rate (a.u.)')
    


ax2 = fig.add_axes([0.63,0.42,0.2,0.12])





plt.plot(np.linspace(0,1500,3000),mean_trig_exc, color = 'black', linewidth = 0.8)
plt.fill_between(np.linspace(0,1500,3000), mean_trig_exc-error_trig_exc, mean_trig_exc+error_trig_exc, alpha = 0.2, color = 'black')

plt.plot(np.linspace(0,1500,3000),mean_trig_fb,color = 'lightseagreen', linewidth = 0.8)
plt.fill_between(np.linspace(0,1500,3000), mean_trig_fb-error_trig_fb, mean_trig_fb+error_trig_fb, alpha = 0.2,color = 'lightseagreen')

plt.plot(np.linspace(0,1500,3000),mean_trig_ff,color = 'deeppink', linewidth = 0.8)
plt.fill_between(np.linspace(0,1500,3000), mean_trig_ff-error_trig_ff, mean_trig_ff+error_trig_ff, alpha = 0.2,color = 'deeppink')
plt.vlines(500,0.95,1.08, color = 'black',linestyles = 'dashed')
plt.xlim([100,1200])

#plt.xticks(ticks = np.arange(300,1000,100), labels = np.arange(-100,250,50))
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from spike (ms)',fontsize = 8)
plt.ylabel('Norm g amp (a.u.)',fontsize = 8)

ax2.tick_params(labelsize = 4)

plt.savefig('norm_firing_rate_gamma.pdf')