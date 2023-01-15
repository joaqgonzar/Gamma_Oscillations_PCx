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


def phase_amp_hist(amp,fase_lenta,numbin):
    
    position=np.zeros(numbin) # this variable will get the beginning (not the center) of each phase bin (in rads)
    winsize = 2*np.pi/numbin # bin de fase
    
    position = []
    for j in np.arange(1,numbin+1):
        position.append(-np.pi+(j-1)*winsize)
        

    nbin=numbin 
    mean_amp = []
    for j in np.arange(0,nbin):  
        boolean_array = np.logical_and(fase_lenta >=  position[j], fase_lenta < position[j]+winsize)
        I = np.where(boolean_array)[0]
        mean_amp.append(np.mean(amp[I]))
        
    mean_amp = [x for x in mean_amp if str(x) != 'nan']
 
    return(mean_amp)

def pac_histograms(spikes_conv,resp,lenta_BandWidth,srate,numbin):
        
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(resp,srate,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
    
    hist_freqs = phase_amp_hist(spikes_conv,faselenta,numbin)
    
    return(hist_freqs)   

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

pac_exc_odors_animals = []
pac_exc_no_odors_animals = []

pac_gamma_animals = []

pac_exc_odors_first_animals = []
pac_exc_odors_last_animals = []

mvl_odors_animals = []
mvl_no_odors_animals = []


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
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    resp = resp[0:lfp.shape[1]]
    
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
   
    
    # get mask variable for resp coupling analyses
    
    odor_onset_awake = odor_onset_srate[odor_onset_srate<lfp.shape[1]-2000]
    odor_times = []
    for x in odor_onset_awake:
        odor_times.append(np.arange(int(x),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask_odors = np.intersect1d(odor_times,np.arange(0,lfp.shape[1]))
    lfp_mask_no_odors = np.setxor1d(odor_times,np.arange(0,lfp.shape[1]))
    
    
    
    
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
    
    
    
    
    # choose inhalation with smell
    
    inh_start = inh_smell
    
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
    exc_spikes_aniamls_odor.append(trig_spikes_exc)
    
    if len(fb_spikes_inh)>0:
        trig_spikes_fb = np.mean(fb_spikes_inh,axis = 0)
        fb_spikes_aniamls_odor.append(trig_spikes_fb)
         
    if len(ff_spikes_inh)>0:
        trig_spikes_ff = np.mean(ff_spikes_inh,axis = 0)
        ff_spikes_aniamls_odor.append(trig_spikes_ff)
        
        
    # get same lenght of odor and odorless cycles    
    
    lenght_odor = lfp_mask_odors.shape[0]
    lfp_mask_no_odors = lfp_mask_no_odors[-lenght_odor:]
    
    # get spike-phase histograms

    numbin = 10
    numbin_phase_diff = 50
    lenta_BandWidth = 2
    Pf1 = 1    
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(np.squeeze(resp),srate,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)
    faselenta_odors = faselenta[lfp_mask_odors]
    faselenta_no_odors = faselenta[lfp_mask_no_odors]
    
    # divide into thirds
    length_odor_series = lfp_mask_odors.shape[0]
    length_odor_third = np.floor(length_odor_series/3)
    
    mask_first = lfp_mask_odors[0:int(length_odor_third)]
    mask_last = lfp_mask_odors[-int(length_odor_third):]
    
    faselenta_first = faselenta[mask_first]
    faselenta_last = faselenta[mask_last]
    
    pac_exc_odor = []
    pac_exc_no_odor = []
    
    pac_exc_odor_first_third = []
    pac_exc_odor_last_third = []
    
    for x in range(exc_neurons_session.shape[0]):
        pac_exc_odor.append(phase_amp_hist(exc_neurons_session[x,lfp_mask_odors],faselenta_odors,numbin))    
        pac_exc_no_odor.append(phase_amp_hist(exc_neurons_session[x,lfp_mask_no_odors],faselenta_no_odors,numbin))    
        pac_exc_odor_first_third.append(phase_amp_hist(exc_neurons_session[x,mask_first],faselenta_first,numbin_phase_diff))    
        pac_exc_odor_last_third.append(phase_amp_hist(exc_neurons_session[x,mask_last],faselenta_last,numbin_phase_diff))    

    electrode = 28
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp[electrode,:], srate, 30, 60)))
    gamma_envelope = gamma_envelope[lfp_mask_odors]
    pac_gamma = phase_amp_hist(gamma_envelope,faselenta_odors,numbin)

    pac_gamma_animals.append(pac_gamma)
    pac_exc_odors_animals.append(pac_exc_odor)
    pac_exc_no_odors_animals.append(pac_exc_no_odor)
    
    pac_exc_odors_first_animals.append(pac_exc_odor_first_third)
    pac_exc_odors_last_animals.append(pac_exc_odor_last_third)

    # check gamma phase mean vector length 
    
    gamma_phase = np.angle(signal.hilbert(eegfilt(lfp[electrode,:], srate, 30, 60)))
    gamma_phase_odor = gamma_phase[lfp_mask_odors]
    gamma_phase_no_odor = gamma_phase[lfp_mask_no_odors]
    
    mvl_odors = []
    mvl_no_odors = []
    
    for x in range(exc_neurons_session.shape[0]):
        spikes_phases_odor = gamma_phase_odor[units_exc[x,lfp_mask_odors]>0]
        spikes_phases_no_odor = gamma_phase_no_odor[units_exc[x,lfp_mask_no_odors]>0]
        
        mvl_odors.append(np.abs(np.mean(np.exp(1j*spikes_phases_odor))))
        mvl_no_odors.append(np.abs(np.mean(np.exp(1j*spikes_phases_no_odor))))

    mvl_odors_animals.append(mvl_odors)
    mvl_no_odors_animals.append(mvl_no_odors)
    
#%%

os.chdir(directory)

np.savez('Figure_5_1.npz', mvl_odors_animals = mvl_odors_animals, mvl_no_odors_animals = mvl_no_odors_animals, 
         pac_gamma_animals = pac_gamma_animals, pac_exc_odors_animals = pac_exc_odors_animals, pac_exc_no_odors_animals= pac_exc_no_odors_animals,
         pac_exc_odors_first_animals = pac_exc_odors_first_animals, pac_exc_odors_last_animals= pac_exc_odors_last_animals,
         exc_spikes_aniamls_odor = exc_spikes_aniamls_odor, fb_spikes_aniamls_odor = fb_spikes_aniamls_odor,
         ff_spikes_aniamls_odor = ff_spikes_aniamls_odor, exc_spikes_aniamls_no_odor = exc_spikes_aniamls_no_odor,
         fb_spikes_aniamls_no_odor = fb_spikes_aniamls_no_odor, ff_spikes_aniamls_no_odor = ff_spikes_aniamls_no_odor)

#%% plot firing rates 

exc_spikes_inh = exc_spikes_aniamls_no_odor
ff_spikes_inh = ff_spikes_aniamls_no_odor
fb_spikes_inh = fb_spikes_aniamls_no_odor

exc_spikes_smell = exc_spikes_aniamls_odor
ff_spikes__smell = ff_spikes_aniamls_odor
fb_spikes_smell = fb_spikes_aniamls_odor

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from matplotlib import gridspec

plt.figure(dpi = 300, figsize = (3,9))
gs = gridspec.GridSpec(4, 1, height_ratios=[2,1,2,1]) 

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
mask_ff = np.mean(ff_spikes_true[:,0:2000],axis = 1)>0.5
ff_spikes_true = ff_spikes_true[mask_ff,:]

mean_exc_smell = np.mean(exc_spikes_aniamls_conc,axis = 0)
mean_fb_inh = np.mean(fb_spikes_true,axis = 0)
mean_ff_inh = np.mean(ff_spikes_true,axis = 0)

error_exc_smell = 0.5*np.std(exc_spikes_aniamls_conc,axis = 0)/np.sqrt(exc_spikes_aniamls_conc.shape[0])
error_fb_inh = 0.5*np.std(fb_spikes_true,axis = 0)/np.sqrt(len(fb_spikes_true))
error_ff_inh = 0.5*np.std(ff_spikes_true,axis = 0)/np.sqrt(len(ff_spikes_true))


plt.subplot(gs[1])
plt.plot(mean_exc_smell, label = 'EXC', color = 'black')
plt.fill_between(np.arange(0,2000), mean_exc_smell-error_exc_smell, mean_exc_smell+error_exc_smell, alpha = 0.2, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from inh start (ms)')
plt.xlim([0,1000])    
#plt.legend()
plt.xticks(ticks = np.arange(0,1050,200), labels = np.arange(0,550,100))
plt.yticks(ticks = np.arange(2,5,1), labels =np.arange(2,5,1))
plt.ylim([2,4.8])
plt.ylabel('Firing Rate (Hz)')

plt.subplot(gs[0])
plt.plot(mean_fb_inh, label = 'FBI', color = 'lightseagreen')
plt.fill_between(np.arange(0,2000), mean_fb_inh-error_fb_inh, mean_fb_inh+error_fb_inh, alpha = 0.2,  color = 'lightseagreen')
plt.plot(mean_ff_inh, label = 'FFI', color = 'deeppink')
plt.fill_between(np.arange(0,2000), mean_ff_inh-error_ff_inh, mean_ff_inh+error_ff_inh, alpha = 0.2, color = 'deeppink')
#plt.legend()
plt.yticks(ticks = np.arange(5,32,5), labels = np.arange(5,32,5))
plt.xticks(ticks = np.arange(0,1050,200), labels = [])
plt.ylim([3,31])
plt.xlim([0,1000])      
plt.ylabel('Firing Rate (Hz)')

plt.title('Odorless cycles')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')


exc_spikes_aniamls_conc = np.concatenate(exc_spikes_smell)

fb_spikes_true = []

for x in range(len(fb_spikes_smell)):
    if np.sum(fb_spikes_smell[x]) > 0:
       fb_spikes_true.append(fb_spikes_smell[x])
       
fb_spikes_true = np.concatenate(fb_spikes_true)


fb_spikes_true = fb_spikes_true[np.sum(fb_spikes_true,axis = 1) > 0,:]

ff_spikes_true = []

for x in range(len(ff_spikes__smell)):
    if np.sum(ff_spikes__smell[x]) > 0:
       ff_spikes_true.append(ff_spikes__smell[x])    

ff_spikes_true = np.concatenate(ff_spikes_true)
ff_spikes_true = ff_spikes_true[np.sum(ff_spikes_true,axis = 1) > 0,:]
ff_spikes_true = ff_spikes_true[mask_ff,:]

mean_exc_smell = np.mean(exc_spikes_aniamls_conc,axis = 0)
mean_fb_inh = np.mean(fb_spikes_true,axis = 0)
mean_ff_inh = np.mean(ff_spikes_true,axis = 0)

error_exc_smell = 0.5*np.std(exc_spikes_aniamls_conc,axis = 0)/np.sqrt(exc_spikes_aniamls_conc.shape[0])
error_fb_inh = 0.5*np.std(fb_spikes_true,axis = 0)/np.sqrt(len(fb_spikes_true))
error_ff_inh = 0.5*np.std(ff_spikes_true,axis = 0)/np.sqrt(len(ff_spikes_true))


plt.subplot(gs[3])
plt.plot(mean_exc_smell, label = 'EXC', color = 'black')
plt.fill_between(np.arange(0,2000), mean_exc_smell-error_exc_smell, mean_exc_smell+error_exc_smell, alpha = 0.2, color = 'black')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xlabel('Time from inh start (ms)')
plt.xlim([0,1000])    
plt.legend()
plt.xticks(ticks = np.arange(0,1050,200), labels = np.arange(0,550,100))
plt.ylim([2,4.8])
plt.yticks(ticks = np.arange(2,5,1), labels =np.arange(2,5,1))
plt.ylabel('Firing Rate (Hz)')

plt.subplot(gs[2])
plt.plot(mean_fb_inh, label = 'FBI', color = 'lightseagreen')
plt.fill_between(np.arange(0,2000), mean_fb_inh-error_fb_inh, mean_fb_inh+error_fb_inh, alpha = 0.2, color = 'lightseagreen')
plt.plot(mean_ff_inh, label = 'FFI', color = 'deeppink')
plt.fill_between(np.arange(0,2000), mean_ff_inh-error_ff_inh, mean_ff_inh+error_ff_inh, alpha = 0.2, color = 'deeppink')
plt.legend()
plt.xticks(ticks = np.arange(0,1050,200), labels = [])
plt.yticks(ticks = np.arange(5,32,5), labels = np.arange(5,32,5))
plt.ylim([3,31])
plt.xlim([0,1000])      
plt.ylabel('Firing Rate (Hz)')

plt.title('Odor cycles')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
    
plt.tight_layout()
plt.savefig('firing_rates_odor_no_odor.pdf')
    

#%% plot spike-resp histograms

pac_exc_animals = np.array(pac_exc_odors_animals)
pac_exc_animals_conc = np.concatenate(pac_exc_animals,axis = 0)


pac_exc_norm = pac_exc_animals_conc/np.sum(pac_exc_animals_conc,axis = 1)[:,np.newaxis]
sort_indexes = np.argsort(np.sum(pac_exc_norm[:,0:5],axis = 1))
data_spikes_plot = np.hstack([pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:]])

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (6,15))

from matplotlib.gridspec import GridSpec

norm_gamma_animals = pac_gamma_animals/np.sum(pac_gamma_animals,axis = 1)[:,np.newaxis]
mean_gamma = np.mean(norm_gamma_animals, axis = 0)
error_gamma = np.std(norm_gamma_animals, axis = 0)/np.sqrt(15)

mean_gamma = np.hstack([mean_gamma,mean_gamma,mean_gamma])
error_gamma = np.hstack([error_gamma,error_gamma,error_gamma])


gs = GridSpec(2, 1, height_ratios=[2, 10],hspace = 0.05)

plt.subplot(gs[0])

plt.plot(np.arange(0,numbin*3),mean_gamma)
plt.fill_between(np.arange(0,numbin*3),mean_gamma-error_gamma,mean_gamma+error_gamma,alpha = 0.3)
plt.xticks(ticks = np.arange(7,28,2),labels = [])
plt.xlim([7,27]) 
plt.ylabel('Gamma Power')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.8, axis = 'both', which = 'Both')

plt.subplot(gs[1])

plt.imshow(data_spikes_plot,aspect = 'auto',interpolation=None,cmap = 'viridis', vmin = 0.03, vmax = 0.18)
plt.colorbar(orientation="horizontal",label = 'Norm Firing Rate', pad = 0.05)
plt.xlabel('Resp Phase (deg)')
plt.ylabel('Principal Neurons (sorted by initial firing)')
   
plt.xticks(ticks = np.arange(7,28,2),labels = np.round(np.arange(0,730,72)).astype(int),rotation = 30)
plt.xlim([7,27]) 

plt.savefig('norm_firing_rates_all_neurons_odor.pdf')
#%%
numbin = 10 

pac_exc_animals = np.array(pac_exc_odors_animals)
pac_exc_animals_conc = np.concatenate(pac_exc_animals,axis = 0)
pac_exc_norm = pac_exc_animals_conc/np.sum(pac_exc_animals_conc,axis = 1)[:,np.newaxis]

max_entrop = np.log(numbin)

mi_exc_odors = []

for x in range(pac_exc_norm.shape[0]):
    
    data = pac_exc_norm[x,:]
    non_zero = data[data>0]
    entrop = -1*np.sum(non_zero*np.log(non_zero))
    mi_exc_odors.append((max_entrop-entrop)/max_entrop)
    
pac_exc_animals = np.array(pac_exc_no_odors_animals)
pac_exc_animals_conc = np.concatenate(pac_exc_animals,axis = 0)
pac_exc_norm = pac_exc_animals_conc/np.sum(pac_exc_animals_conc,axis = 1)[:,np.newaxis]

max_entrop = np.log(numbin)

mi_exc_no_odors = []

for x in range(pac_exc_norm.shape[0]):
    
    data = pac_exc_norm[x,:]
    non_zero = data[data>0]
    entrop = -1*np.sum(non_zero*np.log(non_zero))
    mi_exc_no_odors.append((max_entrop-entrop)/max_entrop)
    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,8))

plt.subplot(312)

plt.scatter(mi_exc_no_odors,mi_exc_odors,s = 5,c = 'black', alpha = 0.3)
plt.yscale('log')
plt.xscale('log')
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))
plt.ylim([5e-4,1.2])
plt.xlim([5e-4,1.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.4, axis = 'both', which = 'Both')
plt.ylabel('Mod Index (odor cycles)')
plt.xlabel('Mod Index (odorless cycles)')
plt.text(0.0007,0.35,'p=0.71')

s_pac,p_pac = stats.ttest_rel(mi_exc_no_odors,mi_exc_odors,alternative = 'greater')
df_pac = len(mi_exc_no_odors)-1

plt.subplot(313)

mvl_odors = np.concatenate(mvl_odors_animals,axis = 0)
mvl_no_odors = np.concatenate(mvl_no_odors_animals,axis = 0)


plt.scatter(mvl_no_odors,mvl_odors,s = 5,alpha = 0.3,c = 'black')
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))
plt.ylim([5e-4,1])
plt.xlim([5e-4,1])

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.4, axis = 'both', which = 'Both')
plt.ylabel('Spike-gamma coupling (odor cycles)')
plt.xlabel('Spike-gamma coupling (odorless cycles)')
plt.yscale('log')
plt.xscale('log')
plt.text(0.0007,0.35,'p<1x10^-19')

s_mvl,p_mvl = stats.ttest_rel(mvl_odors,mvl_no_odors,alternative = 'greater',nan_policy='omit')
df_mvl = len(mvl_odors)-1


pac_exc_odor_first_third = np.concatenate(pac_exc_odors_first_animals)
pac_exc_odor_last_third = np.concatenate(pac_exc_odors_last_animals)

phases = np.tile(np.linspace(0,360,100),(858,1))
phases = np.linspace(0,2*np.pi,100)


pref_phase_first = phases[np.argmax(pac_exc_odor_first_third,axis = 1)]
pref_phase_last = phases[np.argmax(pac_exc_odor_last_third,axis = 1)]

phase_diff = np.angle(np.exp(1j*(pref_phase_first-pref_phase_last)))

plt.subplot(311)

plt.hist(np.rad2deg(phase_diff),bins = 25, color = 'black')

plt.ylabel('# of principal cells')
plt.xlabel('Preferred resp phase difference \n first third - last third of the recording')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.4, axis = 'both', which = 'Both')


plt.tight_layout()
#plt.savefig('stats_firing_odors.pdf')

s_first,p_first = stats.ttest_rel(pref_phase_first,pref_phase_last,alternative = 'greater',nan_policy='omit')
df_first = len(pref_phase_last)-1

#%%

pac_exc_animals = np.array(pac_exc_odors_animals)
pac_exc_animals_conc = np.concatenate(pac_exc_animals,axis = 0)


pac_exc_norm = pac_exc_animals_conc/np.sum(pac_exc_animals_conc,axis = 1)[:,np.newaxis]
sort_indexes = np.argsort(np.sum(pac_exc_norm[:,0:5],axis = 1))
data_spikes_plot = np.hstack([pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:]])

plt.subplot(122)

plt.imshow(data_spikes_plot,aspect = 'auto',interpolation=None,cmap = 'viridis', vmin = 0.03, vmax = 0.15)
plt.colorbar(orientation="horizontal",label = 'Norm Firing Rate', pad = 0.05)
plt.xlabel('Resp Phase (deg)')
plt.ylabel('Principal Neurons (sorted by initial firing)')
   
plt.xticks(ticks = np.arange(7,28,2),labels = np.round(np.arange(0,730,72)).astype(int),rotation = 30)
plt.xlim([7,27]) 

pac_exc_animals = np.array(pac_exc_no_odors_animals)
pac_exc_animals_conc = np.concatenate(pac_exc_animals,axis = 0)


pac_exc_norm = pac_exc_animals_conc/np.sum(pac_exc_animals_conc,axis = 1)[:,np.newaxis]
sort_indexes = np.argsort(np.sum(pac_exc_norm[:,0:5],axis = 1))
data_spikes_plot = np.hstack([pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:],pac_exc_norm[sort_indexes,:]])

plt.subplot(121)

plt.imshow(data_spikes_plot,aspect = 'auto',interpolation=None,cmap = 'viridis', vmin = 0.03, vmax = 0.15)
plt.colorbar(orientation="horizontal",label = 'Norm Firing Rate', pad = 0.05)
plt.xlabel('Resp Phase (deg)')
plt.ylabel('Principal Neurons (sorted by initial firing)')
   
plt.xticks(ticks = np.arange(7,28,2),labels = np.round(np.arange(0,730,72)).astype(int),rotation = 30)
plt.xlim([7,27]) 



