#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 22:22:49 2022

@author: pcanalisis2
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

ic_one_animals = []
variance_pc_one_animals = [] 
pc_proj_animals = []
winners_losers_animals = []
gamma_envelope_animals = []
fb_activity_animals = []
mvl_wl_animals = []
phase_wl_animals = []

for index, name in enumerate(names):
    
    print(name)

    # lfp
    
    os.chdir(directory+'/VGAT/Decimated_LFPs')
    lfp = np.load('PFx_VGAT_lfp_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    srate = 2000
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp[28,:], srate, 30, 60)))
    gamma_phase = np.angle(signal.hilbert(eegfilt(lfp[28,:],srate, 30,60)))
    
    # load spike times, spike data
    
    os.chdir(directory+'/VGAT/processed/'+name)
    spike_times = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['tsec']
    resp = scipy.io.loadmat(name+'_resp.mat')['RRR']
    inh_start = scipy.io.loadmat(name+'_bank1_efd.mat')['efd']['PREX'][0][0][0]*srate
    inh_start = np.squeeze(inh_start)
    positions = mat73.loadmat(name+'_bank1_st.mat')['SpikeTimes']['Wave']
    
    # get odor data
    
    conc_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    if loading[index] == 'A':
        #odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
        odor_series = list(np.array([4,7,8,12,15,16])-1)
    elif loading[index] == 'B':
        odor_series = list(np.array([4,7,8,12,15,16])-1)
   
    # check VGAT+ neurons
    
    laser_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesDuringLaser'][0][0]
    before_spikes = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserSpikes'][0][0][0]['SpikesBeforeLaser'][0][0]
    inh_laser = scipy.io.loadmat(name+'_bank1_efd.mat',struct_as_record = True)['efd']['LaserTimes'][0][0][0]['PREXIndex'][0][0]
    inh_nonlaser = np.delete(inh_start, inh_laser[0][:,0:20][0])
    
    # get only awake times
    
    inh_start = inh_start[inh_start<lfp.shape[1]]
    
    
    # get excitatory and inhibitory neurons
    
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
    vgat_neg_pos = np.mean(y_position_neg)

    vgats = np.vstack([vgat,y_position])    
    
    # get excitatory neuron matrices 
    
    # get exc neurons 
    
    exc_neurons = vgat_neg
    exc_spikes = []
    for x in exc_neurons:
        exc_spikes.append(spike_times[int(x)][0])
        
    if len(exc_spikes)>0:    
        [exc_neurons_session,units_exc] = spike_rate(exc_spikes)
        exc_neurons_session = np.array(exc_neurons_session)[:,0:lfp.shape[1]]
        units_exc = np.array(units_exc)[:,0:lfp.shape[1]]
        
    # z-score neurons
    
    conv_spikes_norm = stats.zscore(exc_neurons_session,axis = 1)
    
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
        conv_fb_spikes_norm = stats.zscore(fb_neurons_session,axis = 1)
    
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
        #index_smells = inh_start==x
        inh_odor.append(inh_start[index_smells])
        if len(inh_start[index_smells]) > 0:
            num_breaths = len(inh_start[index_smells])
            odorants_repeat.append(np.repeat(odorants[index_odor],num_breaths))
            
    inh_smell = np.concatenate(inh_odor,axis = 0)
    odorants = np.concatenate(odorants_repeat)
    
    # get spiking activity and gamma envelopes for all inhalations
    
    gamma_envelope_norm = stats.zscore(gamma_envelope,axis = 0)
    
    smells_trig = []
    gamma_envelope_odor = []  
    fb_smells_trig = []
    gamma_phase_odor = []
    spikes_trig = []
    
    for x in inh_smell:
        if conv_spikes_norm[:,int(x):int(x+2000)].shape[1] == 2000:
            smells_trig.append(conv_spikes_norm[:,int(x):int(x+2000)])
            spikes_trig.append(units_exc[:,int(x):int(x+2000)])
            
            gamma_envelope_odor.append(gamma_envelope_norm[int(x):int(int(x)+2000)])
            gamma_phase_odor.append(gamma_phase[int(x):int(int(x)+2000)])
            
            if len(fb_spikes)>0: 
                fb_smells_trig.append(conv_fb_spikes_norm[:,int(x):int(x+2000)])
        
    mean_envelope_odor = np.mean(gamma_envelope_odor,axis = 0)
    gamma_phase_odor = np.array(gamma_phase_odor)
    smells_trig = np.array(smells_trig)
    spikes_trig = np.array(spikes_trig)
    
    # do ICA analysis
    
    ic_one_weights = []
    
    winners_time = []
    losers_time = []
    
    winners_losers_mvl = []
    winners_losers_phase = []
    
    # define threshold for winners and losers
    
    winners_threshold = 0.0887137180248916
    losers_threshold = 0.013753072267353993
    
    for odor in np.arange(0,6):
        
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
        
        # get gamma phase-locking for winners and losers
        
        spike_counts_odor = spikes_trig[odor_index,:,:]
        spike_counts_odor = np.concatenate(spike_counts_odor,axis = 1)
        gamma_phase_odor_conc = np.concatenate(gamma_phase_odor[odor_index,:],axis = 0)
        
        winners_spikes = spike_counts_odor[np.squeeze(winners),:]
        losers_spikes = spike_counts_odor[np.squeeze(losers),:]
        
        
        mvl_winners = []
        phase_winners = []
        for x in range(winners_spikes.shape[0]):
            winner_spikes_phases = gamma_phase_odor_conc[winners_spikes[x,:]>0]
            if winner_spikes_phases.shape[0]>0:
                mvl_winners.append(np.abs(np.mean(np.exp(1j*winner_spikes_phases)))) 
                phase_winners.append(stats.circmean(winner_spikes_phases,nan_policy = 'omit'))
        #
        mvl_losers = []
        phase_losers = []
        for x in range(losers_spikes.shape[0]):
            losers_spikes_phases = gamma_phase_odor_conc[losers_spikes[x,:]>0]
            if losers_spikes_phases.shape[0]>0:
                mvl_losers.append(np.abs(np.mean(np.exp(1j*losers_spikes_phases)))) 
                phase_losers.append(stats.circmean(losers_spikes_phases,nan_policy = 'omit'))
        
        
        # save results from odor
        winners_time.append(mean_winners)    
        losers_time.append(mean_losers)    
        ic_one_weights.append(ica_weights)
        
        winners_losers_mvl.append([mvl_winners,mvl_losers])
        winners_losers_phase.append([phase_winners,phase_losers])
        

    # get all winners and losers
    
    winners_all = np.nanmean(np.concatenate(winners_time,axis = 0),axis = 0)
    losers_all = np.nanmean(np.concatenate(losers_time,axis = 0),axis = 0)
    
    # save results
    winners_losers_animals.append([np.concatenate(winners_time,axis = 0),np.concatenate(losers_time,axis = 0)])
    ic_one_animals.append(ic_one_weights)
        
    gamma_envelope_animals.append(mean_envelope_odor)
    fb_activity_animals.append(fb_smells_trig)
    
    winners_losers_mvl = np.array(winners_losers_mvl, dtype = 'object')
    mvl_wl_animals.append(winners_losers_mvl)
    
    winners_losers_phase = np.array(winners_losers_phase, dtype = 'object')
    phase_wl_animals.append(winners_losers_phase)
    
    
#%% save results

os.chdir(directory)

np.savez('Figure_6_1.npz', winners_losers_animals = winners_losers_animals, ic_one_animals = ic_one_animals, gamma_envelope_animals = gamma_envelope_animals, fb_activity_animals = fb_activity_animals, mvl_wl_animals = mvl_wl_animals, phase_wl_animals = phase_wl_animals)


#%% load results
os.chdir(directory)
mvl_wl_animals = np.load('Figure_6_1.npz',allow_pickle=(True))['mvl_wl_animals']
phase_wl_animals = np.load('Figure_6_1.npz',allow_pickle=(True))['phase_wl_animals']
#%% plot results       


from scipy.stats import skewtest
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from scipy.stats import skew

fig = plt.figure(dpi = 300, figsize = (4,4))

ax1 = plt.subplot(111)

ic_one_animals = np.array(ic_one_animals,dtype = 'object')


data_dist = np.squeeze(np.concatenate(ic_one_animals[:,0]))
mean1 = np.mean(data_dist)
sk1_test = skewtest(data_dist)
sk1 = stats.skew(data_dist)

sk1 = []
for x in ic_one_animals[:,0]:
    sk1.append(stats.skew(x)[0])

data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'eth bu')

data_dist = np.squeeze(np.concatenate(ic_one_animals[:,1]))
mean2 = np.mean(data_dist)
sk2_test = skewtest(data_dist)
sk2 = stats.skew(data_dist)

sk2 = []
for x in ic_one_animals[:,1]:
    sk2.append(stats.skew(x)[0])
    
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = '2-hex')

data_dist = np.squeeze(np.concatenate(ic_one_animals[:,2]))
mean3 = np.mean(data_dist)
sk3_test = skewtest(data_dist)
sk3 = stats.skew(data_dist)

sk3 = []
for x in ic_one_animals[:,2]:
    sk3.append(stats.skew(x)[0])
    
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'iso')

data_dist = np.squeeze(np.concatenate(ic_one_animals[:,3]))
mean4 = np.mean(data_dist)
sk4_test = skewtest(data_dist)
sk4 = stats.skew(data_dist)

sk4 = []
for x in ic_one_animals[:,3]:
    sk4.append(stats.skew(x)[0])
    
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'hex')

data_dist = np.squeeze(np.concatenate(ic_one_animals[:,4]))
mean5 = np.mean(data_dist)
sk5_test = skewtest(data_dist)
sk5 = stats.skew(data_dist)

sk5 = []
for x in ic_one_animals[:,4]:
    sk5.append(stats.skew(x)[0])

data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'eth ti')

data_dist = np.squeeze(np.concatenate(ic_one_animals[:,5]))
mean6 = np.mean(data_dist)
sk6_test = skewtest(data_dist)
sk6 = stats.skew(data_dist)

sk6 = []
for x in ic_one_animals[:,5]:
    sk6.append(stats.skew(x)[0])
    
data_dist[data_dist>0.15] = 0.15
data_dist[data_dist<-0.1] = -0.1
sns.kdeplot(data_dist,label = 'eth ace')

#plt.legend(fontsize = 6, ncol = 6,handlelength = 1,handletextpad = 0.5) 

plt.vlines(np.mean([mean1,mean2,mean3,mean4,mean5,mean6]),0,30,linestyle = '--',color = 'black')

plt.xlim([-0.1,0.2])
plt.xlabel('1st IC Weights')
plt.xticks(ticks = np.arange(-0.05,0.20,0.05))

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

ax2 = fig.add_axes([0.6,0.4,0.25,0.4])

plt.boxplot(np.array([sk1,sk2,sk3,sk4,sk5,sk6]).T, showfliers = False)


plt.ylim([-1,8.5])
plt.hlines(0,0.6,6.5,color = 'black', linestyles='dashed')
plt.ylabel('Skewness')
plt.xticks(ticks= np.arange(1,7),labels = ['eth bu','2-hex','iso','hex','eth ti','eth ace'], rotation = 90)
plt.xlim([0,7])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.savefig('ICA_distribution_odors.pdf')


#%% get quartiles for each odor

winners_threshold = []
losers_threshold = []
mean_threshold = []

for x in range(6):
    winners_threshold.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals[:,x])),0.95))
    losers_threshold.append(np.quantile(np.squeeze(np.concatenate(ic_one_animals[:,x])),0.5))
    mean_threshold.append(np.mean(np.squeeze(np.concatenate(ic_one_animals[:,x]))))



winners_losers_animals = np.array(winners_losers_animals)

winners = winners_losers_animals[:,0]
losers = winners_losers_animals[:,1]
winners = np.concatenate(winners)
losers = np.concatenate(losers)

mean_winners = np.mean(winners,axis = 0)
error_winners = np.std(winners,axis = 0)/np.sqrt(winners.shape[0])
mean_losers = np.mean(losers,axis = 0)
error_losers = np.std(losers,axis = 0)/np.sqrt(losers.shape[0])


mean_envelope_odor = np.mean(gamma_envelope_animals,axis = 0)
error_envelope_odor = np.std(gamma_envelope_animals,axis = 0)/np.sqrt(15)


fb_activity = []
for x in fb_activity_animals:
    if len(x)>0:
        fb_activity.append(np.mean(x,axis = 0))
    
fb_activity = np.concatenate(fb_activity,axis = 0)    

mean_fb_odor = np.mean(fb_activity,axis = 0)
error_fb_odor = np.std(fb_activity,axis = 0)/np.sqrt(40)


from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,8))
gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1],hspace = 0.05) 

plt.subplot(gs[0])

plt.plot(np.linspace(0,1000,2000),mean_winners, label = 'Winners',color = 'tab:purple')
plt.fill_between(np.linspace(0,1000,2000), mean_winners-error_winners, mean_winners+error_winners,alpha = 0.2,color = 'tab:purple')

plt.plot(np.linspace(0,1000,2000),mean_losers, label = 'Losers',color = 'tab:green')
plt.fill_between(np.linspace(0,1000,2000), mean_losers-error_losers, mean_losers+error_losers,alpha = 0.2,color = 'tab:green')

plt.plot(np.linspace(0,1000,2000),mean_fb_odor, label = 'FBIs',color = 'tab:cyan')
plt.fill_between(np.linspace(0,1000,2000), mean_fb_odor-error_fb_odor, mean_fb_odor+error_fb_odor,alpha = 0.2,color = 'tab:cyan')

plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(50,1100,100),labels = [], rotation = 45)
plt.legend()

plt.subplot(gs[2])
plt.plot(np.linspace(0,1000,2000),mean_losers, label = 'losers',color = 'tab:green')
plt.fill_between(np.linspace(0,1000,2000), mean_losers-error_losers, mean_losers+error_losers,alpha = 0.2,color = 'tab:green')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.hlines(mean_losers[0],0,1000,color = 'gray',linestyles='dashed',alpha = 0.5)
plt.xticks(ticks = np.arange(50,1100,100),labels = np.arange(50,1100,100), rotation = 45)
plt.xlabel('Time from inhalation start (ms)')

plt.subplot(gs[1])
plt.plot(np.linspace(0,1000,2000),mean_envelope_odor, label = 'gamma',color = 'tab:orange')
plt.fill_between(np.linspace(0,1000,2000), mean_envelope_odor-error_envelope_odor, mean_envelope_odor+error_envelope_odor,alpha = 0.2,color = 'tab:orange')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')
plt.xticks(ticks = np.arange(50,1100,100),labels = [], rotation = 45)
plt.ylabel('Z-scored activity')

plt.tight_layout()

plt.savefig('winners_losers_activity.pdf')

#%%


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,10))
             
odors = ['eth bu','2-hex','iso','hex','eth ti','eth ace']

pc_loadings_odors = ic_one_animals[14]

for x in np.arange(0,6):
            
    if x == 0:
        plt.title('1st IC Weight')
    plt.subplot(6,1,x+1)
                
    pca_plot = pc_loadings_odors[x]
    pca_plot[pca_plot>0.15] = 0.15
    pca_plot[pca_plot<-0.1] = -0.1
    
    
    plt.scatter(np.arange(0,pca_plot.shape[0]),pca_plot, s = 12, color = 'black')
    plt.hlines(0,0,pca_plot.shape[0],colors = 'black')
    #plt.hlines(np.mean(winners_threshold),0,pca_plot.shape[0],colors = 'grey',alpha = 0.4)
    
    for p in np.arange(0,pca_plot.shape[0]):
        plt.vlines(p,0,pca_plot[p],linewidth=1, color = 'black')            
    plt.ylim([-0.115,0.165])
    

    plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
    odor = odors[x]
    plt.ylabel(odor)
    
    plt.xticks(ticks = np.arange(0,pca_plot.shape[0],10),labels = [])
    

    
plt.xticks(ticks = np.arange(0,pca_plot.shape[0],10), labels = np.arange(0,pca_plot.shape[0],10))           
plt.xlabel('Principal Neurons')
plt.savefig('ICA_weights_example.pdf')

#%%
from scipy.stats.stats import spearmanr

pc_vector_corr = []
pc_vector_p = []

for animals in np.arange(0,15):
    vector_corr_matrix = np.zeros([6,6]) 
    vector_p_matrix = np.zeros([6,6])
    for x in np.arange(0,5):
        for y in np.arange(x+1,6):
            vector_corr_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals[animals,x]),np.squeeze(ic_one_animals[animals,y]))[0]
            vector_p_matrix[x,y] = pearsonr(np.squeeze(ic_one_animals[animals,x]),np.squeeze(ic_one_animals[animals,y]))[1]
                    
    vector_corr_matrix = vector_corr_matrix+vector_corr_matrix.T
    vector_corr_matrix = vector_corr_matrix + np.identity(6)
    
    vector_p_matrix = vector_p_matrix+vector_p_matrix.T
    
    pc_vector_corr.append(vector_corr_matrix)
    pc_vector_p.append(vector_p_matrix)
    
#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize = (3.3,5))

plt.subplot(211)
population_correlation = np.mean(pc_vector_corr,axis = 0) 

plt.imshow(population_correlation, aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 1)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = [],rotation = 45)
plt.yticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
plt.title('Average 1st IC Weight Correlation')

plt.subplot(212)
population_p = np.mean(pc_vector_p,axis = 0)
population_p = population_p<0.05
population_p = population_p+np.identity(6)
#population_p = np.fill_diagonal(population_p, 0)

plt.imshow(population_p, aspect = 'auto', cmap = 'Greys', vmin = 0, vmax = 1)
plt.colorbar()
plt.xticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'],rotation = 45)
plt.yticks(ticks = np.arange(0,6),labels = ['eth bu', '2-hex', 'iso', 'hex', 'eth ti', 'eth ace'])
plt.xlim([-0.5,5.5])
plt.ylim([-0.5,5.5])
#plt.title('Average Significant (P<0.016) Pairs')

plt.tight_layout()
plt.savefig('correlation_ica.pdf')

#%%

mvl_wl_animals = np.array(mvl_wl_animals)
circ_std_winners = mvl_wl_animals[:,:,0]
circ_std_losers = mvl_wl_animals[:,:,1]

mean_std_winners = []
mean_std_losers = []
for x in range(15):
    mean_std_winners.append(np.nanmean(np.concatenate(circ_std_winners[x,:])))
    mean_std_losers.append(np.nanmean(np.concatenate(circ_std_losers[x,:])))
    
s, p = stats.wilcoxon(mean_std_losers,mean_std_winners,alternative = 'greater')    

s_mvl, p_mvl = stats.ttest_rel(mean_std_losers,mean_std_winners,alternative = 'greater')
df = len(mean_std_winners)-1


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize=(2,8))


plt.subplot(211)

plt.boxplot([mean_std_winners,mean_std_losers],widths = 0.2, showfliers=False)

for x in range(15):
    plt.plot([1.2,1.8],[mean_std_winners[x],mean_std_losers[x]], color = 'grey')
    
plt.ylabel('Spike-Gamma Mean Vector Lenght')
plt.xticks(ticks = [1,2], labels = ['Winners','Losers'])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


phase_wl_animals = np.array(phase_wl_animals)
circ_std_winners = phase_wl_animals[:,:,0]
circ_std_losers = phase_wl_animals[:,:,1]

mean_std_winners = []
mean_std_losers = []
for x in range(15):
    mean_std_winners.append(stats.circmean(np.concatenate(circ_std_winners[x,:]),nan_policy='omit'))
    mean_std_losers.append(stats.circmean(np.concatenate(circ_std_losers[x,:]),nan_policy='omit'))
    
mean_std_winners = np.rad2deg(mean_std_winners)
mean_std_losers = np.rad2deg(mean_std_losers)
    
s, p = stats.wilcoxon(mean_std_losers,mean_std_winners,alternative = 'greater')    

#
s_t, p_t = stats.ttest_rel(mean_std_winners,mean_std_losers)
df = len(mean_std_winners)-1

plt.subplot(212)

plt.boxplot([mean_std_winners,mean_std_losers],widths = 0.2, showfliers=False)

for x in range(15):
    plt.plot([1.2,1.8],[mean_std_winners[x],mean_std_losers[x]], color = 'grey')
    
plt.ylabel('mean gamma phase')
plt.xticks(ticks = [1,2], labels = ['Winners','Losers'])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

#plt.savefig('winners_losers_gamma.pdf')
