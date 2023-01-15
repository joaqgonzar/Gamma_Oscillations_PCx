#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:05:23 2021

@author: joaquin
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import pandas as pd
import os
from scipy import signal
from scipy import stats

#%% load experiment data for thy recordings

names = ['150610','150624','150701','150707','150709']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory+'/THY')

exp_data = pd.read_csv('ExperimentCatalog_THY1.txt', sep=" ", header=None)

pcx = exp_data[6:11]

laser_intensities_first = np.array(pcx[1])
laser_intensities_last = np.array(pcx[2])

laser_packet = 10 # number of laser presentations per intensity

#%% loop through THY animals

spec_aniamls_thy = []
laser_power_thy = []
pre_laser_power_thy = []
intenisity_series_thy = []

for index, name in enumerate(names):
        
    print(name)
    
    # load lfp recordings
    
    os.chdir(directory+'/THY/Decimated_LFPs')
    lfp = np.load('PFx_THY_lfp_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    srate = 2000
    
    
    # load laser data
    os.chdir(directory+'/THY/processed/'+name)
    laser_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    laser_times = laser_data*srate
    channel = 17 # channel to select 
    
    # load lfps around laser onset
    lfp_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x-300):int(x+2700)])
                
    # compute spectrogram for the selected lfps     
    trig_spec = []
    for x in lfp_laser[:-1]:
        if len(x) == 3000:
            f,t,p = signal.spectrogram(x,fs = srate, window = ('hamming'),nperseg = int(srate/25), noverlap = int(srate/30),nfft = 10*srate)
            trig_spec.append(np.abs(p))
                    
    mean_spec = np.mean(trig_spec,axis = 0)

    # save data 
    spec_aniamls_thy.append(mean_spec)
    
    
    # get laser intensities and gamma power 
    
    intensity_first = laser_intensities_first[index]
    intenstity_last = laser_intensities_last[index]
    intenisities_power_series = np.repeat(np.arange(int(intensity_first),int(intenstity_last)+1),laser_packet)
    
    # get one second window before and after laser for computing spectrum
    lfp_laser = []
    lfp_pre_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x):int(x+2000)])
        lfp_pre_laser.append(lfp[channel,int(x-2000):int(x)])
             
    # compute spectrums    
    pxx_laser = []
    for x in lfp_laser:
        if len(x) > 1000:
            f,p_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_laser.append(p_laser)
            
    pxx_pre_laser = []
    for x in lfp_pre_laser[:-1]:
        if len(x) > 1000:        
            f,p_pre_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_pre_laser.append(p_pre_laser)
            
    
    # save data
    laser_power_thy.append(pxx_laser)
    pre_laser_power_thy.append(pxx_pre_laser)
    intenisity_series_thy.append(intenisities_power_series)
    
    
    
    
#%% get telc-thy data from both hemispheres

names_decimated_ipsi = ['PFx_TeLC-Thy_lfp_150627-1_.npz',
                   'PFx_TeLC-Thy_lfp_151213-1_.npz',
                   'PFx_TeLC-Thy_lfp_151116-3_.npz',  
                   'PFx_TeLC-Thy_lfp_151213-2_.npz',
                   'PFx_TeLC-Thy_lfp_151116-4_.npz',  
                   'PFx_TeLC-Thy_lfp_151215-1_.npz',
                   'PFx_TeLC-Thy_lfp_151118-1_.npz', 
                   'PFx_TeLC-Thy_lfp_151220-2_.npz',
                   'PFx_TeLC-Thy_lfp_151118-3_.npz', 
                   'PFx_TeLC-Thy_lfp_151220-3_.npz',
                   'PFx_TeLC-Thy_lfp_151120-2_.npz',
                   'PFx_TeLC-Thy_lfp_151222-1_.npz',
                   'PFx_TeLC-Thy_lfp_151120-3_.npz',
                   'PFx_TeLC-Thy_lfp_151222-2_.npz']

names_decimated_contra = ['PFx_TeLC-Thy_contra_lfp_151117-1_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151117-4_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151119-2_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151119-3_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151122-2_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151214-1_.npz',
                          'PFx_TeLC-Thy_contra_lfp_151216-1_.npz']


os.chdir(directory+'/TeLC-Thy_simul/processed')
exp_data = pd.read_csv('ExperimentCatalog_THY1-TeLC.txt', sep=" ", header=None)
pcx = exp_data[6:11]


# select recordings from experimental catalog
pcx = np.where(exp_data[3]=='P')[0]
ipsi = np.where(exp_data[4]=='T')[0]
contra = np.where(exp_data[4]=='C')[0]

# get mice identities and laser intensities
mice = np.intersect1d(pcx,ipsi)
laser_intensities_first = np.array(exp_data[1][mice])
laser_intensities_last = np.array(exp_data[2][mice])

laser_packet = 10 # number of laser presentations per intensity

# get name and banks of recordings
full_names = exp_data[0][mice]

names = []
for x in full_names:
    names.append(x[-14:-6])
    
banks = []
for x in full_names:
    banks.append(x[-5:])

names_banks = np.vstack([names, banks])


# loop through telc ipsi data

spec_aniamls_telc_ipsi = []
laser_power_telc_ipsi = []
pre_laser_power_telc_ipsi = []
intenisity_series_telc_ipsi = []

for index, name in enumerate(names_decimated_ipsi):
        
    print(name)
    
    # get lfp recording
    os.chdir(directory+'/TeLC-Thy_simul/Decimated_LFPs')
    lfp = np.load(name,allow_pickle=True)['lfp_downsample']   
    srate = 2000
    
    # get laser data
    os.chdir(directory+'/TeLC-Thy_simul/processed/'+name[-13:-5])
    bank = names_banks[1,np.where(names_banks[0,:] == name[-13:-5])[0][0]]
    laser_data = scipy.io.loadmat(name[-13:-5]+'_'+bank+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    laser_times = laser_data*srate
    channel = 17
    
    # load lfps around laser onset
    lfp_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x-300):int(x+2700)])
                
    # compute spectrogram around laser    
    trig_spec = []
    for x in lfp_laser[:-1]:
        if len(x) > 1000:
            f,t,p = signal.spectrogram(x,fs = srate,window = ('hamming'),nperseg = int(srate/25), noverlap = int(srate/30),nfft = 10*srate)
            trig_spec.append(np.abs(p))
                    
    mean_spec = np.mean(trig_spec,axis = 0)

    # save data
    spec_aniamls_telc_ipsi.append(mean_spec)
    
    
    
    # get laser intensities and gamma power 
    intensity_first = laser_intensities_first[index]
    intenstity_last = laser_intensities_last[index]
    intenisities_power_series = np.repeat(np.arange(int(intensity_first),int(intenstity_last)+1),laser_packet)
    
    # get one second window before and after laser for computing spectrum
    lfp_laser = []
    lfp_pre_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x):int(x+2000)])
        lfp_pre_laser.append(lfp[channel,int(x-2000):int(x)])
             
    # compute spectrums        
    pxx_laser = []
    for x in lfp_laser:
        if len(x) > 1000:
            f,p_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_laser.append(p_laser)
            
    pxx_pre_laser = []
    for x in lfp_pre_laser[:-1]:
        if len(x) > 1000:        
            f,p_pre_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_pre_laser.append(p_pre_laser)
            
    
    # save data
    laser_power_telc_ipsi.append(pxx_laser)
    pre_laser_power_telc_ipsi.append(pxx_pre_laser)
    intenisity_series_telc_ipsi.append(intenisities_power_series)
 
    
 
    
#%% run the contralateral hemisphere

# select recordings from experimental catalog
pcx = np.where(exp_data[3]=='P')[0]
ipsi = np.where(exp_data[4]=='T')[0]
contra = np.where(exp_data[4]=='C')[0]

# get mice identities and laser intensities
mice = np.intersect1d(pcx,contra)
laser_intensities_first = np.array(exp_data[1][mice])
laser_intensities_last = np.array(exp_data[2][mice])

laser_packet = 10 # number of laser presentations per intensity

# get name and banks of recordings
full_names = exp_data[0][mice]

names = []
for x in full_names:
    names.append(x[-14:-6])
    
banks = []
for x in full_names:
    banks.append(x[-5:])

names_banks = np.vstack([names, banks])

# loop through telc contra data

spec_aniamls_telc_contra = []
laser_power_telc_contra = []
pre_laser_power_telc_contra = []
intenisity_series_telc_contra = []

for index, name in enumerate(names_decimated_contra):
        
    print(name)
    
    # get lfps
    os.chdir(directory+'/TeLC-Thy_simul/Decimated_LFPs')
    lfp = np.load(name,allow_pickle=True)['lfp_downsample']
    srate = 2000
    
    # get laser data
    os.chdir(directory+'/TeLC-Thy_simul/processed/'+name[-13:-5])
    bank = names_banks[1,np.where(names_banks[0,:] == name[-13:-5])[0][0]]
    laser_data = scipy.io.loadmat(name[-13:-5]+'_'+bank+'.mat',struct_as_record=False)['efd'][0][0].LaserTimes[0][0].LaserOn
    laser_data = laser_data[0][0][0]
    laser_times = laser_data*srate
    channel = 17
    
    # select lfp around laser
    lfp_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x-300):int(x+2700)])
             
    # compute spectrogram around laser    
    trig_spec = []
    for x in lfp_laser[:-1]:
        if len(x) > 1000:
            f,t,p = signal.spectrogram(x,fs = srate, window = ('hamming'),nperseg = int(srate/25), noverlap = int(srate/30),nfft = 10*srate)
            trig_spec.append(np.abs(p))
                    
    mean_spec = np.mean(trig_spec,axis = 0)

    # save results
    spec_aniamls_telc_contra.append(mean_spec)

    # get laser intensities and gamma power 
    intensity_first = laser_intensities_first[index]
    intenstity_last = laser_intensities_last[index]
    intenisities_power_series = np.repeat(np.arange(int(intensity_first),int(intenstity_last)+1),laser_packet)
    
    # get lfp before and after laser
    lfp_laser = []
    lfp_pre_laser = []
    for x in laser_times:
        lfp_laser.append(lfp[channel,int(x):int(x+2000)])
        lfp_pre_laser.append(lfp[channel,int(x-2000):int(x)])
           
    # compute power spectrum before and after laser    
    pxx_laser = []
    for x in lfp_laser:
        if len(x) > 1000:
            f,p_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_laser.append(p_laser)
            
    pxx_pre_laser = []
    for x in lfp_pre_laser[:-1]:
        if len(x) > 1000:        
            f,p_pre_laser = signal.welch(x,fs = srate,nperseg = srate,nfft = 10*srate)
            pxx_pre_laser.append(p_pre_laser)
            
    
    # save data
    laser_power_telc_contra.append(pxx_laser)
    pre_laser_power_telc_contra.append(pxx_pre_laser)
    intenisity_series_telc_contra.append(intenisities_power_series)

#%% save data 
    
os.chdir(directory)
np.savez('Figure_3.npz', laser_power_thy = laser_power_thy, pre_laser_power_thy = pre_laser_power_thy, intenisity_series_thy = intenisity_series_thy, laser_power_telc_ipsi = laser_power_telc_ipsi, pre_laser_power_telc_ipsi = pre_laser_power_telc_ipsi, intenisity_series_telc_ipsi = intenisity_series_telc_ipsi, laser_power_telc_contra = laser_power_telc_contra, pre_laser_power_telc_contra = pre_laser_power_telc_contra, intenisity_series_telc_contra = intenisity_series_telc_contra)

#%%
os.chdir(directory)

laser_power_thy = np.load('Figure_3.npz', allow_pickle=True)['laser_power_thy']
pre_laser_power_thy = np.load('Figure_3.npz', allow_pickle=True)['pre_laser_power_thy']
intenisity_series_thy = np.load('Figure_3.npz', allow_pickle=True)['intenisity_series_thy']
laser_power_telc_ipsi = np.load('Figure_3.npz', allow_pickle=True)['laser_power_telc_ipsi']
laser_power_telc_ipsi = np.load('Figure_3.npz', allow_pickle=True)['laser_power_telc_ipsi']
pre_laser_power_telc_ipsi = np.load('Figure_3.npz', allow_pickle=True)['pre_laser_power_telc_ipsi']
intenisity_series_telc_ipsi = np.load('Figure_3.npz', allow_pickle=True)['intenisity_series_telc_ipsi']
pre_laser_power_telc_contra = np.load('Figure_3.npz', allow_pickle=True)['pre_laser_power_telc_contra']
laser_power_telc_contra = np.load('Figure_3.npz', allow_pickle=True)['laser_power_telc_contra']
intenisity_series_telc_contra = np.load('Figure_3.npz', allow_pickle=True)['intenisity_series_telc_contra']


#%% plot spectrogram figure


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

spec_aniamls_thy = np.array(spec_aniamls_thy)
spec_plot = np.mean(spec_aniamls_thy[1:,:,:], axis = 0)

min_gamma = 8.5
max_gamma = 10
    
plt.figure(dpi = 300, figsize = (5,6))

plt.subplot(311)

plt.imshow(np.log(spec_plot),interpolation = 'gaussian',cmap = 'jet',vmin = min_gamma, vmax = max_gamma,aspect = 'auto')
        
plt.ylim([0,1200])
plt.yticks(ticks = np.arange(0,1300,200),labels = np.arange(0,130,20))
plt.xticks(ticks = np.arange(0,spec_plot.shape[1],spec_plot.shape[1]/5),labels = np.arange(0,1500,300))
plt.plot(np.arange(np.where(t == 0.153)[0],np.where(t == 1.154)[0]),920*np.ones(143),color = 'white')
plt.text(12,1000,'OB Laser ON',color = 'white')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.title('Thy-Control')      


spec_aniamls_telc_ipsi = np.array(spec_aniamls_telc_ipsi)
spec_plot = np.mean(spec_aniamls_telc_ipsi[1:,:,:], axis = 0)

plt.subplot(312)

plt.imshow(np.log(spec_plot),interpolation = 'gaussian',cmap = 'jet',vmin = min_gamma, vmax = max_gamma,aspect = 'auto')
        
plt.ylim([0,1200])
plt.yticks(ticks = np.arange(0,1300,200),labels = np.arange(0,130,20))
plt.xticks(ticks = np.arange(0,spec_plot.shape[1],spec_plot.shape[1]/5),labels = np.arange(0,1500,300))
plt.plot(np.arange(np.where(t == 0.153)[0],np.where(t == 1.154)[0]),920*np.ones(143),color = 'white')
plt.text(12,1000,'OB Laser ON',color = 'white')
plt.title('Thy-TeLC')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.xlabel('Time (ms)')


spec_aniamls_telc_contra = np.array(spec_aniamls_telc_contra)
spec_plot = np.mean(spec_aniamls_telc_contra[:,:,:], axis = 0)

plt.subplot(313)

plt.imshow(np.log(spec_plot),interpolation = 'gaussian',cmap = 'jet',vmin = min_gamma, vmax = max_gamma,aspect = 'auto')
        
plt.ylim([0,1200])
plt.yticks(ticks = np.arange(0,1300,200),labels = np.arange(0,130,20))
plt.xticks(ticks = np.arange(0,spec_plot.shape[1],spec_plot.shape[1]/5),labels = np.arange(0,1500,300))
plt.plot(np.arange(np.where(t == 0.153)[0],np.where(t == 1.154)[0]),920*np.ones(143),color = 'white')
plt.text(12,1000,'OB Laser ON',color = 'white')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.title('Thy-TeLC contralateral')      

plt.tight_layout()

plt.savefig('spectrograms_thy.pdf')

#%% plot second figure

mean_power_laser_control = []
for x in laser_power_thy:
    mean_power_laser_control.append(np.mean(x, axis = 0))

mean_power_pre_control = []
for x in pre_laser_power_thy:
    mean_power_pre_control.append(np.mean(x, axis = 0))
    
    
mean_power_laser_ipsi = []
for x in laser_power_telc_ipsi:
    mean_power_laser_ipsi.append(np.mean(x, axis = 0))

mean_power_pre_ipsi = []
for x in pre_laser_power_telc_ipsi:
    mean_power_pre_ipsi.append(np.mean(x, axis = 0))
    

mean_power_laser_contra = []
for x in laser_power_telc_contra:
    mean_power_laser_contra.append(np.mean(x, axis = 0))

mean_power_pre_contra = []
for x in pre_laser_power_telc_contra:
    mean_power_pre_contra.append(np.mean(x, axis = 0))
    

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,12))

plt.subplot(311)

laser_control = np.mean(mean_power_laser_control*f, axis = 0)
laser_ipsi = np.mean(mean_power_laser_ipsi*f, axis = 0)
laser_contra = np.mean(mean_power_laser_contra*f, axis = 0)

error_control = np.std(mean_power_laser_control*f, axis = 0)/np.sqrt(len(mean_power_laser_control))
error_ipsi = np.std(mean_power_laser_ipsi*f, axis = 0)/np.sqrt(len(mean_power_laser_ipsi))
error_contra = np.std(mean_power_laser_contra*f, axis = 0)/np.sqrt(len(mean_power_laser_contra))

line_freq = int(np.where(f == 60)[0])

laser_control[line_freq-20:line_freq+20] = (laser_control[line_freq-60:line_freq-20]+laser_control[line_freq+20:line_freq+60])/2
error_control[line_freq-20:line_freq+20] = (error_control[line_freq-60:line_freq-20]+error_control[line_freq+20:line_freq+60])/2

laser_ipsi[line_freq-20:line_freq+20] = (laser_ipsi[line_freq-60:line_freq-20]+laser_ipsi[line_freq+20:line_freq+60])/2
error_ipsi[line_freq-20:line_freq+20] = (error_ipsi[line_freq-60:line_freq-20]+error_ipsi[line_freq+20:line_freq+60])/2

laser_contra[line_freq-20:line_freq+20] = (laser_contra[line_freq-60:line_freq-20]+laser_contra[line_freq+20:line_freq+60])/2
error_contra[line_freq-20:line_freq+20] = (error_contra[line_freq-60:line_freq-20]+error_contra[line_freq+20:line_freq+60])/2


plt.plot(f[200:800],laser_control[200:800], label = 'Thy',color = 'black')
plt.fill_between(f[200:800],laser_control[200:800]-error_control[200:800],laser_control[200:800]+error_control[200:800],alpha = 0.3,edgecolor=None,color = 'black')
plt.plot(f[200:800],laser_ipsi[200:800], label = 'Thy-TeLC',color = 'tab:red')
plt.fill_between(f[200:800],laser_ipsi[200:800]-error_ipsi[200:800],laser_ipsi[200:800]+error_ipsi[200:800],alpha = 0.3,edgecolor=None,color = 'tab:red')
plt.plot(f[200:800],laser_contra[200:800], label = 'Thy-TeLC Contra',color = 'tab:blue')
plt.fill_between(f[200:800],laser_contra[200:800]-error_contra[200:800],laser_contra[200:800]+error_contra[200:800],alpha = 0.3,edgecolor=None,color = 'tab:blue')

plt.yscale('log')
plt.xlim([20,80])
plt.ylim([5e3,1e7])
plt.xlabel('Frequency (Hz)')
plt.ylabel('White Power (a.u)')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.8, axis = 'both', which = 'Both')

# plot gamma increases 

laser_control_array = np.mean(np.array(mean_power_laser_control*f)[:,300:550],axis = 1)
laser_ipsi_array = np.mean(np.array(mean_power_laser_ipsi*f)[:,300:550],axis = 1)
laser_contra_array = np.mean(np.array(mean_power_laser_contra*f)[:,300:550],axis = 1)

laser_control = np.mean(laser_control_array,axis = 0)
laser_ipsi = np.mean(laser_ipsi_array,axis = 0)
laser_contra = np.mean(laser_contra_array,axis = 0)

error_control = np.std(laser_control_array,axis = 0)/np.sqrt(len(mean_power_laser_control))
error_ipsi = np.std(laser_ipsi_array,axis = 0)/np.sqrt(len(mean_power_laser_ipsi))
error_contra = np.std(laser_contra_array,axis = 0)/np.sqrt(len(mean_power_laser_contra))

pre_control_array = np.mean(np.array(mean_power_pre_control*f)[:,300:550],axis = 1)
pre_ipsi_array = np.mean(np.array(mean_power_pre_ipsi*f)[:,300:550],axis = 1)
pre_contra_array = np.mean(np.array(mean_power_pre_contra*f)[:,300:550],axis = 1)

pre_control = np.mean(pre_control_array, axis = 0)
pre_ipsi = np.mean(pre_ipsi_array, axis = 0)
pre_contra = np.mean(pre_contra_array, axis = 0)

error_control_pre = np.std(pre_control_array, axis = 0)/np.sqrt(len(mean_power_laser_control))
error_ipsi_pre = np.std(pre_ipsi_array, axis = 0)/np.sqrt(len(mean_power_laser_ipsi))
error_contra_pre = np.std(pre_contra_array, axis = 0)/np.sqrt(len(mean_power_laser_contra))

diff_control = laser_control_array-pre_control_array
diff_ipsi = laser_ipsi_array-pre_ipsi_array
diff_contra = laser_contra_array-pre_contra_array


plt.subplot(312)

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.8, axis = 'both', which = 'Both')

plt.boxplot([diff_control,diff_ipsi,diff_contra], widths = 0.4,showfliers=False)
plt.scatter(np.ones(diff_control.shape[0]),diff_control,color = 'black',alpha = 0.5)
plt.scatter(2*np.ones(diff_ipsi.shape[0]),diff_ipsi,color = 'black',alpha = 0.5)
plt.scatter(3*np.ones(diff_contra.shape[0]),diff_contra,color = 'black',alpha = 0.5)

plt.xlim([0.5,3.5])
plt.ylim([-100000,3e6])
plt.xticks(ticks = [1,2,3], labels = ['Control','TeLC ipsi','TeLC contra'])
plt.ylabel('White Power (a.u)')

# get statistics

s_control, p_control = stats.ttest_ind(diff_control,diff_ipsi,alternative = 'greater')
df_control = diff_control.shape[0]+diff_ipsi.shape[0]-2
s_contra, p_contra = stats.ttest_ind(diff_contra,diff_ipsi,alternative = 'greater')
df_contra = diff_contra.shape[0]+diff_ipsi.shape[0]-2
s_contra_control, p_contra_control = stats.ttest_ind(diff_control,diff_contra)


# plot laser intensities and gamma power

# exclude all animals without a full intensity series

laser_intensity_ipsi = intenisity_series_telc_ipsi
laser_intensity_contra = intenisity_series_telc_contra
laser_intensity_control = intenisity_series_thy

laser_power_telc_ipsi = np.array(laser_power_telc_ipsi)
gamma_ipsi_series = np.empty((laser_power_telc_ipsi.shape[0]-2,80))
gamma_ipsi_series[:] = np.nan

i = 0
for x in laser_power_telc_ipsi:
    data = np.array(x)*f
    if data.shape[0] == 80:
        mean_gamma = np.nanmean(data[:,300:550],axis = 1)
        gamma_ipsi_series[i,0:mean_gamma.shape[0]] = mean_gamma
        i = i+1

#
laser_power_telc_contra = np.array(laser_power_telc_contra)
gamma_contra_series = np.empty((laser_power_telc_contra.shape[0]-1,80))
gamma_contra_series[:] = np.nan

i = 0
for x in laser_power_telc_contra:
    data = np.array(x)*f
    if data.shape[0] == 80:
        mean_gamma = np.nanmean(data[:,300:550],axis = 1)
        gamma_contra_series[i,0:mean_gamma.shape[0]] = mean_gamma
        i = i+1
        
laser_power_thy = np.array(laser_power_thy)
gamma_control_series = np.empty((laser_power_thy.shape[0]-2,80))
gamma_control_series[:] = np.nan

i = 0
for x in laser_power_thy:
        
    data = np.array(x)*f  
    if data.shape[0] == 80:
        mean_gamma = np.nanmean(data[:,300:550],axis = 1)
        gamma_control_series[i,0:mean_gamma.shape[0]] = mean_gamma
        i = i+1
        
gamma_intensity_ipsi = []
gamma_intensity_contra = []
gamma_intensity_control = []

for x in np.arange(0,8):
    
    gamma_intensity_ipsi.append(np.nanmean(gamma_ipsi_series[:,x*10:(x+1)*10],axis = 1))
    gamma_intensity_contra.append(np.nanmean(gamma_contra_series[:,x*10:(x+1)*10],axis = 1))
    gamma_intensity_control.append(np.nanmean(gamma_control_series[:,x*10:(x+1)*10],axis = 1))
    
gamma_intensity_control = np.array(gamma_intensity_control)
gamma_intensity_ipsi = np.array(gamma_intensity_ipsi)
gamma_intensity_contra = np.array(gamma_intensity_contra)
    
plt.subplot(313)

plt.errorbar(np.arange(0,8),np.nanmean(gamma_intensity_control,axis = 1),np.nanstd(gamma_intensity_control,axis = 1)/np.sqrt(3),marker = 'o',markersize=8,linewidth=2,color = 'black')
plt.errorbar(np.arange(0,8),np.nanmean(gamma_intensity_ipsi,axis = 1),np.nanstd(gamma_intensity_ipsi,axis = 1)/np.sqrt(12),marker = 'o',markersize=8,linewidth=2,color = 'tab:red')
plt.errorbar(np.arange(0,8),np.nanmean(gamma_intensity_contra,axis = 1),np.nanstd(gamma_intensity_contra,axis = 1)/np.sqrt(6),marker = 'o',markersize=8,linewidth=2,color = 'tab:blue')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.8, axis = 'both', which = 'Both')
plt.ylabel('Gamma Power (a.u)')
plt.xlabel('Laser Intensity (mW/mm2)')
plt.xticks(ticks = np.arange(0,8), labels = [0,1,5,10,20,30,40,50])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

#plt.savefig('stats_laser.pdf')