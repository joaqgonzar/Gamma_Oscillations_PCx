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


def modulation_index(amp,fase_lenta,numbin):
    
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
    p = mean_amp/np.sum(mean_amp)
    
    entrop = -1*(np.sum(p*np.log(p)))
    mi = (np.log(numbin)-entrop)/np.log(numbin)   

    return(mi)



def pac_cfc_resp(lfp,resp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin,mask):
    
    AmpFreqTransformed1 = []
    PhaseFreq = []
    
    pac_mi = []
    
    for ii in altaVector:
        Af1 = ii
        Af2=Af1+alta_BandWidth
        
        AmpFreq1=eegfilt(lfp,srate,Af1,Af2) # just filtering
        
        analytic_signal1 = signal.hilbert(AmpFreq1)
        AmpFreqTransformed1.append(np.abs(analytic_signal1)[mask])
        

    faselenta = []
    for jj in lentaVector: 
        Pf1 = jj
        Pf2 = Pf1 + lenta_BandWidth
        PhaseFreq=eegfilt(resp,srate,Pf1,Pf2)
        analytic_signal = signal.hilbert(PhaseFreq)
        faselenta.append(np.angle(analytic_signal)[mask])
        
    
    fase_lenta = faselenta
    
    comodulogram = np.empty((len(lentaVector),len(altaVector)))
    
         
    for count_lenta,lenta in enumerate(fase_lenta):
            
        for count_rapida,rapida in enumerate(AmpFreqTransformed1):    
    
            comodulogram[count_lenta,count_rapida] = modulation_index(rapida,lenta,numbin)
                
    
    return(comodulogram)   
#%% experiment data 

names_ipsi = ['150220','150221','150223','150312','150327','150403','150406','150812']
names_contra = ['150221','150312','150327','150403','150406','150812']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos'

os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx')

exp_data = pd.read_csv('ExperimentCatalog_TeLC-PCX.txt', sep=" ", header=None)

loading = exp_data[3][7:15]

#%% run ipsilateral animals

power_animals_ipsi = []
power_resp_ipsi = []
coher_resp_ipsi = []
pac_animals_ipsi = []

index = 0
loading = np.array(exp_data[3][7:15])

for name in names_ipsi:

    print(name)
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
        
    elif  loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        
    
    # load data 
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    lfp = np.load('PFx_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    channel = 17
    lfp = lfp[channel,:]
    srate = 2000
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name) 
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    resp = np.squeeze(resp[0:lfp.shape[0]])

    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
        
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate
    odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate,axis = 1))
    
    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[0]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[0]))
    
    lfp_odorless = lfp[lfp_mask]
    resp_odorless = resp[lfp_mask]
    
    # get power spectrum
    
    f,pxx = signal.welch(lfp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,pxx_resp = signal.welch(resp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,cxx = signal.coherence(lfp_odorless, resp_odorless, fs = srate,nperseg = srate,nfft = 10*srate)
    
    #  pac parameters
    
    lentaVector = np.arange(0.2,10,1)
    altaVector = np.arange(10,150,10)
    lenta_BandWidth = 2
    alta_BandWidth = 10
    numbin = 18
    
    # get phase amplitude coupling
    pac = pac_cfc_resp(lfp,resp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin, lfp_mask)
    
    
    # save results
    
    power_animals_ipsi.append(pxx)
    power_resp_ipsi.append(pxx_resp)
    coher_resp_ipsi.append(cxx)
    pac_animals_ipsi.append(pac)
    index = index+1
    

# run contralateral animals

loading = np.array(exp_data[3][1:7])

power_animals_contra = []
power_resp_contra = []
coher_resp_contra = []
pac_animals_contra = []
index = 0

for name in names_contra:

    print(name)
    
    # select wich odor loading was used
    if loading[index] == 'A':
        odor_series = list(np.array([2,3,4,5,7,8,10,11,12,13,15,16])-1) 
        
    elif  loading[index] == 'D':
        odor_series = list(np.array([7,8,15,16])-1)
        
    # load data 
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/Decimated_LFPs')
    lfp = np.load('PFx_contra_TeLC_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    channel = 17
    lfp = lfp[channel,:]
    srate = 2000
    
    os.chdir(directory+'/Frank_Boldings_Dataset/TeLC_PFx/processed/'+name) 
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    resp = np.squeeze(resp[0:lfp.shape[0]])

    inh_start = scipy.io.loadmat(name+'.mat')['PREX']*srate
    inh_start = np.squeeze(inh_start)
        
    odor_data = scipy.io.loadmat(name+'_bank1_efd'+'.mat',struct_as_record=False)['efd'][0][0].ValveTimes[0][0].PREXTimes[0]
    odor_times = odor_data[odor_series]
    odor_times_srate = odor_times*srate
    odor_times_srate = np.matrix.flatten(np.concatenate(odor_times_srate,axis = 1))
    
    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[0]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[0]))
    
    lfp_odorless = lfp[lfp_mask]
    resp_odorless = resp[lfp_mask]
    
    # get power spectrum
    
    #pxx = []
    #for x in range(lfp.shape[0]):
    f,pxx = signal.welch(lfp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,pxx_resp = signal.welch(resp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,cxx = signal.coherence(lfp_odorless, resp_odorless, fs = srate,nperseg = srate,nfft = 10*srate)
    
    
  
    # parametros generales
    
    lentaVector = np.arange(0.2,10,1)
    altaVector = np.arange(10,150,10)
    lenta_BandWidth = 2
    alta_BandWidth = 10
    numbin = 18
    
    # get phase amplitude coupling
    
    
    pac = pac_cfc_resp(lfp,resp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin, lfp_mask)
    
    
    # save results
    
    power_animals_contra.append(pxx)
    power_resp_contra.append(pxx_resp)
    coher_resp_contra.append(cxx)
    pac_animals_contra.append(pac)
    index = index+1
    
    
# get control animals


names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)

awake_times = np.array([names,np.array(exp_data[2][1:14])])

# define electrode positions in probe 

positions_2 = np.array([8,28,1,30,11,20,10,21,15,16,14,17])
positions_1 = np.array([9,29,0,31,10,21,11,20,14,17,15,16])

loading = np.array(exp_data[3][14:])

# loop through animals

power_animals_control = []
pac_resp_animals_control = []
power_resp_animals_control = []
coher_resp_animals_control = []


index = 0

for name in names:
    
    print(name)
    
    # load data 

    # load lfp
    os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')
    lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    srate = 2000
    
    # get same channel for all recordings
    if (name == '141208-1') or (name =='141208-2') or (name == '141209'):
        channel = 17
    else: 
        channel = 16 
        
    lfp = np.squeeze(lfp[channel,:])    
    
    
    # load respiration and get awake recording
    os.chdir(directory+'/Simul/processed/'+name)
    resp = scipy.io.loadmat(name+'.mat')['RRR']
    resp = np.squeeze(resp[0:int(lfp.shape[0])]) # get only awake times
    
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
    
    # exclude odor deliveries and non-awake times
    inh_no_odor = np.setxor1d(odor_times_srate,resp_start)
    inh_no_odor = inh_no_odor[inh_no_odor<resp.shape[0]]
  
    
  
    # remove odor deliveries and get mask variable
    
    odor_times_awake = odor_times_srate[odor_times_srate<lfp.shape[0]-2000]
    odor_times = []
    for x in odor_times_awake:
        odor_times.append(np.arange(int(x-500),int(x+2000)))
    
    odor_times = np.concatenate(odor_times)   
    lfp_mask = np.setxor1d(odor_times,np.arange(0,lfp.shape[0]))
    
    lfp_odorless = lfp[lfp_mask]
    resp_odorless = resp[lfp_mask]
    
    
    
    # compute power spectrum and coherence
    
    f,pxx = signal.welch(lfp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,pxx_resp = signal.welch(resp_odorless,fs = srate,nperseg = srate,nfft = 10*srate)
    f,cxx = signal.coherence(lfp_odorless, resp_odorless, fs = srate,nperseg = srate,nfft = 10*srate)



    # pac parameters
    
    lentaVector = np.arange(0.2,10,1)
    altaVector = np.arange(10,150,10)
    lenta_BandWidth = 2
    alta_BandWidth = 10
    numbin = 18
    
    # get phase amplitude coupling using slow lfp and resp
     
    pac_resp = pac_cfc_resp(lfp,resp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin,lfp_mask)
            
    # save results
    
    power_animals_control.append(pxx)
    power_resp_animals_control.append(pxx_resp)
    coher_resp_animals_control.append(cxx)
    pac_resp_animals_control.append(pac_resp)
    
    index = index + 1
    #del lfp, pac_lfp, pac_resp, pxx
    
#%% save results

os.chdir(directory)
np.savez('Figure_2.npz',power_animals_ipsi = power_animals_ipsi, power_resp_ipsi= power_resp_ipsi,coher_resp_ipsi = coher_resp_ipsi,pac_animals_ipsi= pac_animals_ipsi,power_animals_contra= power_animals_contra, power_resp_contra = power_resp_contra, coher_resp_contra= coher_resp_contra, pac_animals_contra= pac_animals_contra, power_animals_control= power_animals_control, power_resp_animals_control= power_resp_animals_control, coher_resp_animals_control= coher_resp_animals_control, pac_resp_animals_control= pac_resp_animals_control)
    
#%% load results

os.chdir(directory+'/Frank_Boldings_Dataset')
power_animals_ipsi = np.load('Figure_2.npz')['power_animals_ipsi']
power_animals_contra = np.load('Figure_2.npz')['power_animals_contra']
power_animals_control = np.load('Figure_2.npz')['power_animals_control']

pac_animals_ipsi = np.load('Figure_2.npz')['pac_animals_ipsi']
pac_animals_contra = np.load('Figure_2.npz')['pac_animals_contra']
pac_resp_animals_control = np.load('Figure_2.npz')['pac_resp_animals_control']



#power_animals_ipsi = power_animals_ipsi[:,5:3000]
#power_animals_control = power_animals_control[:,5:3000]
#power_animals_contra = power_animals_contra[:,5:3000]

#%% plot results

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (3,6))

plt.subplot(211)

f = np.arange(0,1000.1,0.1)[5:3000]

mean_power_ipsi = np.mean(power_animals_ipsi,axis = 0)*f
error_power_ipsi = (np.std(power_animals_ipsi,axis = 0)*f)/np.sqrt(13)

# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power_ipsi[line_freq-20:line_freq+20] = (mean_power_ipsi[line_freq-60:line_freq-20]+mean_power_ipsi[line_freq+20:line_freq+60])/2
error_power_ipsi[line_freq-20:line_freq+20] = (error_power_ipsi[line_freq-60:line_freq-20]+error_power_ipsi[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power_ipsi,linewidth = 1.5, color = 'tab:red',label = 'TeLC Ipsi')
plt.fill_between(f,mean_power_ipsi-2*error_power_ipsi,mean_power_ipsi+2*error_power_ipsi,color = 'tab:red',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both', which = 'both')

plt.xscale('log')
plt.xlim([0.5,300])
plt.ylim([10e2,2.5e5])
plt.ylabel('Whitened Power')


mean_power_contra = np.mean(power_animals_control,axis = 0)*f
error_power_contra = (np.std(power_animals_control,axis = 0)*f)/np.sqrt(13)

# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power_contra[line_freq-20:line_freq+20] = (mean_power_contra[line_freq-60:line_freq-20]+mean_power_contra[line_freq+20:line_freq+60])/2
error_power_contra[line_freq-20:line_freq+20] = (error_power_contra[line_freq-60:line_freq-20]+error_power_contra[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power_contra,linewidth = 1.5, color = 'black',label = 'Control')
plt.fill_between(f,mean_power_contra-2*error_power_contra,mean_power_contra+2*error_power_contra,color = 'black',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both', which = 'both')

plt.xscale('log')
plt.xlim([0.5,300])
plt.ylim([10e2,2.5e5])
plt.ylabel('Whitened Power')
plt.legend()

plt.subplot(212)

mean_power_ipsi = np.mean(power_animals_ipsi,axis = 0)*f
error_power_ipsi = (np.std(power_animals_ipsi,axis = 0)*f)/np.sqrt(13)

# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power_ipsi[line_freq-20:line_freq+20] = (mean_power_ipsi[line_freq-60:line_freq-20]+mean_power_ipsi[line_freq+20:line_freq+60])/2
error_power_ipsi[line_freq-20:line_freq+20] = (error_power_ipsi[line_freq-60:line_freq-20]+error_power_ipsi[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power_ipsi,linewidth = 1.5, color = 'tab:red',label = 'TeLC Ipsi')
plt.fill_between(f,mean_power_ipsi-2*error_power_ipsi,mean_power_ipsi+2*error_power_ipsi,color = 'tab:red',alpha = 0.3,edgecolor=None)

mean_power_contra = np.mean(power_animals_contra,axis = 0)*f
error_power_contra = (np.std(power_animals_contra,axis = 0)*f)/np.sqrt(13)

# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power_contra[line_freq-20:line_freq+20] = (mean_power_contra[line_freq-60:line_freq-20]+mean_power_contra[line_freq+20:line_freq+60])/2
error_power_contra[line_freq-20:line_freq+20] = (error_power_contra[line_freq-60:line_freq-20]+error_power_contra[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power_contra,linewidth = 1.5, color = 'tab:blue',label = 'TeLC Contra')
plt.fill_between(f,mean_power_contra-2*error_power_contra,mean_power_contra+2*error_power_contra,color = 'tab:blue',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both', which = 'both')

plt.xscale('log')
plt.xlim([0.5,300])
plt.ylim([10e2,2.5e5])
plt.ylabel('Whitened Power')
plt.xlabel('Frequency(Hz)')
plt.legend()

#plt.savefig('TeLC_power.pdf')


#%%

plt.figure(dpi = 300,figsize = (7,6))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.subplot(222)

mean_power_resp = np.mean(power_resp_ipsi,axis = 0)
mean_coher = np.mean(coher_resp_ipsi,axis = 0)

vmin = 0.0002
vmax = 0.005

data_plot = np.mean(pac_animals_ipsi,axis = 0)

extent = [lentaVector[0]+lenta_BandWidth/2,lentaVector[-1]+lenta_BandWidth/2,altaVector[-1]+alta_BandWidth/2,altaVector[0]+alta_BandWidth/2]

plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')


plt.plot(f,(mean_power_resp*0.0002)+20,color = 'white',linewidth=1.5,label = 'LFP Power')
plt.plot(f,(mean_coher*600)+10,'--',color = 'white',linewidth=1.5,label = 'LFP-Resp') 

legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amp Frequency (Hz)')


plt.subplot(221)

mean_power_resp = np.mean(power_resp_animals_control,axis = 0)
mean_coher = np.mean(coher_resp_animals_control,axis = 0)

vmin = 0.0003
vmax = 0.005

data_plot = np.mean(pac_resp_animals_control,axis = 0)
plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')
plt.plot(f,(mean_power_resp*0.0006)+20,color = 'white',linewidth=1.5,label = 'LFP Power')
plt.plot(f,(mean_coher*600)+10,'--',color = 'white',linewidth=1.5,label = 'LFP-Resp') 

legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amp Frequency (Hz)')

plt.tight_layout()



plt.subplot(223)

mean_power_resp = np.mean(power_resp_ipsi,axis = 0)
mean_coher = np.mean(coher_resp_ipsi,axis = 0)

vmin = 0.0003
vmax = 0.0028

data_plot = np.mean(pac_animals_ipsi,axis = 0)

extent = [lentaVector[0]+lenta_BandWidth/2,lentaVector[-1]+lenta_BandWidth/2,altaVector[-1]+alta_BandWidth/2,altaVector[0]+alta_BandWidth/2]

plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')


plt.plot(f,(mean_power_resp*0.0002)+20,color = 'white',linewidth=1.5,label = 'LFP Power')
plt.plot(f,(mean_coher*600)+10,'--',color = 'white',linewidth=1.5,label = 'LFP-Resp') 

legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amp Frequency (Hz)')


plt.subplot(224)

mean_power_resp = np.mean(power_resp_contra,axis = 0)
mean_coher = np.mean(coher_resp_contra,axis = 0)

vmin = 0.0003
vmax = 0.0028

data_plot = np.mean(pac_animals_contra,axis = 0)
plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')
plt.plot(f,(mean_power_resp*0.0002)+20,color = 'white',linewidth=1.5,label = 'LFP Power')
plt.plot(f,(mean_coher*600)+10,'--',color = 'white',linewidth=1.5,label = 'LFP-Resp') 

legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amp Frequency (Hz)')

plt.tight_layout()

plt.savefig('TeLC_pac.pdf')

#%%

common_animals = [1,3,4,5,6,7]
lentaVector = np.arange(0.2,10,1)
altaVector = np.arange(10,150,10)
lenta_BandWidth = 2
alta_BandWidth = 10
numbin = 18
f = np.arange(0,1000.1,0.1)

from scipy import stats

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


alta = np.where(altaVector==50)[0][0]
baja = np.where(lentaVector==1.2)[0][0]


plt.figure(dpi = 300, figsize = (3,8))

plt.subplot(212)

plt.boxplot([np.squeeze(np.array(pac_resp_animals_control)[:,baja,alta]),np.squeeze(np.array(pac_animals_ipsi)[:,baja,alta]),np.squeeze(np.array(pac_animals_contra)[:,baja,alta])],widths = 0.5)

plt.scatter(np.ones(13),np.squeeze(np.array(pac_resp_animals_control)[:,baja,alta]),c = 'black',alpha = 0.5)
plt.scatter(1+np.ones(8),np.squeeze(np.array(pac_animals_ipsi)[:,baja,alta]),c = 'black',alpha = 0.5)
plt.scatter(2+np.ones(6),np.squeeze(np.array(pac_animals_contra)[:,baja,alta]),c = 'black',alpha = 0.5)

plt.xticks(ticks = [1,2,3], labels = ['Control','Ipsi','Contra'])
plt.ylabel('RR-Gamma Mod Index')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


s_pac_contra,p_pac_contra = stats.ttest_ind(np.log(np.array(pac_animals_ipsi)[:,baja,alta]),np.log(np.array(pac_animals_contra)[:,baja,alta]),alternative = 'less')
s_pac_control,p_pac_control = stats.ttest_ind(np.log(np.array(pac_animals_ipsi)[:,baja,alta]),np.log(np.array(pac_resp_animals_control)[:,baja,alta]),alternative = 'less')

plt.yscale('log')

alta = 500
plt.subplot(211)

plt.boxplot([np.squeeze((np.array(power_animals_control)*f)[:,alta]),np.squeeze((np.array(power_animals_ipsi)*f)[:,alta]),np.squeeze((np.array(power_animals_contra)*f)[:,alta])],widths = 0.5)

plt.scatter(np.ones(13),np.squeeze((np.array(power_animals_control)*f)[:,alta]),c = 'black',alpha = 0.5)
plt.scatter(1+np.ones(8),np.squeeze((np.array(power_animals_ipsi)*f)[:,alta]),c = 'black',alpha = 0.5)
plt.scatter(2+np.ones(6),np.squeeze((np.array(power_animals_contra)*f)[:,alta]),c = 'black',alpha = 0.5)

plt.xticks(ticks = [1,2,3], labels = ['Control','Ipsi','Contra'])
plt.ylabel('Gamma power')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')


s_power_contra,p_power_contra = stats.ttest_ind(np.squeeze((np.array(power_animals_ipsi)*f)[:,alta]),np.squeeze((np.array(power_animals_contra)*f)[:,alta]),alternative = 'less')
df_power_contra = power_animals_ipsi.shape[0]+power_animals_contra.shape[0]-2
s_power_control,p_power_control = stats.ttest_ind(np.squeeze((np.array(power_animals_ipsi)*f)[:,alta]),np.squeeze((np.array(power_animals_control)*f)[:,alta]),alternative = 'less')
df_power_control = power_animals_ipsi.shape[0]+power_animals_control.shape[0]-2
plt.yscale('log')
#plt.savefig('boxplots_telc.pdf')
    