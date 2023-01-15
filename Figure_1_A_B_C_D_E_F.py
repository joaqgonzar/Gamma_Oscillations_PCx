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
from scipy import stats

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



def pac_cfc(lfp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin, mask):
    
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
        PhaseFreq=eegfilt(lfp,srate,Pf1,Pf2)
        analytic_signal = signal.hilbert(PhaseFreq)
        faselenta.append(np.angle(analytic_signal)[mask])
        
    
    fase_lenta = faselenta
    #fase_lenta = np.array(fase_lenta)
    
        
    #lfp1 = np.array(AmpFreqTransformed1)
    comodulogram = np.empty((len(lentaVector),len(altaVector)))
    
         
    for count_lenta,lenta in enumerate(fase_lenta):
            
        for count_rapida,rapida in enumerate(AmpFreqTransformed1):    
    
            comodulogram[count_lenta,count_rapida] = modulation_index(rapida,lenta,numbin)
                
    
    return(comodulogram)         

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
    #fase_lenta = np.array(fase_lenta)
    
        
    #lfp1 = np.array(AmpFreqTransformed1)
    comodulogram = np.empty((len(lentaVector),len(altaVector)))
    
         
    for count_lenta,lenta in enumerate(fase_lenta):
            
        for count_rapida,rapida in enumerate(AmpFreqTransformed1):    
    
            comodulogram[count_lenta,count_rapida] = modulation_index(rapida,lenta,numbin)
                
    
    return(comodulogram)       

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
    p = mean_amp/np.sum(mean_amp)

    return(p)

def pac_histogram_resp(lfp,resp,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin,mask):
    
    AmpFreqTransformed1 = []
    
    for ii in altaVector:
        Af1 = ii
        Af2=Af1+alta_BandWidth
        
        AmpFreq1=eegfilt(lfp,srate,Af1,Af2) # just filtering
        
        analytic_signal1 = signal.hilbert(AmpFreq1)
        AmpFreqTransformed1.append(np.abs(analytic_signal1)[mask])
        
    Pf1 = 2 
    Pf2 = Pf1 + lenta_BandWidth
    PhaseFreq=eegfilt(resp,srate,Pf1,Pf2)
    analytic_signal = signal.hilbert(PhaseFreq)
    faselenta = np.angle(analytic_signal)[mask]
        
    hist_freqs = []
    
    for count_rapida,rapida in enumerate(AmpFreqTransformed1):    
    
        hist_freqs.append(phase_amp_hist(rapida,faselenta,numbin))     
                
    
    return(hist_freqs)    
       

def csd_gamma(lfp,positions):
    
    gamma = []
    for x in positions:
        gamma.append(eegfilt(lfp[x,:],srate = 2000, flow = 30, fhigh = 50))
      
    gamma = np.array(gamma)
    
    peaks = signal.find_peaks(gamma[-1,:])[0]
    
    gamma_trig = []
    for x in peaks[100:-100]:
        if gamma[0,int(int(x)-1000):int(int(x)+1000)].shape[0] == 2000:
            gamma_trig.append(gamma[:,int(int(x)-1000):int(int(x)+1000)]) 
            
    gamma_trig = np.mean(gamma_trig,axis = 0)     
    
    csd = []
    
    for index in range(positions.shape[0]-2):
        
        pos = index+1
        top = index+2
        bottom = index
        
        csd.append((-gamma_trig[top,:]+(gamma_trig[pos,:]*2)-gamma_trig[bottom,:]))
    
    csd = np.array(csd)

    return(csd, gamma_trig)
#%% experiment data 

names = ['141208-1','141208-2','141209','160819','160820','170608','170609','170613','170614','170618','170619','170621','170622']

directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'

os.chdir(directory)

exp_data = pd.read_csv('ExperimentCatalog_Simul.txt', sep=" ", header=None)

awake_times = np.array([names,np.array(exp_data[2][1:14])])

# define electrode positions in the central array of the probe 

positions_2 = np.array([8,28,1,30,11,20,10,21,15,16,14,17])
positions_1 = np.array([9,29,0,31,10,21,11,20,14,17,15,16])

loading = exp_data[3][1:14]
 
#%% loop through animals

power_animals = []
pac_lfp_animals = []
pac_resp_animals = []
phase_amp_hist_animals = []
phase_amp_hist_resp_animals = []
granger_resp_gamma_animals = []
granger_gamma_resp_animals = []
trig_gamma_animals = []
power_resp_animals = []
coher_resp_animals = []
csd_animals = []
gamma_average_animals = []

index = 1

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
    
    lfp_odorless = lfp[:,lfp_mask]
    resp_odorless = resp[lfp_mask]
    
    
    csd,gamma_average = csd_gamma(lfp,positions)
    
    lfp = lfp[channel,:]
    lfp_odorless = np.squeeze(lfp_odorless[channel,:])    
    
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
     
    pac_lfp = pac_cfc(lfp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin,lfp_mask)
    pac_resp = pac_cfc_resp(lfp,resp,lentaVector,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin,lfp_mask)
    #phase_amplitude_hist = pac_histogram(lfp,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin, lfp_mask)
    phase_amplitude_hist_resp = pac_histogram_resp(lfp,resp,altaVector,lenta_BandWidth,alta_BandWidth,srate,numbin, lfp_mask)



    # get g causality 
    
    gamma_envelope = np.abs(signal.hilbert(eegfilt(lfp,srate, 30,60)))
    gamma_envelope = gamma_envelope[lfp_mask]
    gamma_test = np.squeeze(gamma_envelope)
    resp_test = np.squeeze(resp_odorless[0:gamma_envelope.shape[0]])
    
    matrix_test = np.vstack([np.diff(gamma_test),np.diff(resp_test)]).T
    lag = [10]
    gc_res = grangercausalitytests(matrix_test,lag,verbose = False)
    gc_resp_gamma = [gc_res[lag[0]][0]['ssr_ftest'][1],gc_res[lag[0]][0]['ssr_ftest'][0]]
    
    matrix_test_nul = np.vstack([np.diff(resp_test),np.diff(gamma_test)]).T
    lag = [10]
    gc_res = grangercausalitytests(matrix_test_nul,lag,verbose = False)
    gc_gamma_resp = [gc_res[lag[0]][0]['ssr_ftest'][1],gc_res[lag[0]][0]['ssr_ftest'][0]]
    
    
    
    
    # get respiration triggered gamma envelope

    gamma_resp_average = []
    for x in np.squeeze(inh_no_odor):
         if gamma_test[int(x-2000):int(x+2000)].shape[0] == 4000:   
            gamma_resp_average.append(gamma_test[int(x)-2000:int(x)+2000])
      
            
      
    # save results
    
    power_animals.append(pxx)
    power_resp_animals.append(pxx_resp)
    coher_resp_animals.append(cxx)
    pac_lfp_animals.append(pac_lfp)
    pac_resp_animals.append(pac_resp)
    #phase_amp_hist_animals.append(phase_amplitude_hist)
    phase_amp_hist_resp_animals.append(phase_amplitude_hist_resp)
    granger_resp_gamma_animals.append(gc_resp_gamma)
    granger_gamma_resp_animals.append(gc_gamma_resp)
    trig_gamma_animals.append(np.mean(gamma_resp_average,axis = 0))
    csd_animals.append(csd)
    gamma_average_animals.append(gamma_average)
    
    index = index + 1
    #del lfp, pac_lfp, pac_resp, pxx
    
    
#%% save

os.chdir(directory)
np.savez('Figure_1_b_c_d_e_f.npz',power_animals = power_animals, power_resp_animals = power_resp_animals, coher_resp_animals = coher_resp_animals, pac_lfp_animals = pac_lfp_animals, pac_resp_animals = pac_resp_animals, granger_resp_gamma_animals = granger_resp_gamma_animals, granger_gamma_resp_animals = granger_gamma_resp_animals, trig_gamma_animals = trig_gamma_animals, phase_amp_hist_resp_animals = phase_amp_hist_resp_animals, csd_animals = csd_animals, gamma_average_animals = gamma_average_animals)

#%% load data 
directory = '/run/user/1000/gvfs/afp-volume:host=NAS_Sueno.local,user=pcanalisis2,volume=Datos/Frank_Boldings_Dataset'
os.chdir(directory)
results = np.load('Figure_1_b_c_d_e_f.npz')

power_animals = results['power_animals']
power_resp_animals = results['power_resp_animals']
coher_resp_animals = results['coher_resp_animals']
pac_lfp_animals = results['pac_lfp_animals']
pac_resp_animals = results['pac_resp_animals']
granger_gamma_resp_animals = results['granger_gamma_resp_animals']
granger_resp_gamma_animals = results['granger_resp_gamma_animals']
granger_gamma_resp_animals = results['granger_gamma_resp_animals']
granger_resp_gamma_animals = results['granger_resp_gamma_animals']
#phase_amp_hist_animals = results['phase_amp_hist_animals']
phase_amp_hist_resp_animals = results['phase_amp_hist_resp_animals']
trig_gamma_animals = results['trig_gamma_animals']
csd_animals = results['csd_animals']
gamma_average_animals = results['gamma_average_animals']
#%% plot example recordings

spec_aniamls = []

name = names[0]

os.chdir(directory+'/Simul/Decimated_LFPs/Simul/Decimated_LFPs')

lfp = np.load('PFx_lfp_awake_'+name+'_.npz',allow_pickle=True)['lfp_downsample']
    
srate = 2000
    
os.chdir(directory+'/Simul/processed/'+name)
resp = scipy.io.loadmat(name+'.mat')['RRR']
srate_resp = 2000

# 300000
start = 1500000-300
start = 1500000+79000
start = 1500000+354000
length = 4500
window = np.arange(start,int(start+length))

lfp_plot = lfp[16,window]
resp_plot = resp[window]

#
w = 12.
freq = np.linspace(1, int(srate/2), int(srate/2))
widths = w*srate / (2*freq*np.pi)
    
spec = np.abs(signal.cwt(lfp_plot, signal.morlet2, widths, w=w))

    

plt.figure(dpi = 300, figsize = (8,8))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.subplot(312)
plt.plot(lfp_plot)
plt.xlim([0,length])

plt.subplot(311)
plt.plot(resp_plot)
plt.xlim([0,length])
    
    
plt.subplot(313)
plt.imshow(spec,interpolation = 'gaussian',cmap = 'jet',vmin = 300, vmax = 3000,aspect = 'auto',alpha = 1)

    
plt.ylim([30,80])

plt.ylabel('Frequency (Hz)')

plt.tight_layout()
     
plt.xlabel('Time (s)')


#%% plot power spectrum

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize = (4,3))

ax1 = plt.subplot(111)
f = np.arange(0,1000.1,0.1)
f = f[5:]

mean_power = np.mean(power_animals[:,5:],axis = 0)*f
error_power = (np.std(power_animals[:,5:],axis = 0)*f)/np.sqrt(13)


# remove line noise
line_freq = int(np.where(f == 60)[0])
mean_power[line_freq-20:line_freq+20] = (mean_power[line_freq-60:line_freq-20]+mean_power[line_freq+20:line_freq+60])/2
error_power[line_freq-20:line_freq+20] = (error_power[line_freq-60:line_freq-20]+error_power[line_freq+20:line_freq+60])/2

plt.plot(f,mean_power,linewidth = 1.5, color = 'black', label = 'LFP power')
plt.fill_between(f,mean_power-2*error_power,mean_power+2*error_power,color = 'black',alpha = 0.3,edgecolor=None)
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both', which = 'both')
plt.xscale('log')
plt.xlim([0.5,300])
plt.ylim([10e2,0.7e5])
plt.ylabel('Whitened Power')
plt.xlabel('Frequency(Hz)')

ax2 = ax1.twinx()
mean_coher = np.mean(coher_resp_animals[:,5:],axis = 0)


ax2.plot(f,mean_coher, label = 'LFP-Resp Coher')
plt.ylim([0,0.2])
plt.xlim([0.5,300])
#plt.yscale('log')
plt.legend(fontsize = 6)

plt.tight_layout()
plt.savefig('power_spectrum.pdf')

#%% plot pac comodulograms

# pac parameters

lentaVector = np.arange(0.2,10,1)
altaVector = np.arange(10,150,10)
lenta_BandWidth = 2
alta_BandWidth = 10
numbin = 18

plt.figure(dpi = 300,figsize = (12,4))

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.subplot(131)

mean_power = np.mean(power_animals,axis = 0)*f
mean_power_resp = np.mean(power_resp_animals,axis = 0)

mean_coher = np.mean(coher_resp_animals,axis = 0)
data_plot = np.mean(pac_lfp_animals,axis = 0)

vmin = 0.0003
vmax = 0.0014

extent = [lentaVector[0]+lenta_BandWidth/2,lentaVector[-1]+lenta_BandWidth/2,altaVector[-1]+alta_BandWidth/2,altaVector[0]+alta_BandWidth/2]
plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')
plt.plot(f,(mean_power*0.005)-130,color = 'white',linewidth=1.5,label = 'LFP Power')
plt.plot(f,(mean_coher*600)+10,'--',color = 'white',linewidth=1.5,label = 'LFP-Resp') 
legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('LFP Phase Frequency (Hz)')
plt.ylabel('LFP Amp Frequency (Hz)')

plt.subplot(132)

extent = [lentaVector[0]+lenta_BandWidth/2,lentaVector[-1]+lenta_BandWidth/2,altaVector[-1]+alta_BandWidth/2,altaVector[0]+alta_BandWidth/2]


vmin = 0.0012
vmax = 0.005

data_plot = np.mean(pac_resp_animals,axis = 0)
plt.imshow(data_plot.T,extent = extent,interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')
plt.plot(f,(mean_power_resp*0.001)+10,color = 'white',linewidth=1.5,label = 'Resp Power')

legend = plt.legend(frameon=False,prop={'size': 8})
plt.setp(legend.get_texts(), color='w')

plt.colorbar()

plt.yticks(ticks = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.round((lentaVector+lenta_BandWidth/2)-0.2).astype(int))
plt.xlim([1.2,8.2])
plt.ylim([15,125])

plt.xlabel('Resp Phase Frequency (Hz)')
plt.ylabel('LFP Amp Frequency (Hz)')

plt.subplot(133)

t = np.linspace(0,36,1000)
sine_wave = 2*np.sin(2*np.pi*t*0.055)+8

extent = [0,1080,altaVector[-1]+alta_BandWidth/2,altaVector[0]+alta_BandWidth/2]

vmin = 0.05
vmax = 0.068
pac_plot = np.mean(phase_amp_hist_resp_animals, axis = 0)
plt.imshow(np.hstack([pac_plot,pac_plot,pac_plot]),interpolation = 'gaussian',cmap = 'jet',vmin = vmin, vmax = vmax,aspect = 'auto')

plt.yticks(ticks = np.arange(0.5,14.5),labels = ((altaVector+alta_BandWidth/2)+5).astype(int))
plt.xticks(ticks = np.arange(16,52,3),labels = np.round(np.arange(0,720,60)).astype(int),rotation = 30)
plt.plot(np.linspace(16,52,1000),sine_wave,color = 'white',linewidth = 4)
plt.ylim([0,11])
plt.xlim([16,52])

plt.ylabel('Norm Amp Frequency (Hz)')
plt.xlabel('Resp Phase (deg)')

plt.colorbar()


plt.tight_layout()

#plt.savefig('pac_comodulograms_histograms.pdf')

#%% plot directionality results

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.figure(dpi = 300, figsize =(3,5))


plt.subplot(211)

mean_ccg = np.mean(trig_gamma_animals,axis = 0)
err_ccg = np.std(trig_gamma_animals,axis = 0)/np.sqrt(13)

plt.fill_between(np.arange(-1,1,1/2000),mean_ccg-err_ccg,mean_ccg+err_ccg,alpha = 0.2, color = 'black')
plt.plot(np.arange(-1,1,1/2000),mean_ccg, color = 'black')
plt.vlines(0,150,285, linestyle = 'dashed',color = 'black')

plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')
plt.ylim([150,285])
plt.xlim([-1,1])
plt.xlabel('Time from inh start (s)')
plt.ylabel('Gamma Amplitude (a.u.)')


plt.subplot(212)
plt.boxplot([np.log(np.array(granger_resp_gamma_animals)[:,1]),np.log(np.array(granger_gamma_resp_animals)[:,1])], widths = 0.2, showfliers=False)

for x in range(13):
    plt.plot([1.2,1.8],[np.log(np.array(granger_resp_gamma_animals)[:,1])[x],np.log(np.array(granger_gamma_resp_animals)[:,1])[x]], color = 'grey')
    
plt.ylabel('Phase angle ')
plt.xticks(ticks = [1,2], labels = ['Winners','Losers'])
plt.xlim([0.8,2.2])
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.xticks(ticks = [1,2], labels = ['Resp->Gamma','Gamma->Resp'], rotation = 0)
plt.ylabel('Granger Causality')
plt.grid(color='grey', linestyle='--', linewidth=0.7 ,alpha = 0.3, axis = 'both', which = 'Both')

plt.tight_layout()

#plt.savefig('gamma_resp_dir.pdf')

# stats

s_granger, p_granger = stats.ttest_rel(np.log(np.array(granger_resp_gamma_animals)[:,1]),np.log(np.array(granger_gamma_resp_animals)[:,1]), alternative = 'greater')
df_granger = np.array(granger_resp_gamma_animals)[:,1].shape[0]-1

#%% plot current source density  

from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


plt.figure(dpi = 300, figsize =(4,6))

gs = GridSpec(1, 2, width_ratios=(2,1))

plt.subplot(gs[0])

vmin = -10
vmax = 10

csd_animals = np.array(csd_animals)

csd_mean = np.mean(csd_animals,axis = 0)
gamma_mean = np.mean(gamma_average_animals,axis = 0)
length = 2000

plt.imshow(np.flipud(csd_mean[2:,:]),interpolation = 'gaussian',cmap = 'jet', aspect = 'auto', vmin = vmin, vmax = vmax)

vert_sep = 7.5
for x in range(10):
    plt.plot(np.arange(0,length),vert_sep+gamma_mean[x+2,:].T*0.005, color = 'black',linewidth = 1.8)
    vert_sep = vert_sep-0.9
    
    
plt.ylim([-2.5,10])    


plt.xlim([900,1100])
plt.box(None)
plt.yticks([])
plt.xticks([])
plt.text(1120,-0.5,'25 $\mu$m',rotation = 'vertical')
plt.vlines(1105,-0.5,0.4, color = 'black', linewidth = 8)
plt.plot(np.arange(1000,1100),np.zeros(100)-1.9,color = 'black', linewidth = 2)
plt.text(1000,-1.5,'50 ms',rotation = 'horizontal')


plt.subplot(gs[1])

csd_means = np.mean(csd_animals[:,2:,1000], axis = 0)
csd_errors = np.std(csd_animals[:,2:,1000], axis = 0)/np.sqrt(13)

plt.barh(np.arange(0,8),np.flipud(csd_means), color = 'grey')
plt.errorbar(np.flipud(csd_means), np.arange(0,8), xerr = np.flipud(csd_errors), fmt = 'none',color = 'black')
plt.vlines(0,-1,8, color = 'black', linestyles = 'dashed')
plt.grid(color='grey', linestyle='--', linewidth=1 ,alpha = 0.3, axis = 'both')

plt.xlim([-10,15])
plt.ylim([-2.5,10])
plt.box(None)    
plt.yticks(ticks = np.arange(0,8), labels = [])
plt.xticks(ticks = np.arange(-10,15,10), labels = np.arange(-10,15,10))

#plt.savefig('csd_gamma.pdf')

# stats

pvals_csd = []
svals_csd = []

for x in range(csd_animals.shape[1]):
    s,p = scipy.stats.ttest_1samp(csd_animals[:,x,1000], popmean=0,alternative = 'greater')
    pvals_csd.append(p)
    svals_csd.append(s)