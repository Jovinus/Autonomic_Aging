# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from ecgdetectors import Detectors
from biosppy.signals import ecg
import neurokit2 as nk
# %%
with open("../output/dataset/rri_hrv_data/subject_3_0.json") as json_file:
    ecg_json = json.load(json_file)

# %%
detectors = Detectors(1000)
# %%
rri = detectors.christov_detector(np.array(ecg_json['ECG1'][0:1000*10*1]))
# %%
fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot((np.arange(1000*10*1) / 1000), np.array(ecg_json['ECG1'][0:1000*10*1]))
ax.scatter((np.array(rri) / 1000), np.array(ecg_json['ECG1'][0:1000*10*1])[rri], c='r')
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.show()

# %%
out = ecg.ecg(signal=np.array(ecg_json['ECG1'][0:1000*10*1]), sampling_rate=1000, show=False)
rri = out['rpeaks']
ecg_filtered = out['filtered']

fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot((np.arange(1000*10*1) / 1000), ecg_filtered[0:1000*10*1])
ax.scatter((np.array(rri) / 1000), ecg_filtered[0:1000*10*1][rri], c='r')
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.savefig("../figure/rpeak_ecg.png", dpi=500)
plt.show()

# %% Raw RRI
out = ecg.ecg(signal=np.array(ecg_json['ECG1'][0:1000*60*18]), sampling_rate=1000, show=False)

rri = out['rpeaks']

fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot(np.diff(rri))
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.savefig("../figure/rri.png", dpi=500)
plt.show()

# %% Interporlated RRI
fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot(np.arange(0, 60*5*1000, 250) / 1000, np.interp(np.arange(0, 60*5*1000, 250), xp=rri[1:], fp=np.diff(rri)))
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.savefig("../figure/rri.png", dpi=500)
plt.show()
# %%
ecg_filtered = out['filtered']
# %% Neurokit2
test = nk.ecg_process(ecg_json['ECG1'], sampling_rate=1000)
# %%
fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot(np.arange(0, 60*5*1000, 250) / 1000, np.interp(np.arange(0, 60*5*1000, 250), xp=test[1]['ECG_R_Peaks'][1:], fp=np.diff(test[1]['ECG_R_Peaks'])))
plt.grid()
plt.ylabel("RRI", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.savefig("../figure/rri.png", dpi=500)
plt.show()
# %%

plt.rcParams['figure.figsize'] = [15, 9] 


with open("../output/data/subject_3.json") as json_file:
    ecg_json = json.load(json_file)

# %%
signals, info = nk.ecg_process(ecg_json['ECG1'][0:1000*60*2], sampling_rate=1000)

nk.ecg_plot(signals, sampling_rate=1000)
# %%
r_peaks = signals["ECG_Clean"][signals['ECG_R_Peaks'].astype(bool)]
fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot(signals['ECG_Clean'])
ax.scatter(r_peaks.index, r_peaks, c='r')
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.show()


# %%
with open("../output/dataset/rri_hrv_data/subject_3_0.json") as json_file:
    ecg_json = json.load(json_file)

# %%
fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot(ecg_json['RRI'])
plt.grid()
plt.ylabel("RRI", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.show()
# %%
