# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from ecgdetectors import Detectors
from biosppy.signals import ecg
# %%
with open("../output/data/subject_1.json") as json_file:
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
out = ecg.ecg(signal=np.array(ecg_json['ECG1'][0:1000*10*4]), sampling_rate=1000, show=False)
# %%
rri = out['rpeaks']

fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
ax.plot((np.arange(1000*10*4) / 1000), np.array(ecg_json['ECG1'][0:1000*10*4]))
ax.scatter((np.array(rri) / 1000), np.array(ecg_json['ECG1'][0:1000*10*4])[rri], c='r')
plt.grid()
plt.ylabel("mV", fontsize=14)
plt.xlabel("Times (sec)", fontsize=14)
plt.show()
# %%
