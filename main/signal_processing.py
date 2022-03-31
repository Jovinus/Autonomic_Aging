# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from biosppy.signals import ecg

# %%

def _get_interpolate_rri(rpeaks:np.ndarray) -> np.ndarray:
    rri_interp = np.interp(np.arange(0, max(rpeaks), 250), xp=rpeaks[1:], fp=np.diff(rpeaks))
    return rri_interp



def extract_rri(ecg_signal:np.ndarray, sampling_rate:int) -> np.ndarray:
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    rpeaks = out['rpeaks']
    rri_interp = _get_interpolate_rri(rpeaks=rpeaks)
    return rri_interp


# %%
def main():
    with open("../output/data/subject_1.json") as json_file:
        ecg_json = json.load(json_file)
        
    ecg = np.array(ecg_json['ECG1'])
    rri = extract_rri(ecg_signal=ecg, sampling_rate=1000)

    fig, ax = plt.subplots(1, 1, figsize=(20,10), facecolor='white')
    ax.plot(np.arange(0, len(rri)/4, 0.25) / 60, rri)
    plt.grid()
    plt.ylabel("mV", fontsize=14)
    plt.xlabel("Times (mins)", fontsize=14)
    plt.savefig("../figure/rri.png", dpi=500)
    plt.show()

# %%
if __name__ == '__main__':
    main()