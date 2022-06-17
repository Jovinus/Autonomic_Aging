import json
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import pandas as pd

from biosppy.signals import ecg
from IPython.display import display


def _get_interpolate_rri(rpeaks: np.ndarray) -> np.ndarray:
    """
    Interporlate given rri

    Args:
        rpeaks (np.ndarray): rpeaks

    Returns:
        np.ndarray: interporlated rpeaks
    """
    ## Interporation
    rri_interp = np.interp(np.arange(rpeaks[0], max(rpeaks), 250),
                           xp=rpeaks[1:],
                           fp=np.diff(rpeaks))
    return rri_interp


def extract_rri_biosppy(ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Extract RRI from raw ECG by biosppy

    Args:
        ecg_signal (np.ndarray): raw ecgs
        sampling_rate (int): sampling rate, usually 1000

    Returns:
        np.ndarray: interporlated rri
    """
    out = ecg.ecg(signal=ecg_signal,
                  sampling_rate=sampling_rate,
                  show=False)
    rpeaks = out['rpeaks']
    rri_interp = _get_interpolate_rri(rpeaks=rpeaks)
    return rri_interp


def extract_rri_neurokit2(ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Extract RRI from raw ECG by neurokit2

    Args:
        ecg_signal (np.ndarray): raw ecgs
        sampling_rate (int): sampling rate, usually 1000

    Returns:
        np.ndarray: interporlated rri
    """
    ## Sanitize ECG
    ecg_signal = nk.signal_sanitize(ecg_signal)
    
    ## ECG Cleaning
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    
    ## Extract Rpeaks
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned=ecg_cleaned, 
                                          sampling_rate=sampling_rate, 
                                          correct_artifacts=True)
    
    # ## Extract RRI
    # signals, info = nk.ecg_process(ecg_signal,
    #                                sampling_rate=sampling_rate)

    ## Get RRI from results
    rpeaks = rpeaks['ECG_R_Peaks']

    ## Interporlate RRI
    rri_interp = _get_interpolate_rri(rpeaks=rpeaks)
    return rri_interp


def extract_hrv_neurokit2(ecg_signal: np.ndarray, sampling_rate: int) -> pd.DataFrame:
    """
    Extract HRV from raw ECG by neurokit2

    Args:
        ecg_signal (np.ndarray): raw ecgs
        sampling_rate (int): sampling rate, usually 1000

    Returns:
        pd.DataFrame: HRV Features
    """
    ## Sanitize ECG
    ecg_signal = nk.signal_sanitize(ecg_signal)
    
    ## ECG Cleaning
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    
    ## Extract Rpeaks
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned=ecg_cleaned, 
                                          sampling_rate=sampling_rate, 
                                          correct_artifacts=True)
    
    # ## Extract RRI
    # signals, info = nk.ecg_process(ecg_signal,
    #                                sampling_rate=sampling_rate)

    ## Get RRI from results
    rpeaks = rpeaks['ECG_R_Peaks']

    ## Extract HRV
    hrv_table = nk.hrv(peaks=rpeaks, sampling_rate=sampling_rate, show=False)
    
    hrv_table = hrv_table.assign(RRI = lambda x: x["HRV_MeanNN"],
                                 SDNN = lambda x: x["HRV_SDNN"], 
                                 RMSSD = lambda x: x["HRV_RMSSD"], 
                                 pNN50 = lambda x: x["HRV_pNN50"], 
                                 TRI = lambda x: x["HRV_HTI"], 
                                 TINN = lambda x: x["HRV_TINN"], 
                                 logVLF = lambda x: np.log(x["HRV_VLF"]), 
                                 logLF = lambda x: np.log(x["HRV_LF"]), 
                                 LFnu = lambda x: x["HRV_LF"] / (x["HRV_LF"] + x["HRV_HF"]), 
                                 logHF = lambda x: np.log(x["HRV_HF"]), 
                                 HFnu = lambda x: x["HRV_HF"] / (x["HRV_LF"] + x["HRV_HF"]), 
                                 LF_HF = lambda x: x["HRV_LFHF"], 
                                 logTot = lambda x: np.log(x["HRV_VLF"] + x["HRV_LF"] + x["HRV_HF"]),
                                 ApEn = lambda x: x["HRV_ApEn"], 
                                 SampEn = lambda x: x["HRV_SampEn"], 
                                 a1 = lambda x: x["HRV_DFA_alpha1"], 
                                 a2 = lambda x: x["HRV_DFA_alpha2"], 
                                 Cordim = lambda x: x["HRV_CD"], 
                                 SD1 = lambda x: x["HRV_SD1"], 
                                 SD2 = lambda x: x["HRV_SD2"],
                                 )
    
    hrv_features = ["RRI", "SDNN", "RMSSD", "pNN50", "TRI", "TINN", 
                    "logVLF", "logLF", "LFnu", "logHF", "HFnu", "LF_HF", 
                    "logTot", "ApEn", "SampEn", "a1", "a2", "Cordim", "SD1", "SD2"]
    
    return hrv_table[hrv_features]


def main():
    """
    Test codes for rri segmentations
    """
    with open("../output/data/subject_12.json") as json_file:
        ecg_json = json.load(json_file)

    ecg = np.array(ecg_json['ECG1'])
    # rri = extract_rri_neurokit2(ecg_signal=ecg,
    #                             sampling_rate=1000)
    
    hrv = extract_hrv_neurokit2(ecg_signal=ecg,
                                sampling_rate=1000)
    
    display(hrv)

    # fig, ax = plt.subplots(1, 1, figsize=(20, 10), facecolor='white')
    # ax.plot(np.arange(0, len(rri)/4, 0.25) / 60, rri)
    # plt.grid()
    # plt.ylabel("RRI", fontsize=14)
    # plt.xlabel("Times (mins)", fontsize=14)
    # # plt.savefig("../figure/rri.png", dpi=500)
    # plt.show()

if __name__ == '__main__':
    main()