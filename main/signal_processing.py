import json
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

from biosppy.signals import ecg


def _get_interpolate_rri(rpeaks: np.ndarray) -> np.ndarray:
    """
    Interporlate given rri

    Args:
        rpeaks (np.ndarray): rpeaks

    Returns:
        np.ndarray: interporlated rpeaks
    """
    ## Interporation
    rri_interp = np.interp(np.arange(0, max(rpeaks), 250),
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
    ## Extract RRI
    signals, info = nk.ecg_process(ecg_signal,
                                   sampling_rate=1000)

    ## Get RRI from results
    rpeaks = info['ECG_R_Peaks']

    ## Interporlate RRI
    rri_interp = _get_interpolate_rri(rpeaks=rpeaks)
    return rri_interp


def main():
    """
    Test codes for rri segmentations
    """
    with open("../output/data/subject_12.json") as json_file:
        ecg_json = json.load(json_file)

    ecg = np.array(ecg_json['ECG1'])
    rri = extract_rri_neurokit2(ecg_signal=ecg,
                                sampling_rate=1000)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10), facecolor='white')
    ax.plot(np.arange(0, len(rri)/4, 0.25) / 60, rri)
    plt.grid()
    plt.ylabel("RRI", fontsize=14)
    plt.xlabel("Times (mins)", fontsize=14)
    # plt.savefig("../figure/rri.png", dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
