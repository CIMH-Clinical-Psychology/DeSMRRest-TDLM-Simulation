#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 07:27:21 2025

helper script to run participant preprocessing on the cluster.
You can either run it as a __main__ by calling

@author: simon.kern
"""
import numpy as np
import sys
import settings
# overwrite settings dir
settings.cache_dir = '/zi/home/simon.kern/joblib-resting-state/'

import utils
from settings import  cache_dir
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2


#%% Settings
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]

sfreq = 100  # downsample to this frequency. Changing is not supported.
bands = settings.bands_HP  # only use HP filter

final_calculation = True  # this can be set to use the leftout data of the RS

#%% preload some data (e.g. localizer)

def preprocess_participants(subj):
    print(f'running preprocessing for {subj=}')
    # data used for the localizer
    load_localizers_seq12(subj=subj, sfreq=sfreq, bands=bands, autoreject=settings.default_autoreject, ica=settings.default_ica_components)
    # negative examples from the fixation cross before audio cue onset
    load_neg_x_before_audio_onset(subj=subj, sfreq=sfreq, bands=bands,  autoreject=settings.default_autoreject, ica=settings.default_ica_components)
    # resting state data, both eyes open and eyes closed together
    load_RS1(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)
    load_RS2(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)

    # individual sequences for the trials (maybe not necessary)
    utils.get_sequences(subj)
    print(f'Done loading for {subj=}')

if __name__=='__main__':

    print(sys.argv)

    if len(sys.argv)==1:
        for subj_id in range(1, 32):
            if subj_id == 8:
                continue  # missing
            np.random.seed(subj_id)  # for safety
            subj = f'DSMR1{subj_id:02d}'
            try:
                preprocess_participants(subj)
            except Exception as e:
                print(f'ERROR for {subj=}: {e}')

    elif len(sys.argv)==2:
      subj_id = int(sys.argv[1])
      np.random.seed(subj_id)  # for safety

      subj = f'DSMR1{subj_id:02d}'
      preprocess_participants(subj)
    else:
        raise ValueError('either supply one participant number as argument (1-31) or none to process all')
