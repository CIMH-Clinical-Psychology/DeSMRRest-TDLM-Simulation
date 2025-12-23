# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:05:07 2022

This file contains convenience loading functions for different situations and
analysis scenarios. However, this repository only contains the subset
necessary for reproducing the manuscript results

@author: Simon Kern
"""
import logging
import warnings

import numpy as np

import settings

from meg_tools import log_append, load_epochs_bands
from utils import list_files, get_id, get_performance
from utils import num2char, char2num, get_sequences
from meg_tools import load_events, load_segments_bands

default_kwargs = {
    "ica": settings.default_ica_components,
    "sfreq": 100,
    "picks": "meg",
    "bands": settings.default_bands,
    "autoreject": settings.default_autoreject,
}


def filter_trials(data_y, seq, pos):
    """some trials are rejected due to artefact. this function
    harmonizes the trials and seqs that were shown so that they match

    returns a mask with True if the seq is kept, or False if it was rejected
    """
    i_checked=0
    mask = []
    for y in data_y:
        while y!=seq[i_checked][pos-1]:
            i_checked+=1
            mask.append(False)
        mask.append(True)
        i_checked+=1
    while len(mask)<len(seq):
        mask += [False]
    return mask

def stratify_data(data_x, data_y, mode="bootstrap"):
    """
    stratify a dataset such that each class in data_y is contained the same
    times as all other classes. Two modes are supplied, either truncate
    or bootstrap (notimplemented)
    """
    min_items = np.unique(data_y, return_counts=True)[1].min()
    data_x = np.vstack(
        [data_x[np.where(data_y == i)[0][:min_items]] for i in np.unique(data_y)]
    )
    data_y = np.hstack(
        [data_y[np.where(data_y == i)[0][:min_items]] for i in np.unique(data_y)]
    )
    return data_x, data_y


def load_localizers_seq12(subj, **kwargs):
    kwargs = dict(default_kwargs.copy(), **kwargs)
    ica = kwargs["ica"]
    picks = kwargs["picks"]
    sfreq = kwargs["sfreq"]
    stratify = kwargs.get('stratify', True)

    bands = kwargs["bands"]
    tmin = kwargs.get("tmin", -0.1)
    tmax = kwargs.get("tmax", 0.5)

    autoreject = kwargs.get("autoreject")
    event_ids = kwargs.get("event_ids", np.arange(1, 11))
    files = list_files(settings.data_dir, patterns=f"{subj}_localizer*fif")
    assert len(files) >= 2
    # logging.info("loading localizer")
    data_localizer = [
        load_epochs_bands(
            f,
            bands,
            n_jobs=1,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            ica=ica,
            autoreject=autoreject,
            picks=picks,
            event_ids=event_ids,
        )
        for f in files
    ]
    data_x, data_y = [
        np.vstack([d[0] for d in data_localizer]),
        np.hstack([d[1] for d in data_localizer]),
    ]
    if stratify:
        data_x, data_y = stratify_data(data_x, data_y)
    log_append(files[0], "load_func", {"data_x.shape": data_x.shape, "data_y": data_y})
    return [data_x, data_y]


def load_neg_x_before_audio_onset(subj, **kwargs):
    """loads negative examples from the localizer before audio onset"""
    # fixation cross before audio onset 750ms long
    kwargs.update({"event_ids": [98],
                   "stratify": False})
    data_x, _ = load_localizers_seq12(subj, tmin=0, tmax=0.5, **kwargs)
    return data_x


def _load_RS(subj, patterns, final_calculation=False, n_seg=None, **kwargs):
    # logging.info(f'loading RS {patterns=}, {final_calculation=}')
    kwargs = dict(default_kwargs.copy(), **kwargs)
    ica = kwargs['ica']
    picks = kwargs['picks']
    sfreq = kwargs['sfreq']
    bands = kwargs['bands']

    rs_file = list_files(settings.data_dir, patterns=patterns)
    assert len(rs_file)==1
    rs = load_segments_bands(rs_file[0], bands=bands, sfreq=sfreq, ica=ica,
                             picks=picks).T
    # don't need to take only middle segment.
    # the end comes abrupt, so ne end effect
    # beginning should be clear as well
    # rs = rs[int(rs.shape[0]*0.05):int(rs.shape[0]*0.95)] # take middle 90% of data
    subj_id = get_id(subj)

    first_seg = 0 if subj_id<=117 else 1
    if final_calculation:
        # warnings.warn('Running on left-out data of RS')
        first_seg = int(not first_seg) # shift order if running on validation set

    log_append(rs_file[0], 'load_func', {'final_calculation':final_calculation,
                                        'first_seg':first_seg,
                                        'n_seg': n_seg,
                                        'patterns':patterns})

    # the resting state should be around 8 minutes long. In the preregistration
    # we commited to take interleaved 30-second segments. To approximately
    # match that by dividing the segments into 16 segments.
    n_segs = 8*60//30  # 8 minutes by 30 second intervals
    n_segsamp = len(rs)//n_segs
    rs_segs = [rs[i*n_segsamp:(i+1)*n_segsamp] for i in range(first_seg, n_segs, 2)][:n_seg]
    assert len(rs_segs)==8, 'fewer than 8 segements!'
    assert all([abs((len(seg)//sfreq)-30)<2 for seg in rs_segs]), 'some segment is more or less from 30 seconds long'
    rs_stacked = np.vstack(rs_segs)
    return rs_stacked

def load_RS1(subj, **kwargs):
    return _load_RS(subj, patterns=f'{subj.upper()}_RS1*sss*fif', **kwargs)

def load_RS2(subj, **kwargs):
    return _load_RS(subj, patterns=f'{subj.upper()}_RS2*sss*fif', **kwargs)

if __name__ == "__main__":
    # debugging purposes
    kwargs = {
        "ica": settings.default_ica_components,
        "picks": "meg",
        "sfreq": 100,
        "bands": settings.bands_HP,
    }
