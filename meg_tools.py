# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:23:02 2020

@author: Simon Kern
"""
import hashlib
import logging
import os
import warnings
import random

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import picard  # needed for ICA, just imported to see the error immediately
import seaborn as sns
from joblib import Memory, Parallel, delayed
from mne.preprocessing import ICA, read_ica
from scipy.stats import zscore
from collections import namedtuple
from autoreject import get_rejection_threshold
import settings
from settings import (
    cache_dir,
    caching_enabled,
    default_ica_components,
    default_normalize,
    results_dir,
)

memory = Memory(cache_dir if caching_enabled else None)
logging.getLogger().setLevel(logging.INFO)


def log_append(file, key, message):
    """appends data to a logfile in json format

    :param logfile: which file to log to
    :param file: the main file that the log does belong to; that it describes
    :param key: subkey to log to
    :param message: message to save to the logfile to this key

    """
    return
    warnings.warn('REMOVE THIS')
    from utils import get_id, json_dump, json_load

    logfile = results_dir + f"/log/DSMR{get_id(file)}.json"

    if os.path.exists(logfile):
        content = json_load(logfile)
    else:
        content = {}

    filename = os.path.basename(file).split("_")[1]
    if filename not in content:
        content[filename] = {}

    content[filename][key] = message

    json_dump(content, logfile, indent=4, ensure_ascii=True)


def plot_sensors(values, title="Sensors active", mode="size", color=None,
                 ax=None, vmin=None, vmax=None, cmap="Reds", **kwargs,):
    raise DeprecationWarning('moved to meg_utils')

def hash_array(arr, dtype=np.int64):
    raise DeprecationWarning('moved to meg_utils')

def sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"]):
    raise DeprecationWarning('moved to meg_utils')

def repair_epochs_autoreject(raw, epochs, ar_file, picks="meg"):
    raise DeprecationWarning('moved to meg_utils')


def make_meg_epochs(raw, events, tmin=-0.1, tmax=0.5, autoreject=True, picks="meg"):
    """
    Loads a FIF file and cuts it into epochs, normalizes data before returning
    along the sensor timeline

    Parameters
    ----------
    raw : mne.Raw
        a raw object containing epoch markers.
    tmin : float, optional
        Start time before event. Defaults to -0.1.
    tmax : float, optional
        DESCRIPTION. End time after event. Defaults to 0.5.

    Returns
    -------
    data_x : TYPE
        data in format (n_epochs, n_sensors, timepoints).
    data_y : TYPE
        epoch labels in format (n_epochs)
    """
    # create epochs based on event ids
    # events[:, 0] += raw.first_samp

    epochs = mne.Epochs(
        raw,
        events=events,
        picks=picks,
        preload=True,
        proj=False,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        on_missing="warn",
        verbose=False,
    )

    # assert autoreject==True, 'removed the version that rejects data only'

    if autoreject:
        HP, LP = np.round(epochs.info["highpass"], 2), np.round(
            epochs.info["lowpass"], 2
        )
        event_ids = epochs.events[:, 2].astype(np.int64)
        arr_hash = hash_array(event_ids)
        basename = settings.cache_dir + "/" + os.path.basename(raw.filenames[0])
        ar_file = f"{basename}-HP{HP:.1f}-LP{LP:.1f}-{tmin}-{tmax}-{picks}-{arr_hash}.autoreject"
        epochs = repair_epochs_autoreject(raw, epochs, ar_file, picks=picks)

    data_x = epochs.get_data(copy=True)
    data_y = epochs.events[:, 2]
    data_x = default_normalize(data_x)
    return data_x, data_y


@memory.cache(verbose=0)
def load_events(file, event_ids=None):
    """retrieve the events in chronological order from a file"""
    raise Exception('moved to meg_utils')

def load_segments(file, sfreq=100, markers=[[10, 11]], picks='meg',
                  filter_func='lambda x:x', ica=None, verbose='ERROR'):
    """
    Load interval of data between two markers

    Parameters
    ----------
    file : str
        FIF file to load.
    sfreq : int, optional
        frequency to which to downsample. The default is 100.
    markers : list of list of 2 ints, optional
        which trigger values to take the segment between. The default is [[10, 11]].
        the first and last occurence of the marker is taken as the segment length
    slicer : TYPE, optional
        DESCRIPTION. The default is None.
    picks : str or int, optional
        string or int for which channels to load. The default is 'meg'.
    filter_func : str, optional
        lambda string for filtering the segments. The default is 'lambda x:x'.
    ica : int, optional
        how many ICA components to discard. The default is None.
    verbose : str, optional
        MNE verbose marker. The default is 'ERROR'.

    Returns
    -------
    segments : TYPE
        DESCRIPTION.

    """
    raise DeprecationWarning('moved to meg_utils')


@memory.cache
def load_segments_bands(file, bands, sfreq=100, markers=[[10, 11]], picks='meg',
                        ica=settings.default_ica_components, verbose='ERROR',
                        n_jobs=3):
    raise DeprecationWarning('moved to meg_utils')



@memory.cache(ignore=["n_jobs"])
def load_epochs_bands(
    file,
    bands,
    sfreq=100,
    event_ids=None,
    tmin=-0.1,
    tmax=0.5,
    ica=default_ica_components,
    autoreject=True,
    picks="meg",
    event_filter=None,
    n_jobs=1,
):

    raise DeprecationWarning('moved to meg_utils')



def load_epochs(
    file,
    sfreq=100,
    event_ids=None,
    event_filter=None,
    tmin=-0.1,
    tmax=0.5,
    ica=default_ica_components,
    autoreject=True,
    filter_func="lambda x:x",
    picks="meg",
):
    """
    Load data from FIF file and return into epochs given by MEG triggers.
    stratifies the classes, that means each class will have the same
    number of examples.
    """
    raise DeprecationWarning('moved to meg_utils')


def fif2edf(fif_file, chs=None, edf_file=None):
    """
    converts a FIF file to an EDF file using pyedflib
    """
    from pyedflib import highlevel

    raw = mne.io.read_raw_fif(fif_file, preload=True)

    if chs is None:
        n_chs = len(raw.ch_names)
        # load n chandom channels
        load_n_channels = 6
        # load a maximum of 16 channels
        if n_chs <= load_n_channels:
            chs = list(0, range(load_n_channels))
        else:
            chs = np.unique(np.linspace(0, n_chs // 2 - 2, load_n_channels).astype(int))
        chs = [x for x in chs]

        try:
            chs += [raw.ch_names.index("STI101")]
        except Exception:
            pass

    if edf_file is None:
        edf_file = fif_file + ".edf"

    # create the stimulations as annotations
    sfreq = raw.info["sfreq"]
    events = (
        mne.find_events(raw, shortest_event=1, stim_channel="STI101").astype(float).T
    )
    events[0] = (events[0] - raw.first_samp) / sfreq
    annotations = [[s[0], -1 if s[1] == 0 else s[1], str(int(s[2]))] for s in events.T]

    # create stimulation from stim channels instead of events
    stim = raw.copy().pick("stim").get_data().flatten()
    trigger_times = np.where(stim > 0)[0] / sfreq
    trigger_desc = stim[stim > 0]
    where_next = [0] + [x for x in np.where(np.diff(trigger_times) > 1 / sfreq * 2)[0]]
    trigger_times = trigger_times[where_next]
    trigger_desc = trigger_desc[where_next]
    annotations2 = [
        (t, -1, "STIM " + str(d))
        for t, d in zip(trigger_times, trigger_desc, strict=True)
    ]

    picks = raw.pick(chs)
    data = raw.get_data()
    data = zscore(data, 1)
    data = np.nan_to_num(data)
    ch_names = picks.ch_names

    header = highlevel.make_header(technician="fif2edf-skjerns")
    header["annotations"] = annotations

    signal_headers = []
    for name, signal in zip(ch_names, data, strict=True):
        pmin = signal.min()
        pmax = signal.max()
        if pmin == pmax:
            pmin = -1
            pmax = 1
        shead = highlevel.make_signal_header(
            name, sample_rate=sfreq, physical_min=pmin, physical_max=pmax
        )
        signal_headers.append(shead)

    highlevel.write_edf(edf_file, data, signal_headers, header=header)

def get_ch_neighbours(ch_name, n=9, return_idx=False, plot=False):
    """retrieve the n neighbours of a given electrode location.
    Count includes the given origin electrode location"""
    raise DeprecationWarning('moved to meg_utils')





@memory.cache(verbose=0)
def load_events(file, event_ids=None):
    """retrieve event markers in chronological order from a mne readable file
    Parameters
    ----------
    file : str
        filepath of raw file, e.g. .fif or .edf.
    event_ids : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    np.ndarray
        mne.events array (n,3) -> (time, duration, event_id).

    """

    raw = mne.io.read_raw(file)
    min_duration = 3/raw.info['sfreq'] # our triggers are ~5ms long
    events = mne.find_events(raw, min_duration=min_duration,
                                 consecutive=False, verbose='WARNING')
    if event_ids is None:
        event_ids = np.unique(events[:,2])
    event_mask = [e in event_ids for e in events[:,2]]

    return events[event_mask,:]


def load_meg(file, sfreq=100, ica=None, filter_func="lambda x:x", verbose="ERROR"):
    """
    Load MEG data and applies preprocessing to it (resampling, filtering, ICA)

    Parameters
    ----------
    file : str
        Which MEG file to load.
    sfreq : int, optional
        Resample to this sfreq. The default is 100.
    ica : int or bool, optional
        Apply ICA with the number of components as ICA. The default is None.

    filter_func : str, func, optional
        a lambda string or function that will be applied
        to filter the data. The default is 'lambda x:x' (no filtering).

    Returns
    -------
    raw, events : mne.io.Raw, np.ndarray
        loaded raw file and events for the file correctly resampled

    """

    @memory.cache(ignore=["verbose"])
    def _load_meg_ica(file, sfreq, ica, verbose):
        """loads MEG data, calculates artefacted epochs before"""
        from utils import get_id

        tstep = 2.0  # default ICA value for tstep
        raw = mne.io.read_raw_fif(file, preload=True, verbose=verbose)
        raw_orig = raw.copy()
        min_duration = 3 / raw.info["sfreq"]  # our triggers are ~5ms long
        events = mne.find_events(
            raw, min_duration=min_duration, consecutive=False, verbose=verbose
        )
        # resample if requested
        # before all operations and possible trigger jitter, exctract the events
        if sfreq and np.round(sfreq) != np.round(raw.info["sfreq"]):
            raw, events = raw.resample(sfreq, n_jobs=1, verbose=verbose, events=events)

        if ica:
            assert isinstance(ica, int), "ica must be of type INT"
            n_components = ica
            ica_fif = os.path.basename(file).replace(
                ".fif", f"-{sfreq}hz-n{n_components}.ica"
            )
            ica_fif = settings.cache_dir + "/" + ica_fif
            # if we previously applied an ICA with these components,
            # we simply load this previous solution
            if os.path.isfile(ica_fif):
                ica = read_ica(ica_fif, verbose="ERROR")
                assert (
                    ica.n_components == n_components
                ), f"n components is not the same, please delete {ica_fif}"
                assert (
                    ica.method == "picard"
                ), f"ica method is not the same, please delete {ica_fif}"
            # else we compute it
            else:
                ####### START OF AUTOREJECT PART
                # by default, apply autoreject to find bad parts of the data
                # before fitting the ICA
                # determine bad segments, so that we can exclude them from
                # the ICA step, as is recommended by the autoreject guidelines
                logging.info("calculating outlier threshold for ICA")
                equidistants = mne.make_fixed_length_events(raw, duration=tstep)
                # HP filter data as recommended by the autoreject codebook
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=verbose)
                epochs = mne.Epochs(
                    raw_hp,
                    equidistants,
                    tmin=0.0,
                    tmax=tstep,
                    baseline=None,
                    verbose="WARNING",
                )
                reject = get_rejection_threshold(
                    epochs, verbose=verbose, cv=10, random_state=get_id(file)
                )
                epochs.drop_bad(reject=reject, verbose=False)
                log_append(
                    file,
                    "autoreject_raw",
                    {
                        "percentage_removed": epochs.drop_log_stats(),
                        "bad_segments": [x != () for x in epochs.drop_log],
                    },
                )
                ####### END OF AUTOREJECT PART

                ####### START OF ICA PART
                # use picard that simulates FastICA, this is specified by
                # setting fit_params to ortho and extended=True
                ica = ICA(
                    n_components=n_components,
                    method="picard",
                    verbose="WARNING",
                    fit_params=dict(ortho=True, extended=True),
                    random_state=get_id(file),
                )
                # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
                # we later apply the ICA components to the not-filtered signal
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="WARNING")
                ica.fit(raw_hp, picks="meg", reject=reject, tstep=tstep)
                ica.save(ica_fif)  # save ICA to file for later loading
                ####### END OF ICA PART

            assert sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"])
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_orig, ch_name="BIO001", verbose="WARNING"
            )
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, threshold=2, ch_name=["BIO002", "BIO003"], verbose="WARNING"
            )
            emg_indices, emg_scores = ica.find_bads_muscle(raw_orig, verbose="WARNING")

            if len(ecg_indices) == 0:
                warnings.warn("### no ECG component found, is 0")
            if len(eog_indices) == 0:
                warnings.warn("### no EOG component found, is 0")
            components = list(set(ecg_indices + eog_indices + emg_indices))
            ica_log = {
                "ecg_indices": ecg_indices,
                "eog_indices": eog_indices,
                "emg_indices": emg_indices,
            }
            log_append(file, "ica", ica_log)

            ica.exclude = components
            raw = ica.apply(raw, verbose="WARNING")
        return raw, events

    raw, events = _load_meg_ica(file, sfreq=sfreq, ica=ica, verbose=verbose)

    # lamba functions don't work well with caching
    # so allow definition of lambda using strings
    # filtering is done after ICA.
    if filter_func != "lambda x:x":
        print("filtering")
    if isinstance(filter_func, str):
        filter_func = eval(filter_func)
    raw = filter_func(raw)

    # warnings.warn('normalizing zscore per channel!')
    # raw = raw.apply_function(zscore, picks='meg', channel_wise=True)

    return raw, events

@memory.cache(ignore=["n_jobs"])
def load_epochs_bands(
    file,
    bands,
    sfreq=100,
    event_ids=None,
    tmin=-0.1,
    tmax=0.5,
    ica=None,
    autoreject=True,
    picks="meg",
    event_filter=None,
    n_jobs=1,
):

    assert isinstance(bands, dict), f"bands must be dict, but is {type(bands)}"

    if len(bands) > 1 and autoreject:
        raise ValueError("If several bands are used, cannot reject epochs")
    log_append(
        file,
        "parameters_bands",
        {
            "file": file,
            "sfreq": sfreq,
            "ica": ica,
            "event_ids": event_ids,
            "autoreject": autoreject,
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "bands": bands,
            "event_filter": event_filter,
        },
    )

    if n_jobs < 0:
        n_jobs = len(bands) + 1 - n_jobs
    data = Parallel(n_jobs=n_jobs)(
        delayed(load_epochs)(
            file,
            sfreq=sfreq,
            filter_func=f"lambda x: x.filter({lfreq}, {hfreq}, verbose=False, n_jobs=-1)",
            event_ids=event_ids,
            tmin=tmin,
            tmax=tmax,
            ica=ica,
            event_filter=event_filter,
            picks=picks,
            autoreject=autoreject,
        )
        for lfreq, hfreq in bands.values()
    )
    data_x = np.hstack([d[0] for d in data])
    data_y = data[0][1]
    return (data_x, data_y)


def load_epochs(
    file,
    sfreq=100,
    event_ids=None,
    event_filter=None,
    tmin=-0.1,
    tmax=0.5,
    ica=None,
    autoreject=True,
    filter_func="lambda x:x",
    picks="meg",
):
    """
    Load data from FIF file and return into epochs given by MEG triggers.
    stratifies the classes, that means each class will have the same
    number of examples.
    """
    if event_ids is None:
        event_ids = list(range(1, 11))
    raw, events = load_meg(file, sfreq=sfreq, ica=ica, filter_func=filter_func)

    events_mask = [True if idx in event_ids else False for idx in events[:, 2]]
    events = events[events_mask, :]

    if event_filter:
        if isinstance(event_filter, str):
            event_filter = eval(event_filter)
        events = event_filter(events)

    data_x, data_y = make_meg_epochs(
        raw, events=events, tmin=tmin, tmax=tmax, autoreject=autoreject, picks=picks
    )

    # start label count at 0 not at 1, so first class is 0
    data_y -= 1

    return data_x, data_y


def load_segments(file, sfreq=100, markers=[[10, 11]], picks='meg',
                  filter_func='lambda x:x', ica=None, verbose='ERROR'):
    """
    Load interval of data between two markers

    Parameters
    ----------
    file : str
        FIF file to load.
    sfreq : int, optional
        frequency to which to downsample. The default is 100.
    markers : list of list of 2 ints, optional
        which trigger values to take the segment between. The default is [[10, 11]].
        the first and last occurence of the marker is taken as the segment length
    slicer : TYPE, optional
        DESCRIPTION. The default is None.
    picks : str or int, optional
        string or int for which channels to load. The default is 'meg'.
    filter_func : str, optional
        lambda string for filtering the segments. The default is 'lambda x:x'.
    ica : int, optional
        how many ICA components to discard. The default is None.
    verbose : str, optional
        MNE verbose marker. The default is 'ERROR'.

    Returns
    -------
    segments : TYPE
        DESCRIPTION.

    """
    if len(markers)>1:
        raise NotImplementedError('Check end of function, this doesnt work yet')

    # now get segments from data
    raw, events = load_meg(file, sfreq=sfreq, filter_func=filter_func, ica=ica, verbose=verbose)
    print(f'available events: {np.unique(events[:,2])}, looking for {markers}')

    data = raw.get_data(picks=picks)
    triggers = raw.get_data(picks='STI101')

    segments = []
    first_samp = raw.first_samp
    all_markers = np.unique(markers) if isinstance(markers, list) else []
    found_event_idx = [True if e in all_markers else False for e in events[:,2]]
    events = events[found_event_idx, :]

    if len(events)==0:
        warnings.warn(f'No matching events for {all_markers} found in {file}, taking 90% middle segment of file')
        tstart = int(len(raw)*0.05)
        tstop = int(len(raw)*0.95)
        markers = [[0, 1]]
        events = np.array([[tstart, 0, markers[0][0]], [tstop,0, markers[0][1]]])

    start_id, stop_id = markers[0]
    segtuples = np.array(list(zip(events[::2], events[1::2])))

    segments = []
    trigger_val = []
    for start, stop in segtuples:
        assert start[2]==start_id
        assert stop[2]==stop_id
        t_start = start[0] - first_samp
        t_end = stop[0] - first_samp
        seg = data[:, t_start:t_end]
        tpos = np.where(triggers[0, t_start:t_end])[0]
        tval = triggers[0, t_start:t_end][tpos]
        trigger_val.append(list(zip(tpos, tval)))
        segments.append(seg)
    lengths = [seg.shape[-1] for seg in segments]
    if not any(np.diff(np.unique(lengths))>2):
        segments = [seg[:,:min(lengths)] for seg in segments]
    return segments


@memory.cache
def load_segments_bands(file, bands, sfreq=100, markers=[[10, 11]], picks='meg',
                        ica=settings.default_ica_components, verbose='ERROR',
                        n_jobs=1):

    log_append(file, 'parameters_bands', {'file': file, 'sfreq': sfreq,
                                         'ica': ica, 'markers': markers,
                                         'picks': picks, 'bands': bands})

    data_x = Parallel(n_jobs=n_jobs)(delayed(load_segments)
            (file, sfreq=sfreq, markers=markers, ica=ica, picks=picks,
             filter_func=f'lambda x: x.filter({lfreq}, {hfreq}, verbose=False, n_jobs=-1)')
            for lfreq, hfreq in bands.values())
    data_x = np.array(data_x).squeeze()
    data_x = default_normalize(data_x)
    return data_x


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:13 2024

@author: Simon Kern (@skjerns)
"""
import os
import logging
import mne
import numpy as np
import matplotlib.pyplot as plt
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject

def sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"]):
    """
    Checks that the first channel of `channels` is actually containing the
    most ECG events. Comparison is done by  mne.preprocessing.find_ecg_events,
    the channel with the lowest standard deviation between the intervals
    of heartbeats (the most regularly found QRSs) should be the ECG channel

    Parameters
    ----------
    raw : mne.Raw
        a MNE raw object.
    channels : list
        list of channel names that should be compared. First channel
        is the channel that should contain the most ECG events.
        The default is ['BIO001', 'BIO002', 'BIO003'].

    Returns
    -------
    bool
        True if ECG. Assertionerror if not.

    """
    stds = {}
    for ch in channels:
        x = mne.preprocessing.find_ecg_events(raw, ch_name=ch, verbose=False)
        t = x[0][:, 0]
        stds[ch] = np.std(np.diff(t))
    assert (
        np.argmin(stds.values()) == 0
    ), f"ERROR: {channels[0]} should be ECG, but did not have lowest STD: {stds}"
    return True

def repair_epochs_autoreject(raw, epochs, ar_file, picks="meg"):
    """runs autorejec with default parameters on chosen picks

    Parameters
    ----------
    raw : mne.Raw
        the raw object of which the epochs where extracted from.
    epochs : mne.Epochs
        epochs object, chunked mne raw file.
    ar_file : str
        file location where the autoreject results should be saved.
    picks : str/list, optional
        which channels to run autoreject on. The default is "meg".

    Returns
    -------
    epochs_repaired : mne.Epochs
        the epochs with all repaired and remove epochs.

    """
    # raise Exception('this function needs some cleanup')
    from utils import get_id

    # if precomputed solution exists, load it instead
    epochs_repaired_file = f"{ar_file[:-11]}.epochs"
    if os.path.exists(epochs_repaired_file):
        logging.info(f"Loading repaired epochs from {epochs_repaired_file}")
        epochs_repaired = mne.read_epochs(epochs_repaired_file, verbose="ERROR")
        return epochs_repaired

    # apply autoreject on this data to automatically repair
    # artefacted data points

    if os.path.exists(ar_file):
        logging.info(f"Loading autoreject pkl from {ar_file}")
        clf = read_auto_reject(ar_file)
    else:
        from utils import json_dump

        logging.info(f"Calculating autoreject pkl solution and saving to {ar_file}")
        json_dump({"events": epochs.events[:, 2].astype(np.int64)}, ar_file + ".json")
        clf = AutoReject(
            picks=picks, n_jobs=-1, verbose=False, random_state=get_id(ar_file)
        )
        clf.fit(epochs)
        clf.save(ar_file, overwrite=True)

    logging.info("repairing epochs")
    epochs_repaired, reject_log = clf.transform(epochs, return_log=True)

    ar_plot_dir = f"{settings.plot_dir}/autoreject/"
    os.makedirs(ar_plot_dir, exist_ok=True)

    event_ids = epochs.events[:, 2].astype(np.int64)
    arr_hash = hash_array(event_ids)

    n_bad = np.sum(reject_log.bad_epochs)
    arlog = {
        "mode": "repair & reject",
        "ar_file": ar_file,
        "bad_epochs": reject_log.bad_epochs,
        "n_bad": n_bad,
        "perc_bad": n_bad / len(epochs),
        "event_ids": event_ids,
    }

    subj = f"DSMR{get_id(ar_file)}"
    plt.maximize = False
    fig = plt.figure(figsize=[10, 10])
    ax = fig.subplots(1, 1)
    fig = reject_log.plot("horizontal", ax=ax, show=False)
    ax.set_title(f"{subj=} {n_bad=} event_ids={set(event_ids)}")
    fig.savefig(
        f"{ar_plot_dir}/{subj}_{os.path.basename(raw.filenames[0])}-{arr_hash}.png"
    )
    plt.close(fig)

    log_append(
        raw.filenames[0],
        f"autoreject_epochs event_ids={set(event_ids)} n_events={len(event_ids)}",
        arlog,
    )
    print(f"{n_bad}/{len(epochs)} bad epochs detected")
    epochs_repaired.save(epochs_repaired_file, verbose="ERROR")
    logging.info(f"saved repaired epochs to {epochs_repaired_file}")

    return epochs_repaired


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:26 2024

@author: Simon Kern (@skjerns)
"""
from collections import namedtuple
import mne
import hashlib
import numpy as np

def hash_array(arr, length=8, dtype=np.int64):
    """
    create a hash for any array by doing a full hash of the hexdigest

    Parameters
    ----------
    arr : np.ndarray
        any type of array.
    length : int, optional
        how many hash characters to return. The default is 8.
    dtype : np.dtype, optional
        which dtype to convert to, can speed up computation massively.
        The default is np.int64.

    Returns
    -------
    str
        sha1 hash of the hex array.

    """
    arr = arr.astype(dtype)
    return hashlib.sha1(arr.flatten("C")).hexdigest()[:length]


def get_ch_neighbours(ch_name, n=9, return_idx=False,
                      layout_name='Vectorview-all', plot=False):
    """retrieve the n neighbours of a given MEG channel location.
    Count includes the given origin electrode location"""
    layout = mne.channels.read_layout(layout_name)
    positions = {name.replace(' ', ''):pos[:3] for name, pos in zip(layout.names, layout.pos, strict=True)}

    Point = namedtuple('Point', 'name x y z')
    ch = Point(ch_name, *positions[ch_name])
    chs = [Point(ch, *pos) for ch, pos in positions.items()]
    chs = [ch for ch in chs if not (('EOG' in ch.name) or ('IO' in ch.name))]

    dist = lambda p: (p.x - ch.x)**2 + (p.y - ch.y)**2 + (p.z - ch.z)**2

    chs_sorted = sorted(chs, key=dist)

    chs_out = [ch.name for ch in chs_sorted[:n]]

    ch_as_in_raw = sorted([ch.replace(' ', '') for ch in layout.names])

    if plot:
        layout.plot(picks=[list(positions).index(ch) for ch in chs_out])
    return sorted([ch_as_in_raw.index(ch) for ch in chs_out]) if return_idx else chs_out
