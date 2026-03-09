#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 22:47:05 2025

@author: simon
"""
import os
import settings
import tdlm
import utils
from tqdm import tqdm
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mat73
from meg_utils import plotting, sigproc, decoding, misc
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import joblib
from joblib import Parallel, delayed
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2
from utils import get_performance
from utils import plot_correlation, zscore_multiaxis
import warnings
np.random.seed(0)  # just for safety

# plotting for paper default settings
plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)

utils.lowpriority()  # make sure not to clog CPU on multi-user systems
#%% Settings

# this is the classifier that will be used
C = 9.1  # determined previously via cross-validation, see below
clf = decoding.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=2, rng=0)

# list the files
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]

min_acc = 0.3  # minimum accuracy of decoders to include subjects
min_perf = 0.5  # minimum memory performance to include subjects
sfreq = 100  # downsample to this frequency. Changing is not supported.
ms_per_point = 10  # ms per sample point

bands = settings.bands_HP  # only use HP filter

baseline = None  # no baseline correction of localizer
uid = hex(random.getrandbits(128))[2:10]  # get random UID for saving log files
date = time.strftime("%Y-%m-%d")
times = np.arange(-100, 510, 10)
final_calculation = True  # this can be set to use the leftout data of the RS

proba_norm = 'lambda x: x/x.mean(0)'
# proba_norm = 'lambda x:x'

# static colors for plotting
palette = sns.color_palette()
c_fwd = palette[1]
c_bkw = palette[2]

#%% define TDLM parameters
n_shuf = 1000  # do 1000 permutations
max_lag = 30  # 500 ms time lag maximum
alpha_freq = 10 # assume 10 Hz alpha freq
# can be time or subject, either each permutation is zscored or all
# permutations at the subject level, see norm_func
zscore_axes = -1

# create forward transition matrix
tf = tdlm.seq2tf(settings.seq_12)

 # run the simulation twice, once scaling replace n_events from 50-100% once from 0-100% depending on performance
names_perf_scale = ['best-case', 'linear']

#%% simulation parameters

mode = 'erp_diff_all'  # take the ERP difference
lag_sim = 8  # simulate replay at 80 milliseconds
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]
tp = 31  # best decoding index precomputed, ~210ms after stim onset
best_C = 9.1  # cross-validated regularization strength

# refractory period to block before and after each event that we insert
# before, twice the lag must be blocked, as other events can else be put right
# beforehand and their second reactivation would reach into the next one
# After, only one lag must be blocked. See tdlm.utils.insert_events for details
refractory = [lag_sim*2, lag_sim]


# turn off saving resting state simulation data, as the resulting array
# is >64GB large. Mainly used for debugging purposes
save_simulation_data_in_memory = False
n_jobs = -1  # set this to a lower number (eg 1) if you run out of memory

if not final_calculation:
    print('not running final calculation yet, only use this for final paper calculation')

# best_tp will be calculated and replaced later, for debugging it is sometimes
# easier to define it here, so you can run segments without computing everything
best_tp = utils.load_pkl(f"{settings.cache_dir}/best_tp.pkl.zip", {31:31})

palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=101)
hues = {subj:palette[int(100*(get_performance(subj)-0.5)*2)] for subj in subjects}

# gaussian window created by gaussian_filter1d(np.float_([0,0,1,0,0]), 1)
gaussian_weighting = [0.05842299, 0.24210528, 1, 0.24210528, 0.05842299]
#%% preload some data (e.g. localizer)
localizer = {}  # localizer training data
rs1 = {}
rs2 = {}
neg_x = {}  # pre-audio fixation cross neg_x of localizer
seqs = {}  # load sequences of that participant in this dict

for subj in tqdm(subjects, desc="Loading data"):
    # data used for the localizer
    localizer[subj] = load_localizers_seq12(subj=subj, sfreq=sfreq, bands=bands, autoreject=settings.default_autoreject, ica=settings.default_ica_components)
    # negative examples from the fixation cross before audio cue onset
    neg_x[subj] = load_neg_x_before_audio_onset(subj=subj, sfreq=sfreq, bands=bands,  autoreject=settings.default_autoreject, ica=settings.default_ica_components)

    # resting state data, both eyes open and eyes closed together
    rs1[subj] = load_RS1(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)
    rs2[subj] = load_RS2(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)

    # individual sequences for the trials (maybe not necessary)
    seqs[subj] = utils.get_sequences(subj)


max_accuracy = [utils.get_decoding_accuracy(subj, clf=clf, n_splits=20)[0] for subj in subjects]
test_performance = [utils.get_performance(subj, which='test') for subj in subjects]
df_subjects = pd.DataFrame({'subject': subjects,
                            'accuracy': max_accuracy,
                            'performance': test_performance}).sort_values('accuracy')

# define which participants are rejected due to preregistered criteria
excluded_acc = [subj for subj in subjects if utils.get_decoding_accuracy(subj,
                clf=clf, n_splits=20)[0]<0.3]
excluded_perf = [subj for subj in subjects if utils.get_performance(subj, which='test')<0.5]
excluded_miss = [subj for subj in subjects if utils.get_responses_localizer(subj)['n_misses']>60*0.25]

df_excluded = pd.DataFrame({'reason': ['low decoding']*len(excluded_acc) + \
                                      ['low performance']*len(excluded_perf) + \
                                      ['>25% misses']*len(excluded_miss)},
                           index=excluded_acc + excluded_perf + excluded_miss)
df_excluded.sort_index(inplace=True)

subjects_incl = sorted(set(subjects).difference(set(df_excluded.index)))


#%% SUPPL: RS1 vs RS2 no alpha

tp = np.mean(list(best_tp.values())).astype(int)

# put forward and backward sequenceness per subject for RS1/RS2 in these arrays
rs1_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs1_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

# # create axes to plot subject sequenceness into
fig1, axs1 = utils.make_fig(n_axs=len(subjects_incl), bottom_plots=0, suptitle=f'RS control')
fig2, axs2 = utils.make_fig(n_axs=len(subjects_incl), bottom_plots=0, suptitle=f'RS post-learning')

for i, subj in enumerate(tqdm(subjects_incl, desc="subject")):
    # train our classifier using localizer data and negative data
    train_x, train_y = localizer[subj]
    clf.reset_random_state(misc.make_seed(i, "rs"))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # get probability estimates for item reactivation from the resting state
    proba_rs1 = clf.predict_proba(rs1[subj])
    proba_rs2 = clf.predict_proba(rs2[subj])

    # normalize probabilities
    proba_rs1 = eval(proba_norm)(proba_rs1)
    proba_rs2 = eval(proba_norm)(proba_rs2)

    # calculate sequenceness
    seed = utils.get_id(subj)*42

    rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(proba_rs1, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=None,
                                                  seed=misc.make_seed(subj, 'rs1'))
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=None,
                                                  seed=misc.make_seed(subj, 'rs2'))

    # zscore sequenceness scores
    rs1_sf[i, :] = zscore_multiaxis(rs1_sf_subj, axes=zscore_axes)
    rs1_sb[i, :] = zscore_multiaxis(rs1_sb_subj, axes=zscore_axes)
    rs2_sf[i, :] = zscore_multiaxis(rs2_sf_subj, axes=zscore_axes)
    rs2_sb[i, :] = zscore_multiaxis(rs2_sb_subj, axes=zscore_axes)

    # # subject level plot
    tdlm.plot_sequenceness(rs1_sf[i, :], rs1_sb[i, :] , ax=axs1[i], which=['fwd', 'bkw'], rescale=False)
    tdlm.plot_sequenceness(rs2_sf[i, :], rs2_sb[i, :] , ax=axs2[i], which=['fwd', 'bkw'], rescale=False)
    axs1[i].text(0, 0, subj, alpha=0.5)
    axs2[i].text(0, 0, subj, alpha=0.5)
    plt.pause(0.01)

utils.normalize_lims(axs1[:len(subjects_incl)])
utils.normalize_lims(axs2[:len(subjects_incl)])

# create sequenceness dictionary
df_sequenceness = pd.DataFrame()
pkl_sequencenes = f'{settings.cache_dir}/rs1rs2-sequenceness_noalpha.pkl.zip'
for s, subj in enumerate(subjects_incl):
    for c, condition in enumerate(['rs1', 'rs2']):
        sf = [rs1_sf, rs2_sf][c]
        sb = [rs1_sb, rs2_sb][c]

        for d, direction in enumerate(['fwd', 'bkw']):
            sx = np.abs([sf, sb][d])
            sx_subj = sx[s, 0]
            peak_time = np.nanargmax(np.nanmean(sx[:, 0, :], 0))
            peak_sequenceness = sx_subj[peak_time]
            mean_sequenencess = np.nanmean(sx_subj)
            df_tmp = pd.DataFrame({'subject': subj,
                                    'peak': peak_sequenceness,
                                    'mean': mean_sequenencess,
                                    'peak_time': peak_time,
                                    'condition': condition,
                                    'direction': direction},
                                    index=[subj])
            df_sequenceness = pd.concat([df_sequenceness, df_tmp],
                                        ignore_index=True)
joblib.dump(df_sequenceness, pkl_sequencenes)

# plotting of sequenceness curves
fig, axs = plt.subplot_mosaic('11AB\n22CD', figsize=[16, 8])

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

sequence_names = ['Control', 'Post-Learn']

c_fwd = [sns.color_palette("bright")[1], sns.color_palette('dark')[1]]
c_bkw = [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]

# plot forward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='fwd', title=f'Forward Sequenceness ',
                       label='control',
                       ax=axs['1'], rescale=False, color=c_fwd[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='fwd', title=f'Forward Sequenceness',
                       label='post-learn',
                       ax=axs['1'], rescale=False, color=c_fwd[1], clear=False)

# plot backward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='bkw', title=f'Backward Sequenceness',
                       label='control',
                       ax=axs['2'], rescale=False, color=c_bkw[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='bkw', title=f'Backward Sequenceness',
                       label='post-learn',

                       ax=axs['2'], rescale=False, color=c_bkw[1], clear=False)

for ax in (axs_seq:=[axs['1'], axs['2']]):
    ax.set_ylabel('sequenceness\n(u.u., zscored)')
    ax.set_xlabel('time lag (milliseconds)')
    hdl, lbl = ax.get_legend_handles_labels()
    ax.legend([hdl[i] for i in [0, 3, 4, 5]],
              [lbl[i] for i in [0, 3, 4, 5]])
    # ax.legend(['Control', '_', '_', '_', '_', 'Post-Learn'],  loc='lower right')
utils.normalize_lims(axs_seq)

for i, direction in enumerate(['forward', 'backward']):
    ax = [axs['1'], axs['2']][i]
    c = [c_fwd, c_bkw][i][0]
    rs_pre = [rs1_sf, rs1_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_pre[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_pre[:, 0, peak_lag]))
    ax.vlines(peak_lag*10, *ax.get_ylim(), color=c, alpha=0.5)
    ax = [axs['A'], axs['C']][i]
    r1, pval1 = plot_correlation(peak_vals, values=perf_test,
                             color=c, ax=ax,title=f"Control")

    ax.text(ax.get_xlim()[1], .5, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')

    ax = [axs['1'], axs['2']][i]
    c = [c_fwd, c_bkw][i][1]
    rs_post = [rs2_sf, rs2_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_post[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_post[:, 0, peak_lag]))
    ax.vlines(peak_lag*10, *ax.get_ylim(), color=c, alpha=0.5)
    ax = [axs['B'], axs['D']][i]
    r1, pval1 = plot_correlation(peak_vals, values=perf_test,
                             color=c, ax=ax,title=f"Post-Learn")

    ax.text(ax.get_xlim()[1], .5, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')

utils.normalize_lims([ax for desc, ax in axs.items() if not desc.isnumeric()])

fig.suptitle('Sequenceness of Resting States without Control for Alpha Oscillations')
plt.pause(0.1)
fig.tight_layout()
utils.savefig(fig, f'supplement/sequenceness_rs_noalpha.png')
utils.savefig(fig, f'supplement/sequenceness_rs_noalpha.svg')
utils.savefig(fig, f'supplement/sequenceness_rs_noalpha.eps')


#%% SUPPL: TDLM MATLAB simulation probabilities

res = mat73.loadmat('./MATLAB/probabilities.mat')

df_proba = pd.DataFrame()
for subj in range(len(res['probaNonTarget_post'])):
    pre_target = np.squeeze(res['probaTarget_pre'][subj])
    pre_nontarget = np.squeeze(res['probaNonTarget_pre'][subj])
    post_target = np.squeeze(res['probaTarget_post'][subj])
    post_nontarget = np.squeeze(res['probaNonTarget_post'][subj])
    df_proba = pd.concat([df_proba,
                             pd.DataFrame({'proba' : np.squeeze(pre_target),
                                           'class': 'target',
                                           'condition': 'baseline',
                                           'subject':subj}),
                             pd.DataFrame({'proba' : np.squeeze(pre_nontarget),
                                           'class': 'other',
                                           'condition': 'baseline',
                                           'subject':subj}),
                             pd.DataFrame({'proba' : np.squeeze(post_target),
                                           'class': 'target',
                                           'condition': 'with pattern',
                                           'subject':subj}),
                             pd.DataFrame({'proba' : np.squeeze(post_nontarget),
                                           'class': 'other',
                                           'condition': 'with pattern',
                                           'subject':subj}),],
                             ignore_index=True)

fig = plt.figure(figsize=[6, 6])
ax = fig.subplots(1,1)
df_proba = df_proba.groupby(['subject', 'condition', 'class']).mean(0).reset_index()
df_proba = df_proba.sort_values(['condition'], ascending=True).sort_values('class')

sns.boxplot(data=df_proba, x='condition', y='proba', hue='class', ax=ax)
ax.set_ylabel('classifier probability estimate')
ax.set_title(f'MATLAB Simulation \nDecoded probability of item\nat baseline and with added pattern')
utils.savefig(fig, f'supplement/MATLAB-probabilities-boxplot.png')
utils.savefig(fig, f'supplement/MATLAB-probabilities-boxplot.svg')
utils.savefig(fig, f'supplement/MATLAB-probabilities-boxplot.eps')

fig = plt.figure(figsize=[6,6])
# hues = {subj:palette[int(100*(get_performance(subj)-0.5)*2)] for subj in subjects}
df_proba_target = df_proba[df_proba['class']=='target']
df_proba_target = df_proba_target.groupby(['condition', 'subject']).mean(True).reset_index()
df_proba_target = df_proba_target.sort_values('condition', ascending=True)
sns.scatterplot(data=df_proba_target, x='condition', y='proba', hue='subject',
                legend=False, s=(150))
sns.lineplot(data=df_proba_target, x='condition', y='proba', hue='subject',
                legend=False, alpha=0.3)
plt.ylabel('classifier probability estimate')
plt.title(f'MATLAB Simulation\nClassifier Probabilities in Resting State')
utils.savefig(fig, 'supplement/MATLAB-probabilities-lineplot.png')
utils.savefig(fig, 'supplement/MATLAB-probabilities-lineplot.svg')
utils.savefig(fig, 'supplement/MATLAB-probabilities-lineplot.eps')

#%% SUPPL: Raw probabilities per class
from meg_utils.decoding import cross_validation_across_time
ex_per_fold = 8

df_proba = pd.DataFrame()
for subj in tqdm(subjects, desc='calculating probas'):

    data_x, data_y = localizer[subj]
    df_subj, probas = cross_validation_across_time(data_x, data_y, subj=subj,
                                                   n_jobs=-1, tmin=-0.1, tmax=0.5,
                                                   ex_per_fold=ex_per_fold, clf=clf,
                                                   return_probas=True, verbose=False)
    _, n_times, n_classes = probas.shape
    df_proba_subj = pd.DataFrame()
    timepoint = np.repeat(df_subj.timepoint.unique(), n_classes)
    stimuli_de = utils.get_image_names(subj)
    stimuli_en = [settings.stim_translation[name] for name in stimuli_de]
    stim_names = [f'{de}/{en}' for de, en in zip(stimuli_de, stimuli_en)]

    for i, (proba, y) in enumerate(zip(probas, data_y, strict=True)):
        label = ['other']*proba.shape[-1]
        label[y] = 'target'
        df_tmp = pd.DataFrame({'label': np.hstack([label]* n_times),
                               'timepoint': timepoint,
                               'proba': proba.ravel(),
                               'stimulus': stim_names[y]})
        df_proba_subj = pd.concat([df_proba_subj, df_tmp], ignore_index=True)

    df_proba_subj = df_proba_subj.groupby(['label', 'timepoint', 'stimulus']).mean().reset_index()
    df_proba_subj['subject'] = subj
    df_proba = pd.concat([df_proba, df_proba_subj], ignore_index=True)


fix, axs = plt.subplots(4, 3, figsize=[14, 8])
axs = axs.flatten()
ax_b = axs[-1]

for i, stimulus in enumerate(sorted(stim_names)):
    ax = axs[i]
    sns.lineplot(df_proba[df_proba.stimulus==stimulus], x='timepoint', style='subject',
                 y='proba', hue='label', ax=ax, alpha=0.1, legend=False)
    sns.lineplot(df_proba[df_proba.stimulus==stimulus], x='timepoint',
                 y='proba', hue='label', ax=ax)
    ax.set_title(stimulus)
    plt.pause(0.1)

df_proba_mean = df_proba.groupby(['timepoint', 'label', 'subject']).mean(True).reset_index()

sns.lineplot(df_proba_mean, x='timepoint', y='proba', hue='label', style='subject',
             ax=ax_b, alpha=0.1, legend=False)
sns.lineplot(df_proba_mean, x='timepoint', y='proba', hue='label', ax=ax_b)
ax_b.set_title('All stimuli')
plotting.normalize_lims(list(axs))
fig.tight_layout()
plt.pause(0.1)
fig.savefig(settings.plot_dir + f'localizer_perclass.png')
fig.savefig(settings.plot_dir + f'localizer_perclass.svg')
fig.savefig(settings.plot_dir + f'localizer_perclass.eps')




#%% EXTRA RS1 vs RS2 with alphafilter

# tp = np.mean(list(best_tp.values())).astype(int)

clf.set_params(C=best_C)
# put forward and backward sequenceness per subject for RS1/RS2 in these arrays
rs1_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs1_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

rs1_sf_filt = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs1_sb_filt = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sf_filt = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sb_filt = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

# spectrum of probas
psd_rs1, psd_rs2           = [], []      # raw
psd_rs1_filt, psd_rs2_filt = [], []     # notch-filtered

# corr coeffs
corrcoefs_rs1 = [] # store correlation coefficients
corrcoefs_rs2 = [] # store correlation coefficients

fig, axs = plt.subplots(1, 3, figsize=[12, 6])
fig_spec, ax_spec = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for i, subj in enumerate(tqdm(subjects_incl, desc="subject")):
    # train our classifier using localizer data and negative data
    train_x, train_y = localizer[subj]
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # get probability estimates for item reactivation from the resting state
    proba_rs1 = clf.predict_proba(rs1[subj])
    proba_rs2 = clf.predict_proba(rs2[subj])

    # normalize probabilities
    proba_rs1 = eval(proba_norm)(proba_rs1)
    proba_rs2 = eval(proba_norm)(proba_rs2)


    # sort the stimulus order to be comparable across subjects
    names = utils.get_image_names(subj)
    img_idx = [names.index(img) for img in settings.stim_translation]

    # create correlation across stimuli
    corr_rs1 = np.corrcoef(proba_rs1.T[img_idx])
    corr_rs2 = np.corrcoef(proba_rs2.T[img_idx])

    corrcoefs_rs1 += [corr_rs1]
    corrcoefs_rs2 += [corr_rs2]


    for ax in axs.flat: ax.clear()
    axs[0].imshow(corr_rs1)
    axs[0].set_title('RS pre')
    axs[1].imshow(corr_rs2)
    axs[1].set_title('RS post')
    axs[2].imshow(corr_rs2 - corr_rs1)
    axs[2].set_title('diff')
    for ax in axs.flat:
        ax.set_xticks(np.arange(10), settings.stim_translation, rotation=90)
        ax.set_yticks(np.arange(10), settings.stim_translation)
    fig.suptitle(f'Normalized probas correlation coefficients for {subj=}')
    utils.savefig(fig,  f'/corrcoeff/corrcoef_{subj}.png')
    utils.savefig(fig,  f'/corrcoeff/corrcoef_{subj}.svg')
    utils.savefig(fig,  f'/corrcoeff/corrcoef_{subj}.eps')

    # calculate sequenceness
    seed = utils.get_id(subj)*42

    rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(proba_rs1, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=seed)
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=seed+1)

    alpha_peak1 = sigproc.get_alpha_peak(rs1[subj].T, sfreq=100)
    alpha_peak2 = sigproc.get_alpha_peak(rs2[subj].T, sfreq=100)

    # zscore sequenceness scores
    rs1_sf[i, :] = zscore_multiaxis(rs1_sf_subj, axes=zscore_axes)
    rs1_sb[i, :] = zscore_multiaxis(rs1_sb_subj, axes=zscore_axes)
    rs2_sf[i, :] = zscore_multiaxis(rs2_sf_subj, axes=zscore_axes)
    rs2_sb[i, :] = zscore_multiaxis(rs2_sb_subj, axes=zscore_axes)


    # next same for filtered data
    # Apply notch filter at 10 Hz
    rs1_filt = sigproc.notch(rs1[subj].T, freqs=[alpha_peak1], notch_widths=3, sfreq=sfreq, n_jobs=1).T
    rs2_filt = sigproc.notch(rs2[subj].T, freqs=[alpha_peak2], notch_widths=3, sfreq=sfreq, n_jobs=1).T

    # Get probability estimates for filtered data
    proba_rs1_filt = clf.predict_proba(rs1_filt)
    proba_rs2_filt = clf.predict_proba(rs2_filt)

    # Normalize
    proba_rs1_filt = eval(proba_norm)(proba_rs1_filt)
    proba_rs2_filt = eval(proba_norm)(proba_rs2_filt)

    # Compute sequenceness on filtered data
    rs1_sf_filt_subj, rs1_sb_filt_subj = tdlm.compute_1step(proba_rs1_filt, tf=tf, n_shuf=n_shuf,
                                                            max_lag=max_lag, alpha_freq=alpha_freq,
                                                            seed=seed+2)
    rs2_sf_filt_subj, rs2_sb_filt_subj = tdlm.compute_1step(proba_rs2_filt, tf=tf, n_shuf=n_shuf,
                                                            max_lag=max_lag, alpha_freq=alpha_freq,
                                                            seed=seed+3)

    # Z-score and store filtered results
    rs1_sf_filt[i, :] = zscore_multiaxis(rs1_sf_filt_subj, axes=zscore_axes)
    rs1_sb_filt[i, :] = zscore_multiaxis(rs1_sb_filt_subj, axes=zscore_axes)
    rs2_sf_filt[i, :] = zscore_multiaxis(rs2_sf_filt_subj, axes=zscore_axes)
    rs2_sb_filt[i, :] = zscore_multiaxis(rs2_sb_filt_subj, axes=zscore_axes)

    # calculate spectrograms of probabilities
    fs     = sfreq                         # sampling rate already in your config
    nper   = 2 ** 10                      # 4096-point segments; tweak if mem-heavy
    freqs, Pxx_rs1       = signal.welch(proba_rs1,       fs=fs, nperseg=nper, axis=0)
    _,     Pxx_rs2       = signal.welch(proba_rs2,       fs=fs, nperseg=nper, axis=0)
    _,     Pxx_rs1f      = signal.welch(proba_rs1_filt,  fs=fs, nperseg=nper, axis=0)
    _,     Pxx_rs2f      = signal.welch(proba_rs2_filt,  fs=fs, nperseg=nper, axis=0)

    Pxx_rs1 = np.log10(Pxx_rs1).mean(1)
    Pxx_rs2 = np.log10(Pxx_rs2).mean(1)
    Pxx_rs1f = np.log10(Pxx_rs1f).mean(1)
    Pxx_rs2f = np.log10(Pxx_rs2f).mean(1)
    Pxx_rs1  = gaussian_filter1d(Pxx_rs1,  sigma=5)
    Pxx_rs2  = gaussian_filter1d(Pxx_rs2,  sigma=5)
    Pxx_rs1f = gaussian_filter1d(Pxx_rs1f, sigma=5)
    Pxx_rs2f = gaussian_filter1d(Pxx_rs2f, sigma=5)

    mask = freqs <= 35                    # keep 0-35 Hz only
    psd_rs1.append(Pxx_rs1[mask])
    psd_rs2.append(Pxx_rs2[mask])
    psd_rs1_filt.append(Pxx_rs1f[mask])
    psd_rs2_filt.append(Pxx_rs2f[mask])

    # OPTIONAL PER-SUBJECT FIG
    for ax in ax_spec: ax.clear()
    ax_spec[0].plot(freqs[mask], Pxx_rs1[mask],  label='raw')
    ax_spec[0].plot(freqs[mask], Pxx_rs1f[mask], label='notch 10 Hz')
    ax_spec[0].set(title='RS1', xlim=(0, 25), xlabel='Hz', ylabel='PSD')
    ax_spec[1].plot(freqs[mask], Pxx_rs2[mask],  label='raw')
    ax_spec[1].plot(freqs[mask], Pxx_rs2f[mask], label='notch 10 Hz')
    ax_spec[1].set(title='RS2', xlim=(0, 25), xlabel='Hz')
    ax_spec[1].legend(frameon=False)
    # for ax in ax_spec: ax.set_ylim(0, 0.5)
    utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.png')
    utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.svg')
    utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.eps')

    utils.plot_rest_state_spectrograms(rs1[subj],
                                       rs1_filt,
                                       rs2[subj],
                                       rs2_filt,
                                       sfreq=100,
                                       chan_idx = 234,
                                       subj=subj)


for ax in axs.flat: ax.clear()
axs[0].imshow(corr_rs1)
axs[0].set_title('RS pre')
axs[1].imshow(corr_rs2)
axs[1].set_title('RS post')
axs[2].imshow(corr_rs2 - corr_rs1)
axs[2].set_title('diff')
for ax in axs.flat:
    ax.set_xticks(np.arange(10), settings.stim_translation, rotation=90)
    ax.set_yticks(np.arange(10), settings.stim_translation)
fig.suptitle(f'Normalized proba correlation coefficients for n={len(subjects_incl)}')
utils.savefig(fig,  f'/corrcoeff/corrcoef_all.png')
utils.savefig(fig,  f'/corrcoeff/corrcoef_all.svg')
utils.savefig(fig,  f'/corrcoeff/corrcoef_all.eps')

for ax in ax_spec.flat: ax.clear()
ax_spec[0].plot(freqs[mask], Pxx_rs1[mask],  label='raw')
ax_spec[0].plot(freqs[mask], Pxx_rs1f[mask], label='notch 10 Hz')
ax_spec[0].set(title='RS1', xlim=(0, 35), xlabel='Hz', ylabel='PSD')
ax_spec[1].plot(freqs[mask], Pxx_rs2[mask],  label='raw')
ax_spec[1].plot(freqs[mask], Pxx_rs2f[mask], label='notch 10 Hz')
ax_spec[1].set(title='RS2', xlim=(0, 35), xlabel='Hz')
ax_spec[1].legend(frameon=False)
fig_spec.suptitle('Spectrum of probabilities')
utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.png')
utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.svg')
utils.savefig(fig_spec, f'/spectrum/proba_spectrum_{subj}.eps')

sns.set_context('talk')
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# convenience for iterating
c_fwd = [sns.color_palette("bright")[1], sns.color_palette('dark')[1]]
c_bkw = [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]

cfg = [
    # row, col, label, forward data, backward data
    (0, 0, 'RS1 FWD', rs1_sf,       rs1_sb,       'C0', True),
    (0, 0, None,      rs1_sf_filt,  rs1_sb_filt,  'C1', False),
    (0, 1, 'RS1 BKW', rs1_sf,       rs1_sb,       'C0', True,  ['bkw']),
    (0, 1, None,      rs1_sf_filt,  rs1_sb_filt,  'C1', False, ['bkw']),
    (1, 0, 'RS2 FWD', rs2_sf,       rs2_sb,       'C0', True),
    (1, 0, None,      rs2_sf_filt,  rs2_sb_filt,  'C1', False),
    (1, 1, 'RS2 BKW', rs2_sf,       rs2_sb,       'C0', True,  ['bkw']),
    (1, 1, None,      rs2_sf_filt,  rs2_sb_filt,  'C1', False, ['bkw']),
]

import matplotlib as mpl
for row, col, title, fwd, bkw, color, clear, *which in cfg:
    which = which[0] if which else ['fwd']
    ax = axes[row, col]
    tdlm.plot_sequenceness(
        fwd, bkw, sfreq=sfreq, ax=ax, title=title,
        color=color, which=which, clear=clear, despine=False,
        legend=False
    )

# global legend
handles = [
    mpl.lines.Line2D([], [], color='C0', label='raw'),
    mpl.lines.Line2D([], [], color='C1', label='notch 10 Hz')
]
fig.legend(handles=handles, loc='upper right', frameon=False)

sns.despine()
fig.tight_layout(rect=[0, 0, 0.95, 1])

#%% EXTRA: C ~ zero-lag correlation

Cs = np.logspace(-1, 2, 50)                 # 1e‑3 … 1e3
nC, nS = len(Cs), len(subjects_incl)
mean_off  = np.empty((nS, nC))              # per‑subject metrics
beta_corr = np.empty_like(mean_off)

mean_off.fill(np.nan); beta_corr.fill(np.nan)
clf.set_params(max_iter=100)

tqdm_loop = tqdm(total=len(subjects)*len(Cs))

df_seq = pd.DataFrame()
df_reg = pd.DataFrame()
from scipy.stats import zscore

for si, subj in enumerate(subjects_incl):
    train_x, train_y = localizer[subj]
    rs_data      = rs1[subj]                # use RS1; swap if needed

    # subject‑specific stimulus order
    img_idx = [utils.get_image_names(subj).index(img) for img in settings.stim_translation]

    for ci, C in enumerate(Cs):
        clf.set_params(C=C).fit(train_x[:, :, tp], train_y)
        proba = eval(proba_norm)(clf.predict_proba(rs_data))
        colinear = decoding.get_mean_corrcoef(proba.T)
        spatcorr = decoding.get_mean_corrcoef(clf.coef_)

        df_reg = pd.concat([df_reg, pd.DataFrame({'C': C,
                                                  'subj':subj,
                                                  'colinear': colinear,
                                                  'spatcorr': spatcorr
                                                  }, index=[0])], ignore_index=True)

        sf, sb = tdlm.compute_1step(proba, tf, n_shuf=0, alpha_freq=10)
        sf = zscore(sf.squeeze(), nan_policy='omit')
        sb = zscore(sb.squeeze(), nan_policy='omit')

        df_tmp = pd.DataFrame({'C': C,
                               'direction': ['fwd', 'bkw'],
                               'max': [np.nanmax(sf), np.nanmax(sb)],
                               'min': [np.nanmin(sf), np.nanmin(sb)],
                               'abs': [np.nanmax(abs(sf)), np.nanmax(abs(sb))],
                               'subj': subj
                               })
        df_seq = pd.concat([df_seq, df_tmp], ignore_index=True)
        tqdm_loop.update()


joblib.dump([df_reg, df_seq], settings.cache_dir + '/df_c_spatcorr.pkl.gz')

df_reg, df_seq = joblib.load(settings.cache_dir + '/df_c_spatcorr.pkl.gz')

df_reg_long = df_reg.melt(id_vars=['subj','C'], value_vars=['colinear','spatcorr'],
                          var_name='metric', value_name='value')

plt.figure(figsize=(5,3))
ax = sns.lineplot(data=df_reg_long, x='C', y='value', hue='metric', estimator='mean', errorbar='se')
ax.set(xscale='log', xlabel='L1 regularization (C)', ylabel='correlation')
ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=2)
sns.despine()
plt.tight_layout()
utils.savefig(plt.gcf(), 'reg_corr_seaborn.png')
utils.savefig(plt.gcf(), 'reg_corr_seaborn.svg')
utils.savefig(plt.gcf(), 'reg_corr_seaborn.eps')

# 2. sequenceness vs metrics on separate axes
df_seqx = df_seq.merge(df_reg, on=['C','subj'])
df_seq_long = df_seqx.melt(id_vars=['C','subj','direction','colinear','spatcorr'],
                          value_vars=['max','min','abs'],
                          var_name='seq_metric', value_name='seq_value')

preds = {'colinear':'Probability corrcoef',
         'spatcorr':'Sensor spatial corr',
         'C':'L1 reg (C)'}

for pred, label in preds.items():
    fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=False)
    for i, seq_metric in enumerate(['max','min','abs']):
        ax = axes[i]
        sns.regplot(data=df_seqx, x=pred, y=seq_metric, ax=ax, scatter_kws={'s':20,'alpha':0.5})
        ax.set(title=f'{seq_metric} vs {label}', xlabel=label, ylabel='sequenceness', xscale='log')
    fig.tight_layout()
    utils.savefig(fig, f'seq_vs_{pred}.png')
    utils.savefig(fig, f'seq_vs_{pred}.svg')
    utils.savefig(fig, f'seq_vs_{pred}.eps')

#%% EXTRA: heatmap of sequenceness


for C in [0.01, 0.1, 1, 5, 9.1, 20]:
    clf.set_params(C=C)
    # ----- config --------------------------------------------------------------
    timepoints = np.arange(61)  # training tps (samples)

    # containers  (shape: n_time × n_lag)
    rs1_seqx = []
    rs2_seqx = []


    def train_predict_index(train_x, train_y, test_x, clf, proba, t):
        """helper func for making the time a parameter, much faster for joblib"""
        return decoding.train_predict(train_x[:, :, t], train_y, test_x, clf=clf,proba=proba)

    # ----- sweep training time‑points ------------------------------------------
    pool = Parallel(n_jobs)

    for subj in tqdm(subjects_incl):

        train_x, train_y = localizer[subj]

        rs1_probas = pool(delayed(train_predict_index)
                                  (train_x, train_y, rs1[subj],
                                   clf=clf, proba=True, t=t) for t in timepoints)

        rs2_probas = pool(delayed(train_predict_index)
                                  (train_x, train_y, rs2[subj],
                                   clf=clf, proba=True, t=t) for t in timepoints)

        rs1_res = pool(delayed(tdlm.compute_1step)(probas, tf, n_shuf=100,
                                                   max_lag=max_lag, alpha_freq=10) for probas in rs1_probas)

        rs2_res = pool(delayed(tdlm.compute_1step)(probas, tf, n_shuf=100,
                                                   max_lag=max_lag, alpha_freq=10) for probas in rs2_probas)

        rs1_seqx += [rs1_res]
        rs2_seqx += [rs2_res]


    # ----- plotting -------------------------------------------------------------
    rs1_seq = np.squeeze(rs1_seqx)
    rs2_seq = np.squeeze(rs2_seqx)

    rs1_all =  np.concatenate([np.nanmean(rs1_seq, axis=0),
                               np.nanmean(rs1_seq[:,:,:1,:]-rs1_seq[:,:,1:,: ], axis=0)], axis=1)
    rs2_all =  np.concatenate([np.nanmean(rs2_seq, axis=0),
                               np.nanmean(rs2_seq[:,:,:1,:]-rs2_seq[:,:,1:,: ], axis=0)], axis=1)


    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

    rs_names = ['RS pre', 'RS post']
    dirs = ['fwd', 'bkw', 'diff']

    for i, rs in enumerate([rs1_all, rs2_all]):
        for j, sx in enumerate(rs.swapaxes(0,1)):
            ax = axs[i, j]
            df = pd.DataFrame(sx[:, 0, :], columns=np.arange(max_lag+1)*10,
                              index=(np.arange(61)-10)*10)
            sns.heatmap(df, ax=ax, cbar=False, cmap='vlag')
            ax.set(xlabel='time lag', ylabel='train timepoint', title=f'{rs_names[i]} - {dirs[j]}')
    plotting.normalize_lims(axs, 'v')
    fig.suptitle(f'Sequenceness as function of train timepoint (no zscore) {C=}')
    utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_no_zscore_C{C}.png')
    utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_no_zscore_C{C}.svg')
    utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_no_zscore_C{C}.eps')


    ###### with ZSCORE
    for axis in [0, -1]:
        name = ['population', 'time'][axis]
        from scipy.stats import zscore
        rs1_seq = np.squeeze(rs1_seqx)
        rs2_seq = np.squeeze(rs2_seqx)

        rs1_seq[:,:,0, :, :] = zscore(rs1_seq[:,:,0, :, :], nan_policy='omit', axis=axis)
        rs1_seq[:,:,1, :, :] = zscore(rs1_seq[:,:,1, :, :], nan_policy='omit', axis=axis)
        rs2_seq[:,:,0, :, :] = zscore(rs2_seq[:,:,0, :, :], nan_policy='omit', axis=axis)
        rs2_seq[:,:,1, :, :] = zscore(rs2_seq[:,:,1, :, :], nan_policy='omit', axis=axis)

        rs1_all =  np.concatenate([np.nanmean(rs1_seq, axis=0),
                                   np.nanmean(rs1_seq[:,:,:1,:]-rs1_seq[:,:,1:,: ], axis=0)], axis=1)
        rs2_all =  np.concatenate([np.nanmean(rs2_seq, axis=0),
                                   np.nanmean(rs2_seq[:,:,:1,:]-rs2_seq[:,:,1:,: ], axis=0)], axis=1)

        fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

        rs_names = ['RS pre', 'RS post']
        dirs = ['fwd', 'bkw', 'diff']

        for i, rs in enumerate([rs1_all, rs2_all]):
            for j, sx in enumerate(rs.swapaxes(0,1)):
                ax = axs[i, j]
                df = pd.DataFrame(sx[:, 0, :], columns=np.arange(max_lag+1)*10,
                                  index=(np.arange(61)-10)*10)
                sns.heatmap(df, ax=ax, cbar=False, cmap='vlag')
                ax.set(xlabel='time lag', ylabel='train timepoint', title=f'{rs_names[i]} - {dirs[j]}')
        plotting.normalize_lims(axs, 'v')
        fig.suptitle(f'Sequenceness as function of train timepoint (zscore across {axis=} {name}) {C=}')
        utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_zscore-axis{axis}_C{C}.png')
        utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_zscore-axis{axis}_C{C}.svg')
        utils.savefig(fig, f'/heatmap/train_timepoint_sequencess_zscore-axis{axis}_C{C}.eps')
