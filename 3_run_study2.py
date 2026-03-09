#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 22:43:58 2025

All the code to run the hybrid simulation.

@author: simon
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:08:58 2024

@author: simon.kern
"""
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
from meg_utils import  decoding, misc

import joblib
from joblib import Parallel, delayed
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2
from utils import get_performance
from utils import plot_correlation, zscore_multiaxis
from scipy import stats
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


raise misc.Stop('data loaded. continue?')

#%%  %%% STUDY 2:  SIMULATION
#%% Sim-decodability in RS
# in this segment we show that we can decode events that are inserted into
# the resting state by comparing the probability estimates given before
# and after
np.random.seed(42)  # for safety only, should not be necessary

tp = 31 # defined here so no need to recompute

# save forward and backward sequenceness per subject in these lists
sf_sim = []
sb_sim = []

# create axes to plot subject sequenceness into
# fig_proba, axs_proba = utils.make_fig(n_axs=len(subjects_incl), bottom_plots=0, suptitle='Simulation results')

# convert sequence to ints
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]

df_proba = pd.DataFrame()
df_acc = pd.DataFrame()
df_datastat = pd.DataFrame()

for i, subj in enumerate(tqdm(subjects, desc="subject")):
    # train our classifier using localizer data and negative data
    train_x, train_y = localizer[subj]

    clf.reset_random_state(misc.make_seed(subj))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # get data from localizer that is inserted into the resting state
    insert_data_tp = train_x[:, :, tp]  # take the ERP component at peak decodability
    insert_labels = train_y

    insert_class_mean = []
    for label in set(train_y):
        insert_class_mean += [insert_data_tp[train_y==label, :].mean(0)]
    insert_class_mean = np.stack(insert_class_mean)

    if mode=='mean_class':
        # Calculate mean values that should be inserted per class
        insert_data = insert_class_mean
        insert_labels = sorted(set(train_y))

    elif mode=='erp_diff_all':
        # create grand average across all trials
        erp = insert_data_tp.mean(0)

        # now for each class, subtract the mean EPR from the class ERP.
        # this way, we should isolate the pattern that is responsible
        # for the differences
        insert_class_pattern = []
        for label in set(train_y):
            insert_class_pattern += [insert_data_tp[train_y==label, :].mean(0)-erp]
        insert_class_pattern = np.stack(insert_class_pattern)
        insert_data = insert_class_pattern
        insert_labels = sorted(set(train_y))

    elif mode=='erp_diff_others':
        # now for each class, subtract the mean EPR from the class ERP.
        # this way, we should isolate the pattern that is responsible
        # for the differences
        insert_class_pattern = []
        for label in set(train_y):
            erp_others = insert_data_tp[train_y!=label, :].mean(0)
            insert_class_pattern += [insert_data_tp[train_y==label, :].mean(0)-erp_others]
        insert_class_pattern = np.stack(insert_class_pattern)
        insert_data = insert_class_pattern
        insert_labels = sorted(set(train_y))
    else:
        raise Exception

    # insert into resting state data
    rs_pre = rs1[subj]
    rs_sim, df_onsets = tdlm.utils.insert_events(data = rs_pre,
                                                 insert_data = insert_data,
                                                 insert_labels = insert_labels,
                                                 sequence = sequence,
                                                 n_steps=1,
                                                 refractory=15,
                                                 lag=lag_sim,
                                                 n_events = 500,
                                                 return_onsets=True)

    # take some datapoints that have been inserted and some that have been left alone
    df_datastat = pd.concat([df_datastat,
                             pd.DataFrame({'data' : rs_pre[df_onsets.pos[:500], :].ravel(),
                                           'type': 'baseline',
                                           'subj':subj}),
                             pd.DataFrame({'data' : rs_sim[df_onsets.pos[:500], :].ravel(),
                                           'type': 'with pattern',
                                           'subj':subj})],
                             ignore_index=True)

    # get probabilities of simulated resting state
    proba_pre = clf.predict_proba(rs_pre)
    proba_sim = clf.predict_proba(rs_sim)

    proba_pre_raw = proba_pre.copy()
    proba_sim_raw = proba_sim.copy()

    proba_pre = eval(proba_norm)(proba_pre)
    proba_sim = eval(proba_norm)(proba_sim)

    acc_pre = (proba_pre[df_onsets.pos].argmax(1)==df_onsets.class_idx).mean()
    acc_sim = (proba_sim[df_onsets.pos].argmax(1)==df_onsets.class_idx).mean()
    df_acc = pd.concat([df_acc, pd.DataFrame({'accuracy': [acc_pre, acc_sim],
                                              'condition': ['baseline', 'with pattern'],
                                              'subj': subj})], ignore_index=True)

    # extract probabilities from four different positions
    # inserted probability positions - target and non-target class
    # non-inserted positions - target and non-target class
    proba_pre_target = [proba_pre[row.pos][row.class_idx] for _, row in df_onsets.iterrows()]
    proba_pre_nontarget = list(np.ravel([proba_pre[row.pos][~np.in1d(range(10), 5)] for _, row in df_onsets.iterrows()]))

    proba_sim_target = [proba_sim[row.pos][row.class_idx] for _, row in df_onsets.iterrows()]
    proba_sim_nontarget = list(np.ravel([proba_sim[row.pos][~np.in1d(range(10), 5)] for _, row in df_onsets.iterrows()]))

    ## also add the non-normalized version
    proba_pre_target_raw = [proba_pre_raw[row.pos][row.class_idx] for _, row in df_onsets.iterrows()]
    proba_pre_nontarget_raw = list(np.ravel([proba_pre_raw[row.pos][~np.in1d(range(10), 5)] for _, row in df_onsets.iterrows()]))

    proba_sim_target_raw = [proba_sim_raw[row.pos][row.class_idx] for _, row in df_onsets.iterrows()]
    proba_sim_nontarget_raw = list(np.ravel([proba_sim_raw[row.pos][~np.in1d(range(10), 5)] for _, row in df_onsets.iterrows()]))


    df_proba_subj = pd.DataFrame({'proba': proba_pre_target + proba_pre_nontarget +
                                           proba_sim_target + proba_sim_nontarget,
                                  'proba_raw' : proba_pre_target_raw + proba_pre_nontarget_raw +
                                                proba_sim_target_raw + proba_sim_nontarget_raw,
                                  'condition': ['baseline']*len(proba_pre_target) +
                                               ['baseline']*len(proba_pre_nontarget) +
                                               ['with pattern']*len(proba_sim_target) +
                                               ['with pattern']*len(proba_sim_nontarget),
                                  'class': ['target']*len(proba_pre_target) +
                                           ['other']*len(proba_pre_nontarget) +
                                           ['target']*len(proba_sim_target) +
                                           ['other']*len(proba_sim_nontarget),
                                  'subject': subj})

    # sns.violinplot(data=df_proba_subj, x='class', y='proba', ax=axs_proba[i])
    df_proba = pd.concat([df_proba, df_proba_subj])

fig = plt.figure(figsize=[6, 4])
ax = fig.subplots(1,1)
df_proba = df_proba.groupby(['subject', 'condition', 'class']).mean(0).reset_index()
df_proba = df_proba.sort_values(['condition'], ascending=True).sort_values('class')

sns.boxplot(data=df_proba, x='condition', y='proba', hue='class', ax=ax)
ax.set_ylabel('normalized classifier\nprobability estimate')
ax.set_title(f'Decoded probability of item\nat baseline and with added pattern')
utils.savefig(fig, f'supplement/rs_sim_decodability_norm_mode-{mode}.png')

fig = plt.figure(figsize=[6, 4])
ax = fig.subplots(1,1)
sns.boxplot(data=df_acc, x='condition', y='accuracy', ax=ax)
ax.set_ylabel('decoding accuracy')
ax.set_title(f'Decoding accuracy of item\nat baseline and with added pattern')
utils.savefig(fig, 'supplement/rs_sim_accuracy_norm_mode-{mode}.png')

fig, ax = plt.subplots(1, 1, figsize=[6, 4])
sns.histplot(data=df_datastat, x='data', hue='type', bins=300, kde=False,
             stat='percent', ax=ax, alpha=0.5)
ax.set_ylabel('percentage')
ax.set_xlabel('uT')
ax.set_title(f's')
ax.set_xlim(-0.5, 0.5)
ax.set_title('Sensor values at baseline\nand with added pattern')
sns.despine()
utils.savefig(fig, 'supplement/sensor-values-before-after-insertion.png')


# ## plot probabilities at baseline for subjects
# stop


fig, axs = plt.subplots(1, 2, figsize=[12,6])
for y, ax in zip(['proba_raw', 'proba'], axs):
    hues = {subj:palette[int(100*(get_performance(subj)-0.5)*2)] for subj in subjects}
    df_proba_target = df_proba[df_proba['class']=='target']
    df_proba_target = df_proba_target.groupby(['condition', 'subject']).mean(True).reset_index()
    df_proba_target = df_proba_target.sort_values('condition', ascending=True)
    sns.scatterplot(data=df_proba_target, x='condition', y=y, hue='subject',
                    legend=False, s=(150), palette=hues, ax=ax)
    sns.lineplot(data=df_proba_target, x='condition', y=y, hue='subject',
                    legend=False, alpha=0.3, palette=hues, ax=ax)
    ax.set_ylabel('mean probability estimate')
    ax.set_title('Before normalization' if y=='proba_raw' else 'After normalization')
fig.suptitle('Classifier probabilities across participants')
utils.savefig(fig, 'supplement/mean-probability-normalization.png')


# ## correlation between baseline probability magnitude and sequenceness
# fig, axs = plt.subplots(1, 2, figsize=[12, 6])
# df_seq = df_sequenceness[(df_sequenceness.condition=='rs1') & (df_sequenceness.direction=='fwd')]
# df_seq = df_seq.set_index('subject')
# df_prb = df_proba[df_proba.condition=='before insertion'].groupby('subject').mean(True).proba
# df_reg = df_seq.join(df_prb)
# ax = axs[0]
# sns.regplot(df_reg, x='proba', y='peak', ax=axs[0])
# ax.set_xlabel('mean baseline probability')
# ax.set_ylabel('peak sequenceness')

# ax=axs[1]
# sns.regplot(df_reg, x='proba', y='mean', ax=ax)
# ax.set_xlabel('mean baseline probability')
# ax.set_ylabel('mean sequenceness')
# fig.suptitle('Correlation RS pre sequenceness ~ baseline probabilities')
# utils.savefig(fig, 'supplement/regression-baseline-probability-sequenceness.png')



#%% simulate replay in control resting state
print('Warning, this might take around >16GB of RAM')
np.random.seed(42)  # just a failsafes

# warnings.warn('zscore disabled')
# disabled here, but calculated globally instead
zscore_multiaxis = lambda x, axes: x

# here we define replay densities that we want to simulate
# replay density is defined in events/minute or events^-1min
densities = np.arange(0, 210, 10)

# preallocate pool for later reuse
pool = Parallel(n_jobs)


# run twice, once for the regular, linear case, and once for the best-case
# which means scaling from 0-100%

# collect both simulated resting states here
rs_sims = {name:[] for name in names_perf_scale}
sf_sim_all = {name:[] for name in names_perf_scale}
sb_sim_all = {name:[] for name in names_perf_scale}

#### insert events
for name_perf_scale in names_perf_scale:
    if name_perf_scale=='linear':
        # perf_scale = lambda perf: perf  #  linear relationship
        perf_scale = lambda perf: -perf+1.5  # inverse linear relationship
    elif name_perf_scale=='best-case':
        perf_scale = lambda perf: (-perf+1.5-0.5)*2  # toggle this for best case scenario
        # perf_scale = lambda perf: (perf-0.5)*2  # toggle this for reverse best case scenario
    else:
        raise ValueError(f'{name_perf_scale=} unknown')


    pkl_simdensity = settings.cache_dir + f'/simulation-full-{name_perf_scale}.pkl.zip'

    # save resting state simulation data if requested
    # preallocate array of all resting state simulations across all densities
    # warning, this is a very large array, >64 GB
    if save_simulation_data_in_memory:
        arr_size = [len(subjects_incl), len(densities),  *rs1[subj].shape]
        rs_sim_densities = np.full(arr_size, np.nan)

    # save sequenceness result in these arrays
    rs_sim_sf_noalpha = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)
    rs_sim_sf = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)
    rs_sim_sb = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)

    probabilities = []  # collect here for visualization

    #### inserting events for each subject
    for s, subj in enumerate(tqdm(subjects_incl, desc=f"simulating replay across densities for {name_perf_scale}")):
        train_x, train_y = localizer[subj]
        insert_data_tp = train_x[:, :, tp-2:tp+3]  # take the ERP component at peak decodability
        insert_data_tp *= gaussian_weighting  # gaussian weighted events

        # create stimulus-specific patterns
        insert_class_pattern = []
        insert_labels = sorted(set(train_y))
        for label in insert_labels:
            erp_others = insert_data_tp[train_y!=label, :, :].mean(0)
            insert_class_pattern += [insert_data_tp[train_y==label, :, :].mean(0)-erp_others]
        insert_class_pattern = np.stack(insert_class_pattern)
        insert_data = insert_class_pattern

        # we're using the pre-resting state a base for simulation
        rs_pre = rs1[subj]

        # replay density inverse with performance
        perf = get_performance(subj=subj, which='test')
        scale = perf_scale(perf)

        # get probability estimates for item reactivation from the resting state
        # scale the number of replay events by the performance
        n_events_all = [int(np.round(len(rs_pre)/sfreq/60 * dens * scale)) for dens in densities]

        rs_sim_subj = pool(delayed(tdlm.utils.insert_events)(data = rs_pre,
                                      insert_data = insert_data,
                                      insert_labels = np.array(insert_labels),
                                      lag=lag_sim,
                                      refractory=refractory,
                                      distribution='constant',
                                      n_steps=1,
                                      sequence = sequence,
                                      n_events = n_events,
                                      # return_onsets = True,
                                      rng=misc.make_seed(n_events, s)
                                      ) for n_events in n_events_all)

        # if requested, save into memory as well
        if save_simulation_data_in_memory:
            rs_sim_densities[s] = np.array(rs_sim_subj)
            rs_sims[name_perf_scale] = rs_sim_densities


        #### calculate TDLM
        # train our classifier using localizer data and negative data
        train_x, train_y = localizer[subj]
        clf.reset_random_state(misc.make_seed(s))
        clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

        res_d = pool(delayed(clf.predict_proba)(rs_sim_subj[d])
                     for d, _ in enumerate(densities))

        # 1. Pre-calculate normalized probabilities
        # This is computationally cheap and can remain serial or a simple list comp
        probas_norm = [eval(proba_norm)(res_d[d]) for d in range(len(densities))]
        probabilities.extend(probas_norm)

        # 2. Parallelize TDLM calls directly
        # Batch 1: No Alpha
        results_noalpha = pool(delayed(tdlm.compute_1step)(
            prob, tf=tf, n_shuf=n_shuf, max_lag=max_lag,
            alpha_freq=None,
            seed=misc.make_seed(subj, d, 'noalpha')
        ) for d, prob in enumerate(probas_norm))

        # Batch 2: With Alpha (includes progress bar)
        results_alpha = pool(delayed(tdlm.compute_1step)(
            prob, tf=tf, n_shuf=n_shuf, max_lag=max_lag,
            alpha_freq=alpha_freq,
            seed=misc.make_seed(subj, d, 'alpha')
        ) for d, prob in enumerate(probas_norm))

        # 3. Assign results to storage arrays
        for d, ((sf_no, _), (sf, sb)) in enumerate(zip(results_noalpha, results_alpha)):
            rs_sim_sf_noalpha[s, d, :, :] = sf_no
            rs_sim_sf[s, d, :, :] = sf
            rs_sim_sb[s, d, :, :] = sb

    # sanity checks, are all rows filled?
    assert not np.isnan(rs_sim_sf_noalpha[..., 1:].min())
    assert not np.isnan(rs_sim_sf[..., 1:].min())
    assert not np.isnan(rs_sim_sf[..., 1:].min())

    # zscore results to reduce inter-individudal magnitude differences
    rs_sim_sf = stats.zscore(rs_sim_sf, zscore_axes, nan_policy='omit')
    rs_sim_sb = stats.zscore(rs_sim_sb, zscore_axes, nan_policy='omit')

    # add mean sequenceness at 80millisecond of the baseline condition
    # to alleviate baseline shifts
    sf_mean_80 = rs_sim_sf[:, 0, 0, lag_sim].mean(0)
    # sb_mean_80 = rs_sim_sb[:, 0, 0, lag_sim].mean(0)  # not necessary for bkw
    sf_mean_alpha_80 = rs_sim_sf_noalpha[:, 0, 0, lag_sim].mean(0)

    rs_sim_sf[:, :, 0, lag_sim] -= sf_mean_80
    # rs_sim_sb[:, :, 0, lag_sim] -= sb_mean_80
    rs_sim_sf_noalpha[:, :, 0, lag_sim] -= sf_mean_alpha_80

    sf_sim_all[name_perf_scale] = rs_sim_sf
    sb_sim_all[name_perf_scale] = rs_sim_sb

    # save, for some later functions that will load it to speed up calculation
    joblib.dump([rs_sim_sf, rs_sim_sb, rs_sim_sf_noalpha], pkl_simdensity)


#### create sequenceness df
for name_perf_scale in names_perf_scale:
    rs_sim_sf = sf_sim_all[name_perf_scale]
    rs_sim_sb = sb_sim_all[name_perf_scale]
    df_simulation = pd.DataFrame()
    pkl_simulation = f'{settings.cache_dir}/simulation-sequenceness-{name_perf_scale}.pkl.zip'
    for s, subj in enumerate(subjects_incl):
        perf = get_performance(subj, which='test')
        for d, density in enumerate(densities):
            for dr, direction in enumerate(['fwd', 'bkw']):
                sx = [rs_sim_sf, rs_sim_sb][dr]
                sx_subj = zscore_multiaxis(sx[s, d, 0], axes=zscore_axes)
                peak_time = np.nanargmax(np.nanmean(np.abs(sx[:, d, 0, :]), 0))
                peak_sequenceness = sx_subj[peak_time]
                mean_sequenencess = np.nanmean(sx_subj)
                seq80 = sx_subj[lag_sim]
                df_tmp = pd.DataFrame({'subject': subj,
                                       'density': density,
                                       'peak': peak_sequenceness,
                                       'mean': mean_sequenencess,
                                       'peak_time': peak_time,
                                       'direction': direction,
                                       'performance': perf,
                                       'seq80': seq80,
                                        },
                                        index=[subj])
                df_simulation = pd.concat([df_simulation, df_tmp],
                                            ignore_index=True)
    joblib.dump(df_simulation, pkl_simulation)

#%% 1 peak seq. line graph

#### regular sequenceness
rs_sim_sf = sf_sim_all['linear']

fig, axs = plt.subplot_mosaic('AAABB', figsize=[16, 5])

ax = axs['A']

threshholds = np.max(abs(np.mean(rs_sim_sf[:, :, 1:, 1:], 0)), -1)
thresholds_95 = np.quantile(threshholds, 0.95, axis=-1)
thresholds_max = np.nanmax(threshholds, axis=-1)
sequenceness = pd.DataFrame({'sequenceness': rs_sim_sf[:, :, 0, lag_sim].ravel('F'),
                             'density': np.repeat(densities, len(subjects_incl))})
sns.lineplot(sequenceness, x='density', y='sequenceness', errorbar='se', ax=ax,
             label=f'mean sequencenes @ {lag_sim*10}ms', color=sns.color_palette()[1])

ax.plot(densities, thresholds_max, linestyle='-', color='gray', label='perm. max. threshold')
ax.plot(densities, thresholds_95, linestyle='--', color='gray', label='perm. 95% threshold')

seq_peak = sequenceness.groupby('density').mean()
t_max = np.argmax(seq_peak.sequenceness>thresholds_max)
t_95 = np.argmax(seq_peak.sequenceness>thresholds_95)

ax.vlines(seq_peak.index[t_max], 0, 1.5,  linestyle='-', color='red', alpha=0.7)
ax.vlines(seq_peak.index[t_95], 0, 1.5,  linestyle='--', color='red', alpha=0.7)

ax.set_xlabel('replay density (min⁻¹)')
ax.set_ylabel('forward sequenceness\n(u.u., zscored)')
ax.legend(loc='lower right')
sns.despine()
ax.set_title(f'Forward {lag_sim*10} ms sequenceness at simulated replay densities')

#### sign-flip permutation test

ax = axs['B']
ax.clear()
df = pd.DataFrame()
for d, density in enumerate(densities):
    p, t_obs, t_perms = tdlm.signflit_test(rs_sim_sf[:, d, 0, :], rng=d)
    df_tmp = pd.DataFrame({'p-value':p,
                           't-value': t_obs,
                           'density': density}, index=[0])
    df = pd.concat([df, df_tmp], ignore_index=True)
sns.lineplot(df, x='density', y='t-value', label='sign-flip t-value', ax=ax)
idx_5 = np.argmax(df['p-value']<0.05)
idx_01 = np.argmax(df['p-value']<0.001)
ax.axhline(df['t-value'][idx_5], label='p<0.05', c='red', linestyle='--')
ax.axhline(df['t-value'][idx_01], label='p<0.001', c='red', linestyle='-.', linewidth=1)
ax.set_xticks(np.arange(0, 220, 20))
ax.legend()
ax.set_title('Sign-flip Permutation t-values')

utils.savefig(fig, f'figure/simulation-{lag_sim*10}ms-sequenceness.png')

#%% 2 exemplar 4 density curves
# fig, axs = plt.subplots(2, 2, figsize=[12, 12])
fig, axs = plt.subplots(2, 2, figsize=[16, 16])
axs = axs.flatten()

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

rs_sim_sf = sf_sim_all['linear']
rs_sim_sb = sb_sim_all['linear']

for i, density in enumerate([  0,  40,  80, 120]):
    ax = axs[i]
    d = list(densities).index(density)
    ax.clear()
    tdlm.plot_sequenceness(rs_sim_sf[:, d, :], rs_sim_sb[:, d, :], which=['fwd', 'bkw'], ax=ax, rescale=True)
    ax.set_title(f'replay density {density} min⁻¹')

sns.despine(fig)
fig.suptitle('Simulated replay at different densities')
utils.savefig(fig, 'figure/simulation-sequenceness-examples.png')

#%% 3 plot sequenceness curves across densities

fig, axs = plt.subplot_mosaic('XXB\nYYD', figsize=[16, 8])
perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

df_correlation = pd.DataFrame()

for d, density in enumerate(densities):
    # normalize sequenceness scores here
    sf_sim = zscore_multiaxis(rs_sim_sf[:, d, :, :], axes=zscore_axes)
    sb_sim = zscore_multiaxis(rs_sim_sb[:, d, :, :], axes=zscore_axes)

    # first plot RS sequenceness individually
    # ax = axs[i]
    for ax in axs.values(): ax.clear()

    fig.suptitle(f'Simulated replay density {density} min⁻¹')
    tdlm.plot_sequenceness(sf_sim, sb_sim, which=['fwd'], title='forward sequenceness', ax=axs['X'], rescale=False)
    tdlm.plot_sequenceness(sf_sim, sb_sim, which=['bkw'], title='backward sequenceness', ax=axs['Y'], rescale=False)
    # axs['X'].legend(['forward sequenceness', 'max. perm. thresh.', '_', '95% perm. thresh.'])
    # for ax in [axs['X'], axs['Y']]:
    #     hndls = ax.get_legend_handles_labels()
    #     ax.legend([hndls[0][i] for i in [0, 3, 4, 5]],
    #               [hndls[1][i] for i in [0, 3, 4, 5]],
    #               loc='lower right',
    #               ncols=2, fontsize=12)
    # calculate peak sequenceness across participants
    sf_sim_80 = dict(zip(subjects_incl, sf_sim[:, 0, lag_sim]))
    sb_sim_80 = dict(zip(subjects_incl, sb_sim[:, 0, lag_sim]))

    r1, pval1 = plot_correlation(sf_sim_80, values=perf_test, color=c_fwd, title="performance correlation", ax=axs['B'])
    r2, pval2 = plot_correlation(sb_sim_80, values=perf_test, color=c_bkw, title="performance correlation", ax=axs['D'])

    df_correlation = pd.concat([df_correlation,
                                pd.DataFrame({'r': [r1, r2],
                                              'p': [pval1, pval2],
                                              'direction': ['forward', 'backward'],
                                              'density': density})])

    utils.normalize_lims([axs['X'], axs['Y']])
    utils.normalize_lims([axs['B'], axs['D']])

    plt.pause(0.1)
    fig.tight_layout()
    axs['B'].text(axs['B'].get_xlim()[1], 1.0, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')
    axs['D'].text(axs['D'].get_xlim()[1], 1.0, f'r={r2:.2f} p={pval2:.3f}', horizontalalignment='right')
    plt.pause(0.1)
    fig.savefig(f'{settings.plot_dir}/simulation_density-{density:03d}_step1.png')

# plot correlation at 160min-1 with performance
assert name_perf_scale=='linear', 'wrong setting'

sf_sim_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)
fig, ax = plt.subplots(1, 1, figsize=[8, 6])
idx_density_160 = np.argmax(densities>=160)
peak_lag =  np.nanargmax(abs(np.nanmean(sf_sim_normed[:, idx_density_160, 0, :], 0)))
peak_vals = dict(zip(subjects_incl, sf_sim_normed[:, idx_density_160, 0, peak_lag]))
r, pval = plot_correlation(peak_vals, values=perf_test, ax=ax,title=f"Simulation @ 160min-1")

ax.text(ax.get_xlim()[1], .5, f'r={r:.2f} p={pval:.3f}', horizontalalignment='right')


#%% 4 individual sequenceness curves
perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

### plot the individual sequenceness curves into a big plot
fig, axs, ax_b = utils.make_fig(n_axs=len(sf_sim), bottom_plots=[0,0,1])
for d, density in enumerate(tqdm(densities)):
    # normalize sequenceness scores here
    sf_sim = zscore_multiaxis(rs_sim_sf[:, d, :, :], axes=zscore_axes)
    sb_sim = zscore_multiaxis(rs_sim_sb[:, d, :, :], axes=zscore_axes)

    for i, subj in enumerate(subjects_incl):
        sf = sf_sim[i, ...]
        sb = sb_sim[i, ...]
        perf = perf_test[subj]
        title = f'{subj=} {perf=:.2f}'
        tdlm.plot_sequenceness(sf, sb, ax= axs[i], which=['fwd', 'bkw'],
                               title=title, rescale=False)
    tdlm.plot_sequenceness(sf_sim, sb_sim, ax= ax_b, which=['fwd', 'bkw'],
                           title=f'mean with {density=}', rescale=False)

    utils.normalize_lims(axs[:len(subjects_incl)])
    utils.savefig(fig, f'supplement/simulation_density-all-{density:03d}.png')





# ### heatmap
# fig, ax = plt.subplots(1, 1, figsize=[8, 6])
# vmin = np.min(np.mean(rs_sim_sf[:,:,1:,1:],0))
# vmax = np.max(np.mean(rs_sim_sf[:,:,1:,1:],0))

# heatmap = np.nanmean(rs_sim_sf[:, :, 0, :], 0)
# im = ax.imshow(heatmap, aspect='auto', cmap='PiYG', vmin=vmin, vmax=vmax)
# ax.set_xlabel('time lag')
# ax.set_xticks(np.arange(0, 31, 5), np.arange(0, 320, 50))
# ax.set_ylabel('Simulated base density min^-1')
# ax.set_yticks(np.arange(len(densities)), densities)
# ax.set_title("forward sequenceness")
# plt.colorbar(im)
# plt.draw()
# plt.tight_layout()
# utils.log_fig(fig, f'{settings.plot_dir}/simulation-heatmap.png')



#%% 5a Corr with performance

pkl_sim_linear = f'{settings.cache_dir}/simulation-sequenceness-linear.pkl.zip'
pkl_sim_bestcase = f'{settings.cache_dir}/simulation-sequenceness-best-case.pkl.zip'

df_sim_linear = joblib.load(pkl_sim_linear)
df_sim_bestcase = joblib.load(pkl_sim_bestcase)
df_sim_linear['performance scaling'] = 'realistic scaling'
df_sim_bestcase['performance scaling'] = 'maximal scaling'

df_both = pd.concat([df_sim_linear, df_sim_bestcase], ignore_index=True)
df_both = df_both[df_both.direction=='fwd']

df_corr = df_both.groupby(['performance scaling', 'density']).apply(
            lambda group: pd.Series(stats.pearsonr(group.seq80, group.performance), index=['r', 'p'])
        ).reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
palette = sns.color_palette()

sns.lineplot(df_corr, x='density', y='r', hue='performance scaling',
             ax=ax, palette=[palette[0], 'yellowgreen'])

r_sign = utils.calculate_r_threshold(len(df_both.subject.unique()))  # this is the r value at which it will be significant

ax.hlines(-r_sign, *ax.get_xlim(), linestyle='--', color='red', label='p<0.05', alpha=0.5)
ax.legend(loc='upper left', framealpha=1.0)
ax.invert_yaxis()

ax.set_ylabel('corr(perf, sequenceness)')
ax.set_xlabel('replay density (events min⁻¹)')
ax.set_title('Performance correlation across densities')

sns.despine()
utils.savefig(fig, 'figure/correlation-densities.png')
#%% 5b Corr timelag

rs_sim_sf_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)
rs_sim_sf_noalpha_normed = zscore_multiaxis(rs_sim_sf_noalpha, axes=zscore_axes)

perf_test = [get_performance(subj=subj, which="test") for subj in subjects_incl]

df_corr = pd.DataFrame()
for d, density in enumerate(tqdm(densities)):
    for lag in range(1, max_lag+1):
        r1, p1 = stats.pearsonr(rs_sim_sf_normed[:, d, 0, lag], perf_test)
        r2, p2 = stats.pearsonr(rs_sim_sf_noalpha_normed[:, d, 0, lag], perf_test)
        df_corr = pd.concat([df_corr, pd.DataFrame({'density': density,
                                                    'lag': lag*10,
                                                    'r': [r1, r2],
                                                    'p': [p1, p2],
                                                    'alpha-control': ['alpha-ctrl', 'No ctrl']})],
                            ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=[8, 6])

df_corr_sel = df_corr[(df_corr.density==0) & (df_corr['alpha-control']=='alpha-ctrl')]
sns.lineplot(df_corr_sel, x='lag', y='r', marker='o',
             dashes=False, ax=ax, label='r')
ax.set_xlabel('time lag (ms)')
ax.set_ylabel('corr(perf, sequenceness)')

r_sign = utils.calculate_r_threshold(len(subjects_incl))  # this is the r value at which it will be significant

ax.hlines([-r_sign, r_sign], *ax.get_xlim(), color='red',
          alpha=0.5, linestyle='--', label='p=0.05')
ax.legend()
ax.set_title('Performance correlation across time lags at baseline')
sns.despine()
utils.savefig(fig, 'figure/correlation-timelags.png')

#%% 5c power curve, bootstrapping
perf_test = [get_performance(subj=subj, which="test") for subj in subjects_incl]

pkl_simdensity = settings.cache_dir + f'/simulation-full-best-case.pkl.zip'
rs_sim_sf_best, rs_sim_sb_best, _ = joblib.load(pkl_simdensity)

np.random.seed(20241009)  # todays date as random seed
n_repetitions = 10000
sample_sizes = range(15, 200, 5)

density_95 = np.nonzero(densities>=120)[0][0]

def r_val_boostrap(n):
    rs = []
    ps = []
    for repetition in range(n_repetitions):
        idx_sample = np.random.choice(np.arange(len(rs_sim_sf_best)), size=n)
        seq_bootstrapped = rs_sim_sf_best[idx_sample, density_95, 0, lag_sim]
        perf_bootstrapped = [perf_test[x] for x in idx_sample]
        r, p = stats.pearsonr(seq_bootstrapped, perf_bootstrapped)
        rs += [r]
        ps += [p]
    return rs, ps

res = Parallel(n_jobs)(delayed(r_val_boostrap)(n) for n in tqdm(sample_sizes))

df_power = pd.DataFrame({'r': np.ravel([x[0] for x in res]),
                         'p': np.ravel([x[1] for x in res]),
                         'n': np.repeat(sample_sizes, n_repetitions)})

df_power['p<0.05'] = df_power.p<0.05

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
sns.lineplot(df_power, x='n', y='p<0.05', ax=ax, label='% significant', linestyle='--', color='r', alpha=0.5)
# ax2  = ax.twinx()
# sns.lineplot(df_power, x='n', y='r', ax=ax2, label='Mean Pearson\'s r', alpha=0.2)

ax.set_xlabel('bootstrapped sample size')
# ax2.set_ylabel('correlation p value / Pearson\'s r')
ax.set_ylabel('Power')
ax.set_title('Correlation Perf. x Sequ. with increased sample size')
ax.hlines(0.8, *ax.get_xlim(), color='gray', alpha=0.5, linestyle='--', label='80% power')
# ax.vlines(150, 0, 1,
#          color='black', alpha=0.5, linestyle='--')
ax.legend()
sns.despine()
utils.savefig(fig, 'figure/correlation-power.png')

#%% 5d  plot individual sequenceness
pkl_sim_linear = f'{settings.cache_dir}/simulation-sequenceness-linear.pkl.zip'
pkl_sim_bestcase = f'{settings.cache_dir}/simulation-sequenceness-best-case.pkl.zip'

df_sim_baseline = df_simulation[df_simulation.density==0]
df_sim_linear = joblib.load(pkl_sim_linear)
df_sim_bestcase = joblib.load(pkl_sim_bestcase)
df_sim_linear = df_sim_linear[df_sim_linear.density==80]
df_sim_bestcase = df_sim_bestcase[df_sim_bestcase.density==180]

df_sim_baseline['condition'] = 'baseline'
df_sim_linear['condition'] = 'realistic scaling\n(80 min-1)'
df_sim_bestcase['condition'] = 'maximum scaling\n(160 min-1)'

df_sim_baseline['sorter'] = 1
df_sim_linear['sorter'] = 2
df_sim_bestcase['sorter'] = 3

df_sel = pd.concat([df_sim_baseline, df_sim_linear, df_sim_bestcase])
df_sel = df_sel[df_sel.direction=='fwd']
df_sel = df_sel.groupby(['condition', 'subject']).mean(True).reset_index()

df_sel.sort_values('sorter', inplace=True)

# create color palette based on performance
fig, ax = plt.subplots(figsize=[8, 6])
palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=151)[::-1]


hues = {subj:palette[int(100*(get_performance(subj)-0.5)*2)] for subj in df_sel.subject}
mp = sns.scatterplot(df_sel, x='condition', y='seq80', hue='subject', palette=hues,
                  legend=True, s=100, ax=ax)
mp = sns.lineplot(df_sel, x='condition', y='seq80', hue='subject', palette=hues,
                  legend=False, ax=ax, alpha=0.5)


ax.set_ylabel('sequenceness at 80 ms (u.u.)')
ax.set_xlabel('')
ax.set_title('Sequenceness variation across subject')
handles, labels = ax.get_legend_handles_labels()
handles = [x for i, x in enumerate(handles) if i in [0, 2, 6]]
labels = [x for i, x in enumerate(labels) if i in [0, 2, 6]]

ax.legend(handles, ['50%', '75%', '100%'], loc='upper left', title='Performance')

utils.savefig(fig, 'figure/correlation-bestcase-realistisc-case.png')
