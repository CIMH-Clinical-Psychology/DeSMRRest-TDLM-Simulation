#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 10:25:58 2025

Compare classifier probability structure between:
(1) real localizer data, (2) "hybrid" simulation (insert subject ERPs into real RS),
(3) fully synthetic simulation (TDLM demo-style).
Quantify separability via Wasserstein distance + mean target/other ratio, and relate it
to sequenceness as a function of injected pattern strength.

@author: simon
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

from meg_utils import decoding, misc

import joblib
from joblib import Parallel, delayed
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2
from scipy.stats import wasserstein_distance
import warnings
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)  # global RNG safety; per-subject seeds set below

# plotting defaults (paper style)
plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

# suppress known benign warnings from empty slices / NaNs during aggregation
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

# empirical test performance list (used elsewhere for scaling; not central here)
performances = [1.0, 0.67, 0.75, 0.75, 0.83, 0.92, 0.83, 0.5, 0.83, 0.83, 0.92,
                0.92, 0.75, 1.0, 0.75, 0.83, 0.75, 0.5, 1.0, 0.83, 1.0, 0.67,
                0.92, 1.0, 0.58, 1.0, 1.0, 0.92, 0.5, 0.75]

utils.lowpriority()  # avoid hogging shared CPU

#%% Settings

# two classifiers differing only in neg_x_ratio (hybrid/localizer uses 2; Liu-style uses 1)
C = 9.1  # cross-validated regularization strength (see below)
clf1 = decoding.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=2, rng=0)
clf2 = decoding.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=1, rng=0)  # Liu ratio=1

# enumerate subjects based on file naming convention in data_dir
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]

# preregistered inclusion thresholds
min_acc = 0.3   # localizer decoding accuracy
min_perf = 0.5  # memory performance
sfreq = 100     # downsample frequency
ms_per_point = 10  # derived from sfreq, used for plotting labels

bands = settings.bands_HP  # high-pass filtered bands only
baseline = None            # localizer: no baseline correction

uid = hex(random.getrandbits(128))[2:10]  # run id for logs/artifacts
date = time.strftime("%Y-%m-%d")
times = np.arange(-100, 510, 10)
final_calculation = True  # toggles which RS segments are used in loaders

# optional probability normalization hook (currently unused)
proba_norm = 'lambda x: x/x.mean(0)'
# proba_norm = 'lambda x:x'

# plot colors
palette = sns.color_palette()
c_fwd = palette[1]
c_bkw = palette[2]

#%% define TDLM parameters
n_shuf = 1000   # permutations per condition
max_lag = 30    # max lag in samples (at 100 Hz: 300 ms)
alpha_freq = 10 # assumed alpha peak (unused below; kept for context)
zscore_axes = -1  # which permutation axis gets zscored in downstream helpers

# sweep synthetic insertion strength to map discriminability -> sequenceness
strengths = np.linspace(0, 2, 21)

# forward transition matrix for seq_12
tf = tdlm.seq2tf(settings.seq_12)

# labels for alternative performance scaling schemes (not used in this script section)
names_perf_scale = ['best-case', 'linear']

#%% simulation parameters

mode = 'erp_diff_all'  # how insertion patterns are constructed (ERP contrast)
lag_sim = 8            # "true" replay lag = 80 ms (used elsewhere)
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]
tp = 31                # peak decoding timepoint index (~210 ms)
best_C = 9.1           # retained for documentation

# refractory window for insertion (block around injected onsets)
refractory = [lag_sim*2, lag_sim]

# avoid caching huge simulated RS arrays in RAM
save_simulation_data_in_memory = False
n_jobs = -1  # joblib parallelism hint

if not final_calculation:
    print('not running final calculation yet, only use this for final paper calculation')

# cached best decoding timepoint per subject (fallback dict for missing keys)
best_tp = utils.load_pkl(f"{settings.cache_dir}/best_tp.pkl.zip", {31: 31})

# per-subject colors keyed by performance (used only for plotting aesthetics)
palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=101)
hues = {subj: palette[int(100*(utils.get_performance(subj)-0.5)*2)] for subj in subjects}

# gaussian temporal weighting (from gaussian_filter1d([0,0,1,0,0], 1))
gaussian_weighting = [0.05842299, 0.24210528, 1, 0.24210528, 0.05842299]

#%% preload some data (localizer, neg_x, resting state)
localizer = {}  # (X, y) per subject
rs1 = {}
rs2 = {}
neg_x = {}      # fixation cross negatives before audio cue
seqs = {}       # subject sequences (mostly unused below)

for subj in tqdm(subjects, desc="Loading data"):
    localizer[subj] = load_localizers_seq12(
        subj=subj, sfreq=sfreq, bands=bands,
        autoreject=settings.default_autoreject,
        ica=settings.default_ica_components
    )
    neg_x[subj] = load_neg_x_before_audio_onset(
        subj=subj, sfreq=sfreq, bands=bands,
        autoreject=settings.default_autoreject,
        ica=settings.default_ica_components
    )

    # RS data (open/closed concatenated in loaders)
    rs1[subj] = load_RS1(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)
    rs2[subj] = load_RS2(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)

    seqs[subj] = utils.get_sequences(subj)

# subject-level QC metrics for exclusion
max_accuracy = [utils.get_decoding_accuracy(subj, clf=clf1, n_splits=20)[0] for subj in subjects]
test_performance = [utils.get_performance(subj, which='test') for subj in subjects]
df_subjects = pd.DataFrame({
    'subject': subjects,
    'accuracy': max_accuracy,
    'performance': test_performance
}).sort_values('accuracy')

# preregistered exclusion sets
excluded_acc = [subj for subj in subjects if utils.get_decoding_accuracy(subj, clf=clf1, n_splits=20)[0] < 0.3]
excluded_perf = [subj for subj in subjects if utils.get_performance(subj, which='test') < 0.5]
excluded_miss = [subj for subj in subjects if utils.get_responses_localizer(subj)['n_misses'] > 60*0.25]

df_excluded = pd.DataFrame({
    'reason': ['low decoding']*len(excluded_acc)
            + ['low performance']*len(excluded_perf)
            + ['>25% misses']*len(excluded_miss)
}, index=excluded_acc + excluded_perf + excluded_miss)
df_excluded.sort_index(inplace=True)

subjects_incl = sorted(set(subjects).difference(set(df_excluded.index)))

#%% probability separability: target vs non-target

df_strength = pd.DataFrame()
pool = Parallel(8)

def split_proba(proba, idx1, idx2):
    """Extract p(target) and pooled p(other) for given row indices and true classes."""
    assert len(idx1) == len(idx2), 'must be same length idx1 and idx2'

    # p_target: probability assigned to the true class at each sample
    p_target = proba[idx1, idx2]

    # p_other: probabilities of all non-true classes (flattened)
    all_indices = np.arange(max(idx2) + 1)
    mask = all_indices != idx2[:, None]
    p_other = proba[idx1][mask]
    return p_target, p_other


df_data = pd.DataFrame()  # per-subject summary metrics
df_hist = pd.DataFrame()  # per-sample probabilities for histograms

# store synthetic sequenceness curves across strengths for later aggregation
sf_synthetic = np.full([len(subjects_incl), len(strengths), n_shuf, max_lag+1], np.nan)

for s, subj in enumerate(tqdm(subjects_incl)):
    # localizer data at a fixed peak timepoint
    data_x, data_y = localizer[subj]
    data_x = data_x[:, :, tp]
    cv = StratifiedKFold(10)

    # ----- Localizer: cross-validated probability separation
    p_targets = []
    p_others = []
    for idx_train, idx_test in cv.split(data_x, data_y):
        train_x = data_x[idx_train]
        train_y = data_y[idx_train]
        test_x = data_x[idx_test]
        test_y = data_y[idx_test]

        clf1.reset_random_state(misc.make_seed(*idx_train))
        clf1.fit(train_x, train_y, neg_x=neg_x[subj])

        proba = clf1.predict_proba(test_x)
        # proba = eval(proba_norm)(proba)

        p_target, p_other = split_proba(proba, np.arange(len(test_y)), test_y)
        p_targets.extend(p_target)
        p_others.extend(p_other)

    # summary metrics: distribution distance + mean ratio
    d = wasserstein_distance(p_target, p_other)
    ratio = np.mean(p_target) / np.mean(p_other)

    df_data = pd.concat([df_data, pd.DataFrame({
        'subject': subj,
        'type': 'Localizer',
        'wasserstein': d,
        'ratio': ratio
    }, index=[0])], ignore_index=True)

    df_tmp = pd.DataFrame({
        'proba': [*p_target, *p_other],
        'type': 'Localizer',
        'class': ['target']*len(p_target) + ['other']*len(p_other),
        'subject': subj
    })
    df_hist = pd.concat([df_hist, df_tmp], ignore_index=True)

    # ----- Hybrid simulation: insert subject ERP-contrast patterns into real RS
    insert_class_pattern = []
    for label in set(data_y):
        erp_others = data_x[data_y != label, :].mean(0)
        insert_class_pattern += [data_x[data_y == label, :].mean(0) - erp_others]
    insert_class_pattern = np.stack(insert_class_pattern)

    insert_data = insert_class_pattern
    insert_labels = sorted(set(train_y))

    # insert into a short RS segment (memory-safe) and track onsets for labeling
    rs_sim, df_onset = tdlm.simulation.insert_events(
        rs1[subj][:2000],
        sequence=sequence,
        lag=2,                 # smaller lag to reduce computation here
        refractory=[4, 2],
        insert_data=insert_data,
        insert_labels=insert_labels,
        n_events=len(data_x)//2,
        return_onsets=True,
        rng=misc.make_seed(subj)
    )

    # train decoder on last CV fold's train split (note: uses train_x/train_y from above loop)
    clf1.reset_random_state(misc.make_seed(subj))
    clf1.fit(train_x, train_y, neg_x=neg_x[subj])

    proba = clf1.predict_proba(rs_sim)
    p_target, p_other = split_proba(proba, df_onset.pos.values, df_onset.class_idx.values)

    d = wasserstein_distance(p_target, p_other)
    ratio = np.mean(p_target) / np.mean(p_other)

    df_data = pd.concat([df_data, pd.DataFrame({
        'subject': subj,
        'type': 'Hybrid Simulation',
        'wasserstein': d,
        'ratio': ratio
    }, index=[0])], ignore_index=True)

    df_tmp = pd.DataFrame({
        'proba': [*p_target, *p_other],
        'type': 'Hybrid Simulation',
        'class': ['target']*len(p_target) + ['other']*len(p_other),
        'subject': subj
    })
    df_hist = pd.concat([df_hist, df_tmp], ignore_index=True)

    # ----- Fully synthetic simulation: sweep pattern strength, compute separability + TDLM
    clf2 = decoding.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=1, rng=0)

    # generate synthetic M/EEG + synthetic class patterns used for insertion
    rs = tdlm.simulation.simulate_meeg(20, sfreq, 306, rng=misc.make_seed(subj))
    train_x, train_y, patterns = tdlm.simulation.simulate_classifier_patterns(
        n_patterns=10,
        n_channels=306,
        noise=4,
        n_train_per_stim=18,
        rng=misc.make_seed(subj)
    )

    # isolate null class for negatives
    null_x = train_x[train_y == 0]
    train_x = train_x[train_y > 0]
    train_y = train_y[train_y > 0] - 1

    clf2.reset_random_state(s)
    clf2.fit(train_x, train_y, neg_x=null_x)

    n_events = 240  # fixed injection count for the strength sweep

    # precompute inserted RS segments and decoded probabilities across strengths (parallelized)
    res1 = pool(
        delayed(tdlm.simulation.insert_events)(
            rs,
            insert_data=patterns*strength,
            insert_labels=np.arange(10),
            lag=2,
            refractory=[4, 2],
            sequence=sequence,
            n_events=n_events,
            return_onsets=True,
            rng=s
        ) for strength in strengths
    )
    res2 = pool(delayed(clf2.predict_proba)(r[0]) for r in res1)

    for i, strength in enumerate(strengths):
        rs_sim, df_onset = res1[i]
        proba = res2[i]

        # redundant recomputation; left as-is to keep behavior identical
        proba = clf2.predict_proba(rs_sim)

        # discriminability in probability space at injected onsets
        p_target, p_other = split_proba(proba, df_onset.pos.values, df_onset.class_idx.values)
        d = wasserstein_distance(p_target, p_other)
        ratio = np.mean(p_target) / np.mean(p_other)

        # add shared temporal autocorrelation term (matches Liu-style synthetic structure)
        coreP = np.full([len(rs_sim)], np.nan)
        coreP[0] = np.random.randn()

        rand_v = np.random.randn(len(rs_sim))
        for iT in range(1, len(rs_sim)):
            coreP[iT] = 0.95 * (coreP[iT-1] + rand_v[iT])

        proba2 = (proba.T + 0.05 * coreP).T
        p_target, p_other = split_proba(proba2, df_onset.pos.values, df_onset.class_idx.values)

        # record the Liu-strength (=1) synthetic condition for direct comparison
        if strength == 1:
            df_data = pd.concat([df_data, pd.DataFrame({
                'subject': subj,
                'type': 'Synthetic Simulation',
                'wasserstein': d,
                'ratio': ratio
            }, index=[0])], ignore_index=True)

            df_tmp = pd.DataFrame({
                'proba': [*p_target, *p_other],
                'type': 'Synthetic Simulation',
                'class': ['target']*len(p_target) + ['other']*len(p_other),
                'subject': subj
            })
            df_hist = pd.concat([df_hist, df_tmp], ignore_index=True)

        # TDLM sequenceness curve for this strength
        sf, sb = tdlm.compute_1step(
            proba2, tf, n_shuf=n_shuf, max_lag=max_lag, seed=misc.make_seed(s, subj)
        )
        sf_synthetic[s, i] = sf

        # store strength-wise discriminability summary
        df_strength = pd.concat([df_strength, pd.DataFrame({
            'subject': subj,
            'type': 'Synthetic Simulation',
            'wasserstein': d,
            'ratio': ratio,
            'strength': strength
        }, index=[0])], ignore_index=True)

# aggregate synthetic curves to relate strength -> mean sequenceness and null thresholds
seq_means = [np.mean(sf_synthetic[:, i, 0, 2], 0) for i in range(len(strengths))]
perm_maxes = [np.max(abs(np.mean(sf_synthetic[:, i, 1:, 1:], 0)), -1) for i in range(len(strengths))]
thresh_max = [max(perm_maxes[i]) for i in range(len(strengths))]
thresh_95 = [np.quantile(perm_maxes[i], 0.95) for i in range(len(strengths))]

df_seq = pd.DataFrame({
    'Strength': strengths,
    'Mean Sequenceness': seq_means,
    'Perm. Max.': thresh_max,
    'Perm. 95%': thresh_95,
})

# persist intermediate results for plotting / reruns
if len(df_data) > 0:
    joblib.dump([df_data, df_strength, df_hist, df_seq], settings.plot_dir + '/signal_strength.pkl.gz')

#%% plotting

# reload cached results for reproducible plotting
df_data, df_strength, df_hist, df_seq = joblib.load(settings.plot_dir + '/signal_strength.pkl.gz')

# normalize labels for plotting
df_data.type = df_data.type.apply(lambda x: x.lower())
df_strength.type = df_strength.type.apply(lambda x: x.lower())
df_hist.type = df_hist.type.apply(lambda x: x.lower())

# average discriminability per strength and join with sequenceness summary
df_mean_strength = df_strength.groupby('strength').mean(True).reset_index()
df_seq = (
    df_seq
    .rename(columns={'Strength': 'strength'})
    .merge(df_mean_strength, on='strength', how='inner')
)

#%% panel layout + plots

mosaic = 'AABBCC\nAABBCC\nDDEEFF\nDDEEFF\nDDEEFF'
fig, axs = plt.subplot_mosaic(mosaic, figsize=[14, 9])

names = ['localizer', 'hybrid simulation', 'synthetic simulation']
colors = sns.color_palette('muted')

# histograms of target vs other probabilities (log-scale)
for i, name in enumerate(names):
    ax = axs['ABC'[i]]
    df_sel = df_hist[df_hist.type == name].copy()

    color = sns.color_palette()[i]
    if name == 'hybrid simulation':
        name = 'hybrid simulation\n(this study)'
    if name == 'synthetic simulation':
        name = 'synthetic simulation\n(Liu et al. 2021)'

    # balance target/other counts for comparable densities
    min_class = min([len(x) for _, x in df_sel.groupby('class')])
    df_sel = df_sel.groupby('class', group_keys=False).apply(lambda x: x.sample(min_class))

    df_sel['proba'] = np.log10(df_sel.proba)
    sns.histplot(
        df_sel, x='proba', stat='density', hue='class', ax=ax, bins=100,
        palette=['gray', color]
    )

    ax.set_xlabel('log classifier probability\n(at peak decodability)' if name == 'Localizer'
                  else 'log classifier probability\n(after inserting a pattern)')
    ax.set_title(f'{name}')

for ax in [axs['A'], axs['B'], axs['C']]:
    ax.set_xlim(-6, 0)

# KDE of Wasserstein distances across the three data types
ax = axs['D']
ax.clear()
for name in names:
    sns.kdeplot(df_data[df_data.type == name], x='wasserstein', label=name, ax=ax)
ax.legend(loc="upper right", fontsize=12)
ax.set_xlabel("classifier probabilitie's \nWasserstein-distance (target vs other)")
ax.set_title('Pattern Discriminability')

# sequenceness as a function of discriminability, with vertical lines at empirical means
ax = axs['F']
ax.clear()
sns.lineplot(df_seq, x='wasserstein', y='Mean Sequenceness', color='k', ax=ax)
ax.set_xlabel('pattern discriminability\n(Wasserstein distance)')
ax.set_ylabel('Sequenceness\n@ 80 min⁻¹')

for i, name in enumerate(names):
    mean = df_data[df_data.type == name].wasserstein.mean()
    print(f'mean wasserstein distance {name} = {mean}')
    if name == 'synthetic simulation':
        name = 'simulation Liu et al. (2021)'
    ax.vlines(mean, 0, 0.4, linestyle='--', linewidths=2, label=f'{name}', color=colors[i])
ax.legend(title='Pattern disciminability', loc='upper left', fontsize=12)
ax.set_title('Sequenceness by\nPattern Discriminability')

# map synthetic strength -> discriminability, and annotate estimated strength for each empirical condition
ax = axs['E']
sns.lineplot(df_strength, x='strength', y='wasserstein', color='k', ax=ax)
ax.set_xlabel('synthetic simulation pattern strength')
ax.set_ylabel('mean pattern discriminability\n(Wasserstein distance)')
ax.set_title('Pattern Discriminability by \nPattern Strength')

df_mean = df_strength.groupby('strength').mean(True).reset_index()
for i, name in enumerate(names):
    df = df_data[df_data.type == name]
    if name == 'synthetic simulation':
        name = 'simulation Liu et al. (2021)'
    idx = np.argmax(df_mean.wasserstein >= df.wasserstein.mean())
    strength = df_mean.strength[idx]
    ax.vlines(strength, 0, 1, linestyle='--', linewidths=2, label=f'{name}', color=colors[i])
ax.legend(title='Estimated signal strength', loc='upper left', fontsize=12)

fig.suptitle('Pattern Discriminability of Simulated Data and Localizer')
fig.tight_layout()
utils.savefig(fig, 'supplement/signal-strength-probabilities.png')
utils.savefig(fig, 'supplement/signal-strength-probabilities.svg')
utils.savefig(fig, 'supplement/signal-strength-probabilities.eps')
