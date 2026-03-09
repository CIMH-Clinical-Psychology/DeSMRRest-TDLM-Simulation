#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:52:53 2025

Create revision-1 supplemental plots:
- fwd-bkw sequenceness in RS1/RS2 (with sign-flip permutation diagnostics)
- bootstrapped power maps for state-permutation vs sign-flip tests
- simulation variants: shortened sequences, bursty vs uniform replay, multi-step replay, etc.

@author: simon.kern
"""
import sys
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
import mne
from meg_utils import plotting, decoding, misc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib
from joblib import Parallel, delayed
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2
from utils import get_performance
from utils import zscore_multiaxis
import warnings

np.random.seed(0)  # global RNG safety; script also sets section-specific seeds later

# plotting defaults (paper style)
plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

# suppress benign NaN warnings from averaging over masked/empty slices
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

utils.lowpriority()  # avoid hogging shared CPU

#%% Settings

# localizer-trained decoder used throughout (L1 OvA with explicit neg_x)
C = 9.1  # chosen via CV in main analysis
clf = decoding.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=2, rng=0)

# enumerate subjects from data directory
files = utils.list_files(settings.data_dir, patterns=["*DSMR*"])
subjects = [f"DSMR{subj}" for subj in sorted(set(map(utils.get_id, files)))]

# preregistered inclusion thresholds
min_acc = 0.3
min_perf = 0.5
sfreq = 100
ms_per_point = 10

bands = settings.bands_HP
baseline = None

uid = hex(random.getrandbits(128))[2:10]  # run id for caches/logs
date = time.strftime("%Y-%m-%d")
times = np.arange(-100, 510, 10)
final_calculation = True  # controls RS segment selection in loaders

# optional probability normalization (used in RS sections below)
proba_norm = 'lambda x: x/x.mean(0)'
# proba_norm = 'lambda x:x'

palette = sns.color_palette()
c_fwd = palette[1]
c_bkw = palette[2]

#%% TDLM parameters (shared)
n_shuf = 1000
max_lag = 30
alpha_freq = 10
zscore_axes = -1

tf = tdlm.seq2tf(settings.seq_12)

# labels for alternative scaling schemes (used in some appendix sims)
names_perf_scale = ['best-case', 'linear']

#%% simulation parameters (shared)
mode = 'erp_diff_all'
lag_sim = 8  # 80 ms at 100 Hz
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]
tp = 31
best_C = 9.1

# insertion blocking window around events (see tdlm.utils.insert_events)
refractory = [lag_sim*2, lag_sim]

save_simulation_data_in_memory = False  # sims can exceed RAM
n_jobs = -1

if not final_calculation:
    print('not running final calculation yet, only use this for final paper calculation')

# color code subjects by memory performance for some plots
palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=101)
hues = {subj: palette[int(100*(get_performance(subj)-0.5)*2)] for subj in subjects}

# gaussian weighting across 5 samples around tp (matches manuscript sims)
gaussian_weighting = [0.05842299, 0.24210528, 1, 0.24210528, 0.05842299]

#%% preload localizer, negatives, and resting state
localizer = {}
rs1 = {}
rs2 = {}
neg_x = {}
seqs = {}

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

    rs1[subj] = load_RS1(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)
    rs2[subj] = load_RS2(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)

    seqs[subj] = utils.get_sequences(subj)

# subject-level QC + exclusions
max_accuracy = [utils.get_decoding_accuracy(subj, clf=clf, n_splits=20)[0] for subj in subjects]
test_performance = [utils.get_performance(subj, which='test') for subj in subjects]
df_subjects = pd.DataFrame({
    'subject': subjects,
    'accuracy': max_accuracy,
    'performance': test_performance
}).sort_values('accuracy')

excluded_acc = [subj for subj in subjects if utils.get_decoding_accuracy(subj, clf=clf, n_splits=20)[0] < 0.3]
excluded_perf = [subj for subj in subjects if utils.get_performance(subj, which='test') < 0.5]
excluded_miss = [subj for subj in subjects if utils.get_responses_localizer(subj)['n_misses'] > 60*0.25]

df_excluded = pd.DataFrame({
    'reason': ['low decoding']*len(excluded_acc)
            + ['low performance']*len(excluded_perf)
            + ['>25% misses']*len(excluded_miss)
}, index=excluded_acc + excluded_perf + excluded_miss)
df_excluded.sort_index(inplace=True)

subjects_incl = sorted(set(subjects).difference(set(df_excluded.index)))

# raise misc.Stop('Data loaded. Run cells below manually or remove this line.')

#%% SUPPL: fwd-bkw sequenceness
# compute fwd and bkw sequenceness in RS1/RS2 and summarize fwd-bkw difference

rs1_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs1_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sf = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
rs2_sb = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

for i, subj in enumerate(tqdm(subjects_incl, desc="subject")):
    # fit decoder on localizer peak timepoint
    train_x, train_y = localizer[subj]
    clf.reset_random_state(misc.make_seed(i, "rs"))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # decode RS probabilities
    proba_rs1 = clf.predict_proba(rs1[subj])
    proba_rs2 = clf.predict_proba(rs2[subj])

    # optional renormalization to control global probability drift
    proba_rs1 = eval(proba_norm)(proba_rs1)
    proba_rs2 = eval(proba_norm)(proba_rs2)

    # TDLM 1-step sequenceness curves + permutation nulls
    rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(
        proba_rs1, tf=tf, n_shuf=n_shuf, max_lag=max_lag,
        alpha_freq=alpha_freq, seed=misc.make_seed(subj, 'rs1')
    )
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(
        proba_rs2, tf=tf, n_shuf=n_shuf, max_lag=max_lag,
        alpha_freq=alpha_freq, seed=misc.make_seed(subj, 'rs2')
    )

    # z-score along chosen axis (typically within-subject across permutations)
    rs1_sf[i, :] = zscore_multiaxis(rs1_sf_subj, axes=zscore_axes)
    rs1_sb[i, :] = zscore_multiaxis(rs1_sb_subj, axes=zscore_axes)
    rs2_sf[i, :] = zscore_multiaxis(rs2_sf_subj, axes=zscore_axes)
    rs2_sb[i, :] = zscore_multiaxis(rs2_sb_subj, axes=zscore_axes)

#%% SUPPL: plot fwd-bkw and sign-flip null distributions
fig, axs = plt.subplots(2, 2, figsize=[10, 7])
c = [sns.color_palette()[0], sns.color_palette('dark')[0]]

for i, (sf, sb) in enumerate([[rs1_sf, rs1_sb], [rs2_sf, rs2_sb]]):
    cond = ["Control", "Post-Learn"][i]

    # top row: fwd-bkw curve with 95% permutation band
    tdlm.plotting.plot_sequenceness(sf, sb, which=['fwd-bkw'], plot95=True,
                                    color=c[i], ax=axs[0, i])
    axs[0, i].set_title(f'{cond} Sequenceness')

    # bottom row: sign-flip permutation distribution for the fwd-bkw curve
    p, t_obs, t_perms = tdlm.signflit_test((sf - sb)[:, 0, :], n_perms=10000)
    tdlm.plot_tval_distribution(t_obs, t_perms, ax=axs[1, i], color=c[i])
    axs[1, i].set_title(f'{cond} Signflip Perm. Distr')

fig.suptitle('Differential Forward-Backward Sequenceness within Session')
utils.savefig(fig, 'supplement/sequenceness-fwd-bkw.png')
utils.savefig(fig, 'supplement/sequenceness-fwd-bkw.svg')
utils.savefig(fig, 'supplement/sequenceness-fwd-bkw.eps')

#%% bootstrapped sequenceness powermaps (state-permutation thresholds)
# power defined as fraction of bootstraps where mean sequenceness exceeds threshold

np.random.seed(20251103)  # section seed (date tag)

densities = np.arange(0, 210, 10)
pkl_simdensity = settings.cache_dir + f'/simulation-full-linear.pkl.zip'
rs_sim_sf, rs_sim_sb, _ = joblib.load(pkl_simdensity)

n_repetitions = 1000
sample_sizes = range(10, 200, 5)

def bootstrap_perm(x, n):
    """Bootstrap mean sequenceness and permutation thresholds for sample size n."""
    print(f'[{time.ctime()}]{n=} started', flush=True, file=sys.stderr)

    N, D = x.shape[0], x.shape[1]
    rng = np.random.default_rng(n)

    # multinomial counts implement sampling with replacement without materializing indices
    counts = rng.multinomial(n, [1/N]*N, size=n_repetitions)

    # mean sequenceness at fixed lag (observed block [0, lag_sim])
    x_seq = x[:, :, 0, lag_sim]
    seq_means = (counts @ x_seq) / n

    # compute thresholds from permutation blocks (exclude observed/permutation index 0)
    thresh_max = np.empty((n_repetitions, D), dtype=x.dtype)
    thresh_95 = np.empty((n_repetitions, D), dtype=x.dtype)

    x_thr = x[:, :, 1:, 1:]  # (N, D, K-1, L-1)
    K_sub, L_sub = x_thr.shape[2], x_thr.shape[3]
    x_thr_flat = x_thr.reshape(N, -1)  # (N, D*K*L)

    for r in range(n_repetitions):
        mean_flat = counts[r] @ x_thr_flat
        mean_flat /= n
        mean_slice = mean_flat.reshape(D, K_sub, L_sub)

        threshholds = np.max(np.abs(mean_slice), axis=-1)  # collapse lag
        thresh_95[r] = np.quantile(threshholds, 0.95, axis=-1)
        thresh_max[r] = np.nanmax(threshholds, axis=-1)

    return seq_means, thresh_max, thresh_95

sf = rs_sim_sf
res1 = Parallel(-3)(delayed(bootstrap_perm)(sf, n) for n in tqdm(sample_sizes))

means, thmax, th95 = map(np.array, zip(*res1))

powermap_max = pd.DataFrame(np.mean(means > thmax, axis=1),
                            index=sample_sizes, columns=densities)
powermap_95 = pd.DataFrame(np.mean(means > th95, axis=1),
                           index=sample_sizes, columns=densities)

#%% bootstrapped power (sign-flip test)
# subtract mean curve to avoid inflation of false positives at large n

np.random.seed(20251103)

densities = np.arange(0, 210, 10)
pkl_simdensity = settings.cache_dir + f'/simulation-full-linear.pkl.zip'
rs_sim_sf, rs_sim_sb, _ = joblib.load(pkl_simdensity)

rs_sim_sf_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)

n_repetitions = 10000
sample_sizes = range(10, 200, 5)

def bootstrap_signflip(sf, n):
    """Bootstrap sign-flip p-values across densities for sample size n."""
    print(f'[{time.ctime()}] {n=} started', flush=True, file=sys.stderr)
    rng = np.random.default_rng(n)

    n_subj, n_d, max_lag = sf.shape

    # indices: (n_densities, n_repetitions, n)
    idxs = rng.integers(n_subj, size=(len(densities), n_repetitions, n))
    power = np.empty(len(densities))

    for d, density in enumerate(densities):
        sf_d = sf[:, d, :]
        sx = sf_d[idxs[d]]  # (R, n, max_lag)

        # signflit_test is not vectorized; loop over repetitions
        pvals = [tdlm.signflit_test(sx[r], rng=rng)[0] for r in range(n_repetitions)]
        power[d] = np.mean(np.array(pvals) < 0.05)

    return power

sf = rs_sim_sf[:, :, 0, 1:]  # observed curves only (drop lag=0 column)
sf_mean = sf[:, 0, :].mean(0)
sf = sf - sf_mean  # enforce null mean

res2 = Parallel(-3)(delayed(bootstrap_signflip)(sf, n) for n in tqdm(sample_sizes))

powermap_signflip = pd.DataFrame(
    res2,
    index=pd.Index(sample_sizes, name='Bootstrapped sample size'),
    columns=pd.Index(densities, name='Density min-1')
)
joblib.dump([powermap_95, powermap_max, powermap_signflip], settings.cache_dir + '/powermaps.pkl.zip')

#%% powermap plotting
powermap_95, powermap_max, powermap_signflip = joblib.load(settings.cache_dir + '/powermaps.pkl.zip')

fig, axs = plt.subplots(1, 3, figsize=[14, 5], sharex=True)
cmap = 'rocket'

# mne helper draws masked heatmaps; mask highlights power>=0.8 regions
ax = axs[0]
mne.viz.utils._plot_masked_image(
    ax, powermap_95, times=powermap_95.columns, yvals=powermap_95.index, cmap=cmap,
    mask=(powermap_95 >= 0.8).values, mask_cmap=cmap, mask_alpha=1
)
ax.set_title('State-Permutation: Perm 95%')
ax.set_xlabel('replay density min⁻¹')
ax.set_ylabel('bootstrap sample size')
ax.set_yticks(np.arange(20, 200, 20))

ax = axs[1]
mne.viz.utils._plot_masked_image(
    ax, powermap_max, times=powermap_max.columns, yvals=powermap_max.index, cmap=cmap,
    mask=(powermap_max >= 0.8).values, mask_cmap=cmap, mask_alpha=1
)
ax.set_title('State-Permutation: Perm Max')
ax.set_xlabel('replay density min⁻¹')
ax.set_yticks(np.arange(20, 200, 20))

fig.suptitle('TDLM Bootstrapped Powermaps')

ax = axs[2]
im = mne.viz.utils._plot_masked_image(
    ax, powermap_signflip, times=powermap_signflip.columns, yvals=powermap_signflip.index, cmap=cmap,
    mask=(powermap_signflip >= 0.8).values, mask_cmap=cmap, mask_alpha=1
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
fig.colorbar(im[0], cax=cax, label='power (% of bootstraps p<0.05)')

ax.set_title('Signflip-Permutation Test')
ax.set_xlabel('replay density min⁻¹')
ax.set_xticks([0, 25, 50, 75, 100, 150, 200])
ax.set_yticks(np.arange(20, 200, 20))

plotting.savefig(fig, 'supplement/bootstrapped-powermap.png')
plotting.savefig(fig, 'supplement/bootstrapped-powermap.svg')
plotting.savefig(fig, 'supplement/bootstrapped-powermap.eps')

#%% APPENDIX A: sim variants

#%% sim: shorter sequences + density scaling
# keep density scaling, but simulate only transitions that were actually remembered

np.random.seed(42)

densities = np.arange(0, 210, 10)
pool = Parallel(-1)

perf_scale = lambda perf: -perf + 1.5  # inverse linear map: worse perf -> more injected events
clf.set_params(C=9.1)

sf_sim_seq = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)
sb_sim_seq = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)

for s, subj in enumerate(tqdm(subjects_incl, desc="simulating different seq length")):
    train_x, train_y = localizer[subj]

    # build insertion patterns from ERP around tp, Gaussian-weighted in time
    insert_data_tp = train_x[:, :, tp-2:tp+3]
    insert_data_tp *= gaussian_weighting

    insert_class_pattern = []
    insert_labels = sorted(set(train_y))
    for label in insert_labels:
        erp_others = insert_data_tp[train_y != label, :, :].mean(0)
        insert_class_pattern += [insert_data_tp[train_y == label, :, :].mean(0) - erp_others]
    insert_data = np.stack(insert_class_pattern)

    rs_pre = rs1[subj]

    perf = get_performance(subj=subj, which='test')
    scale = perf_scale(perf)

    # convert density (events/min) to absolute counts for this RS duration
    n_events_all = [int(np.round(len(rs_pre)/sfreq/60 * dens * scale)) for dens in densities]

    # derive participant-specific transitions from correctly recalled sequences
    responses = utils.get_sequences(subj, 'test')
    responses = [x for x, y in zip(*responses) if y == 'correct']

    transitions = []
    for response in responses:
        transitions += [[response[0], response[1]]]
        transitions += [[response[1], response[-1]]]

    # simulate per-density, decode, then compute TDLM (kept in-memory only per loop)
    res = pool(
        delayed(tdlm.utils.insert_events)(
            data=rs_pre, insert_data=insert_data, insert_labels=insert_labels,
            lag=lag_sim, distribution='constant', n_steps=1, transitions=transitions,
            n_events=n_events, rng=misc.make_seed(s, n_events)
        ) for n_events in n_events_all
    )

    clf.reset_random_state(misc.make_seed(s))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    probas = pool(delayed(clf.predict_proba)(x) for x in res)
    res = pool(
        delayed(tdlm.compute_1step)(
            x, tf=tf, max_lag=max_lag, seed=misc.make_seed(s, k), n_shuf=n_shuf
        ) for k, x in enumerate(probas)
    )

    sfs, sbs = map(np.array, zip(*res))
    sf_sim_seq[s, :] = sfs
    sb_sim_seq[s, :] = sbs

#%% plotting: example curves + peak lag vs density
fig, axs = plt.subplots(2, 2, figsize=[12, 12])
axs = axs.flatten()

for i, density in enumerate([0, 40, 80, 120]):
    ax = axs[i]
    d = list(densities).index(density)

    sf_sim = zscore_multiaxis(sf_sim_seq[:, d, :, :], axes=zscore_axes)
    sb_sim = zscore_multiaxis(sb_sim_seq[:, d, :, :], axes=zscore_axes)

    ax.clear()
    tdlm.plot_sequenceness(sf_sim, sb_sim, which=['fwd', 'bkw'], ax=ax, rescale=True, plot95=True)
    ax.set_title(f'replay density {density} min⁻¹')

sns.despine(fig)
utils.savefig(fig, 'supplement/rev1-simulation-seqlim-sequenceness-examples.png')
utils.savefig(fig, 'supplement/rev1-simulation-seqlim-sequenceness-examples.svg')
utils.savefig(fig, 'supplement/rev1-simulation-seqlim-sequenceness-examples.eps')

rs_sim_sf_normed = zscore_multiaxis(sf_sim_seq, axes=zscore_axes)
fig, ax = plt.subplots(1, 1, figsize=[12, 6])

threshholds = np.max(abs(np.mean(rs_sim_sf_normed[:, :, 1:, 1:], 0)), -1)
thresholds_95 = np.quantile(threshholds, 0.95, axis=-1)
thresholds_max = np.nanmax(threshholds, axis=-1)

sequenceness = pd.DataFrame({
    'sequenceness': rs_sim_sf_normed[:, :, 0, lag_sim].ravel('F'),
    'density': np.repeat(densities, len(subjects_incl))
})
sns.lineplot(sequenceness, x='density', y='sequenceness', errorbar='se',
             label=f'mean sequencenes @ {lag_sim*10}ms', color=sns.color_palette()[1])

ax.plot(densities, thresholds_max, linestyle='-', color='gray', label='perm. max. threshold')
ax.plot(densities, thresholds_95, linestyle='--', color='gray', label='perm. 95% threshold')

seq_peak = sequenceness.groupby('density').mean()
t_max = np.argmax(seq_peak.sequenceness > thresholds_max)
t_95 = np.argmax(seq_peak.sequenceness > thresholds_95)

ax.vlines(seq_peak.index[t_max], 0, 1.5, linestyle='-', color='red', alpha=0.7)
ax.vlines(seq_peak.index[t_95], 0, 1.5, linestyle='--', color='red', alpha=0.7)

ax.set_xlabel('replay density (min⁻¹)')
ax.set_ylabel('forward sequenceness\n(u.u., zscored)')
ax.legend(loc='lower right')
sns.despine()
fig.suptitle(f'Forward {lag_sim*10} ms sequenceness at simulated replay densities')
utils.savefig(fig, f'supplement/rev1-simulation-seqlim--{lag_sim*10}ms-sequenceness.png')
utils.savefig(fig, f'supplement/rev1-simulation-seqlim--{lag_sim*10}ms-sequenceness.svg')
utils.savefig(fig, f'supplement/rev1-simulation-seqlim--{lag_sim*10}ms-sequenceness.eps')

#%% simulate burst vs uniform replay
from meg_utils import misc
from matplotlib.ticker import FuncFormatter

np.random.seed(20250115)

# --- configuration matching manuscript style ---
rate_events_per_sec = 80        # insert 80 events per second
n_steps = 1
tp = 31                         # peak decoding timepoint already used above
sfreq = 100
zscore_axes = -1                # keep normalization consistent with rest of manuscript
alpha_freq = 10
max_lag = 30

# Gaussian weighting across 5 samples around tp (same as manuscript sim)
gaussian_weighting = np.array([0.05842299, 0.24210528, 1.0, 0.24210528, 0.05842299])

# Convert sequence (A->B->... mapping) to ints as before
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]
tf = tdlm.seq2tf(settings.seq_12)

# Containers: forward/backward sequenceness for both regimens
sf_uniform = np.full([len(subjects_incl), max_lag + 1], np.nan)
sb_uniform = np.full_like(sf_uniform, np.nan)
sf_bursty  = np.full_like(sf_uniform, np.nan)
sb_bursty  = np.full_like(sf_uniform, np.nan)

# Optional: store onset tables for quick checks (not required for plots)
onsets_uniform = {}
onsets_bursty = {}

def create_alternating_dist(length, seglen):
    """Create a normalized probability vector with alternating zero/one blocks.
    """
    pattern = np.r_[np.zeros(seglen, dtype=float), np.ones(seglen, dtype=float)]
    reps = (length + pattern.size - 1) // pattern.size
    p = np.tile(pattern, reps)[:length]

    s = p.sum()
    if s == 0:
        return np.full(length, 1.0 / length, dtype=float)
    return p / s

# Build insertion patterns per subject (same logic as main sim section)
for i, subj in enumerate(tqdm(subjects_incl, desc="Sim burst vs uniform")):
    train_x, train_y = localizer[subj]

    # pattern around tp with Gaussian temporal weighting
    insert_data_tp = train_x[:, :, tp - 2 : tp + 3].copy()
    insert_data_tp *= gaussian_weighting

    # class-specific differential pattern as in `erp_diff_others`
    insert_class_pattern = []
    insert_labels = sorted(set(train_y))
    for lbl in insert_labels:
        erp_others = insert_data_tp[train_y != lbl].mean(0)
        insert_class_pattern += [insert_data_tp[train_y == lbl].mean(0) - erp_others]
    insert_data = np.stack(insert_class_pattern)

    # base data: RS1 per subject
    rs_base = rs1[subj]

    n_events = int(len(rs_base) / 100  * rate_events_per_sec/60)
    # two distributions
    p_burst = create_alternating_dist(len(rs_base), 3000)

    # Uniform regimen
    rs_uni, df_uni = tdlm.utils.insert_events(
        data=rs_base,
        insert_data=insert_data,
        insert_labels=insert_labels,
        sequence=sequence,
        n_events=n_events,
        n_steps=n_steps,
        lag=lag_sim,
        distribution="constant",
        return_onsets=True,
    )

    # Bursty regimen
    rs_bur, df_bur = tdlm.utils.insert_events(
        data=rs_base,
        insert_data=insert_data,
        insert_labels=insert_labels,
        sequence=sequence,
        n_events=n_events,
        n_steps=n_steps,
        lag=lag_sim,
        distribution=p_burst,
        return_onsets=True,
    )

    onsets_uniform[subj] = df_uni
    onsets_bursty[subj] = df_bur

    # Decode probabilities
    clf.reset_random_state(misc.make_seed(subj))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    proba_uni = clf.predict_proba(rs_uni)
    proba_bur = clf.predict_proba(rs_bur)

    proba_uni = eval(proba_norm)(proba_uni)
    proba_bur = eval(proba_norm)(proba_bur)

    # TDLM 1-step sequenceness
    seed = utils.get_id(subj) * 101
    sf_u, sb_u = tdlm.compute_1step(proba_uni, tf=tf, n_shuf=0, max_lag=max_lag, alpha_freq=alpha_freq, seed=seed)
    sf_b, sb_b = tdlm.compute_1step(proba_bur, tf=tf, n_shuf=0, max_lag=max_lag, alpha_freq=alpha_freq, seed=seed)

    # z-score along permutations consistent with manuscript
    sf_uniform[i, :] = zscore_multiaxis(sf_u, axes=zscore_axes)
    sb_uniform[i, :] = zscore_multiaxis(sb_u, axes=zscore_axes)
    sf_bursty[i, :]  = zscore_multiaxis(sf_b, axes=zscore_axes)
    sb_bursty[i, :]  = zscore_multiaxis(sb_b, axes=zscore_axes)

# --- Group plots (forward/backward curves) ---
fig, axs = plt.subplots(2, 2, figsize=[16, 10])

for i, sxx in enumerate([[sf_uniform, sb_uniform], [sf_bursty, sb_bursty]]):
    name = ['uniform', 'bursty'][i]
    colors = [[sns.color_palette("bright")[1], sns.color_palette('dark')[1]],
              [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]]
    for d, sx in enumerate(sxx):
        direction = ['forward', 'backward'][d]
        df_seq = misc.to_long_df(sx, ['subject', 'timelag'], value_name='sequenceness',
                                 timelag={'time lag': np.arange(0, 310, 10)})
        sns.lineplot(df_seq, ax=axs[0, d], x='time lag', y='sequenceness', color=colors[d][i],
                     label=f'{name}')
        axs[0, d].set(title=f'{direction}')


ax = axs[1, 0]
ax.clear()
ax.plot(p_burst, label='bursty')
ax.plot(np.ones_like(p_burst)/len(p_burst), label='uniform')
ax.set_xticks(np.arange(0, 25000, 100*30))
formatter = FuncFormatter(lambda x, pos: f"{int(x//6000)}:{int((x//100)%60):02d}")
ax.xaxis.set_major_formatter(formatter)
ax.set(xlabel="Time (min:sec)", ylabel='sequence start probability',
       title='Replay onset probability distribution ')
ax.legend()

ax = axs[1, 1]
ax.clear()

positions = []
for subj, df_subj in onsets_bursty.items():
    df_sel = df_subj[df_subj.step==0]
    positions += sorted(df_sel.pos)
ax.hist(positions, bins=500, label='bursty', alpha=0.5)

positions = []
for subj, df_subj in onsets_uniform.items():
    df_sel = df_subj[df_subj.step==0]
    positions += sorted(df_sel.pos)

ax.hist(positions, bins=500, label='uniform', alpha=0.5)
ax.legend()


formatter = FuncFormatter(lambda x, pos: f"{int(x//6000)}:{int((x//100)%60):02d}")
ax.xaxis.set_major_formatter(formatter)
ax.set(xlabel="Time (min:sec)", ylabel='#n counts of starts across participants',)
ax.set_title('Empirical sequence starts distribution')
sns.despine(fig)

utils.normalize_lims(axs[0, :])
fig.suptitle('TDLM with simulated bursty replay at 80 min⁻¹', fontsize=22)
utils.savefig(fig, 'supplement/bursty-vs-uniform-replay.png')
utils.savefig(fig, 'supplement/bursty-vs-uniform-replay.svg')
utils.savefig(fig, 'supplement/bursty-vs-uniform-replay.eps')

#%% simulate shortened vs full transitions
# simulate whether simulating ABCDE or ABABA makes a difference,
# i.e., when keeping number of transitions fixed it makes a difference
# whether we simulate the same transitions or the full sequence

np.random.seed(42)  # failsafe

# here we define replay densities that we want to simulate
# replay density is defined in events/minute or events^-1min
density = 80

pool = Parallel(-1)  # pre-initialize pool


sf_sim_orig = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
sf_sim_short = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
sb_sim_orig = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
sb_sim_short = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)


seq_lens = []

for s, subj in enumerate(tqdm(subjects_incl, desc="simulating full sequence vs actual memory")):

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

    # scale number of events by performance
    perf = get_performance(subj=subj, which='test')

    # density to simulate
    dens = 80
    n_events = int(np.round(len(rs_pre)/sfreq/60 * dens))


    # simulate A->B->C->D
    rs_sim1, df_onsets1 = tdlm.utils.insert_events(data = rs_pre,
                              insert_data = insert_data,
                              insert_labels = insert_labels,
                              lag=lag_sim,
                              distribution='constant',
                              n_steps=1,
                              sequence = sequence,
                              n_events = n_events,
                              return_onsets=True,
                              rng=misc.make_seed(subj, n_events))

    # simulate actually learned sequence eg A->B->A->B
    # next, get which sequences were answered correctly
    responses = utils.get_sequences(subj, 'test')

    # filter for correct responses
    responses = [x for x, y in zip(*responses) if y=='correct']

    # these are the tuples that the participan knew
    transitions = []
    for response in responses:
        transitions += [[response[0], response[1]]]
        transitions += [[response[1], response[-1]]]

    rs_sim2, df_onsets2 = tdlm.utils.insert_events(data = rs_pre,
                              insert_data = insert_data,
                              insert_labels = insert_labels,
                              lag=lag_sim,
                              distribution='constant',
                              n_steps=1,
                              transitions = transitions,
                              n_events = n_events,
                              return_onsets=True,
                              rng=misc.make_seed(subj, str(transitions)))

    clf.reset_random_state(misc.make_seed(subj))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    probas_orig = clf.predict_proba(rs_sim1)
    probas_short = clf.predict_proba(rs_sim2)

    sf1, sb1 = tdlm.compute_1step(probas_orig, tf=tf, max_lag=max_lag, n_shuf=n_shuf,
                                seed=misc.make_seed(subj, '1step'))
    sf2, sb2 = tdlm.compute_1step(probas_short, tf=tf, max_lag=max_lag, n_shuf=n_shuf,
                                seed=misc.make_seed(subj, '2step'))

    sf_sim_orig[s, :] = sf1
    sf_sim_short[s, :] = sf2
    sb_sim_orig[s, :] = sb1
    sb_sim_short[s, :] = sb2



#### plotting

fig, axs = plt.subplots(1, 2, figsize=[16, 5])

for i, sxx in enumerate([[sf_sim_orig, sb_sim_orig], [sf_sim_short, sb_sim_short]]):
    name = ['full sequence', 'memorized transitions'][i]
    colors = [[sns.color_palette("bright")[1], sns.color_palette('dark')[1]],
              [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]]
    for d, sx in enumerate(sxx):
        sx = zscore_multiaxis(sx, axes=zscore_axes)
        direction = ['forward', 'backward'][d]
        df_seq = misc.to_long_df(sx[:, 0, :], ['subject', 'timelag'], value_name='sequenceness',
                                 timelag={'time lag': np.arange(0, 310, 10)})
        sns.lineplot(df_seq, ax=axs[d], x='time lag', y='sequenceness', color=colors[d][i],
                     label=f'{name}')
        axs[d].set(title=f'{direction}')

sns.despine()
fig.suptitle('Simulating all graph transitions vs. transitions that were actually remembered\n(density 80 min⁻¹ @ 80ms time lag)')
utils.savefig(fig, f'supplement/rev1-simulation-seq-full-sequence-vs-subset.png')
utils.savefig(fig, f'supplement/rev1-simulation-seq-full-sequence-vs-subset.svg')
utils.savefig(fig, f'supplement/rev1-simulation-seq-full-sequence-vs-subset.eps')

#%% simulate ABC vs AB BC
# does it make a difference whether ABC are in a row or if they are separate
# i.e. we simulate ABCD and AB BC CD, but same number of transitions

np.random.seed(42)  # failsafe

# here we define replay densities that we want to simulate
# replay density is defined in events/minute or events^-1min
densities = np.arange(0, 210, 10)

pool = Parallel(-1)  # pre-initialize pool

perf_scale = lambda perf: -perf+1.5  # inverse linear relationship

clf.set_params(C=9.1)

sf_sim_1step = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
sf_sim_2step = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

sb_sim_1step = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)
sb_sim_2step = np.full([len(subjects_incl), n_shuf, max_lag+1], np.nan)

seq_lens = []

for s, subj in enumerate(tqdm(subjects_incl, desc="simulating ABC vs AB BC")):

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

    # scale number of events by performance
    perf = get_performance(subj=subj, which='test')
    scale = perf_scale(perf)

    # density to simulate
    dens = 80
    n_events_base = int(np.round(len(rs_pre)/sfreq/60 * dens * scale))
    n_events_1step = n_events_base
    n_events_2step = n_events_base//2

    # simulate A->B and C->D separately
    rs_sim1, df_onsets1 = tdlm.utils.insert_events(data = rs_pre,
                              insert_data = insert_data,
                              insert_labels = insert_labels,
                              lag=lag_sim,
                              distribution='constant',
                              n_steps=1,
                              sequence = sequence,
                              n_events = n_events_1step,
                              return_onsets=True,
                              rng=misc.make_seed(subj, n_events_1step, 1))

    # simulate A->B->C
    rs_sim2, df_onsets2 = tdlm.utils.insert_events(data = rs_pre,
                              insert_data = insert_data,
                              insert_labels = insert_labels,
                              lag=lag_sim,
                              distribution='constant',
                              n_steps=2,
                              sequence = sequence,
                              n_events = n_events_2step,
                              return_onsets=True,
                              rng=misc.make_seed(subj, n_events_2step, 2))
    clf.reset_random_state(misc.make_seed(subj))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    probas_1step = clf.predict_proba(rs_sim1)
    probas_2step = clf.predict_proba(rs_sim2)

    sf1, sb1 = tdlm.compute_1step(probas_1step, tf=tf, max_lag=max_lag, n_shuf=n_shuf,
                                seed=misc.make_seed(subj, '1step'))
    sf2, sb2 = tdlm.compute_1step(probas_2step, tf=tf, max_lag=max_lag, n_shuf=n_shuf,
                                seed=misc.make_seed(subj, '2step'))

    sf_sim_1step[s, :] = sf1
    sf_sim_2step[s, :] = sf2
    sb_sim_1step[s, :] = sb1
    sb_sim_2step[s, :] = sb2



#### plotting

fig, axs = plt.subplots(1, 2, figsize=[16, 5])

for i, sxx in enumerate([[sf_sim_1step, sb_sim_1step], [sf_sim_2step, sb_sim_2step]]):
    name = ['A→B [..] B→C', 'A→B→C'][i]
    colors = [[sns.color_palette("bright")[1], sns.color_palette('dark')[1]],
              [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]]
    for d, sx in enumerate(sxx):
        sx = zscore_multiaxis(sx, axes=zscore_axes)
        direction = ['forward', 'backward'][d]
        df_seq = misc.to_long_df(sx[:, 0, :], ['subject', 'timelag'], value_name='sequenceness',
                                 timelag={'time lag': np.arange(0, 310, 10)})
        sns.lineplot(df_seq, ax=axs[d], x='time lag', y='sequenceness', color=colors[d][i],
                     label=f'{name}')
        axs[d].set(title=f'{direction}')

sns.despine()
fig.suptitle('Doublet vs triplet replay with fixed number of 1-step transitions')
utils.savefig(fig, f'supplement/rev1-simulation-seq-ABC-vs-AB-BC.png')
utils.savefig(fig, f'supplement/rev1-simulation-seq-ABC-vs-AB-BC.svg')
utils.savefig(fig, f'supplement/rev1-simulation-seq-ABC-vs-AB-BC.eps')

#%% simulate longer replay sequences
#### calculation
np.random.seed(42)

# here we define replay densities that we want to simulate
# replay density is defined in events/minute or events^-1min
pool = Parallel(-1)

perf_scale = lambda perf: -perf+1.5  # inverse linear relationship

rs_sims = []

n_steps = [1, 2, 3, 4]
density_fractions = np.arange(0, 101, 1)/100

clf.set_params(C=9.1)

sf_sim_steps = np.full([len(subjects_incl), len(n_steps), len(density_fractions), n_shuf, max_lag+1], np.nan)


for s, subj in enumerate(tqdm(subjects_incl, desc="inserting patterns")):

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

    perf = get_performance(subj=subj, which='test')
    scale = perf_scale(perf)

    clf.reset_random_state(misc.make_seed(subj))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    max_densities = []
    for n, n_step in enumerate(n_steps):

        # calculate refractory period, or maximum density
        block_before = lag_sim*(n_step+1)  # else we have overlap
        block_after = lag_sim*(n_step)

        # before, event_len + lag_sim
        # afterwards, 150ms blocking
        block_len = block_before  + block_after + lag_sim
        max_density = 60 * 100 // block_len
        max_densities += [max_density]

        # get probability estimates for item reactivation from the resting state
        # scale the number of replay events by the performance
        n_events_all = [int(np.round(len(rs_pre)/sfreq/60 * max_density * perf * c)) for c in density_fractions]

        # we cannot save all this in memory, would take >50GB RAM
        # so simply do all calculations in this loop and then discard the sims
        rs_sim_densities = pool(delayed(tdlm.utils.insert_events)(data = rs_pre,
                                  insert_data = insert_data,
                                  insert_labels = insert_labels,
                                  lag=lag_sim,
                                  distribution='constant',
                                  refractory=[block_before, block_after],
                                  n_steps=n_step,
                                  sequence = sequence,
                                  n_events = n_events,
                                  rng=misc.make_seed(n_events, n, s)) for n_events in tqdm(n_events_all))

        probas = pool(delayed(clf.predict_proba)(x) for x in rs_sim_densities)

        res = pool(delayed(tdlm.compute_1step)(x,
                                               tf=tf,
                                               max_lag=max_lag,
                                               alpha_freq=10,
                                               seed=misc.make_seed(subj, n_step),
                                               n_shuf=n_shuf) for k, x in enumerate(probas))
        sfs, sbs = map(np.array, zip(*res))
        sf_sim_steps[s, n, ...] = sfs

joblib.dump(sf_sim_steps, settings.cache_dir + '/appendix_multistep.pkl.gz')

#%%## plotting

max_densities = []
for n, n_step in enumerate(n_steps):

    # calculate refractory period, or maximum density
    block_before = lag_sim*(n_step+1)  # else we have overlap
    block_after = lag_sim*(n_step)

    # before, event_len + lag_sim
    # afterwards, 150ms blocking
    block_len = block_before  + block_after + lag_sim
    max_density = 60 * 100 // block_len
    max_densities += [max_density]

sf_sim_steps = joblib.load(settings.cache_dir + '/appendix_multistep.pkl.gz')

sf_sim_steps = zscore_multiaxis(sf_sim_steps, axes=zscore_axes)

df = misc.to_long_df(sf_sim_steps[:, :, :, 0, 8],
                     columns=['subject', 'n_steps', 'density_fraction'],
                     n_steps=n_steps, density_fraction=density_fractions,
                     value_name='sequenceness')

# next add significance thresholds
threshholds = np.max(abs(np.mean(sf_sim_steps[:, :, :, 1:, 1:], 0)), -1)
thr95 = np.quantile(threshholds, 0.95, axis=-1)
thrmax = np.nanmax(threshholds, axis=-1)

df['type'] = 'sequenceness'

df_tmp1 = pd.DataFrame(misc.to_long_df(thrmax, columns=['n_steps', 'density_fraction'],
                                      n_steps=n_steps, density_fraction=density_fractions,
                                      value_name='sequenceness'), )
df_tmp1['type'] = 'perm_max'

df_tmp2 = pd.DataFrame(misc.to_long_df(thr95, columns=['n_steps', 'density_fraction'],
                                      n_steps=n_steps, density_fraction=density_fractions,
                                      value_name='sequenceness'))
df_tmp2['type'] = 'perm_95'

df = pd.concat([df, df_tmp1, df_tmp2], ignore_index=True)

# df = df[df.density_fraction <= 1]

fig, axs = plt.subplots(2, 2, figsize=[12, 8])
palette = sns.color_palette('mako', 6)[:4:][::-1]
units = ['min-1', 'transitions']

for i, step in enumerate(n_steps):
    ax = axs.flat[i]
    ax.clear()
    colors = [palette[i], 'k', 'k']
    df_sel = df[df.n_steps==step]
    df_sel['density'] = df_sel.density_fraction.apply(lambda x: x*max_densities[i])

    sns.lineplot(df_sel, x='density', y='sequenceness',
             style='type', hue='type', err_kws={'alpha': 0.1}, palette=colors,
             ax=ax)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xlabel('Density')
    ax.set_title(f'{step}-step transitions')

    if i==1:
        ax.set_xticks([0, 25, 50, 75, 100, 135])

    labels = [item.get_text() for item in ax.get_xticklabels()]
    lastidx = -2 if i!=1 else -1
    max_dens = int(labels[lastidx])
    labels = [f'{l}\n{int(int(l.replace("−", "-"))*step)}' for l in labels]
    last_label = labels[lastidx]
    line1, line2 = last_label.split('\n')
    labels[lastidx] = f'        {line1} min⁻¹\n             {line2} transitions\n      (~maximum)'
    ax.set_xticklabels(labels)


    thresholds_max = df_sel[df_sel.type=='perm_max'].groupby('density_fraction').mean(True).reset_index()
    thresholds_95 = df_sel[df_sel.type=='perm_95'].groupby('density_fraction').mean(True).reset_index()

    seq_peak = df_sel.groupby('density_fraction').mean(True)
    t_max = np.argmax(seq_peak.sequenceness.values>thresholds_max.sequenceness.values)
    t_95 = np.argmax(seq_peak.sequenceness.values>thresholds_95.sequenceness.values)

    dens_max = int(max_dens * seq_peak.index[t_max])
    dens_95 = int(max_dens * seq_peak.index[t_95])

    x_max = np.ceil(seq_peak.index[t_max] * max_dens)
    x_95 = np.ceil(seq_peak.index[t_95] * max_dens)

    ax.vlines(x_max, 0.2, 1.5,  linestyle='-', color='red', alpha=0.7)
    ax.vlines(x_95, 0.2, 1.5,  linestyle='--', color='red', alpha=0.7)

    ax.text(x_max+1, 0.2, f'~{dens_max} min⁻¹', c='red', ha='left', va='bottom')
    ax.text(x_max+1, 0.2, f'~{dens_max*step} transitions', c='red', ha='left', va='top')

    ax.text(x_95+15, -.2, f'~{dens_95} min⁻¹', c='red', ha='right', va='bottom')
    ax.text(x_95+15, -.2, f'~{dens_95*step} transitions', c='red', ha='right', va='top')
    ax.set_ylim([-0.5, 2.4])

    if i==1:
        ax.set_xticks([0, 25, 50, 75, 100, 135])

fig.tight_layout()
fig.suptitle('Simulating longer replay events')
plotting.savefig(fig, 'supplement/multistep-simulation.png')
plotting.savefig(fig, 'supplement/multistep-simulation.svg')
plotting.savefig(fig, 'supplement/multistep-simulation.eps')
