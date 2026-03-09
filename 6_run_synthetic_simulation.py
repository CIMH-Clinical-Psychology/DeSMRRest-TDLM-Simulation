```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 11:36:03 2025

Synthetic TDLM simulation (matches MATLAB demo): generate resting-state-like M/EEG,
insert replay events with controlled density, decode with a classifier, then estimate
forward/backward sequenceness across lags.

@author: simon
"""
import matplotlib.pyplot as plt
import numpy as np
import tdlm
from tqdm import tqdm
import utils
import settings
import seaborn as sns
import pandas as pd
from meg_utils import decoding, misc
from tdlm.utils import simulate_classifier_patterns

# reduce OS priority for long simulations
misc.low_priority()
np.random.seed(42)

# matplotlib defaults used by downstream plotting
plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

#%% settings

n_subj = 30
# 10-way OvA logistic regression with explicit negative class sampling
clf = decoding.LogisticRegressionOvaNegX(neg_x_ratio=1.0)

# empirical-ish decoding performances (clipped at chance=0.5)
performances = [1.0, 0.67, 0.75, 0.75, 0.83, 0.92, 0.83, 0.5, 0.83, 0.83, 0.92,
                0.92, 0.75, 1.0, 0.75, 0.83, 0.75, 0.5, 1.0, 0.83, 1.0, 0.67,
                0.92, 1.0, 0.58, 1.0, 1.0, 0.92, 0.5, 0.75]

# transition matrix and integer-coded sequence (drop terminal char)
tf = tdlm.seq2tf(settings.seq_12)
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]

sfreq = 100
lag_sim = 8  # event lag in samples (80 ms at 100 Hz)
refractory = [lag_sim*2, lag_sim]  # prevent too-close insertions (sequence / state)

# optional post-hoc normalization of probabilities (currently unused)
proba_norm = 'lambda x: x/x.mean(0)'

# static colors for plotting
palette = sns.color_palette()
c_fwd = palette[1]
c_bkw = palette[2]

#%% synthetic sim: calculation

# replay density grid in events/min
densities = np.arange(0, 210, 1)

# store TDLM outputs per density (filled across subjects)
sf_sim = [[] for _ in densities]
sb_sim = [[] for _ in densities]

# map subject decoding performance to an event-count scaling factor (worse perf -> more inserted events)
perf_scale = lambda perf: -perf + 1 + min(performances)  # inverse linear relationship

for subj in tqdm(range(n_subj)):
    # simulate 60 s resting state with 306 sensors
    rs = tdlm.utils.simulate_meeg(60, sfreq, 306, rng=misc.make_seed(subj))

    # simulate training trials + class patterns (peak timepoint patterns)
    train_x, train_y, patterns = simulate_classifier_patterns(
        n_patterns=10,
        n_channels=306,
        noise=4,
        n_train_per_stim=18,
        rng=misc.make_seed(subj)
    )

    # separate explicit null class from stimulus classes
    null_x = train_x[train_y == 0]
    train_x = train_x[train_y > 0]
    train_y = train_y[train_y > 0] - 1  # relabel stimuli to 0..9

    # fit subject-specific classifier
    clf.reset_random_state(misc.make_seed(subj, 'synthetic'))
    clf.fit(train_x, train_y, neg_x=null_x)

    # subject-specific scaling of inserted replay counts
    perf = performances[subj % len(performances)]
    scale = perf_scale(perf)

    # convert densities (events/min) to absolute counts for 60 s data
    n_events_all = [int(np.round(len(rs)/sfreq/60 * dens * scale)) for dens in densities]

    for d, n_events in enumerate(n_events_all):
        insert_data = patterns          # sensor topographies to inject
        insert_labels = np.arange(10)   # corresponding state labels 0..9

        # insert sequential replay events into resting state
        rs_sim = tdlm.utils.insert_events(
            rs.copy()[:],
            insert_data,
            insert_labels,
            lag=lag_sim,
            refractory=refractory,
            sequence=sequence,
            n_events=n_events,
            rng=subj
        )

        # decode to class probabilities (time x state)
        probas = clf.predict_proba(rs_sim)
        # probas = eval(proba_norm)(probas)

        # add temporal autocorrelation (AR-like) to mimic Liu et al. simulation structure
        coreP = np.full([len(rs_sim)], np.nan)
        coreP[0] = np.random.randn()

        for iT in range(1, len(rs_sim)):
            coreP[iT] = 0.95 * (coreP[iT-1] + np.random.randn())  # shared drift term

        probas = (probas.T + 0.05 * coreP).T

        # compute TDLM 1-step sequenceness across lags + permutation null
        sf, sb = tdlm.compute_1step(
            probas, tf, max_lag=30, n_shuf=100, seed=misc.make_seed(subj)
        )

        sf_sim[d] += [sf]
        sb_sim[d] += [sb]

# convert list-of-lists to arrays: subj x density x (metrics...) after axis swap
sf_sim = np.array(sf_sim)
sb_sim = np.array(sb_sim)

# align dims with older code (subjects first)
sf_sim = np.swapaxes(sf_sim, 0, 1)
sb_sim = np.swapaxes(sb_sim, 0, 1)

#%% synthetic sim: plotting
#### exemplar sequenceness curves at selected densities

fig, axs = plt.subplots(2, 2, figsize=[16, 10])
axs = axs.flatten()

for i, density in enumerate([0, 40, 80, 120]):
    ax = axs[i]
    d = list(densities).index(density)

    # slice: subjects x shuffles x lags (depending on TDLM output shape)
    sf_sim_d = sf_sim[:, d, :, :]
    sb_sim_d = sb_sim[:, d, :, :]

    ax.clear()
    # rescale=True normalizes within-panel for comparability
    tdlm.plot_sequenceness(sf_sim_d, sb_sim_d, which=['fwd', 'bkw'], ax=ax, rescale=True)
    ax.set_title(f'Replay Density {density} min⁻¹')

ax.set_ylim([-1.2, 4.4])
utils.normalize_lims(axs, which='y')
sns.despine(fig)
utils.savefig(fig, 'supplement/synthetic-simulation-sequenceness-examples.png')

#### forward sequenceness at the simulated lag (80 ms) across densities + null thresholds

fig, axs = plt.subplot_mosaic('AAABB', figsize=[16, 5])
ax = axs['A']

# permutation-derived thresholds using mean over subjects (exclude [0] index blocks used for observed stats)
threshholds = np.max(abs(np.mean(sf_sim[:, :, 1:, 1:], 0)), -1)
thresholds_95 = np.quantile(threshholds, 0.95, axis=-1)
thresholds_max = np.nanmax(threshholds, axis=-1)

# observed forward sequenceness at lag_sim
sequenceness = pd.DataFrame({
    'sequenceness': sf_sim[:, :, 0, lag_sim].ravel('F'),
    'density': np.repeat(densities, n_subj)
})

sns.lineplot(
    sequenceness, x='density', y='sequenceness', errorbar='se', ax=ax,
    label=f'mean sequencenes @ {lag_sim*10}ms', color=sns.color_palette()[1]
)

# show max and 95% permutation thresholds across densities
ax.plot(densities, thresholds_max, linestyle='-', color='gray', label='perm. max. threshold')
ax.plot(densities, thresholds_95, linestyle='--', color='gray', label='perm. 95% threshold')

# first density where mean exceeds thresholds
seq_peak = sequenceness.groupby('density').mean()
t_max = np.argmax(seq_peak.sequenceness > thresholds_max)
t_95 = np.argmax(seq_peak.sequenceness > thresholds_95)

x_max = seq_peak.index[t_max]
x_95 = seq_peak.index[t_95]

ax.vlines(x_max, 0, 0.05, linestyle='-', color='red', alpha=0.7)
ax.vlines(x_95, 0, 0.05, linestyle='--', color='red', alpha=0.7)

ax.text(x_max+5, 0.05, f'{x_max} min⁻¹', c='red', ha='center', va='bottom')
ax.text(x_95+2, 0, f'{x_95} min⁻¹', c='red', ha='center', va='top')

ax.set_xlabel('replay density (min⁻¹)')
ax.set_ylabel('forward sequenceness')
ax.legend(loc='lower right')
sns.despine()
fig.suptitle('Synthetic simulation of replay as in Liu et al. 2021')
ax.set_title(f'Synthetic simulation: Forward {lag_sim*10} ms sequenceness across densities')

#%% sign-flip permutation test summary across densities

ax = axs['B']
ax.clear()

# compute density-wise sign-flip p-values and observed t-stats for the forward curve
df = pd.DataFrame()
for d, density in enumerate(densities):
    p, t_obs, t_perms = tdlm.signflit_test(sf_sim[:, d, 0, :], rng=d)
    df_tmp = pd.DataFrame({'p-value': p, 't-value': t_obs, 'density': density}, index=[0])
    df = pd.concat([df, df_tmp], ignore_index=True)

sns.lineplot(df, x='density', y='t-value', label='sign-flip t-value', ax=ax)

# first density crossing significance thresholds
idx_5 = np.argmax(df['p-value'] < 0.05)
idx_01 = np.argmax(df['p-value'] < 0.001)

t_5 = df['t-value'][idx_5]
t_01 = df['t-value'][idx_01]
dens_5 = df.density[idx_5]
dens_01 = df.density[idx_01]

ax.axhline(t_5, label='p<0.05', c='red', linestyle='--')
ax.axhline(t_01, label='p<0.001', c='red', linestyle='-.', linewidth=1)
ax.set_xticks(np.arange(0, 220, 20))

ax.vlines(t_5, 0, 10, linestyle='--', color='red', alpha=0.7)
ax.vlines(t_01, 0, 10, linestyle='-', color='red', alpha=0.7)

ax.text(t_5+5, 10, f'{dens_5} min⁻¹', c='red', ha='center', va='bottom')
ax.text(t_01+15, -0.2, f'{dens_01} min⁻¹', c='red', ha='center', va='top')

ax.legend()
ax.set_title('Sign-flip Permutation t-values')

utils.savefig(fig, f'figure/simulation-synthetic-{lag_sim*10}ms-sequenceness.png')
```
