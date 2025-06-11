# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:08:58 2024

@author: simon.kern
"""
import os
import settings
import tdlm
import utils
from tqdm import tqdm
import time
import random
import compress_pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import mat73
from meg_utils import plotting

from mne.stats import permutation_cluster_1samp_test
import joblib
from joblib import Parallel, delayed
from settings import results_dir, cache_dir
from load_funcs import load_localizers_seq12, load_neg_x_before_audio_onset
from load_funcs import load_RS1, load_RS2
from utils import get_best_timepoint, get_performance, load_pkl_pandas
from utils import plot_correlation, zscore_multiaxis
from scipy.stats import ttest_rel
from scipy import stats
from statsmodels.stats.multitest import multipletests

utils.lowpriority()  # make sure not to clog CPU on multi-user systems
#%% Settings

# this is the classifier that will be used
C = 9.1  # determined previously via cross-validation, see below
clf = utils.LogisticRegressionOvaNegX(C=C, penalty="l1", neg_x_ratio=2)

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
mode = 'erp_diff_all'
lag_sim = 8  # simulate replay at 80 milliseconds
sequence = tdlm.utils.char2num(settings.seq_12)[:-1]
tp = 31


if not final_calculation:
    print('not running final calculation yet, only use this for final paper calculation')

# best_tp will be calculated and replaced later, for debugging it is sometimes
# easier to define it here, so you can run segments without computing everything
best_tp = utils.load_pkl(f"{settings.cache_dir}/best_tp.pkl.zip", 31)

plt.rc("font", size=14)
meanprops = {
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "white",
    "markersize": "10",
}

palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=101)
hues = {subj:palette[int(100*(get_performance(subj)-0.5)*2)] for subj in subjects}

#%% preload some data (e.g. localizer)
localizer = {}  # localizer training data
rs1 = {}
rs2 = {}
neg_x = {}  # pre-audio fixation cross neg_x of localizer
seqs = {}  # load sequences of that participant in this dict

for subj in tqdm(subjects[::-1], desc="Loading data"):
    # data used for the localizer
    localizer[subj] = load_localizers_seq12(subj=subj, sfreq=sfreq, bands=bands, autoreject=settings.default_autoreject, ica=settings.default_ica_components)
    # negative examples from the fixation cross before audio cue onset
    neg_x[subj] = load_neg_x_before_audio_onset(subj=subj, sfreq=sfreq, bands=bands,  autoreject=settings.default_autoreject, ica=settings.default_ica_components)
    # resting state data, both eyes open and eyes closed together
    rs1[subj] = load_RS1(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)
    rs2[subj] = load_RS2(subj=subj, sfreq=sfreq, bands=bands, final_calculation=final_calculation)

    # individual sequences for the trials (maybe not necessary)
    seqs[subj] = utils.get_sequences(subj)


max_accuracy = [utils.get_decoding_accuracy(subj, clf=clf)[0] for subj in subjects]
test_performance = [utils.get_performance(subj, which='test') for subj in subjects]
df_subjects = pd.DataFrame({'subject': subjects,
                            'accuracy': max_accuracy,
                            'performance': test_performance})

# define which participants are rejected due to preregistered criteria
excluded_acc = [subj for subj in subjects if utils.get_decoding_accuracy(subj, clf=clf)[0]<0.3]
excluded_perf = [subj for subj in subjects if utils.get_performance(subj, which='test')<0.5]
excluded_miss = [subj for subj in subjects if utils.get_responses_localizer(subj)['n_misses']>60*0.25]

df_excluded = pd.DataFrame({'reason': ['low decoding']*len(excluded_acc) + \
                                      ['low performance']*len(excluded_perf) + \
                                      ['>25% misses']*len(excluded_miss)},
                           index=excluded_acc + excluded_perf + excluded_miss)
df_excluded.sort_index(inplace=True)

subjects_incl = sorted(set(subjects).difference(set(df_excluded.index)))

input('data loaded. continue?')

#%% %%% STUDY 1
# %% general description

# 0.1 - memory performance
df_stats = pd.DataFrame()
df_stats_blocks = pd.DataFrame()

# # precomputation, this might take a whi
for subj in tqdm(subjects, desc="cross validating decoding performance"):
    utils.get_decoding_accuracy(subj=subj, clf=clf)

for subj in subjects:
    # retrieve precomputed cross validation results
    acc = utils.get_decoding_accuracy(subj=subj, clf=clf)[0]

    # get participant performance for learning and testing
    val_learn = get_performance(subj=subj, which="learning")
    val_test = get_performance(subj=subj, which="test")
    scatter = (np.random.rand() * -0.5) * 0.02 + 1
    df_tmp = pd.DataFrame(
        {
            "participant": subj,
            "performance": [val_learn[-1] * scatter, val_test * scatter],
            "type": ["learning\n(last block)", "retrieval"],
        }
    )
    df_tmp_blocks = pd.DataFrame(
        {
            "participant": subj,
            "performance": [val_learn[0], val_learn[-1], val_test],
            "block": ["first", "last", "test"],
            "n_blocks": len(val_learn),
            "acc": acc,
        }
    )
    df_stats = pd.concat([df_stats, df_tmp])
    df_stats_blocks = pd.concat([df_stats_blocks, df_tmp_blocks])

# store some results for later usage
df_stats_blocks.to_pickle(settings.cache_dir + "/df_stats_blocks.pkl")

# start plotting
fig = plt.figure(figsize=[4, 6])
ax = fig.subplots(1, 1)
sns.despine()
df_stats_blocks = df_stats_blocks[df_stats_blocks.acc >= min_acc]
sns.boxplot(
    data=df_stats_blocks,
    x="block",
    y="performance",
    color=sns.color_palette()[0],
    meanprops=meanprops,
    showmeans=True,
    ax=ax,
)

ax.set_title("Memory performance")
sns.despine()
plt.tight_layout()
fig.savefig(results_dir + "/memory_performance.png")
fig.savefig(results_dir + "/memory_performance.svg")


perf_first, perf_last, perf_test = df_stats_blocks.groupby("block")
p = ttest_rel(perf_last[1].performance, perf_test[1].performance)
mean_diff = perf_last[1].performance.mean() - perf_test[1].performance.mean()

print(f"Learning performance first block {perf_first[1].performance.mean():.2f}")
print(f"Learning performance last block {perf_last[1].performance.mean():.2f}")
print(f"Learning increase last->test {p=}")

#%% Localizer get best C value
import sklearn
utils.lowpriority()
ex_per_fold = 4
df_reg = pd.DataFrame()

clf_x = sklearn.base.clone(clf)

# we originally did this over a much larger selection of C, but to reduce
# computation time, we select this subrange now. Most C outside the range were
# useless.
Cs = np.logspace(-1.2, 2.5, 25, dtype=np.float16)

for C in tqdm(Cs):
    clf_x.set_params(C=C)
    resx = Parallel(4)(delayed(utils.get_best_timepoint)(*localizer[subj], subj=subj,
                                                         clf=clf_x,  ex_per_fold=8,
                                                         add_null_data=True,
                                                         verbose=False)
                               for subj in subjects)
    df_reg = pd.concat([df_reg]+ resx, ignore_index=True)

df_reg['C'] = np.repeat(Cs.astype(float), len(df_reg)/len(Cs))

joblib.dump(df_reg, settings.results_dir + '/localizer_c.pkl.gz')


# plot results
fig, ax = plt.subplots(figsize=[8, 6])
df_reg = joblib.load(settings.results_dir + '/localizer_c.pkl.gz')
df_reg_mean = df_reg[(df_reg.timepoint>150) & (df_reg.timepoint<250)]
df_reg_mean = df_reg_mean.groupby(['C', 'timepoint']).mean(True).reset_index()
df_reg_peak = df_reg_mean.groupby(['C']).mean(True).reset_index()
best_C = df_reg_peak.C[df_reg_peak.accuracy.argmax()]

ax.set(xscale="log")
sns.lineplot(df_reg_mean, x='C', y='accuracy', ax = ax)
ax.vlines(best_C, 0.1, 0.3, color='black')
best_C

ax.set_xlabel('L1 regularization (C), logscale')
ax.set_ylabel('Mean accuracy between 150-250 ms after stim onset')
ax.set_title(f'Best decoding at L1 ~C={best_C:.1f}')
plt.pause(0.1)
plt.tight_layout()
utils.savefig(fig, 'supplement/regularization_C_crossval.png')
#%% localizer decoding accuracy

# (basically a copy of previous publication code)
# Procedure:
#    For each participant:
#    1. calculate best timepoint
#    2. save best tp in list of all participants
#
# Choose best classifier TP for each participants based on LOSO
#   - use data of all other participant for this participants best_tp
#   - in our case, this yields the same TP for all participants


# preload data that has already been computed
pkl_localizer = settings.cache_dir + "/1 localizer.pkl.zip"
results_localizer = load_pkl_pandas(
                                    pkl_localizer,
                                    default=pd.DataFrame(columns=["subject"])
                                    )

### Calculation of best timepoint and leave-one-out cross-validation
ex_per_fold = 1
for i, subj in enumerate(tqdm(subjects, desc="subject")):

    # this is quite time consuming, so caching is implemented.
    # you can continue computation at a later stage if you interrupt.
    # if subj in results_localizer["subject"].values:
    #     # skip subjects that are already computed
    #     print(f"{subj} already computed")
    #     continue

    data_x, data_y = localizer[subj]
    res = get_best_timepoint(
        data_x, data_y, subj=f"{subj}", n_jobs=-1, ex_per_fold=8, clf=clf
    )
    results_localizer = pd.concat([results_localizer, res], ignore_index=True)

    results_localizer.to_pickle(pkl_localizer)  # store intermediate results


max_acc_subj = [
    results_localizer.groupby("subject")
    .get_group(subj)
    .groupby("timepoint")
    .mean(True)["accuracy"]
    .max()
    for subj in subjects
]
# only include participants with decoding accuracy higher than 0.3
included_idx = np.array(max_acc_subj) >= min_acc
included_subj = np.array(subjects)[included_idx]

subj_accuracy_all = (
    results_localizer.groupby(["subject", "timepoint"]).mean(True).reset_index()
)

# compute best decoding time point
best_tp_tmp = [
    subj_accuracy_all.groupby("subject").get_group(subj)["accuracy"].argmax()
    for subj in subjects
]

best_tp_tmp = np.array(best_tp_tmp)
best_acc_tmp = [
    subj_accuracy_all.groupby("subject").get_group(subj)["accuracy"].max()
    for subj in subjects
]
best_acc_tmp = np.array(best_acc_tmp)

# MEAN
best_tp = int(np.round(np.mean(best_tp_tmp[included_idx])))
best_acc = np.mean(best_acc_tmp[included_idx])

# store best timepoint for later retrieval
compress_pickle.dump(best_tp, f"{settings.cache_dir}/best_tp.pkl.zip")

results_localizer_included = pd.concat(
    [results_localizer.groupby("subject").get_group(subj) for subj in included_subj]
)

subj_acc_included = (
    results_localizer_included.groupby(["subject", "timepoint"])
    .mean(True)
    .reset_index()
)

# start plotting of mean decoding accuracy
fig = plt.figure(figsize=[6, 6])
ax = fig.subplots(1, 1)
sns.despine()
utils.plot_decoding_accuracy(
    results_localizer_included, x="timepoint", y="accuracy", ax=ax, color="tab:blue"
)
ax.set_ylabel("Accuracy")
ax.set_xlabel("ms after stimulus onset")
ax.set_title("Localizer: decoding accuracy")
ax.set_ylim(0, 0.5)

ax.axvspan(
    (best_tp * 10 - 100) - 5, (best_tp * 10 - 100) + 5, alpha=0.3, color="orange"
)
ax.legend(["decoding accuracy", "95% conf.", "chance", "peak"], loc="lower right")

plt.tight_layout()
plt.pause(0.1)
fig.savefig(results_dir + "/localizer_decoding.svg", bbox_inches="tight")
fig.savefig(results_dir + "/localizer_decoding.png", bbox_inches="tight")

#%% localizer heatmap transfer across time

# store precomputed results
pkl_loc = cache_dir + "/1b heatmap loc.pkl.zip"

# get decoder accuracy per participants to exclude low decodable participants
accuracies = [utils.get_decoding_accuracy(subj=subj, clf=clf)[0] for subj in subjects]
subj_incl = [subj for subj, acc in zip(subjects, accuracies) if acc > 0.3]

# calculate time by time decoding heatmap from localizer
# basically: How well can a clf trained on t1 predict t2 of the localizer
if os.path.isfile(pkl_loc):
    # either load precomputed results or compute and store
    maps_localizer = compress_pickle.load(pkl_loc)
else:
    # use a parallel pool, as this is computationally quite expensive
    pool = Parallel(len(subj_incl))  # use parallel pool
    maps_localizer = pool(
        delayed(utils.get_decoding_heatmap)(clf, *localizer[subj], n_jobs=2)
        for subj in subj_incl
    )
    compress_pickle.dump(maps_localizer, pkl_loc)


# zscore maps for statistics
maps_loc_norm = np.array(maps_localizer) - np.mean(maps_localizer)
maps_loc_norm = maps_loc_norm / maps_loc_norm.std()

# perform cluster permutation testing, basically same as when doing fMRI
t_thresh = stats.distributions.t.ppf(1 - 0.05, df=len(maps_localizer) - 1)
t_clust, clusters1, p_values1, H0 = permutation_cluster_1samp_test(
    maps_loc_norm,
    tail=1,
    n_jobs=None,
    threshold=t_thresh,
    adjacency=None,
    n_permutations=1000,
    out_type="mask",
)


# create mask from cluster for all clusters of p<0.05
clusters_sum1 = (np.array(clusters1)[p_values1 < 0.05]).sum(0)

# now plot the heatmaps with masking using MNE visualization functions
fig = plt.figure(figsize=[8, 6])
axs = fig.subplots(1, 1)
x = mne.viz.utils._plot_masked_image(
    axs[0],
    np.mean(maps_localizer, 0),
    times=range(61),
    mask=clusters_sum1,
    cmap="viridis",
    mask_style="contour",
    vmin=0.1,
    vmax=0.4,
)
plt.colorbar(x[0])


ax.set_title('Localizer generalization')
times = np.arange(-100, 500, 50)  # t
ax.set_xticks(np.arange(0, 60, 5), times, rotation=40)
ax.set_yticks(np.arange(0, 60, 5), times)
ax.set_xlabel("localizer train time (ms after onset)")
ax.set_ylabel(
    f'{"retrieval " if i==1 else "localizer "} test time (ms after onset)'
)

fig.tight_layout()
fig.savefig(results_dir + "/classifier-transfer.svg")
fig.savefig(results_dir + "/classifier-transfer.png")


#%% RS1 vs RS2

tp = np.mean(list(best_tp.values())).astype(int)

clf.set_params(C=best_C)
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
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=seed)
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=seed+1)

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
pkl_sequencenes = f'{settings.cache_dir}/rs1rs2-sequenceness.pkl.zip'
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
compress_pickle.dump(df_sequenceness, pkl_sequencenes)


#%% RS1RS2 seq in one plot

df_rs1rs2 = pd.DataFrame()
fig, axs = plt.subplots(2, 2, figsize=[12, 12])
axs = axs.flatten()
for i, sx in enumerate([rs1_sf, rs2_sf, rs1_sb, rs2_sb]):
    condition = ['forward control', 'backward post-learn', 'backward control', 'backward post-learn'][i]
    sx = zscore_multiaxis(sx[:, :, :], axes=zscore_axes)
    seq = sx[:, 0, :].ravel()
    time_lags = list(range(0, sx.shape[-1]*10, 10)) * len(subjects_incl)
    df_tmp = pd.DataFrame({'subject': np.repeat(subjects_incl, sx.shape[-1]),
                           'sequenceness': seq,
                           'time lag': time_lags})
    ax = axs[i]
    sns.lineplot(df_tmp, x='time lag', y='sequenceness', hue='subject', ax=ax,
                 palette=hues, legend=False)
    ax.set_title(f'{condition}')

fig.suptitle('Individual Sequenceness Curves of Participants')
utils.normalize_lims(axs)
utils.savefig(fig, 'supplement/sequenceness-individualplots.png')

#%% RS1 and RS2 plots
# plotting of sequenceness curves
fig, axs = plt.subplot_mosaic([['1', '1', 'D1', 'D1'],
                               ['2', '2', 'D2', 'D2'],
                               ['A', 'B', 'C', 'D']], figsize=[14, 14])

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}
perf_diff = {subj: perf_test[subj] - get_performance(subj=subj, which="learning")[-1]  for subj in subjects_incl}

sequence_names = ['Control', 'Post-Learn']

c_fwd = [sns.color_palette("bright")[1], sns.color_palette('dark')[1]]
c_bkw = [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]

# plot forward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='fwd', title=f'Forward Sequenceness ',
                       ax=axs['1'], rescale=False, color=c_fwd[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='fwd', title=f'Forward Sequenceness',
                       ax=axs['1'], rescale=False, color=c_fwd[1], clear=False)

# plot backward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=False, color=c_bkw[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=False, color=c_bkw[1], clear=False)

for ax in (axs_seq:=[axs['1'], axs['2']]):
    ax.set_ylabel('sequenceness\n(u.u., zscored)')
    ax.set_xlabel('time lag (milliseconds)')
    ax.legend(['Control', '_', '_', '_', '_', 'Post-Learn'],  loc='lower right')
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

utils.normalize_lims([ax for desc, ax in axs.items() if not desc.isnumeric() and len(desc)==1])

# fig.suptitle('Sequenceness Analysis Results')
plt.pause(0.1)
fig.tight_layout()
# utils.savefig(fig, f'figure/sequenceness_rs.png')

## calculate for difference in performance
fig, axs = plt.subplots(2, 2, figsize=[12,8 ])

for i, direction in enumerate(['forward', 'backward']):
    ax = axs[i][0]
    c = [c_fwd, c_bkw][i][0]
    rs_pre = [rs1_sf, rs1_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_pre[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_pre[:, 0, peak_lag]))
    r1, pval1 = plot_correlation(peak_vals, values=perf_diff,
                                 color=c, ax=ax,title=f"Control")
    ax.text(ax.get_xlim()[1], -0.2, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')
    ax.set_ylabel('perf. diff. post-pre')

    ax = axs[i][1]
    c = [c_fwd, c_bkw][i][1]
    rs_post = [rs2_sf, rs2_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_post[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_post[:, 0, peak_lag]))
    r1, pval1 = plot_correlation(peak_vals, values=perf_diff,
                                 color=c, ax=ax,title=f"Post-Learn")
    ax.text(ax.get_xlim()[1], -0.2, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')
    ax.set_ylabel('perf. diff. post-pre')
utils.normalize_lims(axs.flatten())

fig.suptitle('Correlation between performance deltas and peak sequenceness')
plt.pause(0.1)
fig.tight_layout()
utils.savefig(fig, f'supplement/correlation_pre-post-difference.png')

#%% RS2-RS1 post cluster permutation
from mne.stats import permutation_cluster_1samp_test
fontdict ={ 'fontsize': 18, 'horizontalalignment':'center'}

rs_pre_sf = rs1_sf[:, 0, 1:]
rs_pre_sb = rs1_sb[:, 0, 1:]
rs_post_sf = rs2_sf[:, 0, 1:]
rs_post_sb = rs2_sb[:, 0, 1:]

n_subj = len(rs_pre_sf)

# create differences between post and pre
diff_sf =  rs_post_sf - rs_pre_sf
diff_sb =  rs_post_sb - rs_pre_sb


df_diff = pd.DataFrame()
for lag in range( max_lag):
    df_diff = pd.concat([df_diff,
                         pd.DataFrame({'participant': list(range(n_subj))*2,
                         'direction': ['forward'] * n_subj + ['backward']*n_subj,
                         'timelag': 10+lag*10,
                         'difference (u.u.)': list(diff_sf[:, lag]) + list(diff_sb[:, lag])
                                       })],
                         ignore_index=True)


for i, direction in enumerate(['forward', 'backward']):
    ax = axs[f'D{i+1}']
    ax.clear()
    ax.hlines(0, 0, max_lag*10+10, color='gray', linestyle='--', alpha=0.2)
    sns.lineplot(df_diff[df_diff.direction==direction], x='timelag', y='difference (u.u.)', ax=ax,
                 color=sns.color_palette()[i+1], err_style="bars", errorbar=("se", 2),
                 err_kws={'fmt':'o-', 'capsize':5})
    ax.set_ylim([x*1.5 for x in ax.get_ylim()])
    ax.set_title(f'{direction} difference post-pre')
    ax.legend(['_', 'diff', 'SE'], loc='lower right')

utils.normalize_lims( [axs[f'D1'], axs[f'D2']])

p_fwd = stats.ttest_rel(rs_pre_sf, rs_post_sf, axis=0, nan_policy='omit')
p_bkw = stats.ttest_rel(rs_pre_sb, rs_post_sb, axis=0, nan_policy='omit')

_, p_fwd_corr, _, _ = multipletests(p_fwd.pvalue, method='fdr_bh')
_, p_bkw_corr, _, _ = multipletests(p_bkw.pvalue, method='fdr_bh')


for idx in np.where(p_fwd.pvalue<0.05)[0]:
    # axs[f'D1'].text(idx*10+10, axs[f'D1'].get_ylim()[1]*0.75,'(*)', fontdict=fontdict)
    print((idx+1)*10, f'{p_fwd.pvalue[idx]=:.3f} {p_fwd_corr[idx]=:.3f}')

for idx in np.where(p_bkw.pvalue<0.05)[0]:
    # axs[f'D2'].text(idx*10+10, axs[f'D2'].get_ylim()[1]*0.75,'(*)', fontdict=fontdict)
    print((idx+1)*10, f'{p_bkw.pvalue[idx]=:.3f} {p_bkw_corr[idx]=:.3f}')

plt.pause(0.1)
fig.tight_layout()

for i, diff in enumerate([diff_sf, diff_sb]):
    t_thresh = stats.distributions.t.ppf(1 - 0.05, df=len(diff) - 1)
    t_clust, clusters1, p_values1, H0 = permutation_cluster_1samp_test(
        diff,
        tail=1,
        n_jobs=None,
        threshold=t_thresh,
        adjacency=None,
        n_permutations=10000,
        out_type="mask",
    )
    print(clusters1, p_values1)

utils.savefig(fig, f'figure/sequenceness_rs.png')


#%% RS1 vs RS2 segments

tp = 31#np.mean(list(best_tp.values())).astype(int)

# # create forward transition matrix
tf = tdlm.seq2tf(settings.seq_12)

# put forward and backward sequenceness per segment for RS1/RS2 in these arrays
rs1_sf_seg = np.zeros([len(subjects_incl), 8, n_shuf, max_lag+1])
rs1_sb_seg = np.zeros([len(subjects_incl), 8, n_shuf, max_lag+1])
rs2_sf_seg = np.zeros([len(subjects_incl), 8, n_shuf, max_lag+1])
rs2_sb_seg = np.zeros([len(subjects_incl), 8, n_shuf, max_lag+1])

# # create axes to plot subject sequenceness into
fig1, axs1 = utils.make_fig(n_axs=len(subjects_incl), bottom_plots=0, suptitle='RS control')
fig2, axs2 = utils.make_fig(n_axs=len(subjects_incl), bottom_plots=0, suptitle='RS post-learning')

n_segments = 8

for i, subj in enumerate(tqdm(subjects_incl, desc="subject")):
    # train our classifier using localizer data and negative data
    train_x, train_y = localizer[subj]
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # there are 8 segments that are approximately 30 seconds long
    segments_rs1 = np.split(rs1[subj], n_segments)
    segments_rs2 = np.split(rs2[subj], n_segments)

    for seg in range(n_segments):

        # get probability estimates for item reactivation from the resting state
        proba_rs1seg = clf.predict_proba(segments_rs1[seg])
        proba_rs2seg = clf.predict_proba(segments_rs2[seg])
        proba_rs1seg = eval(proba_norm)(proba_rs1seg)
        proba_rs2seg = eval(proba_norm)(proba_rs2seg)

        # calculate sequenceness
        seed = utils.get_id(subj) + seg
        rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(proba_rs1seg, tf=tf,
                                                      n_shuf=n_shuf, max_lag=max_lag,
                                                      alpha_freq=alpha_freq,
                                                      seed=seed)
        rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2seg, tf=tf,
                                                      n_shuf=n_shuf, max_lag=max_lag,
                                                      alpha_freq=alpha_freq,
                                                      seed=seed+1)

        rs1_sf_seg[i, seg, :] = zscore_multiaxis(rs1_sf_subj, axes=zscore_axes)
        rs1_sb_seg[i, seg, :] = zscore_multiaxis(rs1_sb_subj, axes=zscore_axes)
        rs2_sf_seg[i, seg, :] = zscore_multiaxis(rs2_sf_subj, axes=zscore_axes)
        rs2_sb_seg[i, seg, :] = zscore_multiaxis(rs2_sb_subj, axes=zscore_axes)


# plot as heatmap across segments
sequenceness = [(rs1_sf_seg, 'forward', 'control'),
                (rs2_sf_seg, 'forward', 'post-learn'),
                (rs1_sb_seg, 'backward', 'control'),
                (rs2_sb_seg, 'backward', 'post-learn')]

fig, axs = plt.subplots(2, 2, figsize=[14, 8])
axs = axs.flatten()

vmins = [np.nanmin(np.nanmean(sx[:,:,1:,:],0)) for sx, *_ in sequenceness]
vmaxs = [np.nanmax(np.nanmean(sx[:,:,1:,:],0)) for sx, *_ in sequenceness]


for i, (sx, direction, condition) in enumerate(sequenceness):
    heatmap = np.nanmean(sx[:, :, 0, :], 0)
    ax = axs[i]
    im = ax.imshow(heatmap, aspect='auto', cmap='PiYG', vmin=min(vmins), vmax=max(vmaxs))
    ax.set_xlabel('time lag')
    ax.set_xticks(np.arange(0, 31, 5), np.arange(0, 310, 50))
    ax.set_ylabel('min. in resting state')
    ax.set_yticks(np.arange(8), np.arange(1, 9))
    ax.set_title(f'{condition} - {direction}')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
fig.colorbar(im, cax=cbar_ax)

plt.pause(0.1)
fig.tight_layout(rect = [0, 0, 0.88, 1])
cbar_ax.set_ylabel('sequenencess\n(u.u. zscored)', labelpad=10)
cbar_ax.yaxis.set_label_position("left")
utils.savefig(fig, 'figure/sequencess_blocks_heatmap.png', tight=False)

# plot separately, as otherwise it get's too crowded in the plot
fig1, axs = plt.subplots(2, 4)
axs = axs.flatten()
for seg in range(8):
    # first plot RS sequenceness individually
    tdlm.plot_sequenceness(rs1_sf_seg[:,seg, ...], rs1_sb_seg[:,seg, ...], which=['fwd', 'bkw'],
                            title=f'Segment {seg}', ax=axs[seg])
    fig1.suptitle('Sequenceness Control Resting State ')
utils.savefig(fig1, 'supplement/segments-rs1.png')

fig2, axs = plt.subplots(2, 4)
axs = axs.flatten()
for seg in range(8):
    # first plot RS sequenceness individually
    tdlm.plot_sequenceness(rs2_sf_seg[:,seg, ...], rs2_sb_seg[:,seg, ...], which=['fwd', 'bkw'],
                            title=f'Segment {seg}', ax=axs[seg])
    fig2.suptitle('Sequenceness Post-Learning Resting State ')
utils.savefig(fig2, 'supplement/segments-rs2.png')


#%%  %%% STUDY 2:  SIMULATION
#%% Sim-decodability in RS
# in this segment we show that we can decode events that are inserted into
# the resting state by comparing the probability estimates given before
# and after
np.random.seed(42)

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
                                                 lag=lag_sim,
                                                 n_events = 1000,
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
# df_sequenceness = compress_pickle.load(pkl_sequencenes)


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
np.random.seed(42)

# gaussian window created by gaussian_filter1d(np.float_([0,0,1,0,0]), 1)
gaussian_weighting = [0.05842299, 0.24210528, 1, 0.24210528, 0.05842299]

# here we define replay densities that we want to simulate
# replay density is defined in events/minute or events^-1min
densities = np.arange(0, 210, 10)
n_steps = 1
pool = Parallel(-1)
# run twice, once for the regular, linear case, and once for the best-case
# which means scaling from 0-100%
for name_perf_scale in names_perf_scale:
    if name_perf_scale=='linear':
        # perf_scale = lambda perf: perf  #  linear relationship
        perf_scale = lambda perf: -perf+1.5  # inverse linear relationship
    elif name_perf_scale=='best-case':
        perf_scale = lambda perf: (-perf+1.5-0.5)*2  # toggle this for best case scenario
        # perf_scale = lambda perf: (perf-0.5)*2  # toggle this for reverse best case scenario
    else:
        raise ValueError(f'{name_perf_scale=} unknown')
    rs_sims = []

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
        perf = perf_scale(perf)

        # insert into resting state data
        # rs_sim_densities = []

        # get probability estimates for item reactivation from the resting state
        # scale the number of replay events by the performance
        n_events_all = [int(np.round(len(rs_pre)/sfreq/60 * dens * perf)) for dens in densities]
        rs_sim_densities = pool(delayed(tdlm.utils.insert_events)(data = rs_pre,
                                      insert_data = insert_data,
                                      insert_labels = insert_labels,
                                      lag=lag_sim,
                                      distribution='constant',
                                      n_steps=n_steps,
                                      sequence = sequence,
                                      n_events = n_events) for n_events in n_events_all)

        # for d, density in enumerate(densities):

        #     # calculate how many events we need for the specific density
        #     # scaled linearly by the participants peformance.
        #     n_events = int(np.round(len(rs_pre)/sfreq/60 * density * perf))

        #     # insert events into the resting state
        #     rs_sim_dens = tdlm.utils.insert_events(data = rs_pre,
        #                                   insert_data = insert_data,
        #                                   insert_labels = insert_labels,
        #                                   lag=lag_sim,
        #                                   distribution='constant',
        #                                   n_steps=n_steps,
        #                                   sequence = sequence,
        #                                   n_events = n_events)

        #     rs_sim_densities.append(rs_sim_dens)
        rs_sims.append(rs_sim_densities)

    rs_sims = np.array(rs_sims)

    tp = 31 # np.mean(list(best_tp.values())).astype(int)

    # save result in this dictionar
    rs_sim_sf_noalpha = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)
    rs_sim_sf = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)
    rs_sim_sb = np.full([len(subjects_incl), len(densities), n_shuf, max_lag+1], np.nan)

    pkl_simdensity = settings.cache_dir + f'/simulation-full-{name_perf_scale}.pkl.zip'
    # if os.path.exists(pkl_simdensity):
    #     _rs_sim_sf, _rs_sim_sb, _rs_sim_sf_noalpha = compress_pickle.load(pkl_simdensity)
    #     if _rs_sim_sf.shape == rs_sim_sf.shape:
    #         rs_sim_sf = _rs_sim_sf
    #         rs_sim_sb = _rs_sim_sb
    #         rs_sim_sf_noalpha = _rs_sim_sf_noalpha
    #     else:
    #         print(f'mismatch of requested shapes: {rs_sim_sb.shape=} != {_rs_sim_sb.shape=}, recalculate everything.')

    probabilities = []

    for s, subj in enumerate(tqdm(subjects_incl, desc="calculating sequenceness")):

        if not np.isnan(rs_sim_sf[s, :, :, :]).all():
            print(f'{subj=} already computed')
            continue

        # train our classifier using localizer data and negative data
        train_x, train_y = localizer[subj]
        clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

        for d, density in enumerate(tqdm(densities, desc='densities')):

            # get probability estimates of resting state data for our classes
            proba_sim = clf.predict_proba(rs_sims[s, d, ...])
            proba_sim = eval(proba_norm)(proba_sim)
            probabilities.append(proba_sim)

            # calculate sequenceness
            seed = utils.get_id(subj)+d
            sf_sim_noalpha_subj, _ = tdlm.compute_1step(proba_sim, tf=tf, n_shuf=n_shuf,
                                                        max_lag=max_lag, alpha_freq=None,
                                                        seed=seed)

            sf_sim_subj, sb_sim_subj = tdlm.compute_1step(proba_sim, tf=tf, n_shuf=n_shuf,
                                                          max_lag=max_lag, alpha_freq=alpha_freq,
                                                          seed=seed+1)

            # we don't normalize here, to leave the choice of normalization
            # up to be performed by a later function (save some compute)
            rs_sim_sf_noalpha[s, d, :, :] = sf_sim_noalpha_subj
            rs_sim_sf[s, d, :, :] = sf_sim_subj
            rs_sim_sb[s, d, :, :] = sb_sim_subj

    compress_pickle.dump([rs_sim_sf, rs_sim_sb, rs_sim_sf_noalpha], pkl_simdensity)


    # create sequenceness dictionary
    df_simulation = pd.DataFrame()
    pkl_simulation = f'{settings.cache_dir}/simulation-sequenceness-{name_perf_scale}-{lag_sim}.pkl.zip'

    for s, subj in enumerate(tqdm(subjects_incl, desc='creating dictionary')):
        perf = get_performance(subj, which='test')
        for d, density in enumerate(densities):
            for dr, direction in enumerate(['fwd', 'bkw']):
                sx = [rs_sim_sf, rs_sim_sb][dr]
                sx_subj = zscore_multiaxis(sx[s, d, 0], axes=zscore_axes)
                peak_time = np.nanargmax(np.nanmean(np.abs(sx[:, d, 0, :]), 0))
                peak_sequenceness = sx_subj[peak_time]
                mean_sequenencess = np.nanmean(sx_subj)
                seq70 = sx_subj[lag_sim]
                df_tmp = pd.DataFrame({'subject': subj,
                                       'density': density,
                                       'peak': peak_sequenceness,
                                       'mean': mean_sequenencess,
                                       'peak_time': peak_time,
                                       'direction': direction,
                                       'performance': perf,
                                       'seq70': seq70,
                                        },
                                        index=[subj])
                df_simulation = pd.concat([df_simulation, df_tmp],
                                            ignore_index=True)
    compress_pickle.dump(df_simulation, pkl_simulation)


#%% 1 plot sequenceness curves across densities

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
    axs['X'].legend(['forward sequenceness', 'max. perm. thresh.', '_', '95% perm. thresh.'])
    axs['Y'].legend(['backward sequenceness', 'max. perm. thresh.', '_', '95% perm. thresh.'])

    # calculate peak sequenceness across participants
    sf_sim_70 = dict(zip(subjects_incl, sf_sim[:, 0, lag_sim]))
    sb_sim_70 = dict(zip(subjects_incl, sb_sim[:, 0, lag_sim]))

    r1, pval1 = plot_correlation(sf_sim_70, values=perf_test, color=c_fwd, title="performance correlation", ax=axs['B'])
    r2, pval2 = plot_correlation(sb_sim_70, values=perf_test, color=c_bkw, title="performance correlation", ax=axs['D'])

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
    fig.savefig(f'{settings.plot_dir}/simulation_density-{density:03d}_step{n_steps}.png')

# plot correlation at 160min-1 with performance
assert name_perf_scale=='linear', 'wrong setting'
sf_sim_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)
fig, ax = plt.subplots(1, 1, figsize=[8, 6])
idx_density_160 = np.argmax(densities>=160)
peak_lag =  np.nanargmax(abs(np.nanmean(sf_sim_normed[:, idx_density_160, 0, :], 0)))
peak_vals = dict(zip(subjects_incl, sf_sim_normed[:, idx_density_160, 0, peak_lag]))
r, pval = plot_correlation(peak_vals, values=perf_test, ax=ax,title=f"Simulation @ 160min-1")

ax.text(ax.get_xlim()[1], .5, f'r={r:.2f} p={pval:.3f}', horizontalalignment='right')


#%% 2 individual sequenceness curves
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


#%% 3 exemplar 4 density curves
# fig, axs = plt.subplots(2, 2, figsize=[12, 12])
fig, axs = plt.subplots(2, 2, figsize=[12, 12])
axs = axs.flatten()

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

for i, density in enumerate([  0,  40,  80, 120]):
    ax = axs[i]
    d = list(densities).index(density)

    # normalize sequenceness scores here
    sf_sim = zscore_multiaxis(rs_sim_sf[:, d, :, :], axes=zscore_axes)
    sb_sim = zscore_multiaxis(rs_sim_sb[:, d, :, :], axes=zscore_axes)
    ax.clear()
    tdlm.plot_sequenceness(sf_sim, sb_sim, which=['fwd', 'bkw'], ax=ax, rescale=True)
    ax.set_title(f'replay density {density} min⁻¹')
    ax.legend([str(i) for i in range(10)])
    ax.legend(['backward sequenceness', '_', '_', '_', '_', 'forward sequenceness'])
sns.despine(fig)
# fig.suptitle('Simulated replay at different densities')
utils.savefig(fig, 'figure/simulation-sequenceness-examples.png')


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


#%% 4 peak seq. line graph
rs_sim_sf_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)
fig, ax = plt.subplots(1, 1, figsize=[12, 6])
threshholds = np.max(abs(np.mean(rs_sim_sf_normed[:, :, 1:, 1:], 0)), -1)
thresholds_95 = np.quantile(threshholds, 0.95, axis=-1)
thresholds_max = np.nanmax(threshholds, axis=-1)
sequenceness = pd.DataFrame({'sequenceness': rs_sim_sf_normed[:, :, 0, lag_sim].ravel('F'),
                             'density': np.repeat(densities, len(subjects_incl))})
sns.lineplot(sequenceness, x='density', y='sequenceness', errorbar='se',
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
fig.suptitle(f'Forward {lag_sim*10} ms sequenceness at simulated replay densities')
utils.savefig(fig, f'figure/simulation-{lag_sim*10}ms-sequenceness.png')

#%% 5a Corr with performance

pkl_sim_linear = f'{settings.cache_dir}/simulation-sequenceness-linear-{lag_sim}.pkl.zip'
pkl_sim_bestcase = f'{settings.cache_dir}/simulation-sequenceness-best-case-{lag_sim}.pkl.zip'

df_sim_linear = compress_pickle.load(pkl_sim_linear)
df_sim_bestcase = compress_pickle.load(pkl_sim_bestcase)
df_sim_linear['performance scaling'] = 'realistic scaling'
df_sim_bestcase['performance scaling'] = 'maximal scaling'

df_both = pd.concat([df_sim_linear, df_sim_bestcase], ignore_index=True)
df_both = df_both[df_both.direction=='fwd']

# Assuming df_both is your DataFrame
df_corr = df_both.groupby(['performance scaling', 'density']).apply(
            lambda group: pd.Series(stats.pearsonr(group.seq70, group.performance), index=['r', 'p'])
        ).reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
palette = sns.color_palette()

sns.lineplot(df_corr, x='density', y='r', hue='performance scaling',
             ax=ax, palette=[palette[0], 'yellowgreen'])

r_sign = utils.calculate_r_threshold(len(df_both.subject.unique()))  # this is the r value at which it will be significant

ax.hlines(-r_sign, *ax.get_xlim(), linestyle='--', color='red', label='p<0.05', alpha=0.5)
ax.legend(loc='upper left', framealpha=1.0)
ax.invert_yaxis()

# invert axis as we have negative correlations
# ax.set_ylim(*[y*-1 for y in ax.get_ylim()])

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
from joblib import Parallel, delayed
perf_test = [get_performance(subj=subj, which="test") for subj in subjects_incl]

pkl_simdensity = settings.cache_dir + f'/simulation-full-best-case.pkl.zip'
rs_sim_sf, rs_sim_sb, _ = compress_pickle.load(pkl_simdensity)

np.random.seed(20241009)  # todays date as random seed
rs_sim_sf_normed = zscore_multiaxis(rs_sim_sf, axes=zscore_axes)
n_repetitions = 10000
sample_sizes = range(15, 200)

density_95 = np.nonzero(densities>=80)[0][0]

def r_val_boostrap(n):
    rs = []
    ps = []
    for repetition in range(n_repetitions):
        idx_sample = np.random.choice(np.arange(len(rs_sim_sf_normed)), size=n)
        seq_bootstrapped = rs_sim_sf_normed[idx_sample, density_95, 0, lag_sim]
        perf_bootstrapped = [perf_test[x] for x in idx_sample]
        r, p = stats.pearsonr(seq_bootstrapped, perf_bootstrapped)
        rs += [r]
        ps += [p]
    return rs, ps

res = Parallel(-1)(delayed(r_val_boostrap)(n) for n in tqdm(sample_sizes))

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
pkl_sim_linear = f'{settings.cache_dir}/simulation-sequenceness-linear-{lag_sim}.pkl.zip'
pkl_sim_bestcase = f'{settings.cache_dir}/simulation-sequenceness-best-case-{lag_sim}.pkl.zip'

df_sim_baseline = df_simulation[df_simulation.density==0]
df_sim_linear = compress_pickle.load(pkl_sim_linear)
df_sim_bestcase = compress_pickle.load(pkl_sim_bestcase)
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
mp = sns.scatterplot(df_sel, x='condition', y='seq70', hue='subject', palette=hues,
                  legend=True, s=100, ax=ax)
mp = sns.lineplot(df_sel, x='condition', y='seq70', hue='subject', palette=hues,
                  legend=False, ax=ax, alpha=0.5)


ax.set_ylabel('sequenceness at 70 ms (u.u.)')
ax.set_xlabel('')
ax.set_title('Sequenceness variation across subject')
ax.legend([*4*['_'], '50%', '_', '75%', '_', '_', '_', '100%'], loc='upper left', title='Performance')

utils.savefig(fig, 'figure/correlation-bestcase-realistisc-case.png')

#%% Cluster permutation of differences
from mne.stats import permutation_cluster_1samp_test
fontdict ={ 'fontsize': 18, 'horizontalalignment':'center'}


all_axs = []

for i, density in enumerate(densities):
    fig, axs = plt.subplots(2, 1)
    all_axs.extend(axs)
    rs_pre_sf = rs_sim_sf[:, 0, 0, 1:]
    rs_pre_sb = rs_sim_sb[:, 0, 0, 1:]
    rs_post_sf = rs_sim_sf[:, i, 0, 1:]
    rs_post_sb = rs_sim_sb[:, i, 0, 1:]

    n_subj = len(rs_pre_sf)

    # create differences between post and pre
    diff_sf =  rs_pre_sf - rs_post_sf
    diff_sb =  rs_pre_sb - rs_post_sb


    df_diff = pd.DataFrame()
    for lag in range( max_lag):
        df_diff = pd.concat([df_diff,
                             pd.DataFrame({'participant': list(range(n_subj))*2,
                             'direction': ['forward'] * n_subj + ['backward']*n_subj,
                             'timelag': 10+lag*10,
                             'difference (u.u.)': list(diff_sf[:, lag]) + list(diff_sb[:, lag])
                                           })],
                             ignore_index=True)


    for i, direction in enumerate(['forward', 'backward']):
        ax = axs[i]
        ax.clear()
        ax.hlines(0, 0, max_lag*10+10, color='gray', linestyle='--', alpha=0.2)
        sns.lineplot(df_diff[df_diff.direction==direction], x='timelag', y='difference (u.u.)', ax=ax,
                     color=sns.color_palette()[i+1], err_style="bars", errorbar=("se", 2),
                     err_kws={'capsize':5})
        ax.set_ylim([x*1.5 for x in ax.get_ylim()])
        ax.set_title(f'{direction} difference pre-post {density=}')
        ax.legend(['_', 'diff', 'SE'], loc='lower right')

    utils.normalize_lims( [axs[0], axs[1]])

    p_fwd = stats.ttest_rel(rs_pre_sf, rs_post_sf, axis=0, nan_policy='omit')
    p_bkw = stats.ttest_rel(rs_pre_sb, rs_post_sb, axis=0, nan_policy='omit')

    _, p_fwd_corr, _, _ = multipletests(p_fwd.pvalue, method='fdr_bh')
    _, p_bkw_corr, _, _ = multipletests(p_bkw.pvalue, method='fdr_bh')


    for idx in np.where(p_fwd.pvalue<0.05)[0]:
        star = '*' if p_fwd_corr[idx]<0.05 else '(*)'
        axs[0].text(idx*10+10, axs[0].get_ylim()[1]*0.75,star, fontdict=fontdict)
        print((idx+1)*10, f'{p_fwd.pvalue[idx]=:.3f} {p_fwd_corr[idx]=:.3f}')

    for idx in np.where(p_bkw.pvalue<0.05)[0]:
        star = '*' if p_fwd_corr[idx]<0.05 else '(*)'
        axs[1].text(idx*10+10, axs[1].get_ylim()[1]*0.75,star, fontdict=fontdict)
        print((idx+1)*10, f'{p_bkw.pvalue[idx]=:.3f} {p_bkw_corr[idx]=:.3f}')

    plt.pause(0.1)
    fig.tight_layout()

    for i, diff in enumerate([diff_sf, diff_sb]):
        t_thresh = stats.distributions.t.ppf(1 - 0.05, df=len(diff) - 1)
        t_clust, clusters1, p_values1, H0 = permutation_cluster_1samp_test(
            diff,
            tail=1,
            n_jobs=None,
            threshold=t_thresh,
            adjacency=None,
            n_permutations=10000,
            out_type="mask",
        )
        print(p_values1)
        for (t, ), p in zip(clusters1, p_values1, strict=True):
            if p<0.05:
                ax = axs[i]
                ax.hlines(ax.get_ylim()[1]*0.8, t.start*10, t.stop*10,
                          color='red', linewidth=5, alpha=0.5)


utils.normalize_lims(all_axs)
# utils.savefig(fig, f'figure/cluster_sim.png')


#%% #### SUPPLEMENT
# here are calculations that I need for the supplement
#%% SUPPL: sensor patterns


patterns = {img:[] for img in ['berg',
                                 'schreibtisch',
                                 'pinsel',
                                 'kuchen',
                                 'apfel',
                                 'zebra',
                                 'clown',
                                 'fahrrad',
                                 'tasse',
                                 'fuß']}

for i, subj in enumerate(tqdm(subjects, desc="subject")):
    # train our classifier using localizer data and negative data
    train_x, train_y = localizer[subj]
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # get data from localizer that is inserted into the resting state
    insert_data_tp = train_x[:, :, tp]  # take the ERP component at peak decodability

    # ERP of all visual evoked activity
    erp = insert_data_tp.mean(0)

    # now for each class, subtract the mean EPR from the class ERP.
    # this way, we should isolate the pattern that is responsible
    # for the differences
    names_subj = utils.get_image_names(subj)

    for i, name in enumerate(names_subj):
        insert_class_pattern = [insert_data_tp[train_y==i, :].mean(0)-erp]
        patterns[name] += insert_class_pattern


fig = plt.figure(figsize=[10, 8])
axs = fig.subplots(3, 4)
axs = axs.flatten()
cmap = sns.color_palette("vlag", as_cmap=True)

vmin = np.min([np.mean(values, 0) for values in patterns.values()])/1.2
vmax = np.max([np.mean(values, 0).max() for values in patterns.values()])/1.2

for i, name in enumerate(patterns):
    plotting.plot_sensors(np.mean(patterns[name], 0),
                          ax=axs[i],
                          vmin=vmin,
                          vmax=vmax,
                          mode="color",
                          title=name, cmap=cmap)

axs[-1].axis("off")
axs[-2].axis("off")

utils.savefig(fig, f'supplement/insertion_patterns.png')

#%% SUPPL: visualize ERP
from meg_utils import plotting
from scipy.stats import zscore

erps = np.zeros([len(localizer), 10, 306])
for s, subj in enumerate(localizer):
    data_x, data_y = localizer[subj]
    names = utils.get_image_names(subj)
    names_base = sorted(names)
    for y in range(10):
        data_img = data_x[data_y==y]
        erps[s, names_base.index(names[y])] = data_img[:, :, 31].mean(0)

erps_mean = erps.mean(0)

fig = plt.figure(figsize=[10, 7])
axs = fig.subplots(3, 4)
axs = axs.flatten()
plotting.plot_sensors(erps_mean.mean(0), ax=axs[0], mode="size",
                      title='mean', cmap='RdYlBu')
for i in range(10):
    plotting.plot_sensors(erps_mean[i], ax=axs[i+1], mode="size",
                          title=names_base[i], cmap='RdYlBu')
axs[-1].axis("off")
axs[-2].axis("off")

fig.suptitle("Sensor value of images")
plt.pause(0.1)
fig.tight_layout
fig.savefig(results_dir + "/image-ERP.svg")
fig.savefig(results_dir + "/image-ERP.png")

#%% SUPPL: visualize sensors betas
from meg_utils import plotting

def fit(clf, subj, *args, **kwargs):
    np.random.seed(utils.get_id(subj))
    return clf.fit(*args, **kwargs)


clfs = Parallel(len(localizer) - 1)(
    delayed(fit)(clf, subj, X=localizer[subj][0][:, :, 31], y=localizer[subj][1])
    for subj in localizer
)

# these are the orders of images as they were assigned to the classes
names = [utils.get_image_names(subj) for subj in subjects]
names_base = sorted(names[0])

# we need to sort the rows of the matrix accordingly
sorting = [[names_base.index(x) for x in name] for name in names]
betas = np.array([clf.coef_[idxs, :] for clf, idxs in zip(clfs, sorting)])
sensors_active = np.mean(betas>0, 0)

fig = plt.figure(figsize=[10, 7])
axs = fig.subplots(3, 4)
axs = axs.flatten()
plotting.plot_sensors(sensors_active.mean(0), ax=axs[0], mode="size",
                      title='mean')
for i in range(10):
    plotting.plot_sensors(sensors_active[i], ax=axs[i+1], mode="size",
                          title=names_base[i])
axs[-1].axis("off")
axs[-2].axis("off")

fig.suptitle("Sensor distribution for different images")
plt.pause(0.1)
fig.tight_layout
fig.savefig(results_dir + "/S6 sensorlocation.svg")
fig.savefig(results_dir + "/S6 sensorlocation.png")

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
                                                  seed=seed)
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=None,
                                                  seed=seed+1)

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
compress_pickle.dump(df_sequenceness, pkl_sequencenes)

# plotting of sequenceness curves
fig, axs = plt.subplot_mosaic('11AB\n22CD', figsize=[16, 8])

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}

sequence_names = ['Control', 'Post-Learn']

c_fwd = [sns.color_palette("bright")[1], sns.color_palette('dark')[1]]
c_bkw = [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]

# plot forward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='fwd', title=f'Forward Sequenceness ',
                       ax=axs['1'], rescale=False, color=c_fwd[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='fwd', title=f'Forward Sequenceness',
                       ax=axs['1'], rescale=False, color=c_fwd[1], clear=False)

# plot backward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=False, color=c_bkw[0], clear=True)
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=False, color=c_bkw[1], clear=False)

for ax in (axs_seq:=[axs['1'], axs['2']]):
    ax.set_ylabel('sequenceness\n(u.u., zscored)')
    ax.set_xlabel('time lag (milliseconds)')
    ax.legend(['Control', '_', '_', '_', '_', 'Post-Learn'],  loc='lower right')
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

# fig.suptitle('Sequenceness Analysis Results')
plt.pause(0.1)
fig.tight_layout()
utils.savefig(fig, f'supplement/sequenceness_rs_noalpha.png')


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
