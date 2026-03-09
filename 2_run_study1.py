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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
from meg_utils import plotting,  decoding, misc

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
best_tp = utils.load_pkl(f"{settings.cache_dir}/best_tp.pkl.zip", 31)

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
fig.savefig(results_dir + "/memory_performance.eps")


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
utils.savefig(fig, 'supplement/regularization_C_crossval.svg')
utils.savefig(fig, 'supplement/regularization_C_crossval.eps')
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
joblib.dump(best_tp, f"{settings.cache_dir}/best_tp.pkl.zip")

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
fig.savefig(results_dir + "/localizer_decoding.eps", bbox_inches="tight")

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
    maps_localizer = joblib.load(pkl_loc)
else:
    # use a parallel pool, as this is computationally quite expensive
    pool = Parallel(n_jobs)  # use parallel pool
    maps_localizer = pool(
        delayed(utils.get_decoding_heatmap)(clf, *localizer[subj], n_jobs=2)
        for subj in subj_incl
    )
    joblib.dump(maps_localizer, pkl_loc)


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
ax = fig.subplots(1, 1)
x = mne.viz.utils._plot_masked_image(
    ax,
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
fig.savefig(results_dir + "/classifier-transfer.eps")

#%% RS1 vs RS2

tp = best_tp

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
    clf.reset_random_state(misc.make_seed(i, "rs"))
    clf.fit(train_x[:, :, tp], train_y, neg_x=neg_x[subj], neg_x_ratio=2.0)

    # get probability estimates for item reactivation from the resting state
    proba_rs1 = clf.predict_proba(rs1[subj])
    proba_rs2 = clf.predict_proba(rs2[subj])

    # normalize probabilities
    proba_rs1 = eval(proba_norm)(proba_rs1)
    proba_rs2 = eval(proba_norm)(proba_rs2)

    # calculate sequenceness
    rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(proba_rs1, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=misc.make_seed(subj, 'rs1'))
    rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2, tf=tf, n_shuf=n_shuf,
                                                  max_lag=max_lag, alpha_freq=alpha_freq,
                                                  seed=misc.make_seed(subj, 'rs2'))

    # zscore sequenceness scores
    rs1_sf[i, :] = zscore_multiaxis(rs1_sf_subj, axes=zscore_axes)
    rs1_sb[i, :] = zscore_multiaxis(rs1_sb_subj, axes=zscore_axes)
    rs2_sf[i, :] = zscore_multiaxis(rs2_sf_subj, axes=zscore_axes)
    rs2_sb[i, :] = zscore_multiaxis(rs2_sb_subj, axes=zscore_axes)

    # # # subject level plot
    tdlm.plot_sequenceness(rs1_sf[i, :], rs1_sb[i, :] , ax=axs1[i], which=['fwd', 'bkw'], rescale=False, plotsignflip=False)
    tdlm.plot_sequenceness(rs2_sf[i, :], rs2_sb[i, :] , ax=axs2[i], which=['fwd', 'bkw'], rescale=False, plotsignflip=False)
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
joblib.dump(df_sequenceness, pkl_sequencenes)

#%% RS1 and RS2 main plot
# plotting of sequenceness curves
fig, axs = plt.subplot_mosaic([['1', '1', 'D1', 'D1'],
                               ['2', '2', 'D2', 'D2'],
                               ['S1', 'S2', 'S3', 'S4'],
                               ['A', 'B', 'C', 'D']], figsize=[14, 14])

perf_test = {subj: get_performance(subj=subj, which="test") for subj in subjects_incl}
perf_diff = {subj: perf_test[subj] - get_performance(subj=subj, which="learning")[-1]  for subj in subjects_incl}

sequence_names = ['Control', 'Post-Learn']

c_fwd = [sns.color_palette("bright")[1], sns.color_palette('dark')[1]]
c_bkw = [sns.color_palette("bright")[2], sns.color_palette('dark')[2]]

#### plot forward and backward
tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='fwd', title=f'Forward Sequenceness ',
                       ax=axs['1'], rescale=True, color=c_fwd[0], clear=True, label='Control')
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='fwd', title=f'Forward Sequenceness',
                       ax=axs['1'], rescale=True, color=c_fwd[1], clear=False, label='Post-Learn')

tdlm.plot_sequenceness(rs1_sf, rs1_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=True, color=c_bkw[0], clear=True, label='Control')
tdlm.plot_sequenceness(rs2_sf, rs2_sb, which='bkw', title=f'Backward Sequenceness',
                       ax=axs['2'], rescale=True, color=c_bkw[1], clear=False, label='Post-Learn')

for ax in (axs_seq:=[axs['1'], axs['2']]):
    ax.set_ylabel('sequenceness\n(u.u., zscored)')
    ax.set_xlabel('time lag (milliseconds)')
    ax.legend(handles=[ln for ln in ax.lines if ln.get_label()])
    ax.plot([0, 0.000001], [0, .000000001], color='gray', linestyle='--', linewidth=1)  # just for legend creation
    ax.plot([0, 0.000001], [0, .000000001], color='gray', linestyle='--', linewidth=1.5)  # just for legend creation
    hndls = ax.get_legend_handles_labels()
    ax.legend([hndls[0][i] for i in [0, 3, 4, 5]],
              [hndls[1][i] for i in [0, 3, 4, 5]],
              loc='lower right',
              ncols=2, fontsize=12)


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


#### signflip test
colors = [sns.color_palette("bright")[1:3], sns.color_palette("dark")[1:3]]

for s, session in enumerate(['Control', 'Post-Learn']):
    for d, direction in enumerate(['Forward', 'Backward']):
        sx = [[rs1_sf, rs1_sb], [rs2_sf, rs2_sb]][s][d]
        pval, t_base, t_perm = tdlm.signflit_test(sx[:, 0, :], rng=misc.make_seed(s, d))
        print(f'{session=} {direction=} {pval=:.3f} {t_base=:.3f}')
        ax = axs[f'S{d*2+s+1}']
        ax.clear()
        color = colors[s][d]
        tdlm.plot_tval_distribution(t_base, t_perm, color=color, ax=ax)
        ax.set_xlabel('sign-flip t-value distribution')
        ax.set_title(f'{session}')

        leg = ax.get_legend()
        plt.setp(leg.get_texts(), fontsize='smaller')
        plt.setp(leg.get_title(), fontsize='smaller')



#### RS2-RS1 post cluster permutation
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
                         'direction': ['Forward'] * n_subj + ['Backward']*n_subj,
                         'timelag': 10+lag*10,
                         'difference (u.u.)': list(diff_sf[:, lag]) + list(diff_sb[:, lag])
                                       })],
                         ignore_index=True)

for i, direction in enumerate(['Forward', 'Backward']):
    ax = axs[f'D{i+1}']
    ax.clear()
    ax.hlines(0, 0, max_lag*10+10, color='gray', linestyle='--', alpha=0.2, label='_nolegend_')
    sns.lineplot(df_diff[df_diff.direction==direction], x='timelag', y='difference (u.u.)', ax=ax,
                 color=sns.color_palette()[i+1], err_style="bars", errorbar=("se", 2),
                 err_kws={'fmt':'o-', 'capsize':5})
    ax.set_ylim([x*1.5 for x in ax.get_ylim()])
    ax.set_title(f'{direction} Difference Post minus Control')
    ax.legend(['diff', 'SE'], loc='lower right')

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
utils.savefig(fig, f'figure/sequenceness_rs.svg')
utils.savefig(fig, f'figure/sequenceness_rs.eps')

#%% correlation with behaviour
fig2, axs2 = plt.subplots(2, 2, figsize=[12,8 ])

for i, direction in enumerate(['forward', 'backward']):
    ax = axs2[i][0]
    ax.clear()
    c = [c_fwd, c_bkw][i][0]
    rs_pre = [rs1_sf, rs1_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_pre[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_pre[:, 0, peak_lag]))
    r1, pval1 = plot_correlation(peak_vals, values=perf_diff,
                                 color=c, ax=ax, title=f"Control")
    ax.text(ax.get_xlim()[1], -0.2, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')
    ax.set_ylabel('perf. diff. post-pre')

    ax = axs2[i][1]
    c = [c_fwd, c_bkw][i][1]
    rs_post = [rs2_sf, rs2_sb][i]
    peak_lag =  np.nanargmax(abs(np.nanmean(rs_post[:, 0, :], 0)))
    peak_vals = dict(zip(subjects_incl, rs_post[:, 0, peak_lag]))
    r1, pval1 = plot_correlation(peak_vals, values=perf_diff,
                                 color=c, ax=ax,title=f"Post-Learn")
    ax.text(ax.get_xlim()[1], -0.2, f'r={r1:.2f} p={pval1:.3f}', horizontalalignment='right')
    ax.set_ylabel('perf. diff. post-pre')
utils.normalize_lims(axs2.flatten())

fig2.suptitle('Correlation between performance deltas and peak sequenceness')
plt.pause(0.1)
fig2.tight_layout()
utils.savefig(fig2, f'supplement/correlation_pre-post-difference.png')
utils.savefig(fig2, f'supplement/correlation_pre-post-difference.svg')
utils.savefig(fig2, f'supplement/correlation_pre-post-difference.eps')
#%% RS1RS2 all sequenceness curves in one plot

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
utils.savefig(fig, 'supplement/sequenceness-individualplots.svg')
utils.savefig(fig, 'supplement/sequenceness-individualplots.eps')

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
    clf.reset_random_state(misc.make_seed(i, "rs"))
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
        rs1_sf_subj, rs1_sb_subj = tdlm.compute_1step(proba_rs1seg, tf=tf,
                                                      n_shuf=n_shuf, max_lag=max_lag,
                                                      alpha_freq=alpha_freq,
                                                      seed=misc.make_seed(subj, seg))
        rs2_sf_subj, rs2_sb_subj = tdlm.compute_1step(proba_rs2seg, tf=tf,
                                                      n_shuf=n_shuf, max_lag=max_lag,
                                                      alpha_freq=alpha_freq,
                                                      seed=misc.make_seed(subj, seg))

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
utils.savefig(fig, 'figure/sequencess_blocks_heatmap.svg', tight=False)
utils.savefig(fig, 'figure/sequencess_blocks_heatmap.eps', tight=False)

# plot separately, as otherwise it get's too crowded in the plot
fig1, axs = plt.subplots(2, 4, figsize=[16, 8], sharex=True, sharey=True)
axs = axs.flatten()
for seg in range(8):
    # first plot RS sequenceness individually
    tdlm.plot_sequenceness(rs1_sf_seg[:,seg, ...], rs1_sb_seg[:,seg, ...], which=['fwd', 'bkw'],
                            title=f'Minute {seg+1}', ax=axs[seg])
    fig1.suptitle('Sequenceness Control Resting State ')
    axs[seg].legend(loc='upper left', ncols=2)
plt.pause(0.1)
utils.savefig(fig1, 'supplement/segments-rs1.png')
utils.savefig(fig1, 'supplement/segments-rs1.svg')
utils.savefig(fig1, 'supplement/segments-rs1.eps')

fig2, axs = plt.subplots(2, 4, figsize=[16, 8], sharex=True, sharey=True)
axs = axs.flatten()
for seg in range(8):
    # first plot RS sequenceness individually
    tdlm.plot_sequenceness(rs2_sf_seg[:,seg, ...], rs2_sb_seg[:,seg, ...], which=['fwd', 'bkw'],
                            title=f'Minute {seg+1}', ax=axs[seg])
    fig2.suptitle('Sequenceness Post-Learning Resting State ')
    axs[seg].legend(loc='upper left', ncols=2)
plt.pause(0.1)

utils.savefig(fig2, 'supplement/segments-rs2.png')
utils.savefig(fig2, 'supplement/segments-rs2.svg')
utils.savefig(fig2, 'supplement/segments-rs2.eps')





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
utils.savefig(fig, f'supplement/insertion_patterns.svg')
utils.savefig(fig, f'supplement/insertion_patterns.eps')

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
fig.savefig(results_dir + "/image-ERP.eps")

#%% SUPPL: visualize sensors betas
from meg_utils import plotting

def fit(clf, subj, *args, **kwargs):
    np.random.seed(misc.make_seed(subj))
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
fig.savefig(results_dir + "/S6 sensorlocation.eps")
