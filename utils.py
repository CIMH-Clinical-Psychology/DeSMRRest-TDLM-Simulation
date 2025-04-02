# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:05:26 2023

collection of utility functions for paper

@author: Simon Kern
"""
import datetime
import inspect
import itertools
import json
import os
import random
import re
import warnings
import hashlib
from pathlib import Path
from types import ModuleType

import compress_pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory, Parallel, delayed
from numpyencoder import NumpyEncoder
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import pearsonr, zscore
from tqdm import tqdm
from scipy.stats import t

import settings

# cache some static results here
memory = Memory(settings.cache_dir if settings.caching_enabled else None)


def calculate_r_threshold(n, alpha=0.05):
    # Degrees of freedom
    df = n - 2
    # Critical t-value for two-tailed test
    t_crit = t.ppf(1 - alpha / 2, df)
    # Calculate r threshold
    r_threshold = t_crit / (t_crit**2 + df)**0.5
    return r_threshold

def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)

def persistent_hash(input_string):
    # Convert input string to bytes
    input_bytes = input_string.encode('utf-8')
    # Calculate hash using both Python's hash() function and SHA-256
    md5_hash = hashlib.md5(input_bytes).hexdigest()
    return md5_hash

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True,
                  **kwargs):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     fill=False, linewidth=1, **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def load_pkl_pandas(pkl_file, default=None):
    """helper function that loads PKL to pandas, with optional default"""
    if os.path.isfile(pkl_file):
        return pd.read_pickle(pkl_file)
    return pd.DataFrame() if default is None else default


def load_pkl(pkl_file, default):
    if os.path.isfile(pkl_file):
        return compress_pickle.load(pkl_file)
    return default


def get_id(name, pattern=r"DSMR[^0-9]*(\d+)/*"):
    """given a file name, extract the subject id."""
    res = re.findall(pattern, name.upper() + "/")
    assert len(res) != 0, f"Pattern {pattern} not found in string {name}"
    assert len(res) == 1, f"Found more or less pattern matches for {pattern} in {name}"
    assert res[0].isdigit(), "{res} does not seem to be a digit?"
    return int(res[0])

def interpolate_vector(vector, new_length):
    from scipy.interpolate import UnivariateSpline
    old_indices = np.arange(0, len(vector))
    new_indices = np.linspace(0,len(vector)-1,new_length)
    spl = UnivariateSpline(old_indices,vector,k=3,s=0)
    new_array = spl(new_indices).round(14)
    return new_array

def interpolate_matrix(matrix):
    from scipy.interpolate import interp1d
    # Check if matrix is of the correct shape
    if matrix.shape[1] != 3 or not (3 <= matrix.shape[0] <= 7):
        raise ValueError("Input matrix must be of shape (x, 3) where 3 <= x <= 7")

    # Original indices (based on the number of rows in the input matrix)
    original_indices = np.linspace(0, 1, matrix.shape[0])

    # New indices for a 7-row matrix
    new_indices = np.linspace(0, 1, 7)

    # Initialize the output matrix
    new_matrix = np.zeros((7, 3))

    # Interpolate each column
    for i in range(3):
        interp_func = interp1d(original_indices, matrix[:, i], kind='linear')
        new_matrix[:, i] = interp_func(new_indices)

    return new_matrix

@memory.cache
def get_sequences(subj, which="test"):
    """
    reads the sequences from a log file
    """
    assert "DSMR" in subj, "subj must contain DSMR, but is {subj=}"
    subj_dict = json_load(f"./data/{subj}.json")
    if which=='test':
        seqs = subj_dict["seqs"]
        response = subj_dict["response"]
        return seqs, response

    elif which=='learning':
        blocks = []
        sdata = subj_dict['blocks']
        # for sanity check, see if performance matches
        perf = get_performance(subj, which='learning')

        for n in range(len(subj_dict['performance-learning'])):
            seq = []
            resp = []
            for trial in range(12):
                n_trial =  str((n*12) + trial)
                block = sdata['Block'][n_trial]
                choice =  sdata['Choice'][n_trial]
                expect =  sdata['Correct'][n_trial]
                # the seq data has format [seq1, seq2, opt1, opt2, opt3, corr]
                seq_tmp = [sdata[f'Seq{i}'][n_trial] for i in [1,2]]
                seq_tmp += [sdata[f'option{i}'][n_trial] for i in [1,2,3]]
                seq_tmp += [num2char(expect)]

                r = 'correct' if choice==expect else 'wrong'
                seq += [char2num(seq_tmp)]
                resp += [r]

            # sanity check
            assert block==n+1  # correct block?
            # performance same as in logs?
            assert (sum([r=='correct' for r in resp])/12*100)//1==((perf[n]*100)//1)
            blocks.append([seq, resp])
        return blocks
    else:
        raise ValueError("which must select test or learning")


@memory.cache
def get_responses_localizer(subj):
    assert "DSMR" in subj, "subj must contain DSMR, but is {subj=}"
    subj_dict = json_load(f"./data/{subj}.json")
    return subj_dict['localizer']

@memory.cache
def get_responses(subj):
    assert "DSMR" in subj, "subj must contain DSMR, but is {subj=}"
    subj_dict = json_load(f"./data/{subj}.json")
    df_blocks = pd.DataFrame(subj_dict['blocks'])
    return df_blocks

@memory.cache
def get_performance(subj, which="test"):
    """
    reads the retention performance of the participant from the log file
    returns 0 if no log file can be found
    """
    assert "DSMR" in subj, "subj must contain DSMR, but is {subj=}"

    subj_dict = json_load(f"./data/{subj}.json")

    if which == "test":
        performance = subj_dict["performance-test"]

    elif which == "learning":
        performance = subj_dict["performance-learning"]
    else:
        raise ValueError("which must select test or learning")

    return performance




def plot_correlation(
    sequenceness, values, ax=None, absolute=False, color=None,
    xlabel="performance", title=None):
    """
    :param sf: forward sequenceness
    :param sb: backwards sequenceness
    :param values: values to form correlation with
    :return: DESCRIPTION
    """
    assert isinstance(values, dict), "values must be supplied with subjects"
    assert isinstance(sequenceness, dict), "sequenceness must be supplied with subjects"
    subjects = list(values)
    # for some analysis, sign isn't meaningful, for others it is.
    transform = lambda arr: np.abs(arr) if absolute else arr
    df = pd.DataFrame(
        {
            "subject": subjects,
            "sequenceness": transform([sequenceness[subj] for subj in subjects]),
            xlabel: [values[subj] for subj in subjects],
        }
    )

    ax = sns.regplot(data=df, x="sequenceness", y=xlabel, ax=ax, color=color)
    x = df["sequenceness"]
    y = df[xlabel]
    r, pval = pearsonr(x[~np.isnan(x)], y[~np.isnan(x)])
    if title is None:
        title = f"{r=:.3f}, {pval=:.3f}"
    ax.set_title(title)
    return r, pval


def valid_filename(string, replacement="_"):
    """
    replace all non-valid filename characters with an underscore
    """
    invalid_chars = '<>:"/\\|?*\n\r\t'
    conversion = {c: replacement for c in invalid_chars}
    conversion['"'] = "'"  # replace by valid quotes
    conversion["<"] = "("  # replace by other brakets
    conversion[">"] = ")"  # replace by other brakets

    string = str(string)
    valid_filename = "".join([conversion.get(c, c) for c in string])
    return valid_filename

def zscore_multiaxis(arr, axes):
    """
    Wrapper for scipy.stats.zscore to normalize along multiple axes.

    Parameters:
    - arr: The input array to normalize.
    - axis: A list or tuple of axes along which to apply the z-score normalization.

    Returns:
    - The normalized array with the same shape as the input.
    """
    if isinstance(axes, bool) and axes==False:
        return arr

    if isinstance(axes, int):
        axes = [axes]

    # Step 1: Move the axes to be normalized to the front, and flatten them
    axes = [a if a >= 0 else a + arr.ndim for a in axes]
    axes = sorted(axes)  # Sort the axis list to avoid reshuffling issues
    other_axes = [i for i in range(arr.ndim) if i not in axes]

    # Transpose to bring specified axes to the front
    transposed_arr = np.transpose(arr, axes=axes + other_axes)

    # Flatten the axes we want to normalize,   Combine the specified axes
    flattened_shape = (-1,) + transposed_arr.shape[len(axes):]
    flattened_arr = transposed_arr.reshape(flattened_shape)

    # Step 2: Apply z-score along the first dimension
    # (flattened data from the specified axes)
    # Normalize along the first axis (ravelled axes)
    normalized_flattened = stats.zscore(flattened_arr, axis=0, nan_policy='omit')

    # Step 3: Reshape back to the original shape
    reshaped_arr = normalized_flattened.reshape(transposed_arr.shape)

    # Transpose the array back to the original order
    inverse_axes = np.argsort(axes + other_axes)  # Inverse the axes order for the final transposition
    final_arr = np.transpose(reshaped_arr, axes=inverse_axes)

    return final_arr

def get_streaks(arr):
    """helper function to get indices of streaks automatically
    i.e. [1,2,3,4,8,9] -> [[1,4], [8,9]]
    transform dict_values to list and then to array.

    returns: min and max of the array across all dimensions"""
    if len(arr)==0:
        return np.array([])
    arr = np.unique([x for x in arr])
    streaks = np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
    streaks = [(s[0], s[-1]) for s in streaks]
    return np.array(streaks)


def list_files(
    path,
    exts=None,
    patterns=None,
    relative=False,
    subfolders=False,
    return_strings=True,
    only_folders=False,
    max_results=None,
):
    """
    will make a list of all files with extention exts (list)
    found in the path and possibly all subfolders and return
    a list of all files matching this pattern

    :param path:  location to find the files
    :type  path:  str
    :param exts:  extension of the files (e.g. .jpg, .jpg or .png, png)
                  Will be turned into a pattern internally
    :type  exts:  list or str
    :param pattern: A pattern that is supported by pathlib.Path,
                  e.g. '*.txt', '**\rfc_*.clf'
    :type:        str
    :param fullpath:  give the filenames with path
    :type  fullpath:  bool
    :param subfolders
    :param return_strings: return strings, else returns Path objects
    :return:      list of file names
    :type:        list of str
    """
    if isinstance(exts, str):
        exts = [exts]
    if isinstance(patterns, str):
        patterns = [patterns]
    assert isinstance(path, str), "path needs to be a str"
    assert os.path.exists(path), "Path {} does not exist".format(path)
    if patterns is None:
        patterns = []
    if exts is None:
        exts = []

    if patterns == [] and exts == []:
        patterns = ["*"]

    for ext in exts:
        ext = ext.replace("*", "")
        pattern = "*" + ext
        patterns.append(pattern.lower())

    # if recursiveness is asked, prepend the double asterix to each pattern
    if subfolders:
        patterns = ["**/" + pattern for pattern in patterns]

    # collect files for each pattern
    files = []
    fcount = 0
    for pattern in patterns:
        # pattern =
        for filename in Path(path).glob(pattern):
            if filename.is_file() and filename not in files:
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results <= fcount:
                    break

    # turn path into relative or absolute paths
    files = [file.relative_to(path) if relative else file.absolute() for file in files]

    # by default: return strings instead of Path objects
    if return_strings:
        files = [os.path.join(file) for file in files]

    return sorted(files)


def json_load(file):
    """load json file without context manager"""
    with open(file, "r") as f:
        c = json.load(f)
    return c


def json_dump(obj, file, *args, **kwargs):
    """write json file without context manager"""
    if not "indent" in kwargs:
        kwargs["indent"] = 4
    if not "cls" in kwargs:
        kwargs["cls"] = NumpyEncoder
    s = json.dumps(obj, *args, **kwargs)
    s = collapse_json(s)
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        f.write(s)


def collapse_json(text, indent=12):
    """Compacts a string of json data by collapsing whitespace after the
    specified indent level

    NOTE: will not produce correct results when indent level is not a multiple
    of the json indent level
    """
    initial = " " * indent
    out = []  # final json output
    sublevel = []  # accumulation list for sublevel entries
    pending = None  # holder for consecutive entries at exact indent level
    for line in text.splitlines():
        if line.startswith(initial):
            if line[indent] == " ":
                # found a line indented further than the indent level, so add
                # it to the sublevel list
                if pending:
                    # the first item in the sublevel will be the pending item
                    # that was the previous line in the json
                    sublevel.append(pending)
                    pending = None
                item = line.strip()
                sublevel.append(item)
                if item.endswith(","):
                    sublevel.append(" ")
            elif sublevel:
                # found a line at the exact indent level *and* we have sublevel
                # items. This means the sublevel items have come to an end
                sublevel.append(line.strip())
                out.append("".join(sublevel))
                sublevel = []
            else:
                # found a line at the exact indent level but no items indented
                # further, so possibly start a new sub-level
                if pending:
                    # if there is already a pending item, it means that
                    # consecutive entries in the json had the exact same
                    # indentation and that last pending item was not the start
                    # of a new sublevel.
                    out.append(pending)
                pending = line.rstrip()
        else:
            if pending:
                # it's possible that an item will be pending but not added to
                # the output yet, so make sure it's not forgotten.
                out.append(pending)
                pending = None
            if sublevel:
                out.append("".join(sublevel))
            out.append(line)
    return "\n".join(out)


def _get_source_file(frame):
    try:
        f_locals = getattr(frame, "f_locals", None)
        f_code = getattr(frame, "f_code", None)
        if f_locals is not None:
            yield f_locals.get("__file__", None)
        if f_code is not None:
            yield getattr(f_code, "co_filename", None)
    except Exception:
        pass

    yield "undefined"


def savefig(fig, filename, tight=True, despine=True):
    """calls plt.pause, tight_figure and then saves to plot dir"""
    plt.pause(0.1)
    if despine:
        sns.despine(fig)
    if tight:
        fig.tight_layout()
    plt.pause(0.1)
    if settings.plot_dir in filename:
        raise ValueError(f'{filename} does not contain the plot dir')
    out_dir = f'{settings.plot_dir}/{os.path.dirname(filename)}'
    os.makedirs(out_dir, exist_ok=True)
    if not filename.endswith(('png', 'jpg', 'svg', 'eps')):
        filename = filename + '.png'
    fig.savefig(f'{out_dir}/{os.path.basename(filename)}')

def log_fig(
    fig, filename, parameters=None, uid=None, logdir=settings.log_dir, tight_layout=True
):

    if uid is None:
        uid = hex(random.getrandbits(128))[2:10]

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    basename, ext = os.path.splitext(filename)
    base_filename = valid_filename(os.path.basename(basename))

    log_file = f"{logdir}/{base_filename}_{uid}.json"
    png_file = f"{basename}_{uid}{ext}"
    fig.tight_layout()
    plt.pause(0.01)
    log_parameters(log_file, parameters=parameters, minlevel=2)
    fig.savefig(png_file, bbox_inches="tight")
    return png_file


def log_parameters(log_file, parameters=None, minlevel=1, maxlevel=99):
    """function to dump parameters as json, additionally adds source code of caller"""
    assert minlevel > 0, "minlevel must be larger than 1"

    def code2string(var, truncate=None):
        """return input as string, truncate the string if larger than 'truncate'"""
        if callable(var):
            try:
                file = inspect.getsourcefile(var)
                if "site-packages" in file:
                    # don't include module code
                    string = repr(var)
                else:
                    string = f"#{var} \n{inspect.getsource(var)}"
            except Exception as e:
                string = str(e)
        else:
            string = str(var)
            if truncate is not None and len(string) > truncate:
                string = string[:truncate] + "[...]"
        return string

    if parameters is None:
        parameters = {}
    parameters = parameters.copy()

    # remove unnecessary parameters
    for key in list(parameters):
        if key.startswith("_"):
            del parameters[key]

    upper_frame = inspect.currentframe()
    for _ in range(minlevel):
        upper_frame = upper_frame.f_back

    sources = {}
    for level in range(1, maxlevel):

        if upper_frame is None:  # that means we reached the top stack
            break

        # get source file of the caller frame
        get_source_file = _get_source_file(upper_frame)
        while (source_file := next(get_source_file)) is None:
            pass

        if "site-packages" in source_file or "/lib/" in source_file:
            break  # that means we have reached interactive-shell level

        if source_file and os.path.exists(source_file):
            with open(source_file, "r") as f:
                source = f.read()
            source = f"# file: {source_file}\n" + source
        else:
            source = f"# [level {level}] SOURCE `{source_file}` NOT FOUND"

        workspace = {}
        for name, var in list(upper_frame.f_locals.items()):
            # remove built-in functions and hidden vars
            if "builtin_function_or_method" in str(type(var)):
                continue
            if isinstance(var, (ModuleType, type)):
                continue
            if name.startswith("_"):
                continue
            workspace[name] = var

        sources[f"level {level}"] = {"code": source, "locals": workspace}
        upper_frame = upper_frame.f_back

    f_trunc = lambda x: code2string(x, truncate=1000)
    f_full = lambda x: code2string(x, truncate=None)

    parameters["date" if "date" in parameters else "_date"] = str(
        datetime.datetime.now()
    )
    parameters["np.random.state"] = np.random.get_state()

    try:
        sources_str = collapse_json(
            json.dumps(sources, indent=2, default=f_trunc, ensure_ascii=False)
        )
        param_str = collapse_json(
            json.dumps(parameters, indent=4, default=f_full, ensure_ascii=False)
        )
    except Exception as e:
        print("cant get source:", e)
        sources_str = "ERROR"
        param_str = "ERROR"

    with open(log_file, "w") as f:
        f.write(f'["parameters":{param_str},\n"sources":{sources_str}]')
    print(f"Saved configuration to {log_file}")
    return log_file


def get_transfer_heatmap(clf, data_x, data_y, test_x, test_y, range_t=None):
    """

    create a heatmap of decoding by varying training and testing time of
    two independent samples
    """
    heatmap = np.zeros([data_x.shape[-1], test_x.shape[-1]])
    for t_train in range(data_x.shape[-1]):
        clf.fit(data_x[:, :, t_train], data_y)
        for t_test in range(test_x.shape[-1]):
            acc = (clf.predict(test_x[:, :, t_test]) == test_y).mean()
            heatmap[t_train, t_test] = acc
    return heatmap


def get_decoding_heatmap(clf, data_x, data_y, ex_per_fold=4, n_jobs=8, range_t=None):
    """

    using cross validation, create a heatmap of decoding by varying training
    and testing times

    :param clf: classifier to use for creation of the heatmap
    :param data_x: data to train on
    :param data_y: data to test on
    :param ex_per_fold: DESCRIPTION, defaults to 4
    :param n_jobs: DESCRIPTION, defaults to 8
    :param range_t: DESCRIPTION, defaults to None
    :return: DESCRIPTION
    :rtype: np.array of

    """
    assert (
        len(set(np.bincount(data_y)).difference(set([0]))) == 1
    ), "WARNING not each class has the same number of examples"
    np.random.seed(0)
    labels = np.unique(data_y)
    idxs_tuples = np.array([np.where(data_y == cond)[0] for cond in labels]).T
    idxs_tuples = [
        idxs_tuples[i : i + ex_per_fold].ravel()
        for i in range(0, len(idxs_tuples), ex_per_fold)
    ]

    if range_t is None:
        range_t = np.arange(data_x.shape[-1])

    tqdm_total = len(idxs_tuples) * (len(range_t) ** 2)
    res = np.zeros([len(idxs_tuples), len(range_t), len(range_t)])

    for i, idxs in enumerate(idxs_tuples):
        idxs_train = ~np.in1d(range(data_x.shape[0]), idxs)
        idxs_test = np.in1d(range(data_x.shape[0]), idxs)
        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]
        test_x = data_x[idxs_test]
        test_y = data_y[idxs_test]
        params = list(itertools.product(range_t, range_t))
        tqdm_initial = i * len(range_t) ** 2
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_predict)(
                train_x[:, :, train_at],
                train_y,
                test_x[:, :, predict_at],
                clf=clf,
                ova=False,
            )
            for train_at, predict_at in tqdm(
                params, total=tqdm_total, initial=tqdm_initial
            )
        )
        accs = np.mean(np.array(results) == test_y, -1)
        res[i, :, :] = accs.reshape([len(range_t), len(range_t)])
    return res.mean(0).squeeze()


def get_decoding_accuracy(subj, clf=settings.default_clf, n_splits=10):
    """wrapper to allow pickling of clf"""
    return _get_decoding_accuracy(subj, clf=clf, clf_str=str(clf), n_splits=n_splits)


@memory.cache(ignore=["clf"])
def _get_decoding_accuracy(subj, clf, clf_str, n_splits, data=None):
    """
    given a classifier, performs a 10-fold cross validation across timepoints
    and returns the best timepoint and its decoding performance

    this function is used to set a criteria for inclusion of a participant
    """
    from load_funcs import load_localizers_seq12
    np.random.seed(get_id(subj))
    if data is None:
        data_x, data_y = load_localizers_seq12(subj, bands=settings.default_bands)
    else:
        data_x, data_y = data

    tps = range(20, 44)
    scores = Parallel(n_jobs=len(tps))(
        delayed(_get_score)(clf, data_x, data_y, tp, n_splits=n_splits) for tp in tps
    )

    return max(scores), tps[np.argmax(scores)]


def _get_score(clf, data_x, data_y, tp, n_splits=5):
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits, shuffle=True)
    scores = []

    for idx_train, idx_test in cv.split(data_x, data_y):
        train_x = data_x[idx_train, :, tp]
        train_y = data_y[idx_train]
        test_x = data_x[idx_test, :, tp]
        test_y = data_y[idx_test]

        clf.fit(train_x, train_y)
        scores.append(clf.score(test_x, test_y))

    return np.mean(scores)


def get_image_names(subj):
    """get the order of images for a participant"""
    idx = get_id(subj) - 100
    seq_file = f"./data/category_graph/sequence_participant_{idx}.csv"
    with open(seq_file, "r") as f:
        c = f.read().strip()
    names = [x.replace(".png", "") for x in c.split(",")]
    return names

def normalize_lims(axs, which='both'):
    """for all axes in axs: set function to min/max of all axs


    Parameters
    ----------
    axs : list
        list of axes to normalize.
    which : string, optional
        Which axis to normalize. Can be 'x', 'y', 'xy' oder 'both'.

    """
    if which=='both':
        which='xy'
    for w in which:
        ylims = [getattr(ax, f'get_{w}lim')() for ax in axs]
        ymin = min([x[0] for x in ylims])
        ymax = max([x[1] for x in ylims])
        for ax in axs:
            getattr(ax, f'set_{w}lim')([ymin, ymax])

def make_fig(
    n_axs=30,
    bottom_plots=2,
    no_ticks=False,
    suptitle="",
    xlabel="Lag in ms",
    ylabel="Sequenceness",
    figsize=None,
    despine=True,
):
    """
    helper function to create a grid space with RxC rows and a
    large row with two axis on the bottom

    returns: fig, axs(size=(rows*columns)), ax_left_bottom, ax_right_bottom
    """

    COL_MULT = 10  # to accomodate also too large axis
    # some heuristic for finding optimal rows and columns
    for columns in [2, 4, 6, 8]:
        rows = np.ceil(n_axs / columns).astype(int)
        if columns >= rows:
            break
    assert columns * rows >= n_axs

    if isinstance(bottom_plots, int):
        bottom_plots = [1 for _ in range(bottom_plots)]
    n_bottom = len(bottom_plots)
    COL_MULT = 1
    if n_bottom > 0:
        for COL_MULT in range(1, 12):
            if (columns * COL_MULT) % n_bottom == 0:
                break
        if not (columns * COL_MULT) % n_bottom == 0:
            warnings.warn(
                f"{columns} cols cannot be evenly divided by {bottom_plots} bottom plots"
            )
    fig = plt.figure(dpi=75, constrained_layout=True, figsize=figsize)
    # assuming maximum 30 participants
    gs = fig.add_gridspec(
        (rows + 2 * (n_bottom > 0)), columns * COL_MULT
    )  # two more for larger summary plots
    axs = []

    # first the individual plot axis for each participant
    for x in range(rows):
        for y in range(columns):
            ax = fig.add_subplot(gs[x, y * COL_MULT : (y + 1) * COL_MULT])
            if no_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            axs.append(ax)
            if len(axs)>=n_axs:
                break
        if len(axs)>=n_axs:
            break
    fig.suptitle(suptitle)

    if len(bottom_plots) == 0:
        return fig, axs

    # second the two graphs with all data combined/meaned
    axs_bottom = []
    step = np.ceil(columns * COL_MULT // n_bottom).astype(int)
    for b, i in enumerate(range(0, columns * COL_MULT, step)):
        if bottom_plots[b] == 0:
            continue  # do not draw* this plot
        ax_bottom = fig.add_subplot(gs[rows:, i : (i + step)])
        if xlabel:
            ax_bottom.set_xlabel(xlabel)
        if ylabel:
            ax_bottom.set_ylabel(ylabel)
        if i > 0 and no_ticks:  # remove yticks on righter plots
            ax_bottom.set_yticks([])
        axs_bottom.append(ax_bottom)
    if despine:
        sns.despine(fig)
    return fig, axs, *axs_bottom


def get_scores(clf, X, method="auto"):
    """
    get either decision function or predict_proba output, depending on what
    is available. Priority is given to decision functio
    """
    if X.ndim == 3:
        warnings.warn(
            "3D array found, assuming first dimension to be trials, second to be time and third to be features"
        )
        arr = [get_scores(clf, _X, method) for _X in X]
        return np.array(arr)
    if method == "auto":
        if hasattr(clf, "decision_function"):
            return clf.decision_function(X)
        elif hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        else:
            raise NameError(f"{clf} has neither decision_function nor predict_proba")

    elif method == "predict_proba_axis0":
        scores = clf.predict_proba(X)
        scores = scores / scores.max(0)
        return scores
    elif method == "decision_function_axis0":
        scores = clf.decision_function(X)
        scores = scores / scores.max(0)
        return scores
    else:
        try:
            return clf.__getattribute__(method)(X)
        except KeyError:
            return type(clf).__getattribute__(method)(clf, X)


def char2num(seq):
    if isinstance(seq, str):
        seq = list(seq.strip())
    assert ord("A") - 65 == 0
    nums = [ord(c.upper().strip()) - 65 for c in seq]
    assert all([0 <= n <= 90 for n in nums])
    return nums


def num2char(arr):
    if isinstance(arr, int):
        return chr(arr + 65)
    arr = np.array(arr, dtype=int)
    return np.array([chr(x + 65) for x in arr.ravel()]).reshape(*arr.shape)


def seq2TF(sequence, nstates=None):
    """
    create a transition matrix from a sequence string,
    e.g. ABCDEFG
    Please note that sequences will not be wrapping automatically,
    i.e. a wrapping sequence should be denoted by appending the first state.

    :param sequence: sequence in format "ABCD..."
    :param seqlen: if not all states are part of the sequence,
                   the number of states can be specified
                   e.g. if the sequence is ABE, but there are also states F,G
                   nstates would be 7

    """

    seq = char2num(sequence)
    if nstates is None:
        nstates = max(seq) + 1
    # assert max(seq)+1==nstates, 'not all positions have a transition'
    TF = np.zeros([nstates, nstates], dtype=int)
    for i, p1 in enumerate(seq):
        if i + 1 >= len(seq):
            continue
        p2 = seq[(i + 1) % len(seq)]
        TF[p1, p2] = 1
    return TF.astype(float)


@memory.cache(ignore=["n_jobs", "plot_confmat", "title_add", "verbose"])
def get_best_timepoint(
    data_x,
    data_y,
    add_null_data=False,
    n_jobs=-2,
    plot_confmat=False,
    title_add="",
    verbose=True,
    ex_per_fold=2,
    simulate=False,
    clf=settings.default_clf,
    ova=False,
    subj="",
    ms_per_point=10,
    return_preds=False,
):
    assert (
        len(set(np.bincount(data_y)).difference(set([0]))) == 1
    ), "WARNING not each class has the same number of examples"
    np.random.seed(get_id(subj))
    labels = np.unique(data_y)
    idxs_tuples = np.array([np.where(data_y == cond)[0] for cond in labels]).T
    idxs_tuples = [
        idxs_tuples[i : i + ex_per_fold].ravel()
        for i in range(0, len(idxs_tuples), ex_per_fold)
    ]

    time_max = data_x.shape[-1]  # 500 ms
    total = len(idxs_tuples)
    tqdm_loop = tqdm(total=total, desc=f"CV Fold {subj}", disable=not verbose)
    df = pd.DataFrame()

    all_preds = np.zeros([time_max, len(data_y)], dtype=int)

    for j, idxs in enumerate(idxs_tuples):
        idxs_train = ~np.in1d(range(data_x.shape[0]), idxs)
        idxs_test = np.in1d(range(data_x.shape[0]), idxs)
        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]
        test_x = data_x[idxs_test]
        test_y = data_y[idxs_test]

        neg_x = np.hstack(train_x[:, :, 0:1].T).T if add_null_data else None
        preds = Parallel(n_jobs=n_jobs)(
            delayed(train_predict)(
                train_x=train_x[:, :, start],
                train_y=train_y,
                test_x=test_x[:, :, start],
                neg_x=neg_x,
                clf=clf,
                ova=ova,
            )
            for start in list(range(0, time_max))
        )

        all_preds[:, idxs_test] = np.array(preds)
        acc_hit_miss = np.array([pred == test_y for pred in preds]).mean(-1)

        df_temp = pd.DataFrame(
            {
                "timepoint": np.arange(
                    -100, len(acc_hit_miss) * ms_per_point - 100, ms_per_point
                ),
                "fold": [j] * len(acc_hit_miss),
                "accuracy": acc_hit_miss,
                "preds": preds,
                "subject": [subj] * len(acc_hit_miss),
            }
        )
        df = pd.concat([df, df_temp], ignore_index=True)
        tqdm_loop.update()
    tqdm_loop.close()
    return (df, all_preds) if return_preds else df


def train_predict(
    train_x, train_y, test_x, clf=None, neg_x=None, proba=False, ova=True
):
    """
    Train a classifier with the given data and return predictions on test_x
    """
    assert not ova, "OVA is not implemented any longer"
    assert test_x.ndim == 2, "test data must be 2d"
    if neg_x is None:
        clf.fit(train_x, train_y)
    else:
        clf.fit(train_x, train_y, neg_x=neg_x)
    pred = clf.predict_proba(test_x) if proba else clf.predict(test_x)
    return pred


def get_sparsity(clf):
    """calculate overall sparsity"""
    assert isinstance(clf, BaseEstimator), "must be sklearn.classifier"
    if not hasattr(clf, "coef_"):
        return np.nan
    sparse_per_class = (np.abs(clf.coef_) < 0.000001).mean(1)
    sparsity = np.mean(sparse_per_class)
    return sparsity


def get_sensor_correlation(clf, axis=None):
    assert isinstance(clf, BaseEstimator), "must be sklearn.classifier"
    if not hasattr(clf, "coef_"):
        return np.nan
    try:
        corrcoef = np.abs(np.corrcoef(clf.coef_))
        corrcoef[np.eye(len(corrcoef)) == 1] = np.nan
        return np.nanmean(corrcoef, axis)
    except:
        return np.nan


class LogisticRegressionOvaNegX(LogisticRegression):
    """one vs all logistic regression classifier including negative examples.

    Under the hood, one separate LogisticRegression is trained per class.
    The LogReg is trained using positive examples (inclass) and negative
    examples (outclass + nullclass).
    """

    def __init__(
        self,
        penalty="l1",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        neg_x_ratio=1.0,
    ):

        self.neg_x_ratio = neg_x_ratio
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        # self.n_pca = n_pca
        LogisticRegression.__init__(
            self,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class="ovr",
        )


    def fit(self, X, y, neg_x=None, neg_x_ratio=None):
        # if self.n_pca is not None:
        #     self.pca = PCA(self.n_pca)
        #     X = self.pca.fit_transform(X)
        #     neg_x = self.pca.transform(neg_x)
        self.classes_ = np.unique(y)
        neg_x_ratio = self.neg_x_ratio if neg_x_ratio is None else neg_x_ratio
        models = []
        intercepts = []
        coefs = []

        for class_ in self.classes_:
            clf = LogisticRegression(penalty = self.penalty, C= self.C,
                                     solver=self.solver, max_iter=self.max_iter)
            idx_class = y == class_
            true_x = X[idx_class]
            false_x = X[~idx_class]

            if neg_x is not None:
                n_null = int(len(X) * neg_x_ratio)
                replace = len(neg_x) < n_null
                idx_neg = np.random.choice(len(neg_x), size=n_null, replace=replace)
                false_x = np.vstack([false_x, neg_x[idx_neg]])

            data_x = np.vstack([true_x, false_x])
            data_y = np.hstack([np.ones(len(true_x)), np.zeros(len(false_x))])

            clf.fit(data_x, data_y)
            models.append(clf)
            intercepts.append(clf.intercept_)
            coefs.append(clf.coef_)

        self.models = models
        self.intercept_ = np.squeeze(intercepts)
        self.coef_ = np.squeeze(coefs)

        return self

    def predict_proba(self, X):
        # if self.n_pca is not None:
        #     X = self.pca.transform(X)
        proba = []
        for clf in self.models:
            p = clf.predict_proba(X)[:, 1]
            proba.append(p)
        return np.array(proba).T


def plot_sf_sb(
    sf,
    sb,
    cTime=None,
    ax=None,
    title=None,
    color=None,
    which=["fwd-bkw", "fwd", "bkw"],
    clear=True,
    rescale=True,
    plot95=True,
    plotmax=True,
    despine=True,
):
    if despine:
        sns.despine()

    def shadedErrorBar(x, y, err, ax=None, **kwargs):
        ax.plot(x, y, **kwargs)
        ax.fill_between(x, y - err, y + err, alpha=0.35, label="_nolegend_", **kwargs)

    sf = np.array(sf, copy=True)
    sb = np.array(sb, copy=True)
    if sf.ndim == 2:
        sf = sf.reshape([1, *sf.shape])
    if sb.ndim == 2:
        sb = sb.reshape([1, *sb.shape])

    sf = np.nan_to_num(sf)
    sb = np.nan_to_num(sb)

    if cTime is None:
        cTime = np.arange(0, sf.shape[-1] * 10, 10)  # just assume sampling frequency

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if clear:
        ax.clear()

    palette = sns.color_palette()

    # First plot fwd-bkw
    div = 1
    if "fwd-bkw" in which:
        c = palette[0] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:, 1:, 1:] - sb[:, 1:, 1:], 0)), -1)
        npThreshMax = max(npThresh)
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95) / div
        dtp = (sf[:, 0, :] - sb[:, 0, :]) / div
        shadedErrorBar(
            cTime, dtp.mean(0), np.std(dtp, 0) / np.sqrt(len(sf)), ax=ax, color=c
        )

    # Now plot fwd
    if "fwd" in which:
        c = palette[1] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:, 1:, 1:], 0)), -1)
        npThreshMax = max(npThresh)
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95) / div
        dtp = sf[:, 0, :] / div
        shadedErrorBar(
            cTime, dtp.mean(0), np.std(dtp, 0) / np.sqrt(len(sf)), ax=ax, color=c
        )

    # now plot bkw
    if "bkw" in which:
        c = palette[2] if color is None else color
        npThresh = np.max(abs(np.mean(sb[:, 1:, 1:], 0)), -1)
        npThreshMax = max(npThresh)
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95) / div
        dtp = sb[:, 0, :] / div
        shadedErrorBar(
            cTime, dtp.mean(0), np.std(dtp, 0) / np.sqrt(len(sb)), ax=ax, color=c
        )

    div = 1 if rescale else npThreshMax
    if plotmax:
        ax.hlines(
            -div,
            cTime[0],
            cTime[-1],
            linestyles="--",
            color="black",
            linewidth=1.5,
            alpha=0.6,
        )
        ax.hlines(
            div,
            cTime[0],
            cTime[-1],
            linestyles="--",
            color="black",
            linewidth=1.5,
            alpha=0.6,
        )
    if plot95:
        ax.hlines(
            npThresh95,
            cTime[0],
            cTime[-1],
            linestyles="--",
            color="black",
            linewidth=1,
            alpha=0.4,
        )
        ax.hlines(
            -npThresh95,
            cTime[0],
            cTime[-1],
            linestyles="--",
            color="black",
            linewidth=1,
            alpha=0.4,
        )
    ax.legend(which, loc="upper right")
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("sequenceness")
    ax.set_ylim(-div * 1.5, div * 1.5)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(cTime[::5])
    ax.set_xticks(cTime[::5], minor=True)
    ax.grid(axis="x", linewidth=1, which="both", alpha=0.3)

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def plot_decoding_accuracy(
    data,
    x="timepoint",
    y="accuracy",
    chance=None,
    title=None,
    ax=None,
    clear=True,
    max_acc=0.6,
    **kwargs,
):

    if ax is None:
        plt.figure(figsize=[7, 7], constrained_layout=False, maximize=False)
        ax = plt.gca()

    if chance is None:
        try:
            chance = 1 / len(np.unique(np.hstack(data["preds"])))
        except Exception:
            chance = 1 / 10
            warnings.warn("data had no preds, simply assuming chance level is 1/10")

    n_points = len(np.unique(data["timepoint"]))
    if clear:
        ax.clear()
    # err_kws = {'label':'_nolegend_'}
    sns.lineplot(data=data, x=x, y=y, ax=ax, **kwargs)
    xlims = ax.get_xlim()
    ax.hlines(chance, *xlims, linestyles="dotted", color="gray")
    ax.legend(
        ["Decoding accuracy", "95% conf", "Chance"], prop={"size": 8}, framealpha=0.5
    )
    ax.set_xlabel("Time after stimulus onset (ms)")
    ax.set_xticks(np.arange(-100, n_points * 10, 50), minor=True)  # plot each 50ms
    ax.set_xticks(np.arange(-100, n_points * 10, 100))  # plot each 100ms
    ax.set_yticks(np.arange(0, int(max_acc * 10)) / 10)  # plot percentage ticks
    ax.set_ylim([0, max_acc])
    ax.set_xlim(xlims)
    ax.grid(axis="x", linewidth=1, which="both", alpha=0.3)
    ax.grid(axis="y", linewidth=1, which="major", alpha=0.3)
    max_acc = data.groupby(x).mean(True)[y].max()
    max_tp = data.groupby(x).mean(True)[y].argmax()
    best_tp_ms = data[x].unique()[max_tp]
    n_subj = len(set(data["subject"])) if "subject" in data else "?"
    if title is None:
        title = f"Average of n={n_subj}, LOSO max {max_acc:.2f} @ {best_tp_ms}ms"
    ax.set_title(title)


def plot_permutation_distribution(sx, ax=None, title=None, **kwargs):
    """plots the means of a TDLM permutation results.

    plots a histogram of mean permutation sequenceness values across the
    time lags. Indicates the base value (permutation index 0) in red.
    Calculates the p value.

    returns: pval, axis_handle
    """
    # Calculate mean across values for each subject
    mean_subjects = np.nanmean(sx, axis=(2,))

    # Calculate mean of means
    mean_of_means = np.nanmean(mean_subjects, axis=0)

    # Plot histogram
    bins = np.histogram_bin_edges(mean_of_means, bins=50)

    ax = sns.histplot(
        mean_of_means, bins=bins, alpha=kwargs.pop('alpha', 0.5), ax=ax,
        stat="count", **kwargs
    )

    # Find the bin that the first permutation falls into
    bin_index = np.searchsorted(bins, mean_of_means[0])

    # Highlight the bin with red color
    axmin = bins[bin_index]
    axmax = bins[bin_index + 1] if (bin_index+1)<len(bins) else axmin*1.02
    ax.axvspan(axmin, axmax, color="red", alpha=0.5)
    p = np.mean(np.abs(mean_of_means[0]) < np.abs(mean_of_means[1:]))

    # Add labels and title
    ax.set_xlabel("Mean sequenceness of permutation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    # ax.text(ax.get_xlim()[1]*0.97, ax.get_ylim()[1]*0.95, f'{p=:.3f}', horizontalalignment='right')
    ax.legend([f"observed\n{p=:.3f}"], fontsize=12, loc="upper left")
    return ax, p


def highlight_cells(mask, ax, color='r', linewidth=1, linestyle='solid'):
    """
    Draws borders around the true entries of the mask array on a heatmap plot.

    Parameters:
    - mask (np.ndarray): A 2D binary mask array where True (or 1) entries indicate
                         the cells that should be highlighted with a border.
    - ax (matplotlib.axes.Axes): The axes on which the heatmap is plotted.
    - color (str): Color of the border lines.
    - linewidth (float): Width of the border lines.
    - linestyle (str): Line style of the border lines.
    """
    # Ensure the mask is a 2D array
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D array.")

    # Check if the mask dimensions match the plotted image dimensions
    image_shape = ax.images[0].get_array().shape
    if mask.shape != image_shape:
        raise ValueError(f"Mask dimensions {mask.shape} do not match the plotted image dimensions {image_shape}.")

    # Function to check if a cell is outside the mask or out of bounds
    def is_outside_mask(i, j, mask):
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return True
        return not mask[i, j]

    # Loop through each cell in the mask to draw borders around masked regions
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                # Draw top border if the cell above is outside the mask
                if is_outside_mask(i - 1, j, mask):
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw bottom border if the cell below is outside the mask
                if is_outside_mask(i + 1, j, mask):
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw left border if the cell to the left is outside the mask
                if is_outside_mask(i, j - 1, mask):
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw right border if the cell to the right is outside the mask
                if is_outside_mask(i, j + 1, mask):
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
