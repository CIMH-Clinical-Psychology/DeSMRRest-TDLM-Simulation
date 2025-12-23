# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:08:58 2024

@author: simon.kern
"""
import os
import warnings
import getpass
import platform

import numpy as np
from sklearn.linear_model import LogisticRegression


def rescale_meg_transform_outlier(arr):
    """
    same as rescale_meg, but also removes all values that are above [-1, 1]
    and rescales them to smaller values
    """

    arr = rescale_meg(arr)

    arr[arr < -1] *= 1e-2
    arr[arr > 1] *= 1e-2
    return arr


def rescale_meg(arr):
    """
    this tries to statically re-scale the values from Tesla to Nano-Tesla,
    such that most sensor values are between -1 and 1

    If possible, individual scaling is applied to magnetometers and
    gradiometers as both sensor types have a different sensitivity and scaling.

    Basically a histogram normalization between the two sensor types

    gradiometers  = *1e10
    magnetometers = *2e11
    """

    # some sanity check, if these
    if arr.min() < -1e-6 or arr.max() > 1e-6:
        warnings.warn(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
        raise Exception(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
    arr = np.array(arr)
    grad_scale = 1e10
    mag_scale = 2e11

    # reshape to 3d to make indexing uniform for all types
    # will be put in its original shape later
    orig_shape = arr.shape
    arr = np.atleast_3d(arr)

    # heuristic to find which dimension is likely the sensor dimension
    for meg_type in [306, 204, 102]:  # mag+grad or grad or mag
        dims = [d for d, size in enumerate(arr.shape) if size % meg_type == 0]
        # how many copies do we have of the sensors?
        stacks = [
            size // meg_type for d, size in enumerate(arr.shape) if size % meg_type == 0
        ]
        if len(dims) > 0:
            break

    if len(dims) != 1:
        warnings.warn(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        raise Exception(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        return arr.reshape(*orig_shape) * grad_scale
    sensor_dim = dims[0]
    n_stack = stacks[0]

    if meg_type == 306:
        slicer_grad = [slice(None) for _ in range(3)]
        slicer_grad[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_grad for i in range(n_stack)]
        )
        arr[tuple(slicer_grad)] *= grad_scale
        slicer_mag = [slice(None) for _ in range(3)]
        slicer_mag[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_mag for i in range(n_stack)]
        )
        arr[tuple(slicer_mag)] *= mag_scale

    if meg_type == 204:
        arr *= grad_scale

    if meg_type == 102:
        arr *= mag_scale

    return arr.reshape(*orig_shape)


def get_free_space(path):
    """return the current free space in the cache dir in GB"""
    import shutil

    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    total //= 1024**3
    used //= 1024**3
    free //= 1024**3
    return free

###############################
#%%userconf
# USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')

curr_dir = os.path.dirname(__file__)

# machine specific configuration overwrites general directory structure
if username == 'simon.kern' and host=='5cd320lfh8':
    cache_dir = f'{home}/Desktop/joblib-resting-state/'
    data_dir = "W:/group_klips/data/data/Simon/DeSMRRest/upload/"
elif username == 'simon.kern':  # any other VM
    # if SLURM_JOB_ID:
        # cache_dir = f'/data/{SLURM_JOB_ID}/joblib-resting-state/'
    # else:
    cache_dir = f'{home}/joblib-resting-state/'
    data_dir = '/zi/flstorage/group_klips/data/data/Simon/DeSMRRest/upload/'
    plot_dir = f'{curr_dir}/plots/'

elif username=='simon' and host=='kubuntu':
    data_dir = '.'
elif username == 'simon' and host in ('thinkpad-simon', 'desktop-dakomj2'):
    cache_dir = f'z:/joblib-simulation/'
    data_dir = "z:"
else:
    warnings.warn(f'No user specific settings found in settings.py for {username=} {host=}')

if 'data_dir' not in locals():
    raise Exception('No data_dir in settings.py!')

#%% checks for stuff
if 'cache_dir' not in locals():
    cache_dir = f"{data_dir}/cache/"  # used for caching
if 'plot_dir' not in locals():
    plot_dir = f"{data_dir}/plots/"  # plots will be stored here
if 'log_dir' not in locals():
    log_dir = f"{data_dir}/plots/logs/"  # log files will be created here

os.environ['JOBLIB_CACHE_DIR'] = cache_dir

results_dir = os.path.expanduser(f"{data_dir}/results/")  # final results here

if data_dir == "":
    raise Exception(f"please set configuration in settings.py")

if not os.path.isdir(data_dir):
    warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
    os.makedirs(plot_dir, exist_ok=True)
if not os.path.isdir(plot_dir):
    warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
    os.makedirs(plot_dir, exist_ok=True)
if not os.path.isdir(log_dir):
    warnings.warn(f"log_dir does not exist at {log_dir}, create")
    # os.makedirs(log_dir, exist_ok=True)
if not os.path.isdir(results_dir):
    warnings.warn(f"log_dir does not exist at {log_dir}, create")
    # os.makedirs(results_dir, exist_ok=True)

if get_free_space(cache_dir) < 20:
    raise RuntimeError(f"Free space for {cache_dir} is below 20GB. Cannot safely run.")

os.environ['JOBLIB_CACHEDIR'] = os.environ.get('JOBLIB_CACHEDIR', cache_dir)

###############################
#%% SETTINGS and CONSTANTS
###############################

bands_delta = {"delta": (0, 4)}
bands_theta = {"theta": (4, 8)}
bands_alpha = {"alpha": (8, 14)}
bands_beta = {"beta": (15, 30)}
bands_gamma = {"gamma": (30, 45)}

# some default brain band definitions
bands_all = {**bands_delta, **bands_theta, **bands_alpha, **bands_beta, **bands_gamma}
bands_lower = {"lower": (0.5, 20)}
bands_HP = {"only_HP": (0.5, None)}
bands_none = {"none": (None, None)}

# corperate colour palette
zi_palette = [
    "#003e65",
    "#006960",
    "#70305a",
    "#c7361b",
    "#3a98cc",
    "#74ba59",
    "#e8326d",
    "#f7ab64",
    "#85cee4",
    "#bfffd7",
    "#d1bcdc",
    "#fcd8c1",
]


stim_translation = {
    'apfel': 'apple',
    'berg': 'mountain',
    'clown': 'clown',
    'fahrrad': 'bicycle',
    'fuß': 'foot',
    'kuchen': 'cake',
    'pinsel': 'brush',
    'schreibtisch': 'desk',
    'tasse': 'cup',
    'zebra': 'zebra'
}

# the sequences with loop included
seq_12 = "ABCDEFGEHIBJAB"

default_predict_function = "predict_proba"  # 'decision_function'

default_seq = seq_12
default_autoreject = True
default_ica_components = 50  # default used by Fungi
default_normalize = rescale_meg_transform_outlier
default_clf_params = {
    "C": 1 / 0.006,
    "max_iter": 1000,
    "penalty": "l1",
    "solver": "liblinear",
}
default_bands = bands_HP

# default classifier to use if non is specified
default_clf = LogisticRegression(**default_clf_params)

caching_enabled = True
timeshift_constant = np.mean(
    [
        1.000559286986059,  # this is the value that we
        1.000559769261213,  # have to multiply the timepoints
        1.0005582875825834,  # of the presentation log files
        1.0005608210420054,  # to get matching positions for the MEG
        1.0005594754801779,  # the numbers on the left
        1.0005585095859724,  # are the mismatched between
        1.0005591506251639,  # individual measurements
        1.0005578477318235,
        1.0005590747786206,
        1.0005578234309724,
        1.0005582714046664,
        1.0005581193610011,
        1.000557486504249,
        1.0005597991661357,
        1.000559275593335,
        1.0005591272826757,
        1.0005586249053116,
        1.0005589597532822,
    ]
)

# this is a lookup table that shows correspondence between
# presentation log file event codes and port codes
event_code_translation = {
    "RS1": 10,
    "RS2": 20,
    "RS1 end": 11,
    "RS2 end": 22,
    "fixation audio": 99,
    "fixation pre audio": 98,
}
event_code_translation.update({f"{x}": x for x in range(256)})

# here some static MEG definitions for Vectorview systems (ELECTRA/NeuroMAG)

idx_grad = np.array(
    [
        1,
        2,
        4,
        5,
        7,
        8,
        10,
        11,
        13,
        14,
        16,
        17,
        19,
        20,
        22,
        23,
        25,
        26,
        28,
        29,
        31,
        32,
        34,
        35,
        37,
        38,
        40,
        41,
        43,
        44,
        46,
        47,
        49,
        50,
        52,
        53,
        55,
        56,
        58,
        59,
        61,
        62,
        64,
        65,
        67,
        68,
        70,
        71,
        73,
        74,
        76,
        77,
        79,
        80,
        82,
        83,
        85,
        86,
        88,
        89,
        91,
        92,
        94,
        95,
        97,
        98,
        100,
        101,
        103,
        104,
        106,
        107,
        109,
        110,
        112,
        113,
        115,
        116,
        118,
        119,
        121,
        122,
        124,
        125,
        127,
        128,
        130,
        131,
        133,
        134,
        136,
        137,
        139,
        140,
        142,
        143,
        145,
        146,
        148,
        149,
        151,
        152,
        154,
        155,
        157,
        158,
        160,
        161,
        163,
        164,
        166,
        167,
        169,
        170,
        172,
        173,
        175,
        176,
        178,
        179,
        181,
        182,
        184,
        185,
        187,
        188,
        190,
        191,
        193,
        194,
        196,
        197,
        199,
        200,
        202,
        203,
        205,
        206,
        208,
        209,
        211,
        212,
        214,
        215,
        217,
        218,
        220,
        221,
        223,
        224,
        226,
        227,
        229,
        230,
        232,
        233,
        235,
        236,
        238,
        239,
        241,
        242,
        244,
        245,
        247,
        248,
        250,
        251,
        253,
        254,
        256,
        257,
        259,
        260,
        262,
        263,
        265,
        266,
        268,
        269,
        271,
        272,
        274,
        275,
        277,
        278,
        280,
        281,
        283,
        284,
        286,
        287,
        289,
        290,
        292,
        293,
        295,
        296,
        298,
        299,
        301,
        302,
        304,
        305,
    ]
)
idx_mag = np.array(
    [
        0,
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
        33,
        36,
        39,
        42,
        45,
        48,
        51,
        54,
        57,
        60,
        63,
        66,
        69,
        72,
        75,
        78,
        81,
        84,
        87,
        90,
        93,
        96,
        99,
        102,
        105,
        108,
        111,
        114,
        117,
        120,
        123,
        126,
        129,
        132,
        135,
        138,
        141,
        144,
        147,
        150,
        153,
        156,
        159,
        162,
        165,
        168,
        171,
        174,
        177,
        180,
        183,
        186,
        189,
        192,
        195,
        198,
        201,
        204,
        207,
        210,
        213,
        216,
        219,
        222,
        225,
        228,
        231,
        234,
        237,
        240,
        243,
        246,
        249,
        252,
        255,
        258,
        261,
        264,
        267,
        270,
        273,
        276,
        279,
        282,
        285,
        288,
        291,
        294,
        297,
        300,
        303,
    ]
)


ch_names =  ['MEG0111',
 'MEG0112',
 'MEG0113',
 'MEG0121',
 'MEG0122',
 'MEG0123',
 'MEG0131',
 'MEG0132',
 'MEG0133',
 'MEG0141',
 'MEG0142',
 'MEG0143',
 'MEG0211',
 'MEG0212',
 'MEG0213',
 'MEG0221',
 'MEG0222',
 'MEG0223',
 'MEG0231',
 'MEG0232',
 'MEG0233',
 'MEG0241',
 'MEG0242',
 'MEG0243',
 'MEG0311',
 'MEG0312',
 'MEG0313',
 'MEG0321',
 'MEG0322',
 'MEG0323',
 'MEG0331',
 'MEG0332',
 'MEG0333',
 'MEG0341',
 'MEG0342',
 'MEG0343',
 'MEG0411',
 'MEG0412',
 'MEG0413',
 'MEG0421',
 'MEG0422',
 'MEG0423',
 'MEG0431',
 'MEG0432',
 'MEG0433',
 'MEG0441',
 'MEG0442',
 'MEG0443',
 'MEG0511',
 'MEG0512',
 'MEG0513',
 'MEG0521',
 'MEG0522',
 'MEG0523',
 'MEG0531',
 'MEG0532',
 'MEG0533',
 'MEG0541',
 'MEG0542',
 'MEG0543',
 'MEG0611',
 'MEG0612',
 'MEG0613',
 'MEG0621',
 'MEG0622',
 'MEG0623',
 'MEG0631',
 'MEG0632',
 'MEG0633',
 'MEG0641',
 'MEG0642',
 'MEG0643',
 'MEG0711',
 'MEG0712',
 'MEG0713',
 'MEG0721',
 'MEG0722',
 'MEG0723',
 'MEG0731',
 'MEG0732',
 'MEG0733',
 'MEG0741',
 'MEG0742',
 'MEG0743',
 'MEG0811',
 'MEG0812',
 'MEG0813',
 'MEG0821',
 'MEG0822',
 'MEG0823',
 'MEG0911',
 'MEG0912',
 'MEG0913',
 'MEG0921',
 'MEG0922',
 'MEG0923',
 'MEG0931',
 'MEG0932',
 'MEG0933',
 'MEG0941',
 'MEG0942',
 'MEG0943',
 'MEG1011',
 'MEG1012',
 'MEG1013',
 'MEG1021',
 'MEG1022',
 'MEG1023',
 'MEG1031',
 'MEG1032',
 'MEG1033',
 'MEG1041',
 'MEG1042',
 'MEG1043',
 'MEG1111',
 'MEG1112',
 'MEG1113',
 'MEG1121',
 'MEG1122',
 'MEG1123',
 'MEG1131',
 'MEG1132',
 'MEG1133',
 'MEG1141',
 'MEG1142',
 'MEG1143',
 'MEG1211',
 'MEG1212',
 'MEG1213',
 'MEG1221',
 'MEG1222',
 'MEG1223',
 'MEG1231',
 'MEG1232',
 'MEG1233',
 'MEG1241',
 'MEG1242',
 'MEG1243',
 'MEG1311',
 'MEG1312',
 'MEG1313',
 'MEG1321',
 'MEG1322',
 'MEG1323',
 'MEG1331',
 'MEG1332',
 'MEG1333',
 'MEG1341',
 'MEG1342',
 'MEG1343',
 'MEG1411',
 'MEG1412',
 'MEG1413',
 'MEG1421',
 'MEG1422',
 'MEG1423',
 'MEG1431',
 'MEG1432',
 'MEG1433',
 'MEG1441',
 'MEG1442',
 'MEG1443',
 'MEG1511',
 'MEG1512',
 'MEG1513',
 'MEG1521',
 'MEG1522',
 'MEG1523',
 'MEG1531',
 'MEG1532',
 'MEG1533',
 'MEG1541',
 'MEG1542',
 'MEG1543',
 'MEG1611',
 'MEG1612',
 'MEG1613',
 'MEG1621',
 'MEG1622',
 'MEG1623',
 'MEG1631',
 'MEG1632',
 'MEG1633',
 'MEG1641',
 'MEG1642',
 'MEG1643',
 'MEG1711',
 'MEG1712',
 'MEG1713',
 'MEG1721',
 'MEG1722',
 'MEG1723',
 'MEG1731',
 'MEG1732',
 'MEG1733',
 'MEG1741',
 'MEG1742',
 'MEG1743',
 'MEG1811',
 'MEG1812',
 'MEG1813',
 'MEG1821',
 'MEG1822',
 'MEG1823',
 'MEG1831',
 'MEG1832',
 'MEG1833',
 'MEG1841',
 'MEG1842',
 'MEG1843',
 'MEG1911',
 'MEG1912',
 'MEG1913',
 'MEG1921',
 'MEG1922',
 'MEG1923',
 'MEG1931',
 'MEG1932',
 'MEG1933',
 'MEG1941',
 'MEG1942',
 'MEG1943',
 'MEG2011',
 'MEG2012',
 'MEG2013',
 'MEG2021',
 'MEG2022',
 'MEG2023',
 'MEG2031',
 'MEG2032',
 'MEG2033',
 'MEG2041',
 'MEG2042',
 'MEG2043',
 'MEG2111',
 'MEG2112',
 'MEG2113',
 'MEG2121',
 'MEG2122',
 'MEG2123',
 'MEG2131',
 'MEG2132',
 'MEG2133',
 'MEG2141',
 'MEG2142',
 'MEG2143',
 'MEG2211',
 'MEG2212',
 'MEG2213',
 'MEG2221',
 'MEG2222',
 'MEG2223',
 'MEG2231',
 'MEG2232',
 'MEG2233',
 'MEG2241',
 'MEG2242',
 'MEG2243',
 'MEG2311',
 'MEG2312',
 'MEG2313',
 'MEG2321',
 'MEG2322',
 'MEG2323',
 'MEG2331',
 'MEG2332',
 'MEG2333',
 'MEG2341',
 'MEG2342',
 'MEG2343',
 'MEG2411',
 'MEG2412',
 'MEG2413',
 'MEG2421',
 'MEG2422',
 'MEG2423',
 'MEG2431',
 'MEG2432',
 'MEG2433',
 'MEG2441',
 'MEG2442',
 'MEG2443',
 'MEG2511',
 'MEG2512',
 'MEG2513',
 'MEG2521',
 'MEG2522',
 'MEG2523',
 'MEG2531',
 'MEG2532',
 'MEG2533',
 'MEG2541',
 'MEG2542',
 'MEG2543',
 'MEG2611',
 'MEG2612',
 'MEG2613',
 'MEG2621',
 'MEG2622',
 'MEG2623',
 'MEG2631',
 'MEG2632',
 'MEG2633',
 'MEG2641',
 'MEG2642',
 'MEG2643']

def char2num(seq):
    """convert list of chars to integers eg ABC=>012"""
    if isinstance(seq, str):
        seq = list(seq)
    assert ord('A')-65 == 0
    nums = [ord(c.upper())-65 for c in seq]
    assert all([0<=n<=90 for n in nums])
    return nums

def num2char(arr):
    """convert list of ints to alphabetical chars eg 012=>ABC"""
    if isinstance(arr, int):
        return chr(arr+65)
    arr = np.array(arr, dtype=int)
    return np.array([chr(x+65) for x in arr.ravel()]).reshape(*arr.shape)

def seq2tf(sequence, n_states=None):
    """
    create a transition matrix from a sequence string,
    e.g. ABCDEFG
    Please note that sequences will not be wrapping automatically,
    i.e. a wrapping sequence should be denoted by appending the first state.

    :param sequence: sequence in format "ABCD..."
    :param seqlen: if not all states are part of the sequence,
                   the number of states can be specified
                   e.g. if the sequence is ABE, but there are also states F,G
                   n_states would be 7

    """

    seq = char2num(sequence)
    if n_states is None:
        n_states = max(seq)+1
    # assert max(seq)+1==n_states, 'not all positions have a transition'
    TF = np.zeros([n_states, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+1>=len(seq): continue
        p2 = seq[(i+1) % len(seq)]
        TF[p1, p2] = 1
    return TF.astype(float)

transition_matrix = seq2tf(seq_12)
