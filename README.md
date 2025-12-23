# DeSMRRest-RestingState-TDLM-Simulation

Code for the resting state analysis and simulation data using TDLM for the paper "Challenges in Replay Detection by TDLM in Post-Encoding Resting State"

For an overview of the experiment and results head to GH-pages: [Paper: Challenges in Replay Detection by TDLM in Post-Encoding Resting State](https://cimh-clinical-psychology.github.io/DeSMRRest-TDLM-Simulation/)

Find the preprint at [eLife](https://elifesciences.org/reviewed-preprints/108023)

### 1. Getting started

First of all, clone the repository and init the submodule `meg_utils`

```bash
git clone https://github.com/CIMH-Clinical-Psychology/DeSMRRest-TDLM-Simulation.git
cd DeSMRRest-TDLM-Simulation
git submodule init
git submodule update
```

Then install the requirements using pip `pip install -r requirements.txt`. It is recommended to run this in a dedicated environment not to mix up your current Python installation. You can do so e.g. using [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

```bash
conda create --name tdlm-sim python=3.10
conda activate tdlm-sim
# assuming you are in the folder of the repository
pip install -r requirements.txt
```

### 2. Download and setup

Then you need to specify your settings. Open `settings.py` and around line 117 insert where you want to store the data, or (if you have already downloaded it), where the data was saved. You can leave the other parameters the same.

```python
data_dir = '/path/to/data/'           # directory containing the FIF files
cache_dir = f'{data_dir}/cache/'      # used for caching
plot_dir = f'{data_dir}/plots/'       # plots will be stored here
log_dir = f'{data_dir}/plots/logs/'   # log files will be created here
```

Download the experiment files from [Zenodo](https://zenodo.org/record/8001755) into a common folder. The files are split up into two repositories, 10.5281/zenodo.8001755 and 10.5281/zenodo.15629081. Instead of downloading them individually, you they can be downloaded automatically by running `python 0_download_dataset.py`. This will utilize the Python API `pyzenodo3` and download the 140 GB dataset into your `data_dir`. This can take a while.

### 3. Scripts

The analysis is split into several scripts:

| Script                               | Description                                                                                                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0_download_dataset.py`              | Downloads the dataset from Zenodo. Set your datadir beforehand in settings.py                                                                              |
| `1_run_preprocessing.py`             | Preprocesses the MEG data for each participant. This includes downsampling, filtering, and segmenting the data.                                            |
| `2_run_study1.py`                    | Runs the first study, which investigates sequenceness in resting-state data.                                                                               |
| `3_run_study2.py`                    | Runs the second study, which is a hybrid simulation to investigate the effect of replay on sequenceness.                                                   |
| `4_run_supplement.py`                | Runs supplementary analyses, including sensor pattern analysis and ERP visualization.                                                                      |
| `5_run_revision1.py`                 | Runs analyses for the first revision of the paper.                                                                                                         |
| `6_run_synthetic_simulation.py`      | Runs a purely synthetic simulation of replay similar to the proposed by [Liu et al 2021](https://github.com/YunzheLiu/TDLM/blob/master/Simulate_Replay.m). |
| `7_run_discriminability_analysis.py` | Compares the discriminability of classifier probabilities between real and simulated data.                                                                 |

### 4. Running the analysis

To run the analysis, you first need to run the preprocessing for all participants. You can do this by running:

```bash
python 1_run_preprocessing.py
```

This will preprocess the data for all participants and save the results in the cache directory. This can take a while (~1h per participant).

If you are working on a cluster, you can also submit the preprocessing as a job. An example sbatch script is provided in `1_run_preprocessing.sbatch`.

After the preprocessing is finished, you can run the other scripts to reproduce the analyses and figures from the paper. For example, to run the first study, you can run:

```bash
python 2_run_study1.py
```

All plots should appear in your `plot_dir` which you defined in your `settings.py`.
