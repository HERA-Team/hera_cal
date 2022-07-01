# hera_cal
[![](https://github.com/HERA-Team/hera_cal/workflows/Run%20Tests/badge.svg?branch=master)](https://github.com/HERA-Team/hera_cal/actions)
[![codecov](https://codecov.io/gh/HERA-Team/hera_cal/branch/master/graph/badge.svg)](https://codecov.io/gh/HERA-Team/hera_cal)


The hera_cal package includes modules and scripts for the calibration and LST-binning of [Hydrogen Epoch of Reionization Array (HERA)](http://reionization.org/) data, along with various helpful methods for filtering and smoothing of data and calibration solutions. These are meant for use interatively, as part of offline analysis (e.g. [IDR 2.2](http://reionization.org/manual_uploads/HERA069_IDR2.2_Memo_v2.html)), or as part of HERA's realtime analysis pipeline using [`hera_opm`](https://github.com/HERA-Team/hera_opm/).

This package only officially supports python 3, though most functionality will still work in python 2.

## Package Details

### Modules


#### I/O and Data Handling

* `hera_cal.io`: contains `HERACal` and `HERAData` that wrap `pyuvdata` equivalents and enable easy I/O of data and calibration files.

* `hera_cal.datacontainer`: contains the `DataContainer` object, a dictionary-like container for visibility data with various useful abstractions

#### Calibration

* `hera_cal.redcal`: redundant calibration module, with `firstcal`, `logcal`, `lincal`, and `omnical` and helper functions for finding and manipulating sets of redundant baselines.

* `hera_cal.abscal`: absolute calibnration module, largely used to calibrate out redcal degeneraices post-redundant calibration using an externally calibrated data set.

* `hera_cal.apply_cal`: functions to apply calibration solutions (and flags) to data in memory or on disk

* `hera_cal.reflections`: functions for fitting per-antenna cable reflections and other per-baseline high-delay systematics (e.g. cross-talk)

* `hera_cal.tempcal`: functions for calibrating using external temperature data


#### LST-Binning

* `hera_cal.lstbin`: module for LST-binning, including aligning, rephasing, and MAD clipping, and associated I/O


#### Filtering and Smoothing

* `hera_cal.smooth_cal`: utilities for smoothing calibration solutions in frequency, time, or both

* `hera_cal.vis_clean`: base module interface to aipy CLEAN for low- and high-pass filtering visibility data along the time or frequency axis

* `hera_cal.delay_filter`: specialization of `vis_clean` for performing delay filtering (e.g. wedge filtering) of visibility data

* `hera_cal.frf`: specialization of `vis_clean` for performing fringe-rate (e.g. time) filtering of visibility data

#### Other Utilities

* `hera_cal.noise`: utilities for calculating visibility noise from interleaved differences and for predicting visibility noise from autocorrelations

* `hera_cal.autos`: module for extracting and saving autocorrelation data

* `hera_cal.utils`: grabbag of useful functions, including polarization string handling, FFT-based delay fitting, time and LST math, solar position calculation, chi^2 calculations, etc.

* `hera_cal.flag_utils`: utilities for applying, synthesizing, and factoring flags


### Scripts


* `apply_cal.py`: apply calibration solutions (as associated antenna-based flags) to data
* `auto_reflection_run.py`: estimate cable reflection gains from autocorrelations
* `delay_filter_run.py`: perform delay filtering outside the wedge
* `extract_autos.py`: extract autocorrelation visibilities and save them
* `extract_hh.py`: extract data only from the core HERA Hex
* `lstbin_run.py`: run the LST-binner
* `noise_from_autos.py`: infer noise on visibilities and save as per-antenna noise standard deviation
* `post_redcal_abscal_run.py`: run abscal post-redundant calibration and save updated calibration solutions
* `redcal_run.py`: run redundant calibration and save firstcal and omnical visibility abd calibration solutions
* `smooth_cal_run.py`: smooth calibration solutions in time, frequency, or both

### Documentation

The only guaranteed up-to-date documentation of individual functions and classes are their docstrings.

The [IDR2.2 Release Memo](https://github.com/HERA-Team/hera_sandbox/blob/master/jsd/IDR2_2/IDR2.2_Memo.ipynb) is a jupyter notebook that can run at NRAO and contains useful examples of data access and visualization.

Many modules have [instructional notebooks avaible here](../tree/master/scripts/notebooks), though some of those are out of date.

While `hera_cal` has a [Read the Docs](http://hera_cal.readthedocs.io/en/latest/), it is wildly out of date.


## Installation
Preferred installation method is `pip install .` in top-level directory. Alternatively,
one can use `python setup.py install`. This will attempt to install all dependencies.
If you prefer to explicitly manage dependencies, see below.

### Dependencies
Those who use `conda` (preferred) may wish to install the following manually before
installing `hera_cal`:

`conda install -c conda-forge "numpy>=1.10" scipy scikit-learn h5py astropy pyuvdata`

(note that `h5py` is a dependency of `hera_qm`, not `hera_cal`).

Other dependencies that will be installed from PyPI on-the-fly are:
* [linsolve](https://github.com/HERA-Team/linsolve)
* [hera_qm](https://github.com/HERA-Team/hera_qm)
* [hera_filters](https://github.com/HERA-Team/hera_filters)

`hera_cal` also has the _optional_ dependency of `aipy`, and some
functions will not work without this dependency. To install all optional dependencies, use
`pip install .[all]` or `pip install git+git://github.com/HERA-Team/hera_cal.git[all]`.

### Development Environment
To install a full development environment for `hera_cal`, it can be useful to work with
a fresh `conda` environment. These steps will get you up and running::

    $ conda create -n hera_cal python=3
    $ conda activate hera_cal
    $ conda env update -n hera_cal -f environment.yml
    $ pip install -e .

This installs extra packages than those required to use `hera_cal`, including `hera_sim`
and `pytest`.

If you are developing `hera_cal` please install pre-commit: `pip install pre-commit` and
then `pre-commit install` in the top-level directory. This will check your style before
you make commits.

### Running tests
Tests use the `pytest` framework. To run all tests, call `pytest` or
`python -m pytest` from the base directory of the repo.

## Issues and Contribution

Issues [are tracked here](https://github.com/HERA-Team/hera_cal/issues). Please submit bugs, feature requests, etc. Contributions to this repo via pull request are welcome, though they require thorough peer review before merging into the master branch. To the best of our ability, all code should be covered with tests. The primary maintainer of `hera_cal` is [@jsdillon](https://github.com/jsdillon). Other maintiners who can update the master branch include [@AaronParsons](https://github.com/AaronParsons), [@nkern](https://github.com/nkern), [@adampbeardsley](https://github.com/adampbeardsley), and [@plaplant](https://github.com/plaplant).
