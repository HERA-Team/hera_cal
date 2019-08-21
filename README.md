# hera_cal

[![Build Status](https://travis-ci.org/HERA-Team/hera_cal.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_cal)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_cal/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_cal?branch=master)

The hera_cal package includes modules and scripts for calibration of HERA as part of the real time system.

Full documentation available on [Read the Docs.](http://hera_cal.readthedocs.io/en/latest/)

## Package Details

### Modules

* `hera_cal.firstcal`: module includes the `FirstCalRedundantInfo` class, `FirstCal` 
  class that solves for delays, and other helper functions.

* `hera_cal.omni`: includes functions and classes for interfacing with and running 
  `omnical`.

* `hera_cal.abscal`: includes `AbsCal` class for phasing and scaling visibility data to 
  an absolute reference. See `scripts/notebook/running_abscal.ipynb` for a quick 
  tutorial. Functionalities include:
    1. scaling raw data to an absolute (visibility) reference
    2. scaling omnicalibrated data to an absolute (visibility) reference
    3. scaling omnical unique-baseline visibility solutions to an absolute reference 
       (still beta testing this capability)

* `hera_cal.lstbin`: includes functions for LST-aligning visibility data, and LST-binning 
  files overlapping in LST. See `scripts/notebooks/running_lstbin.ipynb` for a brief 
  tutorial.

### Scripts

* `firstcal\_run.py`: runs firstcal on a per file basis.
* `omni\_run.py`: runs omnical on a per file basis.
* `omni\_apply.py`: apply calibration solutions to miriad files.
* `abscal\_run.py`: run absolute calibration given a data file and model file(s)
* `lstbin\_run.py`: run LST binning on series of files overlapping in LST.

## Installation
Preferred installation method is `pip install .` in top-level directory. Alternatively,
one can use `python setup.py install`. This will attempt to install all dependencies.
If you prefer to explicitly manage dependencies, see below.

### Dependencies
Those who use `conda` (preferred) may wish to install the following manually before 
installing `hera_cal`:

`conda install -c conda-forge "numpy>=1.10" scipy scikit-learn h5py astropy pyuvdata

(note that `h5py` is a dependency of `hera_qm`, not `hera_cal`).

Other dependencies that will be installed from PyPI on-the-fly are:
* pyephem
* [uvtools](https://github.com/HERA-Team/uvtools)
* [linsolve](https://github.com/HERA-Team/linsolve)
* [hera_qm](https://github.com/HERA-Team/hera_qm)

`hera_cal` also has the _optional_ dependency of `aipy`, and some functions will not
work without this dependency. To install all optional dependencies, use
`pip install .[all]` or `pip install git+git://github.com/HERA-Team/hera_cal.git[all]`.

### Development Environment
To install a full development environment for `hera_cal`, it is preferred to work with
a fresh `conda` environment. These steps will get you up and running::

    $ conda create -n hera_cal python=3
    $ conda activate hera_cal
    $ conda env update -n hera_cal -f environment.yml
    $ pip install -e . 

This installs extra packages than those required to use `hera_cal`, including `hera_sim`
and `pytest`.

### Running tests
Tests use the `pytest` framework. To run all tests, call `pytest` or
`python -m pytest` from the base directory of the repo.