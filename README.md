# hera_cal

[![Build Status](https://travis-ci.org/HERA-Team/hera_cal.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_cal)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_cal/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_cal?branch=master)

The hera_cal package includes modules and scripts for calibration of HERA as part of the real time system.

Full documentation available on [Read the Docs.](http://hera_cal.readthedocs.io/en/latest/)

# Package Details

## Modules

* hera_cal.firstcal: module includes the FirstCalRedundantInfo class, FirstCal class that solves for delays, and other helper functions.

* hera_cal.omni: includes functions and classes for interfacing with and running omnical.

* hera_cal.abscal: includes AbsCal class for phasing and scaling visibility data to an absolute reference. See scripts/notebook/running_abscal.ipynb for a quick tutorial. Functionalities include:
    1. scaling raw data to an absolute (visibility) reference
    2. scaling omnicalibrated data to an absolute (visibility) reference
    3. scaling omnical unique-baseline visibility solutions to an absolute reference (still beta testing this capability)

* hera_cal.lstbin: includes functions for LST-aligning visibility data, and LST binning files overlapping in LST. See scripts/notebooks/running_lstbin.ipynb for a brief tutorial.

## Scripts

* firstcal\_run.py: runs firstcal on a per file basis.
* omni\_run.py: runs omnical on a per file basis.
* omni\_apply.py: apply calibration solutions to miriad files.
* abscal\_run.py: run absolute calibration given a data file and model file(s)
* lstbin\_run.py: run LST binning on series of files overlapping in LST.

# Installation
## Dependencies
First install dependencies. 

* numpy >= 1.10
* scipy
* astropy >=1.2
* pyephem
* [uvtools](https://github.com/HERA-Team/uvtools)
* [aipy](https://github.com/HERA-Team/aipy/)
* [pyuvdata](https://github.com/HERA-Team/pyuvdata/)
* [linsolve](https://github.com/HERA-Team/linsolve)
* [hera_qm](https://github.com/HERA-Team/hera_qm)
* [hera_sim](https://github.com/HERA-Team/hera_sim) (Only required for unit tests.)

## Install hera_cal
Install with ```python setup.py install```

## Running tests
Tests use the `pytest` framework. To run all tests, call `pytest` or
`python -m pytest` from the base directory of the repo.