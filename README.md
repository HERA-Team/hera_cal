# hera_cal

[![Build Status](https://travis-ci.org/HERA-Team/hera_cal.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_cal)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_cal/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_cal?branch=master)

The hera_cal package inlcudes modules and scripts that are required to run redundant calibration on HERA as part of the real time system.

Full documentation available on [Read the Docs.](http://hera_cal.readthedocs.io/en/latest/)

# Package Details

## Modules

* hera_cal.firstcal: module includes the FirstCalRedundantInfo class, FirstCal class that solves for delays, and other helper functions.

* hera_cal.omni: includes functions and classes for interfacing with and running omnical.

* hera_cal.abscal: includes AbsCal class for phasing and scaling visibility data to an absolute reference. See scripts/notebook/running_abscal.ipynb for a quick tutorial. Functionalities include:
    1. scaling raw data to an absolute (visibility) reference
    2. scaling omnicalibrated data to an absolute (visibility) reference
    3. scaling omnical unique-baseline visibility solutions to an absolute reference (still beta testing this capability)

## Scripts

* firstcal\_run.py: runs firstcal on a per file basis.
* omni\_run.py: runs omnical on a per file basis.
* omni\_apply.py: apply calibration solutions to miriad files.

# Installation
## Dependencies
First install dependencies. 

* numpy >= 1.10
* scipy
* matplotlib
* astropy >=1.2
* pyephem
* [aipy](https://github.com/HERA-Team/aipy/)
* [pyuvdata](https://github.com/HERA-Team/pyuvdata/)
* [omnical](https://github.com/HERA-Team/omnical/) >= 5.0.2
* [linsolve](https://github.com/HERA-Team/linsolve)
* [hera_qm](https://github.com/HERA-Team/hera_qm)

## Install hera_cal
Install with ```python setup.py install```
