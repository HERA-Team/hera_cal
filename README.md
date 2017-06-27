# hera_cal

[![Build Status](https://travis-ci.org/HERA-Team/hera_cal.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_cal)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_cal/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_cal?branch=master)

The hera_cal package inlcudes modules and scripts that are required to run redundant calibration on HERA as part of the real time system.

Full documentation available on [Read the Docs.](http://hera_cal.readthedocs.io/en/latest/)

# Package Details

## Modules

* hera_cal.firstcal: module includes the FirstCalRedundantInfo class, FirstCal class that solves for delays, and other helper functions.
* hera_cal.omni: includes functions and classes for interfacing with and running omnical.

## Scripts

* firstcal\_run.py: runs firstcal on a per file basis.
* omni\_run.py: runs omnical on a per file basis.
* omni\_apply.py: apply calibration solutions to miriad files.


# Installation
## Dependencies
First install dependencies. 

* numpy >= 1.10
* scipy
* astropy >=1.2
* pyephem
* aipy
* pyuvdata
* omnical >= 5.0.2
* matplotlib

## Install hera_cal
Install with ```python setup.py install```
