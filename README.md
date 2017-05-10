# heracal

[![Build Status](https://travis-ci.org/HERA-Team/heracal.svg?branch=master)](https://travis-ci.org/HERA-Team/heracal)
[![Coverage Statue](https://coveralls.io/repos/github/HERA-Team/heracal/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/heracal?branch=master)

The heracal package inlcudes modules and scripts that are required to run redundant calibration on HERA as part of the real time system.

# Package Details

## Modules

* heracal.firstcal: module includes the FirstCalRedundantInfo class, FirstCal class that solves for delays, and other helper functions.
* heracal.omni: includes functions and classes for interfacing with and running omnical.
* heracal.metrics: includes functions for determining metrics of the array, e.g. determining bad antennas.
* heracal.xrfi: suite of rfi excision algorithms.

## Scripts

* omni\_run.py: runs omnical on a per file basis.
* run\_firstcal.py: runs firstcal on a per file basis.
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

## Install heracal
Install with ```python setup.py install```
