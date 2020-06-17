#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""
Command-line driver script for xtalk Filtering with DAYENU that allows for parallelization across baselines.
This script is meant to be called by OPM which is responsible for determining chunks of baselines to process in each
runs of this script on different compute nodes and writing appropriate antpairpol list strings for each script call.

Each run of this script writes out a pyuvdata uvh5 data file for the chunk of baselines process by each
run on the compute node. 
"""
from hera_cal import xtalk_filter
from hera_cal import vis_clean as vc
import sys

parser = xtalk_filter.xtalk_filter_argparser(mode='dayenu', parallelization_mode='baselines')

a = parser.parse_args()

# set kwargs
filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
antpairpol_list = vc._parse_antpairpol_list_string(a.antpairpol_list)
spw_range = a.spw_range
# Run Delay Filter
delay_filter.load_xtalk_filter_and_write_baseline_list(datafile_list=a.datafile_list, calfile_list=a.calfile_list,
                                         antpairpol_list=antpairpol_list, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode='dayenu',
                                         add_to_history=' '.join(sys.argv), **filter_kwargs)
