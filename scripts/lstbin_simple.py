#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""
command-line drive script for lstbin.lst_bin_files()
Assume our current working directory is data/, which looks like
data/
    2485042/
        zen.2485042.1000.xx.HH.uv
        zen.2485042.2000.xx.HH.uv
        zen.2485042.3000.xx.HH.uv
    2485043/
        zen.2485043.1200.xx.HH.uv
        zen.2485043.2200.xx.HH.uv
        zen.2485043.3200.xx.HH.uv
    2485044/
        zen.2485044.1400.xx.HH.uv
        zen.2485044.2400.xx.HH.uv
        zen.2485044.3400.xx.HH.uv
In order to LST-bin all of the above files, our call to lstbin_run.py would look like
lstbin_run.py '2485042/zen.2485042.*.xx.HH.uv' \
              '2485043/zen.2485043.*.xx.HH.uv' \
              '2485044/zen.2485044.*.xx.HH.uv'
Arguments can be specified before the search strings, like
lstbin_run.py --lst_init np.pi --dlst 0.001 --align --outdir './' \
              '2485042/zen.2485042.*.xx.HH.uv' \
              '2485043/zen.2485043.*.xx.HH.uv' \
              '2485044/zen.2485044.*.xx.HH.uv'
Note: make sure the search strings are bounded by quotations!
"""

from hera_cal import lstbin_simple as lstbin
import sys
import glob
import json
from hera_cal._cli_tools import setup_logger
import logging
import importlib
from pathlib import Path
from hera_cal.profiling_utils import add_profiling_args, run_with_profiling

logger = logging.getLogger('hera_cal')
setup_logger()
a = lstbin.lst_bin_arg_parser()
add_profiling_args(a)
args = a.parse_args()

history = ' '.join(sys.argv)

# get kwargs
kwargs = dict(vars(args))

logger.setLevel(kwargs.pop("log_level"))

# configure history
kwargs['history'] += history

# configure data_files
data_files = [sorted(glob.glob(s.strip("'").strip('"'))) for s in args.data_files]
del kwargs['data_files']

# ensure data_files is a set of nested lists
if not isinstance(data_files[0], list):
    raise ValueError("data_files is not a set of nested lists. check input to data_files. See lstbin_run.py doc-string for examples.")

write_kwargs = json.loads(kwargs.pop('write_kwargs'))
if not write_kwargs:
    write_kwargs = {}

if not isinstance(write_kwargs, dict):
    raise ValueError("write_kwargs must be a json dictionary")

# configure vis_units
if args.vis_units is None:
    del kwargs['vis_units']

if 'vis_units' in kwargs:
    write_kwargs['vis_units'] = kwargs.pop('vis_units')

# handle output_file_select fed as None
if kwargs['output_file_select'] == ['None']:
    del kwargs['output_file_select']

# Handle calfile rules
# Turn a list into a list of 2-tuples.
crules = kwargs.pop("calfile_rules")
calfile_rules = [(crules[i], crules[i+1]) for i in range(len(crules)//2)]

del kwargs['profile']
del kwargs['profile_funcs']
del kwargs['profile_output']

kwargs['save_channels'] = tuple(int(ch) for ch in kwargs['save_channels'].split(','))
kwargs['golden_lsts'] = tuple(float(lst) for lst in kwargs['golden_lsts'].split(','))

run_with_profiling(
    lstbin.lst_bin_files, 
    a, 
    data_files=data_files,
    calfile_rules=calfile_rules, 
    write_kwargs=write_kwargs, 
    **kwargs
)
