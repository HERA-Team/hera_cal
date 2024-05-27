#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""command-line drive script for lstbin.lst_bin_files()"""

from hera_cal import lst_stack as lstbin
import sys
import json
from hera_cal._cli_tools import setup_logger, parse_args, filter_kwargs, run_with_profiling
from pathlib import Path

a = lstbin.wrappers.lst_bin_arg_parser()
args = parse_args(a)

history = ' '.join(sys.argv)

# get kwargs
kwargs = filter_kwargs(dict(vars(args)))


# configure history
kwargs['history'] += history

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
if crules is not None:
    calfile_rules = [(crules[i], crules[i + 1]) for i in range(len(crules) // 2)]
else:
    calfile_rules = None

inprules = kwargs.pop("where_inpainted_file_rules")
if inprules is not None:
    inpaint_rules = [(inprules[i], inprules[i + 1]) for i in range(len(inprules) // 2)]
else:
    inpaint_rules = None

kwargs['save_channels'] = tuple(int(ch) for ch in kwargs['save_channels'].split(','))
kwargs['golden_lsts'] = tuple(float(lst) for lst in kwargs['golden_lsts'].split(','))

kwargs['output_flagged'] = not kwargs.pop('no_flagged_mode')
kwargs['output_inpainted'] = kwargs.pop("do_inpaint_mode")
if kwargs['sigma_clip_subbands'] is not None:
    kwargs['sigma_clip_subbands'] = [(int(low), int(high)) for b in kwargs["sigma_clip_subbands"].split(",") for low, high in b.split("~")]

run_with_profiling(
    lstbin.lst_bin_files,
    args,
    config_file=kwargs.pop('configfile'),
    calfile_rules=calfile_rules,
    where_inpainted_file_rules=inpaint_rules,
    write_kwargs=write_kwargs,
    **kwargs
)
