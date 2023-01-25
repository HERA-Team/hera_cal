#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command line driver for utils.select_spw_ranges"""

from hera_cal import utils
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

ap = utils.select_spw_ranges_argparser()
args = parse_args(ap)

kw = filter_kwargs(vars(args))
run_with_profiling(
    utils.select_spw_ranges,
    args,
    **kw
)
