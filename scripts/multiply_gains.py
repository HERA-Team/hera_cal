#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

"""Command-line driver script for multiplying gains."""

from hera_cal.abscal import multiply_gains_argparser, multiply_gains
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

argparser = multiply_gains_argparser()
args = parse_args(argparser)
argvars = filter_kwargs(vars(args))
run_with_profiling(multiply_gains, args, **argvars)
