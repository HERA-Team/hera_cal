#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

"""Command-line driver script for model based calibration."""

from hera_cal.abscal import model_calibration_argparser, run_model_based_calibration
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

argparser = model_calibration_argparser()
args = parse_args(argparser)
argvars = filter_kwargs(vars(args))

run_with_profiling(run_model_based_calibration, args, **argvars)
