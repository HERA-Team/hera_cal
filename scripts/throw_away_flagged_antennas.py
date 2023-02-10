#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for throw away all data with apriori flags. Saves downstream I/O, memory etc..."""

import sys
import argparse
from hera_cal import io
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs


ap = io.throw_away_flagged_ants_parser()
args = parse_args(ap)
run_with_profiling(io.throw_away_flagged_ants, args, **filter_kwargs(vars(args)))
