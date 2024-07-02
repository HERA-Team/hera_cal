#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import reflections
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

parser = reflections.auto_reflection_argparser()

a = parse_args(parser)

kwargs = filter_kwargs(dict(vars(a)))
kwargs.pop('data')
kwargs.pop('output_fname')
kwargs.pop('dly_ranges')

run_with_profiling(
    reflections.auto_reflection_run,
    a,
    data=a.data,
    delay_ranges=a.dly_ranges,
    output_fname=a.output_fname,
    **kwargs
)
