#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import reflections

parser = reflections.auto_reflection_argparser()
a = parser.parse_args()

kwargs = dict(vars(a))
kwargs.pop('data')
kwargs.pop('output_fname')
kwargs.pop('dly_ranges')

reflections.auto_reflection_run(a.data, a.dly_ranges, a.output_fname, **kwargs)
