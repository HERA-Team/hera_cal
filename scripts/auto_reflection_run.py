#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from hera_cal import reflections

parser = reflections.auto_reflection_argparser()
a = parser.parse_args()

kwargs = dict(vars(a))
kwargs.pop('data')
kwargs.pop('dly_range')
kwargs.pop('output_fname')

reflections.auto_reflection_run(a.data, a.dly_range, a.output_fname, **kwargs)
