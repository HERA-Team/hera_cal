#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

"""Command-line driver script for model based calibration."""

from hera_cal import model_calibration_argparser, run_model_based_calibration

arparser = model_calibration_argparser()
argvars = vars(argparser.parse_args())
run_model_based_calibration(**argvars)
