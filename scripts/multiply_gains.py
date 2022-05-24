#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

"""Command-line driver script for multiplying gains."""

from hera_cal.abscal import multiply_gains_argparser, multiply_gains

argparser = multiply_gains_argparser()
argvars = vars(argparser.parse_args())
multiply_gains(**argvars)
