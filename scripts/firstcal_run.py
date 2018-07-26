#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

import sys
from hera_cal.firstcal import firstcal_run, firstcal_option_parser

o = firstcal_option_parser()
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

firstcal_run(files, opts, history)
