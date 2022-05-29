#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

from hera_cal.redcal import nightly_median_firstcal_delays, nightly_median_firstcal_delays_argparser

ap = update_redcal_phase_degeneracy_argparser()
argvars = vars(ap.parse_args())
update_redcal_phase_degeneracy(**argvars)
