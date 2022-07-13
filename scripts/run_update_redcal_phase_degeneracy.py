#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

from hera_cal.redcal import update_redcal_phase_degeneracy, update_redcal_phase_degeneracy_argparser

ap = update_redcal_phase_degeneracy_argparser()
argvars = vars(ap.parse_args())
update_redcal_phase_degeneracy(**argvars)
