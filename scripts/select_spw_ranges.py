#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command line driver for utils.select_spw_ranges"""

from hera_cal import utils
ap = utils.select_spw_ranges_argparser()
args = ap.parse_args()
utils.select_spw_ranges(**vars(args))
