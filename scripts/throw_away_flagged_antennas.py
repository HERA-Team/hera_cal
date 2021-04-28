#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for throw away all data with apriori flags. Saves downstream I/O, memory etc..."""

import sys
import argparse
from hera_cal import io


ap = io.throw_away_flagged_ants_parser()
args = ap.parse_args()
io.throw_away_flagged_ants(**vars(args))
