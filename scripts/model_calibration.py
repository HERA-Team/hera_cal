#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License

"""Command-line driver script for model based calibration."""

from hera_cal import abscal_parser, AbsCal

a = abscal_parser()

args = a.parse_args()
