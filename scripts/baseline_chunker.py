#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
"""
Script for generating text files containing baseline list strings.
Each text file lists the baselines to be processed in each
compute job in baseline parallelization mode.
"""
from hera_cal import io
from hera_cal._cli_tools import parse_args, run_with_profiling

parser = io.antpairpol_parallelization_parser()
a = parse_args(parser)
run_with_profiling(
    io.generate_antpairpol_parallelization_files,
    a,
    filename=a.template_file,
    writedir=a.directory,
    bls_per_chunk=a.bls_per_chunk,
    polarizations=a.polarizations
)
