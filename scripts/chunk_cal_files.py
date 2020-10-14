#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


""" Command Line Driver for data file chunker."""

from hera_cal import chunker

a = chunker.chunk_cal_parser()
args = a.parse_args()

chunker.chunk_cal_files(filenames=a.filenames, outputfile=a.outputfile,
                        inputfile=a.inputfile, spw=a.spw)
