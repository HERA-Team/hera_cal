#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""command-line drive script for lstbin.lst_align_files()"""

from hera_cal import lstbin
import sys
import os
import glob

a = lstbin.lst_align_arg_parser()
args = a.parse_args()
history = ' '.join(sys.argv)

# get kwargs
kwargs = dict(vars(args))
del kwargs['data_files']
# configure verbose
kwargs['verbose'] = kwargs['silence'] is False
del kwargs['silence']

lstbin.lst_align_files(args.data_files, **kwargs)
