#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

import numpy as np
import optparse
from hera_cal import omni
import sys

# Options
o = omni.get_optionParser('omni_apply')
opts, args = o.parse_args(sys.argv[1:])
args = np.sort(args)

omni.omni_apply(args, opts)
