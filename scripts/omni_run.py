#! /usr/bin/env python

import aipy
import numpy as np
from heracal.omni import omni_run
import pyuvdata
import optparse
import os
import sys

o = optparse.OptionParser()
o.set_usage(
    "omni_run.py -C [calfile] -p [pol] --firstcal=[firstcal path] [options] *.uvc")
o.set_description(__doc__)
aipy.scripting.add_standard_options(o, cal=True, pol=True)
o.add_option('--omnipath', dest='omnipath', default='.', type='string',
             help='Path to save omnical solutions.')
o.add_option('--ex_ants', dest='ex_ants', default=None,
             help='Antennas to exclude, separated by commas.')
o.add_option('--firstcal', dest='firstcal', type='string',
             help='Path and name of firstcal file. Can pass in wildcards.')
o.add_option('--minV', action='store_true',
             help='Toggle V minimization capability. This only makes sense in the case of 4-pol cal, which will set crosspols (xy & yx) equal to each other')
o.add_option('--median', action='store_true',
             help='Take the median over time of the starting calibration gains (e.g. firstcal).')
opts, files = o.parse_args(sys.argv[1:])

history = ' '.join(sys.argv)

omni_run(files, opts, history)
