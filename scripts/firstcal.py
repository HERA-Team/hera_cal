#!/usr/bin/env python2.7

import sys
from heracal.firstcal import firstcal_run, firstcal_option_parser

o = firstcal_option_parser()
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

firstcal_run(files, opts, history)
