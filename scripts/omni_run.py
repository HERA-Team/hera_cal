#! /usr/bin/env python

from hera_cal import omni
import optparse
import sys

o = omni.get_optionParser('omni_run')
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

omni.omni_run(files, opts, history)
