#!/usr/bin/env python
import numpy as np
import optparse
from heracal import omni
import sys

### Options ###
o = omni.get_optionParser('omni_apply')
opts, args = o.parse_args(sys.argv[1:])
args = np.sort(args)

omni.omni_apply(args, opts)
