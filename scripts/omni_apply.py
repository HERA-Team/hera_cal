#!/usr/bin/env python
import numpy as np
import optparse
from heracal.omni import omni_apply
import aipy, sys

### Options ###
o = optparse.OptionParser()
o.set_usage('omni_apply.py [options] *uvc')
o.set_description(__doc__)
aipy.scripting.add_standard_options(o, pol=True)
o.add_option('--omnipath', dest='omnipath', default='*.fits', type='string',
             help='Filename or format string that gets passed to glob for omnical/firstcal solution fits files.')
o.add_option('--median', action='store_true',
             help='Take the median in time before applying solution. Applicable only in delay.')
o.add_option('--firstcal', action='store_true',
             help='Applying firstcal solutions.')
o.add_option('--extension', dest='extension', default='O', type='string',
             help='Filename extension to be appended to the input filename')
opts, args = o.parse_args(sys.argv[1:])
args = np.sort(args)

omni_apply(args, opts)
