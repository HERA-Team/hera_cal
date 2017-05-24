#! /usr/bin/env python
"""
This script uses heracal.xrfi.xrfi to create flags based 
on the omnical chisq output (currently average over chisq
per antenna. This creates a new calfits file with a 
flag array with it.
"""

import numpy as np, optparse, sys
from heracal import omni, xrfi
from pyuvdata import UVCal

o = optparse.OptionParser()
o.set_usage('omni_xrfi.py [options] zen.*.omni.calfits')
o.set_description(__doc__)
opts,args = o.parse_args(sys.argv[1:])


for calfile in args:
    uvcal = UVCal()
    uvcal.read_calfits(calfile)
    avg_chisq_over_ant = np.mean(uvcal.quality_array, axis=0)[0] # takes nspw=0
    for ip,pol in enumerate(uvcal.jones_array):
        flags = xrfi.xrfi(avg_chisq_over_ant[:,:,ip])
        for i, ant in enumerate(uvcal.ant_array):
            uvcal.flag_array[i, 0, :, :, ip] = np.logical_or(uvcal.flag_array[i, 0, :, :, ip],flags)
    uvcal.write_calfits('.'.join(calfile.split('.')[:-1]) + '.xrfi.calfits')
