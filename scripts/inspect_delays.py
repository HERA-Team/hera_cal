#! /usr/bin/env python

import aipy as a
import numpy as np
import optparse
import sys
import pyuvdata
import glob
import pylab as p

# Options
o = optparse.OptionParser()
o.set_usage('inspect_delays.py [options] *firstcal.fits')
o.set_description(__doc__)
a.scripting.add_standard_options(o, pol=True)
opts, args = o.parse_args(sys.argv[1:])

delays = {}
for f in args:
    cal = pyuvdata.UVCal()
    cal.read_calfits(f)
    print " Reading calibration: {0}".format(f)

    if cal.cal_type != 'delay':
        print "Not a file with delays, exiting..."
        exit()

    for i, ant in enumerate(cal.ant_array):
        if ant not in delays:
            delays[ant] = []
        delays[ant].append(cal.delay_array[i, 0, :, 0])

for ant in cal.ant_array:
    p.plot(1e9 * np.concatenate(delays[ant]).flatten(), '.', label=str(ant))
p.xlabel('time bins')
p.ylabel('delays (ns)')
p.legend(loc='best')
p.show()
