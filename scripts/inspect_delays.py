#! /usr/bin/env python

import aipy as a, numpy as np 
import optparse, sys, pyuvdata, glob
import pylab as p

### Options ###
o = optparse.OptionParser()
o.set_usage('inspect_delays.py [options] *firstcal.fits')
o.set_description(__doc__)
a.scripting.add_standard_options(o,pol=True)
opts,args = o.parse_args(sys.argv[1:])

delays = {}
for f in args:
    cal = pyuvdata.UVCal()
    cal.read_calfits(f)
    print " Reading calibration: {0}".format(f)

    if cal.cal_type!='delay':
        print "Not a file with delays, exiting..."  
        exit()
    
    for i, ant in enumerate(cal.antenna_numbers):
        if not ant in delays: delays[ant] = []
        delays[ant].append(cal.delay_array[i,0,:,0])
    
import IPython; IPython.embed()    
for ant in cal.antenna_numbers:
    p.plot(np.array(delays[ant]).flatten(), 'o')
p.show()
