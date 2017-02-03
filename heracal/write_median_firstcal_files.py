#! /usr/bin/env python
'''Takes raw firstcal files and rewrites to give the median gain solution.'''

import numpy as np, glob, optparse, sys
from uvdata import UVCal

def load_dmv(files, offsets=False, verbose=False):
    delays = {}
    means = {}
    medians = {}
    vars = {}
    if offsets: ss = 'o'
    else: ss = 'd'
    for f in files:
        if verbose:
            print 'Reading %s'%f    
        cal = UVCal()
        cal.read_calfits(f)
        if cal.cal_type == 'delay': 
            for i, ant in enumerate(cal.antenna_numbers):
                if ant not in delays.keys():
                    delays[ant] = cal.delay_array[i,:,:,0]
                    continue 
                delays[ant] = np.hstack((delays[ant], cal.delay_array[i,:,:,0]))
        else:
            raise ValueError(("{0} does not contain delays. Check input files and retry".format(f)))
    
    for k in delays.keys():
        delays[k] = np.array(delays[k]).flatten()
        means[k] = np.mean(delays[k])
        medians[k] = np.median(delays[k])
        vars[k] = np.var(delays[k])
        
    return delays,means,medians,vars


o = optparse.OptionParser()
o.add_option('--mean', action='store_true', 
    help='Use the mean value of antenna delays for gain solutions.')
o.add_option('--median', action='store_true',
    help='Use the median value of antenna delays for gain solutions.')
opts,args = o.parse_args(sys.argv[1:])

if opts.mean and opts.median:
    raise ValueError('Must choose between mean and median values.')
    sys.exit(1)

#fqs = np.load(args[0])['freqs']
_,means,medians,_ = load_dmv(args)
# create load in input cal object. only works on files with delays. Error will be thrown above. 
cal = UVCal()
cal.read_calfits(args[0])
delays = []
for i in cal.antenna_numbers:
    delays.append(medians[i])
delays = np.array(delays).reshape(len(delays),1,1,1)
cal.delay_array = delays
cal.flag_array = np.zeros_like(cal.delay_array, dtype=np.bool)
cal.quality_array = np.ones_like(cal.delay_array)
# Update number of times in UVCal object
cal.Ntimes = 1
cal.time_array = cal.time_array[:1]

# save to fits
for f in args:
    fname = '.'.join(f.split('.')[:-2] + ['median', 'fits'])
    cal.write_calfits(fname)
