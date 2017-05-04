#! /usr/bin/env python

import aipy as a, numpy as np 
import optparse, sys, pyuvdata, heracal, glob

### Options ###
o = optparse.OptionParser()
o.set_usage('omni_apply.py [options] *uvcRRE')
o.set_description(__doc__)
a.scripting.add_standard_options(o,pol=True)
o.add_option('--xtalk',dest='xtalk',default=False,action='store_true',
            help='Toggle: apply xtalk solutions to data. Default=False')
o.add_option('--omnipath',dest='omnipath',default='%s.fits',type='string',
            help='Format string (e.g. "path/%s.npz", where you actually type the "%s") which converts the input file name to the omnical npz path/file.')
o.add_option('--firstcal', action='store_true', 
            help='Applying firstcal solutions.')
opts,args = o.parse_args(sys.argv[1:])
args = np.sort(args)


filedict = {}
solution_files = np.sort(glob.glob(opts.omnipath))
for i, f in enumerate(args):
        filedict[f] = str(solution_files[i])
print args, filedict    
for f in args:
    mir = pyuvdata.UVData()
    print "  Reading {0}".format(f)
    mir.read_miriad(f)
    cal = pyuvdata.UVCal()
    print "  Reading calibration : {0}".format(filedict[f])
    cal.read_calfits(filedict[f])

    print "  Calibrating..."
    antenna_index = dict(zip(*(cal.antenna_numbers,range(cal.Nants_data))))
    for p,pol in enumerate(mir.polarization_array):
        for bl,k in zip(*np.unique(mir.baseline_array, return_index=True)):
            blmask = np.where(mir.baseline_array == bl)[0]
            ai, aj = mir.baseline_to_antnums(bl)
            if not ai in cal.antenna_numbers or not aj in cal.antenna_numbers:
                continue
            for nsp, nspws in enumerate(mir.spw_array):
                if cal.cal_type == 'gain' and cal.gain_convention == 'multiply':
                    mir.data_array[blmask, nsp, :, p] = \
                                mir.data_array[blmask, nsp, :, p] * \
                                cal.gain_array[antenna_index[ai], nsp, :, :, p].T * \
                                np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p].T)
                                                               
                if cal.cal_type == 'gain' and cal.gain_convention == 'divide':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                cal.gain_array[antenna_index[ai], nsp, :, :, p].T / \
                                np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p].T)
    
                if cal.cal_type == 'delay' and cal.gain_convention == 'multiply':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] * \
                                np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[ai], nsp, :, p].reshape(-1,1),cal.freq_array)) * \
                                np.conj(np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[aj], nsp, :, p].reshape(-1,1),cal.freq_array)))

                if cal.cal_type == 'delay' and cal.gain_convention == 'divide':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[ai], nsp, :, p].reshape(-1,1),cal.freq_array)) / \
                                np.conj(np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[aj], nsp, :, p].reshape(-1,1),cal.freq_array)))



    if opts.firstcal:
        print " Writing {0}".format(f+'F')
        mir.write_miriad(f +'F') 
    else:
        print " Writing {0}".format(f+'O')
        mir.write_miriad(f +'O') 

