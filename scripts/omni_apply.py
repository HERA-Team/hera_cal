#! /usr/bin/env python

import aipy as a, numpy as np, re, os
import optparse, sys, pyuvdata, heracal, glob

### Options ###
o = optparse.OptionParser()
o.set_usage('omni_apply.py [options] *uvcRRE')
o.set_description(__doc__)
a.scripting.add_standard_options(o,pol=True)
o.add_option('--xtalk',dest='xtalk',default=False,action='store_true',
            help='Toggle: apply xtalk solutions to data. Default=False')
o.add_option('--omnipath',dest='omnipath',default='%s.fits',type='string',
            help='Format string (e.g. "path/%s.fits", where you actually type the "%s") which converts the input file name to the omnical npz path/file.')
o.add_option('--firstcal', action='store_true', 
            help='Applying firstcal solutions.')
opts,args = o.parse_args(sys.argv[1:])
args = np.sort(args)
pols = opts.pol.split(',')

def file2pol(filename): return filename.split('.')[3] #XXX assumes file naming format
def isLinPol(polstr): return len(list(set(polstr)))==1

linear_pol_keys = []
for pp in pols:
    if isLinPol(pp):
        linear_pol_keys.append(pp)

filedict = {}
solution_files = sorted(glob.glob(opts.omnipath))

if opts.firstcal:
    firstcal_files = {}
    nf=0
    for pp in pols:
        if isLinPol(pp):
            firstcal_files[pp] = sorted([s for s in glob.glob(opts.firstcal) if pp in s])
            nf += len(firstcal_files[pp])    

for i, f in enumerate(args):
    pp = file2pol(f)
    if not opts.firstcal:
        if len(pols)==1:
            fexpected = '%s/%s.fits'%(os.path.dirname(opts.omnipath),os.path.basename(f))
        else: 
            fexpected = '%s/%s.fits'%(os.path.dirname(opts.omnipath), os.path.basename(f).replace('.%s'%pp,''))
        try:
            ind = solution_files.index(fexpected)
            filedict[f] = str(solution_files[ind])
        except ValueError:
           raise Exception('Solution file %s.fits expected; not found.'%f)
    else:
        if nf == len(solution_files)*len(pols): # atomic firstcal application
            filedict[f] = solution_files[i] # XXX this is fragile
        else: # one firstcal file for many data files
            if isLinPol(pp):
                filedict[f] = [firstcal_files[pp][0]]
            else:
                filedict[f] = [firstcal_files[lpk][0] for lpk in linear_pol_keys]

jonesLookup = {
    -5: (-5,-5),
    -6: (-6,-6),
    -7: (-5,-6),
    -8: (-6,-5)
}

for f in args:
    mir = pyuvdata.UVData()
    print "  Reading {0}".format(f)
    mir.read_miriad(f)
    cal = pyuvdata.UVCal()
    print "  Reading calibration : {0}".format(filedict[f])
    cal.read_calfits(filedict[f])
    
    print "  Calibrating..."
    antenna_index = dict(zip(*(cal.ant_array,range(cal.Nants_data))))
    for p,pol in enumerate(mir.polarization_array):
        p1,p2 = [list(cal.jones_array).index(pk) for pk in jonesLookup[pol]] #XXX could replace with numpy function instead of casting to list
        for bl,k in zip(*np.unique(mir.baseline_array, return_index=True)):
            blmask = np.where(mir.baseline_array == bl)[0]
            ai, aj = mir.baseline_to_antnums(bl)
            if not ai in cal.antenna_numbers or not aj in cal.antenna_numbers:
                continue
            for nsp, nspws in enumerate(mir.spw_array):
                if cal.cal_type == 'gain' and cal.gain_convention == 'multiply':
                    mir.data_array[blmask, nsp, :, p] = \
                                mir.data_array[blmask, nsp, :, p] * \
                                cal.gain_array[antenna_index[ai], nsp, :, :, p1].T * \
                                np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p2].T)
                                                               
                if cal.cal_type == 'gain' and cal.gain_convention == 'divide':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                cal.gain_array[antenna_index[ai], nsp, :, :, p1].T / \
                                np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p2].T)
    
                if cal.cal_type == 'delay' and cal.gain_convention == 'multiply':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] * \
                                np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[ai], nsp, :, p1].reshape(-1,1),cal.freq_array)) * \
                                np.conj(np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[aj], nsp, :, p2].reshape(-1,1),cal.freq_array)))

                if cal.cal_type == 'delay' and cal.gain_convention == 'divide':
                    mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[ai], nsp, :, p1].reshape(-1,1),cal.freq_array)) / \
                                np.conj(np.exp(-2j*np.pi*np.dot(cal.delay_array[antenna_index[aj], nsp, :, p2].reshape(-1,1),cal.freq_array)))



    if opts.firstcal:
        print " Writing {0}".format(f+'F')
        mir.write_miriad(f +'F') 
    else:
        print " Writing {0}".format(f+'O')
        mir.write_miriad(f +'O') 

