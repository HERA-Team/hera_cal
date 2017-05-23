#! /usr/bin/env python

import aipy as a, numpy as np, re, os
import optparse, sys, pyuvdata, heracal, glob

### Options ###
o = optparse.OptionParser()
o.set_usage('omni_apply.py [options] *uvcRRE')
o.set_description(__doc__)
a.scripting.add_standard_options(o,pol=True)
o.add_option('--omnipath',dest='omnipath',default='*.fits',type='string',
            help='Filename or format string that gets passed to glob for omnical/firstcal solution fits files.')
o.add_option('--median', action='store_true',
            help='Take the median in time before applying solution. Applicable only in delay.')
o.add_option('--firstcal', action='store_true', 
            help='Applying firstcal solutions.')
opts,args = o.parse_args(sys.argv[1:])
args = np.sort(args)
pols = opts.pol.split(',')

def file2pol(filename): return filename.split('.')[3] #XXX assumes file naming format
def file2djd(filename): return re.findall("\d+\.\d+",filename)[0]
def isLinPol(polstr): return len(list(set(polstr)))==1

jonesLookup = {
    -5: (-5,-5),
    -6: (-6,-6),
    -7: (-5,-6),
    -8: (-6,-5)
}

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
            firstcal_files[pp] = sorted([s for s in glob.glob(opts.omnipath) if pp in s])
            nf += len(firstcal_files[pp])    

for i, f in enumerate(args):
    pp = file2pol(f)
    djd = file2djd(f)
    if not opts.firstcal:
        if len(pols)==1:
            # atomic solution application
            fexpected = solution_files[i] #XXX this is fragile
        else:
            # one solution file per djd
            fexpected = next((s for s in solution_files if djd in s),None)
        try:
            ind = solution_files.index(fexpected)
            filedict[f] = str(solution_files[ind])
        except ValueError:
           raise Exception('Solution file %s expected; not found.'%fexpected)
    
    
    else:
        if nf == len(solution_files)*len(pols): # atomic firstcal application
            filedict[f] = solution_files[i] # XXX this is fragile
        else: # one firstcal file for many data files
            if isLinPol(pp):
                filedict[f] = [firstcal_files[pp][0]]
            else:
                filedict[f] = [firstcal_files[lpk][0] for lpk in linear_pol_keys]

for f in args:
    import IPython;IPython.embed()
    mir = pyuvdata.UVData()
    print "  Reading {0}".format(f)
    mir.read_miriad(f)
    if mir.phase_type != 'drift':
        mir.unphase_to_drift()
    cal = pyuvdata.UVCal()
    print "  Reading calibration : {0}".format(filedict[f])
    if len(pols)==1 or not opts.firstcal:
        cal.read_calfits(filedict[f])
    else:
        if isLinPol(file2pol(f)):
            cal.read_calfits(filedict[f][0])
        else:
            cal = heracal.omni.concatenate_UVCal_on_pol(filedict[f])
    
    print "  Calibrating..."
    antenna_index = dict(zip(*(cal.ant_array,range(cal.Nants_data))))
    for p,pol in enumerate(mir.polarization_array):
        p1,p2 = [list(cal.jones_array).index(pk) for pk in jonesLookup[pol]] #XXX could replace with numpy function instead of casting to list
        for bl,k in zip(*np.unique(mir.baseline_array, return_index=True)):
            blmask = np.where(mir.baseline_array == bl)[0]
            ai, aj = mir.baseline_to_antnums(bl)
            if not ai in cal.ant_array or not aj in cal.ant_array:
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
                    if opts.median:
                        mir.data_array[blmask, nsp, :, p] =  \
                                    mir.data_array[blmask, nsp, :, p] * \
                                    heracal.omni.get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[ai], nsp, :, p1])).reshape(1,-1) * \
                                    np.conj(heracal.omni.get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[aj], nsp, :, p2])).reshape(1,-1))
                    else:
                        mir.data_array[blmask, nsp, :, p] =  \
                                    mir.data_array[blmask, nsp, :, p] * \
                                    heracal.omni.get_phase(cal.freq_array, cal.delay_array[antenna_index[ai], nsp, :, p1]) * \
                                    np.conj(heracal.omni.get_phase(cal.freq_array, cal.delay_array[antenna_index[aj], nsp, :, p2]))

                if cal.cal_type == 'delay' and cal.gain_convention == 'divide':
                    if opts.median:
                        mir.data_array[blmask, nsp, :, p] =  \
                                    mir.data_array[blmask, nsp, :, p] / \
                                    heracal.omni.get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[ai], nsp, :, p1])).reshape(1,-1) / \
                                    np.conj(heracal.omni.get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[aj], nsp, :, p2])).reshape(1,-1))
                    else:
                        mir.data_array[blmask, nsp, :, p] =  \
                                    mir.data_array[blmask, nsp, :, p] / \
                                    heracal.omni.get_phase(cal.freq_array, cal.delay_array[antenna_index[ai], nsp, :, p1]).T / \
                                    np.conj(heracal.omni.get_phase(cal.freq_array, cal.delay_array[antenna_index[aj], nsp, :, p2]).T)
                 



    if opts.firstcal:
        print " Writing {0}".format(f+'F')
        mir.write_miriad(f +'F') 
    else:
        print " Writing {0}".format(f+'O')
        mir.write_miriad(f +'O') 

