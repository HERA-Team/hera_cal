#! /usr/bin/env python

import aipy
import numpy as np
from heracal.omni import from_fits, aa_to_info, run_omnical, compute_xtalk, HERACal
from heracal.miriad import read_files
import pyuvdata
import optparse
import os, sys, glob

o = optparse.OptionParser()
o.set_usage("omni_run.py [options] *pp.uvcRRE (only provide one polarization's-worth of files. This script assumes other polarizations of the same file title are in the same directory as the respective arguments, and finds them itself)") #XXX should this detail go in the description?
o.set_description(__doc__)
aipy.scripting.add_standard_options(o, cal=True, pol=True)
o.add_option('--omnipath', dest='omnipath', default='', type='string',
             help='Path to save omnical solutions.')
o.add_option('--ex_ants', dest='ex_ants', default=None,
             help='Antennas to exclude, separated by commas.')
o.add_option('--firstcal', dest='firstcal', type='string',
             help='Path and name of firstcal file. Can pass in wildcards.')
o.add_option('--minV',action='store_true',
            help='Toggle V minimization capability. This only makes sense in the case of 4-pol cal, which will set crosspols (xy & yx) equal to each other')
opts, args = o.parse_args(sys.argv[1:])

if len(args)==0:
    raise AssertionError('Please provide visibility files.')

args = np.sort(args)
pols = opts.pol.split(',')

if opts.minV and len(list(set(''.join(pols))))==1:
    raise AssertionError('Stokes V minimization requires crosspols in the "-p" option.')

def write_uvdata_vis(filename, aa, m, v, xtalk=False, returnuv=True):
    '''
    Given meta information and visibilities, write out a uvfits file.
    filename: filename to write uvfits object.(str)
    aa      : aipy antenna array object (object)
    m       : dictionary of information (dict)
    v       : dictionary of visibilities with keys antenna pair and pol (dict)
    xtalk   : visibilities given are xtalk visibilities. (bool)
    returnuv: return uvdata object. If returned, file is not written. (bool)
    '''

    pols = v.keys()
    bls = v[pols[0]].keys()

    uv = pyuvdata.UVData()
    if xtalk:
        uv.Ntimes = 1
    else:
        uv.Ntimes = len(m['times'])
    uv.Npols = len(pols)
    uv.Nbls = len(v[pols[0]].keys())
    uv.Nblts = uv.Nbls * uv.Ntimes
    uv.Nfreqs = len(m['freqs'])
    data = {}
    for p in v.keys():
        if p not in data.keys():
            data[p] = []
        for bl in v[p].keys():
            data[p].append(v[p][bl])
        data[p] = np.array(data[p]).reshape(uv.Nblts, uv.Nfreqs)

    uv.data_array = np.expand_dims(np.array([data[p] for p in pols]).T.swapaxes(0, 1), axis=1)
    uv.vis_units = 'uncalib'
    uv.nsample_array = np.ones_like(uv.data_array, dtype=np.float)
    uv.flag_array = np.zeros_like(uv.data_array, dtype=np.bool)
    uv.Nspws = 1  # this is always 1 for paper and hera(currently)
    uv.spw_array = np.array([uv.Nspws])
    blts = np.array([bl for bl in bls for i in range(uv.Ntimes)])
    if xtalk:
        uv.time_array = np.array(list(m['times'][:1]) * uv.Nbls)
        uv.lst_array = np.array(list(m['lsts'][:1]) * uv.Nbls)
    else:
        uv.time_array = np.array(list(m['times']) * uv.Nbls)
        uv.lst_array = np.array(list(m['lsts']) * uv.Nbls)

    # generate uvw
    uvw = []
    for t in range(uv.Ntimes):
        for bl in bls:
            uvw.append(aa.gen_uvw(*bl, src='z').reshape(3, -1))
    uv.uvw_array = np.array(uvw).reshape(-1, 3)

    uv.ant_1_array = blts[:, 0]
    uv.ant_2_array = blts[:, 1]
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array,
                                               uv.ant_2_array)

    uv.freq_array = m['freqs'].reshape(1, -1) * 1e9  # Turn into MHz.
    poldict = {'xx': -5, 'yy': -6, 'xy': -7, 'yx': -8}
    uv.polarization_array = np.array([poldict[p] for p in pols])
    if xtalk:
        # xtalk integration time is averaged over the whole file
        uv.integration_time = m2['inttime'] * len(m2['times'])
    else:
        uv.integration_time = m2['inttime']
    uv.channel_width = np.diff(uv.freq_array[0])[0]

    # observation parameters
    uv.object_name = 'zenith'
    uv.telescope_name = 'HERA'
    uv.instrument = 'HERA'
    tobj = pyuvdata.uvtel.get_telescope(uv.telescope_name)
    uv.telescope_location = tobj.telescope_location
    uv.history = m2['history']

    # phasing information
    uv.phase_type = 'drift'
    uv.zenith_ra = uv.lst_array
    uv.zenith_dec = np.array([aa.lat] * uv.Nblts)

    # antenna information
    uv.Nants_telescope = 128
    uv.antenna_numbers = np.arange(uv.Nants_telescope, dtype=int)
    uv.Nants_data = len(np.unique(np.concatenate([uv.ant_1_array, uv.ant_2_array]).flatten()))
    uv.antenna_names = np.array(['ant{0}'.format(ant) for ant in uv.antenna_numbers])
    antpos = []
    for k in aa:
        antpos.append(k.pos)

    uv.antenna_positions = np.array(antpos)

    if returnuv:
        return uv
    else:
        print('   Saving {0}'.format(filename))
        uv.write_uvfits(filename, force_phase=True, spoof_nonessential=True)

# Create info
# generate reds from calfile
aa = aipy.cal.get_aa(opts.cal, np.array([.15]))
print 'Getting reds from calfile'
if opts.ex_ants:  # assumes exclusion of the same antennas for every pol
    ex_ants = map(int,opts.ex_ants.split(','))
    print '   Excluding antennas:', sorted(ex_ants)
else:
    ex_ants = []
info = aa_to_info(aa, pols=list(set(''.join(pols))), ex_ants=ex_ants, crosspols=pols, minV=opts.minV)
reds = info.get_reds()
bls = [bl for red in reds for bl in red]

# Dictionary of calpar gains and files
fcalfiles = {}
firstcal_files = {}
if not opts.firstcal:
    #XXX this requires a firstcal file for any implementation
    raise ValueError('Please provide a firstcal file. Exiting...')
else:
    for pp in pols:
        if not len(list(set(pp))) > 1:
            # we cannot use cross-pols to firstcal
            if '*' in opts.firstcal:
                firstcal_files[pp] = np.sort([s for s in glob.glob(opts.firstcal) if pp in s])
            elif ',' in opts.firstcal:
                firstcal_files[pp] = np.sort([s for s in opts.firstcal.split(',') if pp in s])             
            else:
                firstcal_files[pp] = [str(opts.firstcal)]
                if len(firstcal_files)==0:
                    raise ValueError('Cannot parse --firstcal argument')
#import IPython;IPython.embed()
for f, filename in enumerate(args):
    fcalfiles[filename] = {}
    if len(firstcal_files) == len(args) or (len(firstcal_files) == 2*len(args) and len(pols)==2):  # each data file has its own firstcal file.
        print 'Each data file has its own firstcal file' #XXX remember to delete
        for pp in pols:
            #if not pp in files[filename]: 
            fcalfiles[filename][pp] = {}
            if len(list(set(pp))) == 1:
                fcalfiles[filename][pp]['firstcal'] = str(firstcal_files[pp][f])
            elif len(list(set(pp))) == 2:
                try:
                    fcalfiles[filename][pp]['firstcal'] = [str(firstcal_files[pp[0]*2][f]), str(firstcal_files[pp[1]*2][f])]
                except(IndexError):
                    raise Exception("Provide firstcal files for both linear pols if cross pols require calibration.")
            else:
                raise IOError("Provide valid polarization.")
    else:  # use firstcal file for all input files
        print 'Use one firstcal file for all input files' #XXX remember to delete
        for pp in pols:
            #if not pp in files[filename]:
            fcalfiles[filename][pp] = {}
            if len(list(set(pp))) == 1:
                fcalfiles[filename][pp]['firstcal'] = str(firstcal_files[pp][0])
                #files[filename]['firstcal'] = str(firstcal_files[pp][0])
            elif len(list(set(pp))) == 2:
                try:
                    fcalfiles[filename][pp]['firstcal'] = [str(firstcal_files[pp[0]*2][0]), str(firstcal_files[pp[1]*2][0])]
                    #files[filename]['firstcal'] = [str(firstcal_files[pp[0]*2][0]), str(firstcal_files[pp[1]*2][0])]
                except(IndexError):
                    raise Exception("Provide firstcal files for both linear pols if cross pols require calibration.")
            else:
                raise IOError("Provide valid polarization.")

### Omnical-ing! Loop Through Files ###
for f, filename in enumerate(args):
    #file_group = files[filename]  # dictionary with pol indexed files
    file_group = {}
    
    for pp in pols:
        fn = filename.split('.')
        fn[3] = pp
        file_group[pp] = '.'.join(fn)
    
    print 'Reading:'
    for key in file_group.keys():
        print '   ' + file_group[key]
    
    for pp in pols:
        filename_pp = file_group[pp]
        fitsname = '%s/%s.fits'%(opts.omnipath,filename_pp)# Full filename + .fits to keep notion of history.
        if os.path.exists(fitsname):
            print '   %s exists. Skipping...' % fitsname
            continue
        print 'Reading %s'%str(fcalfiles[filename][pp]['firstcal'])
        _, g0, _, _ = from_fits(fcalfiles[filename][pp]['firstcal'])  # read in firstcal data
        
        uvd = pyuvdata.UVData()
        uvd.read_miriad(filename_pp)
        uvd.select(times=np.unique(uvd.time_array)[:3])
        t_jd = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
        t_lst = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
        freqs = uvd.freq_array[0]
        SH = (uvd.Ntimes, uvd.Nfreqs)  # shape of file data (ex: (19,203))
        d, f = {}, {}
        for p in g0.keys():
            for i in g0[p]:
                if g0[p][i].shape != (len(t_jd), len(freqs)):
                    g0[p][i] = np.resize(g0[p][i], SH)  # resize gains like data
                else: continue

        for ip, pol in enumerate(uvd.polarization_array):
            pol = aipy.miriad.pol2str[pol]
            if pol != opts.pol:
                continue
            for nbl, (i,j) in enumerate(map(uvd.baseline_to_antnums, uvd.baseline_array[:uvd.Nbls])):
                if not (i, j) in bls and not (j, i) in bls:
                    continue
                d[(i,j)] = {pol: uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nspws, uvd.Nfreqs, uvd.Npols)[:, nbl, 0, :, ip]}
                f[(i,j)] = {pol: np.logical_not(uvd.flag_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nspws, uvd.Nfreqs, uvd.Npols)[:, nbl, 0, :, ip])}

        print '   Running Omnical'
        m2, g3, v3 = run_omnical(d, info, gains0=g0)

        # need wgts for xtalk
        wgts, xtalk = {}, {}
        for pp in pols:
            wgts[pp] = {}  # weights dictionary by pol
            for i,j in f:
                if (i,j) in bls:
                    wgts[pp][(i, j)] = np.logical_not(f[i,j][pp]).astype(np.int)
                else:  # conjugate
                    wgts[pp][(j, i)] = np.logical_not(f[i,j][pp]).astype(np.int)

        xtalk = compute_xtalk(m2['res'], wgts)  # xtalk is time-average of residual
        m2['history'] = 'OMNI_RUN: ' + ''.join(sys.argv) + '\n'
        m2['times'] = t_jd
        m2['lsts'] = t_lst
        m2['freqs'] = freqs
        m2['inttime'] = uvd.integration_time
        optional = {'observer': 'obs (obs@institution.edu)'}

        print '   Saving %s' % fitsname
        hc = HERACal(m2, g3, ex_ants=ex_ants, optional=optional)
        hc.write_calfits(fitsname)
        write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.vis.fits', aa, m2, v3, returnuv=False)
        write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits', aa, m2, xtalk, xtalk=True, returnuv=False)
