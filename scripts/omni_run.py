#! /usr/bin/env python

import omnical
import aipy
import numpy as np
from heracal.omni import from_fits, aa_to_info, run_omnical, compute_xtalk, HERACal
from heracal.miriad import read_files
import pickle
import optparse
import os
import sys
import pyuvdata
import glob

o = optparse.OptionParser()
o.set_usage('omni_run.py [options] *uvcRRE')
o.set_description(__doc__)
aipy.scripting.add_standard_options(o, cal=True, pol=True)
o.add_option('--omnipath', dest='omnipath', default='', type='string',
             help='Path to save omnical solutions. Include final / in path.')
o.add_option('--ex_ants', dest='ex_ants', default=None,
             help='Antennas to exclude, separated by commas.')
o.add_option('--firstcal', dest='firstcal', type='string',
             help='Path and name of firstcal file. Can pass in wildcards.')
opts, args = o.parse_args(sys.argv[1:])
args = np.sort(args)


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
        uv.Ntimes = len(m['jds'])
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
        uv.time_array = np.array(list(m['jds'][:1]) * uv.Nbls)
        uv.lst_array = np.array(list(m['lsts'][:1]) * uv.Nbls)
    else:
        uv.time_array = np.array(list(m['jds']) * uv.Nbls)
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
        uv.integration_time = m2['inttime'] * len(m2['jds'])
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


# Dictionary of calpar gains and files
pols = opts.pol.split(',')
files = {}
if not opts.firstcal:
    raise ValueError('Please provide a firstcal file. Exiting...')
else:
    firstcal_files = np.sort(glob.glob(opts.firstcal))
for f, filename in enumerate(args):
    files[filename] = {}
    if len(firstcal_files) == len(args):
        files[filename]['firstcal'] = str(firstcal_files[f])  # one firstcal file per input file.
    elif len(args) > len(firstcal_files):
        files[filename]['firstcal'] = str(firstcal_files[0])  # only use first firstcal file for all input files.
    for p in pols:
        fn = filename.split('.')
        fn[3] = p
        files[filename][p] = '.'.join(fn)
    
# Create info
# generate reds from calfile
aa = aipy.cal.get_aa(opts.cal, np.array([.15]))
print 'Getting reds from calfile'
if opts.ex_ants:  # assumes exclusion of the same antennas for every pol
    ex_ants = []
    for a in opts.ex_ants.split(','):
        ex_ants.append(int(a))
    print '   Excluding antennas:', sorted(ex_ants)
else:
    ex_ants = []
info = aa_to_info(aa, pols=list(set(''.join(pols))), ex_ants=ex_ants, crosspols=pols)
reds = info.get_reds()
bls = [bl for red in reds for bl in red]

### Omnical-ing! Loop Through Files ###
for f, filename in enumerate(args):
    file_group = files[filename]  # dictionary with pol indexed files
    print 'Reading:'
    for key in file_group.keys():
        print '   ' + file_group[key]

    fitsname = opts.omnipath + filename + '.fits'  # Full filename + .fits to keep notion of history.
    if os.path.exists(fitsname):
        print '   %s exists. Skipping...' % fitsname
        continue
    
    _, g0, _, _ = from_fits(file_group['firstcal'])  # read in firstcal data

    # timeinfo, d, f = read_files([file_group[key] for key in file_group.keys() if key != 'firstcal'],
    #                            antstr='cross', polstr=opts.pol)
    file_pol = filename.split('/')[-1].split('.')[3]
    uvd = pyuvdata.UVData()
    uvd.read_miriad(file_group[file_pol])
    t_jd = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
    t_lst = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
    freqs = uvd.freq_array[0]
    SH = (uvd.Ntimes, uvd.Nfreqs)  # shape of file data (ex: (19,203))
    data, wgts, xtalk = {}, {}, {}
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
    

    data = d  # indexed by bl and then pol (backwards from everything else)
    for p in pols:
        wgts[p] = {}  # weights dictionary by pol
        for i,j in f:
            if (i,j) in bls:
                wgts[p][(i, j)] = np.logical_not(f[i,j][p]).astype(np.int)
            else:  # conjugate
                wgts[p][(j, i)] = np.logical_not(f[i,j][p]).astype(np.int)
    print '   Running Omnical'
    m2, g3, v3 = run_omnical(data, info, gains0=g0)

    xtalk = compute_xtalk(m2['res'], wgts)  # xtalk is time-average of residual
    m2['history'] = 'OMNI_RUN: ' + ''.join(sys.argv) + '\n'
    m2['jds'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs
    m2['inttime'] = uvd.integration_time
    optional = {'observer': 'Zaki Ali (zakiali@berkeley.edu)'}

    print '   Saving %s' % fitsname
    hc = HERACal(m2, g3, optional=optional)
    hc.write_calfits(fitsname)
    write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.vis.fits', aa, m2, v3, returnuv=False)
    write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits', aa, m2, xtalk, xtalk=True, returnuv=False)
