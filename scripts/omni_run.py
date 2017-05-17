#! /usr/bin/env python

import aipy
import numpy as np
from heracal.omni import from_fits, aa_to_info, run_omnical, compute_xtalk, HERACal, write_uvdata_vis
from heracal.miriad import read_files
import pyuvdata
import optparse
import os, sys, glob

o = optparse.OptionParser()
o.set_usage("omni_run.py [options] *.uvcRRE")
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

args = np.sort(args)
pols = opts.pol.split(',')

if len(args)==0: raise AssertionError('Please provide visibility files.')
if opts.minV and len(list(set(''.join(pols))))==1: raise AssertionError('Stokes V minimization requires crosspols in the "-p" option.')

def getPol(fname): return fname.split('.')[3] #XXX assumes file naming format
def linPol(polstr): return len(list(set(polstr)))==1

linear_pol_keys = []
for pp in pols:
    if linPol(pp):
        linear_pol_keys.append(pp)

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

### Collect all firstcal files ###
firstcal_files = {}
if not opts.firstcal:
    raise ValueError('Please provide a firstcal file. Exiting...') #XXX this requires a firstcal file for any implementation
else:
    Nf=0
    for pp in pols:
        if linPol(pp):
            # we cannot use cross-pols to firstcal
            if '*' in opts.firstcal or '?' in opts.firstcal: flist = glob.glob(opts.firstcal)    
            elif ',' in opts.firstcal:flist = opts.firstcal.split(',')             
            else: flist = [str(opts.firstcal)]
            firstcal_files[pp] = np.sort([s for s in flist if pp in s])
            Nf += len(firstcal_files[pp])

### Match firstcal files according to mode of calibration ###
filesByPol = {}
for pp in pols: filesByPol[pp] = []
file2firstcal = {}

for f, filename in enumerate(args):
    if Nf == len(args)*len(pols): fi = f # atomic firstcal application
    else: fi = 0 # one firstcal file serves all visibility files
    pp = getPol(filename)
    if linPol(pp): file2firstcal[filename] = [firstcal_files[pp][fi]]
    else: file2firstcal[filename] = [firstcal_files[lpk][fi] for lpk in linear_pol_keys]
    filesByPol[pp].append(filename)

# XXX can these be combined into one loop?

### Execute Omnical stages ###
for filenumber in range(len(args)/len(pols)):
    file_group = {} #there is one file_group per djd
    for pp in pols: file_group[pp] = filesByPol[pp][filenumber]
    if len(pols) == 1:
        bname = os.path.basename(file_group[pols[0]])
    else:
        bname = os.path.basename(file_group[pols[0]]).replace('.%s'%pols[0],'')
    fitsname = '%s/%s.fits'%(opts.omnipath, bname)
    
    if os.path.exists(fitsname):
        print '   %s exists. Skipping...' % fitsname
        continue
    
    # get correct firstcal files
    # XXX not a fan of the way this is done, open to suggestions
    fcalfile = None
    if len(pols) == 1: #single pol
        fcalfile = file2firstcal[file_group[pols[0]]]
    else:
        for pp in pols: #4 pol
            if pp not in linear_pol_keys:
                fcalfile = file2firstcal[file_group[pp]]
                break
    if not fcalfile: #2 pol
        fcalfile = [file2firstcal[file_group[pp]][0] for pp in linear_pol_keys]
       
    _, g0, _, _ = from_fits(fcalfile)
    
    #uvd = pyuvdata.UVData()
    #uvd.read_miriad([file_group[pp] for pp in pols])
    #XXX THIS WILL BECOME MUCH SIMPLER WHEN PYUVDATA CAN READ MULTIPLE MIRIAD FILES AT ONCE
    
    ## collect metadata -- should be the same for each file
    f0 = file_group[pols[0]]
    uvd = pyuvdata.UVData()
    uvd.read_miriad(f0)
    t_jd = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
    t_lst = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:,0]
    t_int = uvd.integration_time
    freqs = uvd.freq_array[0]
    SH = (uvd.Ntimes, uvd.Nfreqs)  # shape of file data (ex: (19,203))
    
    uvd_dict = {}
    for pp in pols:
        uvd = pyuvdata.UVData()
        uvd.read_miriad(file_group[pp])
        uvd_dict[pp] = uvd
    
    ## format g0 for application to data
    for p in g0.keys():
        for i in g0[p]:
            if g0[p][i].shape != (len(t_jd), len(freqs)): #not a big fan of this if/else
                g0[p][i] = np.resize(g0[p][i], SH)  # resize gains like data
            else: continue
    
    ## read data into dictionaries
    d,f = {},{}
    for ip,pp in enumerate(pols):
        uvdp = uvd_dict[pp]
        
        if ip==0:
            for nbl, (i,j) in enumerate(map(uvdp.baseline_to_antnums, uvdp.baseline_array[:uvdp.Nbls])):
                if not (i, j) in bls and not (j, i) in bls: continue
                d[(i,j)] = {}
                f[(i,j)] = {}
        
        #XXX I *really* don't like looping again, but I'm not sure how better to do it
        for nbl, (i,j) in enumerate(map(uvdp.baseline_to_antnums, uvdp.baseline_array[:uvdp.Nbls])):
            if not (i, j) in bls and not (j, i) in bls: continue
            d[(i,j)][pp] = uvdp.data_array.reshape(uvdp.Ntimes, uvdp.Nbls, uvdp.Nspws, uvdp.Nfreqs, uvdp.Npols)[:, nbl, 0, :, 0]
            f[(i,j)][pp] = np.logical_not(uvdp.flag_array.reshape(uvdp.Ntimes, uvdp.Nbls, uvdp.Nspws, uvdp.Nfreqs, uvdp.Npols)[:, nbl, 0, :, 0])
    
    ## Finally prepared to run omnical
    print '   Running Omnical'
    m2, g3, v3 = run_omnical(d, info, gains0=g0)
    
    ## Collect weights for xtalk
    wgts, xtalk = {}, {}
    for pp in pols:
        wgts[pp] = {}  # weights dictionary by pol
        for i,j in f:
            if (i,j) in bls:
                wgts[pp][(i, j)] = np.logical_not(f[i,j][pp]).astype(np.int)
            else:  # conjugate
                wgts[pp][(j, i)] = np.logical_not(f[i,j][pp]).astype(np.int)
    # xtalk is time-average of residual: data - omnical model
    xtalk = compute_xtalk(m2['res'], wgts)  
    
    ## Append metadata parameters 
    m2['history'] = 'OMNI_RUN: '+' '.join(sys.argv) + '\n'
    m2['times'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs
    m2['inttime'] = t_int
    optional = {'observer': 'obs (example@institution.edu)'}

    print '   Saving %s' % fitsname
    hc = HERACal(m2, g3, ex_ants = ex_ants,  optional=optional)
    hc.write_calfits(fitsname)
    fsj = '.'.join(fitsname.split('.')[:-1])
    write_uvdata_vis('%s.vis.fits'%fsj, aa, m2, v3, returnuv=False)
    write_uvdata_vis('%s.xtalk.fits'%fsj, aa, m2, xtalk, xtalk=True, returnuv=False)

