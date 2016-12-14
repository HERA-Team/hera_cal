import numpy as np
import subprocess
import datetime
from astropy.io import fits
from uvdata import CALFITS
import omni
import sys
import os

def writefits(outfn, meta, gains, vismdl, xtalk, repopath=None, ex_ants=[], name_dict={}):
    '''
    This function writes the solution from output npz files from omni_run to a fits file.
    npzfiles can be a list of npz files with solutions for different polarizations but for
    the same obs id. repopath is for writing the program of origin and git hash, e.g., if
    the solution comes from capo, then repopath=/path/to/capo/. ex_ants is used to indicate
    which antennas are flagged. name_dict is for writing antenna names to fits.
    '''
    p2pol = {'EE': 'x', 'NN': 'y', 'EN': 'cross', 'NE': 'cross'}

    if os.path.exists(outfn):
        print '   %s exists, skipping...' % outfn
        return 0

    today = datetime.date.today().strftime("Date: %d, %b %Y")
    if not repopath == None:
        githash = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repopath)
        ori = subprocess.check_output(['git','remote','show','origin'], cwd=repopath)
        ori = ori.split('\n')[1].split(' ')[-1]
        githash = githash.replace('\n','')
    else:
        githash = ''
        ori = ''

    chisqdict = {}
    datadict = {}
    ants = []
    for pol in gains:
        for ant in gains[pol]:
            datadict['%d%s' %(ant,pol)] = gains[pol][ant]
            if not ant in ants:
                ants.append(ant)

    ants.sort()
    if name_dict == {}: tot = ants + ex_ants
    else: tot = name_dict.keys()
    tot.sort()
    time = meta['jds']
    freq = meta['freqs']/1e6
    pol = ['EE', 'NN', 'EN', 'NE']
    nt = time.shape[0]
    nf = freq.shape[0]
    na = len(tot)
    nam = []
    for nn in range(0,na):
        try: nam.append(name_dict[tot[nn]])
        except(KeyError): nam.append('ant'+str(tot[nn]))
    datarray = []
    flgarray = []
    for ii in range(0,4):
        dd = []
        fl = []
        for jj in range(0,na):
            try: dd.append(datadict[str(tot[jj])+p2pol[pol[ii]]])
            except(KeyError): dd.append(np.ones((nt,nf)))
            if tot[jj] in ex_ants: fl.append(np.ones((nt,nf),dtype=bool))
            else: fl.append(np.zeros((nt,nf),dtype=bool))
        datarray.append(dd)
        flgarray.append(fl)
    datarray = np.array(datarray)
    #import IPython; IPython.embed()
    datarray = datarray.swapaxes(0,3).swapaxes(0,1)
    flgarray = np.array(flgarray)
    flgarray = flgarray.swapaxes(0,3).swapaxes(0,1)
    tarray = np.resize(time,(4*nf*na,nt)).transpose()
    parray = np.array((['EE']*(nf*na)+['NN']*(nf*na)+['EN']*(nf*na)+['NE']*(nf*na))*nt)
    farray = np.array(list(np.resize(freq,(na,nf)))).transpose()
    numarray = np.array(tot*4*nt*nf)
    namarray = np.array(nam*4*nt*nf)

    chisqarray = []
    for i in range(4):
        ch = []
        for jj in range(na):
            try:
                ch.append(meta['chisq'+str(tot[jj])+p2pol[pol[ii]]])
            except(KeyError): 
                ch.append(np.ones((nt,nf))) # if key error (b/c) of pol
        chisqarray.append(ch)

    chisqarray = np.array(chisqarray)
    chisqarray = chisqarray.swapaxes(0,3).swapaxes(0,1)

    calfits_object = CALFITS()
    calfits_object.set_gain()

    calfits_object.Nfreqs = nf
    calfits_object.Npols = len(pol)
    calfits_object.Ntimes = nt
    calfits_object.history = ''
    calfits_object.Nants_data = na  # what value?
    calfits_object.antenna_names = namarray
    calfits_object.antenna_numbers = numarray
    calfits_object.Nants_telescope = na
    calfits_object.Nspws = 1

    calfits_object.freq_array = farray
    calfits_object.polarization_array = range(-8, -4)[::-1]
    calfits_object.time_array = tarray
    calfits_object.gain_convention = 'divide'
    calfits_object.flag_array = flgarray
    calfits_object.quality_array = flgarray  # what is this array supposed to be?
    calfits_object.cal_type = 'gain'
    calfits_object.x_orientation = 'E'
    calfits_object.gain_array = datarray
    calfits_object.quality_array = chisqarray




    
def read_fits(filename, pols):
    ### This function reads in the solution from fits file, which returns a dictionary of polarization,  ###
    ### each polarization is a dictionary of antenna indexes, which has a value as an numpy array with   ###
    ### the shape (Ntimes, Nfreqs)                                                                       ###
    g0 = {}
    poldict = {'EE': 'xx', 'NN': 'yy', 'EN': 'xy', 'NE': 'yx'}
    hdu = fits.open(filename)
    Ntimes = hdu[0].header['NTIMES']
    Nfreqs = hdu[0].header['NFREQS']
    Npols = hdu[0].header['NPOLS']
    Nants = hdu[0].header['NANTS']
    ant_index = hdu[1].data['ANT INDEX'][0:Nants]
    pol_list = hdu[1].data['POL'][0:Nfreqs*Nants*Npols].reshape(Npols,Nants*Nfreqs)[:,0]
    data_list = hdu[1].data['GAIN'].reshape((Ntimes,Npols,Nfreqs,Nants)).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2) #Npols,Nants,Ntimes,Nfreqs
    for ii in range(0,Npols):
        polarization = poldict[pol_list[ii]]
        if not polarization in pols: continue
        g0[polarization[0]] = {}
        for jj in range(0,Nants):
            g0[polarization[0]][ant_index[jj]]=data_list[ii][jj]
    return g0


def fc_gains_to_fits(npznames,filename,repopath=None,name_dict={}):
    #### For firstcal solution without time axis ###
    if not repopath == None:
        githash = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repopath)
        ori = subprocess.check_output(['git','remote','show','origin'], cwd=repopath)
        ori = ori.split('\n')[1].split(' ')[-1]
        githash = githash.replace('\n','')
    else:
        ori = ''
        githash = ''
    today = datetime.date.today().strftime("Date: %d, %b %Y")
    outname = '%s.fc.fits'%filename
    print outname
    if os.path.exists(outname):
        print '   %s exists, skipping...' % outname
        return 0
    datadict = {}
    ant = []
    for npz in npznames:
        data = np.load(npz)
        for ii, ss in enumerate(data):
            if ss[0].isdigit():
                datadict[ss] = data[ss][0]
                intss = int(ss[0:-1])
                if not intss in ant:
                    ant.append(intss)
    ant.sort()
    try: ex_ants = list(data['ex_ants'])
    except(KeyError): ex_ants = []
    if name_dict == {}: tot = ant + ex_ants
    else: tot = name_dict.keys()
    tot.sort()
    freq = data['freqs']/1e6  #in MHz
    pol = ['x', 'y']
    Na = len(tot)
    Nf = freq.shape[0]
    parray = np.array(['x']*Nf*Na+['y']*Nf*Na)
    farray = np.array(list(np.resize(freq,(Na,Nf)).transpose().reshape(Na*Nf))*2)
    datarray = []
    flgarray = []
    for ii in range(0,2):
        dd = []
        fl = []
        for jj in range(0,Na):
            try: dd.append(datadict[str(tot[jj])+pol[ii]])
            except(KeyError): dd.append(np.ones((Nf),dtype=float))
            if tot[jj] in ex_ants: fl.append(np.ones((Nf),dtype=bool))
            else: fl.append(np.zeros((Nf),dtype=bool))
        datarray.append(dd)
        flgarray.append(fl)
    datarray = np.array(datarray)
    datarray = datarray.swapaxes(1,2).reshape(2*Nf*Na)
    flgarray = np.array(flgarray)
    flgarray = flgarray.swapaxes(1,2).reshape(2*Nf*Na)
    nam = []
    for nn in range(0,Na):
        try: nam.append(name_dict[tot[nn]])
        except(KeyError): nam.append('ant'+str(tot[nn]))
    numarray = np.array(tot*2*Nf)
    namarray = np.array(nam*2*Nf)

    prihdr = fits.Header()
    prihdr['DATE'] = today
    prihdr['ORIGIN'] = ori
    prihdr['HASH'] = githash
    prihdr['PROTOCOL'] = 'Divide uncalibrated data by these gains to obtain calibrated data.'
    prihdr['NFREQS'] = Nf
    prihdr['NANTS'] = Na
    prihdr['NPOLS'] = 2
    prihdu = fits.PrimaryHDU(header=prihdr)
    colnam = fits.Column(name='ANT NAME', format='A10', array=namarray)
    colnum = fits.Column(name='ANT INDEX', format='I',array=numarray)
    colf = fits.Column(name='FREQ (MHZ)', format='E', array=farray)
    colp = fits.Column(name='POL', format='A4', array=parray)
    coldat = fits.Column(name='GAIN', format='M', array=datarray)
    colflg = fits.Column(name='FLAG', format='L', array=flgarray)
    cols = fits.ColDefs([colnam, colnum, colf, colp, coldat, colflg])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist = fits.HDUList([prihdu, tbhdu])
    hdulist.writeto(outname)

def fc_gains_from_fits(filename):
### for reading firstcal solution without time axis ###
    g0 = {}
    hdu = fits.open(filename)
    Nfreqs = hdu[0].header['NFREQS']
    Npols = hdu[0].header['NPOLS']
    Nants = hdu[0].header['NANTS']
    ant_index = hdu[1].data['ANT INDEX'][0:Nants]
    pol_list = hdu[1].data['POL'].reshape(Npols,Nants*Nfreqs)[:,0]
    data_list = hdu[1].data['GAIN'].reshape((Npols, Nfreqs, Nants)).swapaxes(1,2)
    for ii in range(0,Npols):
        p = pol_list[ii]
        g0[p] = {}
        for jj in range(0,Nants):
            g0[p][ant_index[jj]] = data_list[ii][jj]
    return g0
