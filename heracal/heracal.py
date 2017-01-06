from uvdata import UVCal


class HERACal(UVCal):
    '''Class that loads in hera omnical data,'''
    def load_omnical(self, meta, gains, ex_ants = []):
        '''given meta and gain dictionary from omni_run.py before to_npz() call,
           to reshuffle data and populating the UVCal class.'''
        
    chisqdict = {}
    datadict = {}
    ants = []
    for pol in gains:
        for ant in gains[pol].sort():
            datadict['%d%s' %(ant,pol)] = gains[pol][ant]
            if not ant in ants:
                ants.append(ant)

    allants = ants + ex_ants
    allants = allants.sort()
    time = meta['jds']
    freq = meta['freqs']/1e6 # in GHz
    pols = ['x', 'y']; npol = len(pol)
    ntimes = time.shape[0]
    nfreqs = freq.shape[0]
    nants = len(allants)
    antnames = ['ant'+str(ant) for ant in allants]
    datarray = []
    flgarray = []
    for ii in range(npol):
        dd = []
        fl = []
        for jj in range(nants):
            try: dd.append(datadict[str(allants[jj])+pols[ii]])
            except(KeyError): dd.append(np.ones((ntimes,nfreqs)))
            if allants[jj] in ex_ants: fl.append(np.ones((ntimes,nfreqs),dtype=bool))
            else: fl.append(np.zeros((ntimes,nfreqs),dtype=bool))
        datarray.append(dd)
        flgarray.append(fl)

    datarray = np.array(datarray)
    datarray = datarray.swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)

    flgarray = np.array(flgarray)
    flgarray = flgarray.swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)
    
    tarray = np.resize(time,(4*nfreqs*nants,ntimes)).transpose().reshape(nants*nfreqs*npol*ntimes)
    parray = np.concantenate([[pol]*(nfreq*nants) for pol in pols])
    farray = np.array(list(np.resize(freq,(nants,nfreqs)).transpose().reshape(nants*nfreqs))*npol*ntimes)
    numarray = np.array(allants*npol*ntimes*nfreqs)
    namarray = np.array(antnames*npol*ntimes*nfreqs)

    chisqarray = []
    for i in range(npol):
        ch = []
        for jj in range(nants):
            try:
                ch.append(meta['chisq'+str(allants[jj])+pols[ii]]
            else:
                ch.append(np.ones((nt,nf)))
        chisqarray.append(ch)
    chisqarray = np.array(chisqarray).swapaxes(0,3).swapaxes(0,1)
    
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
    calfits_object.polarization_array = parray
    calfits_object.time_array = tarray
    calfits_object.gain_convention = 'divide'
    calfits_object.flag_array = flgarray
    calfits_object.quality_array = chisqarray  # what is this array supposed to be?
    calfits_object.cal_type = 'gain'
    calfits_object.x_orientation = 'E'
    calfits_object.gain_array = datarray
    calfits_object.quality_array = chisqarray

