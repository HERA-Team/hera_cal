from uvdata import calfits.CALFITS


class HERACal(CALFITS):
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
    
    self.set_gain()

    self.Nfreqs = nf
    self.Npols = len(pol)
    self.Ntimes = nt
    self.history = ''
    self.Nants_data = na  # what value?
    self.antenna_names = namarray
    self.antenna_numbers = numarray
    self.Nants_telescope = na
    self.Nspws = 1

    self.freq_array = farray
    self.polarization_array = parray
    self.time_array = tarray
    self.gain_convention = 'divide'
    self.flag_array = flgarray
    self.quality_array = chisqarray  # what is this array supposed to be?
    self.cal_type = 'gain'
    self.x_orientation = 'E'
    self.gain_array = datarray
    self.quality_array = chisqarray

