from uvdata.calfits import CALFITS
import numpy as np


class HERACal(CALFITS):
    '''Class that loads in hera omnical data,'''
    def __init__(self, meta, gains, ex_ants = []):
        '''given meta and gain dictionary from omni_run.py before to_npz() call,
           to reshuffle data and populating the UVCal class.'''
        
        super(HERACal, self).__init__()
        
        chisqdict = {}
        datadict = {}
        ants = []
        for pol in gains:
            for ant in np.sort(gains[pol].keys()):
                datadict['%d%s' %(ant,pol)] = gains[pol][ant]
                if not ant in ants:
                    ants.append(ant)

        allants = ants + ex_ants
        allants = np.sort(allants)
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
        
        tarray = np.resize(time,(npol*nfreqs*nants,ntimes)).transpose().reshape(nants*nfreqs*npol*ntimes)
        parray = np.concatenate([[pol]*(nfreqs*nants) for pol in pols])
        farray = np.array(list(np.resize(freq,(nants,nfreqs)).transpose().reshape(nants*nfreqs))*npol*ntimes)
        numarray = np.array(allants*npol*ntimes*nfreqs)
        namarray = np.array(antnames*npol*ntimes*nfreqs)

        chisqarray = []
        for ii in range(npol):
            ch = []
            for jj in range(nants):
                try:
                    ch.append(meta['chisq'+str(allants[jj])+pols[ii]])
                except:
                    ch.append(np.ones((ntimes,nfreqs)))
            chisqarray.append(ch)
        chisqarray = np.array(chisqarray).swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)
        

        self.Nfreqs = nfreqs
        self.Npols = len(pols)
        self.Ntimes = ntimes
        self.history = ''
        self.Nants_data = len(ants)  # only ants with data
        self.antenna_names = namarray
        self.antenna_numbers = numarray
        self.Nants_telescope = nants # total number of antennas
        self.Nspws = 1

        self.freq_array = farray
        self.polarization_array = parray
        self.time_array = tarray
        self.gain_convention = 'divide'
        self.flag_array = flgarray
        self.quality_array = chisqarray  
        self.cal_type = 'gain'
        self.x_orientation = 'east'
        self.gain_array = datarray
        self.quality_array = chisqarray

        self.set_gain()

