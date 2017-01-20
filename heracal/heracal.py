from uvdata.cal import UVCal
import numpy as np


class HERACal(UVCal):
    '''Class that loads in hera omnical data,'''
    def __init__(self, meta, gains, ex_ants = []):
        '''given meta and gain dictionary from omni_run.py before to_npz() call,
           to reshuffle data and populating the UVCal class.'''
        
        super(HERACal, self).__init__()
    
        str2pol = {'x': -5, 'y': -6}
        pol2str = {-5: 'x', -6: 'y'}
        
        chisqdict = {}
        datadict = {}
        ants = []
        for pol in gains:
            for ant in np.sort(gains[pol].keys()):
                datadict['%d%s' %(ant,pol)] = gains[pol][ant]
                if not ant in ants:
                    ants.append(ant)

        # drop antennas that are not solved for. 
        allants = ants + ex_ants
        ants = np.sort(ants)
        allants = np.sort(allants)
        time = meta['jds']
        freq = meta['freqs'] # in GHz
        pols = [ str2pol[p] for p in gains.keys() ]; npol = len(pols)
        ntimes = time.shape[0]
        nfreqs = freq.shape[0]
        nants = len(ants)
        antnames = ['ant'+str(ant) for ant in ants]
        datarray = []
        flgarray = []
        for ii in range(npol):
            dd = []
            fl = []
            for ant in ants:
                try: 
                    dd.append(datadict[str(ant)+pol2str[pols[ii]]])
                    fl.append(np.zeros((ntimes,nfreqs),dtype=bool))
                #if antenna not in data dict (aka, a bad antenna)
                except(KeyError): 
                    print "Can't find antenna {0}".format(ant)
            datarray.append(dd)
            flgarray.append(fl)

        '''
        datarray = np.array(datarray)
        datarray = datarray.swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)

        flgarray = np.array(flgarray)
        flgarray = flgarray.swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)
        
        tarray = np.resize(time,(npol*nfreqs*nants,ntimes)).transpose().reshape(nants*nfreqs*npol*ntimes)
        parray = np.concatenate([[pol]*(nfreqs*nants*ntimes) for pol in pols])
        farray = np.array(list(np.resize(freq,(nants,nfreqs)).transpose().reshape(nants*nfreqs))*npol*ntimes)
        numarray = np.array(list(ants)*npol*ntimes*nfreqs, dtype=np.int16)
        namarray = np.array(antnames*npol*ntimes*nfreqs)

        chisqarray = []
        for ii in range(npol):
            ch = []
            for ant in ants:
                try:
                    ch.append(meta['chisq'+str(ant)+pol2str[pols[ii]]])
                except:
                    ch.append(np.ones((ntimes,nfreqs)))
            chisqarray.append(ch)
        chisqarray = np.array(chisqarray).swapaxes(0,3).swapaxes(0,1).reshape(nants*nfreqs*npol*ntimes)
        '''
        datarray = np.array(datarray)
        datarray = datarray.swapaxes(0,3).swapaxes(0,1)

        flgarray = np.array(flgarray)
        flgarray = flgarray.swapaxes(0,3).swapaxes(0,1)
        
        tarray = time
        parray = np.array(pols)
        farray = np.array(freq)
        numarray = np.array(list(ants))
        namarray = np.array(antnames)

        chisqarray = []
        for ii in range(npol):
            ch = []
            for ant in ants:
                try:
                    ch.append(meta['chisq'+str(ant)+pol2str[pols[ii]]])
                except:
                    ch.append(np.ones((ntimes,nfreqs)))
            chisqarray.append(ch)
        chisqarray = np.array(chisqarray).swapaxes(0,3).swapaxes(0,1)

        
#       import IPython; IPython.embed()
        self.Nfreqs = nfreqs
        self.Npols = len(pols)
        self.Ntimes = ntimes
        self.history = 'Test file.'
        self.Nants_data = len(ants)  # only ants with data
        self.antenna_names = namarray[:self.Nants_data]
        self.antenna_numbers = numarray[:self.Nants_data]
        self.Nants_telescope = nants # total number of antennas
        self.Nspws = 1

        self.freq_array = farray[:self.Nfreqs].reshape(self.Nspws,-1)
        self.polarization_array = parray[:self.Npols]
        self.time_array = tarray[:self.Ntimes]
        self.gain_convention = 'divide'
        self.flag_array = flgarray
        self.quality_array = chisqarray  
        self.cal_type = 'gain'
        self.x_orientation = 'east'
        self.gain_array = datarray
        self.quality_array = chisqarray

