import numpy as np
import os
from pyuvdata import UVCal, UVData


class HERACal(UVCal):
    '''
       Class that loads in hera omnical data into a pyuvdata calfits object.
       This can then be saved to a file, plotted, etc.
    '''

    def __init__(self, meta, gains, flags=None, DELAY=False, ex_ants=[], appendhist='', **optional):
        '''Initialize a UVCal object.
            Args:
                meta: meta information dictionary. As returned by from_fits or from_npz.
                gains: dictionary of complex gain solutions or delays.
                flags (optional): Optional input flags for gains.
                DELAY (optional): toggle if calibration solutions in gains are delays.
                ex_ants (optional): antennas that are excluded from gains.
                appendhist (optional): string to append to history
                optional (optional): dictionary of optional parameters to be passed to UVCal object.

            Notes:
                The UVCal parameter cal_style is by default set to 'redundant'. It can be changed
                by feeding cal_style as a keyword argument to HERACal. If cal_style is set to 'sky',
                then there are additional required parameters. See pyuvdata.UVCal doc for details.
                If cal_style is neither redundant or sky, it is set to unknown.
        '''

        super(HERACal, self).__init__()

        # helpful dictionaries for antenna polarization of gains
        str2pol = {'x': -5, 'y': -6}
        pol2str = {-5: 'x', -6: 'y'}

        chisqdict = {}
        datadict = {}
        flagdict = {}

        # drop antennas that are not solved for. Since we are feeding in omnical/firstcal solutions into this,
        # if we provided an ex_ants those antennas will not have a key in gains. Need to provide ex_ants list
        # to HERACal object.
        # create set to get unique antennas from both pol
        ants = np.array(list(set([ant for pol in gains.keys() for ant in gains[pol].keys()])))
        allants = np.sort(np.concatenate([ants, np.array(ex_ants)])).astype(np.int)  # total number of antennas
        ants = np.sort(ants)
        # antenna names for all antennas
        antnames = np.array(['ant' + str(int(ant)) for ant in allants])
        time = meta['times']
        freq = meta['freqs']  # this is in Hz (should be anyways)
        pols = [str2pol[p] for p in gains.keys()]  # all of the polarizations

        # get sizes of things
        nspw = 1  # This is by default 1. No support for > 1 in pyuvdata.
        npol = len(pols)
        ntimes = time.shape[0]
        nfreqs = freq.shape[0]

        datarray = np.array([[gains[pol2str[pol]][ant] for ant in ants]
                             for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        if flags:
            flgarray = np.array([[flags[pol2str[pol]][ant] for ant in ants]
                                 for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        else:
            if DELAY:
                flgarray = np.zeros((len(ants), nfreqs, ntimes, npol), dtype=bool)
            else:
                # dont need to swap since datarray alread same shape
                flgarray = np.zeros(datarray.shape, dtype=bool)
        # do the same for the chisquare, which is the same shape as the data
        try:
            chisqarray = np.array([[meta['chisq' + str(ant) + pol2str[pol]]
                                    for ant in ants] for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        except:
            # XXX EXCEPT WHAT?
            chisqarray = np.ones(datarray.shape, dtype=float)
        # get the array-wide chisq, which does not have separate axes for
        # antennas or polarization
        try:
            totchisqarray = np.array(meta['chisq']).swapaxes(0, 1)
            # add a polarization axis until this is fixed properly
            totchisqarray = totchisqarray[:, :, np.newaxis]
            # repeat polarization axis npol times for proper shape
            totchisqarray = np.repeat(totchisqarray, npol, axis=-1)
        except:
            # XXX EXCEPT WHAT?
            # leave it empty
            totchisqarray = None
        
        pols = np.array(pols)
        freq = np.array(freq)
        antarray = np.array(list(map(int, ants)))
        numarray = np.array(list(map(int, allants)))

        # set UVCal attributes
        self.telescope_name = 'HERA'
        self.Nfreqs = nfreqs
        self.Njones = len(pols)
        self.Ntimes = ntimes
        try:
            self.history = meta['history'].replace('\n', ' ') + appendhist
        except KeyError:
            self.history = appendhist
        self.Nants_data = len(ants)  # only ants with data
        self.Nants_telescope = len(allants)  # all antennas in telescope
        self.antenna_names = antnames
        self.antenna_numbers = numarray
        self.ant_array = np.array(antarray[:self.Nants_data])
        
        self.Nspws = nspw # XXX: needs to change when we support more than 1 spw!
        self.spw_array = np.array([0])
        
        self.freq_array = freq.reshape(self.Nspws, -1)
        self.channel_width = np.diff(self.freq_array)[0][0]
        self.jones_array = pols
        self.time_array = time
        self.integration_time = meta['inttime']
        self.gain_convention = 'divide'
        self.set_redundant()
        self.x_orientation = 'east'
        self.time_range = [self.time_array[0], self.time_array[-1]]
        self.freq_range = [self.freq_array[0][0], self.freq_array[0][-1]]
        
        # set the optional attributes to UVCal class, which may overwrite previous
        # parameters set by default
        for key in optional:
            setattr(self, key, optional[key])

        # look for cal_style in optional
        if 'cal_style' in optional:
            if optional['cal_style'] == 'redundant':
                pass
            elif optional['cal_style'] == 'sky':
                # set cal style to sky, and set required parameters
                self.set_sky()
                reqs = ['ref_antenna_name', 'sky_catalog', 'sky_field']
                for r in reqs:
                    if r not in optional:
                        raise AttributeError("if cal_style=='sky', then {} must be fed".format(r))
                    setattr(self, r, optional[r])
            else:
                self.set_unknown_cal_type()

        # adding new axis for the spectral window axis. This is default to 1.
        # This needs to change when support for Nspws>1 in pyuvdata.
        self.quality_array = chisqarray[:, np.newaxis, :, :, :]
        self.flag_array = flgarray.astype(np.bool)[:, np.newaxis, :, :, :]
        if DELAY:
            self.set_delay()
            self.delay_array = datarray[:, np.newaxis, :, :, :]  # units of seconds
        else:
            self.set_gain()
            self.gain_array = datarray[:, np.newaxis, :, :, :]
        if totchisqarray is not None:
            self.total_quality_array = totchisqarray[np.newaxis, :, :, :]

        # run a check
        self.check()


        