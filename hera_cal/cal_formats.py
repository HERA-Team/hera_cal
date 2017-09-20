import numpy as np
import os
from pyuvdata import UVCal, UVData


class HERACal(UVCal):
    '''
       Class that loads in hera omnical data into a pyuvdata calfits object.
       This can then be saved to a file, plotted, etc.
    '''

    def __init__(self, meta, gains, flags=None, DELAY=False, ex_ants=[], appendhist='', optional={}):
        '''Initialize a UVCal object.
            Args:
                meta: meta information dictionary. As returned by from_fits or from_npz.
                gains: dictionary of complex gain solutions or delays.
                flags (optional): Optional input flags for gains.
                DELAY (optional): toggle if calibration solutions in gains are delays.
                ex_ants (optional): antennas that are excluded from gains.
                appendhist (optional): string to append to history
                optional (optional): dictionary of optional parameters to be passed to UVCal object.
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
        ants = list(set([ant for pol in gains.keys() for ant in gains[pol].keys()]))
        allants = np.sort(ants + ex_ants)  # total number of antennas
        ants = np.sort(ants)
        # antenna names for all antennas
        antnames = ['ant' + str(ant) for ant in allants]
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
        antarray = list(map(int, ants))
        numarray = list(map(int, allants))

        # set the optional attributes to UVCal class.
        for key in optional:
            setattr(self, key, optional[key])
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
        self.x_orientation = 'east'
        self.time_range = [self.time_array[0], self.time_array[-1]]
        self.freq_range = [self.freq_array[0][0], self.freq_array[0][-1]]
        
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



class AbsCal(UVCal):
    '''
    Load gains obtained via an absolute calibration into a UVCal object.

    This class is for defining a way to create a pyuvdata UVCal object from
    gains derived from an absolute calibration.
    '''

    def __init__(self, miriad_filename, abscal_npznames, polname, gain_convention,
                 ex_ants=[], optional={}, append2hist=''):
        '''
        Initialize an AbsCal object. It requires the gains themselves (assumed to be stored
        in an npz format) and the accompanying miriad file that was used to generate them
        (to get important metadata). It assumes that the gains have been specified as a function
        of frequency, and pads them out to apply to all times in the miriad file.

        Args:
        ====================
        miriad_filename (string) -- full path to miriad file used to generate absolute calibration
        abscal_npzname (list of strings) -- full paths to npz file containing the gains (can have multiple
           npz files for a cross-pol miriad file)
        polname (string) -- visibility polarization to be calibrated (e.g., 'xx')
        gain_convention (string) -- definition of how gains should be applied to the data
           (must be "mulitply" or "divide")
        ex_ants (list of ints, optional) -- antenna numbers not included in the calibration
        optional (dictionary, optional) -- optional parameters to be passed to the UVCal object
        append2hist (string, optional) -- string to append to history

        Returns:
        ====================
        uvc -- UVCal object containing the gains to be applied to the miriad file
        '''
        super(AbsCal, self).__init__()

        # define dictionaries for converting between polarization string and number
        str2pol = {'x': -5, 'y': -6}
        pol2str = {-5: 'x', -6: 'y'}

        if gain_convention not in ["multiply", "divide"]:
            raise AssertionError("gain_convention must be 'mulitply' or 'divide'")

        # make sure that npz and miriad file have the polarization specified
        if polname not in miriad_filename:
            raise AssertionError("specified polarization does not match the one in the name of the miriad file")

        # make sure that we have all the polarizations necessary for the calibration
        for polchar in set(polname):
            have_abscal = False
            for npz in abscal_npznames:
                basename = os.path.basename(npz)
                if polchar in basename:
                    have_abscal = True
            if not have_abscal:
                raise AssertionError("required polarization solution {} not "
                                     "contained in abscal list".format(polchar))

        # check that we have the right number of abscal files for the specified polarization type
        if len(abscal_npznames) != len(set(polname)):
            raise AssertionError("expected {} abscal files, instead got {}".format(len(set(polname)),
                                                                                   len(abscal_npznames)))

        # load npzs
        npzdict, filedict = {}, {}
        for npz in abscal_npznames:
            basename = os.path.basename(npz)
            for polchar in set(polname):
                if polchar in basename:
                    _d = dict(np.load(npz))
                if 'flags' in _d.keys():
                    _f = _d.pop('flags')
                else:
                    _f = None
                npzdict[polchar] = _d

        # check that for the case of 2 abscal files, we have the same antennas
        if len(abscal_npznames) > 1:
            ants1 = npzdict[polname[0]].keys()
            ants2 = npzdict[polname[1]].keys()
            if sorted(nants1) != sorted(ants2):
                raise AssertionError("The two abscal files do not have the same antennas")
            ants_gains = ants1
            nants_gains = len(ants_gains)
        else:
            ants_gains = npzdict[polname[0]].keys()
            nants_gains = len(ants_gains)

        # save miriad filename too
        filedict[polname] = miriad_filename

        # get metadata from the miriad file
        uvd = UVData()
        uvd.read_miriad(miriad_filename)
        times = np.unique(uvd.time_array)
        freqs = uvd.freq_array
        pols = [str2pol[p] for p in list(set(''.join(polname)))]

        history = uvd.history
        inttime = uvd.integration_time

        # sizes of data
        nspw = 1
        npol = len(pols)
        ntimes = times.shape[0]
        nfreqs = freqs.shape[1]

        # do a quick check that there are enough antennas in the gains file for the data
        nants_exants = len(ex_ants)
        if uvd.Nants_data > nants_gains + nants_exants:
            raise AssertionError("The number of antennas in the miriad file is greater than the "
                                 "number of gains provided plus excluded antennas.")

        # make a list of "good antennas"
        ants = sorted(map(int, ants_gains))
        good_antnames = sorted(ants_gains)
        allants = np.sort(ants + ex_ants)
        antnames = ['ant{}'.format(a) for a in allants]

        # construct data and flag array
        tmp_spec = npzdict[polname[0]][good_antnames[0]]
        # check to see if our data are already the right size
        if tmp_spec.shape != (ntimes, nfreqs):
            # make sure we've got the right number of frequency channels
            if tmp_spec.shape != (nfreqs,):
                raise AssertionError("Cannot interpret gain spectrum shape")
        if _f is not None:
            if _f.shape != tmp_spec.shape:
                raise AssertionError("Flags must be the same shape as gain spectrum")
            flags = _f
        else:
            flags = np.zeros_like(tmp_spec)

        data_array = np.array([[npzdict[polchar][ant] * np.ones((ntimes, nfreqs))
                                for ant in good_antnames
                                for polchar in set(polname)]]).swapaxes(0, 3).swapaxes(0, 1)

        flag_array = np.array([[flags * np.ones((ntimes, nfreqs))
                                for ant in good_antnames
                                for polchar in set(polname)]]).swapaxes(0, 3).swapaxes(0, 1)

        # fill out other required arrays
        chisqarray = np.ones(data_array.shape)
        totchisqarray = None
        parray = np.array(pols)
        farray = np.array(freqs)
        tarray = times
        antarray = list(ants)
        numarray = list(allants)
        namarray = antnames

        for key in optional:
            setattr(self, key, optional[key])

        self.telescope_name = 'HERA'
        self.Nfreqs = nfreqs
        self.Njones = len(pols)
        self.Ntimes = ntimes
        self.history = history + append2hist
        self.Nants_data = len(ants)
        self.Nants_telescope = len(allants)
        self.antenna_names = namarray[:self.Nants_telescope]
        self.antenna_numbers = numarray[:self.Nants_telescope]
        self.ant_array = np.array(antarray[:self.Nants_data])
        self.Nspws = nspw
        self.spw_array = np.array([0])
        self.freq_array = farray[:self.Nfreqs].reshape(self.Nspws, -1)
        self.channel_width = np.diff(self.freq_array)[0][0]
        self.jones_array = parray[:self.Njones]
        self.time_array = tarray[:self.Ntimes]
        self.integration_time = inttime
        self.gain_convention = gain_convention
        self.x_orientation = 'east'
        self.time_range = [self.time_array[0], self.time_array[-1]]
        self.freq_range = [self.freq_array[0][0], self.freq_array[0][-1]]
        self.quality_array = chisqarray[:, np.newaxis, :, :, :]

        self.flag_array = flag_array.astype(np.bool)[:, np.newaxis, :, :, :]

        self.set_gain()
        self.gain_array = data_array[:, np.newaxis, :, :, :]

        # check that we have a valid object
        self.check()
