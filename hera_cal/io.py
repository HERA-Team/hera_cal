import numpy as np
from pyuvdata import UVCal, UVData
from collections import OrderedDict as odict
from pyuvdata.utils import polstr2num, polnum2str
from copy import deepcopy
from hera_cal.datacontainer import DataContainer
import operator

str2antpol = {'x': -5, 'y': -6}
antpol2str = {antpol: string for string,antpol in str2antpol.items()}
#TODO: update to use jones strings (e.g. 'jxx' and 'jyy'). Currently uses pyuvdata for visibility polarizations.

def load_vis(input_data, return_meta=False, filetype='miriad', pop_autos=False, pick_data_ants=True, nested_dict=False):
    '''Load miriad or uvfits files or UVData objects into DataContainers, optionally returning
    the most useful metadata. More than one spectral window is not supported. Assumes every baseline
    has the same times present and that the times are in order.

    Arguments:
        input_data: data file path, or UVData instance, or list of either strings of data file paths
            or list of UVData instances to concatenate into a single dictionary
        return_meta:  boolean, if True: also return antpos, ants, freqs, times, lsts, and pols
        filetype: either 'miriad' or 'uvfits', can be ignored if input_data is UVData objects
        pop_autos: boolean, if True: remove autocorrelations
        pick_data_ants: boolean, if True and return_meta=True, return only antennas in data
        nested_dict: boolean, if True replace DataContainers with the legacy nested dictionary filetype
            where visibilities and flags are accessed as data[(0,1)]['xx']

    Returns:
        if return_meta is True:
            (data, flags, antpos, ants, freqs, times, lsts, pols)
        else:
            (data, flags)

        data: DataContainer containing baseline-pol complex visibility data with keys
            like (0,1,'xx') and with shape=(Ntimes,Nfreqs)
        flags: DataContainer containing data flags
        antpos: dictionary containing antennas numbers as keys and position vectors
        ants: ndarray containing unique antenna indices
        freqs: ndarray containing frequency channels (Hz)
        times: ndarray containing julian date bins of data
        lsts: ndarray containing LST bins of data (radians)
        pol: ndarray containing list of polarization strings
    '''

    uvd = UVData()
    if isinstance(input_data, (tuple, list, np.ndarray)): #List loading
        if np.all([isinstance(id, str) for id in input_data]): #List of visibility data paths
            if filetype == 'miriad':
                uvd.read_miriad(list(input_data))
            elif filetype == 'uvfits':
                 #TODO: implement this
                raise NotImplementedError('This function has not been implemented yet.')
            else:
                raise NotImplementedError("Data filetype must be either 'miriad' or 'uvfits'.")
        elif np.all([isinstance(id, UVData) for id in input_data]): #List of uvdata objects
            uvd = reduce(operator.add, input_data)
        else:
            raise TypeError('If input is a list, it must be only strings or only UVData objects.')
    elif isinstance(input_data, str): #single visibility data path
        if filetype == 'miriad':
            uvd.read_miriad(input_data)
        elif filetype == 'uvfits':
             #TODO: implement this
            raise NotImplementedError('This function has not been implemented yet.')
        else:
            raise NotImplementedError("Data filetype must be either 'miriad' or 'uvfits'.")
    elif isinstance(input_data, UVData): #single UVData object
        uvd = input_data
    else:
        raise TypeError('Input must be a UVData object, a string, or a list of either.')

    data, flags = odict(), odict()
    # create nested dictionaries of visibilities in the data[bl][pol] filetype, removing autos if desired
    for nbl, (i, j) in enumerate(uvd.get_antpairs()):
        if (not pop_autos) or (i != j):
            if (i, j) not in data:
                data[i, j], flags[i, j] = odict(), odict()
            for ip, pol in enumerate(uvd.polarization_array):
                pol = polnum2str(pol)
                data[(i, j)][pol] = deepcopy(uvd.get_data((i, j, pol)))
                flags[(i, j)][pol] = deepcopy(uvd.get_flags((i, j, pol)))

    # If we don't want nested dicts, convert to DataContainer
    if not nested_dict:
        data, flags = DataContainer(data), DataContainer(flags)

    # get meta
    if return_meta:
        freqs = np.unique(uvd.freq_array)
        times = np.unique(uvd.time_array)
        lsts = np.unique(uvd.lst_array)
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
        antpos = odict(zip(ants, antpos))
        pols = np.array([polnum2str(polnum) for polnum in uvd.polarization_array])
        return data, flags, antpos, ants, freqs, times, lsts, pols
    else:
        return data, flags


def write_vis(outfilename, data, flags, filetype='miriad', history='', clobber=False, **kwargs):
    '''TODO: migrate in hera_cal.utils.data_to_miriad and generalize to also write uvfits.'''
    raise NotImplementedError('This function has not been implemented yet.')


def update_uvdata(uvd, data=None, flags=None, add_to_history='', **kwargs):
    '''Updates a UVData object with data or parameters. Cannot modify the shape of
    data arrays. More than one spectral window is not supported. Assumes every baseline
    has the same times present and that the times are in order.

    Arguments:
        uv: UVData object to be updated
        data: dictionary or DataContainer of complex visibility data to update. Keys
            like (0,1,'xx') and shape=(Ntimes,Nfreqs). Default (None) does not update.
        flags: dictionary or DataContainer of data flags to update.
            Default (None) does not update.
        add_to_history: appends a string to the history of the UVData object
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # set data and/or flags
    if data is not None or flags is not None:
        for (i, j) in uvd.get_antpairs():
            this_bl = (uvd.baseline_array == uvd.antnums_to_baseline(i, j))
            for ip, pol in enumerate(uvd.polarization_array):
                if data is not None:
                    uvd.data_array[this_bl, 0, :, ip] = data[(i, j, polnum2str(pol))]
                if flags is not None:
                    uvd.flag_array[this_bl, 0, :, ip] = flags[(i, j, polnum2str(pol))]

    # set additional attributes
    uvd.history += add_to_history
    for attribute, value in kwargs.items():
        uvd.__setattr__(attribute, value)


def update_vis(infilename, outfilename, filetype_in='miriad', filetype_out='miriad',
               data=None, flags=None, add_to_history='', clobber=False, **kwargs):
    '''Loads an existing file with pyuvdata, modifies some subset of of its parameters, and
    then writes a new file to disk. Cannot modify the shape of data arrays. More than one
    spectral window is not supported. Assumes every baseline has the same times present
    and that the times are in order.

    Arguments:
        infilename: filename of the base visibility file to be updated, or UVData object
        outfilename: filename of the new visibility file
        filetype_in: either 'miriad' or 'uvfits' (ignored if infile is a UVData object)
        filetype_out: either 'miriad' or 'uvfits'
        data: dictionary or DataContainer of complex visibility data to update. Keys
            like (0,1,'xx') and shape=(Ntimes,Nfreqs). Default (None) does not update.
        flags: dictionary or DataContainer of data flags to update.
            Default (None) does not update.
        add_to_history: appends a string to the history of the output file
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVData documentation for more info.
    '''

    # Load infile
    if type(infilename) == UVData:
        uvd = deepcopy(infilename)
    else:
        uvd = UVData()
        if filetype_in == 'miriad':
            uvd.read_miriad(infilename)
        elif filetype_in == 'uvfits':
            # TODO: implement this
            raise NotImplementedError('This function has not been implemented yet.')
        else:
            raise TypeError("Input filetype must be either 'miriad' or 'uvfits'.")

    update_uvdata(uvd, data=data, flags=flags, add_to_history=add_to_history, **kwargs)
    uvd.check()

    # write out results
    if filetype_out == 'miriad':
        uvd.write_miriad(outfilename, clobber=clobber)
    elif filetype_out == 'uvfits':
        # TODO: implement this
        raise NotImplementedError('This function has not been implemented yet.')
    else:
        raise TypeError("Input filetype must be either 'miriad' or 'uvfits'.")


def load_cal(input_cal, return_meta=False):
    '''Load calfits files or UVCal objects into dictionaries, optionally returning
    the most useful metadata. More than one spectral window is not supported.

    Arguments:
        input_cal: path to calfits file, UVCal object, or a list of either
        return_meta: if True, returns additional information (see below)

    Returns:
        if return_meta is True:
            (gains, flags, quals, total_qual, ants, freqs, times, pols)
        else:
            (gains, flags)

        gains: Dictionary of complex calibration gains as a function of time
            and frequency with keys in the (1,'x') format
        flags: Dictionary of flags in the same format as the gains
        quals: Dictionary of of qualities of calibration solutions in the same
            format as the gains (e.g. omnical chi^2 per antenna)
        total_qual: ndarray of toal calibration quality for the whole array
            (e.g. omnical overall chi^2)
        ants: ndarray containing unique antenna indices
        freqs: ndarray containing frequency channels (Hz)
        times: ndarray containing julian date bins of data
        pols: list of antenna polarization strings
    '''
    #load UVCal object
    cal = UVCal()
    if isinstance(input_cal, (tuple, list, np.ndarray)): #List loading
        if np.all([isinstance(ic, str) for ic in input_cal]): #List of calfits paths
            cal.read_calfits(list(input_cal))
        elif np.all([isinstance(ic, UVCal) for ic in input_cal]): #List of UVCal objects
            cal = reduce(operator.add, input_cal)
        else:
            raise TypeError('If input is a list, it must be only strings or only UVCal objects.')
    elif isinstance(input_cal, str): #single calfits path
        cal.read_calfits(input_cal)
    elif isinstance(input_cal, UVCal): #single UVCal object
        cal = input_cal
    else:
        raise TypeError('Input must be a UVCal object, a string, or a list of either.')

    #load gains, flags, and quals into dictionaries
    gains, quals, flags = odict(), odict(), odict()
    for i, ant in enumerate(cal.ant_array):
        for ip, pol in enumerate(cal.jones_array):
            gains[(ant, antpol2str[pol])] = cal.gain_array[i, 0, :, :, ip].T
            flags[(ant, antpol2str[pol])] = cal.flag_array[i, 0, :, :, ip].T
            quals[(ant, antpol2str[pol])] = cal.quality_array[i, 0, :, :, ip].T

    #return quantities
    if return_meta:
        total_qual = cal.total_quality_array
        ants = cal.ant_array
        freqs = np.unique(cal.freq_array)
        times = np.unique(cal.time_array)
        pols = [antpol2str[j] for j in cal.jones_array]
        return gains, flags, quals, total_qual, ants, freqs, times, pols
    else:
        return gains, flags


def write_cal():
    '''TODO: copy over code from hera_cal.cal_formats.HERACal'''
    raise NotImplementedError('This function has not been implemented yet.')


def update_uvcal(cal, gains=None, flags=None, quals=None, add_to_history='', **kwargs):
    '''Update UVCal object with gains, flags, quals, history, and/or other parameters
    Cannot modify the shape of gain arrays. More than one spectral window is not supported.

    Arguments:
        cal: UVCal object to be updated
        gains: Dictionary of complex calibration gains with shape=(Ntimes,Nfreqs)
            with keys in the (1,'x') format. Default (None) leaves unchanged.
        flags: Dictionary like gains but of flags. Default (None) leaves unchanged.
        quals: Dictionary like gains but of per-antenna quality. Default (None) leaves unchanged.
        add_to_history: appends a string to the history of the UVCal object
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVCal documentation for more info.
    '''
    # Set gains, flags, and/or quals
    for i, ant in enumerate(cal.ant_array):
        for ip, pol in enumerate(cal.jones_array):
            if gains is not None:
                cal.gain_array[i, 0, :, :, ip] = gains[(ant, antpol2str[pol])].T
            if flags is not None:
                cal.flag_array[i, 0, :, :, ip] = flags[(ant, antpol2str[pol])].T
            if quals is not None:
                cal.quality_array[i, 0, :, :, ip] = quals[(ant, antpol2str[pol])].T

    # Set additional attributes
    cal.history += add_to_history
    for attribute, value in kwargs.items():
        cal.__setattr__(attribute, value)


def update_cal(infilename, outfilename, gains=None, flags=None, quals=None, add_to_history='', clobber=False, **kwargs):
    '''Loads an existing calfits file with pyuvdata, modifies some subset of of its parameters,
    and then writes a new calfits file to disk. Cannot modify the shape of gain arrays.
    More than one spectral window is not supported.

    Arguments:
        infilename: filename of the base calfits file to be updated, or UVCal object
        outfilename: filename of the new calfits file
        gains: Dictionary of complex calibration gains with shape=(Ntimes,Nfreqs)
            with keys in the (1,'x') format. Default (None) leaves unchanged.
        flags: Dictionary like gains but of flags. Default (None) leaves unchanged.
        quals: Dictionary like gains but of per-antenna quality. Default (None) leaves unchanged.
        add_to_history: appends a string to the history of the output file
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVCal documentation for more info.
    '''

    # Load infile
    if type(infilename) == UVCal:
        cal = deepcopy(infilename)
    else:
        cal = UVCal()
        cal.read_calfits(infilename)

    update_uvcal(cal, gains=gains, flags=flags, quals=quals,
                 add_to_history=add_to_history, **kwargs)
    cal.check()

    # Write to calfits file
    cal.write_calfits(outfilename, clobber=clobber)
