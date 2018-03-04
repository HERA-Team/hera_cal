import numpy as np
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from collections import OrderedDict as odict
from copy import deepcopy
from hera_cal.datacontainer import DataContainer
from hera_cal import utils
import operator
import os
from hera_cal.abscal import echo
import argparse


polnum2str = {-5: "xx", -6: "yy", -7: "xy", -8: "yx"}
polstr2num = {"xx": -5, "yy": -6 ,"xy": -7, "yx": -8}


jonesnum2str = {-5: 'jxx', -6: 'jyy', -7: 'jxy', -8: 'jyx'}
jonesstr2num = {'jxx': -5, 'jyy': -6, 'jxy': -7, 'jyx': -8}


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
                pol = polnum2str[pol]
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
        pols = np.array([polnum2str[polnum] for polnum in uvd.polarization_array])
        return data, flags, antpos, ants, freqs, times, lsts, pols
    else:
        return data, flags


def write_vis(fname, data, lst_array, freq_array, antpos, time_array=None, flags=None, nsamples=None,
              filetype='miriad', write_file=True, outdir="./", overwrite=False, verbose=True, history=" ",
              return_uvd=False, longitude=21.42830, start_jd=None, instrument="HERA", 
              telescope_name="HERA", object_name='EOR', vis_units='uncalib', dec=-30.72152,
              telescope_location=np.array([5109325.85521063,2005235.09142983,-3239928.42475395]),
              **kwargs):
    """
    Take DataContainer dictionary, export to UVData object and write to file. See pyuvdata.UVdata
    documentation for more info on these attributes.

    Parameters:
    -----------
    data : type=DataContainer, holds complex visibility data.

    lst_array : type=float ndarray, contains unique LST time bins [radians] of data (center of integration).

    freq_array : type=ndarray, contains frequency bins of data [Hz]. 

    antpos : type=dictionary, antenna position dictionary. keys are antenna integers and values
             are position vectors in meters in ENU (TOPO) frame.

    time_array : type=ndarray, contains unique Julian Date time bins of data (center of integration).

    flags : type=DataContainer, holds data flags, matching data in shape.

    nsamples : type=DataContainer, holds number of points averaged into each bin in data (if applicable).

    filetype : type=str, filetype to write-out, options=['miriad'].

    write_file : type=boolean, write UVData to file if True.

    outdir : type=str, output directory for output file.

    overwrite : type=boolean, if True, overwrite output files.

    verbose : type=boolean, if True, report feedback to stdout.

    history : type=str, history string for UVData object

    return_uvd : type=boolean, if True return UVData instance.

    longitude : type=float, longitude of observer in degrees East

    start_jd : type=float, starting integer Julian Date of time_array if time_array is None.

    instrument : type=str, instrument name.

    telescope_name : type=str, telescope name.

    object_name : type=str, observing object name.

    vis_unit : type=str, visibility units.

    dec : type=float, declination of observer in degrees North.

    telescope_location : type=ndarray, telescope location in xyz in ITRF (earth-centered frame).

    kwargs : type=dictionary, additional parameters to set in UVData object.
    
    Output:
    -------
    if return_uvd: return UVData instance
    """
    ## configure UVData parameters
    # get pols
    pols = np.unique(map(lambda k: k[-1], data.keys()))
    Npols = len(pols)
    polarization_array = np.array(map(lambda p: polstr2num[p], pols))

    # get times
    if time_array is None:
        if start_jd is None:
            raise AttributeError("if time_array is not fed, start_jd must be fed")
        time_array = utils.LST2JD(lst_array, start_jd, longitude=longitude)
    Ntimes = len(time_array)
    integration_time = np.median(np.diff(time_array)) * 24 * 3600.

    # get freqs
    Nfreqs = len(freq_array)
    channel_width = np.median(np.diff(freq_array))
    freq_array = freq_array.reshape(1, -1)
    spw_array = np.array([0])
    Nspws = 1

    # get baselines keys
    bls = sorted(data.bls())
    Nbls = len(bls)
    Nblts = Nbls * Ntimes

    # reconfigure time_array and lst_array
    time_array = np.repeat(time_array[np.newaxis], Nbls, axis=0).ravel()
    lst_array = np.repeat(lst_array[np.newaxis], Nbls, axis=0).ravel()

    # get data array
    data_array = np.moveaxis(map(lambda p: map(lambda bl: data[str(p)][bl], bls), pols), 0, -1)

    # resort time and baseline axes
    data_array = data_array.reshape(Nblts, 1, Nfreqs, Npols)
    if nsamples is None:
        nsample_array = np.ones_like(data_array, np.float)
    else:
        nsample_array = np.moveaxis(map(lambda p: map(lambda bl: nsamples[str(p)][bl], bls), pols), 0, -1)
        nsample_array = nsample_array.reshape(Nblts, 1, Nfreqs, Npols)

    # flags
    if flags is None:
        flag_array = np.zeros_like(data_array, np.float).astype(np.bool)
    else:
        flag_array = np.moveaxis(map(lambda p: map(lambda bl: flags[str(p)][bl].astype(np.bool), bls), pols), 0, -1)
        flag_array = flag_array.reshape(Nblts, 1, Nfreqs, Npols)

    # configure baselines
    bls = np.repeat(np.array(bls), Ntimes, axis=0)

    # get ant_1_array, ant_2_array
    ant_1_array = bls[:,0]
    ant_2_array = bls[:,1]

    # get baseline array
    baseline_array = 2048 * (ant_2_array+1) + (ant_1_array+1) + 2^16

    # get antennas in data
    data_ants = np.unique(np.concatenate([ant_1_array, ant_2_array]))
    Nants_data = len(data_ants)

    # get telescope ants
    antenna_numbers = np.unique(antpos.keys())
    Nants_telescope = len(antenna_numbers)
    antenna_names = map(lambda a: "HH{}".format(a), antenna_numbers)

    # set uvw assuming drift phase i.e. phase center is zenith
    uvw_array = np.array([antpos[k[1]] - antpos[k[0]] for k in zip(ant_1_array, ant_2_array)])

    # get antenna positions in ITRF frame
    tel_lat_lon_alt = uvutils.LatLonAlt_from_XYZ(telescope_location)
    antenna_positions = np.array(map(lambda k: antpos[k], antenna_numbers))
    antenna_positions = uvutils.ECEF_from_ENU(antenna_positions.T, *tel_lat_lon_alt).T - telescope_location

    # get zenith location: can only write drift phase
    phase_type = 'drift'
    zenith_dec_degrees = np.ones_like(baseline_array) * dec
    zenith_ra_degrees = utils.JD2RA(time_array, longitude)
    zenith_dec = zenith_dec_degrees * np.pi / 180
    zenith_ra = zenith_ra_degrees * np.pi / 180

    # instantiate object
    uvd = UVData()

    # assign parameters
    params = ['Nants_data', 'Nants_telescope', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Nspws', 'Ntimes',
              'ant_1_array', 'ant_2_array', 'antenna_names', 'antenna_numbers', 'baseline_array',
              'channel_width', 'data_array', 'flag_array', 'freq_array', 'history', 'instrument',
              'integration_time', 'lst_array', 'nsample_array', 'object_name', 'phase_type',
              'polarization_array', 'spw_array', 'telescope_location', 'telescope_name', 'time_array',
              'uvw_array', 'vis_units', 'antenna_positions', 'zenith_dec', 'zenith_ra']   
    local_params = locals()           

    # overwrite paramters by kwargs
    local_params.update(kwargs)

    # set parameters in uvd
    for p in params:
        uvd.__setattr__(p, local_params[p])

    # write to file
    if write_file:
        if filetype == 'miriad':
            # check output
            fname = os.path.join(outdir, fname)
            if os.path.exists(fname) and overwrite is False:
                if verbose:
                    print("{} exists, not overwriting".format(fname))
            else:
                if verbose:
                    print("saving {}".format(fname))
                uvd.write_miriad(fname, clobber=True)

        else:
            raise AttributeError("didn't recognize filetype: {}".format(filetype))

    if return_uvd:
        return uvd


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
                    uvd.data_array[this_bl, 0, :, ip] = data[(i, j, polnum2str[pol])]
                if flags is not None:
                    uvd.flag_array[this_bl, 0, :, ip] = flags[(i, j, polnum2str[pol])]

    # set additional attributes
    uvd.history += add_to_history
    for attribute, value in kwargs.items():
        uvd.__setattr__(attribute, value)
    uvd.check()


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

    # load gains, flags, and quals into dictionaries
    gains, quals, flags = odict(), odict(), odict()
    for i, ant in enumerate(cal.ant_array):
        for ip, pol in enumerate(cal.jones_array):
            gains[(ant, jonesnum2str[pol])] = cal.gain_array[i, 0, :, :, ip].T
            flags[(ant, jonesnum2str[pol])] = cal.flag_array[i, 0, :, :, ip].T
            quals[(ant, jonesnum2str[pol])] = cal.quality_array[i, 0, :, :, ip].T

    #return quantities
    if return_meta:
        total_qual = cal.total_quality_array
        ants = cal.ant_array
        freqs = np.unique(cal.freq_array)
        times = np.unique(cal.time_array)
        pols = [jonesnum2str[j] for j in cal.jones_array]
        return gains, flags, quals, total_qual, ants, freqs, times, pols
    else:
        return gains, flags


def write_cal(fname, gains, freqs, times, flags=None, quality=None, write_file=True,
              return_uvc=True, outdir='./', overwrite=False, gain_convention='divide', 
              history=' ', x_orientation="east", telescope_name='HERA', cal_style='redundant',
              **kwargs):
    '''Format gain solution dictionary into pyuvdata.UVCal and write to file

    Arguments:
        fname : type=str, output file basename
        gains : type=dictionary, holds complex gain solutions. keys are antenna + pol
                tuple pairs, e.g. (2, 'x'), and keys are 2D complex ndarrays with time
                along [0] axis and freq along [1] axis.
        freqs : type=ndarray, holds unique frequencies channels in Hz
        times : type=ndarray, holds unique times of integration centers in Julian Date
        flags : type=dictionary, holds boolean flags (True if flagged) for gains.
                Must match shape of gains.
        quality : type=dictionary, holds "quality" of calibration solution. Must match
                  shape of gains. See pyuvdata.UVCal doc for more details.
        write_file : type=bool, if True, write UVCal to calfits file
        return_uvc : type=bool, if True, return UVCal object
        outdir : type=str, output file directory
        overwrite : type=bool, if True overwrite output files
        gain_convention : type=str, gain solutions formatted such that they 'multiply' into data
                          to get model, or 'divide' into data to get model
                          options=['multiply', 'divide']
        history : type=str, history string for UVCal object.
        x_orientation : type=str, orientation of X dipole, options=['east', 'north']
        telescope_name : type=str, name of telescope
        cal_style : type=str, style of calibration solutions, options=['redundant', 'sky']. If
                    cal_style == sky, additional params are required. See pyuvdata.UVCal doc.
        kwargs : additional atrributes to set in pyuvdata.UVCal
    Returns:
        if return_uvc: returns UVCal object
        else: returns None
    '''
    # get antenna info
    antenna_numbers = np.array(sorted(map(lambda k: k[0], gains.keys())), np.int)
    antenna_names = np.array(map(lambda a: "ant{}".format(a), antenna_numbers))
    Nants_data = len(antenna_numbers)
    Nants_telescope = len(antenna_numbers)
    ant_array = np.arange(Nants_data)

    # get polarization info
    pol_array = np.array(sorted(set(map(lambda k: k[1].lower(), gains.keys()))))
    jones_array = np.array(map(lambda p: jonesstr2num[p], pol_array), np.int)
    Njones = len(jones_array)

    # get time info
    time_array = np.array(times, np.float)
    Ntimes = len(time_array)
    time_range = np.array([time_array.min(), time_array.max()], np.float)
    integration_time = np.median(np.diff(time_array)) * 24. * 3600.

    # get frequency info
    freq_array = np.array(freqs, np.float)
    Nfreqs = len(freq_array)
    Nspws = 1
    freq_array = freq_array[None, :]
    spw_array = np.arange(Nspws)
    channel_width = np.median(np.diff(freq_array))

    # form gain, flags and qualities
    gain_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.complex)
    flag_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.bool)
    quality_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.float)
    for i, p in enumerate(pol_array):
        for j, a in enumerate(antenna_numbers):
            # ensure (a, p) is in gains
            if (a, p) in gains:
                gain_array[j, :, :, :, i] = gains[(a, p)].T[None, :, :]
                if flags is not None:
                    flag_array[j, :, :, :, i] = flags[(a, p)].T[None, :, :]
                else:
                    flag_array[j, :, :, :, i] = np.zeros((Nspws, Nfreqs, Ntimes), np.bool)
                if quality is not None:
                    quality_array[j, :, :, :, i] = quality[(a, p)].T[None, :, :]
                else:
                    quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.float)
            else:
                gain_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.complex)
                flag_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.bool)
                quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.float)

    # instantiate UVCal
    uvc = UVCal()

    # enforce 'gain' cal_type
    uvc.cal_type = "gain"

    # create parameter list
    params = ["Nants_data", "Nants_telescope", "Nfreqs", "Ntimes", "Nspws", "Njones",
              "ant_array", "antenna_numbers", "antenna_names", "cal_style", "history",
              "channel_width", "flag_array", "gain_array", "quality_array", "jones_array",
              "time_array", "spw_array", "freq_array", "history", "integration_time",
              "time_range", "x_orientation", "telescope_name", "gain_convention"]

    # create local parameter dict
    local_params = locals()

    # overwrite with kwarg parameters
    local_params.update(kwargs)

    # set parameters
    for p in params:
        uvc.__setattr__(p, local_params[p])

    # run check
    uvc.check()

    # write to file
    if write_file:
        # check output
        fname = os.path.join(outdir, fname)
        if os.path.exists(fname) and overwrite is False:
            print("{} exists, not overwriting...".format(fname))
        else:
            print "saving {}".format(fname)
            uvc.write_calfits(fname, clobber=True)

    # return object
    if return_uvc:
        return uvc


def update_uvcal(cal, gains=None, flags=None, quals=None, add_to_history='', **kwargs):
    '''Update UVCal object with gains, flags, quals, history, and/or other parameters
    Cannot modify the shape of gain arrays. More than one spectral window is not supported.

    Arguments:
        cal: UVCal object to be updated
        gains: Dictionary of complex calibration gains with shape=(Ntimes,Nfreqs)
            with keys in the (1,'x') format. Default (None) leaves unchanged.
        flags: Dictionary like gains but of flags. Default (None) leaves unchanged.
        quals: Dictionary like gains but of per-antenna quality. Default (None) leaves unchanged.
        add_to_history: appends a string to the history of the output file
        overwrite: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributs to their new values. 
            See pyuvdata.UVCal documentation for more info.
    '''
    # Set gains, flags, and/or quals
    for i, ant in enumerate(cal.ant_array):
        for ip, pol in enumerate(cal.jones_array):
            if gains is not None:
                cal.gain_array[i, 0, :, :, ip] = gains[(ant, jonesnum2str[pol])].T
            if flags is not None:
                cal.flag_array[i, 0, :, :, ip] = flags[(ant, jonesnum2str[pol])].T
            if quals is not None:
                cal.quality_array[i, 0, :, :, ip] = quals[(ant, jonesnum2str[pol])].T

    # Set additional attributes
    cal.history += add_to_history
    for attribute, value in kwargs.items():
        cal.__setattr__(attribute, value)
    cal.check()


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

    # Write to calfits file
    cal.write_calfits(outfilename, clobber=clobber)


def apply_cal(uvfiles, apply_gain_files=None, unapply_gain_files=None, ext='C', filetype='miriad', outdir=None, 
              overwrite=False, flag_missing=True, history=None, verbose=True):
    """
    Apply calibration gain solutions in *.calfits file to complex visibility data in uvfiles, and write
    to file with extension.

    Parameters:
    -----------
    uvfiles : type=list, path to visibility file, or list of paths to files

    apply_gain_files : type=str, path to calibration gain solution *.calfits file(s) to apply to uvfile(s)

    ext : type=str, file extension to input uvfile when writing to disk

    filetype : type=str, uvfile filetype. options=['miriad']

    outdir : type=str, output directory when writing outputfile, default is path to input file

    overwrite : type=bool, if True, overwrite output files

    flag_missing : type=bool, if True, flag antennas in uvfile if missing in *.calfits file

    unapply_gain_files : type=list, path to gain solution file(s) to un-apply to uvfiles(s)
    
    history : type=str, history string to append to data history. If None, will append names of calfits files.

    verbose : type=bool, if True, report feedback to stdout

    Notes:
    ------
    The gain convention is taken from the first file in apply_gain_files, which if None, is then taken from
    the first file in unapply_gain_files.
    """
    # check type of lists
    if isinstance(uvfiles, str):
        uvfiles = [uvfiles]
    if apply_gain_files is not None and isinstance(apply_gain_files, str):
        apply_gain_files = [apply_gain_files]
    if unapply_gain_files is not None and isinstance(unapply_gain_files, str):
        unapply_gain_files = [unapply_gain_files]

    # check both apply_gain_files and unapply_gain_files
    if apply_gain_files is None and unapply_gain_files is None:
        raise ValueError("apply_gain_files and unapply_gain_files can't both be None")

    # load apply_gain calfits file(s) and merge gains appropriately
    if apply_gain_files is not None:
        for i, gf in enumerate(apply_gain_files):
            # load UVCal object
            echo("reading {}".format(gf), type=1, verbose=verbose)
            uvc = UVCal()
            uvc.read_calfits(gf)
            if i == 0:
                gain_convention = uvc.gain_convention
                gains, flags = load_cal(uvc)
            else:
                gc = 'multiply'
                if uvc.gain_convention != gain_convention: gc = 'divide'
                g, f = load_cal(uvc)
                gains, flags = utils.merge_gains([gains, g], flags=[flags, f], gain_convention=gc)

    # load unapply_gain calfits file(s) and merge gains appropriately
    if unapply_gain_files is not None:
        for i, gf in enumerate(unapply_gain_files):
            # load UVCal object
            echo("reading {}".format(gf), type=1, verbose=verbose)
            uvc = UVCal()
            uvc.read_calfits(gf)
            if i == 0:
                if uvc.gain_convention == 'multiply':
                    unapply_gain_convention = 'divide'
                elif uvc.gain_convention == 'divide':
                    unapply_gain_convention = 'multiply'
                unapply_gains, unapply_flags = load_cal(uvc)
            else:
                gc = 'multiply'
                if uvc.gain_convention != unapply_gain_convention: gc = 'divide'
                ug, uf = load_cal(uvc)
                unapply_gains, unapply_flags = utils.merge_gains([unapply_gains, ug], flags=[unapply_flags, uf], gain_convention=gc)

        # merge with gains if necessary
        if apply_gain_files is not None:
            gains, flags = utils.merge_gains([gains, unapply_gains], flags=[flags, unapply_flags], gain_convention='divide')
        else:
            gains = unapply_gains
            flags = unapply_flags
            gain_convention = unapply_gain_convention

    # iterate over data files
    for i, uvf in enumerate(uvfiles):
        # get outdir
        if outdir is None:
            outpath = os.path.dirname(uvf)
        else:
            outpath = outdir

        # construct fname and check output
        fname = os.path.join(outpath, os.path.basename(uvf)) + ext
        if os.path.exists(fname) and overwrite is False:
            echo("{} exists, not overwriting...".format(fname), type=1, verbose=verbose)
            continue

        # load data
        echo("reading {}".format(uvf), type=1, verbose=verbose)
        uvd = UVData()

        # read uvfile
        if filetype == 'miriad':
            uvd.read_miriad(uvf)
        else:
            raise AttributeError("didn't recognize filetype {}".format(filetype))

        # ensure drift-scan
        if uvd.phase_type != 'drift':
            uvd.unphase_to_drift()

        # load data into dictionary
        data, data_flags = load_vis(uvd)
        data_pols = map(lambda p: polnum2str[p], uvd.polarization_array)

        # apply gains
        new_data, new_data_flags = utils.apply_gains(gains, data, gain_flags=flags, data_flags=data_flags,
                                                     flag_missing=flag_missing, gain_convention=gain_convention)

        # insert into uvdata object
        for k in new_data.keys():
            indices = uvd.antpair2ind(*k[:2])
            p = data_pols.index(k[2])
            uvd.data_array[indices, 0, :, p] = new_data[k]
            uvd.flag_array[indices, 0, :, p] = new_data_flags[k]

        # append history
        if history is None:
            history = ''
            if apply_gain_files is not None:
                for gf in apply_gain_files:
                    history += " : applied {} :".format(os.path.basename(gf))
            if unapply_gain_files is not None:
                for gf in unapply_gain_files:
                    history += " : un-applied {} :".format(os.path.basename(gf))

        uvd.history += history

        # write to file
        if filetype == 'miriad':
            echo("writing {}".format(fname), type=0, verbose=verbose)
            uvd.write_miriad(fname, clobber=True)


def apply_cal_arg_parser():
    """
    argparser for apply_cal() function.
    """
    a = argparse.ArgumentParser(description="Apply (or un-apply) *.calfits file(s) to visibility file(s).")
    a.add_argument("--uvfiles", type=str, nargs='*', help="space-delimited paths to visibility file(s).")
    a.add_argument("--apply_gains", default=None, type=str, nargs='*', help="space-delimited path to *.calfits file(s) to apply on uvfiles.")
    a.add_argument("--unapply_gains", default=None, type=str, nargs='*', help="space-delimited path to *.calfits file(s) to unapply on uvfiles.")
    a.add_argument("--ext", type=str, default="C", help="filename extension to uvfile when writing out data.")
    a.add_argument("--outdir", type=str, default=None, help="output directory to write data, default is path to uvfile")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--noflag_missing", default=False, action='store_true', help="don't flag antennas in uvfile that are missing in *.calfits solution.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output to stdout")
    return a


def combine_calfits(files, fname, outdir=None, overwrite=False, broadcast_flags=True, verbose=True):
    """
    multiply together multiple calfits gain solutions (overlapping in time and frequency)

    Parameters:
    -----------
    files : type=list, dtype=str, list of files to multiply together

    fname : type=str, output filename

    outdir : type=str, path to output directory

    overwrite : type=bool, overwrite output file

    broadcast_flags : type=bool, if True, take logical AND of flags across all calfits, else
        take logical OR of flags across all calfits
    """
    # get io params
    if outdir is None:
        outdir = "./"

    output_fname = os.path.join(outdir, fname)
    if os.path.exists(fname) and overwrite is False:
        raise IOError("{} exists, not overwriting".format(output_fname))

    # iterate over files
    for i, f in enumerate(files):
        if i == 0:
            echo("...loading {}".format(f), verbose=verbose)
            uvc = UVCal()
            uvc.read_calfits(f)

            # set flagged data to unity
            uvc.gain_array[uvc.flag_array] /= uvc.gain_array[uvc.flag_array]

        else:
            uvc2 = UVCal()
            uvc2.read_calfits(f)

            # set flagged data to unity
            gain_array = uvc2.gain_array
            gain_array[uvc2.flag_array] /= gain_array[uvc2.flag_array]

            # multiply gain solutions in
            uvc.gain_array *= uvc2.gain_array

            # pass flags
            if broadcast_flags:
                uvc.flag_array += uvc2.flag_array
            else:
                uvc.flag_array *= uvc2.flag_array

    # write to file
    echo("...saving {}".format(output_fname), verbose=verbose)
    uvc.write_calfits(output_fname, clobber=True)


def combine_calfits_arg_parser():
    """
    argparser for combine_calfits() function.
    """
    a = argparse.ArgumentParser(description="Multiply together gains from multiple *.calfits files.")
    a.add_argument("files", type=str, nargs='*', help="list of calfits files to combine.")
    a.add_argument("--fname", type=str, help="output calfits filename.", required=True)
    a.add_argument("--outdir", default=None, type=str, help="path to output directory")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--no_broadcast", default=False, action='store_true', help="don't broadcast flags from each input calfits file")
    a.add_argument("--silence", action='store_true', default=False, help="stop feedback to stdout.")
    return a

