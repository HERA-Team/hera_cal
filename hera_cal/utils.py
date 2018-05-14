import numpy as np
import aipy
import astropy.constants as const
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as unt
import pyuvdata.utils as uvutils
from pyuvdata import UVCal, UVData
import os
import hera_cal
import copy
from scipy.interpolate import interp1d
from collections import OrderedDict as odict


class AntennaArray(aipy.pol.AntennaArray):
    def __init__(self, *args, **kwargs):
        aipy.pol.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')
        # yes, this is a thing. cm per meter
        self.cm_p_m = 100.

    def update(self):
        aipy.pol.AntennaArray.update(self)

    def get_params(self, ant_prms={'*': '*'}):
        try:
            prms = aipy.pol.AntennaArray.get_params(self, ant_prms)
        except(IndexError):
            return {}
        return prms

    def set_params(self, prms):
        changed = aipy.pol.AntennaArray.set_params(self, prms)
        for i, ant in enumerate(self):
            ant_changed = False
            top_pos = np.dot(self._eq2zen, ant.pos)
            try:
                top_pos[0] = prms[str(i)]['top_x']
                ant_changed = True
            except(KeyError):
                pass
            try:
                top_pos[1] = prms[str(i)]['top_y']
                ant_changed = True
            except(KeyError):
                pass
            try:
                top_pos[2] = prms[str(i)]['top_z']
                ant_changed = True
            except(KeyError):
                pass
            if ant_changed:
                #rotate from zenith to equatorial, convert from meters to ns
                ant.pos = np.dot(np.linalg.inv(self._eq2zen), top_pos) / aipy.const.len_ns * self.cm_p_m
            changed |= ant_changed
        if changed:
            self.update()
        return changed


def get_aa_from_uv(uvd, freqs=[0.15]):
    '''
    Generate an AntennaArray object from a pyuvdata UVData object.

    This function creates an AntennaArray object from the metadata
    contained in a UVData object. It assumes that the antenna_positions
    array in the UVData object is in earth-centered, earth-fixed (ECEF)
    coordinates, relative to the center of the array, also given in
    ECEF coordinates. We must add these together, and then rotate so that
    the x-axis is aligned with the local meridian (rotECEF). rotECEF is the
    coordinate system for Antenna objects in the AntennaArray object (which
    inherits this behavior from MIRIAD). It is also expected that distances
    are given in nanoseconds, rather than meters, also because of the
    default behavior in MIRIAD.

    Arguments
    ====================
    uvd: a pyuvdata UVData object containing the data.
    freqs (optional): list of frequencies to pass to aa object. Defaults to single frequency
        (150 MHz), suitable for computing redundancy and uvw info.

    Returns
    ====================
    aa: AntennaArray object that can be used to calculate redundancies from
       antenna positions.
    '''
    # center of array values from file
    cofa_lat, cofa_lon, cofa_alt = uvd.telescope_location_lat_lon_alt
    location = (cofa_lat, cofa_lon, cofa_alt)

    # get antenna positions from file
    antpos = {}
    for i, antnum in enumerate(uvd.antenna_numbers):
        # we need to add the CofA location to the relative coordinates
        pos = uvd.antenna_positions[i, :] + uvd.telescope_location
        # convert from meters to nanoseconds
        c_ns = const.c.to('m/ns').value
        pos = pos / c_ns

        # rotate from ECEF -> rotECEF
        rotECEF = uvutils.rotECEF_from_ECEF(pos, cofa_lon)

        # make a dict for parameter-setting purposes later
        antpos[antnum] = {'x': rotECEF[0], 'y': rotECEF[1], 'z': rotECEF[2]}

    # make antpos_ideal array
    nants = np.max(antpos.keys()) + 1
    antpos_ideal = np.zeros(shape=(nants, 3), dtype=float) - 1
    # unpack from dict -> numpy array
    for k in antpos.keys():
        antpos_ideal[k, :] = np.array([antpos[k]['x'], antpos[k]['y'], antpos[k]['z']])
    freqs = np.asarray(freqs)
    # Make list of antennas.
    # These are values for a zenith-pointing antenna, with a dummy Gaussian beam.
    antennas = []
    for i in range(nants):
        beam = aipy.fit.Beam(freqs)
        phsoff = {'x': [0., 0.], 'y': [0., 0.]}
        amp = 1.
        amp = {'x': amp, 'y': amp}
        bp_r = [1.]
        bp_r = {'x': bp_r, 'y': bp_r}
        bp_i = [0., 0., 0.]
        bp_i = {'x': bp_i, 'y': bp_i}
        twist = 0.
        antennas.append(aipy.pol.Antenna(0., 0., 0., beam, phsoff=phsoff,
                        amp=amp, bp_r=bp_r, bp_i=bp_i, pointing=(0., np.pi / 2, twist)))

    # Make the AntennaArray and set position parameters
    aa = AntennaArray(location, antennas, antpos_ideal=antpos_ideal)
    pos_prms = {}
    for i in antpos.keys():
        pos_prms[str(i)] = antpos[i]
    aa.set_params(pos_prms)
    return aa


def get_aa_from_calfile(freqs, calfile, **kwargs):
    '''
    Generate an AntennaArray object from the specified calfile.

    Arguments:
    ====================
    freqs: list of frequencies in data file, in GHz
    calfile: name of calfile, without the .py extension (e.g., hsa7458_v001). Note that this
        file must be in sys.path.

    Returns:
    ====================
    aa: AntennaArray object
    '''
    exec('from {calfile} import get_aa'.format(calfile=calfile))

    # generate aa
    return get_aa(freqs, **kwargs)


def JD2LST(JD, longitude=21.42830):
    """
    Input:
    ------
    JD : type=float or list of floats containing Julian Date(s) of an observation

    longitude : type=float, longitude of observer in degrees East, default=HERA longitude

    Output:
    -------
    Local Apparent Sidreal Time [radians]

    Notes:
    ------
    The Local Apparent Sidereal Time is *defined* as the right ascension in the current epoch.
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # iterate over JD
    LST = []
    for jd in JD:
        # construct astropy Time object
        t = Time(jd, format='jd', scale='utc')
        # get LST in radians at epoch of jd
        LST.append(t.sidereal_time('apparent', longitude=longitude*unt.deg).radian)
    LST = np.array(LST)

    if _array:
        return LST
    else:
        return LST[0]


def LST2JD(LST, start_jd, longitude=21.42830):
    """
    Convert Local Apparent Sidereal Time -> Julian Date via a linear fit
    at the 'start_JD' anchor point.

    Input:
    ------
    LST : type=float, local apparent sidereal time [radians]

    start_jd : type=int, integer julian day to use as starting point for LST2JD conversion

    longitude : type=float, degrees East of observer, default=HERA longitude

    Output:
    -------
    JD : type=float, Julian Date(s). accurate to ~1 milliseconds
    """
    # get LST type
    if isinstance(LST, list) or isinstance(LST, np.ndarray):
        _array = True
    else:
        LST = [LST]
        _array = False

    # get start_JD
    base_jd = float(start_jd)

    # iterate over LST
    jd_array = []
    for lst in LST:
        while True:
            # calculate fit
            jd1 = start_jd
            jd2 = start_jd + 0.01
            lst1, lst2 = JD2LST(jd1, longitude=longitude), JD2LST(jd2, longitude=longitude)
            slope = (lst2 - lst1) / 0.01
            offset = lst1 - slope * jd1

            # solve y = mx + b for x
            JD = (lst - offset) / slope

            # redo if JD isn't on starting JD
            if JD - base_jd < 0:
                start_jd += 1
            elif JD - base_jd > 1:
                start_jd -= 1
            else:
                break
        jd_array.append(JD)

    jd_array = np.array(jd_array)

    if _array:
        return jd_array
    else:
        return jd_array[0]


def JD2RA(JD, longitude=21.42830, latitude=-30.72152, epoch='current'):
    """
    Convert from Julian date to Equatorial Right Ascension at zenith
    during a specified epoch.

    Parameters:
    -----------
    JD : type=float, a float or an array of Julian Dates

    longitude : type=float, longitude of observer in degrees east, default=HERA longitude

    latitude : type=float, latitude of observer in degrees north, default=HERA latitutde
               This only matters when using epoch="J2000"

    epoch : type=str, epoch for RA calculation. options=['current', 'J2000'].
            The 'current' epoch is the epoch at JD. Note that
            LST is defined as the zenith RA in the current epoch. Note that
            epoch='J2000' corresponds to the ICRS standard.

    Output:
    -------
    RA : type=float, right ascension [degrees] at zenith JD times
         in the specified epoch.
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # setup RA list
    RA = []

    # iterate over jd
    for jd in JD:

        # use current epoch calculation
        if epoch == 'current':
            ra = JD2LST(jd, longitude=longitude) * 180 / np.pi
            RA.append(ra)

        # use J2000 epoch
        elif epoch == 'J2000':
            loc = crd.EarthLocation(lat=latitude*unt.deg, lon=longitude*unt.deg)
            t = Time(jd, format='jd', scale='utc')
            zen = crd.SkyCoord(frame='altaz', alt=90*unt.deg, az=0*unt.deg, obstime=t, location=loc)
            RA.append(zen.icrs.ra.degree)

        else:
            raise ValueError("didn't recognize {} epoch".format(epoch))

    RA = np.array(RA)

    if _array:
        return RA
    else:
        return RA[0]


def get_sun_alt(jds, longitude=21.42830, latitude=-30.72152):
    """
    Given longitude and latitude, get the Solar alittude at a given time.

    Parameters
    ----------
    jds : float or ndarray of floats
        Array of Julian Dates

    longitude : float
        Longitude of observer in degrees East

    latitude : float
        Latitude of observer in degrees North

    Returns
    -------
    alts : float or ndarray
        Array of altitudes [degrees] of the Sun
    """
    # type check
    array = True
    if isinstance(jds, (float, np.float, np.float64, int, np.int, np.int32)):
        jds = [jds]
        array = False 

    # get earth location
    e = crd.EarthLocation(lat=latitude*unt.deg, lon=longitude*unt.deg)

    # get AltAz frame
    a = crd.AltAz(location=e)

    # get Sun locations
    alts = np.array(map(lambda t: crd.get_sun(Time(t, format='jd')).transform_to(a).alt.value, jds))

    if array:
        return alts
    else:
        return alts[0]


def solar_flag(flags, times=None, flag_alt=0.0, longitude=21.42830, latitude=-30.72152, inplace=False, 
               interp=False, interp_Nsteps=11, verbose=True):
    """
    Apply flags at times when the Sun is above some minimum altitude.

    Parameters
    ----------
    flags : flag ndarray, or DataContainer, or pyuvdata.UVData object

    start_jd : int
        Integer Julian Date to perform calculation for

    times : 1D float ndarray
        If flags is an ndarray or DataContainer, this contains the time bins 
        of the data's time axis in Julian Date

    flag_alt : float
        If the Sun is greater than this altitude [degrees], we flag the data.

    longitude : float
        Longitude of observer in degrees East (if flags is a UVData object, 
        use its stored longitude instead)

    latitude : float
        Latitude of observer in degrees North (if flags is a UVData object, 
        use its stored latitude instead)
        
    inplace: bool
        If inplace, edit flags instead of returning a new flags object.

    interp : bool
        If True, evaluate Solar altitude with a coarse grid and interpolate at times values.

    interp_Nsteps : int
        Number of steps from times.min() to times.max() to use in get_solar_alt call.
        If the range of times is <= a single day, Nsteps=11 is a good-enough resolution.

    verbose : bool
        if True, print feedback to standard output

    Returns
    -------
    flags : solar-applied flags, same format as input
    """
    # type check
    if isinstance(flags, hera_cal.datacontainer.DataContainer):
        dtype = 'DC'
    elif isinstance(flags, np.ndarray):
        dtype = 'ndarr'
    elif isinstance(flags, UVData):
        if verbose: print "Note: using latitude and longitude in given UVData object"
        latitude, longitude, altitude = flags.telescope_location_lat_lon_alt_degrees
        times = np.unique(flags.time_array)
        dtype = 'uvd'
    if dtype in ['ndarr', 'DC']:
        assert times is not None, "if flags is an ndarray or DataContainer, must feed in times"

    # inplace
    if not inplace:
        flags = copy.deepcopy(flags)

    # get solar alts
    if interp:
        # first evaluate coarse grid, then interpolate
        _times = np.linspace(times.min(), times.max(), interp_Nsteps)
        _alts = get_sun_alt(_times, longitude=longitude, latitude=latitude)

        # interpolate _alts
        alts = interp1d(_times, _alts, kind='quadratic')(times)
    else:
        # directly evaluate solar altitude at times
        alts = get_sun_alt(times, longitude=longitude, latitude=latitude)

    # apply flags
    if dtype == 'DC':
        for k in flags.keys():
            flags[k][alts > flag_alt, :] = True
    elif dtype == 'ndarr':
        flags[alts > flag_alt, :] = True
    elif dtype == 'uvd':
        for t, a in zip(times, alts):
            if a > flag_alt:
                flags.flag_array[np.isclose(flags.time_array, t)] = True

    if not inplace:
        return flags


def combine_calfits(files, fname, outdir=None, overwrite=False, broadcast_flags=True, verbose=True):
    """
    multiply together multiple calfits gain solutions (overlapping in time and frequency)

    Parameters:
    -----------
    files : type=list, dtype=str, list of files to multiply together

    fname : type=str, path to output filename

    outdir : type=str, path to output directory

    overwrite : type=bool, overwrite output file

    broadcast_flags : type=bool, if True, broadcast flags from each calfits to final solution
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
            hera_cal.abscal_funcs.echo("...loading {}".format(f), verbose=verbose)
            uvc = UVCal()
            uvc.read_calfits(f)
            f1 = copy.copy(f)

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
                uvc.flag_array = uvc.flag_array * uvc2.flag_array

    # write to file
    hera_cal.abscal_funcs.echo("...saving {}".format(output_fname), verbose=verbose)
    uvc.write_calfits(output_fname, clobber=True)


def get_miriad_times(filepaths, add_int_buffer=False):
    """
    Use aipy to get filetimes for Miriad file paths.
    All times mark the center of an integration, which is not the standard Miriad format.

    Parameters:
    ------------
    filepaths : type=list, list of filepaths

    add_int_buffer : type=bool, if True, extend stop times by an integration duration.

    Output: (file_starts, file_stops, int_times)
    -------
    file_starts : file starting point in LST [radians]

    file_stops : file ending point in LST [radians]

    int_times : integration duration in LST [radians]
    """
    _array = True
    # check filepaths type
    if type(filepaths) == str:
        _array = False
        filepaths = [filepaths]

    # form empty lists
    file_starts = []
    file_stops = []
    int_times = []

    # get Nfiles
    Nfiles = len(filepaths)

    # iterate over filepaths and extract time info
    for i, f in enumerate(filepaths):
        uv = aipy.miriad.UV(f)
        # get integration time
        int_time = uv['inttime'] * 2*np.pi / (23.9344699*3600.)
        # get start and stop
        start = uv['lst']
        stop = start + (uv['ntimes']-1) * int_time
        # add integration buffer to end of file if desired
        if add_int_buffer:
            stop += int_time
        # add half an integration to get center of integration
        start += int_time / 2
        stop += int_time / 2
        file_starts.append(start)
        file_stops.append(stop)
        int_times.append(int_time)

    file_starts = np.array(file_starts)
    file_stops = np.array(file_stops)
    int_times = np.array(int_times)

    # make sure times don't wrap
    file_starts[np.where(file_starts < 0)] += 2*np.pi
    file_stops[np.where(file_stops >= 2*np.pi)] -= 2*np.pi

    if _array is False:
        file_starts = file_starts[0]
        file_stops = file_stops[0]
        int_times = int_times[0]

    return file_starts, file_stops, int_times


def lst_rephase(data, bls, freqs, dlst, lat=-30.72152, inplace=True, array=False):
    """
    Shift phase center of each integration in data by amount dlst [radians] along right ascension axis.
    If inplace == True, this function directly edits the arrays in 'data' in memory, so as not to 
    make a copy of data.

    Parameters:
    -----------
    data : type=DataContainer, holding 2D visibility data, with [0] axis time and [1] axis frequency

    bls : type=dictionary, same keys as data, values are 3D float arrays holding baseline vector
                            in ENU frame in meters

    freqs : type=ndarray, frequency array of data [Hz]

    dlst : type=ndarray or float, delta-LST to rephase by [radians]. If a float, shift all integrations
                by dlst, elif an ndarray, shift each integration by different amount w/ shape=(Ntimes)

    lat : type=float, latitude of observer in degrees North

    inplace : type=bool, if True edit arrays in data in memory, else make a copy and return

    array : type=bool, if True, treat data as a visibility ndarray and bls as a baseline vector

    Notes:
    ------
    The rephasing uses aipy.coord.top2eq_m and aipy.coord.eq2top_m matrices to convert from
    array TOPO frame to Equatorial frame, induces time rotation, converts back to TOPO frame,
    calculates new pointing vector s_prime and inserts a delay plane into the data for rephasing.

    This method of rephasing follows Eqn. 21 & 22 of Zhang, Y. et al. 2018 "Unlocking Sensitivity..."
    """
    # check format of dlst
    if isinstance(dlst, list):
        lat = np.ones_like(dlst) * lat
        dlst = np.array(dlst)
        zero = np.zeros_like(dlst)
    elif isinstance(dlst, np.ndarray):
        lat = np.ones_like(dlst) * lat
        zero = np.zeros_like(dlst)

    else:
        zero = 0

    # get top2eq matrix
    top2eq = uvutils.top2eq_m(zero, lat*np.pi/180)

    # get eq2top matrix
    eq2top = uvutils.eq2top_m(-dlst, lat*np.pi/180)

    # get full rotation matrix
    rot = np.einsum("...jk,...kl->...jl", eq2top, top2eq)

    # make copy of data if desired
    if inplace == False:
        data = copy.deepcopy(data)

    # turn array into dict
    if array:
        inplace = False
        data = {'data': data}
        bls = {'data': bls}

    # iterate over data keys
    for i, k in enumerate(data.keys()):

        # get new s-hat vector
        s_prime = np.einsum("...ij,j->...i", rot, np.array([0.0, 0.0, 1.0]))
        s_diff = s_prime - np.array([0., 0., 1.0])

        # get baseline vector
        bl = bls[k]

        # dot bl with difference of pointing vectors to get new u: Zhang, Y. et al. 2018 (Eqn. 22)
        u = np.einsum("...i,i->...", s_diff, bl)

        # get delay
        tau = u / (aipy.const.c / 100.0)

        # reshape tau
        if type(tau) == np.ndarray:
            pass
        else:
            tau = np.array([tau])

        # get phasor
        phs = np.exp(-2j*np.pi*freqs[None, :]*tau[:, None])

        # multiply into data
        data[k] *= phs

    if array:
        data = data['data']
        
    if inplace == False:
        return data


def synthesize_ant_flags(flags, threshold=0.0):
    '''
    Synthesizes flags on visibilities into flags on antennas. For a given antenna and
    a given time and frequency, if the fraction of flagged pixels in all visibilities with that
    antenna exceeds 'threshold', the antenna gain is flagged at that time and frequency. This
    excludes contributions from antennas that are completely flagged, i.e. are dead.

    Arguments:
        flags: DataContainer containing boolean data flag waterfalls
        threshold: float, fraction of flagged pixels across all visibilities (with a common antenna)
            needed to flag that antenna gain at a particular time and frequency.

    Returns:
        ant_flags: dictionary mapping antenna-pol keys like (1,'x') to boolean flag waterfalls
    '''
    # type check
    assert isinstance(flags, hera_cal.datacontainer.DataContainer), "flags must be fed as a datacontainer"

    # get Ntimes and Nfreqs
    Ntimes, Nfreqs = flags[flags.keys()[0]].shape

    # get antenna-pol keys
    antpols = set([ap for (i,j,pol) in flags.keys() for ap in [(i, pol[0]), (j, pol[1])]])

    # get dictionary of completely flagged ants to exclude
    is_excluded = {ap: True for ap in antpols}
    for (i,j,pol), flags_here in flags.items():
        if not np.all(flags_here): 
            is_excluded[(i,pol[0])] = False
            is_excluded[(j,pol[1])] = False

    # construct dictionary of visibility count (number each antenna touches)
    # and dictionary of number of flagged visibilities each antenna has (excluding dead ants)
    # per time and freq
    ant_Nvis = {ap: 0 for ap in antpols}
    ant_Nflag = {ap: np.zeros((Ntimes, Nfreqs), np.float) for ap in antpols}
    for (i, j, pol), flags_here in flags.items():
        # get antenna keys
        ap1 = (i, pol[0])
        ap2 = (j, pol[1])
        # only continue if not in is_excluded
        if not is_excluded[ap1] and not is_excluded[ap2]:
            # add to Nvis count
            ant_Nvis[ap1] += 1
            ant_Nvis[ap2] += 1
            # Add to Nflag count
            ant_Nflag[ap1] += flags_here.astype(np.float)
            ant_Nflag[ap2] += flags_here.astype(np.float)

    # iterate over antpols and construct antenna gain dictionaries
    ant_flags = {}
    for ap in antpols:
        # create flagged arrays for excluded ants
        if is_excluded[ap]:
            ant_flags[ap] = np.ones((Ntimes, Nfreqs), np.bool)
        # else create flags based on threshold
        else:
            # handle Nvis = 0 cases
            if ant_Nvis[ap] == 0: ant_Nvis[ap] = 1e-10
            # create antenna flags
            ant_flags[ap] = (ant_Nflag[ap] / ant_Nvis[ap]) > threshold

    return ant_flags

