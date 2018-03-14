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

    add_int_buffer : type=bool, if True, extend start and stop times by a single integration duration
        except for start of first file, and stop of last file.

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
        # add integration buffer to beginning and end if desired
        if add_int_buffer:
            if i != 0:
                start -= int_time
            if i != (Nfiles-1):
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
