import numpy as np
import aipy
import astropy.constants as const
import pyuvdata.utils as uvutils

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


def get_aa_from_uv(uvd):
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

    Returns
    ====================
    aa: AntennaArray object that can be used to calculate redundancies from
       antenna positions.
    '''
    # Define parameters necessary for the AntnnaArary object.
    # The exact values are not important, the object just requires
    #    that these are set.
    prms = {
        'amps': dict(zip(range(128), np.ones(128))),
        'amp_coeffs': np.array([1] * 128),
        'bp_r': np.array([[1.]] * 128),
        'bp_i': np.array([[0., 0., 0.]] * 128),
        'twist': np.array([0] * 128),
        'beam': aipy.fit.Beam2DGaussian,
        'bm_prms': {'bm_xwidth': 3.39 * np.pi / 180,
                    'bm_ywidth': 3.39 * np.pi / 180}
        # Gaussian Beam is put in because something is needed.
        #   here we are specifying a 8 deg FWHM beam, bm_xwidth = 1 sigma
    }

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
    freqs = np.array([0.15])
    # Make list of antennas.
    # These are values for a zenith-pointing antenna, with a dummy Gaussian beam.
    antennas = []
    for i in range(nants):
        beam = aipy.fit.Beam(freqs)
        phsoff = {'x': [0., 0.], 'y': [0., 0.]}
        amp = prms['amps'].get(i, 4e-3)
        amp = {'x': amp, 'y': amp}
        bp_r = prms['bp_r'][i]
        bp_r = {'x': bp_r, 'y': bp_r}
        bp_i = prms['bp_i'][i]
        bp_i = {'x': bp_i, 'y': bp_i}
        twist = prms['twist'][i]
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
