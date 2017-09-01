import numpy as np
import aipy
import pyuvdata.utils as uvutils
cm_p_m = 100 #yes, this is a thing. cm per meter

class AntennaArray(aipy.pol.AntennaArray):
    def __init__(self, *args, **kwargs):
        aipy.pol.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')


    def update(self):
        aipy.pol.AntennaArray.update(self)

    def get_params(self, ant_prms={'*': '*'}):
        try:
            prms = aipy.pol.AntennaArray.get_params(self, ant_prms)
        except(IndexError):
            return {}
        for k in ant_prms:
            try:
                #rotate from equatorial to zenith
                top_pos = np.dot(self._eq2zen, self[int(k)].pos)
                #convert from ns to m
                top_pos *= aipy.const.len_ns / cm_p_m

            except(ValueError):
                continue
            if ant_prms[k] == '*':
                prms[k].update({'top_x': top_pos[0], 'top_y': top_pos[1], 'top_z': top_pos[2]})
            else:
                for val in ant_prms[k]:
                    if val == 'top_x':
                        prms[k]['top_x'] = top_pos[0]
                    elif val == 'top_y':
                        prms[k]['top_y'] = top_pos[1]
                    elif val == 'top_z':
                        prms[k]['top_z'] = top_pos[2]
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
                ant.pos = np.dot(np.linalg.inv(self._eq2zen), top_pos) / aipy.const.len_ns * cm_p_m
            changed |= ant_changed
        if changed:
            self.update()
        return changed


def get_HERA_aa(freqs,calfile='hera_cm',**kwargs):
    # create an aipy AntennaArray object using positions and hookup info in M&C.
    #
    # Inputs:
    # freqs = numpy array of frequencies in GHz
    #   - used to compute uvws by some aipy functions
    #   - for applications using antenna locations in meters, a nominal input
    #        value of np.array([0.15]) is probably fine
    # calfile = python library containing get_aa function.
    #   - default value is hera_cal.hera_cm handles import from M&C
    #   - don't include the .py
    #   - needs to be in your python path
    # array_epoch_jd = julian date of desired configuration
    #   - if not input uses default date set in calfile, hera_cal.hera_cm
    # locations_file = antenna location csv file exported from m&c
    #   - default file included in hera_cal
    #   - create a new one with hera_mc/scripts/write_antenna_location_file.py
    #

    #sometimes the input is set to None, which implies default
    if calfile is None:
        calfile='hera_cm'

    exec('from {calfile} import get_aa'.format(calfile=calfile))
    if calfile!='hera_cm':
        #array position is set based on user input cal file
        return get_aa(freqs)
    else:
        #use the time and position file aware get_aa
        return get_aa(freqs,**kwargs)

def get_aa_from_uv(uvd):
    '''
    Generate an AntennaArray object from a pyuvdata UVData object.

    This function creates an AntennaArray object from the metadata
    contained in a UVData object. Critically, it assumes that the
    antenna positions are in the correct miriad format (ECEF, rotated
    so the x-axis passes through the local meridian), and that the CofA
    (center of array) values are set properly for performing coordinate
    transformations.

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
    # MAKE SURE COORDINATES ARE CORRECT (aipy is expecting lat/lon in radians)
    cofa_lat, cofa_lon, cofa_alt = uvd.telescope_location_lat_lon_alt
    location = (cofa_lat, cofa_lon, cofa_alt)

    # get antenna positions from file
    antpos = {}
    for i, antnum in enumerate(uvd.antenna_numbers):
        # miriad antenna positions are rotated ECEF, so we need to convert:
        #   rotECEF -> ECEF -> ENU (east-north-up)
        pos = uvd.antenna_positions[i, :] + uvd.telescope_location
        ecef = uvutils.ECEF_from_rotECEF(pos, cofa_lon)
        enu = uvutils.ENU_from_ECEF(ecef, cofa_lat, cofa_lon, cofa_alt)

        # make a dict for parameter-setting purposes later
        antpos[antnum] = {'top_x': enu[0], 'top_y': enu[1], 'top_z': enu[2]}

    # make antpos_ideal array
    nants = np.max(antpos.keys()) + 1
    antpos_ideal = np.zeros(shape=(nants, 3), dtype=float) - 1
    tops = {'top_x': 0, 'top_y': 1, 'top_z': 2}
    # unpack from dict -> numpy array
    for k in antpos.keys():
        for i, m in enumerate(antpos[k]):
            antpos_ideal[k, tops[m]] = antpos[k][m]
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
