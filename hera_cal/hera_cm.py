import numpy as np
import pandas as pd
import os
import aipy
import pyuvdata.utils as uv_utils
from astropy.time import Time
from dateutil.parser import parse

cm_p_m = 100 #centimeters per meter, used in AntennaArray

# default locations file
locations_file = os.path.join(os.path.dirname(__file__), 'data/hera_ant_locs_05_16_2017.csv')

# default locations epoch
array_epoch_jd = 2457458


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


def cofa_latlonalt(locations_file):
    stn_locs_df = pd.read_csv(locations_file)

    cofa = stn_locs_df[stn_locs_df['station_type'] == 'cofa'].to_dict('records')
    cofa_lat = cofa[0]['latitude']
    cofa_lon = cofa[0]['longitude']
    cofa_alt = cofa[0]['elevation']

    return cofa_lat, cofa_lon, cofa_alt


def antpos_enu(locations_file, array_epoch_jd, station_type='herahex'):
    cofa_lat, cofa_lon, cofa_alt = cofa_latlonalt(locations_file)

    stn_locs_df = pd.read_csv(locations_file)
    stn_locs_df = stn_locs_df[stn_locs_df['station_type'] == station_type]
    stn_list = stn_locs_df.to_dict('records')
    ant_pos = {}
    for stn in stn_list:
        start_jd = Time(parse(stn['start_date']), format='datetime').jd
        try:
            stop_jd = Time(parse(stn['stop_date']), format='datetime').jd
        except:
            stop_jd = None
        if (start_jd < array_epoch_jd and (stop_jd > array_epoch_jd or stop_jd is None)):
            ecef_xyz = uv_utils.XYZ_from_LatLonAlt(stn['latitude'] * np.pi / 180.0,
                                                   stn['longitude'] * np.pi / 180.0,
                                                   stn['elevation'])
            # now need to convert ecef to ENU (east, north, up = topocentric) at HERA
            enu = uv_utils.ENU_from_ECEF(ecef_xyz, cofa_lat * np.pi / 180.0,
                                         cofa_lon * np.pi / 180.0, cofa_alt)
            ant_pos[stn['antenna_number']] = {'top_x': enu[0], 'top_y': enu[1], 'top_z': enu[2]}
    return ant_pos


def antpos_enu_mc(station_type='herahex'):
    '''
    Perform a live call to the M&C database to get current antenna positions.

    This function uses the get_cminfo_correlator function from hera_mc to get
    the current locations of the array antennas. There is no support for
    historical data, so post-dated processing should be done by using the
    antpos_enu function above with a generated calfile.

    Arguments
    ====================
    station_type: what kind of antenna we should filter on (e.g., herahex,
        paperimaging, etc.)

    Returns
    ====================
    ant_pos: dictionary-of-dictionaries of antenna positions. Primary
        keys are station numbers. Secondary keys are 'top_x', 'top_y',
        and 'top_z', which are the antenna positions in local ENU
        coordinates.
    '''
    # connect to M&C database
    from hera_mc import geo_handling
    h = geo_handling.Handling()
    stations_conn = h.get_cminfo_correlator()

    # antenna info
    antenna_names = cminfo['antenna_numbers']
    antenna_positions = cminfo['antenna_positions']
    station_types = cminfo['station_types']

    # cofa info
    cofa_loc = h.cofa()[0]

    # build up dict of ant_pos
    ant_pos = {}
    for i, stn in enumerate(antenna_names):
        # make sure this station is the correct type
        if station_types[i] != station_type:
            continue

        # get positions in rotated ECEF
        rotECEF = antenna_positions[i, :]

        # rotate to ECEF
        ecef = uv_utils.ECEF_from_rotECEF(rotECEF, cofa_loc.lon)

        # get ENU (east, north, up) at HERA
        enu = uv_utils.ENU_from_ECEF(ecef, cofa_loc.lat * np.pi / 180.0,
                                     cofa_loc.lon * np.pi / 180.0, cofa_alt)
        ant_pos[stn] = {'top_x': enu[0], 'top_y': enu[1], 'top_z': enu[2]}
    return ant_pos


prms = {#'loc': (cofa_lat, cofa_lon),
        #'antpos_ideal': antpos_enu(locations_file,array_epoch_jd),
        'amps': dict(zip(range(128), np.ones(128))),
        'amp_coeffs': np.array([1] * 128),
        'bp_r': np.array([[1.]] * 128),
        'bp_i': np.array([[0., 0., 0.]] * 128),
        'twist': np.array([0] * 128),
        'beam': aipy.fit.Beam2DGaussian,
        'bm_prms': {'bm_xwidth':3.39*np.pi/180,'bm_ywidth':3.39*np.pi/180}
        # Gaussian Beam is put in because something is needed.
        #   here we are specifying a 8 deg FWHM beam, bm_xwidth = 1 sigma
}


def get_aa(freqs, locations_file=locations_file, array_epoch_jd=array_epoch_jd):
    '''Return the HERA AntennaArray.'''
    cofa_lat, cofa_lon, cofa_alt = cofa_latlonalt(locations_file)
    location = (cofa_lat * np.pi/180, cofa_lon * np.pi/180)
    antennas = []
    try:
        # make a live call to get current configuration
        antpos = antpos_enu_mc(station_type='herahex')
    except:
        # load the positions from the file, filter by epoch
        antpos = antpos_enu(locations_file, array_epoch_jd, station_type='herahex')
    nants = np.max(antpos.keys()) + 1
    antpos_ideal = np.zeros(shape=(nants, 3), dtype=float) - 1
    tops = {'top_x': 0, 'top_y': 1, 'top_z': 2}
    for k in antpos.keys():
        for i, m in enumerate(antpos[k]):
            antpos_ideal[k, tops[m]] = antpos[k][m]
    for i in range(nants):
        beam = prms['beam'](freqs)
        try:
            beam.set_params(prms['bm_prms'])
        except(AttributeError):
            pass
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
    aa = AntennaArray(location, antennas, antpos_ideal=antpos_ideal)
    pos_prms = {}
    for i in antpos.keys():
        pos_prms[str(i)] = antpos[i]
    aa.set_params(pos_prms)
    return aa
