import numpy as np
import pandas as pd
import os
import aipy
import pyuvdata.utils as uv_utils
from astropy.time import Time
from dateutil.parser import parse

cm_p_m = 100 #centimeters per meter, used in AntennaArray

#default locations file
locations_file = os.path.join(os.path.dirname(__file__), 'data/hera_ant_locs_05_16_2017.csv')

#default locations epoch
array_epoch_jd = 2457458


class AntennaArray(aipy.pol.AntennaArray):
    def __init__(self, *args, **kwargs):
        aipy.pol.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')

    def update_gains(self):
        gains = self.gain * self.amp_coeffs
        for i, gain in zip(self.ant_layout.flatten(), gains.flatten()):
            self[i].set_params({'amp_x': gain})
            self[i].set_params({'amp_y': gain})

    def update_delays(self):
        ns, ew = np.indices(self.ant_layout.shape)
        dlys = ns * self.tau_ns + ew * self.tau_ew + self.dly_coeffs
        for i, tau in zip(self.ant_layout.flatten(), dlys.flatten()):
            self[i].set_params({'dly_x': tau})
            self[i].set_params({'dly_y': tau + self.dly_xx_to_yy.flatten()[i]})

    def update(self):
        # self.update_gains()
        # self.update_delays()
        aipy.pol.AntennaArray.update(self)

    def get_params(self, ant_prms={'*': '*'}):
        try:
            prms = aipy.pol.AntennaArray.get_params(self, ant_prms)
        except(IndexError):
            return {}
        for k in ant_prms:
            if k == 'aa':
                if 'aa' not in prms:
                    prms['aa'] = {}
                for val in ant_prms[k]:
                    if val == 'tau_ns':
                        prms['aa']['tau_ns'] = self.tau_ns
                    elif val == 'tau_ew':
                        prms['aa']['tau_ew'] = self.tau_ew
                    elif val == 'gain':
                        prms['aa']['gain'] = self.gain
            else:
                try:
                    top_pos = np.dot(self._eq2zen, self[int(k)].pos)
                # XXX should multiply this by len_ns to match set_params.
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
                ant.pos = np.dot(np.linalg.inv(self._eq2zen), top_pos) / aipy.const.len_ns * cm_p_m
            changed |= ant_changed
        try:
            self.tau_ns, changed = prms['aa']['tau_ns'], 1
        except(KeyError):
            pass
        try:
            self.tau_ew, changed = prms['aa']['tau_ew'], 1
        except(KeyError):
            pass
        try:
            self.gain, changed = prms['aa']['gain'], 1
        except(KeyError):
            pass
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


def antpos_enu(locations_file,array_epoch_jd,station_type='herahex'):
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



prms = {#'loc': (cofa_lat, cofa_lon),
        #'antpos_ideal': antpos_enu(locations_file,array_epoch_jd),
        'amps': dict(zip(range(128), np.ones(128))),
        'amp_coeffs': np.array([1] * 128),
        'bp_r': np.array([[1.]] * 128),
        'bp_i': np.array([[0., 0., 0.]] * 128),
        'twist': np.array([0] * 128),
        'beam': aipy.fit.Beam2DGaussian,
        'bm_prms': {'bm_xwidth':8*np.pi/180,'bm_ywidth':8*np.pi/180}#no idea what the units are here
        # or even if Beam2DGaussian will work as advertised
}


def get_aa(freqs,locations_file,array_epoch_jd=array_epoch_jd):
    '''Return the HERA AntennaArray.'''
    cofa_lat, cofa_lon, cofa_alt = cofa_latlonalt(locations_file)
    location = (cofa_lat, cofa_lon)
    antennas = []
    #load the positions from the file, filter by epoch
    antpos = antpos_enu(locations_file,array_epoch_jd,station_type='herahex')
    print("found {0} antennas".format(len(antpos)))
    nants = np.max(antpos.keys())+1
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
