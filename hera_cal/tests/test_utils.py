import nose.tools as nt
#from hera_cal.utils import get_HERA_aa
import numpy as np,sys
from hera_cal.calibrations import CAL_PATH
freqs = np.array([0.15])

# def count_ants(aa):
#     nants = 0
#     for antpos in aa.antpos_ideal:
#         if antpos[0]!=-1 and antpos[1] != -1 and antpos[2] != -1:
#             nants += 1
#     return nants
# class Test_utils(object):
#     # add directory with calfile
#     if CAL_PATH not in sys.path:
#         sys.path.append(CAL_PATH)
#     global calfile
#     calfile = "hera_test_calfile"
#     def test_get_HERA_aa_default_cal(self):
#         aa = get_HERA_aa(freqs)
#         nt.assert_equal(len(aa),113)
#     def test_get_HERA_aa_mycal(self):
#         aa = get_HERA_aa(freqs,calfile=calfile)
#         nt.assert_equal(len(aa),128)
#         nt.assert_almost_equal(aa.lat,-30.7215261207*np.pi/180,places=6)
#         nt.assert_almost_equal(aa.lon,21.4283038269*np.pi/180,places=6)
#     def test_get_HERA_aa_cofa(self):
#         aa = get_HERA_aa(freqs)
#         #check the position is correct to ~6m
#         nt.assert_almost_equal(aa.lat,-30.7215261207*np.pi/180,places=6)
#         nt.assert_almost_equal(aa.lon,21.4283038269*np.pi/180,places=6)
#     def test_mc_nants(self):
#         aa = get_HERA_aa(freqs,array_epoch_jd=2457458)
#         nt.assert_equal(count_ants(aa),20)
#         aa = get_HERA_aa(freqs,array_epoch_jd=2457962)
#         nt.assert_equal(count_ants(aa),34)
#     def test_aa_get_params(self):
#         #this is a super rough check that we've round-tripped the topo coords
#         #   through equatorial nanoseconds without massively missing on units
#         # just tests that we get to within 10%.
#         aa = get_HERA_aa(freqs,array_epoch_jd=2457458)
#         prms = aa.get_params({'9':['x','top_x','top_y','top_z']}) #get the position of antenna 8
#         nt.assert_equal(prms['9']['x'],-143.64149291430934)
#         radius_of_earth = 6371e3
#         #use positions taken from hera_ant_locs_05_16_2017.csv
#         topo_x_rough = (21.4272063669-21.4283038269)*np.pi/180
#         topo_x_rough *= np.cos(-30.7215261207*np.pi/180)*radius_of_earth
#         topo_y_rough = (-30.7222964631 - -30.7215261207)*np.pi/180
#         topo_y_rough *= radius_of_earth
#         x_err = np.abs((topo_x_rough - prms['9']['top_x'])/prms['9']['top_x'])
#         y_err = np.abs((topo_y_rough - prms['9']['top_y'])/prms['9']['top_y'])
#         nt.assert_true(x_err<0.1)
#         nt.assert_true(y_err<0.1)
