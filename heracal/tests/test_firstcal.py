'''Tests for firstcal.py'''
import nose.tools as nt
import os
import numpy as np
import aipy
import optparse
import sys
from pyuvdata import UVCal, UVData
import heracal.firstcal as firstcal
from heracal.omni import compute_reds
from heracal.data import DATA_PATH
from heracal.calibrations import CAL_PATH


class Test_FirstCal(object):

    def setUp(self):
        antpos = np.array([[14.60000038, -25.28794098,   1.],
                           [21.89999962, -12.64397049,   1.],
                           [14.60000038,  25.28794098,   1.],
                           [-21.89999962, -12.64397049,   1.],
                           [-14.60000038,   0.,   1.],
                           [21.89999962,  12.64397049,   1.],
                           [29.20000076,   0.,   1.],
                           [-14.60000038, -25.28794098,   1.],
                           [0.,  25.28794098,   1.],
                           [0., -25.28794098,   1.],
                           [0.,   0.,   1.],
                           [-7.30000019, -12.64397049,   1.],
                           [-7.30000019,  12.64397049,   1.],
                           [-21.89999962,  12.64397049,   1.],
                           [-29.20000076,   0.,   1.],
                           [14.60000038,   0.,   1.],
                           [-14.60000038,  25.28794098,   1.],
                           [7.30000019, -12.64397049,   1.]])
        reds = [[(0, 8), (9, 16)],
                [(13, 15), (14, 17), (3, 0), (4, 1), (16, 5), (12, 6)],
                [(3, 17), (4, 15), (7, 0), (11, 1),
                 (16, 2), (12, 5), (10, 6), (14, 10)],
                [(3, 6), (14, 5)],
                [(0, 9), (1, 17), (2, 8), (4, 14), (6, 15), (8, 16), (12, 13), (11, 3),
                 (10, 4), (9, 7), (15, 10), (17, 11)],
                [(3, 8), (11, 2), (9, 5)],
                [(3, 9), (4, 17), (12, 15), (11, 0),
                 (10, 1), (8, 5), (13, 10), (14, 11)],
                [(0, 13), (1, 16)],
                [(0, 4), (1, 12), (6, 8), (9, 14), (15, 16), (17, 13)],
                [(0, 5), (3, 16), (7, 12), (17, 2), (11, 8)],
                [(0, 10), (7, 14), (10, 16), (11, 13),
                 (6, 2), (9, 4), (15, 8), (17, 12)],
                [(1, 9), (2, 12), (5, 10), (6, 17), (8, 13),
                 (12, 14), (10, 3), (17, 7), (15, 11)],
                [(2, 3), (5, 7)],
                [(16, 17), (12, 0), (8, 1), (13, 9)],
                [(0, 17), (1, 15), (3, 14), (4, 13), (9, 11), (10, 12), (12, 16), (5, 2), (7, 3),
                 (11, 4), (6, 5), (17, 10)],
                [(3, 15), (4, 5), (7, 1), (13, 2), (11, 6)],
                [(5, 15), (8, 12), (10, 11), (13, 14), (15, 17), (1, 0), (6, 1), (4, 3), (12, 4),
                 (11, 7), (17, 9), (16, 13)],
                [(0, 15), (1, 5), (3, 13), (4, 16), (9, 10),
                 (11, 12), (15, 2), (7, 4), (10, 8)],
                [(0, 6), (3, 12), (4, 8), (7, 10),
                 (9, 15), (14, 16), (10, 2), (17, 5)],
                [(8, 17), (2, 1), (13, 7), (12, 9), (16, 11)],
                [(0, 2), (7, 16), (9, 8)], [(4, 6), (14, 15), (3, 1), (13, 5)],
                [(0, 14), (1, 13), (6, 16)],
                [(2, 14), (6, 7), (5, 3)],
                [(2, 9), (8, 7)],
                [(2, 4), (5, 11), (6, 9), (8, 14), (15, 7)], [(1, 14), (6, 13)]]
        self.freqs = np.linspace(.1, .2, 64)
        self.times = np.arange(1)
        ants = np.arange(len(antpos))
        reds = compute_reds(len(ants), 'x', antpos, tol=0.1)

        self.info = firstcal.FirstCalRedundantInfo(len(antpos))
        self.info.init_from_reds(reds, antpos)

        # Simulate unique "true" visibilities
        np.random.seed(21)
        self.vis_true = {'xx': {}}
        i = 0
        for rg in reds:
            self.vis_true['xx'][rg[0]] = np.array(1.0 * np.random.randn(len(self.times), len(
                self.freqs)) + 1.0j * np.random.randn(len(self.times), len(self.freqs)), dtype=np.complex64)

        # Generate and apply firstcal gains
        self.fcgains = {}
        self.delays = {}
        for i in ants:
            if i == len(ants) - 1:
                self.delays[i] = -1 * \
                    np.sum([delay for delay in self.delays.values()])
            else:
                self.delays[i] = np.random.randn() * 30
            fcspectrum = np.exp(2.0j * np.pi * self.delays[i] * self.freqs)
            self.fcgains[i] = np.array(
                [fcspectrum for t in self.times], dtype=np.complex64)
            self.delays[i] /= 1e9

        # Generate fake data
        bl2ublkey = {bl: rg[0] for rg in reds for bl in rg}
        self.data = {}
        self.wgts = {}
        for rg in reds:
            for (i, j) in rg:
                self.data[(i.val, j.val)] = {}
                self.wgts[(i.val, j.val)] = {}
                for pol in ['xx']:
                    self.data[(i.val, j.val)][pol] = np.array(np.conj(self.fcgains[
                        i.val]) * self.fcgains[j.val] * self.vis_true['xx'][rg[0]], dtype=np.complex64)
                    self.wgts[(i.val, j.val)][pol] = np.ones_like(
                        self.data[(i.val, j.val)][pol], dtype=np.bool)

    def test_data_to_delays(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        w = fcal.data_to_delays()
        for (i, k), (l, m) in w.keys():
            nt.assert_almost_equal(w[(i, k), (l, m)][0], self.delays[
                                   i] - self.delays[k] - self.delays[l] + self.delays[m], places=16)

    def test_data_to_delays_average(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        w = fcal.data_to_delays(average=True)
        for (i, k), (l, m) in w.keys():
            nt.assert_almost_equal(w[(i, k), (l, m)][0], self.delays[
                                   i] - self.delays[k] - self.delays[l] + self.delays[m], places=16)

    def test_get_N(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        # the only requirement on N is it's shape.
        nt.assert_equal(fcal.get_N(len(fcal.info.bl_pairs)).shape,
                        (len(fcal.info.bl_pairs), len(fcal.info.bl_pairs)))

    def test_get_M(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        nt.assert_equal(fcal.get_M().shape, (len(
            self.info.bl_pairs), len(self.times)))
        _M = np.array([1 * (self.delays[i] * np.ones(len(self.times)) - self.delays[k] * np.ones(len(self.times)) - self.delays[l]
                            * np.ones(len(self.times)) + self.delays[m] * np.ones(len(self.times))) for (i, k), (l, m) in self.info.bl_pairs])
        nt.assert_equal(np.testing.assert_almost_equal(
            _M, fcal.get_M(), decimal=16), None)

    def test_run(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        sols = fcal.run()
        solved_delays = []
        for pair in fcal.info.bl_pairs:
            ant_indexes = fcal.info.blpair2antind(pair)
            dlys = fcal.xhat[ant_indexes]
            solved_delays.append(dlys[0] - dlys[1] - dlys[2] + dlys[3])
        solved_delays = np.array(solved_delays).flatten()
        nt.assert_equal(np.testing.assert_almost_equal(
            fcal.M.flatten(), solved_delays, decimal=16), None)

    def test_run_average(self):
        fcal = firstcal.FirstCal(self.data, self.wgts, self.freqs, self.info)
        sols = fcal.run(average=True)
        solved_delays = []
        for pair in fcal.info.bl_pairs:
            ant_indexes = fcal.info.blpair2antind(pair)
            dlys = fcal.xhat[ant_indexes]
            solved_delays.append(dlys[0] - dlys[1] - dlys[2] + dlys[3])
        solved_delays = np.array(solved_delays).flatten()
        nt.assert_equal(np.testing.assert_almost_equal(
            fcal.M.flatten(), solved_delays, decimal=16), None)

    def test_flatten_reds(self):
        reds = [[(0, 1), (1, 2)], [(2, 3), (3, 4)]]
        freds = firstcal.flatten_reds(reds)
        nt.assert_equal(freds, [(0, 1), (1, 2), (2, 3), (3, 4)])
        return

    def test_UVData_to_dict(self):
        str2pol = {'xx': -5, 'yy': -6, 'xy': -7, 'yy': -8}
        filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
        uvd = UVData()
        uvd.read_miriad(filename)
        if uvd.phase_type != 'drift':
            uvd.unphase_to_drift()

        d, f = firstcal.UVData_to_dict([uvd, uvd])
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, 0, :, uvpol], f[i, j][pol].shape))

        d, f = firstcal.UVData_to_dict([filename, filename])
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, 0, :, uvpol], f[i, j][pol].shape))

    def test_process_ex_ants(self):
        ex_ants = ''
        xants = firstcal.process_ex_ants(ex_ants)
        nt.assert_equal(xants, [])

        ex_ants = '0,1,2'
        xants = firstcal.process_ex_ants(ex_ants)
        nt.assert_equal(xants, [0, 1, 2])

        ex_ants = '0,obvious_error'
        nt.assert_raises(AssertionError, firstcal.process_ex_ants, ex_ants)
        return

    def test_process_ubls(self):
        ubls = ''
        ubaselines = firstcal.process_ubls(ubls)
        nt.assert_equal(ubaselines, [])

        ubls = '0_1,1_2,2_3'
        ubaselines = firstcal.process_ubls(ubls)
        nt.assert_equal(ubaselines, [(0, 1), (1, 2), (2, 3)])

        ubls = '0_1,1,2'
        nt.assert_raises(AssertionError, firstcal.process_ubls, ubls)
        return


class TestFCRedInfo(object):

    def test_init_from_reds(self):
        antpos = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        reds = compute_reds(4, 'x', antpos)
        blpairs = [((0, 1), (1, 2)), ((0, 1), (2, 3)),
                   ((1, 2), (2, 3)), ((0, 2), (1, 3))]
        A = np.array([[1, -2, 1, 0], [1, -1, -1, 1],
                      [0, 1, -2, 1], [1, -1, -1, 1]])
        i = firstcal.FirstCalRedundantInfo(4)
        i.init_from_reds(reds, antpos)
        nt.assert_true(np.all(i.subsetant == np.arange(4, dtype=np.int32)))
        nt.assert_equal(i.reds, reds)
        nt.assert_equal(i.bl_pairs, blpairs)
        nt.assert_true(i.blperant[0] == 2)
        nt.assert_true(i.blperant[1] == 3)
        nt.assert_true(i.blperant[2] == 3)
        nt.assert_true(i.blperant[3] == 2)
        nt.assert_true(np.all(i.A == A))

    def test_bl_index(self):
        antpos = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        reds = compute_reds(4, 'x', antpos)
        i = firstcal.FirstCalRedundantInfo(4)
        i.init_from_reds(reds, antpos)
        bls_order = [bl for ublgp in reds for bl in ublgp]
        for k, b in enumerate(bls_order):
            nt.assert_equal(i.bl_index(b), k)

    def test_blpair_index(self):
        antpos = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        reds = compute_reds(4, 'x', antpos)
        blpairs = [((0, 1), (1, 2)), ((0, 1), (2, 3)),
                   ((1, 2), (2, 3)), ((0, 2), (1, 3))]
        i = firstcal.FirstCalRedundantInfo(4)
        i.init_from_reds(reds, antpos)
        for k, bp in enumerate(blpairs):
            nt.assert_equal(i.blpair_index(bp), k)

    def test_blpair2antindex(self):
        antpos = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        reds = compute_reds(4, 'x', antpos)
        blpairs = [((0, 1), (1, 2)), ((0, 1), (2, 3)),
                   ((1, 2), (2, 3)), ((0, 2), (1, 3))]
        i = firstcal.FirstCalRedundantInfo(4)
        i.init_from_reds(reds, antpos)
        for bp in blpairs:
            nt.assert_true(np.all(i.blpair2antind(bp) == map(
                i.ant_index, np.array(bp).flatten())))


class Test_firstcal_run(object):
    global calfile
    global xx_vis
    calfile = "hsa7458_v001"
    xx_vis = "zen.2457698.40355.xx.HH.uvcAA"

    # add directory with calfile
    if CAL_PATH not in sys.path:
        sys.path.append(CAL_PATH)

    def test_empty_fileset(self):
        o = firstcal.firstcal_option_parser()
        cmd = "-C {0} -p xx".format(calfile)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        nt.assert_raises(AssertionError, firstcal.firstcal_run,
                         files, opts, history)
        return

    def test_single_file_execution(self):
        objective_file = os.path.join(
            DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.first.calfits')
        xx_vis4real = os.path.join(DATA_PATH, xx_vis)
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = firstcal.firstcal_option_parser()
        cmd = "-C {0} -p xx --ex_ants=81 {1}".format(calfile, xx_vis4real)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        firstcal.firstcal_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        os.remove(objective_file)
        return
