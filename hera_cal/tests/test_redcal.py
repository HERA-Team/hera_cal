import hera_cal.redcal as om
import numpy as np
import unittest

np.random.seed(0)

def build_reds_linear(nants, sep=14.7):
    antpos = {i: np.array([sep*i, 0, 0]) for i in range(nants)}
    return om.get_reds(antpos), antpos

def build_reds_hex(hexNum, sep=14.7):
    antpos, i = {}, 0
    for row in range(hexNum-1,-(hexNum),-1):
        for col in range(2*hexNum-abs(row)-1):
            xPos = ((-(2*hexNum-abs(row))+2)/2.0 + col)*sep;
            yPos = row*sep*3**.5/2;
            antpos[i] = np.array([xPos, yPos, 0])
            i += 1
    return om.get_reds(antpos), antpos

class TestMethods(unittest.TestCase):
    def test_noise(self):
        n = om.noise((1024,1024))
        self.assertEqual(n.shape, (1024,1024))
        self.assertAlmostEqual(np.var(n), 1, 2)
    def test_sim_red_data(self):
        reds,antpos = build_reds_linear(10)
        pols = ['xx']
        gains, true_vis, data = om.sim_red_data(reds, pols, stokes_v_indep=True)
        self.assertEqual(len(gains), 10)
        self.assertEqual(len(data), 45)
        for bls in reds:
            bl0 = bls[0]
            ai,aj = bl0
            ans0 = data[bl0+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            for bl in bls[1:]:
                ai,aj = bl
                ans = data[bl+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                np.testing.assert_almost_equal(ans0, ans, 7)
        pols = ['xx','yy','xy','yx']
        gains, true_vis, data = om.sim_red_data(reds, pols, stokes_v_indep=True)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4*(45))
        for bls in reds:
            bl0 = bls[0]
            ai,aj = bl0
            ans0xx = data[bl0+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            ans0xy = data[bl0+('xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
            ans0yx = data[bl0+('yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
            ans0yy = data[bl0+('yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
            for bl in bls[1:]:
                ai,aj = bl
                ans_xx = data[bl+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                ans_xy = data[bl+('xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
                ans_yx = data[bl+('yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
                ans_yy = data[bl+('yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)
        gains, true_vis, data = om.sim_red_data(reds, pols, stokes_v_indep=False)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4*(45))
        for bls in reds:
            bl0 = bls[0]
            ai,aj = bl0
            ans0xx = data[bl0+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            ans0xy = data[bl0+('xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
            ans0yx = data[bl0+('yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
            ans0yy = data[bl0+('yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
            np.testing.assert_almost_equal(ans0xy, ans0yx, 7)
            for bl in bls[1:]:
                ai,aj = bl
                ans_xx = data[bl+('xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                ans_xy = data[bl+('xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
                ans_yx = data[bl+('yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
                ans_yy = data[bl+('yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)

class TestRedundantCalibrator(unittest.TestCase):
    def test_build_eq(self):
        reds, antpos = build_reds_linear(3)
        bls = reduce(lambda x,y: x+y, reds)
        info = om.RedundantCalibrator(reds, antpos)
        eqs = info.build_eqs(bls, ['xx'])
        self.assertEqual(len(eqs), 3)
        self.assertEqual(eqs['g1x * g0x_ * u0xx'], (1,0,'xx'))
        self.assertEqual(eqs['g2x * g1x_ * u0xx'], (2,1,'xx'))
        self.assertEqual(eqs['g2x * g0x_ * u1xx'], (2,0,'xx'))
        pols = ['xx','yy','xy','yx']
        info = om.RedundantCalibrator(reds, antpos)
        eqs = info.build_eqs(bls, pols)
        self.assertEqual(len(eqs), 3*4)
        self.assertEqual(eqs['g1x * g0y_ * u0xy'], (1,0,'xy'))
        self.assertEqual(eqs['g2x * g1y_ * u0xy'], (2,1,'xy'))
        self.assertEqual(eqs['g2x * g0y_ * u1xy'], (2,0,'xy'))
        self.assertEqual(eqs['g1y * g0x_ * u0yx'], (1,0,'yx'))
        self.assertEqual(eqs['g2y * g1x_ * u0yx'], (2,1,'yx'))
        self.assertEqual(eqs['g2y * g0x_ * u1yx'], (2,0,'yx'))
        info = om.RedundantCalibrator(reds, antpos,stokes_v_indep=False)
        eqs = info.build_eqs(bls, pols)
        self.assertEqual(len(eqs), 3*4)
        self.assertEqual(eqs['g1x * g0y_ * u0xy'], (1,0,'xy'))
        self.assertEqual(eqs['g2x * g1y_ * u0xy'], (2,1,'xy'))
        self.assertEqual(eqs['g2x * g0y_ * u1xy'], (2,0,'xy'))
        self.assertEqual(eqs['g1y * g0x_ * u0xy'], (1,0,'yx'))
        self.assertEqual(eqs['g2y * g1x_ * u0xy'], (2,1,'yx'))
        self.assertEqual(eqs['g2y * g0x_ * u1xy'], (2,0,'yx'))
    def test_solver(self):
        reds, antpos = build_reds_linear(3)
        info = om.RedundantCalibrator(reds, antpos)
        gains, true_vis, d = om.sim_red_data(reds, ['xx'])
        w = {}
        w = dict([(k,1.) for k in d.keys()])
        def solver(data, wgts, sparse, **kwargs):
            np.testing.assert_equal(data['g1x * g0x_ * u0xx'], d[1,0,'xx'])
            np.testing.assert_equal(data['g2x * g1x_ * u0xx'], d[2,1,'xx'])
            np.testing.assert_equal(data['g2x * g0x_ * u1xx'], d[2,0,'xx'])
            if len(wgts) == 0: return
            np.testing.assert_equal(wgts['g1x * g0x_ * u0xx'], w[1,0,'xx'])
            np.testing.assert_equal(wgts['g2x * g1x_ * u0xx'], w[2,1,'xx'])
            np.testing.assert_equal(wgts['g2x * g0x_ * u1xx'], w[2,0,'xx'])
            return
        info._solver(solver, d)
        info._solver(solver, d, w)
    def test_logcal(self):
        NANTS = 18
        reds, antpos = build_reds_linear(NANTS)
        info = om.RedundantCalibrator(reds, antpos)
        gains, true_vis, d = om.sim_red_data(reds, ['xx'], gain_scatter=.55)
        w = dict([(k,1.) for k in d.keys()])
        sol = info.logcal(d)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i,'x')].shape, (10,10))
        for bls in reds:
            ubl = sol[bls[0]+('xx',)]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl+('xx',)]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
    def test_lincal(self):
        NANTS = 18
        reds, antpos = build_reds_linear(NANTS)
        info = om.RedundantCalibrator(reds, antpos)
        #gains, true_vis, d = om.sim_red_data(reds, ['xx'], gain_scatter=.01) # XXX causes svd error
        gains, true_vis, d = om.sim_red_data(reds, ['xx'], gain_scatter=.0099999)
        w = dict([(k,1.) for k in d.keys()])
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(info.compute_ubls(d,sol0))
        #sol0 = info.logcal(d)
        #for k in sol0: sol0[k] += .01*capo.oqe.noise(sol0[k].shape)
        meta, sol = info.lincal(d, sol0)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i,'x')].shape, (10,10))
        for bls in reds:
            ubl = sol[bls[0]+('xx',)]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl+('xx',)]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)

    def test_svd_convergence(self):
        for hexnum in (2,3,4):
            for dtype in (np.complex64, np.complex128):
                reds, antpos = build_reds_hex(hexnum)
                rc = om.RedundantCalibrator(reds, antpos)
                gains, _, d = om.sim_red_data(reds, ['xx'], gain_scatter=.01)
                d = {k:dk.astype(dtype) for k,dk in d.items()}
                w = {k:1. for k in d.keys()}
                gains = {k:gk.astype(dtype) for k,gk in gains.items()}
                sol0 = {k:np.ones_like(gk) for k,gk in gains.items()}
                sol0.update(rc.compute_ubls(d,sol0))
                meta, sol = rc.lincal(d, sol0) # should not raise 'np.linalg.linalg.LinAlgError: SVD did not converge'
    
    def test_lincal_hex_end_to_end_with_remove_degen(self):
        reds, antpos = build_reds_hex(3)
        rc = om.RedundantCalibrator(reds, antpos)
        gains, true_vis, d = om.sim_red_data(reds, ['xx'], gain_scatter=.01)
        w = dict([(k,1.) for k in d.keys()])
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(rc.compute_ubls(d,sol0))
        meta, sol = rc.lincal(d, sol0)

        np.testing.assert_array_less(meta['iter'], 50*np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'],0,10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i,'x')].shape, (10,10))
        for bls in reds:
            ubl = sol[bls[0]+('xx',)]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl+('xx',)]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        
        sol_rd = rc.remove_degen(sol)
        ants = [key for key in sol_rd.keys() if len(key)==2]
        gainSols = np.array([sol_rd[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]+('xx',)]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl+('xx',)]
                mdl = sol_rd[(bl[0],'x')] * sol_rd[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(sol, degen_sol=gains)
        for key,val in sol_rd.items():
            if len(key)==2: np.testing.assert_almost_equal(val,gains[key],10)
            if len(key)==3: np.testing.assert_almost_equal(val,true_vis[key],10)

if __name__ == '__main__':
    unittest.main()
