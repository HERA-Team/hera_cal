import hera_cal.redcal as om
import numpy as np
import unittest

np.random.seed(0)

def build_linear_array(nants, sep=14.7):
    antpos = {i: np.array([sep*i, 0, 0]) for i in range(nants)}
    return antpos

def build_hex_array(hexNum, sep=14.7):
    antpos, i = {}, 0
    for row in range(hexNum-1,-(hexNum),-1):
        for col in range(2*hexNum-abs(row)-1):
            xPos = ((-(2*hexNum-abs(row))+2)/2.0 + col)*sep;
            yPos = row*sep*3**.5/2;
            antpos[i] = np.array([xPos, yPos, 0])
            i += 1
    return antpos

class TestMethods(unittest.TestCase):

    def test_noise(self):
        n = om.noise((1024,1024))
        self.assertEqual(n.shape, (1024,1024))
        self.assertAlmostEqual(np.var(n), 1, 2)
    
    def test_sim_red_data(self):
        antpos = build_linear_array(10)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 10)
        self.assertEqual(len(data), 45)
        for bls in reds:
            bl0 = bls[0]
            ai,aj,pol = bl0
            ans0 = data[bl0] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            for bl in bls[1:]:
                ai,aj,pol = bl
                ans = data[bl] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                np.testing.assert_almost_equal(ans0, ans, 7)
        
        reds = om.get_reds(antpos, pols=['xx','yy','xy','yx'], pol_mode='4pol')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4*(45))
        for bls in reds:
            bl0 = bls[0]
            ai,aj,pol = bl0
            ans0xx = data[(ai,aj,'xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            ans0xy = data[(ai,aj,'xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
            ans0yx = data[(ai,aj,'yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
            ans0yy = data[(ai,aj,'yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
            for bl in bls[1:]:
                ai,aj,pol = bl
                ans_xx = data[(ai,aj,'xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                ans_xy = data[(ai,aj,'xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
                ans_yx = data[(ai,aj,'yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
                ans_yy = data[(ai,aj,'yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)

        reds = om.get_reds(antpos, pols=['xx','yy','xy','yx'], pol_mode='4pol_minV')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4*(45))
        for bls in reds:
            bl0 = bls[0]
            ai,aj,pol = bl0
            ans0xx = data[(ai,aj,'xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
            ans0xy = data[(ai,aj,'xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
            ans0yx = data[(ai,aj,'yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
            ans0yy = data[(ai,aj,'yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
            np.testing.assert_almost_equal(ans0xy, ans0yx, 7)
            for bl in bls[1:]:
                ai,aj,pol = bl
                ans_xx = data[(ai,aj,'xx',)] / (gains[(ai,'x')] * gains[(aj,'x')].conj())
                ans_xy = data[(ai,aj,'xy',)] / (gains[(ai,'x')] * gains[(aj,'y')].conj())
                ans_yx = data[(ai,aj,'yx',)] / (gains[(ai,'y')] * gains[(aj,'x')].conj())
                ans_yy = data[(ai,aj,'yy',)] / (gains[(ai,'y')] * gains[(aj,'y')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)

    def test_check_polLists_minV(self):
        polLists = [['xy']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['xx','xy']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['xx','xy','yx']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['xy','yx'],['xx'],['yy'],['xx'],['yx','xy'],['yy']]
        self.assertTrue(om.check_polLists_minV(polLists))

    def test_parse_pol_mode(self):
        reds = [[(0,1,'xx')]]
        self.assertEqual(om.parse_pol_mode(reds), '1pol')
        reds = [[(0,1,'xx')], [(0,1,'yy')]]
        self.assertEqual(om.parse_pol_mode(reds), '2pol')
        reds = [[(0,1,'xx')],[(0,1,'xy')],[(0,1,'yx')],[(0,1,'yy')]]
        self.assertEqual(om.parse_pol_mode(reds), '4pol')
        reds = [[(0,1,'xx')],[(0,1,'xy'), (0,1,'yx')],[(0,1,'yy')]]
        self.assertEqual(om.parse_pol_mode(reds), '4pol_minV')

        reds = [[(0,1,'xx')],[(0,1,'xy'), (0,1,'yx')],[(0,1,'zz')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0,1,'xx')],[(0,1,'xy')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0,1,'xy')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0,1,'xx')],[(0,1,'xy'), (0,1,'yy')],[(0,1,'yx')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')


class TestRedundantCalibrator(unittest.TestCase):
    
    def test_build_eq(self):
        antpos = build_linear_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3)
        self.assertEqual(eqs['g1x * g0x_ * u0xx'], (1,0,'xx'))
        self.assertEqual(eqs['g2x * g1x_ * u0xx'], (2,1,'xx'))
        self.assertEqual(eqs['g2x * g0x_ * u1xx'], (2,0,'xx'))
        
        reds = om.get_reds(antpos, pols=['xx','yy','xy','yx'], pol_mode='4pol')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3*4)
        self.assertEqual(eqs['g1x * g0y_ * u4xy'], (1,0,'xy'))
        self.assertEqual(eqs['g2x * g1y_ * u4xy'], (2,1,'xy'))
        self.assertEqual(eqs['g2x * g0y_ * u5xy'], (2,0,'xy'))
        self.assertEqual(eqs['g1y * g0x_ * u6yx'], (1,0,'yx'))
        self.assertEqual(eqs['g2y * g1x_ * u6yx'], (2,1,'yx'))
        self.assertEqual(eqs['g2y * g0x_ * u7yx'], (2,0,'yx'))


        reds = om.get_reds(antpos, pols=['xx','yy','xy','yx'], pol_mode='4pol_minV')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3*4)
        self.assertEqual(eqs['g1x * g0y_ * u4xy'], (1,0,'xy'))
        self.assertEqual(eqs['g2x * g1y_ * u4xy'], (2,1,'xy'))
        self.assertEqual(eqs['g2x * g0y_ * u5xy'], (2,0,'xy'))
        self.assertEqual(eqs['g1y * g0x_ * u4xy'], (1,0,'yx'))
        self.assertEqual(eqs['g2y * g1x_ * u4xy'], (2,1,'yx'))
        self.assertEqual(eqs['g2y * g0x_ * u5xy'], (2,0,'yx'))


    def test_solver(self):
        antpos = build_linear_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds)
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
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.05)
        w = dict([(k,1.) for k in d.keys()])
        sol = info.logcal(d)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i,'x')].shape, (10,10))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
    

    def test_lincal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.0099999)
        w = dict([(k,1.) for k in d.keys()])
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(info.compute_ubls(d,sol0))
        #sol0 = info.logcal(d)
        #for k in sol0: sol0[k] += .01*capo.oqe.noise(sol0[k].shape)
        meta, sol = info.lincal(d, sol0)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i,'x')].shape, (10,10))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
    

    def test_lincal_hex_end_to_end_1pol_with_remove_degen(self):
        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.1)
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
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0],'x')] * sol[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)
        ants = [key for key in sol_rd.keys() if len(key)==2]
        gainSols = np.array([sol_rd[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            self.assertEqual(ubl.shape, (10,10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol_rd[(bl[0],'x')] * sol_rd[(bl[1],'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        for key,val in sol_rd.items():
            if len(key)==2: np.testing.assert_almost_equal(val,gains[key],10)
            if len(key)==3: np.testing.assert_almost_equal(val,true_vis[key],10)

        rc.pol_mode = 'unrecognized_pol_mode'
        with self.assertRaises(ValueError):
            sol_rd = rc.remove_degen(antpos, sol)


    def test_lincal_hex_end_to_end_4pol_with_remove_degen(self):
        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['xx','xy','yx','yy'], pol_mode='4pol')
        rc = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.01, shape=(3,4))
        w = dict([(k,1.) for k in d.keys()])
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(rc.compute_ubls(d,sol0))
        meta, sol = rc.lincal(d, sol0)

        np.testing.assert_array_less(meta['iter'], 50*np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'],0,10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i,'x')].shape, (3,4))
            self.assertEqual(sol[(i,'y')].shape, (3,4))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol[(bl[0],bl[2][0])] * sol[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        
        sol_rd = rc.remove_degen(antpos, sol)
        
        ants = [key for key in sol_rd.keys() if len(key)==2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key)==3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='x']), axis=0), 0, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='y']), axis=0), 0, 10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0],bl[2][0])] * sol_rd[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        
        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='x']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='y']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='x']), axis=0), 
            np.mean(np.angle(degenGains[gainPols=='x']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='y']), axis=0), 
            np.mean(np.angle(degenGains[gainPols=='y']), axis=0), 10)

        for key,val in sol_rd.items():
            if len(key)==2: np.testing.assert_almost_equal(val,gains[key],10)
            if len(key)==3: np.testing.assert_almost_equal(val,true_vis[key],10)


    def test_lincal_hex_end_to_end_4pol_minV_with_remove_degen(self):

        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['xx','xy','yx','yy'], pol_mode='4pol_minV')

        rc = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.01, shape=(3,4))
        w = dict([(k,1.) for k in d.keys()])
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(rc.compute_ubls(d,sol0))
        meta, sol = rc.lincal(d, sol0)

        np.testing.assert_array_less(meta['iter'], 50*np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'],0,10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i,'x')].shape, (3,4))
            self.assertEqual(sol[(i,'y')].shape, (3,4))
        for bls in reds:
            ubl = sol[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol[(bl[0],bl[2][0])] * sol[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        
        sol_rd = rc.remove_degen(antpos, sol)
        
        ants = [key for key in sol_rd.keys() if len(key)==2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key)==3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        visPolsStr = np.array([bl[2] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0],bl[2][0])] * sol_rd[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)


        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0],bl[2][0])] * sol_rd[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        
        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='x']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='y']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 
            np.mean(np.angle(degenGains), axis=0), 10)
        
        visSols = np.array([sol_rd[bl] for bl in bl_pairs])
        degenVis = np.array([true_vis[bl] for bl in bl_pairs])
        np.testing.assert_almost_equal(np.mean(np.abs(visSols[visPolsStr=='xx']), axis=0), 
            np.mean(np.abs(degenVis[visPolsStr=='xx']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.abs(visSols[visPolsStr=='yy']), axis=0), 
            np.mean(np.abs(degenVis[visPolsStr=='yy']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.abs(visSols[visPolsStr=='xy']), axis=0), 
            np.mean(np.abs(degenVis[visPolsStr=='xy']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(visSols), axis=0), 
            np.mean(np.angle(degenVis), axis=0), 10)

        for key,val in sol_rd.items():
            if len(key)==2: np.testing.assert_almost_equal(val,gains[key],10)
            if len(key)==3: np.testing.assert_almost_equal(val,true_vis[key],10)

    def test_lincal_hex_end_to_end_2pol_with_remove_degen(self):
        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['xx','yy'], pol_mode='2pol')
        rc = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.01, shape=(3,4))
        sol0 = dict([(k,np.ones_like(v)) for k,v in gains.items()])
        sol0.update(rc.compute_ubls(d,sol0))
        meta, sol = rc.lincal(d, sol0)

        np.testing.assert_array_less(meta['iter'], 50*np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'],0,10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i,'x')].shape, (3,4))
            self.assertEqual(sol[(i,'y')].shape, (3,4))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol[(bl[0],bl[2][0])] * sol[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        
        sol_rd = rc.remove_degen(antpos, sol)

        ants = [key for key in sol_rd.keys() if len(key)==2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key)==3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 1, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='x']), axis=0), 0, 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='y']), axis=0), 0, 10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                self.assertEqual(ubl.shape, (3,4))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0],bl[2][0])] * sol_rd[(bl[1],bl[2][1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl*mdl.conj()), 0, 10)
        

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        
        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='x']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='x']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.abs(gainSols[gainPols=='y']), axis=0), 
            np.mean(np.abs(degenGains[gainPols=='y']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='x']), axis=0), 
            np.mean(np.angle(degenGains[gainPols=='x']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='y']), axis=0), 
            np.mean(np.angle(degenGains[gainPols=='y']), axis=0), 10)

        for key,val in sol_rd.items():
            if len(key)==2: np.testing.assert_almost_equal(val,gains[key],10)
            if len(key)==3: np.testing.assert_almost_equal(val,true_vis[key],10)


if __name__ == '__main__':
    unittest.main()

#TODO: check that mean visibility amplitude is identical to the degen_sol mean visbility amplitude 