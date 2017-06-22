import unittest
import glob
import heracal.xrfi as xrfi
import numpy as np
import pylab as plt

np.random.seed(0)

SIZE = 100
VERBOSE = False
PLOT = False

FILES = {
    'paper': glob.glob('xrfi_data/paper/chisq0*.npz'),
    'hera': glob.glob('xrfi_data/hera/chisq0*.npz'),
}


def get_accuracy(f, rfi, verbose=VERBOSE):
    correctly_flagged = np.average(f[rfi])
    m = f.copy()
    m[rfi] = 0
    false_positive = float(np.sum(m)) / (m.size - len(rfi[0]))
    if verbose:
        print '\t Found RFI: %1.3f\n\t False Positive: %1.3f' % (correctly_flagged, false_positive)
    return correctly_flagged, false_positive


def plot_waterfall(data, f, mx=10, drng=10, mode='lin'):
    if not PLOT:
        return
    plt.subplot(121)
    capo.plot.waterfall(data, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.subplot(122)
    capo.plot.waterfall(f, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.show()


def plot_result(f, rfi):
    if not PLOT:
        return
    plt.plot(rfi[0], rfi[1], 'ko')
    fi = np.where(f)
    plt.plot(fi[0], fi[1], 'r.')
    plt.show()


class Template(unittest.TestCase):

    def setUp(self):
        raise unittest.SkipTest  # setUp has to be overridden to actually run a test
    rfi_gen = None  # Need to override this for each TestCase, usually in setUp

    def _run_test(self, func, correct_flag, false_positive, nsig=4):
        for data, rfi in self.rfi_gen():
            f = func(data)
            if VERBOSE:
                print self.__class__, func.__name__
            # plot_waterfall(data, f)
            f = np.where(f > nsig, 1, 0)
            cf, fp = get_accuracy(f, rfi)
            # plot_result(f, rfi)
            self.assertGreater(cf, correct_flag)
            self.assertLess(fp, false_positive)
    ans = {
        'detrend_deriv': (.9, .1),
        'detrend_medfilt': (.99, .01),
        'detrend_medminfilt': (.97, .05),
        'xrfi_simple': (.99, .1),
        'xrfi': (.99, .01),
    }

    def test_detrend_deriv(self):
        cf, fp = self.ans['detrend_deriv']
        self._run_test(xrfi.detrend_deriv, cf, fp, nsig=4)

    def test_detrend_medfilt(self):
        cf, fp = self.ans['detrend_medfilt']
        self._run_test(xrfi.detrend_medfilt, cf, fp, nsig=4)

    def test_detrend_medminfilt(self):
        cf, fp = self.ans['detrend_medminfilt']
        self._run_test(xrfi.detrend_medminfilt, cf, fp, nsig=6)

    def test_xrfi_simple(self):
        cf, fp = self.ans['xrfi_simple']
        self._run_test(xrfi.xrfi_simple, cf, fp, nsig=.5)

    def test_xrfi(self):
        cf, fp = self.ans['xrfi']
        self._run_test(xrfi.xrfi, cf, fp, nsig=.5)


class TestSparseScatter(Template):

    def setUp(self):
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = np.random.normal(size=(SIZE, SIZE))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen


class TestDenseScatter(Template):

    def setUp(self):
        RFI = 1000
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = np.random.normal(size=(SIZE, SIZE))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = (.33, .1)
        self.ans['xrfi_simple'] = (.90, .1)


class TestCluster(Template):

    def setUp(self):
        RFI = 10
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = np.random.normal(size=(SIZE, SIZE))
                x, y = (np.random.randint(SIZE - 1, size=RFI),
                        np.random.randint(SIZE - 1, size=RFI))
                x = np.concatenate([x, x, x + 1, x + 1])
                y = np.concatenate([y, y + 1, y, y + 1])
                rfi = (np.array(x), np.array(y))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['xrfi_simple'] = (.39, .1)
        self.ans['detrend_deriv'] = (-.05, .1)


class TestLines(Template):

    def setUp(self):
        RFI = 3
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = np.random.normal(size=(SIZE, SIZE))
                x, y = (np.random.randint(SIZE, size=RFI),
                        np.random.randint(SIZE, size=RFI))
                mask = np.zeros_like(data)
                mask[x] = 1
                mask[:, y] = 1
                data += mask * NSIG
                yield data, np.where(mask)
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = (.0, .1)
        self.ans['xrfi_simple'] = (.75, .1)
        self.ans['xrfi'] = (.97, .01)


class TestBackground(Template):

    def setUp(self):
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                sin_t = np.sin(np.linspace(0, 2 * np.pi, SIZE))
                sin_t.shape = (-1, 1)
                sin_f = np.sin(np.linspace(0, 4 * np.pi, SIZE))
                sin_f.shape = (1, -1)
                data = 5 * sin_t * sin_f + np.random.normal(size=(SIZE, SIZE))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = (.83, .1)
        self.ans['detrend_medminfilt'] = (.2, .1)
        self.ans['xrfi'] = (.75, .1)
        self.ans['xrfi_simple'] = (.90, .1)

# class TestHERA(Template):
#    def setUp(self):
#        def rfi_gen():
#            for f in FILES['hera']:
#                data = np.load(f)['chisq']
#                rfi = np.where(xrfi.xrfi(data)) # XXX actual answers?
#                yield data, rfi
#            return
#        self.rfi_gen = rfi_gen
#        self.ans['detrend_deriv'] = (.05, .1)
#        self.ans['detrend_medfilt'] = (.5, .1)
#        self.ans['detrend_medminfilt'] = (.30, .1)
#        self.ans['xrfi_simple'] = (.40, .3)
#
# class TestPAPER(Template):
#    def setUp(self):
#        def rfi_gen():
#            for f in FILES['paper']:
#                data = np.load(f)['chisq']
#                rfi = np.where(xrfi.xrfi(data)) # XXX actual answers?
#                yield data, rfi
#            return
#        self.rfi_gen = rfi_gen
#        self.ans['detrend_deriv'] = (.0, .1)
#        self.ans['detrend_medfilt'] = (.1, .1)
#        self.ans['detrend_medminfilt'] = (.0, .35)
#        self.ans['xrfi_simple'] = (.3, .5)


# TODO: noise tilts
# TODO: faint RFI
# TODO: combination of everything


if __name__ == '__main__':
    unittest.main()
