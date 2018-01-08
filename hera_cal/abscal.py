"""
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linerizations of the
(complex) calibration equation:

V_ij^model = g_i x g_j^* x V_ij^data

where

V_ij^model = exp(eta_ij^model + i x phi_ij^model)
g_i = exp(eta_i + i x phi_i)
g_j = exp(eta_j + i x phi_j)
V_ij^data = exp(eta_ij^data + i x phi_ij^data)

There are five calibration methods, where the
RHS of each equation contains the free parameters:

1. per-antenna amplitude logarithmic calibration
---------------------------------------------

eta_ij^model - eta_ij^data = eta_i + eta_j


2. per-antenna phase logarithmic calibration
-----------------------------------------

phi_ij^model - phi_ij^data = phi_i - phi_j


3. Average amplitude linear calibration
----------------------------------------

|V_ij^model| / |V_ij^data| = |g_avg|


4. Tip-Tilt phase logarithmic calibration 
-----------------------------------------

phi_ij^model - phi_ij^data = PSI + dot(THETA, B_ij)

where PSI is an overall gain phase scalar, 
THETA is the gain phase slope vector [radians / meter]
and B_ij is the baseline vector between antenna i and j.


5. delay linear calibration
---------------------------

tau_ij^model - tau_ij^data = tau_i - tau_j

where tau is the delay that can be turned
into a complex gain via: g_i = exp(i x 2pi x tau_i x nu)
"""
from abscal_funcs import *


class AbsCal(object):

    def __init__(self, model, data, wgts=None, antpos=None, freqs=None, times=None, pols=[None]):
        """
        AbsCal object for absolute calibration of flux scale and phasing
        given a visibility model and measured data. model, data and weights
        should be fed as dictionary types,

        Parameters:
        -----------
        model : dict of visibility data of refence model, type=dictionary
            keys are antenna pair tuples, values are complex ndarray visibilities
            these visibilities must be 3D arrays, with the [0] axis indexing time,
            the [1] axis indexing frequency and the [2] axis indexing polarization

            Example: Ntimes = 2, Nfreqs = 3, Npol = 2
            model = {(0, 1): np.array([ [[1+0j, 2+1j, 0+2j],
                                         [3-1j,-1+2j, 0+2j]],
                                        [[3+1j, 4+0j,-1-3j],
                                         [4+2j, 0+0j, 0-1j]] ]), ...}

        data : dict of visibility data of measurements, type=dictionary
            keys are antenna pair tuples (must match model), values are
            complex ndarray visibilities, with shape matching model

        antpos : dict of antenna position vectors in TOPO frame in meters, type=dictionary
            keys are antenna integers and values are 2D or 3D ndarray
            position vectors in meters (topocentric coordinates),
            with [0] index containing X (E-W) distance, and [1] index Y (N-S) distance.

        wgts : dict of weights of data, type=dictionry, [default=None]
            keys are antenna pair tuples (must match model), values are real floats
            matching shape of model and data

        verbose : print output, type=boolean, [default=False]

        freqs : ndarray of frequency array, type=ndarray, dtype=float
            1d array containing visibility frequencies in Hz. Needed to write out to calfits.
    
        times : ndarray of time array, type=ndarray, dtype=float
            1d array containing visibility times in Julian Date. Needed to write out to calfits.

        pols : ndarray of polarization array, type=ndarray, dtype=int
            array containing polarization integers in pyuvdata.UVData.polarization_array 
            format. Needed to write out to calfits.
        """
        # append attributes
        self.model = model
        self.data = data

        # setup frequencies
        self.Nfreqs = model[model.keys()[0]].shape[1]
        self.freqs = freqs

        # setup polarization
        self.str2pol = {"xx": -5, "yy": -6, "xy": -7, "yx": -8}
        self.Npols = model[model.keys()[0]].shape[2]
        if pols is not None:
            if type(pols) is list:
                if type(pols[0]) is str:
                    pols = map(lambda x: self.str2pol[x], pols)
            elif type(pols) is str:
                pols = [self.str2pol[pols]]
            elif type(pols) is int:
                pols = [pols]
        self.pols = pols

        # setup weights
        if wgts is None:
            wgts = copy.deepcopy(data)
            for i, k in enumerate(data.keys()):
                wgts[k] = np.ones_like(wgts[k], dtype=np.float)
        self.wgts = wgts

        # setup times
        self.Ntimes = model[model.keys()[0]].shape[0]
        self.times = times

        # setup ants
        self.ants = np.unique(sorted(np.array(map(lambda x: [x[0], x[1]], model.keys())).ravel()))
        self.Nants = len(self.ants)

        # setup baselines and antenna positions
        self.antpos = antpos
        if self.antpos is not None:
            self.bls = odict([((x[0], x[1]), self.antpos[x[1]] - self.antpos[x[0]]) for x in self.model.keys()])
            self.antpos = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos -= np.median(self.antpos, axis=0)

    def abs_amp_lincal(self, unravel_freq=False, unravel_time=False, unravel_pol=False,
                       apply_cal=False, verbose=True):
        """
        call abscal_funcs.abs_amp_lincal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization

        apply_cal : turn calibration solution into gains and apply to data
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1)
            unravel(wgts, 'f', 1)

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2)
            unravel(wgts, 'p', 2)

        # run linsolve
        fit = abs_amp_lincal(model, data, wgts=wgts, verbose=verbose)
        self.abs_amp = copy.copy(np.sqrt(fit['amp']))

        ### TODO apply_cal

    def TT_phs_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False,
                      verbose=True, zero_psi=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)
        bls = copy.deepcopy(self.bls)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0, copy_dict=bls)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1, copy_dict=bls)
            unravel(wgts, 'f', 1)

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2, copy_dict=bls)
            unravel(wgts, 'p', 2)
           
        # run linsolve
        fit = TT_phs_logcal(model, data, bls, wgts=wgts, verbose=verbose, zero_psi=zero_psi)
        self.psi = copy.copy(fit['psi'])
        self.TT_PHI = copy.copy(np.array([fit['PHIx'], fit['PHIy']]))

    def amp_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=True):
        """
        call abscal_funcs.amp_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1)
            unravel(wgts, 'f', 1)

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2)
            unravel(wgts, 'p', 2)

        # run linsolve
        fit = amp_logcal(model, data, wgts=wgts, verbose=verbose)
        self.ant_eta = np.array(map(lambda a: copy.copy(fit['eta{}'.format(a)]), self.ants))

    def phs_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=True):
        """
        call abscal_funcs.phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1)
            unravel(wgts, 'f', 1)

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2)
            unravel(wgts, 'p', 2)

        # run linsolve
        fit = phs_logcal(model, data, wgts=wgts, verbose=verbose)
        self.ant_phi = np.array(map(lambda a: copy.copy(fit['phi{}'.format(a)]), self.ants))


    def delay_lincal(self, kernel=(1, 11), verbose=True):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Parameters:
        -----------
        kernel : size of median filter across (time, freq) axes, type=(int, int)
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)

        df = np.median(np.diff(self.freqs))

        # iterate over polarizations
        dlys = odict(map(lambda a: ('tau{}'.format(a), []), self.ants))
        for i, p in enumerate(self.pols):
            # run linsolve
            m = odict(zip(model.keys(), map(lambda k: model[k][:, :, i], model.keys())))
            d = odict(zip(data.keys(), map(lambda k: data[k][:, :, i], data.keys())))
            fit = delay_lincal(m, d, df=df, kernel=kernel, verbose=verbose, time_ax=0, freq_ax=1)
            for j, k in enumerate(fit.keys()):
                dlys[k].append(fit[k])

        # turn into array
        self.delays = np.array(map(lambda a: np.moveaxis(dlys['tau{}'.format(a)], 0, 2), self.ants))


    def make_gains(self, gains2dict=False, verbose=True):
        """
        use self.gain_amp and self.gain_phi and self.gain_psi
        to construct a complex gain array per antenna assuming
        a gain convention of multiply.

        Parameters:
        -----------
        gains2dict : boolean, if True convert gains into dictionary form
            with antenna number as key and ndarray as value
        """
        # form blank gain array
        gain_array = np.ones((self.Nants, self.Ntimes, self.Nfreqs, self.Npols), dtype=np.complex)

        # multiply absolute amplitude
        try:
            gain_array = self.abs_amp[np.newaxis]
        except AttributeError:
            pass

        # multiply overall phase
        try:
            gain_array *= np.exp(-1j*self.psi[np.newaxis])
        except AttributeError:
            pass

        # multiply phase slope
        try:
            gain_array *= np.exp(-1j*np.einsum("ijkl, hi -> hjkl", self.TT_PHI, self.antpos[:, :2]))
        except AttributeError:
            pass

        # multiply delay
        try:
            gain_array *= np.exp(-1j*2*np.pi*self.freqs.reshape(-1, 1)*self.delays)
        except AttributeError:
            pass

        # multiply ant amp
        try:
            gain_array *= np.exp(self.ant_amp)
        except:
            pass

        # multiply ant phase
        try:
            gain_array *= np.exp(-1j*self.ant_phi)
        except:
            pass

        self.gain_array = gain_array

        if gains2dict:
            self.gain_array = odict((a, self.gain_array[i]) for i, a in enumerate(self.ants))

    def write_calfits(self, calfits_fname, verbose=True, overwrite=False, gain_convention='multiply'):
        """
        """
        echo("saving {}".format(calfits_fname), type=1, verbose=verbose)
        gains2calfits(calfits_fname, self.gain_array, self.freqs, self.times, self.pols,
                      gain_convention=gain_convention, inttime=10.7, overwrite=overwrite)


''' TO DO
    def smooth_data(self, data, flags=None, kind='linear'):
        """


        Parameters:
        -----------
        data : 


        flags : 
    
        """
        # configure parameters



        Ntimes
        Nfreqs

        # interpolate flagged data


        # sphere training data x-values


        # ravel training data


        if kind == 'poly':
            # fit polynomial
            data = smooth_data(Xtrain_raveled, ytrain_raveled, Xpred_raveled, kind=kind, degree=degree)

        if kind == 'gp':
            # construct GP mean function from a degree-order polynomial
            MF = make_pipeline(PolynomialFeatures(degree), linear_model.RANSACRegressor())
            MF.fit(Xtrain_raveled, ytrain_raveled)
            y_mean = MF.predict(Xtrain_raveled)

            # make residual and normalize by std
            y_resid = (ytrain_raveled - y_mean).reshape(Ntimes, Nfreqs)
            y_std = np.sqrt(astats.biweight_midvariance(y_resid.ravel()))
            y_resid /= y_std

            # average residual across time
            ytrain = np.mean(y_resid, axis=0)

            # ravel training data
            Xtrain_raveled = Xtrain[0, :].reshape(-1, 1)
            ytrain_raveled = ytrain
            Xpred_raveled = Xpred[0, :]

            # fit GP and predict MF
            y_pred = smooth_data(Xtrain_raveled, ytrain_raveled, Xpred_raveled) * y_std
            y_pred = np.repeat(y_pred, Ntimes)
            data = (y_pred + y_mean).reshape(Ntimes, Nfreqs)
'''