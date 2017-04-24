'''
Tools for dealing with redundant array configurations.
'''

import numpy as n
import aipy as a
import pylab as p
import time 
import multiprocessing as mpr

def fit_line(phs, fqs, valid, offset=False):
    fqs = fqs.compress(valid)
    dly = phs.compress(valid)
    B = n.zeros((fqs.size,1)); B[:,0] = dly
    if offset:
        A = n.zeros((fqs.size,2)); A[:,0] = fqs*2*n.pi; A[:,1] = 1
    else:
        A = n.zeros((fqs.size,1)); A[:,0] = fqs*2*n.pi#; A[:,1] = 1
    try:
        if offset:
            dt,off = n.linalg.lstsq(A,B)[0].flatten()
        else:
            dt = n.linalg.lstsq(A,B)[0][0][0]
            off = 0.0
        return dt,off
    except ValueError:
        import IPython 
        IPython.embed()

def mpr_clean(args):
    return a.deconv.clean(*args,tol=1e-4)[0]

def redundant_bl_cal_simple(d1,w1,d2,w2,fqs, cleantol=1e-4, window='none', finetune=True, verbose=False, plot=False, noclean=True, average=False, offset=False):
    '''Gets the phase differnce between two baselines by using the fourier transform and a linear fit to that residual slop. 
        Parameters
        ----------
        d1,d2 : NXM numpy arrays.
            Data arrays to find phase difference between. First axis is time, second axis is frequency. 
        w1,w2 : NXM numpy arrays. 
            corrsponding data weight arrays.
        fqs   : 1XM numpy array
            Array of frequencies in GHz.
        window: str
            Name of window function to use in fourier transform. Default is 'none'.
        finetune  : boolean 
            Flag if you want to fine tune the phase fit with a linear fit. Default is true.
        verbose: boolean
            Be verobse. Default is False.
        plot: boolean
            Turn on low level plotting of phase ratios. Default is False.
        cleantol: float
            Clean tolerance level. Default is 1e-4.
        noclean: boolean
            Don't apply clean deconvolution to remove the sampling function (weights).
        average: boolean
            Average the data in time before applying analysis. collapses NXM -> 1XM.
        offset: boolean
            Solve for a offset component in addition to a delay component.

        Returns
        -------
        delays : ndarrays
            Array of delays (if average == False), or single delay.    
        offsets: ndarrays
            Array of offsets (if average == False), or single delay.
    
'''
    d12 = d2 * n.conj(d1)
    # For 2D arrays, assume first axis is time. 
    if average:
        if d12.ndim > 1:
            d12_sum = n.sum(d12,axis=0).reshape(1,-1)
            d12_wgt = n.sum(w1*w1,axis=0).reshape(1,-1)
        else:
            d12_sum = d12.reshape(1,-1)
            d12_wgt = w1.reshape(1,-1)*w2.reshape(1,-1)
    else:
        d12_sum = d12
        d12_wgt = w1*w2
    #normalize data to maximum so that we minimize fft articats from RFI
    d12_sum *= d12_wgt
    d12_sum = d12_sum/n.where(n.abs(d12_sum)==0., 1., n.abs(d12_sum)) 
    window = a.dsp.gen_window(d12_sum[0,:].size, window=window)
    dlys = n.fft.fftfreq(fqs.size, fqs[1]-fqs[0])
    # FFT and deconvolve the weights to get the phs
    _phs = n.fft.fft(window*d12_sum,axis=-1)
    _wgt = n.fft.fft(window*d12_wgt,axis=-1)
    _phss = n.zeros_like(_phs)
    if not noclean and average:
        _phss = a.deconv.clean(_phs, _wgt, tol=cleantol)[0]
    elif not noclean:
        pool=mpr.Pool(processes=4)
        _phss = pool.map(mpr_clean, zip(_phs,_wgt))
    else:
        _phss = _phs
    _phss = n.abs(_phss)
    #get bin of phase
    mxs = n.argmax(_phss, axis=-1)
    #Fine tune with linear fit.
    mxs[mxs>_phss.shape[-1]/2] -= _phss.shape[-1]
    dtau = mxs / (fqs[-1] - fqs[0])
    mxs = n.dot(mxs.reshape(len(mxs),1),n.ones((1,3),dtype=int)) + n.array([-1,0,1])
    taus = n.sum(_phss[n.arange(mxs.shape[0],dtype=int).reshape(-1,1),mxs] * dlys[mxs],axis=-1) / n.sum(_phss[n.arange(mxs.shape[0]).reshape(-1,1),mxs],axis=-1)
    dts,offs = [],[]
    if finetune:
    #loop over the linear fits
        t1 = time.time()
        for ii,(tau,d) in enumerate(zip(taus,d12_sum)):
            valid = n.where(d != 0, 1, 0) # Throw out zeros, which NaN in the log below
            valid = n.logical_and(valid, n.logical_and(fqs>.11,fqs<.19))
            dly = n.angle(d*n.exp(-2j*n.pi*tau*fqs))
            dt,off = fit_line(dly,fqs,valid,offset=offset)
            dts.append(dt), offs.append(off)
            if plot:
                p.subplot(411)
                p.plot(fqs,n.angle(d12_sum[ii]), linewidth=2)
                p.plot(fqs,d12_sum[ii], linewidth=2)
                p.plot(fqs, n.exp((2j*n.pi*fqs*(tau+dt))+off))
                p.hlines(n.pi, .1,.2,linestyles='--',colors='k')
                p.hlines(-n.pi, .1,.2,linestyles='--',colors='k')
                p.subplot(412)
                p.plot(fqs,n.unwrap(dly)+2*n.pi*tau*fqs, linewidth=2)
                p.plot(fqs,dly+2*n.pi*tau*fqs, linewidth=2,ls='--')
                p.plot(fqs,2*n.pi*tau*fqs, linewidth=2,ls='-.')
                p.plot(fqs,2*n.pi*(tau+dt)*fqs + off, linewidth=2,ls=':')
                p.subplot(413)
                p.plot(dlys, n.abs(_phs[ii]),'-.')
                p.xlim(-400,400)
                p.subplot(414)
                p.plot(fqs,dly, linewidth=2)
                p.plot(fqs,off+dt*fqs*2*n.pi, '--')
                p.hlines(n.pi, .1,.2,linestyles='--',colors='k')
                p.hlines(-n.pi, .1,.2,linestyles='--',colors='k')
                print 'tau=', tau
                print 'tau + dt=', tau+dt
                p.xlabel('Frequency (GHz)', fontsize='large')
                p.ylabel('Phase (radians)', fontsize='large')
        p.show()

        dts = n.array(dts)
        offs = n.array(offs)

    info = {'dtau':dts, 'off':offs, 'mx':mxs}
    if verbose: print info, taus, taus+dts, offs
    return taus+dts,offs


class FirstCalRedundantInfo(omnical.info.FirstCalRedundantInfo):
    def __init__(self, nant):
        omnical.info.FirstCalRedundantInfo.__init__(self)
        self.nant = nant
        print 'Loading FirstCalRedundantInfo class'

    def order_data(self, dd):
        '''Create a data array ordered for use in _omnical.redcal.  'dd' is
        a dict whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
        the conjugation convention of the provided data.  'dd' values are 2D arrays
        of (time,freq) data.''' # XXX does time/freq ordering matter.  should data be 2D instead?
        return np.array([dd[bl] if dd.has_key(bl) else dd[bl[::-1]].conj()
            for bl in self.bl_order()]).transpose((1,2,0))

def compute_reds(nant, pols, *args, **kwargs):
    _reds = omnical.arrayinfo.compute_reds(*args, **kwargs)
    reds = []
    for pi in pols:
        for pj in pols:
            reds += [[(Antpol(i, pi, nant), Antpol(j, pj, nant)) for i, j in gp] for gp in _reds]
    return reds


class FirstCal(object):
    def __init__(self, data, wgts, fqs, info):
        self.data = data
        self.fqs = fqs
        self.info = info
        self.wgts = wgts

    def data_to_delays(self, **kwargs):
        '''data = dictionary of visibilities.
           info = FirstCalRedundantInfo class
           can give it kwargs:
                supports 'window': window function for fourier transform. default is none
                         'tune'  : to fit and remove a linear slope to phase.
                         'plot'  : Low level plotting in the red.redundant_bl_cal_simple script.
                         'clean' : Clean level when deconvolving sampling function out.
           Returns 2 dictionaries:
                1. baseline pair : delays
                2. baseline pari : offset
        '''
        verbose = kwargs.get('verbose', False)
        blpair2delay = {}
        blpair2offset = {}
        dd = self.info.order_data(self.data)
        ww = self.info.order_data(self.wgts)
        for (bl1, bl2) in self.info.bl_pairs:
            if verbose:
                print (bl1, bl2)
            d1 = dd[:, :, self.info.bl_index(bl1)]
            w1 = ww[:, :, self.info.bl_index(bl1)]
            d2 = dd[:, :, self.info.bl_index(bl2)]
            w2 = ww[:, :, self.info.bl_index(bl2)]
            delay, offset = redundant_bl_cal_simple(d1, w1, d2, w2, self.fqs, **kwargs)
            blpair2delay[(bl1, bl2)] = delay
            blpair2offset[(bl1, bl2)] = offset
        return blpair2delay, blpair2offset

    def get_N(self, nblpairs):
        return sps.eye(nblpairs)

    def get_M(self, **kwargs):
        blpair2delay, blpair2offset = self.data_to_delays(**kwargs)
        sz = len(blpair2delay[blpair2delay.keys()[0]])
        M = np.zeros((len(self.info.bl_pairs), sz))
        O = np.zeros((len(self.info.bl_pairs), sz))
        for pair in blpair2delay:
            M[self.info.blpair_index(pair), :] = blpair2delay[pair]
            O[self.info.blpair_index(pair), :] = blpair2offset[pair]
        return M, O

    def run(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        offset = kwargs.get('offset', False)
        # make measurement matrix
        print "Geting M,O matrix"
        self.M, self.O = self.get_M(**kwargs)
        print "Geting N matrix"
        N = self.get_N(len(self.info.bl_pairs))
        # XXX This needs to be addressed. If actually do invers, slows code way down.
        # self._N = np.linalg.inv(N)
        self._N = N  # since just using identity now

        # get coefficients matrix,A
        self.A = sps.csr_matrix(self.info.A)
        print 'Shape of coefficient matrix: ', self.A.shape

#        solve for delays
        print "Inverting A.T*N^{-1}*A matrix"
        invert = self.A.T.dot(self._N.dot(self.A)).todense()  # make it dense for pinv
        dontinvert = self.A.T.dot(self._N.dot(self.M))  # converts it all to a dense matrix
#        definitely want to use pinv here and not solve since invert is probably singular.
        self.xhat = np.dot(np.linalg.pinv(invert), dontinvert)
        # solve for offset
        if offset:
            print "Inverting A.T*N^{-1}*A matrix"
            invert = self.A.T.dot(self._N.dot(self.A)).todense()
            dontinvert = self.A.T.dot(self._N.dot(self.O))
            self.ohat = np.dot(np.linalg.pinv(invert), dontinvert)
            # turn solutions into dictionary
            return dict(zip(self.info.subsetant, zip(self.xhat, self.ohat)))
        else:
            # turn solutions into dictionary
            return dict(zip(self.info.subsetant, self.xhat))

    def get_solved_delay(self):
        solved_delays = []
        for pair in self.info.bl_pairs:
            ant_indexes = self.info.blpair2antind(pair)
            dlys = self.xhat[ant_indexes]
            solved_delays.append(dlys[0] - dlys[1] - dlys[2] + dlys[3])
        self.solved_delays = np.array(solved_delays)

# PYUV
def get_phase(fqs,tau, offset=False):
    fqs = fqs.reshape(-1,1) #need the extra axis
    if offset:
        delay = tau[0]
        offset = tau[1]
        return np.exp(-1j*(2*np.pi*fqs*delay) - offset)
    else:
        return np.exp(-2j*np.pi*fqs*tau)

# PYUV
def save_gains_fc(s,fqs,pol,filename,ubls=None,ex_ants=None,verbose=False):
    """
    s: solutions
    fqs: frequencies
    pol: polarization of single antenna i.e. 'x', or 'y'.
    filename: if a specific file was used (instead of many), change output name
    ubls: unique baselines used to solve for s'
    ex_ants: antennae excluded to solve for s'
    """
#    if isinstance(filename, list): filename=filename[0] #XXX this is evil. why?
    NBINS = len(fqs)
    s2 = {}
    for k,i in s.iteritems():
        #len > 1 means that one is using the "tune" parameter in omni.firstcal
        if len(i)>1:
            s2[str(k)+pol] = get_phase(fqs,i,offset=True).T
            s2['d'+str(k)] = i[0]
            s2['o'+str(k)] = i[1]
            if verbose: print 'dly=%f , off=%f'%i
        else:
            s2[str(k)+pol] = get_phase(fqs,i).T #reshape plays well with omni apply
            s2['d'+str(k)] = i
            if verbose: print 'dly=%f'%i
    if not ubls is None: s2['ubls']=ubls
    if not ex_ants is None: s2['ex_ants']=ex_ants
    s2['freqs']=fqs#in GHz
    outname='%s.fc.npz'%filename
    import sys
    s2['cmd'] = ' '.join(sys.argv)
    print 'Saving fcgains to %s'%outname
    np.savez(outname,**s2)
