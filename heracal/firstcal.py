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
