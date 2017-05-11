'''Module for all things Radio Frequency Interference Flagging'''
import numpy as np
from scipy.signal import medfilt

def medmin(d):
    #return np.median(np.min(chisq,axis=0))
    mn = np.min(d,axis=0)
    return 2*np.median(mn) - np.min(mn)

def medminfilt(d, K=8):
    d_sm = np.empty_like(d)
    for i in xrange(d.shape[0]):
        for j in xrange(d.shape[1]):
            i0,j0 = max(0,i-K), max(0,j-K)
            i1,j1 = min(d.shape[0], i+K), min(d.shape[1], j+K)
            d_sm[i,j] = medmin(d[i0:i1,j0:j1])
    return d_sm

#def omni_chisq_to_flags(chisq, K=8, sigma=6, sigl=2):
#    '''Returns a mask of RFI given omnical's chisq statistic'''
#    if False:
#        w_sm = np.empty_like(chisq)
#        sig = np.empty_like(chisq)
#        #get smooth component of chisq
#        for i in xrange(chisq.shape[0]):
#            for j in xrange(chisq.shape[1]):
#                i0,j0 = max(0,i-K), max(0,j-K)
#                i1,j1 = min(chisq.shape[0], i+K), min(chisq.shape[1], j+K)
#                #w_sm[i,j] = np.median(chisq[i0:i1,j0:j1])
#                w_sm[i,j] = medmin(chisq[i0:i1,j0:j1])
#    else: w_sm = medfilt(chisq, 2*K+1)
#    #the residual from smooth component
#    w_rs = chisq - w_sm 
#    w_sq = np.abs(w_rs)**2
#    #get the standard deviation of the media.
#    if False:
#        for i in xrange(chisq.shape[0]):
#            for j in xrange(chisq.shape[1]):
#                i0,j0 = max(0,i-K), max(0,j-K)
#                i1,j1 = min(chisq.shape[0], i+K), min(chisq.shape[1], j+K)
#                #sig[i,j] = np.sqrt(np.median(w_sq[i0:i1,j0:j1]))
#                sig[i,j] = np.sqrt(medmin(w_sq[i0:i1,j0:j1]))
#    else: sig = np.sqrt(medfilt(w_sq, 2*K+1))
#    #Number of sigma above the residual unsmooth part is.
#    f1 = w_rs / sig
#    return watershed_flag(f1, sig_init=sigma, sig_adj=sigl)


def watershed_flag(d, f=None, sig_init=6, sig_adj=2):
    '''Returns a watershed flagging of an array that is in units of standard
    deviation (i.e. how many sigma the datapoint is from the center).'''
    #mask off any points above 'sig' sigma and nan's.
    f1 = np.ma.array(d, mask=np.where(d > sig_init,1,0)) 
    f1.mask |= np.isnan(f1)
    if not f is None: f1.mask |= f
    
    # Loop over flagged points and examine adjacent points to see if they exceed sig_adj
    #Start the watershed
    prevx,prevy = 0,0
    x,y = np.where(f1.mask)
    while x.size != prevx and y.size != prevy:
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            prevx,prevy = x.size, y.size
            xp, yp = (x+dx).clip(0,f1.shape[0]-1), (y+dy).clip(0,f1.shape[1]-1)
            i = np.where(f1[xp,yp] > sig_adj)[0] # if sigma > 'sigl'
            f1.mask[xp[i],yp[i]] = 1
            x,y = np.where(f1.mask)
    return f1.mask
    
def toss_times_freqs(mask, sig_t=6, sig_f=6):
    f1ch = np.average(f1.mask, axis=0); f1ch.shape = (1,-1)
    #The cut off value is a made up number here...sig = 'sig' if none flagged.
    f1.mask = np.logical_or(f1.mask, np.where(f1 > sig_init*(1-f1ch), 1, 0))
    f1t = np.average(f1.mask, axis=1) # band-avg flag vs t
    ts = np.where(f1t > 2*np.median(f1t))
    f1.mask[ts] = 1
    f1f_sum = np.sum(f1.filled(0), axis=0)
    f1f_wgt = np.sum(np.logical_not(f1.mask), axis=0)
    f1f = f1f_sum / f1f_wgt.clip(1,np.Inf)
    fs = np.where(f1f > 2)
    f1.mask[:,fs] = 1
    mask = f1.mask
    return mask

def xrfi_simple(d, f=None, nsig_df=6, nsig_dt=6, nsig_all=0):
    if f is None: f = np.zeros(d.shape, dtype=np.bool)
    if nsig_df > 0:
        d_df = d[:,1:-1] - .5 * (d[:,:-2] + d[:,2:])
        d_df2 = np.abs(d_df)**2
        sig2 = np.median(d_df2, axis=1) # XXX 1 or 0 here?
        sig2.shape = (-1,1)
        f[:,0] = 1; f[:,-1] = 1
        f[:,1:-1] = np.where(d_df2 / sig2 > nsig_df**2, 1, f[:,1:-1])
    if nsig_dt > 0:
        d_dt = d[1:-1,:] - .5 * (d[:-2,:] + d[2:,:])
        d_dt2 = np.abs(d_dt)**2
        sig2 = np.median(d_dt2, axis=0) # XXX 0 or 1 here?
        sig2.shape = (1,-1)
        f[0,:] = 1; f[-1,:] = 1
        f[1:-1,:] = np.where(d_dt2 / sig2 > nsig_dt**2, 1, f[1:-1,:])
    if nsig_all > 0:
        ad = np.abs(d)
        med = np.median(ad)
        sig = np.sqrt(np.median(np.abs(ad-med)**2))
        f = np.where(ad > med + nsig_all * sig, 1, f)
    return f

def detrend_deriv(d, dt=True, df=True):
    '''XXX This only works ok on sparse RFI.'''
    if df:
        d_df = np.empty_like(d)
        d_df[:,1:-1] = (d[:,1:-1] - .5 * (d[:,:-2] + d[:,2:])) / np.sqrt(1.5)
        d_df[:,0] = (d[:,0] - d[:,1]) / np.sqrt(2)
        d_df[:,-1] = (d[:,-1] - d[:,-2]) / np.sqrt(2)
    else: d_df = d
    if dt:
        d_dt = np.empty_like(d_df)
        d_dt[1:-1] = (d_df[1:-1] - .5 * (d_df[:-2] + d_df[2:])) / np.sqrt(1.5)
        d_dt[0] = (d_df[0] - d_df[1]) / np.sqrt(2)
        d_dt[-1] = (d_df[-1] - d_df[-2]) / np.sqrt(2)
    else: d_d= d_df
    d2 = np.abs(d_dt)**2
    # model sig as separable function of 2 axes
    sig_f = np.median(d2, axis=0); sig_f.shape = (1,-1)
    sig_t = np.median(d2, axis=1); sig_t.shape = (-1,1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    return d_dt / sig

def detrend_medminfilt(d, K=8):
    d_sm = medminfilt(np.abs(d), 2*K+1)
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    sig = np.sqrt(medminfilt(d_sq, 2*K+1)) * (K/.64) # puts minmed on same scale as average
    f = d_rs / sig
    return f

def detrend_medfilt(d, K=8):
    d = np.concatenate([d[K-1::-1],d,d[:-K-1:-1]], axis=0)
    d = np.concatenate([d[:,K-1::-1],d,d[:,:-K-1:-1]], axis=1)
    d_sm = medfilt(d, 2*K+1)
    d_rs = d - d_sm 
    d_sq = np.abs(d_rs)**2
    sig = np.sqrt(medfilt(d_sq, 2*K+1) / .456) # puts median on same scale as average
    f = d_rs / sig
    return f[K:-K,K:-K]

def xrfi(d, f=None, K=8, sig_init=6, sig_adj=2):
    nsig = detrend_medfilt(d, K=K)
    f = watershed_flag(np.abs(nsig), f=f, sig_init=sig_init, sig_adj=sig_adj)
    return f

# XXX split off median filter as one type of flagger



