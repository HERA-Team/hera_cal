# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Module for predicting noise on visibilties using autocorrelations and for producing noise standard deviation data files."""

from __future__ import print_function, division, absolute_import

import numpy as np
import argparse

from . import io
from .utils import split_pol, predict_noise_variance_from_autos
from .datacontainer import DataContainer

def per_antenna_noise_stdev(autos, dt=None, df=None):
    '''Predicts per-antenna noise using autocorrelation data. The result is a noise standard deviation. To predict
    the noise variance on e.g. (1, 2, 'xy'), one would use noise[1, 1, 'xx'] * noise[2, 2, 'yy'].
    
    Arguments:
        autos: DataContainer containing autocorrelation data
        dt: integration time in seconds. If None, will try infer this from the times stored in the DataContainer
        df: channel width in Hz. If None, will try to infer this from the frequencies stored in the DataContainer

    Returns:
        noise: DataContainer mapping autocorrelation baselines like (1, 1, 'xx') to predictions for the standard
            deviation of noise for that antenna.
    '''
    noise = {}
    for bl in autos:
        if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1]):
            noise[bl] = np.sqrt(predict_noise_variance_from_autos(bl, autos, dt=dt, df=df))
    return DataContainer(noise)


