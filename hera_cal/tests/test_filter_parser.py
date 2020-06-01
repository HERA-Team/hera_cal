# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
import sys
import filter_parser as fp

class Test_FilterParser(object):

    def test_filter_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--spw_range', '0', '20']
        parser = fp.delay_filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20

    def test_delay_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris']
        parser = fp.delay_filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.window == 'blackmanharris'

    def test_delay_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/']
        parser = fp.delay_filter_argparser(mode='dayenu')
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'

    def test_xtalk_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris', '--max_frate_coeffs', '0.024', '-0.229']
        parser = fp.xtalk_filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.window == 'blackmanharris'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] = 0.229

    def test_xtalk_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/', '--max_frate_coeffs', '0.024', '-0.229']
        parser = fp.xtalk_filter_argparser(mode='dayenu')
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == 0.229
