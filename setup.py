from setuptools import setup
import glob
import os.path as path
from os import listdir

__version__ = '0.0.0'

setup_args = {
    'name': 'hera_cal',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_cal',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'package_dir': {'hera_cal': 'hera_cal'},
    'packages': ['hera_cal'],
    #    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    'package_data': {'hera_cal': ['data/*py', 'data/*uv*/*', 'data/test_input/*', 'calibrations/*']},
    #    'install_requires': ['numpy>=1.10', 'scipy', 'pyuvdata', 'astropy>1.2', 'aipy']
    #    'dependency_links': ['https://github.com/zakiali/omnical/tarball/master#egg=omnical-dev',]
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
