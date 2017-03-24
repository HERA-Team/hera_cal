from setuptools import setup
import glob
import os.path as path
from os import listdir

__version__ = '0.0.0'

setup_args = {
    'name': 'heracals',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/heracals',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'package_dir': {'heracals': 'heracal'},
    'packages': ['heracal'],
#    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    # 'package_data':
    'install_requires': ['numpy>=1.10', 'scipy', 'pyuvdata', 'astropy>1.2', 'aipy'],
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
