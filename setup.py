from setuptools import setup
import glob
import os.path as path
from os import listdir
import sys,os

__version__ = '0.0.0'

def package_files(package_dir,subdirectory):
    #walk the input package_dir/subdirectory
    #return a package_data list
    paths = []
    directory = os.path.join(package_dir,subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir+'/','')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('hera_cal','data') + package_files('hera_cal','calibrations')

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
    'package_data': {'hera_cal': data_files},
    #    'install_requires': ['numpy>=1.10', 'scipy', 'pyuvdata', 'astropy>1.2', 'aipy']
    #    'dependency_links': ['https://github.com/zakiali/omnical/tarball/master#egg=omnical-dev',]
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
