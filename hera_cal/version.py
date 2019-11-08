# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import os
import sys
import subprocess
import json
import inspect

hera_cal_dir = os.path.dirname(os.path.realpath(__file__))


def _get_git_output(args, capture_stderr=False):
    """Get output from git, ensuring that it is of the ``str`` type,
    not bytes."""

    argv = ['git', '-C', hera_cal_dir] + args

    if capture_stderr:
        data = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    else:
        data = subprocess.check_output(argv)

    data = data.strip()
    return data.decode('utf8')


def _get_gitinfo_file(git_file=None):
    """Get saved info from GIT_INFO file that was created when installing package"""
    if git_file is None:
        git_file = os.path.join(hera_cal_dir, 'GIT_INFO')

    with open(git_file) as data_file:
        data = [x for x in json.loads(data_file.read().strip())]
        git_origin = data[0]
        git_hash = data[1]
        git_description = data[2]
        git_branch = data[3]

    return {'git_origin': git_origin, 'git_hash': git_hash,
            'git_description': git_description, 'git_branch': git_branch}


def construct_version_info():
    version_file = os.path.join(hera_cal_dir, 'VERSION')
    with open(version_file) as f:
        version = f.read().strip()

    git_origin = ''
    git_hash = ''
    git_description = ''
    git_branch = ''

    version_info = {'version': version, 'git_origin': '', 'git_hash': '',
                    'git_description': '', 'git_branch': ''}

    try:
        version_info['git_origin'] = _get_git_output(['config', '--get', 'remote.origin.url'], capture_stderr=True)
        version_info['git_hash'] = _get_git_output(['rev-parse', 'HEAD'], capture_stderr=True)
        version_info['git_description'] = _get_git_output(['describe', '--dirty', '--tag', '--always'])
        version_info['git_branch'] = _get_git_output(['rev-parse', '--abbrev-ref', 'HEAD'], capture_stderr=True)
    except subprocess.CalledProcessError:  # pragma: no cover
        try:
            # Check if a GIT_INFO file was created when installing package
            version_info.update(_get_gitinfo_file())
        except (IOError, OSError):
            pass

    return version_info


def history_string(notes=''):
    '''Creates a standardized history string that all functions that write to disk can use. Optionally add notes.'''
    history = '\n------------\nThis file was produced by the function ' + str(inspect.stack()[1][3]) + '()'
    # inspect.stack()[1][3] is the name of the function that called this function
    history += ' in ' + os.path.basename(inspect.stack()[1][1]) + ' using: '
    # inspect.stack()[1][1] is path to the file that contains the function that called this function
    version_info = construct_version_info()
    for v in sorted(version_info.keys()):
        history += '\n    ' + v + ': ' + version_info[v]
    if (notes is not None) and (notes != ''):
        history += '\n\nNotes:\n'
        history += notes
    return history + '\n------------\n'


version_info = construct_version_info()
version = version_info['version']
git_origin = version_info['git_origin']
git_hash = version_info['git_hash']
git_description = version_info['git_description']
git_branch = version_info['git_branch']


def main():
    print('Version = {0}'.format(version))
    print('git origin = {0}'.format(git_origin))
    print('git branch = {0}'.format(git_branch))
    print('git description = {0}'.format(git_description))


if __name__ == '__main__':
    main()
