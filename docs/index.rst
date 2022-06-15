.. hera_cal documentation master file, created by
   sphinx-quickstart on Tue May 16 13:29:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hera_cal
=======

.. image:: https://travis-ci.org/HERA-Team/hera_cal.svg?branch=master
           :target: https://travis-ci.org/HERA-Team/hera_cal

.. image:: https://coveralls.io/repos/github/HERA-Team/hera_cal/badge.svg?branch=master
           :target: https://coveralls.io/github/HERA-Team/hera_cal?branch=master

hera_cal is a package that contains modules and scripts that are required to run
redundant calibration on the Hydrogen Epoch of Reionization Array (HERA) as part
of the real time calibration system. For more on HERA, visit http://reionization.org/


Package Details
===============
hera_cal aims to have a well supported and tested scripts to run calibration analysis.
All new functions, classes, modules, and scripts shall be well tested. Test coverage
should be > 95%.

Installation Instructions
============

Dependencies
------------

First Install all dependencies.

-  numpy >= 1.10
-  scipy
-  astropy >=1.2
-  aipy
-  pyuvdata >= 1.1
-  omnical >= 5.0.2
-  matplotlib

Installing hera_cal
---------------
To get the latest version of hera_cal, clone the repository with
``git clone https://github.com/HERA-Team/hera_cal.git``

Navigate to into the hera_cal directory and run ``python setup.py install``.

To install without dependencies, run
``python setup.py install --no-dependencies``

Tests
-----
Requires installation of ``nose`` package. From the source hera_cal
directory run ``nosetests hera_cal``.


Further Documentation
=====================

.. toctree::
   :maxdepth: 1

   scripts_overview
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
