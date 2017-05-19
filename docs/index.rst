.. heracal documentation master file, created by
   sphinx-quickstart on Tue May 16 13:29:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

heracal
=======

.. image:: https://travis-ci.org/HERA-Team/heracal.svg?branch=master
           :target: https://travis-ci.org/HERA-Team/heracal

.. image:: https://coveralls.io/repos/github/HERA-Team/heracal/badge.svg?branch=master
           :target: https://coveralls.io/github/HERA-Team/heracal?branch=master

heracal is a package that contains modules and scripts that are required to run
redundant calibration on the hydrogen epoch of reionization array (HERA) as part
of the real time calibration system.


Package Details
===============
heracal aims to have a well supported and tested scripts to run calibration analysis.
All new functions, classes, modules, and scripts shall be well tested. Test coverage 
should be > 95%. 

Installation
============

Dependencies
------------

First Install all dependencies.

-  numpy >= 1.10
-  scipy
-  astropy >=1.2
-  pyephem
-  aipy
-  pyuvdata >= 1.1
-  omnical >= 5.0.2
-  matplotlib

Install heracal
---------------
To get the latest version of heracal, clone the repository with 
``git clone https://github.com/HERA-Team/heracal.git``

Navigate to into the heracal directory and run ``python setup.py install``.

To install without dependencies, run
``python setup.py install --no-dependencies``

Tests
-----
Requires installation of ``nose`` package. From the source heracal 
directory run ``nosetests heracal``.


Further Documentation
=====================

.. toctree::
   :maxdepth: 1
    
   omni
   firstcal
   metrics
   xrfi



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

