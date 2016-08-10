#!/usr/bin/env python

from distutils.core import setup

setup(name='pyERA',
  version='0.1',
  url='https://github.com/mpatacchiola/pyERA',
  description='Implementation of the Epigenetic Robotic Architecture (ERA). It includes standalone classes for Self-Organizing Maps (SOM) and Hebbian Learning',
  author='Massimiliano Patacchiola',
  packages = ['pyERA'],
  package_data={'pyERA': ['Readme.md']},
  include_package_data=True,
  license="The MIT License (MIT)",
  py_modules=['pyERA'],
  requires = ['numpy']
 )
