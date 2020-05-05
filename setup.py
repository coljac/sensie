#!/usr/bin/env python

from distutils.core import setup

setup(name='Sensie',
      version='0.1',
      description='Sensie Neural Network Probe',
      author='Colin Jacobs',
      author_email='colin@coljac.space',
      url='https://github.com/coljac/sensie/',
      packages=['sensie'],
      install_requires = ["matplotlib", "numpy",
        "pandas", "pymc3==3.7",
        "scikit-learn", "scipy"
       ]
     )

