from setuptools import setup
import os
setup(name='NOISE_LISA',
      version='0.1',
      description='Obtaining the noise of LISA and calculate its influence on the signal',
      url='https://github.com/eabram/synthlisa/tree/master_ester/calculations/NOISE_LISA',
      author='Ester Abram',
      author_email='esterabram@hotmail.com',
      license='Nikhef/TNO',
      packages=['NOISE_LISA'],
      zip_safe=False,
      data_files=['/NOISE_LISA/parameters/']
      )

#include_package_data=True
