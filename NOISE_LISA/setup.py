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
      package_dir={'NOISE_LISA': 'NOISE_LISA'},
      package_data={'NOISE_LISA': ['parameters/*.txt']},
      zip_safe=False,
      #include_package_data=True
      #data_files=['/NOISE_LISA/parameters/*']
      #package_data = {'':['*.txt'],}
      )

