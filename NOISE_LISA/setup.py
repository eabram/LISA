from setuptools import setup
import os
import glob

data_folder = 'parameters/'
filename_list=['parameters/Waluschka/tele_offset.txt','parameters/Abram/tele_offset.txt']
#for filename in glob.iglob(data_folder+'/**/*'):
#    print (filename)
#    filename_list.append(filename.split(data_folder)[-1])

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
      data_files=[('parameters',filename_list)],
      #package_data = {'':['*.txt'],}
      )

