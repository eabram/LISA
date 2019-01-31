
import PAA_LISA
import NOISE_LISA
import os

import matplotlib.pyplot as plt
import numpy as np
import random
import os
from fractions import Fraction
import math
import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
#warnings.filterwarnings("error")
import scipy.optimize

year2sec=32536000
day2sec=year2sec/365.25
c=300000000



#
#input_param = {
#        'calc_method': 'Waluschka',
#        'plot_on':False, #If plots will be made
#        'dir_savefig': os.getcwd(), # The directory where the figures will be saved. If False, it will be in the current working directory
#        'noise_check':False,
#        'home':'/home/ester/git/synthlisa/', # Home directory
#        'directory_imp': False,
#        'num_back': 0,
#        'dir_orbits': '/home/ester/git/synthlisa/orbits/', # Folder with orbit files
#        'length_calc': 20, # Length of number of imported datapoints of orbit files. 'all' is also possible
#        'dir_extr': 'zzzWaluschka_no_abberation', # This will be added to the folder name of the figures
#        'timeunit':'Default', # The timeunit of the plots (['minutes'],['days']['years'])
#        'LISA_opt':True, # If a LISA object from syntheticLISA will be used for further calculations (not sure if it works properly if this False)
#        'arm_influence': True, # Set True to consider the travel time of the photons when calculating the nominal armlengths
#        'tstep':False,
#        'delay':True, #'Not ahead' or False
#        'method':'fsolve', # Method used to solve the equation for the photon traveling time
#        'valorfunc':'Function', #
#        'select':'Hallion', # Select which orbit files will be imported ('all' is all)
#        'test_calc':False,
#        'abberation':False,
#        'delay': True
#        }
#
#data_all = PAA_LISA.runfile.do_run(input_param)
#
#for k in range(0,len(data_all)/2):
#    #data = PAA_res[str(k+1)]
#    data = data_all[str(k+1)]
#t_vec = data.t_all
#t_plot = np.linspace(t_vec[0],t_vec[-1],len(t_vec)*10)
#
#LA = PAA_LISA.la()
