from .imports import *
import PAA_LISA
import functions
import calc_values
import plotfile

from .WFE import WFE
from .plotfile import plot_func
from .calc import Noise, TDI
from .AIM import AIM

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

import parameters
para = NOISE_LISA.parameters.__dict__
for k in para:
    globals()[k] = para[k]



