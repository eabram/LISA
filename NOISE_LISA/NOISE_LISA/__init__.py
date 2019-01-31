print('Start initializing')

import imports

from .WFE import WFE
from .plotfile import plot_func
from .calc import Noise, TDI
from .AIM import AIM

# Read parameters
import parameters
para = parameters.__dict__
for k in para:
    globals()[k] = para[k]
print('Done')



