import numpy as np
import matplotlib.pyplot as plt
import NOISE_LISA
import PAA_LISA
param = NOISE_LISA.parameters.__dict__
for k in param.keys():
    globals()[k] = param[k]

wfe = NOISE_LISA.WFE(orbit='Hallion',duration=400)

# Calculations
wfe.get_pointing(PAAM_method='full control',tele_method='no control',iteration=1)
plot_func = NOISE_LISA.plot_func(wfe)
