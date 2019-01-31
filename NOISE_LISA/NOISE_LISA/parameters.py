import numpy as np
import os

year2sec=32536000
day2sec=year2sec/365.25
c=300000000

#Parameters:
labda =1065*(10**-9) # m
eta_opt = 0.23 # ...look up
eta_pd = 0.68 # A/W
P_L = 1 # W ...verify with new LISA technical speifications
D = 0.20 # Diameter [m]
MAGNIFICATION = 80 # Check value in Technote

nu_0 = c/labda

h = 1.0/(6.241506*(10**18))

#Parameters for point PAAM
PAA_out_lim = 0.5*0.000001
PAA_out_marge = 0.1*0.000001

D = 0.240 # blz. 12 Optical quality criterion of a truncated laser beam for extremely long distance heterodyne interferometry
gamma_0 = 1.12 # blz. 12 
L_tele = 1 # meter telecope length
E0 = 1 #...Adjust
FOV = np.float64(8e-6) #Field of View telescope
#Calculations
w0_laser = D/(2*gamma_0) # blz. 12
k = (2*np.pi)/labda

home_run = os.getcwd()


#def do_para():
#    para = {}
#
#    para['year2sec']=32536000
#    para['day2sec']=para['year2sec']/365.25
#    para['c']=300000000
#
#    # Parameters:
#    para['labda'] =1065*(10**-9) # m
#    para['eta_opt'] = 0.23 # ...look up
#    para['eta_pd'] = 0.68 # A/W
#    para['P_L'] = 1 # W ...verufy with new LISA technical speifications
#    para['D'] = 0.20 # Diameter [m]
#    para['MAGNIFICATION'] = 5
#
#    para['nu_0'] = para['c']/para['labda']
#    #h = 6.62607004*(10**-34) # Js
#    #c=300000000
#    #h = 1.98644568*(10**-25) # J/m
#    #c = 1
#    para['h'] = 1.0/(6.241506*(10**18))
#
#    # Parameters for point PAAM
#    para['PAA_out_lim'] = 0.5*0.000001
#    para['PAA_out_marge'] = 0.1*0.000001
#
#    para['D'] = 0.240 # blz. 12 Optical quality criterion of a truncated laser beam for extremely long distance heterodyne interferometry
#    para['gamma_0'] = 1.12 # blz. 12 
#    para['L_tele'] = 1 # meter telecope length
#    para['E0'] = 1 #...Adjust
#
#    # Calculation
#    para['w0'] = para['D']/(2*para['gamma_0']) # blz. 12
#    para['k'] = (2*np.pi)/para['labda']
#
#
#    return para
#
#def do_obtain_var(dic,output=False):
#    for keys in dic.keys():
#        print(keys)
#        globals()[keys] = dic[keys]
#        if output!=False:
#            if output==True:
#                setattr(type(dic[keys]),keys,dic[keys])
#
#
#
#
