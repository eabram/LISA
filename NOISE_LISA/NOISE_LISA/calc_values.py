from imports import *
from functions import *
from parameters import *
import PAA_LISA
import NOISE_LISA

def make_t_calc(wfe,t0=False,tend=False,dt=False):
    calc=False
    if dt==False:
        try:
            wfe.t_calc
        except AttributeError:
            calc=True
            dt = wfe.t_all[1]-wfe.t_all[0]
            pass
    else:
        calc=True

    if calc==True:
        if t0==False:
            t0 = wfe.t_all[0]
        if tend==False:
            tend = wfe.t_all[-1]
        N = int(np.round((tend-t0)/dt))+1
        t_plot = np.linspace(t0,tend,N)

    elif calc==False:
        t_plot = wfe.t_calc

    return t_plot


def piston(wfe,SC=[1,2,3],side=['l','r'],dt=False,meas='piston',lim=[0,-1],aim='Default'):
    t_vec = make_t_calc(wfe,dt=dt)
    if type(SC)==int:
        SC = [SC]
    if type(side)==str:
        side=[side]
    
    len_short=False
    if type(meas)==str:
        meas = [meas]
        if meas[0]!='all_val':
            len_short=True
    if len(meas)<4 and meas[0]!='all_val':
        size='small'
    else:
        size='big'

    ret={}
    ret['mean']={}
    ret['var']={}
    for i in SC:
        ret['mean'][str(i)]={}
        ret['var'][str(i)]={}
        for s in side:
            mean={}
            var={}
            #calc_all = lambda t: wfe.calc_piston_val(i,t,s,ret=meas,size='big',aim=aim)

            for t in t_vec[lim[0]:lim[-1]]:
                print(t)
                #calc = calc_all(t)
                calc = wfe.calc_piston_val(i,t,s,ret=meas,size=size,aim=aim)
                for m in calc.keys():
                #calc = wfe.calc_piston_val(i,t,s,ret=m)
                    try:
                        mean[m]
                    except KeyError:
                        mean[m] = []
                        var[m] = []
                        
                    if calc[m][-1]=='vec':
                        mean[m].append([t,calc[m][0]])
                        var[m].append([t,np.nan])
                    else:
                        mean[m].append([t,calc[m][0]])
                        var[m].append([t,calc[m][1]])

            ret['mean'][str(i)][s] = mean
            ret['var'][str(i)][s] = var

    ret_sort={}
    for meanvar in ret.keys():
        for SC in ret[meanvar].keys():
            for s in ret[meanvar][SC].keys():
                for m in ret[meanvar][SC][s].keys():
                    try:
                        ret_sort[m]
                    except KeyError:
                        ret_sort[m]={}
                        ret_sort[m]['mean']={}
                        ret_sort[m]['var']={}

                    try:
                        ret_sort[m][meanvar][SC]
                    except KeyError:
                        ret_sort[m][meanvar][SC]={}

                    try:
                        ret_sort[m][meanvar][SC][s]
                    except KeyError:
                        ret_sort[m][meanvar][SC][s]={}

                    ret_sort[m][meanvar][SC][s] = ret[meanvar][SC][s][m]

            #ret['mean'][str(i)][s]=mean
            #ret['var'][str(i)][s]=var


    ret_all={}
    if meas[0]=='all_val':
        meas = calc.keys()

    for m in meas:
        title = 'Title:: Telescope control: '+wfe.tele_control+', PAAM control: '+ wfe.PAAM_control_method
        iteration = 'Iteration:: '+ str(aim.iteration)
        measurement = 'Measurement:: '+m

        ret_all[m] = title, iteration, measurement, ret_sort[m]

    if len_short==True:
        return ret_all[meas[0]]
    else:
        return ret_all
            





def ttl(wfe,tele_control=False,PAAM_control_method=False,simple=True):
    
    if tele_control==False:
        tele_control = wfe.tele_control
    if PAAM_control_method==False:
        PAAM_control_method = wfe.PAAM_control_method

    wfe.get_pointing(PAAM_method = PAAM_control_method,tele_method = tele_control) 
     
    ttl = {}
    wfe.ttl_pointing_function(option='all')
    ttl['pointing_all'] = wfe.ttl_l,wfe.ttl_r
    wfe.ttl_pointing_function(option='tilt')
    ttl['pointing_tilt'] = wfe.ttl_l,wfe.ttl_r
    wfe.ttl_pointing_function(option='piston')
    ttl['pointing_piston'] = wfe.ttl_l,wfe.ttl_r

    ttl_PAAM_l = []
    ttl_PAAM_r = []
    ttl_PAAM = wfe.Ndata.PAAMnoise(wfe=wfe)
    for i in range(1,4):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        keyl = str(i_self)+str(i_left)
        keyr = str(i_self)+str(i_right)
        ttl_PAAM_l.append(ttl_PAAM[keyl])
        ttl_PAAM_r.append(ttl_PAAM[keyr])

    ttl['PAAM'] = PAA_LISA.utils.func_over_sc(ttl_PAAM_l),PAA_LISA.utils.func_over_sc(ttl_PAAM_r)

    wfe.ttl_val = ttl
    
    return ttl

def ang(wfe,iteration=[0,1]): 
    tele_l={}
    PAAM_l={}

    tele_r={}
    PAAM_r={}

    PAAM_methods=['no control','full control','SS']
    tele_methods = PAAM_methods

    for i in iteration:
        ikey=str(i)
        tele_l[ikey]={}
        PAAM_l[ikey]={}
        tele_r[ikey]={}
        PAAM_r[ikey]={}

        for tele_method in tele_methods:
            tele_l[ikey][tele_method]={}
            PAAM_l[ikey][tele_method]={}
            tele_r[ikey][tele_method]={}
            PAAM_r[ikey][tele_method]={}
            
            for PAAM_method in PAAM_methods:
                key=tele_method+', '+PAAM_method
                wfe.get_pointing(PAAM_method=PAAM_method,tele_method=tele_method,iteration=i)
                tele_l[ikey][tele_method][PAAM_method] = wfe.aim.tele_l_ang
                PAAM_l[ikey][tele_method][PAAM_method] = wfe.aim.beam_l_ang
                tele_r[ikey][tele_method][PAAM_method] = wfe.aim.tele_r_ang
                PAAM_r[ikey][tele_method][PAAM_method] = wfe.aim.beam_r_ang
    
    return tele_l,PAAM_l,tele_r,PAAM_r
            
        
