from synthlisa import *
import numpy as np
import matplotlib.pyplot as plt
import os
from fractions import Fraction
import math
import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize
from sympy import *
from imports import *
#import NOISE_LISA

year2sec=32536000
day2sec=year2sec/365.25
c=300000000

class la():
    def norm(OBJ,v):
        return np.linalg.norm(v)

    def unit(OBJ,v):
        if OBJ.norm(v)==0:
            return v #...adjust
            raise ValueError
        else:
            return v/OBJ.norm(v)

    def angle(OBJ,v1,v2,dot=False):
        norm_v1 = OBJ.norm(v1)
        norm_v2 = OBJ.norm(v2)
        if norm_v1!=0 and norm_v2!=0:
            if dot==False:
                sin = OBJ.norm(np.cross(v1,v2)/(norm_v1*norm_v2))
                return np.arcsin(sin)
            elif dot == True:
                cos = np.dot(v1,v2)/(norm_v1*norm_v2)
                return np.sign(np.dot(v1,v2))*np.arccos(cos)
        else:
            #print('norm v1: '+str(norm_v1))
            #print('norm v2: '+str(norm_v2))

            return np.nan
    def abberation(OBJ,thetas,v):
        return (np.cos(thetas) - v/c)/(1-(v/c)*np.cos(thetas))

    def inplane(OBJ,v,n):
        inplane_calc = v - (np.dot(v,n)/(np.linalg.norm(n)**2))*n
        return inplane_calc

    def outplane(OBJ,v,n):
        outplane_calc = (np.dot(v,n)/(np.linalg.norm(n)**2))*n
        return outplane_calc

    def ang_out(OBJ,v,n):
        sign = np.sign(np.dot(OBJ.outplane(v,n),n))
        return sign*OBJ.angle(OBJ.inplane(v,n),v)

    def ang_in(OBJ,v,n,r):
        #ang_out_calc = OBJ.ang_out(v,n)
        inplane_calc = OBJ.inplane(v,n)
        ang_in_calc = OBJ.angle(inplane_calc,r)

        return ang_in_calc

    def ang_in_dot(OBJ,v,v_stat,n,r):
        inplane_calc = OBJ.inplane(v,n)
        costheta_beam = np.dot(inplane_calc,r)/(OBJ.norm(inplane_calc)*OBJ.norm(r))
        inplane_calc = OBJ.inplane(v_stat,n)
        costheta_sc = np.dot(inplane_calc,r)/(OBJ.norm(inplane_calc)*OBJ.norm(r))

        return np.abs(np.arccos(costheta_beam)) - np.abs(np.arccos(costheta_sc))


    def ang_in_direct(OBJ,v,v_stat,n,r):
        inplane_calc = OBJ.inplane(v,n)
        inplane_stat = OBJ.inplane(v_stat,n)
        ang_out_calc = OBJ.angle(inplane_calc,inplane_stat)
        #ang1 = OBJ.angle(inplane_calc,r)
        #ang2 = OBJ.angle(inplane_stat,r)
        #sign = 1#np.sign(ang1 - ang2)

        return ang_out_calc#ang1-ang2

    def print_component(OBJ,v,v_in,v_out,v_arm):
        n = OBJ.norm(v)
        n_in = OBJ.norm(v_in)
        n_out = OBJ.norm(v_out)
        n_arm = OBJ.norm(v_arm)

        print(n_in/n)
        print((n_out**2+n_in**2+n_arm**2)/n**2)
        print('')

        return 0

    def ang_in_out(OBJ,v1,v2,n,r,give='all'):
        n = OBJ.unit(n)
        v1_out = (np.dot(v1,n)*n)/(OBJ.norm(n)**2)
        #v1_arm = (np.dot(v1,v_stat)*v_stat)/(OBJ.norm(v_stat)**2)
        #v1_in = v1 - v1_out - v1_arm

        v2_out = (np.dot(v2,n)*n)/(OBJ.norm(n)**2)
        #v2_arm = (np.dot(v2,v_stat)*v_stat)/(OBJ.norm(v_stat)**2)
        #v2_in = v2 - v2_out - v2_arm

        ang_out_1 = np.arcsin(OBJ.norm(v1_out)/OBJ.norm(v1))
        ang_out_1 = ang_out_1 * np.sign(np.dot(v1_out,n))
        ang_out_2 = np.arcsin(OBJ.norm(v2_out)/OBJ.norm(v2))
        ang_out_2 = ang_out_2 * np.sign(np.dot(v2_out,n))
        #ang_out = ang_out_1 - ang_out_2

        #q = np.cross(n,v_stat)
        #q = OBJ.unit(q)

        v1_in = v1 - v1_out
        v2_in = v2 - v2_out

        ang_in_1 = OBJ.angle(v1_in,r)
        ang_in_2 = OBJ.angle(v2_in,r)
        ang_in = ang_in_1 - ang_in_2
        ang_out = ang_out_1 - ang_out_2
        #ang_in = np.arcsin(np.linalg.norm(np.cross(v1_in,v2_in))/(np.linalg.norm(v1_in)*np.linalg.norm(v2_in)))


        #v1_in_calc = v1 - v1_out
        #v2_in_calc = v2 - v2_out
        #ang_in_1 = np.arcsin(OBJ.norm(np.cross(v1_in_calc,r))/(OBJ.norm(v1_in_calc)*OBJ.norm(r)))
        #ang_in_2 = np.arcsin(OBJ.norm(np.cross(v2_in_calc,r))/(OBJ.norm(v2_in_calc)*OBJ.norm(r)))


        #ang_in = OBJ.norm(np.cross(v1_in_calc,v2_in_calc))/(OBJ.norm(v1_in_calc)*OBJ.norm(v2_in_calc))
        #ang_in_1 = OBJ.norm(np.cross(v1_in_calc,v_stat))/(OBJ.norm(v1_in_calc)*OBJ.norm(v_stat))
        #ang_in_2 = OBJ.norm(np.cross(v2_in_calc,v_stat))/(OBJ.norm(v2_in_calc)*OBJ.norm(v_stat))

        #ang_in = abs(ang_in_1)+abs(ang_in_2)


        if give=='all':
            return [ang_in,ang_out]
        elif give=='in':
            return ang_in
        elif give=='out':
            return ang_out


    def rotate(OBJ, v,n,ang,mag=False):
        R = np.empty((3,3))
        c=np.cos(ang)
        s = np.sin(ang)
        [x,y,z]=n
        R[0,0] = c+(x**2)*(1-c)
        R[0,1] = x*y*(1-c) - z*s
        R[0,2] = x*z*(1-c)+y*s
        R[1,0] = y*x*(1-c) + z*s
        R[1,1] = c+(y**2)*(1-c)
        R[1,2] = y*z*(1-c)-x*s
        R[2,0] = z*x*(1-c) - y*s
        R[2,1] = z*y*(1-c) + x*s
        R[2,2] = c + (z**2)*(1-c)

        ret = np.dot(R,v)

        return ret

    def matmul(OBJ,A,v): #Matrix multiplication

        return np.array([np.dot(A[0],v),np.dot(A[1],v),np.dot(A[2],v)])

    def beam_coor(OBJ,beam,tele,ntele):
        # Converts beam coordinates from position to procetion on telescope
        #...check if tele is indeed a unit vector
        beam = -beam 
        zbeam = np.dot(beam,OBJ.unit(tele))*OBJ.unit(tele)
        ybeam = OBJ.outplane(beam-zbeam,ntele)#...adjusted with unit 13-12-2018
        xbeam = beam-zbeam-ybeam
      
        return np.array([np.linalg.norm(xbeam),np.linalg.norm(ybeam),np.linalg.norm(zbeam)])
     
    def tele_coor(OBJ,tele_pos,beam,nbeam):
        # Converts tele coordinates from position beam components
        #...check if tele is indeed a unit vector
        ztele = np.dot(tele_pos,OBJ.unit(beam))*OBJ.unit(beam)
        ytele = OBJ.outplane(tele_pos - ztele,nbeam)#...adjusted with unit 13-12-2018
        xtele = tele_pos-ztele-ytele
      
        return np.array([np.linalg.norm(xtele),np.linalg.norm(ytele),np.linalg.norm(ztele)])
    

    def beam_ang(OBJ,beam,tele,ntele):
        beam_conv = OBJ.beam_coor(beam,OBJ.unit(tele),ntele)
        mag = np.linalg.norm(beam_conv)
        ang_x = np.sin(abs(beam_conv[0])/mag)
        ang_y = np.sin(abs(beam_conv[1])/mag)

        return [ang_x,ang_y]

    def flatten(OBJ,y):
        ynew=[]
        check=True
        try:
            len(y)
        except TypeError:
            ynew = [y]
            check=False
            pass
        
        if check==True:
            for i in range(0,len(y)):
                try:
                    for j in range(0,len(y[i])):
                        ynew.append(y[i][j])
                except TypeError:
                    ynew.append(y[i])
            
        return ynew


def nominal_arm(OBJ,i,t):
    L_vec=[]
    t_vec=OBJ.orbit.t
    for j in range(0,len(t_vec)):
        L_vec.append(np.linalg.norm(orbit.L[i-1][j]))

    f=interp1d(t_vec,L_vec,bounds_error)

    return f(t)

def LISA_obj(OBJ,type_select='cache'):
    func_nominal_arm = lambda i,time: nominal_arm(OBJ,i,time)
    lisa=OBJ.orbit.lisa_obj
    lisa_orb=PyLISA(lisa,func_nominal_arm)
    lisa_cache=CacheLISA(lisa_orb)

    OBJ.t_all = OBJ.orbit.t
    if type(type_select)==str:
        if type_select=='cache':
            OBJ.LISA = lisa_cache
        elif type_select=='Py':
            OBJ.LISA = lisa_orb
        elif type_select=='other':
            OBJ.LISA = lisa
    else:
        OBJ.LISA=type_select
    return 0

def i_slr(i,side='all'):
    '''Obtaining the correct spacecraft numbers'''

    i_OBJ = i
    i_left = (i+1)%3
    i_right = (i+2)%3

    i_ret = [i_OBJ,i_left,i_right]
    for j in range(0,len(i_ret)):
        if i_ret[j]==0:
            i_ret[j]=3

    if side=='all':
        return i_ret
    elif side=='l':
        return [i_ret[0],i_ret[1]]
    elif side=='r':
        return [i_ret[0],i_ret[2]]

def get_link(i,side):
    if side=='l':
        link=(i+2)%3
    if side=='r':
        link=(i+1)%3

    if link==0:
        link=3

    return link

def get_armvec_func(OBJ,i,side):
    [i_OBJ,i_next] = i_slr(i,side=side)
    arm_vec = lambda time: np.array(OBJ.LISA.putp(i_next,time)) - np.array(OBJ.LISA.putp(i_OBJ,time))

    return arm_vec


def solve_num(func,guess,method='fsolve'):
    '''Select method for numerically solving an equation'''
    if method == 'fsolve':
        guess = np.array(guess)
        ret = scipy.optimize.fsolve(func,guess)
    elif method == 'excitingmixing':
        ret = scipy.optimize.excitingmixing(func,guess)
    elif method == 'linearmixing':
        ret = scipy.optimize.linearmixing(func,guess)
    elif method == 'newton_krylov':
        guess = [guess]
        ret = scipy.optimize.newton_krylov(func,guess)
    elif method == 'anderson':
        ret = scipy.optimize.anderson(func,guess)
    elif method == 'broyden1':
        ret = scipy.optimize.broyden1(func,guess)

    return ret

def func_pos(OBJ,i):
    '''Generate functions of the positions'''
    L = lambda time: np.array(OBJ.LISA.putp(i,time))
    return L

def solve_L_PAA(OBJ,t,pos_OBJ,pos_left,pos_right,select='sl',calc_method='Waluschka'):
    if OBJ.LISA==False:
        t_guess = np.linalg.norm(OBJ.orbit.p[0][0,:] - OBJ.orbit.p[1][0,:])/c
    else:
        t_guess = np.linalg.norm(np.array(OBJ.LISA.putp(1,0)) - np.array(OBJ.LISA.putp(2,0)))/c

    if select=='sl' or select=='rl':
        s1 = lambda x: pos_left(x)
    elif select=='sr' or select=='rr':
        s1 = lambda x: pos_right(x)

    s2 = lambda x: pos_OBJ(x)
    x_0 = t
    if select=='sl' or select=='sr':
        if calc_method=='Abram':
            s3 = lambda dt: s1(x_0+dt) - s2(x_0)
        else:
            s3 = lambda dt: s1(x_0+dt) - s2(x_0+dt)
    elif select=='rl' or select=='rr':
        if calc_method=='Abram':
            s3 = lambda dt: -s1(x_0-dt) + s2(x_0)
        else:
            s3 = lambda dt: -s1(x_0-dt) + s2(x_0-dt)
    s4 = lambda dt: np.linalg.norm(s3(dt))
    s5 = lambda dt: s4(dt) - c*dt

    res = scipy.optimize.brentq(s5,0,t_guess*4)

    return res




def L_PAA(OBJ,pos_OBJ,pos_left,pos_right,calc_method='Walushka'):
    '''Obtain time of flight of beam between spacecrafts'''

    selections = ['sl','sr','rl','rr']

    L_sl_func =  lambda time: solve_L_PAA(OBJ,time,pos_OBJ,pos_left,pos_right,select=selections[0],calc_method=calc_method)
    L_sr_func =  lambda time: solve_L_PAA(OBJ,time,pos_OBJ,pos_left,pos_right,select=selections[1],calc_method=calc_method)
    L_rl_func =  lambda time: solve_L_PAA(OBJ,time,pos_OBJ,pos_left,pos_right,select=selections[2],calc_method=calc_method)
    L_rr_func =  lambda time: solve_L_PAA(OBJ,time,pos_OBJ,pos_left,pos_right,select=selections[3],calc_method=calc_method)

    return [L_sl_func,L_sr_func,L_rl_func,L_rr_func]

def n_r_lisa(i,time,LISA,m=[2,2,2],ret='all'):
    '''Obtaining normal, r and COM vectors'''
    [i_OBJ,i_left,i_right] = i_slr(i)

    v_l = np.array(LISA.putp(i_left,time)) - np.array(LISA.putp(i_OBJ,time))
    v_r = np.array(LISA.putp(i_right,time)) - np.array(LISA.putp(i_OBJ,time))
    COM = (m[i_left-1]*np.array(LISA.putp(i_left,time)) + m[i_right-1]*np.array(LISA.putp(i_right,time)) + m[i_OBJ-1]*np.array(LISA.putp(i_OBJ,time)))/sum(m)

    r = COM(time) - np.array(LISA.putp(i_OBJ,time))

    n = np.cross(v_l(time)/np.linalg.norm(v_l(time)),v_r(time)/np.linalg.norm(v_r(time)))

    if ret=='all':
        return [n,r]
    elif ret=='n':
        return n
    elif ret=='r':
        return r

def r_calc(v_l,v_r,i,m=[2,2,2]):

    [i_OBJ,i_left,i_right] = i_slr(i)
    r =  (v_l*m[i_left-1]+v_r*m[i_right-1])/(m[i_left-1]+m[i_right-1])

    return r

def func_over_sc(func_tot):
    f = lambda i,t: func_tot[i-1](t)

    return f

def send_func(OBJ,i,calc_method='Waluschka',print_on=False):
    if print_on==True:
        print('Selected calculation method is: '+ calc_method)
        print('')
    LA = la()
    [i_OBJ,i_left,i_right] = i_slr(i)

    pos_left = func_pos(OBJ,i_left)
    pos_OBJ = func_pos(OBJ,i_OBJ)
    pos_right = func_pos(OBJ,i_right)

    if OBJ.delay==True:
        [L_sl,L_sr,L_rl,L_rr] = L_PAA(OBJ,pos_OBJ,pos_left,pos_right,calc_method=calc_method)
    elif OBJ.delay=='not ahead':
        L_sl = lambda t: np.linalg.norm(pos_left(t) - pos_OBJ(t))/c #func_arm_sec(orbit,i_OBJ,'l',LISA=LISA)
        L_sr = lambda t: np.linalg.norm(pos_right(t) - pos_OBJ(t))/c #func_arm_sec(orbit,i_OBJ,'r',LISA=LISA)
        L_rl=L_sl
        L_rr=L_sr

    elif OBJ.delay=='constant':
        L_sl = lambda t: 2500000000/c
        L_sr = lambda t: 2500000000/c
        L_rl=L_sl
        L_rr=L_sr


    elif OBJ.delay==False:
        L_sl = lambda t: 0
        L_sr = lambda t: 0
        L_rl=L_sl
        L_rr=L_sr

    if calc_method=='Abram':
        #print('Selected method is: '+method)
        #print('')
        #Abram2018
        v_send_l_calc = lambda t: pos_left(t+L_sl(t)) - pos_OBJ(t)
        v_send_l = lambda t: (L_sl(t)*c*v_send_l_calc(t))/(np.linalg.norm(v_send_l_calc(t)))
        v_send_r_calc = lambda t: pos_right(t+L_sr(t)) - pos_OBJ(t)
        v_send_r = lambda t: (L_sr(t)*c*v_send_r_calc(t))/(np.linalg.norm(v_send_r_calc(t)))

        v_rec_l0 = lambda t: pos_OBJ(t) - pos_left(t - L_rl(t))
        v_rec_r0 = lambda t: pos_OBJ(t) - pos_right(t - L_rr(t))
        if OBJ.abb==False:
            v_rec_l = v_rec_l0
            v_rec_r = v_rec_r0
        elif OBJ.abb==True:
            v_rec_l = lambda t: relativistic_aberrations(OBJ,i,t,L_rl(t),'l',relativistic=OBJ.relativistic)
            v_rec_r = lambda t: relativistic_aberrations(OBJ,i,t,L_rr(t),'r',relativistic=OBJ.relativistic) 
    elif calc_method=='Waluschka':
        #Waluschka2003
        v_send_l = lambda t: pos_left(t+L_sl(t)) - pos_OBJ(t+L_sl(t))
        v_send_r = lambda t: pos_right(t+L_sr(t)) - pos_OBJ(t+L_sr(t))
        v_rec_l0 = lambda t: pos_OBJ(t-L_rl(t)) - pos_left(t - L_rl(t))
        v_rec_r0 = lambda t: pos_OBJ(t-L_rr(t)) - pos_right(t - L_rr(t))
        if OBJ.abb==False:
            v_rec_l = v_rec_l0
            v_rec_r = v_rec_r0
        elif OBJ.abb==True:
            v_rec_l = lambda t: np.linalg.norm(v_rec_l0(t))*(LA.unit(LA.unit(v_rec_l0(t-L_rl(t)))*OBJ.c+(OBJ.v_abs(i_OBJ,t-L_rl(t)) - OBJ.v_abs(i_left,t-L_rl(t)))))
            v_rec_r = lambda t: np.linalg.norm(v_rec_r0(t))*(LA.unit(LA.unit(v_rec_r0(t-L_rr(t)))*OBJ.c+(OBJ.v_abs(i_OBJ,t-L_rr(t)) - OBJ.v_abs(i_right,t-L_rr(t)))))
    #if OBJ.abb==True:
        v_rec_l = lambda t: relativistic_aberrations(OBJ,i,t,L_rl(t),'l',relativistic=OBJ.relativistic)
        v_rec_r = lambda t: relativistic_aberrations(OBJ,i,t,L_rr(t),'r',relativistic=OBJ.relativistic)          
    #return [[v_send_l,v_send_r],[L_sl,L_sr,L_rl,L_rr]]
    return [[v_send_l,v_send_r,v_rec_l,v_rec_r],[L_sl,L_sr,L_rl,L_rr],[v_rec_l0,v_rec_r0]]



def relativistic_aberrations(OBJ,i,t,tdel,side,relativistic=True):
    LA=la()
    [i_self,i_left,i_right] = i_slr(i)
    if OBJ.calc_method=='Abram':
        tdel0=0
    elif OBJ.calc_method=='Waluschka':
        tdel0 = tdel

    if side=='l':
        u_not_ab = np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_left,t-tdel))
        u_ab = np.linalg.norm(u_not_ab)*(LA.unit(LA.unit(u_not_ab)*OBJ.c+(OBJ.v_abs(i_self,t-tdel0) - OBJ.v_abs(i_left,t-tdel))))

    elif side=='r':
        u_not_ab = np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_right,t-tdel))
        u_ab = np.linalg.norm(u_not_ab)*(LA.unit(LA.unit(u_not_ab)*OBJ.c+(OBJ.v_abs(i_self,t-tdel0) - OBJ.v_abs(i_right,t-tdel))))
 
    if relativistic==False:
        return u_ab
    
    elif relativistic==True:
        LA=la()
        coor = NOISE_LISA.functions.coor_SC(OBJ,i,t)
        if side=='l':
            velo = (OBJ.v_abs(i_self,t-tdel0) - OBJ.v_abs(i_left,t-tdel))
        elif side=='r':
            velo = (OBJ.v_abs(i_self,t-tdel0) - OBJ.v_abs(i_right,t-tdel))

        c_vec = LA.unit(u_not_ab)*c

        r = coor[0]
        x_prime = LA.unit(velo)
        n_prime = LA.unit(np.cross(velo,r))
        r_prime = LA.unit(np.cross(n_prime,x_prime))

        coor_velo = np.array([r_prime,n_prime,x_prime])
        c_velo = LA.matmul(coor_velo,c_vec)
        v = np.linalg.norm(velo)
        den = 1.0 - ((v/(c**2))*coor_velo[2])
        num = ((1.0-((v**2)/(c**2)))**0.5)

        ux_prime = (c_velo[2] - v)/den
        ur_prime = (num*c_velo[0])/den
        un_prime = (num*c_velo[1])/den
        c_prime = ux_prime*x_prime + un_prime*n_prime +ur_prime*r_prime
        u_new=LA.unit(c_prime)*np.linalg.norm(u_not_ab)
   
    elif relativistic=='test':
         
        if side=='l':
            c_send = LA.unit(OBJ.u_l0_func_tot(i_self,t))*np.float64(c)
            velo_send_0 = OBJ.v_abs(i_left,t-tdel)
        elif side=='r':
            c_send = LA.unit(OBJ.u_r0_func_tot(i_self,t))*np.float64(c)
            velo_send_0 = OBJ.v_abs(i_right,t-tdel)
        velo_0_send = -velo_send_0

        c_0 = LA.unit(add_velo(velo_0_send,c_send))*np.float64(c)
        velo_rec_0 = OBJ.v_abs(i_self,t-tdel0)
        c_rec = LA.unit(add_velo(velo_rec_0,c_0))*np.float64(c)

        u_new=LA.unit(c_rec)*np.linalg.norm(u_not_ab)
    
    return u_new

def add_velo(v,u):
    c=np.float64(300000000.0)
    v_mag = np.linalg.norm(v)
    gamma = 1.0/((1-((v/c)**2))**0.5)
    u_prime = (1.0/(1.0-(np.dot(u,v)/(c**2))))*((u/gamma) - v+(1.0/(c**2))*(gamma/(1+gamma))*(np.dot(u,v))*v)

    return u_prime

def get_receiving(OBJ,i,t,side):
    LA = la()
    [i_self,i_left,i_right] = i_slr(i)
    
    if side=='l':
        tdel = OBJ.L_rl_func_tot(i_self,t)
        v_send = OBJ.v_abs(i_left,t-tdel)
    elif side=='r':
        tdel = OBJ.L_rr_func_tot(i_self,t)
        v_send = OBJ.v_abs(i_right,t-tdel)
    
    if OBJ.calc_method=='Abram':
        tdel0=0
    elif OBJ.calc_method=='Waluschka':
        tdel0=tdel
    
    if OBJ.abb==True:
        v_rec = OBJ.v_abs(i_self,t)
        if OBJ.relativistic==True:
            ux = v_rec
            v = v_send
            v_rec_sendframe = add_velo(v,ux)
            if side=='l':
                c_vec_sendframe = LA.unit(OBJ.v_r_func_tot(i_left,t-tdel))*np.float64(c)
            elif side=='r':
                c_vec_sendframe = LA.unit(OBJ.v_l_func_tot(i_right,t-tdel))*np.float64(c)

            c_vec_recframe = add_velo(v_rec_sendframe,c_vec_sendframe)
            return (tdel*np.float64(c)*c_vec_recframe)/(np.linalg.norm(c_vec_recframe))
        elif OBJ.relativistic==False:
            if side=='l':
                c_vec =  np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_left,t-tdel))
            elif side=='r':
                c_vec = np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_right,t-tdel))
            c_vec = (np.float64(c)*c_vec)/(np.linalg.norm(c_vec))
            ret = c_vec + (v_rec-v_send)
            return (tdel*np.float64(c)*ret)/np.linalg.norm(ret)


    elif OBJ.abb==False:
        if side=='l':
            return np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_left,t-tdel))
        elif side=='r':
            return np.array(OBJ.LISA.putp(i_self,t-tdel0)) - np.array(OBJ.LISA.putp(i_right,t-tdel))

    #angle = LA.angle(-c_vec_recframe,wfe.data.v_l_func_tot(i,t))

    #return angle


def calc_PAA_ltot(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_l(i,t),OBJ.v_out_mag_l(i,t),OBJ.v_arm_mag_l(i,t)]
    #else:
    calc_ang=LA.angle(OBJ.v_l_func_tot(i,t),-OBJ.u_l_func_tot(i,t))
    return calc_ang

def calc_PAA_rtot(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_l(i,t),OBJ.v_out_mag_l(i,t),OBJ.v_arm_mag_l(i,t)]
    #else:
    calc_ang=LA.angle(OBJ.v_r_func_tot(i,t),-OBJ.u_r_func_tot(i,t))
    return calc_ang

def calc_PAA_lin(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_l(i,t),OBJ.v_out_mag_l(i,t),OBJ.v_arm_mag_l(i,t)]
    #else:
    calc_ang=LA.ang_in_out(OBJ.v_l_func_tot(i,t),-OBJ.u_l_func_tot(i,t),OBJ.n_func(i,t),OBJ.r_func(i,t),give='in')
    return calc_ang

def calc_PAA_lout(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_l(i,t),OBJ.v_out_mag_l(i,t),OBJ.v_arm_mag_l(i,t)]
    #else:
    calc_ang=LA.ang_in_out(OBJ.v_l_func_tot(i,t),-OBJ.u_l_func_tot(i,t),OBJ.n_func(i,t),OBJ.r_func(i,t),give='out')
    return calc_ang

def calc_PAA_rin(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_r(i,t),OBJ.v_out_mag_r(i,t),OBJ.v_arm_mag_r(i,t)]
    #else:
    calc_ang=LA.ang_in_out(OBJ.v_r_func_tot(i,t),-OBJ.u_r_func_tot(i,t),OBJ.n_func(i,t),OBJ.r_func(i,t),give='in')
    
    return calc_ang

def calc_PAA_rout(OBJ,i,t):
    LA = la()
    #if OBJ.abb:
    #    v_abb = [OBJ.v_in_r(i,t),OBJ.v_out_mag_r(i,t),OBJ.v_arm_mag_r(i,t)]
    #else:
    calc_ang=LA.ang_in_out(OBJ.v_r_func_tot(i,t),-OBJ.u_r_func_tot(i,t),OBJ.n_func(i,t),OBJ.r_func(i,t),give='out')
    
    return calc_ang

#def calc_PAA(OBJ,i,t,side,ab=True):
#    LA = la()
#    coor = NOISE_LISA.functions.coor_SC(OBJ,i,t)
#    if side=='l':
#        if ab==True:
#            beam_in = -(LA.matmul(coor,OBJ.u_l_func_tot(i,t)))
#        elif ab==False:
#            beam_in = -(LA.matmul(coor,OBJ.u_l0_func_tot(i,t)))
#        beam_out = (LA.matmul(coor,OBJ.v_l_func_tot(i,t)))
#    elif side=='r':
#        if ab==True:
#            beam_in = -(LA.matmul(coor,OBJ.u_r_func_tot(i,t)))
#        elif ab==False:
#            beam_in = -(LA.matmul(coor,OBJ.u_r0_func_tot(i,t)))
#        beam_out = (LA.matmul(coor,OBJ.v_r_func_tot(i,t)))
#
#    beam_in_in = (np.array([beam_in[2],beam_in[0]]))
#    beam_in_out = (np.array([beam_in[1],beam_in[0]]))
#    beam_out_in = (np.array([beam_out[2],beam_out[0]]))
#    beam_out_out = (np.array([beam_out[1],beam_out[0]]))
#    #
#    #PAA_in_calc = beam_out_in - beam_in_in
#    #PAA_in = (np.arctan(PAA_in_calc[1]/PAA_in_calc[0]))
#    #
#    #
#    #PAA_out_calc = beam_out_out - beam_in_out
#    #PAA_out = abs(np.arctan(PAA_out_calc[1]/PAA_out_calc[0]))
#
#
#    #PAA_in = np.arccos(np.dot(beam_out_in,beam_in_in)/(np.linalg.norm(beam_out_in)*np.linalg.norm(beam_in_in)))
#    #PAA_out = np.arccos(np.dot(beam_out_out,beam_in_out)/(np.linalg.norm(beam_out_out)*np.linalg.norm(beam_in_out)))
#    
#    PAA_in = LA.angle(beam_out_in,beam_in_in)
#    PAA_out = LA.angle(beam_out_out,beam_in_out)
#    PAA_tot = LA.angle(beam_out,beam_in)
#
#    return [PAA_in,PAA_out,PAA_tot]





# Velocity
def velocity_abs_calc(OBJ,i_select,t,hstep):
    v = (np.array(OBJ.LISA.putp(i_select,np.float64(t+hstep)))-np.array(OBJ.LISA.putp(i_select,t)))/hstep
    #print(i_select)
    #print('')
    return v


def velocity_abs(OBJ,hstep=1.0):
    hstep = np.float128(hstep)

    #v_ret=[]
    #for i_select in range(1,4):
        #print(i_select)
        #v_ret.append(lambda time: velocity_abs_calc(OBJ,i_select,time,hstep))
    v_ret = lambda i,time: velocity_abs_calc(OBJ,i,time,hstep)

    OBJ.v_abs = v_ret

    return OBJ.v_abs
    
def velocity_calc(OBJ,i,time,hstep,side,rs):
    LA=la()

    [i_OBJ,i_next] = i_slr(i,side=side)
    v_pos_l = OBJ.v_l_stat_func_tot(i_OBJ,time)
    v_pos_r = OBJ.v_r_stat_func_tot(i_OBJ,time)
    n = np.cross(v_pos_r,v_pos_l)
    n = LA.unit(n)
    if side=='l':
        v_pos = v_pos_l
    elif side=='r':
        v_pos = v_pos_r 

    pos_OBJ = np.array(OBJ.LISA.putp(i_OBJ,time))
    pos_next = np.array(OBJ.LISA.putp(i_next,time))
    pos_OBJ_h = np.array(OBJ.LISA.putp(i_OBJ,np.float64(time+hstep)))
    pos_next_h = np.array(OBJ.LISA.putp(i_next,time+np.float64(hstep)))
    v = ((pos_next_h-pos_next) - (pos_OBJ_h - pos_OBJ))/hstep

    #v = OBJ.v_abs(i_next,time) - OBJ.v_abs(i_OBJ,time)
    v_out = n*(np.dot(v,n))
    v_arm = v*(np.dot(LA.unit(v),LA.unit(v_pos)))
    v_in = v - v_out - v_arm

    v_out_sign = np.sign(np.dot(v_out,n))
    v_arm_sign = np.sign(np.dot(LA.unit(v),LA.unit(v_pos)))
    v_in_sign = np.sign(np.dot(v_in,v_pos_r - v_pos_l))

    v_out_mag = np.linalg.norm(v_out)*v_out_sign
    v_in_mag = np.linalg.norm(v_in)*v_in_sign
    v_arm_mag = np.linalg.norm(v_arm)*v_arm_sign

    ret =  [v,v_in,v_out,v_arm,v_in_mag,v_out_mag,v_arm_mag]

    return ret[rs]

def velocity_func(OBJ,hstep=1.0):
    LA=la()
    hstep = np.float128(hstep)
    OBJ.v_l= lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',0)
    OBJ.v_in_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',1)
    OBJ.v_out_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',2)
    OBJ.v_arm_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',3)
    OBJ.v_in_mag_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',4)
    OBJ.v_out_mag_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',5)
    OBJ.v_arm_mag_l = lambda i,time: velocity_calc(OBJ,i,time,hstep,'l',6)
    OBJ.v_r= lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',0)
    OBJ.v_in_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',1)
    OBJ.v_out_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',2)
    OBJ.v_arm_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',3)
    OBJ.v_in_mag_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',4)
    OBJ.v_out_mag_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',5)
    OBJ.v_arm_mag_r = lambda i,time: velocity_calc(OBJ,i,time,hstep,'r',6)
    

    return 0

def high_precision(p):
    Y = p
    for i in range(0,len(p)):
        y_p=[]
        for j in range(0,len(p[i][0])):
            y = p[i,:,j]
            y_inv = scipy.fftpack.ifft(y)
            y_new = scipy.fftpack.fft(y_inv)
            Y[i,:,j] = np.real(y_new)

    return Y












