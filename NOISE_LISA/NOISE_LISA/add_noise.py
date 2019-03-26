from imports import *
import matplotlib.pyplot
import PAA_LISA
import NOISE_LISA
import control

def tele_param(dz_dis=False):
    m=100.0
    c=0.1
    k=0.1
    dz = c/(2*(m*k)**0.5)
    if dz_dis==False:
        C=c
    else:
        C=dz_dis *2*(m*k)**0.5
    oo0 = (k/m)**0.5
    oo1 = oo0*(1-dz_dis**2)**0.5
    print(dz_dis,oo1)
    sys = control.tf([1],[m,C,k])

    return sys


# Angular noise
def response(i,side,aim,dz_dis=0.05,dt=False,t_start=False,t_end=False,component='tele'):
    sys = tele_param(dz_dis=dz_dis)
    if t_start==False:
        t_start = aim.wfe.t_all[3]
    else:
        t_start = t_start - 24*3600
    if t_end==False:
        t_end = aim.wfe.t_all[-4]
    if dt==False:
        dt = aim.wfe.t_all[1]-aim.wfe.t_all[0]
    
    T = np.linspace(t_start,t_end,int((t_end-t_start)/dt)+1)
    if side=='l':
        if component=='tele':
            func0 = lambda t: aim.tele_l_ang(i,t)
        elif component=='PAAM':
            func0 = lambda t: aim.beam_l_ang(i,t)
    elif side=='r':
        if component=='tele':
            func0 = lambda t: aim.tele_r_ang(i,t)
        elif component=='PAAM':
            func0 = lambda t: aim.beam_r_ang(i,t)
    U=[]
    for t in T:
        U.append(func0(t))
        print(U[-1])
    U=np.array(U)
    #U = np.array([func0(t) for t in T])
    
    print('Start calculsting response')
    out = control.forced_response(sys,T=T,U=U,X0=U[0])
    print('Done\n\n')
    
    lim = int((3600*24.0)/dt)
    out_x = out[0][lim:-1]
    out_y = out[1][lim:-1]
    
    f = NOISE_LISA.functions.interpolate(out_x,out_y)

    return func0,f,out_x,out_y

def aim_noise(i,side,aim,componenet,offset=1e-6,dz_dis=0.05,dt=False,t_start=False,t_end=False):
    PSD = lambda f:400
    f0=1e-6
    f_max=1e-3
    N=4096
    func0,f,out_x,out_y = response(i,side,aim,dz_dis=dz_dis,dt=dt,t_start=t_start,t_end=t_end,component=componenet)
    noise = NOISE_LISA.calc.Noise(aim)
    func_noise = noise.Noise_time(f0,f_max,N,PSD,out_x[-1])[1]
    
    ret = lambda t: func_noise(t)*offset+f(t)

    return ret,out_x,out_y


# PAAM 
def get_static():
    std_x = 1e-3 #...source Peijnenburg article
    std_y = 50e-6

    Dx={}
    Dy={}

    for i in range(1,4):
        Dx[i]={}
        Dy[i]={}
        for s in ['l','r']:
            Dx[i][s] = np.random.normal(loc=0,scale=std_x) 
            Dy[i][s] = np.random.normal(loc=0,scale=std_y)

    return Dx, Dy

def get_PAAM_jitter(aim):
    t_plot =aim.wfe.t_all[4:-4]
    aim.Dx_stat,aim.Dy_stat = get_static()
    n = lambda f: (1+0.0028/f)**2 # noise shape function

    try:
        aim.noise
    except AttributeError:
        aim.noise = NOISE_LISA.Noise(aim)

    f0=1.0e-6
    f_max=1.0e-3
    N=4096
    angular_jitter_all = {}
    long_jitter_all = {}
    rotax_jitter_all = {}
    
    PSD_ang_jit = [lambda f: 8.0*n(f),1e-9]
    PSD_long_jit = [lambda f: 0.28*n(f),1e-9]
    PSD_rotax_jit = [lambda f: 0.30*n(f),1e-9]

    

    for i in range(1,4):
        angular_jitter_all[i] ={}
        long_jitter_all[i]={}
        rotax_jitter_all[i] = {}

        for s in ['l','r']:
            [x,y],func = aim.noise.Noise_time(f0,f_max,N,PSD_ang_jit[0],t_plot[-1])
            angular_jitter = NOISE_LISA.functions.interpolate(x,np.real(y)*PSD_ang_jit[1])

            [x,y],func = aim.noise.Noise_time(f0,f_max,N,PSD_long_jit[0],t_plot[-1])
            long_jitter = NOISE_LISA.functions.interpolate(x,np.real(y)*PSD_long_jit[1])

            [x,y],func = aim.noise.Noise_time(f0,f_max,N,PSD_rotax_jit[0],t_plot[-1])
            rotax_jitter = NOISE_LISA.functions.interpolate(x,np.real(y)*PSD_rotax_jit[1])

            angular_jitter_all[i][s] = angular_jitter
            long_jitter_all[i][s] = long_jitter
            rotax_jitter_all[i][s] = rotax_jitter

    aim.noise.angular_jitter = angular_jitter_all
    aim.noise.long_jitter = long_jitter_all
    aim.noise.rotax_jitter = rotax_jitter_all

    return 0

def get_OPD(aim,beam_ang=False):

    if beam_ang==False:
        beam_l_ang = aim.beam_l_ang
        beam_r_ang = aim.beam_r_ang
    else:
        [beam_l_ang,beam_r_ang] = beam_ang
    
    Dx_all={}
    Dy_all={}
    alpha_all={}
    OPD_all={}
    for s in ['l','r']:
        Dx = lambda i,t: aim.Dx_stat[i][s](t) +aim.noise.long_jitter[i][s](t)+aim.noise.rotax_jitter[i][s](t)
        Dy = lambda i,t: aim.Dy_stat[i][s](t)
        if s=='l':
            alpha = lambda i,t: beam_l_ang(i,t) + aim.noise.angular_jitter[i][s](t)
        elif s=='r':
            alpha = lambda i,t: beam_r_ang(i,t) + aim.noise.angular_jitter[i][s](t)

        OPD = lambda i,t: (Dy(i,t)-Dx(i,t)*np.tan(alpha(i,t)))*(np.sin(alpha(i,t))/np.sin(np.radians(135)))*(1-np.cos(np.radians(90)-2*alpha(i,t)))

        Dx_all[s] = Dx
        Dy_all[s] = Dy
        alpha_all[s] = alpha
        OPD_all[s] = OPD

    aim.noise.Dx = Dx_all
    aim.noise.Dy = Dy_all
    aim.noise.OPD = OPD_all
    aim.noise.alpha = alpha_all

    return 0

def get_jittered_aim(aim,dt_tele=False,dt_PAAM = False):
    aim_new = NOISE_LISA.AIM(wfe=False)
    aim_new.wfe = aim.wfe
    aim_new.tele_method = aim.tele_method
    aim_new.PAAM_method = aim.PAAM_method

    if dt_PAAM==False:
        dt_PAAM = aim.wfe.t_all[1]-aim.wfe.t_all[0]
    if dt_tele==False:
        if 'SS' in aim_new.tele_method:
            dt_tele=100
        else:
            dt_tele = aim.wfe.t_all[1]-aim.wfe.t_all[0]


    tele_l_ang = {}
    tele_r_ang = {}
    beam_l_ang = {}
    beam_r_ang = {}
    for i in range(1,4):
        tele_l_ang[i]= response(i,'l',aim,dt=dt_tele,component='tele')[1]
        tele_r_ang[i]= response(i,'r',aim,dt=dt_tele,component='tele')[1]
        beam_l_ang[i]= response(i,'l',aim,dt=dt_PAAM,component='PAAM')[1]
        beam_r_ang[i]= response(i,'r',aim,dt=dt_PAAM,component='PAAM')[1]

    aim_new.tele_l_ang = lambda i,t: tele_l_ang[i](t)
    aim_new.tele_r_ang = lambda i,t: tele_r_ang[i](t)
    #aim_new.beam_l_ang = lambda i,t: beam_l_ang[i](t)
    #aim_new.beam_r_ang = lambda i,t: beam_r_ang[i](t)
    
    beam_ang = [lambda i,t: beam_l_ang[i](t),lambda i,t: beam_r_ang[i](t)]

    try:
        del aim_new.noise
    except:
        pass
    NOISE_LISA.add_noise.get_PAAM_jitter(aim_new)
    NOISE_LISA.add_noise.get_OPD(aim_new,beam_ang)

    aim_new.beam_l_ang = aim_new.noise.alpha['l']
    aim_new.beam_r_ang = aim_new.noise.alpha['r']

    aim_new.offset_tele = aim.offset_tele
    aim_new.get_coordinate_systems(option='self')
    aim_new.tele_option = aim.tele_option
    aim_new.PAAM_option = aim.PAAM_option
    aim_new.iteration = aim.iteration
    
    return aim_new







