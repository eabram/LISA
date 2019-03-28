from imports import *
from functions import *
import parameters 
from .AIM import AIM
#import PAA_LISA
#import NOISE_LISA

class WFE(): 
    def __init__(self,**kwargs):
        para = NOISE_LISA.parameters.__dict__
        for k in para:
            globals()[k] = para[k]
            setattr(self,k,para[k])

        self.data = kwargs.pop('data',False)
        adjust={}
        adjust['select']= kwargs.pop('orbit','')
        adjust['dir_orbits']=kwargs.pop('dir_orbits',os.getcwd())
        adjust['home']=kwargs.pop('home',os.getcwd())
        adjust['length_calc']= kwargs.pop('duration',40)
        adjust['dir_savefig']=kwargs.pop('home',home_run+'/Default_folder/')
        adjust['calc_method']=kwargs.pop('calc_method','Waluschka')
        adjust['LISA_opt']=kwargs.pop('LISA_opt','cache')
        self.LISA = adjust['LISA_opt']
        if self.data==False:
            self.get_PAA_LISA(para,adjust=adjust)


        self.tele_control = kwargs.pop('tele_control','no_control')
        self.PAAM_control_method = kwargs.pop('PAAM_control','SS')
        self.side = kwargs.pop('side','l')
        self.speed_on = kwargs.pop('speed_on',0)
        self.simple=True
        self.jitter=[False,False]
        self.tele_aim_done = False
        self.jitter_tele_done = False
        self.tilt_send = False
        self.zmn_func = {}
        self.thmn_func = {}
        self.zmn={}
        self.thmn = {}
        self.t_all = self.data.t_all

        if self.data==False:
            print('Please input PAA_LISA object')
        else:
            LA = PAA_LISA.utils.la()
            self.pupil()
            self.scale=1

        self.t_calc = NOISE_LISA.calc_values.make_t_calc(self)

    def get_PAA_LISA(self,para,adjust={}):
        options = { 
        'calc_method': 'Waluschka',
        'plot_on':False, #If plots will be made
        'dir_savefig': os.getcwd(), # The directory where the figures will be saved. If False, it will be in the current working directory
        'noise_check':False,
        'home':'/home/ester/git/synthlisa/', # Home directory
        'directory_imp': False,
        'num_back': 0,
        'dir_orbits': '/home/ester/git/synthlisa/orbits/', # Folder with orbit files
        'length_calc': 'all', # Length of number of imported datapoints of orbit files. 'all' is also possible
        'dir_extr': 'zzzWaluschka_no_abberation', # This will be added to the folder name of the figures
        'timeunit':'Default', # The timeunit of the plots (['minutes'],['days']['years'])
        'LISA_opt':True, # If a LISA object from syntheticLISA will be used for further calculations (not sure if it works properly if this False)
        'arm_influence': True, # Set True to consider the travel time of the photons when calculating the nominal armlengths
        'tstep':False,
        'delay':True, #'Not ahead' or False
        'method':'fsolve', # Method used to solve the equation for the photon traveling time
        'valorfunc':'Function', #
        'select':'Hallion', # Select which orbit files will be imported ('all' is all)
        'test_calc':False,
        'abberation':False,
        'delay': True
        }

        for k in adjust.keys():
            if k in options.keys():
                print(adjust[k])
                options[k] = adjust[k]

        data_all = PAA_LISA.runfile.do_run(options,para)

        print('Len data_all: '+str(len(data_all)))

        for k in range(0,len(data_all)/2):
            data = data_all[str(k+1)]
        self.t_vec = data.t_all
        self.data = data
        self.status_init_pointing=False


    def get_pointing(self,tele_method = False,PAAM_method=False,iteration=0,tele_ang_extra=True,PAAM_ang_extra=False,init=False,sampled=False,aim_old=False,aim0=False,option_tele='wavefront',option_PAAM='wavefront',count=0,offset_tele='read'): #...add more variables
        
        #if tele_ang_extra==True:
        #    tele_ang_extra = NOISE_LISA.functions.get_extra_ang_mean(self,'tele')
        if PAAM_ang_extra==True:
            PAAM_ang_extra = NOISE_LISA.functions.get_extra_ang_mean(self,'PAAM')


        self.iteration=iteration 
        if tele_method==False:
            tele_method = self.tele_control
        else:
            self.tele_control = tele_method
        
        if PAAM_method==False:
            PAAM_method = self.PAAM_control_method
        else:
            self.PAAM_control_method = PAAM_method

        if aim_old==False:
            tele_l_ang=False
            tele_r_ang=False
            PAAM_l_ang=False
            PAAM_r_ang=False
        else:
            tele_l_ang=aim_old.tele_l_ang
            tele_r_ang=aim_old.tele_r_ang
            PAAM_l_ang=aim_old.beam_l_ang
            PAAM_r_ang=aim_old.beam_r_ang
        angles_old = [tele_l_ang,PAAM_l_ang,tele_r_ang,PAAM_r_ang]

        if aim0==False:
            tele_l_ang0=False
            tele_r_ang0=False
            PAAM_l_ang0=False
            PAAM_r_ang0=False
        else:
            tele_l_ang0=aim0.tele_l_ang
            tele_r_ang0=aim0.tele_r_ang
            PAAM_l_ang0=aim0.beam_l_ang
            PAAM_r_ang0=aim0.beam_r_ang
        angles0 = [tele_l_ang0,PAAM_l_ang0,tele_r_ang0,PAAM_r_ang0]

        aim = AIM(self,init=init,sampled=sampled,angles0=angles0,angles_old=angles_old,count=count,offset_tele=offset_tele)
        aim.tele_aim(self,method=tele_method,iteration=iteration,tele_ang_extra=tele_ang_extra,option=option_tele)
        out = aim.PAAM_control(method=PAAM_method,PAAM_ang_extra=PAAM_ang_extra,option=option_PAAM)
        try:
            aim.tele_option
        except AttributeError:
            aim.tele_option = option_tele
            aim.PAAM_option = option_PAAM
        
        return out

    # Beam properties equations
    
    def get_pointing_with_iteration(self,tele_method = False,PAAM_method=False,iteration=0,tele_ang_extra=False,PAAM_ang_extra=False,option_tele='wavefront',option_PAAM='wavefront',sampled=True):
        
        if PAAM_method!='no_control':
            PAAM_testmode='full_control'
        else:
            PAAM_testmode=PAAM_method
        if tele_method!='no_control':
            tele_testmode='full_control'
        else:
            tele_testmode=tele_method

        try:
            del self.aim
        except AttributeError:
            pass
        
        aim0 = self.get_pointing(tele_method = 'no_control',PAAM_method='no_control',iteration=0,tele_ang_extra=tele_ang_extra,PAAM_ang_extra=PAAM_ang_extra,init=True,sampled=False,option_tele=option_tele,option_PAAM=option_PAAM)
       
        if iteration==0:
            aimnew = self.get_pointing(tele_method = tele_method,PAAM_method=PAAM_method,iteration=0,tele_ang_extra=tele_ang_extra,PAAM_ang_extra=PAAM_ang_extra,aim_old=aim0,aim0=aim0,sampled=sampled,option_tele=option_tele,option_PAAM=option_PAAM)
            aimnew.iteration = 0
        else:
            aimnew = self.get_pointing(tele_method = tele_testmode,PAAM_method=PAAM_testmode,iteration=0,tele_ang_extra=tele_ang_extra,PAAM_ang_extra=PAAM_ang_extra,aim_old=aim0,aim0=aim0,sampled=sampled,option_tele=option_tele,option_PAAM=option_PAAM)
            aimnew.iteration = 0

        if iteration>0:
            for i in range(0,iteration):
                aimold = aimnew
                del aimnew
                if i<iteration-1:
                    aimnew = self.get_pointing(tele_method = tele_testmode,PAAM_method=PAAM_testmode,iteration=0,tele_ang_extra=tele_ang_extra,PAAM_ang_extra=PAAM_ang_extra,aim_old=aimold,aim0=aim0,sampled=True,option_tele=option_tele,option_PAAM=option_PAAM,count=i+1)
                else:
                    aimnew = self.get_pointing(tele_method = tele_method,PAAM_method=PAAM_method,iteration=0,tele_ang_extra=tele_ang_extra,PAAM_ang_extra=PAAM_ang_extra,aim_old=aimold,aim0=aim0,sampled=True,option_tele=option_tele,option_PAAM=option_PAAM,count=i+1)
        
        aimnew.iteration = iteration
        self.aim = aimnew
        
        self.iteration = iteration
        self.aim.iteration = iteration

        return aimnew




    def pupil(self,Nbins=2,**kwargs):
        D_calc=kwargs.pop('D',self.D)


        if D_calc=='Default':
            D = self.para['D'] 
        xlist = np.linspace(-D_calc*0.5,D_calc*0.5,Nbins+1)
        self.xlist = xlist[0:-1]+0.5*(xlist[1]-xlist[0])
        self.ylist = self.xlist
        self.Deltax = self.xlist[1]-self.xlist[0]
        self.Deltay = self.ylist[1]-self.ylist[0]
        self.Nbinsx = len(self.xlist)
        self.Nbinsy = len(self.ylist)

    def w(self,z):
        zR = np.pi*(self.w0_laser**2)/labda

        return self.w0_laser*((1+((z/zR)**2))**0.5)

    def R(self,z,guess=False):
        if z!=np.nan:
            zR = np.pi*(self.w0_laser**2)/labda

            if guess==False:
                return abs(z*(1+((zR/z)**2)))

            elif guess==True:
                return z
        else:
            return np.nan

    def z_solve(self,x,y,z,calc_R=False,ret='piston',R_guess=True):
        try:
            if z!=np.nan:
                x = np.float64(x)
                y = np.float64(y)
                z = np.float64(z)
                #lim = (x**2+y**2) 
                #R_new = lambda dz: self.R(z+dz,guess=False)
                f_solve = lambda dz: (self.R(z+dz,guess=R_guess) - (self.R(z+dz,guess=R_guess)**2 - (x**2+y**2))**0.5) - dz
                f_solve_2 = lambda dz: (z- (((z+dz)**2 - x**2 -y**2 )**0.5))
                dz_sol = scipy.optimize.brentq(f_solve,-0.5*z,0.5*z,xtol=1e-64)
                #dz_sol_3 = scipy.optimize.brentq(f_solve_2,-lim,lim,xtol=1e-64)
                #dz_sol_2 = scipy.optimize.fsolve(f_solve,0,xtol=1e-128)
            else:
                dz_sol=np.nan
                dz_sol_2=np.nan
                dz_sol_3=np.nan
                raise ValueError
            
            if calc_R==True:
                return self.R(z+dz_sol,guess=R_guess)
            else:
                if ret=='piston':
                    return z+dz_sol
                elif ret=='all':
                    #print(dz_sol,1e-64)
                    #print(x,y)
                    #print(dz_sol,dz_sol_2,dz_sol_3)
                    #print(x,y,z)
                    return [z+dz_sol,dz_sol]
            
        except RuntimeError:
            #print(x,y,z)
            if ret=='piston':
                return np.nan
            elif ret=='all':
                return [np.nan,np.nan]


# WFE send
    def WFE_send(self,i,t,side='l',xlist=False,ylist=False): #...adjust wirt real WFE instead of 0
        if self.speed_on==2:
            xlist=np.array([0])
            ylist=np.array([0])
        else:
            if xlist==False:
                xlist = self.xlist
            if ylist == False:
                ylist =self.ylist

        if side=='l':
            angy = self.aim.beam_l_ang(i,t)
        elif side=='r':
            angy = self.aim.beam_r_ang(i,t)
        angx = 0 #...to do: Add jitter
       
        labda = self.data.labda
        function = lambda x,y: 2*np.pi*((x*np.sin(angx)+y*np.sin(angy))/labda)

        w = self.aperture(xlist,ylist,function,dType = np.float64)

        return w

    def w0(self,angx,angy,ksi,aim=False):
        # Opposite wfe
        if aim==False:
            aim=self.aim

        #if side=='l':
        #    angy = self.aim.beam_l_ang(i,t)
        #elif side == 'r':
        #    angy = self.aim.beam_r_ang(i,t)
        angx=0#...add jitter

        [x,y] = ksi
        
        labda = self.data.labda
        w_error = 2*np.pi*((x*np.sin(angx)+y*np.sin(angy))/labda)

        return w_error

    def u0(self,ksi):#...for gaussian beam
        w = self.w(0)
        [x,y] = ksi

        return np.exp(-((x**2+y**2)/(w**2)))

# WFE receive # Used
    
    def u_rz_calc(self,r,z,angx,angy,xlist=False,ylist=False,aim=False):
        if xlist==False:
            xlist = self.xlist
        if ylist==False:
            ylist = self.ylist
        labda = self.data.labda
        k = (2*np.pi)/labda
        
        if len(xlist)==1 and len(ylist)==1:
            dksi = (self.D**2)*(np.pi/4.0)
        else:
            dksi = (xlist[1]-xlist[0])*(ylist[1]-ylist[0])

        ret=0
        for i in range(0,len(xlist)):
            for j in range(0,len(ylist)):
                ksi = np.array([xlist[i],ylist[j]])
                T1 = np.exp((1j*k*np.dot(r,ksi))/z)
                T2 = self.u0(ksi)
                T3 = np.exp(1j*self.w0(angx,angy,ksi,aim=aim))

                ret = ret+T1*T2*T3
        ret = ret*dksi*(1j*k*np.exp(-(1j*k*(np.linalg.norm(r)**2))/(2*z))/(2*np.pi*z))

        return ret

    def u_rz(self,zmn,thmn,ksi,i,t,side='l',xlist=False,ylist=False):
        [x0,y0] = ksi
        z0 = zmn['00']
        angx = zmn['11']*np.cos(thmn['11'])
        angy = zmn['11']*np.sin(thmn['11'])
        x = x0*np.cos(angx)
        y = y0*np.cos(angy)
        z = z0 + x0*np.sin(angx)+y0*np.sin(angy)
        r = np.array([x,y])

        u = self.u_rz_calc(r,z,i,t,side,xlist=xlist,ylist=ylist)
        w = self.w(z) #...check if proper z is used (z0)
        wac = (self.data.D/2)/w
        norm = ((np.pi*((2*np.pi)/self.data.labda)*w**2*(1-np.exp(-1/(wac**2))))/(2*np.pi*z))**2
        I = (abs(u)**2)/norm
        phase = np.angle(u)

        return u,I,phase

    def u_rz_aperture(self,zmn,thmn,i,t,side='l',xlist=False,ylist=False,mode='power'):
        if zmn==np.nan or thmn==np.nan:
            function = lambda x,y: np.nan
        else:
            if self.speed_on>=1: #Only canculates center (works correctly for only piston and tilt
                xlist = np.array([0])
                ylist = np.array([0])
             
            if mode=='power':
                function = lambda x,y: self.u_rz(zmn,thmn,np.array([x,y]),i,t,side=side,xlist=xlist,ylist=ylist)[1]
            elif mode=='u':
                function = lambda x,y: abs(self.u_rz(zmn,thmn,np.array([x,y]),i,t,side=side,xlist=xlist,ylist=ylist)[0])
            elif mode=='phase':
                function = lambda x,y: self.u_rz(zmn,thmn,np.array([x,y]),i,t,side=side,xlist=xlist,ylist=ylist)[2]

        ps = self.aperture(xlist,ylist,function,dType=np.float64)
        
        
        return ps

    


    def aperture(self,xlist,ylist,function,dType=np.float64): # Creates matrix of function over an aperture (circle)
        #print('Type of telescope control is: ' + self.tele_control)
        if type(xlist)==bool:
            if xlist==False:
                xlist = self.xlist
        if type(ylist)==bool:
            if ylist==False:
                ylist = self.ylist

        Nbins = len(xlist)
        if len(xlist)!=0:
            step = xlist[1]-xlist[0]
        else:
            step = self.D
        ps = np.empty((Nbins,Nbins),dtype=dType)
        for i in range(0,len(xlist)):
            for j in range(0,len(ylist)):
                x = xlist[i]
                y = ylist[j]
                #r = (x**2+y**2)**0.5
                if x**2+y**2<=0.25*(self.D**2):
                    val = function(x,y)
                    if type(val)==list:
                        val=val[0]
                    ps[i,j] = val

                else:
                    ps[i,j] = np.nan
        
        return ps

#Obtining TTL by pointing
    def offset(self,tstart=False,tend=False,dt=False):
        if tstart==False:
            tstart = self.t_all[0]
        if tend==False:
            tend=self.t_all[-1]
        if dt==False:
            dt=self.t_all[1]-self.t_all[0]

        t_sample = np.linspace(tstart,tend,int((tend-tstart)/dt)+1)
        
        offset=[{},{},{},{}]
        for i in range(1,4):
            ang_l_in = np.array([self.data.ang_in_l(i,t) for t in t_sample])
            ang_l_out = np.array([self.data.ang_out_l(i,t) for t in t_sample])
            ang_r_in = np.array([self.data.ang_in_r(i,t) for t in t_sample])
            ang_r_out = np.array([self.data.ang_out_r(i,t) for t in t_sample])
            A = [ang_l_in,ang_l_out,ang_r_in,ang_r_out]

            for j in range(0,len(A)):
                offset[j][i] = np.nansum(A[j])/(list(np.isnan(A[j])).count(False))

        return offset


    def mean_angin(self,i,side,dt=False): #Used
        t_vec = self.data.t_all

        if dt==False:
            dt = t_vec[1]-t_vec[0]
        t_vec = np.linspace(t_vec[2],t_vec[-1],int(((t_vec[-1]-t_vec[2])/dt)+1))
        #print(t_vec)
        ang=[]
        
        if side=='l':
            ang = -abs(np.nanmean(np.array([self.data.ang_in_l(i,t) for t in t_vec]))) - np.radians(-30)
        elif side=='r':
            ang = abs(np.nanmean(np.array([self.data.ang_in_r(i,t) for t in t_vec])))-np.radians(30)

        return ang

    def do_mean_angin(self,dt=False): #Used
        self.mean_angin_l={}
        self.mean_angin_r={}

        for i in range(1,4):
            self.mean_angin_l[str(i)] = self.mean_angin(i,'l',dt=dt)
            self.mean_angin_r[str(i)] = self.mean_angin(i,'r',dt=dt)

        return 0

    def zern_aim(self,i_self,t,side='l',ret=[],ksi=[0,0],mode='auto',angin=False,angout=False,offset=False,aim='Default',**kwargs):
        # This function calculates the tilt and piston when receiving the beam in a telescope.
        if aim=='Default':
            aim = self.aim

        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i_self)
        # Tilt by function
        if side=='l':

            tdel = self.data.L_rl_func_tot(i_self,t)
            angx_func_send = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_left,t-tdel,'r',aim.beam_r_ang(i_left,t-tdel),'angx',mode='opposite',ksi=ksi)
            angy_func_send = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_left,t-tdel,'r',aim.beam_r_ang(i_left,t-tdel),'angy',mode='opposite',ksi=ksi)
            point_val = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_self,t,'l',aim.beam_r_ang(i_left,t-tdel),'all',mode='self',ksi=ksi,precision=1)

        elif side=='r':
            tdel = self.data.L_rr_func_tot(i_self,t)
            angx_func_send = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_right,t-tdel,'l',aim.beam_l_ang(i_right,t-tdel),'angx',mode='opposite',ksi=ksi)
            angy_func_send = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_right,t-tdel,'l',aim.beam_l_ang(i_right,t-tdel),'angy',mode='opposite',ksi=ksi)
            point_val = NOISE_LISA.functions.get_wavefront_parallel(self,aim,i_self,t,'r',aim.beam_l_ang(i_right,t-tdel),'all',mode='self',ksi=ksi,precision=1)
        
        
        angx_tilt = np.arctan(abs(point_val.get("bd_receiving_frame")[2]/point_val.get("bd_receiving_frame")[0]))*np.sign(np.dot(point_val.get("bd_original_frame"),point_val.get("coor_end")[2]))

        angy_tilt = np.arctan(abs(point_val.get("bd_receiving_frame")[1]/point_val.get("bd_receiving_frame")[0]))*np.sign(np.dot(point_val.get("bd_original_frame"),point_val.get("coor_end")[1]))

               
        # Zernike polynomials
        angx_tot = point_val.get("angx_func_rec")
        angy_tot = point_val.get("angy_func_rec")
        
        u=self.u_rz_calc(point_val.get("r"),point_val.get("zoff"),point_val.get("angx_func_rec"),point_val.get("angy_func_rec"),xlist=False,ylist=False,aim=aim)

        power = (abs(u)**2)[0]*np.cos(point_val.get("angx_func_rec"))*np.cos(point_val.get("angy_func_rec"))
        vec = point_val.get("bd_receiving_frame")
        
        beam_inc_tele_frame = point_val.get("bd_receiving_frame")
        if side=='l':
            PAAM_ang=aim.beam_l_ang(i_self,t)
            tele_ang=aim.tele_l_ang(i_self,t)
        elif side=='r':
            PAAM_ang=aim.beam_r_ang(i_self,t)
            tele_ang=aim.tele_r_ang(i_self,t)
        
        

        OPD = aim.noise.OPD[i_self][side](t)
        Dx = aim.noise.Dx[i_self][side](t)
        Dy = aim.noise.Dy[i_self][side](t)


        if type(ret)!=list:
            ret = [ret]
        ret_arg=[]
        for ret_val in ret:
            try:
                ret_arg.append(locals()[ret_val])
            except KeyError:
                try:
                    ret_arg.append(point_val.get(ret_val))
                except NameError,e:
                    print(e)
                    print('Please select proper return value')
            except NameError,e:
                print(e)
                print('Please select proper return value')
            
            return ret_arg


#            elif ret=='surface':
#                thmn11 = np.arctan(angy/angx)
#                zmn11 = (angx**2 + angy**2)**0.5
#                zmn00 = piston
#                zmn={}
#                thmn={}
#                zmn['00'] = zmn00
#                zmn['11'] = zmn11
#                thmn['11'] = thmn11
#                self.zmn = zmn
#                self.thmn = thmn
#                power_angle = [angx_tot,angy_tot]
#                return [xoff_0,yoff_0,zoff_0],zmn,thmn

    def calc_piston_val(self,i,t,side,ret=[],size='small',aim='Default'):
        if size == 'big' or ret=='all_val':
            calc0 = lambda x,y: self.zern_aim(i,t,side=side,ret='all_val',ksi=[x,y],aim=aim)
            calc={}
            meas = calc0(0,0).keys()
            for k in meas:
                calc[k] = lambda x,y: calc0(x,y)[k]
            size='big'
        else:
            calc={}
            meas = ret
            for m in meas:
                calc[m] = lambda x,y: self.zern_aim(i,t,side=side,ret=m,ksi=[x,y],aim=aim)
            size='small'



        ret={}
        for m in meas:
            if size=='small':
                piston = calc[m]
            elif size=='big':
                piston  = lambda x,y: calc0(x,y)[m]

            if self.speed_on==2:
                xlist=[0]
                ylist=[0]
            else:
                xlist=False
                ylist=False
            
            try:
                piston_ttl = self.aperture(xlist,ylist,piston,dType=np.float64)
                N_flat = PAA_LISA.utils.flatten(piston_ttl)
                N = len(N_flat) - np.sum(np.isnan(N_flat))
                 
                try:
                    ret[m] = np.nanmean(piston_ttl),np.nanvar(piston_ttl)/np.float64(N)
                except RuntimeWarning:
                    print(piston(0,0))
                    ret[m] = np.nan,np.nan
            except ValueError,e:
                if str(e)=='setting an array element with a sequence.':
                    ret[m] =  piston(0,0),'vec'

        return ret


    def piston_val(self):
        self.piston_val_l = lambda i,t: self.calc_piston_val(i,t,'l',ret='piston')
        self.piston_val_r = lambda i,t: self.calc_piston_val(i,t,'r',ret='piston')
        self.ztilt_val_l = lambda i,t: self.calc_piston_val(i,t,'l',ret='z_extra')
        self.ztilt_val_r = lambda i,t: self.calc_piston_val(i,t,'r',ret='z_extra')

        return 0 







# Calulate P
    def P_calc(self,i,t,side='l'): # Only simple calculation (middel pixel)
        if side=='l':
            P = np.cos(self.zern_aim(i,t,side='l')[1]['11'])*self.data.P_L
        elif side=='r':
            P = np.cos(self.zern_aim(i,t,side='r')[1]['11'])*self.data.P_L
        
        return P

    def get_zern_poly(self,zmn=False,thmn=False,nmax=4):
        if zmn==False:
            zmn = self.zmn
        if thmn==False:
            thmn = self.thmn

        for n in range(0,nmax+1):
            for m in range(-n,n+1):
                if (m%2)==(n%2):
                    key=str(m)+str(n)
                    if key not in zmn.keys():
                        zmn[key]=0
                    if key not in thmn.keys():
                        thmn[key]=0

        self.zmn=zmn
        self.thmn=thmn

        return zmn,thmn


    def zern_para(self,z=False,zmn=False,thmn=False,x=0,y=0):
        if zmn==False:
            zmn = self.zmn
        if thmn==False:
            thmn = self.thmn

        x0 = x
        y0 = y
        angx = zmn['11']*np.cos(thmn['11'])
        angy = zmn['11']*np.sin(thmn['11'])

        x = x0*np.cos(angx)
        y = y0*np.cos(angy)
        if z==False:
            z = zmn['00']
        z = z + x0*np.sin(angx)+y0*np.sin(angy)

        labda = self.data.labda
        w = self.w(z)
        r0 = self.data.D*0.5
        k = (2*np.pi)/labda #...adjust for laser noise
        wac = r0/w
        q = -1.0/(wac**2) # ...In Sasso paper not with minus sign!!!
        print(wac,q) 
        zmn,thmn = self.get_zern_poly(zmn=zmn,thmn=thmn)

        z02 = zmn['02']
        z04 = zmn['04']
        z22abs = zmn['22']
        z33abs = zmn['33']
        th33 = thmn['33']
        th22 = thmn['22']
        th11 = thmn['11']
        z13abs = zmn['13']
        th13 = thmn['13']

        dzx = zmn['11']*np.cos(thmn['11'])#... adjust to (36a)
        dzy = zmn['11']*np.sin(thmn['11'])#...ajust to (36b)


        A2 = (1+np.exp(q)+2*(1-np.exp(q))*(wac**2))/(1-np.exp(q))
        A4 = (1-np.exp(q)+6*(1+np.exp(q))*(wac**2) + 12*(1-np.exp(q))*(wac**4))/(1-np.exp(q))
        B = (-2*(1+3*(wac**2)+6*(wac**4)+6*(1-np.exp(q))*(wac**6)))/(1-np.exp(q))
        C = (-2*(1+5*(wac**2)+2*(7+2*np.exp(q))*(wac**4)+18*(1-np.exp(q))*(wac**6)))/(1-np.exp(q))
        D = (4*(np.exp(q)+6*np.exp(q)*(wac**2)-2*(2-np.exp(q) - np.exp(2*q))*(wac**4) - 12*((1-np.exp(q))**2)*(wac**6)))/((1-np.exp(q))**2)
        G = (24*((np.exp(q)*(wac**2))-(2-9*np.exp(q)+np.exp(2*q))*(wac**4) -2*(7-2*np.exp(q)-5*np.exp(2*q))*(wac**6) -30*(1-np.exp(q))*(wac**8)))/((1-np.exp(q))**2)

        #G = 0#3(24*(np.exp(q)*(wac**2)-(2-9*np.exp(q) + np.exp(2*q))*(wac**4) - 2*(7-2*np.exp(q) - 5*np.exp(2*q))*(wac**6) -30((1-np.exp(q))**2)*(wac**8)))/((1-np.exp(q))**2)
        E = (2*(np.exp(q) - ((1-np.exp(2*q))**2)*(wac**4)))/((1-np.exp(q))**2)
        F = (-1*(1+2*(wac**2)+2*(1-np.exp(q))*(wac**4)))/(1-np.exp(q))
        H = (6*(2*np.exp(q)*(wac**2) - (1-np.exp(2*q))*(wac**4)-4*((1-np.exp(q))**2)*(wac**6)))/((1-np.exp(q))**2)

        b0 = A2*z02+A4*z04
        b1 = B*z22abs*z33abs*np.cos(th33-th22-th11)+C*z22abs*z13abs*np.cos(th22-th13-th11)+D*z02*z13abs*np.cos(th13-th11)+G*z04*z13abs*np.cos(th13-th11)
        b2 = E*z02+F*z22abs*np.cos(th22-2*th11)+H*z04

        b00 = b0
        b10 = B*np.cos(th33-th22)*z33abs*z22abs +C*np.cos(th22-th13)*z13abs*z22abs+D*np.cos(th13)*z13abs*z02 +G*np.cos(th13)*z13abs*z22abs
        b01 = B*np.sin(th33-th22)*z33abs*z22abs +C*np.sin(th22-th13)*z13abs*z22abs+D*np.sin(th13)*z13abs*z02 +G*np.sin(th13)*z13abs*z22abs
        b20 = E*z02+F*np.cos(th22)*z22abs+H*z04
        b02 = E*z02-F*np.cos(th22)*z22abs+H*z04
        b11 = 2*F*np.sin(th22)*z22abs

        
        # Power density
        c1 = -(4*(1+2*(2+np.exp(q))*(wac**2)+6*(1-np.exp(q))*(wac**4)))/(1-np.exp(-q))
        c2 = (2*(1+(1-np.exp(q))*(wac**2)))/(1-np.exp(-q))
        I = 1+c1*z13abs*(np.cos(th13)*dzx+np.sin(th13)*dzy)-c2*(dzx**2+dzy**2)
        print(c1,c2,I)
       
        u0,[a0,a1,a2,a3] = self.WFE_rec(zmn,thmn,xlist=False,ylist=False,fast_cal=True)

        u_2 = ((np.pi**2)*(w**4)*abs(a0+1j*a1+a2)**2*(k**2))/((2*np.pi*z)**2)
        I_2 = u_2/(((np.pi*k*w**2)*(1-np.exp(-1.0/(wac**2)))/(2*np.pi*z))**2)




        return u_2,I,I_2








    def obtain_ttl(self,zmn,thmn,offset,piston=True,tilt=True,mode='ttl'):
        ps=np.zeros((self.Nbinsx,self.Nbinsy),dtype=np.float64)
        if piston==False:
            zmn['00']=False
            thmn['00']=False
        if tilt==False:
            zmn['11']=False
            thmn['11']=False
        for n in range(0,2):
            for m in range(-n,n+1):
                if ((m%2) == (n%2)):
                    ps = ps + self.zern(m,n,zmn=zmn,thmn=thmn,offset=offset,mode=mode)
        if mode=='phase':
            ps = ps%(2*np.pi)
        wave_ttl = np.nanmean(ps)

        return ps, wave_ttl

    def ttl_pointing_function_calc(self,i,t,mode='ttl',side='l',piston=True,tilt=True,ret='value',offset=False):
        [xoff,yoff,zoff],zmn,thmn = self.zern_aim(i,t,side=side)
        if offset==False:
            offset=[0,0] # Because of additional tilt in offset, this can be set to 0 
        elif offset==True:
            offset = [xoff,yoff]
        ps,wave_ttl = self.obtain_ttl(zmn,thmn,offset,piston=piston,tilt=tilt,mode=mode)
        
        if ret=='value':
            return wave_ttl

        elif ret=='aperture':
            return ps

        elif ret=='all':
            return ps, wave_ttl


    def ttl_pointing_function(self,mode='ttl',ret='value',option='all'):
        
        if option=='piston':
            self.ttl_l = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='l',piston=True,tilt=False,ret=ret)
            self.ttl_r = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='r',piston=True,tilt=False,ret=ret)
        elif option=='tilt':
            self.ttl_l = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='l',piston=False,tilt=True,ret=ret)
            self.ttl_r = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='l',piston=False,tilt=True,ret=ret)
        elif option=='all':
            self.ttl_l = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='l',piston=True,tilt=True,ret=ret)
            self.ttl_r = lambda i,t: self.ttl_pointing_function_calc(i,t,mode=mode,side='l',piston=True,tilt=True,ret=ret)
        

















