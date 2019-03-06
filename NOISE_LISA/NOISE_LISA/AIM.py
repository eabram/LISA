from imports import *
import functions
from parameters import *
#import PAA_LISA
import NOISE_LISA as pack

# tele and PAA aim
class AIM():
    global LA
    LA = PAA_LISA.la()

    def __init__(self,wfe,**kwargs):
        print('Start calculating telescope and PAAM aim')
        
        self.PAAM_method = wfe.PAAM_control_method
        if self.PAAM_method =='SS_lim':
            self.FOV_control = kwargs.pop('FOV_control',1e-6)
        self.tele_method = wfe.tele_control
        self.compensation_tele = kwargs.pop('compensation_tele',True)
        self.sampled_on = kwargs.pop('sampled',False)

        global LA
        LA = PAA_LISA.la()
        import imports
        
        self.wfe = wfe
        self.init_set = kwargs.pop('init',False)
        if self.init_set==True:
            self.get_t_sampled()
        else:
            self.aim0 = kwargs.pop('aim0',False)
            self.aim_old = kwargs.pop('aim_old',False)
            if self.aim_old==False:
                print('Please select init or previous aim iteration')
                raise ValueError

        self.count=kwargs.pop('count',0)
        print('')
        print('Iteration count: '+str(self.count))
        print('')
        self.noise = pack.Noise(wfe=wfe)
        self.PAAM_method = wfe.PAAM_control_method
        self.tele_method = wfe.tele_control
        self.sampled_val = False


    def get_t_sampled(self):
        t_sample0 = self.wfe.t_all

        t_sample_all_l = []
        t_sample_all_r = []

        for i_self in range(1,4):
            t_add_l = self.wfe.data.L_rl_func_tot
            t_add_r = self.wfe.data.L_rr_func_tot

            t_sample_l=[]
            t_sample_r=[]
            for t in t_sample0:
                t_sample_l.append(t - t_add_l(i_self,t))
                t_sample_r.append(t - t_add_r(i_self,t))
                t_sample_l.append(t)
                t_sample_r.append(t)
            t_sample_all_l.append(t_sample_l)
            t_sample_all_r.append(t_sample_r)
        
        self.t_sample_l = t_sample_all_l
        self.t_sample_r = t_sample_all_r

        return 0                  
    
    def get_funcions_from_sampling(self,samples):
        [[y_l_tele,y_r_tele],[y_l_beam,y_r_beam]] = samples
        
        f_l_tele=[]
        f_r_tele=[]
        f_l_beam=[]
        f_r_beam=[]
        for i in range(0,len(y_l_tele)):
            [xlt,ylt] = y_l_tele[i]
            [xrt,yrt] = y_r_tele[i]
            [xlb,ylb] = y_l_beam[i]
            [xrb,yrb] = y_r_beam[i]
            f_l_tele.append(pack.functions.interpolate(xlt,ylt))
            f_r_tele.append(pack.functions.interpolate(xrt,yrt))
            f_l_beam.append(pack.functions.interpolate(xlb,ylb))
            f_r_beam.append(pack.functions.interpolate(xrb,yrb))

        f_l_tele_ret = PAA_LISA.utils.func_over_sc(f_l_tele)
        f_r_tele_ret = PAA_LISA.utils.func_over_sc(f_r_tele)
        f_l_beam_ret = PAA_LISA.utils.func_over_sc(f_l_beam)
        f_r_beam_ret = PAA_LISA.utils.func_over_sc(f_r_beam)

        return [[f_l_tele_ret,f_r_tele_ret],[f_l_beam_ret,f_r_beam_ret]]


    def get_sampled_pointing(self,option='start',ret='val'):
        [f_l_tele,f_r_tele] = self.sampled_pointing('tele',option=option,ret=ret)
        [f_l_beam,f_r_beam] = self.sampled_pointing('PAAM',option=option,ret=ret)
        
        self.sampled_val =  [[f_l_tele,f_r_tele],[f_l_beam,f_r_beam]]
        self.sampled_val_type = ret

        return self.sampled_val

    def sampled_pointing(self,mode,option='start',ret='val'):
        if option=='start':
            aim_use = self.aim0
        elif option=='previous':
            aim_use = self.aim_old
        elif option=='self':
            aim_use = self
        # Sample pointing

        f_l_ang=[]
        f_l_coor=[]
        f_l_vec=[]
        f_l_start=[]
        f_l_direction=[]
        f_r_ang=[]
        f_r_coor=[]
        f_r_vec=[]
        f_r_start=[]
        f_r_direction=[]

        t_sample_l = self.aim0.t_sample_l
        t_sample_r = self.aim0.t_sample_r
        
        y_l_ang_all=[]
        y_r_ang_all=[]
        for i_self in range(1,4):
            x_l = t_sample_l[i_self-1]
            x_r = t_sample_r[i_self-1]
            if mode=='tele':
                y_l_ang = np.array([aim_use.tele_l_ang(i_self,t) for t in x_l])
                #y_l_coor = np.array([aim_use.tele_l_coor(i_self,t) for t in x_l])
                #y_l_vec = np.array([aim_use.tele_l_vec(i_self,t) for t in x_l])

                y_r_ang = np.array([aim_use.tele_r_ang(i_self,t) for t in x_r])
                #y_r_coor = np.array([aim_use.tele_r_coor(i_self,t) for t in x_r])
                #y_r_vec = np.array([aim_use.tele_r_vec(i_self,t) for t in x_r])
                
                f_l_ang.append(pack.functions.interpolate(x_l,y_l_ang))
                #f_l_coor.append(pack.functions.interpolate(x_l,y_l_coor))
                #f_l_vec.append(pack.functions.interpolate(x_l,y_l_vec))
                f_r_ang.append(pack.functions.interpolate(x_r,y_r_ang))
                #f_r_coor.append(pack.functions.interpolate(x_r,y_r_coor))
                #f_r_vec.append(pack.functions.interpolate(x_r,y_r_vec))
 
            elif mode=='PAAM':
                y_l_ang = np.array([aim_use.beam_l_ang(i_self,t) for t in x_l])
                #y_l_start = np.array([aim_use.beam_l_start(i_self,t) for t in x_l])
                #y_l_coor = np.array([aim_use.beam_l_coor(i_self,t) for t in x_l])
                #y_l_direction = np.array([aim_use.beam_l_direction(i_self,t) for t in x_l])

                y_r_ang = np.array([aim_use.beam_r_ang(i_self,t) for t in x_r])
                #y_r_start = np.array([aim_use.beam_r_start(i_self,t) for t in x_r])
                #y_r_coor = np.array([aim_use.beam_r_coor(i_self,t) for t in x_r])
                #y_r_direction = np.array([aim_use.beam_r_direction(i_self,t) for t in x_r])

                f_l_ang.append(pack.functions.interpolate(x_l,y_l_ang))
                #f_l_start.append(pack.functions.interpolate(x_l,y_l_start))
                #f_l_coor.append(pack.functions.interpolate(x_l,y_l_coor))
                #f_l_direction.append(pack.functions.interpolate(x_l,y_l_direction))
                f_r_ang.append(pack.functions.interpolate(x_r,y_r_ang))
                #f_r_start.append(pack.functions.interpolate(x_r,y_r_start))
                #f_r_coor.append(pack.functions.interpolate(x_r,y_r_coor))
                #f_r_direction.append(pack.functions.interpolate(x_r,y_r_direction))

            y_l_ang_all.append([x_l,y_l_ang])
            y_r_ang_all.append([x_r,y_r_ang])


        
        #f_l={}
        #f_r={}

        if ret=='function':
            if mode=='tele':
                f_l = PAA_LISA.utils.func_over_sc(f_l_ang)
                #f_l['coor'] = f_l_coor
                #f_l['vec'] = f_l_vec
                f_r = PAA_LISA.utils.func_over_sc(f_r_ang)
                #f_r['coor'] = f_r_coor
                #f_r['vec'] = f_r_vec
            elif mode=='PAAM':
                f_l = PAA_LISA.utils.func_over_sc(f_l_ang)
                #f_l['start'] = f_l_start
                #f_l['coor'] = f_l_coor
                #f_l['direction'] = f_l_direction
                f_r = PAA_LISA.utils.func_over_sc(f_r_ang)
                #f_r['start'] = f_l_start
                #f_r['coor'] = f_l_coor
                #f_r['direction'] = f_l_direction
            
            return [f_l,f_r]
        elif ret=='val':
            return [y_l_ang_all,y_r_ang_all]
    
    def get_wavefront_parallel(self,i,t,side):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)


    def get_aim_accuracy(self,i,t,side,component=False,option='wavefront'):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        if option=='center':
            if component=='tele':
                if side=='l':
                    tdel=self.wfe.data.L_rl_func_tot(i_self,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    end = self.aim_old.tele_l_start(i_self,t-tdel0)
                    start = self.aim_old.tele_r_start(i_left,t-tdel)
                    coor_start = self.aim_old.beam_r_coor(i_left,t-tdel)
                    coor_end = self.aim_old.tele_l_coor(i_self,t)
                    #vec = self.wfe.data.v_l_func_tot(i,t)
                elif side=='r':
                    tdel = self.wfe.data.L_rr_func_tot(i_self,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    end = self.aim_old.tele_r_start(i_self,t-tdel0)
                    start = self.aim_old.tele_l_start(i_right,t-tdel)
                    coor_start = self.aim_old.beam_l_coor(i_right,t-tdel)
                    coor_end = self.aim_old.tele_r_coor(i_self,t)
                    #vec = self.wfe.data.v_r_func_tot(i,t)
                
                [z,y,x] = LA.matmul(coor_start,end-start)
                angx = np.sign(x)*abs(np.arctan(x/z))
                angy = np.sign(y)*abs(np.arctan(y/z))
                delay = tdel
            elif component=='PAAM':        
                if side=='l':
                    tdel = self.wfe.data.L_sr_func_tot(i_left,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    start = self.aim_old.tele_r_start(i_left,t+tdel0)
                    end = self.aim_old.tele_l_start(i_self,t+tdel)
                    coor_start = self.aim_old.beam_r_coor(i_left,t)
                elif side=='r':
                    tdel = self.wfe.data.L_sl_func_tot(i_right,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    start = self.aim_old.tele_l_start(i_right,t+tdel0)
                    end = self.aim_old.tele_r_start(i_self,t+tdel)
                    coor_start = self.aim_old.beam_l_coor(i_right,t)
                 
                [z,y,x] = LA.matmul(coor_start,end-start)
                angx = np.sign(x)*abs(np.arctan(x/z))
                angy = np.sign(y)*abs(np.arctan(y/z))
                delay = tdel

            return [angx,-angy,delay]
        
        elif option=='wavefront':
            if component=='tele':
                if side=='l':
                    tdel = self.wfe.data.L_rl_func_tot(i_self,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    coor_start = self.aim_old.tele_r_coor(i_left,t-tdel)
                    coor_end = self.aim_old.tele_l_coor(i_self,t)
                    direct = self.aim_old.beam_r_direction(i_left,t-tdel)
                    start = self.aim_old.tele_r_start(i_left,t-tdel)
                    end = self.aim_old.tele_l_start(i_self,t-tdel0)
                elif side=='r':
                    tdel = self.wfe.data.L_rr_func_tot(i_self,t)
                    if self.wfe.data.calc_method=='Waluschka':
                        tdel0=tdel
                    elif self.wfe.data.calc_method=='Abram':
                        tdel0=0
                    coor_start = self.aim_old.tele_l_coor(i_right,t-tdel)
                    coor_end = self.aim_old.tele_r_coor(i_self,t)
                    direct = self.aim_old.beam_l_direction(i_right,t-tdel)
                    start = self.aim_old.tele_l_start(i_right,t-tdel)
                    end = self.aim_old.tele_r_start(i_self,t-tdel0)

                [zoff,yoff,xoff] = LA.matmul(coor_start,end-start)
                R = zoff #Not precise
                R_vec = np.array([(R**2-xoff**2-yoff**2)**0.5,yoff,xoff])
                R_vec_origin = LA.matmul(np.linalg.inv(coor_start),R_vec)
                [z,y,x] = LA.matmul(coor_end,-R_vec_origin)
                #[z,y,x]=LA.matmul(coor_end,-direct)
                angx = np.sign(x)*abs(np.arctan(x/z))
                angy = np.sign(y)*abs(np.arctan(y/z))
                delay=tdel

            elif component=='PAAM':
                angy_solve = lambda PAAM_ang: pack.functions.get_wavefront_parallel(self.wfe,self.aim_old,i_self,t,side,PAAM_ang,'angy',mode='opposite')
                try:
                    angy =  scipy.optimize.brentq(angy_solve,-1e-1,1e-1)
                    #angy =  scipy.optimize.fmin(angy_solve,0,disp=False)[0]
                    #angy = scipy.optimize.minimize(angy_solve,0)['fun']
                except ValueError:
                    angy=np.nan
                delay=False
                angx=False

        return [angx,angy,delay]


    def tele_control_ang_fc(self,option='wavefront'):
        # Option 'wavefront' means poiting with the purpose of getting a small tilt of the receiving wavefront
        # 'center' means pointing it to te center of the receiving telescope
        i_left = lambda i: PAA_LISA.utils.i_slr(i)[1]
        i_right = lambda i: PAA_LISA.utils.i_slr(i)[2]

        print('Telescope pointing strategy: '+option)
        # Obtaines functions for optimal telescope pointing vector
        if option=='center':
            scale=1
        elif option=='wavefront':
            scale=1

        delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='tele',option=option)[2]
        delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='tele',option=option)[2]
        if option=='wavefront':
            #ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t+delay_l(i,t),'r',component='tele',option=option)[0]*scale #...
            #ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t+delay_r(i,t),'l',component='tele',option=option)[0]*scale
            ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='tele',option=option)[0]*scale #...
            ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='tele',option=option)[0]*scale

        
        elif option=='center':
            ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t+delay_l(i,t),'r',component='tele',option=option)[0]*scale
            ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t+delay_r(i,t),'l',component='tele',option=option)[0]*scale
            #ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='tele',option=option)[0]*scale
            #ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='tele',option=option)[0]*scale


        if self.sampled_on==True:
            if type(self.sampled_val)==bool:
                self.get_sampled_pointing(option='previous')
            [[tele_l,tele_r],[beam_l,beam_r]] = self.get_funcions_from_sampling(self.sampled_val)
        else:
            tele_l = self.aim_old.tele_l_ang
            tele_r = self.aim_old.tele_r_ang
            beam_l = self.aim_old.beam_l_ang
            beam_r = self.aim_old.beam_r_ang
        
        if option=='wavefront':
            #self.tele_ang_l_fc = lambda i,t: tele_l(i,t)+ang_tele_extra_l(i,t+delay_l(i,t))
            #self.tele_ang_r_fc = lambda i,t: tele_r(i,t)+ang_tele_extra_r(i,t+delay_r(i,t))
            self.tele_ang_l_fc = lambda i,t: tele_l(i,t)+ang_tele_extra_l(i,t)
            self.tele_ang_r_fc = lambda i,t: tele_r(i,t)+ang_tele_extra_r(i,t)
        elif option=='center':
            self.tele_ang_l_fc = lambda i,t: tele_l(i,t)+ang_tele_extra_l(i,t)
            self.tele_ang_r_fc = lambda i,t: tele_r(i,t)+ang_tele_extra_r(i,t)


        
        self.tele_option = option
        return 0


    def tele_aim(self,method=False,dt=3600*24*10,jitter=False,tau=3600*24*5,mode='overdamped',iteration=0,tele_ang_extra=False,option='wavefront'):
        self.option_tele=option
        if self.init_set==False:
            self.get_sampled_pointing(option='previous')
            self.tele_control_ang_fc(option=option)
            tele_l = self.tele_ang_l_fc
            tele_r = self.tele_ang_r_fc

        if method == False:
            method = self.tele_method
        else:
            self.tele_method = method

        print('The telescope control method is: '+method)
        print(' ')
        
        # Calculating telescope angle for 'full control', 'no control' and 'SS' (step and stair)
        if method=='full control':
            tele_l = tele_l
            tele_r = tele_r

        elif method=='no control':
            #self.do_static_tele_angle('tele')
            if tele_ang_extra==False:
                offset_l = [0,0,0]
                offset_r = [0,0,0]
            else:
                [offset_l,offset_r] = tele_ang_extra

            tele_l = lambda i,t: np.radians(-30)+offset_l[i-1]*0.5
            tele_r = lambda i,t: np.radians(30)+offset_r[i-1]*0.5


        elif method=='SS': #After dt, the telescope is pointed again
            tele_l_SS = lambda i,t: self.tele_ang_l_fc(i,t-(t%dt))
            tele_r_SS = lambda i,t: self.tele_ang_r_fc(i,t-(t%dt))
            print('Taken '+mode+' step response for telescope SS control with tau='+str(tau)+'sec')
            #tele_l = self.step_response(tele_l_SS,'tele',dt,tau=tau,mode=mode)
            #tele_r = self.step_response(tele_r_SS,'tele',dt,tau=tau,mode=mode)
            self.tele_l_ang_SS = tele_l_SS
            self.tele_r_ang_SS = tele_r_SS
        
        elif method=='SS FOV':
            ret={}
            for link in range(1,4):
                ret = pack.functions.get_SS_FOV(self.wfe,self.aim_old,link,ret=ret)

            self.tele_l_ang_SS_FOV = lambda i,t: ret[str(i)]['l'](t)
            self.tele_r_ang_SS_FOV = lambda i,t: ret[str(i)]['l'](t)
            tele_l = self.tele_l_ang_SS_FOV
            tele_r = self.tele_r_ang_SS_FOV

        else:
            raise ValueError('Please select a valid telescope pointing method')

        # Adding jitter
        if jitter!=False:
            self.tele_l_ang = lambda i,t: self.add_jitter(tele_l,i,t,1e-6,1e10,dt=0.1)
            self.tele_r_ang = lambda i,t: self.add_jitter(tele_r,i,t,1e-6,1e10,dt=0.1)
        else:
            #try:
            #    tele_l_ang_old = self.tele_l_ang
            #    tele_r_ang_old = self.tele_r_ang
            #    self.tele_l_ang = lambda i,t: (tele_l_ang_old(i,t)+tele_l(i,t))/np.float64(2.0)
            #    self.tele_r_ang = lambda i,t: (tele_r_ang_old(i,t)+tele_r(i,t))/np.float64(2.0)
            #except AttributeError:
            #        self.tele_l_ang = tele_l
            #        self.tele_r_ang = tele_r


            #self.tele_l_ang = lambda i,t: (tele_l(i,t) - self.tele_l_ang(i,t))*0.5 +self.tele_l_ang(i,t)
            #self.tele_r_ang = lambda i,t: (tele_r(i,t) - self.tele_r_ang(i,t))*0.5 +self.tele_r_ang(i,t)
            
            if method=='SS':
                self.tele_l_ang = tele_l_SS
                self.tele_r_ang = tele_r_SS
            else:
                self.tele_l_ang = tele_l
                self.tele_r_ang = tele_r
        
        
        ## Calculating new pointing vectors and coordinate system
        #self.tele_l_coor = lambda i,t: pack.functions.coor_tele(self.wfe,i,t,self.tele_l_ang(i,t))
        #self.tele_r_coor = lambda i,t: pack.functions.coor_tele(self.wfe,i,t,self.tele_r_ang(i,t))
        #self.tele_l_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,self.tele_l_ang(i,t))[0])*L_tele
        #self.tele_r_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,self.tele_r_ang(i,t))[0])*L_tele

        return 0

    def get_tele_coor(self,i,t,tele_l_ang,tele_r_ang):
        # Calculating new pointing vectors and coordinate system
        tele_l_coor = pack.functions.coor_tele(self.wfe,i,t,tele_l_ang(i,t))
        tele_r_coor = pack.functions.coor_tele(self.wfe,i,t,tele_r_ang(i,t))
        tele_l_vec = LA.unit(pack.functions.coor_tele(self.wfe,i,t,tele_l_ang(i,t))[0])*L_tele
        tele_r_vec = LA.unit(pack.functions.coor_tele(self.wfe,i,t,tele_r_ang(i,t))[0])*L_tele
        tele_l_start = tele_l_vec+np.array(self.wfe.data.LISA.putp(i,t))
        tele_r_start = tele_r_vec+np.array(self.wfe.data.LISA.putp(i,t))

        return [[tele_l_coor,tele_r_coor],[tele_l_vec,tele_r_vec],[tele_l_start,tele_r_start]]


    def tele_aim_vec(self,ang):
        tele_l_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,ang[0](i,t))[0])*L_tele
        tele_r_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,ang[1](i,t))[0])*L_tele

        return [tele_l_vec,tele_r_vec]

    
    def iteration_tele_calc(self,i,t,side,tele_vec):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        
        if side=='l':
            tele_send = tele_vec[0](i_self,t)
            i_next = i_left
            tdel = self.wfe.data.L_sl_func_tot(i_self,t) 
            tele_rec = LA.unit(tele_vec[1](i_next,t+tdel))*self.wfe.L_tele
        elif side=='r':
            tele_send = tele_vec[1](i_self,t)
            i_next = i_right
            tdel = self.wfe.data.L_sr_func_tot(i_self,t)
            tele_rec = LA.unit(tele_vec[0](i_next,t+tdel))*self.wfe.L_tele
        
        tele_send = LA.unit(tele_send)*tdel*c
        n = self.wfe.data.n_func(i_self,t)
        v = tele_send-tele_rec
        ang_extr = LA.ang_in(v,n,tele_send)

        return ang_extr

    #def iteration_tele(self,ang0):
    #    self.ang_extr_l = lambda i,t:  self.iteration_tele_calc(i,t,'l')
    #    self.ang_extr_r = lambda i,t:  self.iteration_tele_calc(i,t,'r')
    #    
    #    #Adjust telescope angle
    #    tele_l_ang_fc = ang0[0]
    #    tele_r_ang_fc = ang0[1]

    #    tele_pp_l_ang = lambda i,t: tele_l_ang_fc(i,t)+self.ang_extr_l(i,t)
    #    tele_pp_r_ang = lambda i,t: tele_l_ang_fc(i,t)+self.ang_extr_r(i,t)

    #    self.tele_l_ang_fc = lambda i,t: self.tele_l_ang_fc(i,t)+self.ang_extr_l(i,t)
    #    self.tele_r_ang_fc = lambda i,t: self.tele_r_ang_fc(i,t)+self.ang_extr_r(i,t)

    #    return [tele_pp_l_ang,tele_pp_r_ang]
    #
    #def tele_aim(self,method=False,dt=3600*24*10,jitter=False,tau=3600*24*5,mode='overdamped',iteration=1):
    #    
    #    self.tele_control_ang_fc()
    #    ang_iter = [self.tele_ang_l_fc,self.tele_ang_r_fc]

    #    if method == False:
    #        method = self.tele_method
    #    else:
    #        self.tele_method = method

    #    if iteration>0 and method!='no control':
    #        for step in range(1,iteration+1):
    #            self.tele_aim_ang(method='full control',dt=dt,jitter=False,tau=tau,mode=mode)
    #            ang_iter = self.iteration_tele(ang_iter)
    #            self.tele_aim_vec(ang=ang_iter)
    #            print('Iteration: '+str(step))

    #    self.tele_aim_ang(method=method,dt=dt,jitter=jitter,tau=tau,mode=mode)
    #    self.tele_aim_vec()

    #    return 0



         

        
        










        


    def add_jitter(self,ang_func,i,scale_v,dt=3600,PSD=False,f0=1e-6,f_max=1e-3,N=4096,offset=1,scale_tot=1):
        t_stop = self.wfe.t_all[-2]
        if PSD==False:
            PSD = lambda f: 16*1e-9
        func_noise = self.noise.Noise_time(f0,f_max,N,PSD,t_stop)[1]
        
        
        offset = offset*scale_tot
        scale_v = scale_v*scale_tot
        
        # add position jitter
        # add velocity jitter
        v = lambda t: (ang_func(t) - ang_func(t-dt))/dt
        ret = lambda t: func_noise(t)*(offset+v(t)*scale_v)+ang_func(t)
        #return np.random.normal(ang_func(i,t),dang*(1+v*scale_v))#...adjust: make correlated errors
        return ret

    
    def SS_FOV_control(self,i,f_PAA,xlim=False,accuracy=3600,FOV=1e-6,step=False,step_response=False):
        wfe=self.wfe
        if xlim==False:
            xlim=[wfe.t_all[1],wfe.t_all[-2]]
        if step==False:
            step=0.5*FOV
        self.FOV_control = FOV
        
        x0=xlim[0]
        function=lambda t: f_PAA(i,t)

        steps = [(function(x0)-function(x0)%step)]
        PAAM = [steps[-1]]
        t_PAAM = [x0]
        x_list=[x0]
        while x0<=xlim[1]:
            fb = f_PAA(i,x0)
            if fb-steps[-1]>step:
                steps.append(steps[-1])
                steps.append(step+steps[-1])
                x_list.append(x0-10)
                x_list.append(x0)
                t_PAAM.append(x0)
                PAAM.append(steps[-1])
            elif fb-steps[-1]<-step:
                steps.append(steps[-1])
                steps.append(-step+steps[-1])
                x_list.append(x0-10)
                x_list.append(x0)
                t_PAAM.append(x0)
                PAAM.append(steps[-1])
 
            x0=x0+accuracy
        steps.append(steps[-1])
        x_list.append(x0)
        
        try:
            PAAM_func = pack.functions.interpolate(np.array(x_list),np.array(steps))
        except ValueError:
            PAAM_func = lambda t: 0
        else:
            ret=[t_PAAM,PAAM,PAAM_func]

        return ret


    def SS_control(self,function,i,t,dt=False,xlim=False,accuracy=3600,FOV=1e-6,step=False):
        if dt == False:
            ret = self.SS_FOV_control(i,function,xlim=xlim,accuracy=accuracy,FOV=FOV,step=step)
        
        else:
            t0 = t-(t%dt)
            if t0==0 or t0==self.wfe.data.t_all[-1]:
                ret = np.nan
            else:
                ret = function(i,t-(t%dt))

        return ret

    def PAAM_control_ang_fc(self,option='center'):
        self.option_PAAM=option
        i_left = lambda i: PAA_LISA.utils.i_slr(i)[1]
        i_right = lambda i: PAA_LISA.utils.i_slr(i)[2]

        print('PAAM pointing strategy: '+option)
        # Obtains functions for optimal telescope pointing vector
        #if option!='wavefront':
        #    delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='PAAM',option=option)[2]
        #    delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='PAAM',option=option)[2]

        
        if option=='wavefront':
            #ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t+delay_l(i,t),'r',component='PAAM',option=option)[1]
            #ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t+delay_r(i,t),'l',component='PAAM',option=option)[1]
            if self.count>0:
                ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='PAAM',option=option)[1]
                ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='PAAM',option=option)[1]

        elif option=='center':
            ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t,'r',component='PAAM',option=option)[1]
            ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t,'l',component='PAAM',option=option)[1]

            #ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='PAAM',option=option)[1]
            #ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='PAAM',option=option)[1]


        if self.sampled_on==True:
            [[tele_l,tele_r],[beam_l,beam_r]] = self.get_funcions_from_sampling(self.sampled_val)
        else:
            tele_l = self.aim_old.tele_l_ang
            tele_r = self.aim_old.tele_r_ang
            beam_l = self.aim_old.beam_l_ang
            beam_r = self.aim_old.beam_r_ang

        
        if option=='wavefront':
            if self.count>0:
                self.PAAM_ang_l_fc = lambda i,t: ang_PAAM_extra_l(i,t)
                self.PAAM_ang_r_fc = lambda i,t: ang_PAAM_extra_r(i,t)
            else:
                self.PAAM_ang_l_fc = self.aim_old.beam_l_ang
                self.PAAM_ang_r_fc = self.aim_old.beam_r_ang

        elif option=='center':
            self.PAAM_ang_l_fc = lambda i,t: beam_l(i,t)+ang_PAAM_extra_l(i,t)
            self.PAAM_ang_r_fc = lambda i,t: beam_r(i,t)+ang_PAAM_extra_r(i,t)


        self.PAAM_option = option


    def PAAM_control(self,method=False,dt=3600*24,jitter=False,tau=1,mode='overdamped',PAAM_ang_extra=False,option='center'):
        if method==False:
            method = self.PAAM_method
        else:
            self.PAAM_method = method

        print('The PAAM control method is: ' +method)
        print(' ')

        if self.init_set==False:
            self.PAAM_control_ang_fc(option=option)
            ang_fc_l = lambda i,t: self.PAAM_ang_l_fc(i,t)
            ang_fc_r = lambda i,t: self.PAAM_ang_r_fc(i,t)
            self.PAAM_fc_ang_l = ang_fc_l
            self.PAAM_fc_ang_r = ang_fc_r

        #ang_fc_l = lambda i,t: self.wfe.data.PAA_func['l_out'](i,t)
        #ang_fc_r = lambda i,t: self.wfe.data.PAA_func['r_out'](i,t)
        #ang_fc_l = lambda i,t: self.PAAM_ang_l_fc(i,t)*2
        #ang_fc_r = lambda i,t: self.PAAM_ang_r_fc(i,t)*2
        #self.PAAM_fc_ang_l = ang_fc_l
        #self.PAAM_fc_ang_r = ang_fc_r

        # Obtaining PAAM angles for 'fc' (full control), 'nc' (no control) and 'SS' (step and stair)
        
        if method=='full control':
            ang_l = ang_fc_l
            ang_r = ang_fc_r
        elif method=='no control':
            #self.do_static_tele_angle('PAAM')
            if PAAM_ang_extra==False:
                ang_l = lambda i,t: 0
                ang_r = lambda i,t: 0
            else:
                [offset_l,offset_r] = PAAM_ang_extra
                ang_l = lambda i,t: offset_l[i-1]*0.5
                ang_r = lambda i,t: offset_r[i-1]*0.5

        elif method=='SS':
            ang_l_SS = lambda i,t: ang_fc_l(i,t-(t%dt)) # Adjusting the pointing every dt seconds
            ang_r_SS = lambda i,t: ang_fc_r(i,t-(t%dt))
            print('Taken '+method+' step response for PAAM SS control with tau='+str(tau)+' sec')
            mode='overdamped'

        elif method=='SS_lim':
            ang_l_SS = lambda i: self.SS_control(ang_fc_l,i,False,dt=False,xlim=False,accuracy=3600,FOV=self.FOV_control,step=False)
            ang_r_SS = lambda i: self.SS_control(ang_fc_r,i,False,dt=False,xlim=False,accuracy=3600,FOV=self.FOV_control,step=False)
            print('Taken '+method+' step response for PAAM SS control with tau='+str(tau)+' sec and step limit='+str(self.FOV_control*1e6)+' radians')
            mode='not_damped' #...damped SS not implemented jet for SS_lim
        else:
            raise ValueError('Please select a valid PAAM pointing method')


        if 'SS' in method:
            ang_l = self.step_response(ang_l_SS,'PAAM',dt,tau=tau,mode=mode)
            ang_r = self.step_response(ang_r_SS,'PAAM',dt,tau=tau,mode=mode)
            f_noise_l = lambda i,t: (ang_l(i,t)-ang_l_SS(i,t))**2
            f_noise_r = lambda i,t: (ang_r(i,t)-ang_r_SS(i,t))**2
            self.PAAM_ang_l_SS = ang_l_SS
            self.PAAM_ang_r_SS = ang_r_SS
            self.PAAM_step = dt


        # Adding jitter
        if jitter!=False:
            self.beam_l_ang = lambda i,t: self.add_jitter(ang_l,i,t,1e-8,1e20,dt=3600)
            self.beam_r_ang = lambda i,t: self.add_jitter(ang_r,i,t,1e-8,1e20,dt=3600)
        else:
            self.beam_l_ang = ang_l
            self.beam_r_ang = ang_r

        #self.PAAM_l_ang = lambda i,t: self.beam_l_ang(i,t)*wfe.MAGNIFICATION
        #self.PAAM_r_ang = lambda i,t: self.beam_r_ang(i,t)*wfe.MAGNIFICATION


        ## Calculating new pointing vectors and coordinate system
        #self.beam_l_coor = lambda i,t: pack.functions.beam_coor_out(self.wfe,i,t,self.tele_l_ang(i,t),self.beam_l_ang(i,t))
        #self.beam_r_coor = lambda i,t: pack.functions.beam_coor_out(self.wfe,i,t,self.tele_r_ang(i,t),self.beam_r_ang(i,t))
       
        ## Calculating the Transmitted beam direction and position of the telescope aperture
        #self.beam_l_direction = lambda i,t: self.beam_l_coor(i,t)[0]
        #self.beam_r_direction = lambda i,t: self.beam_r_coor(i,t)[0]
        #self.beam_l_start = lambda i,t: self.beam_l_direction(i,t)+np.array(self.wfe.data.LISA.putp(i,t))
        #self.beam_r_start = lambda i,t: self.beam_r_direction(i,t)+np.array(self.wfe.data.LISA.putp(i,t))

        ##self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*self.wfe.L_tele
        ##self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*self.wfe.data.L_sl_func_tot(i,t)*c
        ##self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*np.linalg.norm(self.wfe.data.v_l_func_tot(i,t))
        ##self.beam_r_vec = lambda i,t: self.beam_r_coor(i,t)[0]*self.wfe.data.L_sr_func_tot(i,t)*c
        ##self.beam_r_vec = lambda i,t: self.beam_l_coor(i,t)[0]*np.linalg.norm(self.wfe.data.v_r_func_tot(i,t))
        
 
        self.get_coordinate_systems(iteration_val=self.sampled_on,option='self')
        
        return self
    
    
    def get_beam_coor(self,i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang):
        # Calculating new pointing vectors and coordinate system
        beam_l_coor = pack.functions.beam_coor_out(self.wfe,i,t,tele_l_ang(i,t),beam_l_ang(i,t))
        beam_r_coor = pack.functions.beam_coor_out(self.wfe,i,t,tele_r_ang(i,t),beam_r_ang(i,t))

        # Calculating the Transmitted beam direction and position of the telescope aperture
        beam_l_direction = beam_l_coor[0]
        beam_r_direction = beam_r_coor[0]
        beam_l_start = beam_l_direction+np.array(self.wfe.data.LISA.putp(i,t))
        beam_r_start = beam_r_direction+np.array(self.wfe.data.LISA.putp(i,t))

        return [[beam_l_coor,beam_r_coor],[beam_l_direction,beam_r_direction],[beam_l_start,beam_r_start]]

    def get_coordinate_systems(self,iteration_val=False,option='self'):
        if iteration_val==False:
            tele_l_ang = self.tele_l_ang
            tele_r_ang = self.tele_r_ang
            beam_l_ang = self.beam_l_ang
            beam_r_ang = self.beam_r_ang
        else:
            [[tele_l_ang,tele_r_ang],[beam_l_ang,beam_r_ang]] = self.get_funcions_from_sampling(self.get_sampled_pointing(option=option))

        self.tele_l_coor = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[0][0]
        self.tele_r_coor = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[0][1]
        self.tele_l_vec = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[1][0]
        self.tele_r_vec = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[1][1]
        self.tele_l_start = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[2][0]
        self.tele_r_start = lambda i,t: self.get_tele_coor(i,t,tele_l_ang,tele_r_ang)[2][1]


        self.beam_l_coor = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[0][0]
        self.beam_r_coor = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[0][1]
        self.beam_l_direction = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[1][0]
        self.beam_r_direction = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[1][1]
        self.beam_l_start = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[2][0]
        self.beam_r_start = lambda i,t: self.get_beam_coor(i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang)[2][1]

        self.tele_l_ang_calc = tele_l_ang
        self.tele_r_ang_calc = tele_r_ang
        self.beam_l_ang_calc = beam_l_ang
        self.beam_r_ang_calc = beam_r_ang
        
        return 0 #[tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang,tele_l_coor,tele_r_coor,tele_l_vec,tele_r_vec,beam_l_coor,beam_r_coor,beam_l_direction,beam_r_direction,beam_l_start,beam_r_start]

    def get_received_beam_duration(self,i,t,side,ksi=[0,0]):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        
        # Calculate new tdel
        #... code has to be adjusted for telescope length
        if side=='l':
            L_r = self.wfe.data.L_rl_func_tot(i_self,t)
            L_s = self.wfe.data.L_sl_func_tot(i_self,t)
        elif side=='r':
            L_r = self.wfe.data.L_rr_func_tot(i_self,t)
            L_s = self.wfe.data.L_sr_func_tot(i_self,t)

        # Received beams (Waluschka)
        if side=='l':
            coor_start_beam = self.beam_r_coor(i_left,t-L_r)
            coor_start_tele = self.tele_r_coor(i_left,t-L_r)
            #coor_start = self.beam_r_coor(i_left,t-L_r)
            start =  self.tele_r_start(i_left,t-L_r)
            direction_out = self.beam_r_direction(i_left,t-L_r)
            if self.wfe.data.calc_method=='Abram':
                end = self.tele_l_start(i_self,t)
                coor_end_beam = self.beam_l_coor(i_self,t)
                coor_end_tele = self.tele_l_coor(i_self,t)
                direction_in = self.beam_l_direction(i_self,t)
            else:
                end = self.tele_l_start(i_self,t-L_r)
                coor_end_beam = self.beam_l_coor(i_self,t)
                coor_end_tele = self.tele_l_coor(i_self,t)
                direction_in = self.beam_l_direction(i_self,t+L_s)

        elif side=='r':
            coor_start_beam = self.beam_l_coor(i_right,t-L_r)
            coor_start_tele = self.tele_l_coor(i_right,t-L_r)
            start =  self.tele_l_start(i_right,t-L_r)
            direction_out = self.beam_l_direction(i_right,t-L_r)
            if self.wfe.data.calc_method=='Abram':
                end = self.tele_r_start(i_self,t)
                coor_end_beam = self.beam_r_coor(i_self,t)
                coor_end_tele = self.tele_r_coor(i_self,t)
                direction_in = self.beam_r_direction(i_self,t)
            else:
                end = self.tele_r_start(i_self,t-L_r)
                coor_end_beam = self.beam_r_coor(i_self,t)
                coor_end_tele = self.tele_r_coor(i_self,t)
                direction_in = self.beam_r_direction(i_self,t+L_s)        
        
        # ksi is in receiiving telescope frame so adapt ksi in beam send frame
        [ksix,ksiy]=ksi

        ksix_vec = coor_end_tele[2]*ksiy
        ksiy_vec = coor_end_tele[1]*ksiy
        end = end+ksix_vec+ksiy_vec
        target_pos_beam = LA.matmul(coor_start_beam,end-start)
        target_pos_tele = LA.matmul(coor_start_tele,end-start)
        target_direction = LA.matmul(coor_start_tele,direction_in) #... not inportant
        

        ret={}
        ret['start'] = start
        ret['end'] = end
        ret['beam_out'] = direction_out
        ret['beam_in'] = direction_in
        ret['target_pos_beam'] = target_pos_beam #w.r.t. beam
        ret['target_pos_tele'] = target_pos_tele #w.r.t. telescope
        ret['target_direction'] = target_direction #w.r.t. telescope
        ret['coor_start_beam'] = coor_start_beam
        ret['coor_start_tele'] = coor_start_tele
        ret['coor_end_beam'] = coor_end_beam
        ret['coor_end_tele'] = coor_end_tele
        ret['L_s'] = L_s
        ret['L_r'] = L_r

        return ret
    









    
    def step_response_calc(self,function,i,t,dt,tau,mode='overdamped'):
        if mode=='overdamped':
            #if self.PAAM_method=='SS':
            t0 = t-(t%dt)
            t1 = t0+dt
            Y0 = function(i,t0)
            Y1 = function(i,t1)
            #elif self.PAAM_method=='SS_lim':
            #    [t_PAAM,PAAM] = function(i)
            #    k = NOISE_LISA.get_nearest_smaller_value(t_PAAM,t)
            #    t0 = t_PAAM[k]
            #    t1 = t_PAAM[k+1]
            #    Y0 = PAAM[k]
            #    Y1 = PAAM[k+1]
            if t<self.wfe.t_all[2] or t>self.wfe.t_all[-2]:
                return np.nan
            else:
                if t0==0:
                    Y0=Y1
                return Y1+(Y0-Y1)*np.exp(-(t-t0)/tau)
        elif mode==False:
            return function(i,t)

    def step_response(self,function,select,dt,tau=3600,mode='overdamped'):
        if select=='PAAM' and self.PAAM_method=='SS_lim':
            f = []
            for i in range(1,4):
                [t_PAAM,PAAM,func] = function(i)
                if mode=='overdamped':
                    ret=[]
                    for j in range(1,len(t_PAAM)):
                        t0 = t_PAAM[j-1]
                        t1 = t_PAAM[j]
                        Y0 = PAAM[j-1]
                        Y1 = PAAM[j]

                        ret.append(lambda t: Y1+(Y0-Y1)*np.exp(-(t-t0)/tau))

                    pos = lambda t: pack.functions.get_nearest_smaller_value(t_PAAM,t)
                    f.append(lambda t: ret[pos(t)])
                else:
                    func_ret = lambda t: pack.functions.make_nan(func,t,[t_PAAM[0],t_PAAM[-1]])
                    f.append(func_ret)

            return PAA_LISA.utils.func_over_sc(f)
        else:
            return lambda i,t: self.step_response_calc(function,i,t,dt,tau=tau,mode=mode)
 



