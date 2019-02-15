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
        self.offset_control = kwargs.pop('offset_control',False)
        self.compensation_tele = kwargs.pop('compensation_tele',True)
        global LA
        LA = PAA_LISA.la()
        import imports
        
        self.wfe = wfe
        self.noise = pack.Noise(wfe=wfe)
        self.PAAM_method = wfe.PAAM_control_method
        self.tele_method = wfe.tele_control

    def static_tele_angle(self,select,i,dt=False,side='l'):
        if select=='PAAM':
            if side=='l':
                #func = self.wfe.data.PAA_func['l_out']
                func = self.wfe.data.ang_out_l
                #func_y = self.wfe.data.PAA_func['l_out']
            elif side=='r':
                #func = self.wfe.data.PAA_func['r_out']
                func = self.wfe.data.ang_out_r
                #func_y = self.wfe.data.PAA_func['r_out']
        
        elif select=='tele':
            if side=='l':                                
                func = self.tele_ang_l_fc    
            elif side=='r':
                func = self.tele_ang_r_fc 

        t_all = self.wfe.data.t_all
        if dt==False:
            dt = t_all[1]-t_all[0]
        t_vec = np.linspace(t_all[0],t_all[-1],(t_all[1]-t_all[0])/dt)
        val=[]
        for t in t_vec:
            val.append(func(i,t))

        return np.mean(val)

    def do_static_tele_angle(self,select,dt=False):
        tele_ang_off_l = lambda i: self.static_tele_angle(select,i,dt=dt,side='l')
        tele_ang_off_r = lambda i: self.static_tele_angle(select,i,dt=dt,side='r')
        
        if select=='PAAM':
            self.offset_PAAM_l = tele_ang_off_l
            self.offset_PAAM_r = tele_ang_off_r
        elif select=='tele':
            self.offset_tele_l = tele_ang_off_l
            self.offset_tele_r = tele_ang_off_r

        return 0

    def tele_control_ang_fc_calc(self,i,t,side='l'):
        coor = functions.coor_SC(self.wfe,i,t)
        if side=='l':
            v = -self.wfe.data.u_l_func_tot(i,t)
        elif side=='r':
            v = -self.wfe.data.u_r_func_tot(i,t)
        
        if self.compensation_tele==True:
            v = LA.unit(v)*(np.linalg.norm(v)-self.wfe.L_tele)

        v_SC = LA.matmul(coor,v)
        
        #print(v_SC)
        import warnings
        warnings.simplefilter("error", RuntimeWarning)
        try:
            ang = np.arcsin(v_SC[2]/np.linalg.norm(v_SC[0])) # Angle betweed x and r component, whih is the optimal telescope pointing angle (inplane)
        except RuntimeWarning:
            print(v_SC[2],v_SC[0])
            print(i,t,v)
            ang = np.nan
            pass

        return ang

    def get_aim_accuracy(self,i,t,side):
        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        if side=='l':
            i_calc = i_left
            s_calc = 'r'
        elif side=='r':
            i_calc=i_right
            s_calc='l'

        ret =  self.wfe.aim0.get_received_beam_duration(i_calc,t,s_calc,ksi=[0,0])
        [z,y,x] = LA.matmul(ret['coor_end'],ret['beam_in'])
        
        angx = np.sign(x)*abs(np.arctan(x/z))
        angy = np.sign(y)*abs(np.arctan(y/z))
        
        delay = ret['L_r']

        return [angx,angy,delay]

    def tele_control_ang_fc(self):
        # Obtaines functions for optimal telescope pointing vector
        delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l')[2]
        delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r')[2]
        ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l')[0]
        ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r')[0]
 
        self.tele_ang_l_fc = lambda i,t: self.wfe.aim0.tele_l_ang(i,t)+ang_tele_extra_l(i,t+delay_l(i,t))
        self.tele_ang_r_fc = lambda i,t: self.wfe.aim0.tele_r_ang(i,t)+ang_tele_extra_r(i,t+delay_r(i,t))

        #self.tele_ang_l_fc = lambda i,t: self.tele_control_ang_fc_calc(i,t,side='l')
        #self.tele_ang_r_fc = lambda i,t: self.tele_control_ang_fc_calc(i,t,side='r')

        return 0


    def tele_aim(self,method=False,dt=3600*24*10,jitter=False,tau=3600*24*5,mode='overdamped',iteration=0,tele_ang_extra=False):

        self.tele_control_ang_fc()

        if method == False:
            method = self.tele_method
        else:
            self.tele_method = method

        print('The telescope control method is: '+method)
        print(' ')
        
        tele_l0 = self.tele_ang_l_fc
        tele_r0 = self.tele_ang_r_fc
        if iteration>0:
            print('Number of pointing iterations is '+str(iteration))
            tele_l_extr0=lambda i,t: 0
            tele_r_extr0=lambda i,t: 0
            step=0
            while step<iteration:
                tele_vec0 = self.tele_aim_vec([tele_l0,tele_r0])
                tele_l_extr1 = lambda i,t: self.iteration_tele_calc(i,t,'l',tele_vec0)
                tele_r_extr1 = lambda i,t: self.iteration_tele_calc(i,t,'r',tele_vec0)
                tele_l1 = lambda i,t: tele_l0(i,t)+tele_l_extr1(i,t)
                tele_r1 = lambda i,t: tele_r0(i,t)+tele_r_extr1(i,t)
                #del tele_l0, tele_r0, tele_l_extr0, tele_r_extr0,tele_vec0
                #tele_l0 = tele_l1
                #tele_r0 = tele_r1
                #tele_l_extr0 = tele_l_extr1
                #tele_r_extr0 = tele_r_extr1
                #del tele_vec0, tele_l_extr1, tele_r_extr1, tele_l1, tele_r1

                step=step+1
            tele_l = tele_l1
            tele_r = tele_r1
        else:
            tele_l = tele_l0
            tele_r = tele_r0

                

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
            tele_l = self.step_response(tele_l_SS,'tele',dt,tau=tau,mode=mode)
            tele_r = self.step_response(tele_r_SS,'tele',dt,tau=tau,mode=mode)
            self.tele_l_ang_SS = tele_l_SS
            self.tele_r_ang_SS = tele_r_SS

        else:
            raise ValueError('Please select a valid telescope pointing method')

        # Adding jitter
        if jitter!=False:
            self.tele_l_ang = lambda i,t: self.add_jitter(tele_l,i,t,1e-6,1e10,dt=0.1)
            self.tele_r_ang = lambda i,t: self.add_jitter(tele_r,i,t,1e-6,1e10,dt=0.1)
        else:
            self.tele_l_ang = tele_l
            self.tele_r_ang = tele_r
        
        
        # Calculating new pointing vectors and coordinate system
        self.tele_l_coor = lambda i,t: pack.functions.coor_tele(self.wfe,i,t,self.tele_l_ang(i,t))
        self.tele_r_coor = lambda i,t: pack.functions.coor_tele(self.wfe,i,t,self.tele_r_ang(i,t))
        self.tele_l_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,self.tele_l_ang(i,t))[0])*L_tele
        self.tele_r_vec = lambda i,t: LA.unit(pack.functions.coor_tele(self.wfe,i,t,self.tele_r_ang(i,t))[0])*L_tele

        return 0


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

    def PAAM_control_ang_fc(self):
        # Obtaines functions for optimal telescope pointing vector
        delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l')[2]
        delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r')[2]
        ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l')[1]
        ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r')[1]

        self.PAAM_ang_l_fc = lambda i,t: self.wfe.aim0.beam_l_ang(i,t)+ang_PAAM_extra_l(i,t+delay_l(i,t))
        self.PAAM_ang_r_fc = lambda i,t: self.wfe.aim0.beam_r_ang(i,t)+ang_PAAM_extra_r(i,t+delay_r(i,t))


    def PAAM_control(self,method=False,dt=3600*24,jitter=False,tau=1,mode='overdamped',PAAM_ang_extra=False):
        if method==False:
            method = self.PAAM_method
        else:
            self.PAAM_method = method

        print('The PAAM control method is: ' +method)
        print(' ')

        #ang_fc_l = lambda i,t: self.wfe.data.PAA_func['l_out'](i,t)
        #ang_fc_r = lambda i,t: self.wfe.data.PAA_func['r_out'](i,t)
        self.PAAM_control_ang_fc()
        ang_fc_l = lambda i,t: self.PAAM_ang_l_fc(i,t)
        ang_fc_r = lambda i,t: self.PAAM_ang_r_fc(i,t)
        
        
        self.PAAM_fc_ang_l = ang_fc_l
        self.PAAM_fc_ang_r = ang_fc_r

        # Obtaining PAAM angles for 'fc' (full control), 'nc' (no control) and 'SS' (step and stair)
        
        if method=='full control':
            ang_l = ang_fc_l
            ang_r = ang_fc_r
        elif method=='no control':
            self.do_static_tele_angle('PAAM')
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


        # Calculating new pointing vectors and coordinate system
        self.beam_l_coor = lambda i,t: pack.functions.beam_coor_out(self.wfe,i,t,self.tele_l_ang(i,t),self.beam_l_ang(i,t))
        self.beam_r_coor = lambda i,t: pack.functions.beam_coor_out(self.wfe,i,t,self.tele_r_ang(i,t),self.beam_r_ang(i,t))
       
        # Calculating the Transmitted beam direction and position of the telescope aperture
        self.beam_l_direction = lambda i,t: self.beam_l_coor(i,t)[0]
        self.beam_r_direction = lambda i,t: self.beam_r_coor(i,t)[0]
        self.beam_l_start = lambda i,t: self.beam_l_direction(i,t)+np.array(self.wfe.data.LISA.putp(i,t))
        self.beam_r_start = lambda i,t: self.beam_r_direction(i,t)+np.array(self.wfe.data.LISA.putp(i,t))

        #self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*self.wfe.L_tele
        #self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*self.wfe.data.L_sl_func_tot(i,t)*c
        #self.beam_l_vec = lambda i,t: self.beam_l_coor(i,t)[0]*np.linalg.norm(self.wfe.data.v_l_func_tot(i,t))
        #self.beam_r_vec = lambda i,t: self.beam_r_coor(i,t)[0]*self.wfe.data.L_sr_func_tot(i,t)*c
        #self.beam_r_vec = lambda i,t: self.beam_l_coor(i,t)[0]*np.linalg.norm(self.wfe.data.v_r_func_tot(i,t))

        return 0
    

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
            coor_start = self.beam_r_coor(i_left,t-L_r)
            start =  self.beam_r_start(i_left,t-L_r)
            direction_in = self.beam_r_direction(i_left,t-L_r)
            if self.wfe.data.calc_method=='Abram':
                end = self.beam_l_start(i_self,t)
                #coor_end = self.beam_l_coor(i_self,t)
                coor_end = self.tele_l_coor(i_self,t)
                direction_out = self.beam_l_direction(i_self,t)
            else:
                end = self.beam_l_start(i_self,t)
                #coor_end = self.beam_l_coor(i_self,t-L_r)
                coor_end = self.tele_l_coor(i_self,t)
                direction_out = self.beam_l_direction(i_self,t)

        elif side=='r':
            coor_start = self.beam_l_coor(i_right,t-L_r)
            start =  self.beam_l_start(i_right,t-L_r)
            direction_in = self.beam_l_direction(i_right,t-L_r)
            if self.wfe.data.calc_method=='Abram':
                end = self.beam_r_start(i_self,t)
                #coor_end = self.beam_r_coor(i_self,t)
                coor_end = self.tele_r_coor(i_self,t)
                direction_out = self.beam_r_direction(i_self,t)
            else:
                end = self.beam_r_start(i_self,t)
                #coor_end = self.beam_r_coor(i_self,t)
                coor_end = self.tele_r_coor(i_self,t)
                direction_out = self.beam_r_direction(i_self,t)
        # ksi is in receiiving telescope frame so adapt ksi in beam send frame
        [ksix,ksiy]=ksi

        ksix_vec = coor_end[2]*ksiy
        ksiy_vec = coor_end[1]*ksiy
        end = end+ksix_vec+ksiy_vec
        target_pos = LA.matmul(coor_start,end-start)
        #target_direction = LA.matmul(coor_start,direction)
        

        ret={}
        ret['start'] = start
        ret['end'] = end
        ret['beam_out'] = direction_out
        ret['beam_in'] = direction_in
        ret['target_pos'] = target_pos
        #ret['target_direction'] = target_direction
        ret['coor_start'] = coor_start
        ret['coor_end'] = coor_end
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
 



