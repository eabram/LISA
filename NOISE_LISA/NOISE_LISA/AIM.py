from imports import *
import functions
from parameters import *
#import PAA_LISA
import NOISE_LISA as pack

# tele and PAA aim
class AIM():
    global LA
    LA = PAA_LISA.utils.la()

    def __init__(self,wfe,**kwargs):
        self.wfe = wfe
        print(wfe)
        offset_tele = kwargs.pop('offset_tele','read')

        if wfe!=False:
            print('Start calculating telescope and PAAM aim')
            self.get_offset_inplane(offset_tele)
            self.PAAM_method = wfe.PAAM_control_method
            if self.PAAM_method =='SS_lim':
                self.FOV_control = kwargs.pop('FOV_control',1e-6)
            self.tele_method = wfe.tele_control
            self.compensation_tele = kwargs.pop('compensation_tele',True)
            self.sampled_on = kwargs.pop('sampled',False)

            global LA
            LA = PAA_LISA.la()
            import imports
            
            self.init_set = kwargs.pop('init',False)
            if self.init_set!=True:
                self.angles0=kwargs.pop('angles0',False)
                self.aim0=AIM(False)
                self.aim0.wfe=wfe
                self.aim0.tele_l_ang = self.angles0[0]
                self.aim0.tele_r_ang = self.angles0[2]
                self.aim0.beam_l_ang = self.angles0[1]
                self.aim0.beam_r_ang = self.angles0[3]
                self.aim0.get_coordinate_systems(option='self')
                self.aim0.offset_tele = self.offset_tele

                self.angles_old=kwargs.pop('angles_old',False)
                self.aim_old=AIM(False)
                self.aim_old.tele_l_ang = self.angles_old[0]
                self.aim_old.tele_r_ang = self.angles_old[2]
                self.aim_old.beam_l_ang = self.angles_old[1]
                self.aim_old.beam_r_ang = self.angles_old[3]
                self.aim_old.offset_tele = self.offset_tele
                self.aim_old.wfe=wfe
                self.aim_old.get_coordinate_systems(option='self')
                #if self.aim_old==False:
                #    if 'SS' in self.tele_method:
                #        self.aim_old=True
                #        self.aim0=True
                #    else:
                #        print('Please select init or previous aim iteration')
                #        raise ValueError

            self.count=kwargs.pop('count',0)
            print('')
            print('Iteration count: '+str(self.count))
            print('')
            #self.noise = pack.Noise(wfe=wfe)
            self.PAAM_method = wfe.PAAM_control_method
            self.tele_method = wfe.tele_control
            self.iteration=0
    
    def copy_aim(self,aim_old,option=False): #...copying all essential parameters
        for attr in aim_old.__dict__.keys():
            if attr not in self.__dict__.keys() or attr=='wfe':
                t = str(aim_old.__dict__[attr])
                go=False
                if option!=False:
                    if option=='new_angles':
                        if 'instance' in t:
                            print(t)
                            if 'WFE' in t:
                                if type(self.wfe)==bool:
                                    go=True
                                else:
                                    go=False
                            else:
                                go=False
                        elif 'function' in t:
                            go =False
                        else:
                            go=True
                else:
                    go=True
                if go==True:
                    #print(attr)
                    setattr(self,attr,aim_old.__dict__[attr])

        return 0 



    def get_noise(self,aim_old=False,dt=100):
        if aim_old==False:
            aim_old = self
        tele_l={}
        tele_r={}
        PAAM_l={}
        PAAM_r={}
        for SC in range(1,4):
            tele_l[SC] = pack.add_noise.aim_noise(SC,'l',aim_old,'tele',dt=dt)[0]
            tele_r[SC] = pack.add_noise.aim_noise(SC,'r',aim_old,'tele',dt=dt)[0]
            PAAM_l[SC] = pack.add_noise.aim_noise(SC,'l',aim_old,'tele',dt=dt)[0]
            PAAM_l[SC] = pack.add_noise.aim_noise(SC,'r',aim_old,'tele',dt=dt)[0]
        aim_new = pack.AIM(wfe=False)
        aim_new.wfe = aim_old.wfe
        aim_new.tele_l_ang = lambda i,t: tele_l[i](t)
        aim_new.tele_r_ang = lambda i,t: tele_r[i](t)
        aim_new.beam_l_ang = lambda i,t: beam_l[i](t)
        aim_new.beam_r_ang = lambda i,t: beam_r[i](t)
        aim_new.get_coordinate_systems(option='self') #...add position jitter
        aim_new.offset_tele = aim_old.offset_tele

        return aim_new




    def get_sampled(self,dt=False):
        try:
            del self.aim_sampled
        except AttributeError:
            pass
        wfe = self.wfe

        self.aim_sampled=AIM(False)
        if dt==False:
            dt = self.wfe.t_all[1]-self.wfe.t_all[0]
        start = wfe.t_all[3]
        end = wfe.t_all[-3]
        t_sample = np.linspace(start,end,int((end-start)/dt)+1)

        ang_l_tele_list=[]
        ang_r_tele_list=[]
        ang_l_PAAM_list=[]
        ang_r_PAAM_list=[]

        for SC in range(1,4):
            ang_l_tele = [self.tele_l_ang(SC,t) for t in t_sample]
            ang_r_tele = [self.tele_r_ang(SC,t) for t in t_sample]
            ang_l_PAAM = [self.beam_l_ang(SC,t) for t in t_sample]
            ang_r_PAAM = [self.beam_r_ang(SC,t) for t in t_sample]
            
            ang_l_tele_list.append(pack.functions.interpolate(t_sample,ang_l_tele))
            ang_r_tele_list.append(pack.functions.interpolate(t_sample,ang_r_tele))
            ang_l_PAAM_list.append(pack.functions.interpolate(t_sample,ang_l_PAAM))
            ang_r_PAAM_list.append(pack.functions.interpolate(t_sample,ang_r_PAAM))
        
        self.aim_sampled.tele_l_ang = PAA_LISA.utils.func_over_sc(ang_l_tele_list)
        self.aim_sampled.tele_r_ang = PAA_LISA.utils.func_over_sc(ang_r_tele_list)
        self.aim_sampled.beam_l_ang = PAA_LISA.utils.func_over_sc(ang_l_PAAM_list)
        self.aim_sampled.beam_r_ang = PAA_LISA.utils.func_over_sc(ang_r_PAAM_list)
        
        self.aim_sampled.wfe= wfe
        self.aim_sampled.get_coordinate_systems(iteration_val=False,option='self')

        self.aim_sampled.tele_option = self.tele_option
        self.aim_sampled.PAAM_option = self.PAAM_option
        self.aim_sampled.iteration = 0
        self.aim_sampled.t_all = t_sample
        

        return 0 
    
    def get_offset_inplane(self,option):

        offset = {'l': {1: 0.0, 2: 0.0, 3: 0.0},
 'r': {1: 0.0, 2: 0.0, 3: 0.0}}
        if option==0:
            pass

        elif option==True:
            offset = {'l': {1: -0.00018360462896226676, 2: 0.0, 3: 0.0},
 'r': {1: 0.0, 2: 0.0, 3: 0.0}}
        
        elif '/' in option:
            ret = pack.functions.read(direct=option)
            #inp = ret['full_control']['full_control']['0']['tele_center__PAAM_center']['angx_func_send mean']
            for k1 in ret:
                for k2 in ret[k1]:
                    for k3 in ret[k1][k2]:
                        for k4 in ret[k1][k2][k3]:
                            inp = ret[k1][k2][k3][k4]['angx_func_rec mean']
                            break

            for side in ['l','r']:
                for SC in range(1,4):
                    if side=='l':
                        label='SC'+str(SC)+', left'
                    elif side=='r':
                        label='SC'+str(SC)+', right'
                    ang = inp[label]['y'][3:-3]

                    offset[side][SC] = -np.nanmean(ang)
        elif 'read'==option:
            read_folder = os.path.dirname(os.path.realpath(__file__))+'/parameters/'+self.wfe.data.calc_method+'/'
            #print(read_folder)

            for (dirpath, dirnames, filenames) in os.walk(read_folder):
                for f in filenames:
                    read_file = open(dirpath+f)
                    break
            out=''
            count=0
            for line in read_file:
                count=count+1
                if count>1:
                    out=out+line.split('\n')[0]
            offset = yaml.load(out)
            read_file.close()

        else:
            raise ValueError("Please select offset tele values or method")

        self.offset_tele = offset

        return 0


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
                except ValueError:
                    angy=np.nan
                delay=False
                angx=False

        return [angx,angy,delay]


    def tele_control_ang_fc(self,option='wavefront'):
        # Option 'wavefront' means poiting with the purpose of getting a small tilt of the receiving wavefront
        # 'center' means pointing it to te center of the receiving telescope
        if option=='center':
            i_left = lambda i: PAA_LISA.utils.i_slr(i)[1]
            i_right = lambda i: PAA_LISA.utils.i_slr(i)[2]

            print('Telescope pointing strategy: '+option)
            scale=1

            delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='tele',option=option)[2]
            delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='tele',option=option)[2] 
            ang_tele_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t+delay_l(i,t),'r',component='tele',option=option)[0]*scale
            ang_tele_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t+delay_r(i,t),'l',component='tele',option=option)[0]*scale

            tele_l = self.aim_old.tele_l_ang
            tele_r = self.aim_old.tele_r_ang 
            ang_l = lambda i,t: tele_l(i,t)+ang_tele_extra_l(i,t)
            ang_r = lambda i,t: tele_r(i,t)+ang_tele_extra_r(i,t)


        elif option=='wavefront':
            lim=1e-10
            i_right = lambda i: PAA_LISA.utils.i_slr(i)[2]

            ang_l = lambda i,t: pack.functions.rotate_tele_wavefront(self.wfe,self.aim0,PAA_LISA.utils.get_link(i,'l'),t,count_max=np.inf,lim=lim,scale=1)[0]
            ang_r = lambda i,t: pack.functions.rotate_tele_wavefront(self.wfe,self.aim0,PAA_LISA.utils.get_link(i,'r'),t+self.wfe.data.L_rl_func_tot(i_right(i),t),count_max=np.inf,lim=lim,scale=1)[2]

        self.tele_option = option
        return [ang_l,ang_r]


    def tele_aim(self,wfe,method=False,lim=1e-10,tele_ang_extra=True,option='wavefront',iteration=0):
        if method==False:
            method=self.tele_method
        else:
            self.tele_method=method
        
        try:
            print('The telescope control method is: '+method)
        except:
            print('The telescope control method is: user defined')

        print(' ')

        if method=='no_control':
            if tele_ang_extra==False:
                offset_l = {'1':0,'2':0,'3':0}
                offset_r = {'1':0,'2':0,'3':0}
            elif tele_ang_extra==True:
                try:
                    offset_l = self.wfe.mean_angin_l
                    offset_r = self.wfe.mean_angin_r
                except:
                    self.wfe.do_mean_angin()
                    offset_l = self.wfe.mean_angin_l
                    offset_r = self.wfe.mean_angin_r
            else:
                [offset_l,offset_r] = tele_ang_extra

            self.offset =[offset_l,offset_r]
            self.tele_l_ang = lambda i,t: np.radians(-30)+offset_l[str(i)]
            self.tele_r_ang = lambda i,t: np.radians(30)+offset_r[str(i)]
        
        elif method=='full_control':
            [self.tele_ang_l_fc,self.tele_ang_r_fc] = self.tele_control_ang_fc(option=option)
            self.tele_l_ang = self.tele_ang_l_fc
            self.tele_r_ang = self.tele_ang_r_fc

        elif 'min' in method:
            if 'spot'==method.split('_')[-1]:
                self.tele_l_ang = lambda i,t: pack.functions.spotsize_limit(self.wfe,self.aim0,i,t,'l',limit=-self.wfe.spotsize)
                self.tele_r_ang = lambda i,t: pack.functions.spotsize_limit(self.wfe,self.aim0,i,t,'r',limit=-self.wfe.spotsize)
        elif 'max' in method:
            if 'spot'==method.split('_')[-1]:
                self.tele_l_ang = lambda i,t: pack.functions.spotsize_limit(self.wfe,self.aim0,i,t,'l',limit=self.wfe.spotsize)
                self.tele_r_ang = lambda i,t: pack.functions.spotsize_limit(self.wfe,self.aim0,i,t,'r',limit=self.wfe.spotsize)


        elif 'SS' in method:
            ### Obtai full control
            [self.tele_ang_l_fc,self.tele_ang_r_fc] = self.tele_control_ang_fc(option=option)
            if option=='center':
                print('Still have to implement') #...adjust

            m = method.split('SS_')[-1]
            print('SS by '+m)
            print('')
            ret={}
            t_all={}
            ang_adjust={}
            for link in range(1,4):
                #if self.count==0:
                try:
                    self.beam_r_ang
                except AttributeError:
                    self.beam_l_ang = self.wfe.data.ang_out_l
                    self.beam_r_ang = self.wfe.data.ang_out_r
                ret,t_all,ang_adjust = pack.functions.get_SS(self.wfe,self,link,ret=ret,m=m,t_all=t_all,ang_output=ang_adjust)
                #else:
                #    ret,t_all,tele_ang_adjust,PAAM_ang_adjust = pack.functions.get_SS(self.wfe,self.aim_old,link,ret=ret,m=m,t_all=t_all,tele_ang=tele_ang_adjust,PAAM_ang=PAAM_ang_adjust)

            self.t_adjust = t_all
            self.tele_ang_adjust = ang_adjust

            self.tele_l_ang_SS = lambda i,t: ret['tele'][str(i)]['l'](t)
            self.tele_r_ang_SS = lambda i,t: ret['tele'][str(i)]['r'](t)
            #self.PAAM_l_ang_SS = lambda i,t: ret['PAAM'][str(i)]['l'](t)
            #self.PAAM_r_ang_SS = lambda i,t: ret['PAAM'][str(i)]['r'](t)


            self.tele_l_ang = self.tele_l_ang_SS
            self.tele_r_ang = self.tele_r_ang_SS
        
        elif type(method)==list and method[0]=='Imported pointing':
            print(method[0])
            self.tele_l_ang = lambda i,t: pack.functions.get_tele_SS(False,False,i,t,'l',x=method[1]['SC'+str(i)+', left']['x'],y=method[1]['SC'+str(i)+', left']['y'])
            self.tele_r_ang = lambda i,t: pack.functions.get_tele_SS(False,False,i,t,'r',x=method[1]['SC'+str(i)+', right']['x'],y=method[1]['SC'+str(i)+', right']['y'])
            
            t_adjust={}
            tele_ang_adjust = {}
            for i in range(1,4):
                t_adjust[str(i)]={}
                tele_ang_adjust[str(i)]={}
                t_adjust[str(i)]['l']=method[1]['SC'+str(i)+', left']['x']
                t_adjust[str(i)]['r']=method[1]['SC'+str(i)+', right']['x']
                tele_ang_adjust[str(i)]['l']=method[1]['SC'+str(i)+', left']['y']
                tele_ang_adjust[str(i)]['r']=method[1]['SC'+str(i)+', right']['y']

            
            self.t_adjust = t_adjust
            self.tele_ang_adjust = tele_ang_adjust

        else:
            raise ValueError('Please select valid telescope pointing method')


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

    
    def PAAM_control_ang_fc(self,option='center'):
        self.option_PAAM=option
        
        if option=='center':
            ang_l = lambda i,t: pack.functions.rotate_PAA_wavefront(self.wfe,self.aim_old,i,t,'l','yoff')
            ang_r = lambda i,t: pack.functions.rotate_PAA_wavefront(self.wfe,self.aim_old,i,t,'r','yoff')
        
        elif option=='wavefront':
            ang_l = lambda i,t: pack.functions.rotate_PAA_wavefront(self.wfe,self.aim_old,i,t,'l','angy')
            ang_r = lambda i,t: pack.functions.rotate_PAA_wavefront(self.wfe,self.aim_old,i,t,'r','angy')

        self.PAAM_option = option
        return [ang_l,ang_r]

        
        #i_left = lambda i: PAA_LISA.utils.i_slr(i)[1]
        #i_right = lambda i: PAA_LISA.utils.i_slr(i)[2]

        #print('PAAM pointing strategy: '+option)
        ## Obtains functions for optimal telescope pointing vector
        ##if option!='wavefront':
        ##    delay_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='PAAM',option=option)[2]
        ##    delay_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='PAAM',option=option)[2]

        #
        #if option=='wavefront':
        #    #ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t+delay_l(i,t),'r',component='PAAM',option=option)[1]
        #    #ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t+delay_r(i,t),'l',component='PAAM',option=option)[1]
        #    if self.count>0:
        #        ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i,t,'l',component='PAAM',option=option)[1]
        #        ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i,t,'r',component='PAAM',option=option)[1]

        #elif option=='center':
        #    ang_PAAM_extra_l = lambda i,t: self.get_aim_accuracy(i_left(i),t,'r',component='PAAM',option=option)[1]
        #    ang_PAAM_extra_r = lambda i,t: self.get_aim_accuracy(i_right(i),t,'l',component='PAAM',option=option)[1]

        #if self.sampled_on==True:
        #    [[tele_l,tele_r],[beam_l,beam_r]] = self.get_funcions_from_sampling(self.sampled_val)
        #else:
        #    tele_l = self.aim_old.tele_l_ang
        #    tele_r = self.aim_old.tele_r_ang
        #    beam_l = self.aim_old.beam_l_ang
        #    beam_r = self.aim_old.beam_r_ang

        #
        #if option=='wavefront':
        #    if self.count>0:
        #        self.PAAM_ang_l_fc = lambda i,t: ang_PAAM_extra_l(i,t)
        #        self.PAAM_ang_r_fc = lambda i,t: ang_PAAM_extra_r(i,t)
        #    else:
        #        self.PAAM_ang_l_fc = self.aim_old.beam_l_ang
        #        self.PAAM_ang_r_fc = self.aim_old.beam_r_ang

        #elif option=='center':
        #    self.PAAM_ang_l_fc = lambda i,t: beam_l(i,t)+ang_PAAM_extra_l(i,t)
        #    self.PAAM_ang_r_fc = lambda i,t: beam_r(i,t)+ang_PAAM_extra_r(i,t)




    def PAAM_control(self,method=False,dt=3600*24,jitter=False,tau=1,mode='overdamped',PAAM_ang_extra=False,option='center'):
        if method==False:
            method = self.PAAM_method
        else:
            self.PAAM_method = method

        print('The PAAM control method is: ' +method)
        print(' ')

        #if self.init_set==False:
        #    try:
        #        self.PAAM_control_ang_fc(option=option)
        #        ang_fc_l = lambda i,t: self.PAAM_ang_l_fc(i,t)
        #        ang_fc_r = lambda i,t: self.PAAM_ang_r_fc(i,t)
        #    except AttributeError,e:
        #        if option=='center':
        #            ang_fc_l = self.wfe.data.PAA_func['l_out']
        #            ang_fc_r = self.wfe.data.PAA_func['r_out']
        #            pass
        #        else:
        #            raise AttributeError(str(e))
        #    self.PAAM_fc_ang_l = ang_fc_l
        #    self.PAAM_fc_ang_r = ang_fc_r

        #ang_fc_l = lambda i,t: self.wfe.data.PAA_func['l_out'](i,t)
        #ang_fc_r = lambda i,t: self.wfe.data.PAA_func['r_out'](i,t)
        #ang_fc_l = lambda i,t: self.PAAM_ang_l_fc(i,t)*2
        #ang_fc_r = lambda i,t: self.PAAM_ang_r_fc(i,t)*2
        #self.PAAM_fc_ang_l = ang_fc_l
        #self.PAAM_fc_ang_r = ang_fc_r

        # Obtaining PAAM angles for 'fc' (full_control), 'nc' (no_control) and 'SS' (step and stair)
        
        if method=='full_control':
            [ang_l,ang_r] = self.PAAM_control_ang_fc(option=option)
            #except UnboundLocalError:
            #    ang_l=self.wfe.data.PAA_func['l_out']
            #    ang_r=self.wfe.data.PAA_func['r_out']

        elif method=='no_control':
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
 
        self.get_coordinate_systems(iteration_val=False,option='self')
        
        return self
    
    
    def get_beam_coor(self,i,t,tele_l_ang,tele_r_ang,beam_l_ang,beam_r_ang):
        # Calculating new pointing vectors and coordinate system
        beam_l_coor = pack.functions.beam_coor_out(self.wfe,i,t,tele_l_ang(i,t),beam_l_ang(i,t),self.offset_tele['l'])
        beam_r_coor = pack.functions.beam_coor_out(self.wfe,i,t,tele_r_ang(i,t),beam_r_ang(i,t),self.offset_tele['r'])

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

