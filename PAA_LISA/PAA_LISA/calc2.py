from imports import *

year2sec=32536000
day2sec=year2sec/365.25
c=300000000

class PAA():
    def __init__(self,para,**kwargs):
        for k in para:
            globals()[k] = para[k]
            setattr(self,k,para[k])

        self.home = kwargs.pop('home',os.getcwd())
        self.filename = kwargs.pop('filename','')
        self.directory_imp = kwargs.pop('directory_imp','')
        self.read_max = kwargs.pop('read_max','all')
        self.num_back = kwargs.pop('num_back',0)
        self.plot_on = kwargs.pop('plot_on',True)
        self.scale = kwargs.pop('scale','Default')
        self.relativistic = kwargs.pop('relativistic',True)
        if self.scale=='Default':
            print('Getting scale by filename:')
            a = self.filename
            a1 = a.split('.')[0]
            a1 = a1.split('_')
            for k in range(0,len(a1)):
                if 'scale' == a1[k]:
                    self.scale = float(a1[k+1])
                    print(self.scale)
                    
        else:
            print(self.scale)
        print('')
        
        self.method = kwargs.pop('method','fsolve')        
        self.dir_savefig = kwargs.pop('dir_savefig',os.getcwd())
        self.dir_extr = kwargs.pop('dir_extr','')
        self.new_folder = kwargs.pop('new_folder',True)
        self.timeunit = kwargs.pop('timeunit','Default')
        if self.timeunit=='Default':
            print('Getting timestep by filename:')
            a = self.filename
            a1 = a.split('.')[0]
            a1 = a1.split('_')
            for k in range(0,len(a1)):
                if 'timestep' == a1[k]:
                    self.timeunit = a1[k+1]
                    print(self.timeunit)
            if self.timeunit!='days' and self.timeunit!='seconds':
                print('Could not obtain proper timestep')
        else:
            print(self.timeunit)
        print('')
            
        self.delay = kwargs.pop('delay',True)
        self.LISA_opt = kwargs.pop('LISA_opt',False)
        self.arm_influence = kwargs.pop('arm_influence',True)
        self.tstep = kwargs.pop('tstep',False)
        self.valorfunc = kwargs.pop('valorfunc','Value')
        self.calc_method = kwargs.pop('calc_method','Waluschka')
        print(self.calc_method)
        self.abb = kwargs.pop('abberation',False)
    
    
    def PAA_func(self):
        import PAA_LISA.utils as utils
        print('')
        print('Importing Orbit')
        tic=time.clock()
        Orbit=orbit(home=self.home,filename=self.filename,directory_imp=self.directory_imp,num_back=self.num_back,scale=self.scale,read_max=self.read_max,plot_on=False,timeunit=self.timeunit,LISA_opt=self.LISA_opt)
        print(str(Orbit.linecount)+' datapoints')
        self.orbit = Orbit
        utils.LISA_obj(self,type_select=self.LISA_opt)
        print('Done in '+str(time.clock()-tic))
        self.SC = range(1,4)

        # Calculations
        LA=utils.la()
        v_l_func_tot=[]
        v_r_func_tot=[]
        u_ltest_func_tot=[]
        u_rtest_func_tot=[]
        u_l0test_func_tot=[]
        u_r0test_func_tot=[]
        L_sl_func_tot=[]
        L_sr_func_tot=[]
        L_rl_func_tot=[]
        L_rr_func_tot=[]
        v_l_stat_func_tot=[]
        v_r_stat_func_tot=[]
        pos_func=[]

        for i in range(1,4):
            #--- Obtaining Velocity
            utils.velocity_func(self,hstep=100)
            utils.velocity_abs(self,hstep=100)

            [[v_l_func,v_r_func,u_l_func,u_r_func],[L_sl_func,L_sr_func,L_rl_func,L_rr_func],[u_l0_func,u_r0_func]] = utils.send_func(self,i,calc_method = self.calc_method)
            #[[v_l_func,v_r_func],[L_sl_func,L_sr_func,L_rl_func,L_rr_func]] = utils.send_func(self,i,calc_method = self.calc_method)

            v_l_func_tot.append(v_l_func)
            v_r_func_tot.append(v_r_func)
            u_ltest_func_tot.append(u_l_func)
            u_rtest_func_tot.append(u_r_func)
            u_l0test_func_tot.append(u_l0_func)
            u_r0test_func_tot.append(u_r0_func)
            
            L_sl_func_tot.append(L_sl_func)
            L_sr_func_tot.append(L_sr_func)
            L_rl_func_tot.append(L_rl_func)
            L_rr_func_tot.append(L_rr_func)
            
            [i_self,i_left,i_right] = utils.i_slr(i)
            v_l_stat_func_tot.append(utils.get_armvec_func(self,i_self,'l'))
            v_r_stat_func_tot.append(utils.get_armvec_func(self,i_self,'r'))
            pos_func.append(utils.func_pos(self,i))

        self.v_l_func_tot = utils.func_over_sc(v_l_func_tot)
        self.v_r_func_tot = utils.func_over_sc(v_r_func_tot)
        self.u_l_func_tot = lambda i,t: utils.get_receiving(self,i,t,'l')
        self.u_r_func_tot = lambda i,t: utils.get_receiving(self,i,t,'r')
        self.u_ltest_func_tot = utils.func_over_sc(u_ltest_func_tot)
        self.u_rtest_func_tot = utils.func_over_sc(u_rtest_func_tot)
        self.u_l0test_func_tot = utils.func_over_sc(u_l0test_func_tot)
        self.u_r0test_func_tot = utils.func_over_sc(u_r0test_func_tot)

        self.L_sl_func_tot = utils.func_over_sc(L_sl_func_tot)
        self.L_sr_func_tot = utils.func_over_sc(L_sr_func_tot)
        self.L_rl_func_tot = utils.func_over_sc(L_rl_func_tot)
        self.L_rr_func_tot = utils.func_over_sc(L_rr_func_tot)
        
        self.v_l_stat_func_tot = utils.func_over_sc(v_l_stat_func_tot)
        self.v_r_stat_func_tot = utils.func_over_sc(v_r_stat_func_tot)
        self.n_func = lambda i,t: LA.unit(np.cross(self.v_l_stat_func_tot(i,t),self.v_r_stat_func_tot(i,t)))
        self.r_func = lambda i,t: utils.r_calc(self.v_l_stat_func_tot(i,t),self.v_r_stat_func_tot(i,t),i)
        self.pos_func = utils.func_over_sc(pos_func)

        self.v_l_in_func_tot = lambda i,t: LA.inplane(self.v_l_func_tot(i,t),self.n_func(i,t))
        self.v_r_in_func_tot = lambda i,t: LA.inplane(self.v_r_func_tot(i,t),self.n_func(i,t))
        self.u_l_in_func_tot = lambda i,t: LA.inplane(self.u_l_func_tot(i,t),self.n_func(i,t))
        self.u_r_in_func_tot = lambda i,t: LA.inplane(self.u_r_func_tot(i,t),self.n_func(i,t))
        self.v_l_out_func_tot = lambda i,t: LA.outplane(self.v_l_func_tot(i,t),self.n_func(i,t))
        self.v_r_out_func_tot = lambda i,t: LA.outplane(self.v_r_func_tot(i,t),self.n_func(i,t))
        self.u_l_out_func_tot = lambda i,t: LA.outplane(self.u_l_func_tot(i,t),self.n_func(i,t))
        self.u_r_out_func_tot = lambda i,t: LA.outplane(self.u_r_func_tot(i,t),self.n_func(i,t))
        

        ##--- Obtaining Velocity
        #utils.velocity_func(self,hstep=100)
        #utils.velocity_abs(self,hstep=100)

        #--- Obtaining PAA --- 
        print('Abberation: '+str(self.abb))
        selections=['l_in','l_out','r_in','r_out']
        PAA_func_val={}
        PAA_func_val[selections[0]] = lambda i,t: utils.calc_PAA_lin(self,i,t)
        PAA_func_val[selections[1]] = lambda i,t: utils.calc_PAA_lout(self,i,t)
        PAA_func_val[selections[2]] = lambda i,t: utils.calc_PAA_rin(self,i,t)
        PAA_func_val[selections[3]] = lambda i,t: utils.calc_PAA_rout(self,i,t)
        PAA_func_val['l_tot'] = lambda i,t: utils.calc_PAA_ltot(self,i,t)
        PAA_func_val['r_tot'] = lambda i,t: utils.calc_PAA_rtot(self,i,t)

        self.PAA_func = PAA_func_val 
       
        self.ang_breathing_din = lambda i, time: LA.angle(self.v_l_func_tot(i,time),self.v_r_func_tot(i,time))
        self.ang_breathing_stat = lambda i, time: LA.angle(self.v_l_stat_func_tot(i,time),self.v_r_stat_func_tot(i,time))
        
        self.ang_in_l = lambda i,t: LA.ang_in(self.v_l_func_tot(i,t),self.n_func(i,t),self.r_func(i,t))
        self.ang_in_r = lambda i,t: LA.ang_in(self.v_r_func_tot(i,t),self.n_func(i,t),self.r_func(i,t))
        self.ang_out_l = lambda i,t: LA.ang_out(self.v_l_func_tot(i,t),self.n_func(i,t))
        self.ang_out_r = lambda i,t: LA.ang_out(self.v_r_func_tot(i,t),self.n_func(i,t))

        return self #...adjust


