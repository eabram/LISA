from imports import *
from functions import *
from parameters import *
import calc_values

import PAA_LISA
import NOISE_LISA

class plot_func():
    def __init__(self,wfe,**kwargs):
        self.wfe = wfe
        self.dt = kwargs.pop('dt',3600)
        self.make_t_plot(wfe,dt=self.dt)
        self.ttl_sample_all={}
        self.save_fig()
        self.override = kwargs.pop('override',True)
    def save_fig(self,wfe=False,directory=False,extra_folder=False):
        if directory==False:
            directory = 'Figures/TTL/'

        if wfe==False:
            wfe=self.wfe
        extra = wfe.aim.tele_method+'_'+wfe.aim.PAAM_method

        directory = os.getcwd()+'/'+directory+'/'+extra+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory



    def make_t_plot(self,wfe=False,t0=False,tend=False,dt=3600):
        if wfe==False:
            wfe=self.wfe
        if t0==False:
            t0 = wfe.t_all[1]
        if tend==False:
            tend = wfe.t_all[-2]
        N = int(np.round((tend-t0)/dt))+1
        self.t_plot = np.linspace(t0,tend,N)
   
    def do_savefig(self,f,title,directory=False,override='default'):
        if override=='default':
            override=self.override
        if directory==False:
            directory = self.directory
        
        lst = []
        excists=False
        n=0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png") and title in file:
                    excists=True
                    print(file)
                    N = file.split('(')[-1].split(')')[0]
                    print(N)
                    if '.' not in N:
                        check = int(N)
                        if check>n:
                            n=check
        if excists==True:
            if n==0:
                if override==True:
                    n=''
                else:
                    n='(1)'
            else:
                if override==True:
                   n='('+str(n)+')'
                else:
                   n='('+str(n+1)+')'
        else:
            n=''

        f.savefig(directory+title+n+'.png')
        print(title+n+'.png'+' saved in '+directory)

        return 0


    def plot_piston(self,wfe=False,dt=False,title_extr='',SC=3):
        if dt != False:
            self.make_t_plot(dt=dt)

        if wfe==False:
            wfe=self.wfe

        print(wfe.aim.tele_method,wfe.aim.tele_method)
        piston_mean_l={}
        piston_var_l={}
        piston_mean_r={}
        piston_var_r={}
        
        for i in range(1,SC+1):
            piston_mean_l[str(i)]=[]
            piston_var_l[str(i)]=[]
            piston_mean_r[str(i)]=[]
            piston_var_r[str(i)]=[]
            for t in self.t_plot:
                print(t/self.t_plot[-1])
                calc = wfe.piston_val_l(i,t)
                piston_mean_l[str(i)].append(calc[0])
                piston_var_l[str(i)].append(calc[1])

                calc = wfe.piston_val_r(i,t)
                piston_mean_r[str(i)].append(calc[0])
                piston_var_r[str(i)].append(calc[1])

        
        f,ax = plt.subplots(4,3,figsize=(20,20))
        plt.subplots_adjust(hspace=0.6,wspace=0.2)
        f.suptitle('Telescope control: '+wfe.aim.tele_method+', PAAM control: '+ wfe.aim.PAAM_method)

        for i in piston_mean_l.keys():
            ax[0,int(i)-1].plot(self.t_plot/day2sec,piston_mean_l[i],label='SC'+i+', left')
            if SC!=1:
                ax[1,int(i)-1].plot(self.t_plot/day2sec,piston_var_l[i],label='SC'+i+', left')
                ax[2,int(i)-1].plot(self.t_plot/day2sec,piston_mean_r[i],label='SC'+i+', right')
                ax[3,int(i)-1].plot(self.t_plot/day2sec,piston_var_r[i],label='SC'+i+', right')

        i_label=['Mean armlength','Mean variance armlength','Mean armlength','Mean variance armlength']
        for i in range(0,len(ax)):
            for j in range(0,len(ax[i])):
                ax[i,j].set_xlabel('Time (sec)')
                if i%2==0:
                    ax[i,j].set_ylabel('Distance (m)')
                else:
                    ax[i,j].set_ylabel('Distance^2 (m^2)')
                ax[i,j].legend(loc='best')
                ax[i,j].set_title(i_label[i])

        direct = self.directory+'Piston'+title_extr+'.png'
        self.do_savefig(f,'Piston')

        return f,ax

    def plot_ang(self,i,side,wfe=False,dt=False):

        if wfe==False:
            wfe=self.wfe
        tele_l,PAAM_l,tele_r,PAAM_r = NOISE_LISA.calc_values.ang(wfe)

        [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
        if side=='l':
            tele = tele_l
            PAAM = PAAM_r
            i_next = i_left

        elif side=='r':
            tele = tele_r
            PAAM = PAAM_l
            i_next = i_right

        for ikey in tele.keys():
            f,ax = plt.subplots(2,3,figsize=(15,15))
            plt.subplots_adjust(hspace=0.6,wspace=0.2)
            f.suptitle('Transmitting telescope and receiving PAAM pointing, SC'+str(i_self)+', side'+side+', iter='+ikey)
            count=0
            for key_tele in tele[ikey].keys():
                for key_PAAM in tele[ikey][key_tele].keys():
                    tele_calc=[]
                    PAAM_calc=[]
                    for t in self.t_plot:
                        tele_calc.append(tele[ikey][key_tele][key_PAAM](i_self,t))
                        PAAM_calc.append(PAAM[ikey][key_tele][key_PAAM](i_next,t))
                    #tele_calc=np.arrray(tele_calc)
                    #PAAM_calc=np.arrray(PAAM_calc)

                    ax[0][count].plot(self.t_plot/day2sec,tele_calc,label='PAAM: '+key_PAAM)
                    ax[1][count].plot(self.t_plot/day2sec,PAAM_calc,label='PAAM: '+key_PAAM)
                ax[0][count].set_title('Telescope angle for telescope control='+key_tele)
                ax[1][count].set_title('PAAM angle for telescope control='+key_tele)

                ax[0][count].legend(loc='best')
                ax[1][count].legend(loc='best')

                ax[0][count].set_xlabel('Time (days)')
                ax[1][count].set_xlabel('Time (days)')
                ax[0][count].set_ylabel('Angle (rad)')
                ax[1][count].set_ylabel('Angle (micro rad)')
                count = count+1

            self.do_savefig(f,'Pointing_SC'+str(i)+'_iter'+ikey+'.png')

        return 0



    def plot_ttl(self,i,side,wfe=False,title='',dt=False):
        if wfe==False:
            wfe = self.wfe
        title=' of '+title 
        ttl = wfe.ttl_val
        t_vec = wfe.t_all

        if dt!=False:
            self.make_t_plot(dt=dt)
        t_plot = self.t_plot
            
        if side=='l':
            side=0
        elif side =='r':
            side=1

        ttl_sample={}
        for t in t_plot:
            for k in ttl.keys():
                if k not in ttl_sample.keys():
                    ttl_sample[k]=[]
                ttl_sample[k].append(ttl[k][side](i,t))
        
        for k in ttl_sample.keys():
            ttl_sample[k] = np.array(ttl_sample[k])

        f,ax = plt.subplots(len(ttl_sample.keys()),1,figsize=(20,10))
        plt.subplots_adjust(hspace=0.6,wspace=0.2)
        for k in range(0,len(ttl_sample.keys())):
            key = ttl_sample.keys()[k]
            y = np.array(ttl_sample[key])
            ax[k].plot(t_plot,y,label='SC'+str(i))
            ax[k].set_title(key)
            ax[k].set_xlabel('time (s)')
            ax[k].set_ylabel('Length (m)')

        f.suptitle('TTL'+title+', telescope control='+wfe.tele_control+' , PAAM control = '+wfe.PAAM_control_method)
        
        fig_title = wfe.tele_control+'_'+wfe.PAAM_control_method
        f.savefig(self.directory+fig_title+'.png')
        
        self.f_ttl = f
        self.ttl_sample = ttl_sample
        key_tele = wfe.tele_control
        key_PAAM = wfe.PAAM_control_method
        if key_tele not in self.ttl_sample_all.keys():
            self.ttl_sample_all[key_tele]={}
        self.ttl_sample_all[key_tele][key_PAAM] = ttl_sample

        return [[t_plot,ttl_sample],[f,ax]]

    def plot_ttl_overview(self,ttl_sample_all=False):
        t_plot = self.t_plot
        if ttl_sample_all==False:
            ttl_sample_all = self.ttl_sample_all

        f,ax = plt.subplots(4,1,figsize=(20,10))
        plt.subplots_adjust(hspace=0.6,wspace=0.2)
        ttl0 = ttl_sample_all['no control']['nc']
        for tele_key in ttl_sample_all.keys():
            for PAAM_key in ttl_sample_all[tele_key].keys():
                ttl = ttl_sample_all[tele_key][PAAM_key]
                
                for k in range(0,len(ttl.keys())):
                    key = ttl.keys()[k]
                    y = ttl[key]-ttl0[key]
                    ax[k].semilogy(t_plot,y,label='tele='+tele_key+', PAAM='+PAAM_key)
        
        for i in range(0,len(ax)):
            ax[i].legend(loc='best')
            ax[i].set_title(ttl.keys()[i])
            ax[i].set_xlabel('time (s)')
            ax[i].set_ylabel('TTL (m)')
        f.suptitle('TTL difference compared to no telescope and PAAM control')

        f.savefig('Figures/TTL/Overview.png')

    def plot_P(self,i,side='l'):
        
        y = []
        for t in self.t_plot:
            y.append(self.wfe.P_calc(i,t,side=side))
        plt.plot(self.t_plot,y)
        plt.title('Power for SC'+str(i)+', side='+side)
        plt.xlabel('Time (s)')
        plt.ylabel('Relative power')
        plt.show()

