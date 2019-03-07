from imports import *
from functions import *
from parameters import *
import calc_values

import PAA_LISA
import NOISE_LISA

from matplotlib.font_manager import FontProperties

def compare_methods(wfe,SC,side,read_folder=False,ret=False,meas_plot='all',methods1=['no_control','no_control'],methods2=['full_control','full_control'],lim=[3,-3]):
    def get_output(ret,methods,SC,side):
        return ret[methods[0]][methods[1]][str(methods[2])][methods[3]]

    def get_values(wfe,ret,meas,i,side,xref,lim='Default'):
        if lim=='Default':
            lim=[0,len(wfe.t_all)]
        status=True
        try:
            if side=='l':
                d = ret[meas]['SC'+str(i)+', left']
            elif side =='r':
                d = ret[meas]['SC'+str(i)+', right']
            x_all = d['x']
            y_all = d['y']
            x = x_all[lim[0]:lim[1]]
            if len(x_all)*3==len(y_all):
                y = y_all[lim[0]*3:(lim[1]-lim[0])*3+lim[0]*3]
            else:
                y = y_all[lim[0]:lim[1]]
        except KeyError, e:
            print(meas)
            print(e)
            # = np.array([np.nan]*(lim[1]-lim[0]))
            # = np.array([np.nan]*(lim[1]-lim[0]))
            status = False
            pass
        #for i in d:
        #    x.append(i[0])
        #    y.append(i[1])
        #x = d['x'][lim[0]:lim[1]]
        #y = d['y'][lim[0]:lim[1]]
        
        if status==True:
            try:
                if len(x)!=len(y):
                    y_new=[]
                    for i in range(0,len(x)):
                        y_new.append(np.array([y[i*3],y[i*3+1],y[i*3+2]]))
                    y = np.array(y_new)        
                
                upperlimit=len(x)
                for i in range(1,len(x)):
                    if x[i]<x[i-1]:
                        upperlimit=i
                        break
                return x[0:upperlimit],y[0:upperlimit]
            except IndexError:
                print(meas)
                print(len(x),len(y))
                print(x)
                print(d)
                print(y)
                status=False
                pass

        if status==False:
            x = xref[lim[0]:lim[-1]]
            y = np.array([np.nan]*len(x))
            return x,y


#    def get_FOV(pl):
#        x,y = pl['beam_inc_tele_frame mean']
#        x2,y2 = pl['R_vec_tele_rec mean']
#        FOV = []
#        R=[]
#        for i in range(0,len(x)):
#            FOV.append(np.arccos(-y[i][0]/np.linalg.norm(y[i])))
#        for i in range(0,len(x2)):
#            R.append(np.linalg.norm(y2[i]))
#        
#        pl['FOV_calc mean'] = x,np.array(FOV)
#        pl['R mean'] = x,np.array(R)
#        return pl
        
    def max_power(wfe,SC,t,side):
        i=1
        I0 = wfe.P_L
        if side=='l':
            z = np.linalg.norm(wfe.data.u_l_func_tot(i,t))
        elif side=='r':
            z = np.linalg.norm(wfe.data.u_r_func_tot(i,t))
        A = (wfe.D**2)*(np.pi/4.0)
        return ((1.0/(np.float64(wfe.labda)*z))*A)**2
        #w = wfe.w(z)
        #w0 = wfe.w0_laser
        #return I0*((w0/w)**2)

    # Set default iterations and pointing options
    if len(methods1)==2:
        methods1.append(0)
    if len(methods2)==2:
        methods2.append(0)
    if len(methods1)==3:
        methods1.append('tele_wavefront__PAAM_wavefront')
    if len(methods2)==3:
        methods2.append('tele_wavefront__PAAM_wavefront')





    length = len(wfe.t_all)
    if lim[1]<0:
        lim[1]=length+lim[1]

    if read_folder==False:
        if ret==False:
            raise ValueError
    else:
        ret = NOISE_LISA.functions.read(direct=read_folder)

    class BreakIt(Exception): pass
    try:
        for k1 in ret.keys():
            for k2 in ret[k1].keys():
                for k3 in ret[k1][k2].keys():
                    for k31 in ret[k1][k2][k3].keys():
                        for k4 in ret[k1][k2][k3][k31].keys():
                            if meas_plot=='all':
                                meas_plot=ret[k1][k2][k3][k31].keys()
                            for k5 in ret[k1][k2][k3][k31][k4]:
                                xref = ret[k1][k2][k3][k31][k4][k5]['x']
                                print(k5)
                                raise BreakIt
    except BreakIt:
        pass


    ret1 = get_output(ret,methods1,SC,side)
    ret2 = get_output(ret,methods2,SC,side)

    pl1 = {}
    pl2 = {}
    for m in meas_plot:
        #print('Check')
        #print(wfe,ret1)
        #print(m,SC,side,xref,lim)
        x1,y1 = get_values(wfe,ret1,m,SC,side,xref,lim=lim)
        x2,y2 = get_values(wfe,ret2,m,SC,side,xref,lim=lim)
        pl1[m] = x1,y1
        pl2[m] = x2,y2
#    if 'FOV_calc mean' in meas_plot:
#        pl1 = get_FOV(pl1)
#        pl2 = get_FOV(pl2)

    meas = pl1.keys()
    meas.sort()

    ref={}
    ref['angx_tot mean'] = lambda t: 0
    ref['angy_tot mean'] = lambda t: 0
    if side=='l':
        ref['piston mean'] = lambda t: np.linalg.norm(wfe.data.u_l_func_tot(SC,t))
    elif side=='r':
        ref['piston mean'] = lambda t: np.linalg.norm(wfe.data.u_r_func_tot(SC,t))

    ref['R mean'] = ref['piston mean']
    ref['R_vec_tele_rec mean'] = ref['piston mean']
    if side=='l':
        ref['piston mean'] = lambda t: wfe.c*wfe.data.L_rl_func_tot(SC,t)
    elif side=='r':
        ref['piston mean'] = lambda t: wfe.c*wfe.data.L_rir_func_tot(SC,t)
    ref['power mean'] = lambda t: max_power(wfe,SC,t,side)
    #ref['FOV mean'] = lambda t: wfe.FOV
    if side=='l':
        scale=-1
    elif side=='r':
        scale=1
    ref['tele_ang mean'] = lambda t: np.radians(30)*scale
    if methods1[1]=='full_control' and methods2[1]=='full_control':
        if side=='l':
            ref['PAAM_ang mean'] = lambda t: -wfe.data.PAA_func['l_out'](SC,t)
        elif side=='r':
            ref['PAAM_ang mean'] = lambda t: -wfe.data.PAA_func['r_out'](SC,t)

    for m in meas:
        if m not in ref.keys():
            if 'FOV' in m:
                ret[m] = lambda t: wfe.FOV
            else:
                ref[m] = lambda t: 0

    unit={}
    for m in meas:
        if 'ang' in m or 'FOV' in m:
            unit[m] = ['Angle (microrad)',1e6]
        elif 'piston' in m or 'r ' == m[0:2]  or 'z_extra' in m or 'zoff' in m or 'R mean'==m:
            unit[m] = ['Distance (km)',0.001]
        elif 'power' in m:
            unit[m]=['Power (W)',1]
        else:
            unit[m]=['AU',1]

    f_all=[] 
    label1= 'tele: '+methods1[0]+', '+ methods1[3].split('_')[1]+', PAAM: '+methods1[1]+', '+ methods1[3].split('_')[4]+'\n'+'Iteration: '+str(methods1[2])
    label2= 'tele: '+methods2[0]+', '+ methods2[3].split('_')[1]+', PAAM: '+methods2[1]+', '+ methods2[3].split('_')[4]+'\n'+'Iteration: '+str(methods2[2])
    lim=[3,-3]
    print(lim)
    for m in range(0,len(meas_plot)):
        print(meas_plot[m])
        print(methods1,methods2)
        try:
            x1,y1 = pl1[meas_plot[m]]
            x2,y2 = pl2[meas_plot[m]]
            x1_sec = x1[lim[0]:lim[1]]
            x1 = x1_sec/wfe.day2sec
            y1 = y1[lim[0]:lim[1]]*unit[meas_plot[m]][1]
            try:
                y1[0][0]
                y1 = np.array([np.linalg.norm(i) for i in y1])
                title_add='|'
            except IndexError:
                title_add=''
                pass

            x2 = x2[lim[0]:lim[1]]/wfe.day2sec
            y2 = y2[lim[0]:lim[1]]*unit[meas_plot[m]][1]
            try:
                y2[0][0]
                y2 = np.array([np.linalg.norm(i) for i in y2])
            except IndexError:
                pass

            f,ax = plt.subplots(2,2,figsize=(10,12))
            plt.subplots_adjust(hspace=0.3,wspace=0.6)
            f.suptitle(title_add+meas_plot[m]+title_add)
            ax[0,0].plot(x1,y1)
            ax[0,0].set_title(label1,pad=20)

            ax[0,1].plot(x2,y2)
            ax[0,1].set_title(label2,pad=20)

            y_ref = np.array([ref[meas_plot[m]](t)*unit[meas_plot[m]][1] for t in x1_sec])
            ax[1,0].plot(x1,y1-y_ref,label=label1)
            ax[1,0].plot(x2,y2-y_ref,label=label2)
            ax[1,0].legend(loc='best',bbox_to_anchor=(1.2, -0.12))
            ax[1,0].set_title('Relative difference w.r.t. optimal/standard',pad=20)

            ax[1,1].plot(x1,y1,label=label1)
            ax[1,1].plot(x2,y2,label=label2)
            ax[1,1].plot(x1,y_ref,color='r',linestyle='--')
            ax[1,1].legend(loc='best',bbox_to_anchor=(1.2, -0.12))
            ax[1,1].set_title('Overview',pad=20)

            for i in range(0,len(ax)):
                for j in range(0,len(ax[i])):
                    ax[i,j].set_xlabel('Time (days)')
                    ax[i,j].set_ylabel(unit[meas_plot[m]][0])


            f_all.append([meas_plot[m],label1,label2,f])
            plt.close('all')
        except KeyError:
            pass
    #if ready==True:
    return f_all,ret1,ret2,meas_plot,pl1,pl2

class plot_func():
    def __init__(self,wfe,**kwargs):
        self.wfe = wfe
        self.dt = kwargs.pop('dt',3600)
        self.make_t_plot(wfe,dt=self.dt)
        self.ttl_sample_all={}
        self.save_fig()
        self.override = kwargs.pop('override',True)
        self.show = kwargs.pop('show',False)

    def showfig(self,**kwargs):
        fig = kwargs.pop('fig',False)

        if self.show==True:
            plt.show()
        elif self.show==False:
            plt.close(fig)

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

        return f 


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
        ztilt_mean_l={}
        ztilt_mean_r={}
        ztilt_var_l={}
        ztilt_var_r={}
        
        for i in range(1,SC+1):
            piston_mean_l[str(i)]=[]
            piston_var_l[str(i)]=[]
            piston_mean_r[str(i)]=[]
            piston_var_r[str(i)]=[]
            ztilt_mean_l[str(i)]=[]
            ztilt_mean_r[str(i)]=[]
            ztilt_var_l[str(i)]=[]
            ztilt_var_r[str(i)]=[]

            for t in self.t_plot:
                print(t/self.t_plot[-1])
                calc = wfe.piston_val_l(i,t)
                piston_mean_l[str(i)].append(calc[0])
                piston_var_l[str(i)].append(calc[1])
                calc = wfe.ztilt_val_l(i,t)
                ztilt_mean_l[str(i)].append(calc[0])
                ztilt_var_l[str(i)].append(calc[1])

                calc = wfe.piston_val_r(i,t)
                piston_mean_r[str(i)].append(calc[0])
                piston_var_r[str(i)].append(calc[1])
                calc = wfe.ztilt_val_r(i,t)
                ztilt_mean_r[str(i)].append(calc[0])
                ztilt_var_r[str(i)].append(calc[1])


        num_subplots = [8,3]
        f,ax = plt.subplots(num_subplots[0],num_subplots[1],figsize=(2*5*num_subplots[1],5*num_subplots[0]))
        plt.subplots_adjust(hspace=0.6,wspace=0.2)
        f.suptitle('Telescope control: '+wfe.aim.tele_method+', PAAM control: '+ wfe.aim.PAAM_method)

        for i in piston_mean_l.keys():
            ax[0,int(i)-1].plot(self.t_plot/day2sec,piston_mean_l[i],label='SC'+i+', left')
            if SC!=1:
                ax[1,int(i)-1].plot(self.t_plot/day2sec,piston_var_l[i],label='SC'+i+', left')
                ax[2,int(i)-1].plot(self.t_plot/day2sec,piston_mean_r[i],label='SC'+i+', right')
                ax[3,int(i)-1].plot(self.t_plot/day2sec,piston_var_r[i],label='SC'+i+', right')
                ax[4,int(i)-1].plot(self.t_plot/day2sec,ztilt_mean_l[i],label='SC'+i+', left')
                ax[5,int(i)-1].plot(self.t_plot/day2sec,ztilt_var_l[i],label='SC'+i+', left')
                ax[6,int(i)-1].plot(self.t_plot/day2sec,ztilt_mean_r[i],label='SC'+i+', right')
                ax[7,int(i)-1].plot(self.t_plot/day2sec,ztilt_var_r[i],label='SC'+i+', right')


        i_label=['Mean armlength','Mean variance armlength','Mean armlength','Mean variance armlength','Mean armlength due to tilt','Mean variance armlength due to tilt','Mean armlength due to tilt','Mean variance armlength due to tilt']
        for i in range(0,len(ax)):
            for j in range(0,len(ax[i])):
                ax[i,j].set_xlabel('Time (sec)')
                if i%2==0:
                    ax[i,j].set_ylabel('Distance (m)')
                else:
                    ax[i,j].set_ylabel('Distance^2 (m^2)')
                ax[i,j].legend(loc='best')
                ax[i,j].set_title(i_label[i])
                ax[i,j].set_xlabel('Time (days)')
                ax[i,j].set_ylabel('Length (m)')

        direct = self.directory+'Piston'+title_extr+'.png'
        self.do_savefig(f,'Piston')
        
        self.showfig(fig=f)

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
            self.showfig(fig=f)


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

        self.showfig(fig=f)
        
        return [[t_plot,ttl_sample],[f,ax]]

    def plot_ttl_overview(self,ttl_sample_all=False):
        t_plot = self.t_plot
        if ttl_sample_all==False:
            ttl_sample_all = self.ttl_sample_all

        f,ax = plt.subplots(4,1,figsize=(20,10))
        plt.subplots_adjust(hspace=0.6,wspace=0.2)
        ttl0 = ttl_sample_all['no_control']['nc']
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

        self.showfig(fig=f)

    def plot_P(self,i,side='l'):
        
        y = []
        for t in self.t_plot:
            y.append(self.wfe.P_calc(i,t,side=side))
        plt.plot(self.t_plot,y)
        plt.title('Power for SC'+str(i)+', side='+side)
        plt.xlabel('Time (s)')
        plt.ylabel('Relative power')
        self.showfig()

