from imports import *
import matplotlib.pyplot 
import PAA_LISA
import NOISE_LISA

global LA

def get_nearest_smaller_value(lst,val):
    lst.sort()
    if val<lst[0]:
         pos = np.nan #...check if this holds
    else:
        for i in range(1,len(lst)):
            if val<lst[i] and val>=lst[i-1]:
                pos = i-1
                break

    return pos

def make_nan(function,t,lim):
    [a,b]=lim
    if t<a or t>b:
        return np.nan
    else:
        return function(t)

def string_length(l,string):
    while len(string)<l:
        string = '0'+string

    return string

def get_date(option='date'):
    now = datetime.datetime.now()
    if option=='date':
        ret=string_length(2,str(now.year))+string_length(2,str(now.month))+string_length(2,str(now.day))
    elif option=='time':
        ret=string_length(2,str(now.hour))+string_length(2,str(now.minute))+string_length(2,str(now.second))
    #date=date+'-'+dir_extr
    return ret

def get_folder(direct=False):
    if direct==False:
        date = get_date(option='date')
        direct = os.getcwd()+'/Results/'+date+'/'

    if not os.path.exists(direct):
        os.makedirs(direct)

    return direct

def savefig(f,title='',direct=True,newtime=False,extension='png'):
    
    if newtime==True:
        time = get_date(option='time')
    else:
        try:
            time
        except NameError:
            time='000000'
            pass
    
    date = get_date(option='date')

    if direct==True:
        direct = get_folder()
    
    if not os.path.exists(direct):
        os.makedirs(direct)
    
    title=direct+'/'+time+'-'+title+extension
    f.savefig(title)
    print('Saved as '+title)

    return 0

def flatten(y):
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

def nanfilter(l):
    l = flatten(l)
    l_copy = []
    for i in l:
        if i!=np.nan:
            l_copy.append(i)
    
    return l

def nanmean(l):
    return np.mean(nanfilter(l))




def write(inp,title='',direct ='',extr='',list_inp=False):
    date = get_date(option='date')
    time = get_date(option='time')
    
    if direct=='':
        direct=get_folder()
    direct=direct+'_'+extr+'/'
    if not os.path.exists(direct):
        os.makedirs(direct)


    title=date+'_'+time+'_'+title+'.txt'
    writefile = open(direct+'/'+title,'w')

    #if len(inp)==1:
    #    inp=[inp]
    
    if type(inp)==dict:
        inp_new = []
        for k in inp.keys():
            inp_new.append(inp[k])
        inp = inp_new
        del inp_new
    elif type(inp)!=list:
        inp=[inp]

    for m in inp:
        if type(m)==list:
            if len(m)==3 and 'Figure' in str(type(m[0])):
                f= m[0]
                ax = flatten(m[1])
                title=f._suptitle.get_text()
                print(title.split('iter_'))
                writefile.write('Title:: '+f._suptitle.get_text()+'\n')
                writefile.write('Iteration:: '+str(m[2])+'\n')
                for i in range(0,len(ax)):
                    ax_calc=ax[i]
                    ax_title = ax_calc.get_title()
                    line = 'ax_title:: '+ax_title
                    writefile.write(line+'\n')
                    
                    for l in range(0,len(ax_calc.lines)):
                        label = str(ax_calc.lines[l]._label)
                        writefile.write('Label:: '+label+'\n')
                        xy = ax_calc.lines[l]._xy
                        for k in xy:
                            writefile.write(str(k[0])+';'+str(k[1])+'\n')
        elif type(m)==tuple and type(m[4])==dict:
            for out in m[0:-2]:
                writefile.write(out+'\n')
            for k in sorted(m[-1].keys()):
                writefile.write(m[3]+' '+k+'\n')
                for SC in sorted(m[-1][k].keys()):
                    for side in sorted(m[-1][k][SC].keys()):
                        if side=='l':
                            side_wr='left'
                        elif side=='r':
                            side_wr='right'
                        writefile.write('Label:: SC'+SC+', '+side_wr+'\n')
                        for point in m[-1][k][SC][side]:
                             writefile.write(str(point[0])+';'+str(point[1])+'\n')


            

    writefile.close()

    print(title+' saved in:')
    print(direct)

    return direct

def rdln(line,typ='text'):
    if '[array(' in line:
        newline = line.split('array(')
        line = newline[-1].split(')')[0]+']'
        A = line[0:-1]
        #print(A)
        #print('')
        A = A.replace('[','')
        A = A.replace(']','')
        A = A.replace(' ','')
        A = A.split(',')
        #print(A)
        B=[]
        for i in A:
            B.append(np.float64(i))
        B = B
        #print(B,len(B))
        return [B]
    else:
        ret = line[0:-1]
        if typ=='float':
            return np.float64(ret)
        else:
            return ret

def read(filename='',direct=''):
    ret={}
    if direct=='':
        direct = get_folder()

    if filename=='':
        f_get=[]
        f_list=[]
        for (dirpath, dirnames, filenames) in os.walk(direct):
            #filenames.sort()
            for f in filenames:
                f_list.append(dirpath+'/'+f.split('/')[-1])

        filenames=f_list
    else:
        print('Please select filename or leave blank')

    
    try:
        filenames
        go =True
    except UnboundLocalError:
        print('Please select proper title and/or directory')
        go=False
        pass

    if go==True:
        for filename_select in filenames:
            #print(filenames)
            print('Reading '+filename_select)

            readfile = open(filename_select,'r')

            for line in readfile:
                if 'Title' in line:
                    key1 = rdln(line.split(':: ')[-1])
                    keys = rdln(line).replace(':',',').split(',')
                    print(keys)
                    key0 = (keys[3]+' ')[1:-1]
                    key1 = (keys[5]+' ')[1:-1]
                    if key0 not in ret.keys():
                        ret[key0] = {}
                    if key1 not in ret[key0].keys():
                        ret[key0][key1]={}
                elif 'Iteration' in line:
                    iteration = rdln(line.split(':: ')[-1])
                    if iteration not in ret[key0][key1].keys():
                        ret[key0][key1][iteration] = {}
                elif 'Option' in line:
                    option = rdln(line.split(':: ')[-1])
                    if option not in ret[key0][key1][iteration].keys():
                        ret[key0][key1][iteration][option]={}
                elif 'ax_title' in line:
                    key2 = rdln(line.split(':: ')[-1])
                    if key2 not in ret[key0][key1][iteration][option].keys():
                        ret[key0][key1][iteration][option][key2]={}
                elif 'Measurement' in line:
                    key2 = rdln(line.split(':: ')[-1])
                    if key2 not in ret[key0][key1][iteration][option].keys():
                        ret[key0][key1][iteration][option][key2]={}
 
                elif 'Label' in line:
                    key3 = rdln(line.split(':: ')[-1])
                    if key3 not in ret[key0][key1][iteration][option][key2].keys():
                        ret[key0][key1][iteration][option][key2][key3]={}
                        ret[key0][key1][iteration][option][key2][key3]['x']=np.array([])
                        ret[key0][key1][iteration][option][key2][key3]['y']=np.array([])

                else:
                    try:
                        del x,y 
                    except NameError:
                        pass
                    try:
                        [x,y] = line.split(';')
                        ret[key0][key1][iteration][option][key2][key3]['x'] = np.append(ret[key0][key1][iteration][option][key2][key3]['x'],rdln(x,typ='float'))
                        try:
                            ret[key0][key1][iteration][option][key2][key3]['y'] =np.append(ret[key0][key1][iteration][option][key2][key3]['y'],rdln(y,typ='float'))
                            value=True
                        except ValueError:
                            value=False
                    except ValueError,e:
                        print(e)
                        print(line)
                    if value==False:
                        #except ValueError:
                        ynew_list = rdln(y)[1:-1].split(' ')
                        ynew_write=[]
                        for ynew in ynew_list:
                            try:
                                ynew_write.append(np.float64(ynew))
                            except:
                                pass
                        ret[key0][key1][iteration][option][key2][key3]['y'] = np.append(ret[key0][key1][iteration][option][key2][key3]['y'],np.array(ynew_write))


            
            readfile.close()

    return ret


### Pointing functions

### Telescope pointing
def get_extra_angle(wfe,SC,side,component,tmin=False,tmax=False,ret='value'):
    if ret=='value':
        A = NOISE_LISA.calc_values.piston(wfe,SC=[SC],side=[side],dt=False,meas='R_vec_tele_rec')
        WF = A[3]['mean'][str(SC)][side]
        t=[]
        angx=[]
        angy=[]
        ang=[]
        if tmin==False:
            tmin = wfe.t_all[0]
        if tmax==False:
            tmax = wfe.t_all[-1]
        for i in range(0,len(WF)):
            vec = -WF[i][1]
            ang.append(PAA_LISA.la().angle(vec,np.array([1,0,0])))
            t.append(WF[i][0])
            if t[-1]>=tmin and t[-1]<=tmax:
                angx.append(np.sign(vec[2])*np.arctan(abs(vec[2]/vec[0])))
                angy.append(np.sign(vec[1])*np.arctan(abs(vec[1]/vec[0])))
        
        if component=='tele':
            return angx
        elif component=='PAAM':
            return angy

    elif ret=='function':
        vec = lambda t: -wfe.calc_piston_val(SC,t,side,ret='R_vec_tele_rec')
        if component=='tele':
            angx = lambda t: np.sign(vec(t)[2])*np.arctan(abs(vec(t)[2]/vec(t)[0]))
            return angx
        elif component=='PAAM':
            angy = lambda t: np.sign(vec(t)[1])*np.arctan(abs(vec(t)[1]/vec(t)[0]))
            return angy


def get_extra_ang_mean(wfe,component):
    offset_l=[]
    offset_r=[]
    for SC in range(1,4):
        offset_l.append(np.mean(get_extra_angle(wfe,SC,'l',component,ret='value')))
        offset_r.append(np.mean(get_extra_angle(wfe,SC,'r',component,ret='value')))

    return [offset_l,offset_r]

def get_wavefront_parallel(wfe,aim,i,t,side,PAAM_ang,ret,mode='opposite',precision=0,ksi=[0,0]):
    [i_self,i_left,i_right] = PAA_LISA.utils.i_slr(i)
    if mode=='opposite':
        if side=='l':
            tdel = wfe.data.L_sl_func_tot(i_self,t)
            if wfe.data.calc_method=='Waluschka':
                tdel0=tdel
            elif wfe.data.calc_method=='Abram':
                tdel0=0
            tele_ang = aim.tele_l_ang(i_self,t+tdel0)
            coor_start = beam_coor_out(wfe,i_self,t,tele_ang,PAAM_ang)
            coor_end = aim.tele_r_coor(i_left,t+tdel)
            start=aim.tele_l_start(i_self,t+tdel0)
            end=aim.tele_r_start(i_left,t+tdel)+coor_end[1]*ksi[1]+coor_end[2]*ksi[0]

        elif side=='r':
            tdel=wfe.data.L_sr_func_tot(i_self,t)
            if wfe.data.calc_method=='Waluschka':
                tdel0=tdel
            elif wfe.data.calc_method=='Abram':
                tdel0=0
            tele_ang = aim.tele_r_ang(i_self,t+tdel0)
            coor_start =  beam_coor_out(wfe,i_self,t,tele_ang,PAAM_ang)
            coor_end = aim.tele_l_coor(i_right,t+tdel)
            start = aim.tele_r_start(i_self,t+tdel0)
            end=aim.tele_l_start(i_right,t+tdel)+coor_end[1]*ksi[1]+coor_end[2]*ksi[0]

        [zoff,yoff,xoff]=LA.matmul(coor_start,end-start)
        if precision==0:
            R = zoff # Not precise
        elif precision==1:
            try:
               [piston,z_extra] = wfe.z_solve(xoff,yoff,zoff,ret='all')
            except:
                [piston,z_extra] = [np.nan,np.nan]
            R = wfe.R(piston)

        R_vec = np.array([(R**2-xoff**2-yoff**2)**0.5,yoff,xoff])
        tele_vec = LA.matmul(coor_start,-coor_end[0])
        angx_R = np.sign(R_vec[2])*abs(np.arctan(R_vec[2]/R_vec[0]))
        angy_R = np.sign(R_vec[1])*abs(np.arctan(R_vec[1]/R_vec[0]))
        angx_tele = np.sign(tele_vec[2])*abs(np.arctan(tele_vec[2]/tele_vec[0]))
        angy_tele = np.sign(tele_vec[1])*abs(np.arctan(tele_vec[1]/tele_vec[0]))
        angx = (angx_tele-angx_R)
        angy = (angy_tele-angy_R)
 
    elif mode=='self':
        if side=='l':
            tdel = wfe.data.L_rl_func_tot(i_self,t)
            if wfe.data.calc_method=='Waluschka':
                tdel0=tdel
            elif wfe.data.calc_method=='Abram':
                tdel0=0
            tele_ang = aim.tele_r_ang(i_left,t-tdel)
            coor_start = beam_coor_out(wfe,i_left,t-tdel,tele_ang,PAAM_ang)
            coor_end = aim.tele_l_coor(i_self,t)
            start = aim.tele_r_start(i_left,t-tdel)
            end = aim.tele_l_start(i_self,t-tdel0)+coor_end[1]*ksi[1]+coor_end[2]*ksi[0]

        
        elif side=='r':
            tdel = wfe.data.L_rr_func_tot(i_self,t)
            if wfe.data.calc_method=='Waluschka':
                tdel0=tdel
            elif wfe.data.calc_method=='Abram':
                tdel0=0
            tele_ang = aim.tele_l_ang(i_right,t-tdel)
            coor_start = beam_coor_out(wfe,i_right,t-tdel,tele_ang,PAAM_ang)
            coor_end = aim.tele_r_coor(i_self,t)
            start = aim.tele_l_start(i_right,t-tdel)+coor_start[1]*ksi[1]+coor_start[2]*ksi[0]
            end = aim.tele_r_start(i_self,t-tdel0)+coor_end[1]*ksi[1]+coor_end[2]*ksi[0]
            
        [zoff,yoff,xoff]=LA.matmul(coor_start,end-start)
        if precision==0:
            R = zoff # Not precise
        elif precision==1:
            try:
               [piston,z_extra] = wfe.z_solve(xoff,yoff,zoff,ret='all')
            except:
                [piston,z_extra] = [np.nan,np.nan]
            R = wfe.R(piston)

        R_vec = np.array([(R**2-xoff**2-yoff**2)**0.5,yoff,xoff])
        R_vec_tele = LA.matmul(coor_end,-LA.matmul(np.linalg.inv(coor_start),R_vec))
        angx = np.arctan(abs(R_vec_tele[2]/R_vec_tele[0]))*np.sign(R_vec_tele[2])
        angy = np.arctan(abs(R_vec_tele[1]/R_vec_tele[0]))*np.sign(R_vec_tele[1])

    if ret=='angy':
        return angy
    elif ret=='angx':
        return angx
    elif ret=='tilt':
        return (angx**2+angy**2)**0.5
    elif ret=='all':
        ret_val={}
        ret_val['start']=start
        ret_val['end']=end
        ret_val['zoff']=zoff
        ret_val['yoff']=yoff
        ret_val['xoff']=xoff
        ret_val['coor_start']=coor_start
        ret_val['coor_end']=coor_end
        ret_val['bd_original_frame'] = np.array(coor_start[0])
        ret_val['bd_receiving_frame'] = LA.matmul(coor_end,ret_val['bd_original_frame'])
        ret_val['angx_func_rec'] = angx
        ret_val['angy_func_rec'] = angy
        #ret_val['tilt']=(angx**2+angy**2)**0.5
        #ret_val['tilt']=LA.angle(R_vec_tele,(angx**2+angy**2)**0.5
        ret_val['piston']=piston
        ret_val['z_extra'] = z_extra
        ret_val['R']=R
        ret_val["R_vec"] = R_vec

        return ret_val






















LA = PAA_LISA.la()

# Changes of coordinate system
def coor_SC(wfe,i,t):
    # r,n,x (inplane) format
    t_calc=t

    r = LA.unit(wfe.data.r_func(i,t_calc))
    n = LA.unit(wfe.data.n_func(i,t_calc))
    x = np.cross(n,r)
    #offset = wfe.data.LISA.putp(i,t)

    return np.array([r,n,x])

def coor_tele(wfe,i,t,ang_tele):
    # Retunrs the coordinate system of telescope (same as SC but rotated over ang_tele inplane)
    L_tele = wfe.L_tele
    [r,n,x] = coor_SC(wfe,i,t)
    tele = r*L_tele
    tele = LA.rotate(tele,n,ang_tele)
    r = LA.unit(tele)
    x = np.cross(n,r)

    return np.array([r,n,x])

def pos_tele(wfe,i,t,side,ang_tele):
    offset = np.array(wfe.data.LISA.putp(i,t))
    pointing = coor_tele(wfe,i,t,ang_tele)

    return offset+pointing

def beam_coor_out(wfe,i,t,ang_tele,ang_paam):
    # Retunrs the coordinate system of the transmitted beam (same as SC but rotated over ang_tele inplane and ang_tele outplane)
    [r,n,x] = coor_tele(wfe,i,t,ang_tele) #Telescope coordinate system

    r = LA.unit(LA.rotate(r,x,ang_paam)) # Rotate r in out of plane over ang_paam
    n = np.cross(r,x)

    return np.array([r,n,x])

def i_slr(i):
    i_self = i
    i_left = (i+1)%3
    i_right = (i+2)%3

    i_ret = [i_self,i_left,i_right]
    for j in range(0,len(i_ret)):
        if i_ret[j]==0:
            i_ret[j]=3

    return i_ret

def delay(data,l_array,t,para='X',delay_on=True):
    t_del = 0
    if delay_on==True:
        for k in range(0,len(l_array)):
            j = -1-k
            i_r = (abs(l_array[j])+1)%3
            try:
                if l_array[j]>0:
                    t_del = t_del - data.L_rl[i_r-1](t - t_del)
                elif l_array[j]<0:
                    t_del = t_del - data.L_rr[i_r-1](t - t_del)
            except:
                pass

    return t_del

def PSD(f_list,SD_list):
    return interp1d(f_list,SD_list,bounds_error=False,fill_value=0)

def PowerLaw(SD_val,f0,exp=1):
    return lambda f: (SD_val)*((f/f0)**exp)

def add_func_help(func_list,f):

    func_ret = func_list[0]

    if len(func_list)>1:
        for i in range(1,len(func_list)):
            func_ret = func_ret(f)+func_list[i](f)

    return func_ret

def add_func(func_list):

    return lambda f: add_func_help(func_list,f)

def get_matrix_from_function(A,t):
    ret=[]
    for i in range(0,len(A)):
        vec=[]
        for j in range(0,len(A[i])):
            vec.append(A[i][j](t))
        ret.append(np.array(vec))

    return np.array(ret)

def interpolate(x,y,method='interp1d'):
    if method=='interp1d':
        if str(type(y[0]))!="<type 'numpy.ndarray'>":
            return interp1d(x,y,bounds_error=False)
        else:
            type_dim = str(type(y[0,0]))
            if type_dim!="<type 'numpy.ndarray'>":
                ret=[]
                for l in range(0,len(y[0])):
                    ret.append(interp1d(x,y[:,l],bounds_error=False))

                return lambda t: np.array([ret[0](t),ret[1](t),ret[2](t)])
            else:
                ret=[]
                for i in range(0,len(y[0])):
                    vec=[]
                    for j in range(0,len(y[0][i])):
                        vec.append(interp1d(x,y[:,i,j],bounds_error=False))
                    ret.append(np.array(vec))
                return lambda t: get_matrix_from_function(np.array(ret),t)
 

    else:
        print('Please select proper interpolation method (interp1d)')


