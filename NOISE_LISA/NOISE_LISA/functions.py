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
            

    writefile.close()

    print(title+' saved in:')
    print(direct)

    return direct

def rdln(line):
    return line[0:-1]

def read(filename='',ret={},direct=''):
    if direct=='':
        direct = get_folder()

    if filename=='':
        for (dirpath, dirnames, filenames) in os.walk(direct):
            filenames.sort()

    for filename_select in filenames:
        #print(filenames)
        print('Reading '+filename_select)

        readfile = open(direct+filename_select,'r')

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
            elif 'ax_title' in line:
                key2 = rdln(line.split(':: ')[-1])
                if key2 not in ret[key0][key1][iteration].keys():
                    ret[key0][key1][iteration][key2]={}
            elif 'Label' in line:
                key3 = rdln(line.split(':: ')[-1])
                if key3 not in ret[key0][key1][iteration][key2].keys():
                    ret[key0][key1][iteration][key2][key3]={}
                    ret[key0][key1][iteration][key2][key3]['x']=[]
                    ret[key0][key1][iteration][key2][key3]['y']=[]
            else:
                [x,y] = line.split(';')
                ret[key0][key1][iteration][key2][key3]['x'].append(np.float64(rdln(x)))
                ret[key0][key1][iteration][key2][key3]['y'].append(np.float64(rdln(y)))
        
        readfile.close()
    return ret


 

    






LA = PAA_LISA.la()

# Changes of coordinate system
def coor_SC(wfe,i,t):
    # r,n,x (inplane) format

    r = LA.unit(wfe.data.r_func(i,t))
    n = LA.unit(wfe.data.n_func(i,t))
    x = np.cross(n,r)

    return np.array([r,n,x])

def coor_tele(wfe,i,t,ang_tele,L_tele=2):
    # Retunrs the coordinate system of telescope (same as SC but rotated over ang_tele inplane)

    [r,n,x] = coor_SC(wfe,i,t)
    tele = r*L_tele
    tele = LA.rotate(tele,n,ang_tele)
    r = LA.unit(tele)
    x = np.cross(n,r)

    return np.array([r,n,x])

def beam_tele(wfe,i,t,ang_tele,ang_paam):
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

def interpolate(x,y,method='interp1d'):
    if method=='interp1d':
        return interp1d(x,y,bounds_error=False)
    else:
        print('Please select proper interpolation method (interp1d)')


