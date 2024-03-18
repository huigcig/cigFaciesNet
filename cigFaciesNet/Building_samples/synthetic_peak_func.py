"""
Code for generating peak data of several major types of seismic phase
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,interpolate
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def introduction(showfig=True):
    introtext = """
    ###### Description of functions in this code ######
    
    Parallel function:  generate_parallel() # Parallel or subparallel

    Prograding clinoform function: generate_clinoform0() # Sigmoid symmetric clinoform
                                   generate_clinoform1() # Sigmoid divergent clinoform
                                   generate_clinoform2() # Asymmetric top-heavy clinoform
                                   generate_clinoform3() # Asymmetric bottom-heavy clinoform
                                   generate_clinoform4() # Parallel oblique clinoform
                                   generate_clinoform5() # Oblique clinoform

    Fill function: generate_fill0() # Divergent fill
                   generate_fill1() # Onlap fill
                   generate_fill2() # Mounded onlap fill
                   generate_fill3() # Prograded fill
                   generate_fill4() # Chaotic fill
                   generate_fill5() # Complex fill

    Hummocky function: generate_hummocky0() # Pinnacle with velocity pull-up hummocky
                       generate_hummocky1() # Fan complex simple hummocky
                       generate_hummocky2() # Bank edge with velocity sag hummocky
                       generate_hummocky3() # Homogeneous with drape hummocky
                       generate_hummocky4() # Slump hummocky

    Chaotic function:  generate_chaotic() # Chaotic
                                   """
    print(introtext)
    
    if showfig is True:
        plt.figure(figsize=(24,16))
        plt.subplot(4,6,1)
        p1 = generate_parallel() 
        plt.imshow(p1.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Parallel and subparallel",fontsize=15)
        
        plt.subplot(4,6,4)
        ch1 = generate_chaotic() 
        plt.imshow(ch1.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Chaotic",fontsize=15)
        
        plt.subplot(4,6,7)
        c0 = generate_clinoform0() 
        plt.imshow(c0.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Sigmoid symmetric clinoform",fontsize=15)
        
        plt.subplot(4,6,8)
        c1 = generate_clinoform1() 
        plt.imshow(c1.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Sigmoid divergent clinoform",fontsize=15)
        
        plt.subplot(4,6,9)
        c2 = generate_clinoform2() 
        plt.imshow(c2.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Asymmetric top-heavy clinoform",fontsize=15)
        
        plt.subplot(4,6,10)
        c3 = generate_clinoform3() 
        plt.imshow(c3.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Asymmetric bottom-heavy clinoform",fontsize=15)
        
        plt.subplot(4,6,11)
        c4 = generate_clinoform4() 
        plt.imshow(c4.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Parallel oblique clinoform",fontsize=15)
        
        plt.subplot(4,6,12)
        c5 = generate_clinoform5() 
        plt.imshow(c5.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Oblique clinoform",fontsize=15)
        
        plt.subplot(4,6,13)
        f0 = generate_fill0() 
        plt.imshow(f0.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Divergent fill",fontsize=15)
        
        plt.subplot(4,6,14)
        f1 = generate_fill1() 
        plt.imshow(f1.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Onlap fill",fontsize=15)
        
        plt.subplot(4,6,15)
        f2 = generate_fill2() 
        plt.imshow(f2.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Mounded onlap fill",fontsize=15)
        
        plt.subplot(4,6,16)
        f3 = generate_fill3() 
        plt.imshow(f3.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Prograded fill",fontsize=15)
        
        plt.subplot(4,6,17)
        f4 = generate_fill4() 
        plt.imshow(f4.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Chaotic fill",fontsize=15)
        
        plt.subplot(4,6,18)
        f5 = generate_fill5() 
        plt.imshow(f5.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Complex fill",fontsize=15)
        
        plt.subplot(4,6,19)
        h0 = generate_hummocky0() 
        plt.imshow(h0.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Pinnacle with velocity pull-up hummocky",fontsize=15)
        
        plt.subplot(4,6,20)
        h1 = generate_hummocky1() 
        plt.imshow(h1.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Fan complex simple hummocky",fontsize=15)
        
        plt.subplot(4,6,21)
        h2 = generate_hummocky2() 
        plt.imshow(h2.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Bank edge with velocity sag hummocky",fontsize=15)
        
        plt.subplot(4,6,22)
        h3 = generate_hummocky3() 
        plt.imshow(h3.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Homogeneous with drape hummocky",fontsize=15)
        
        plt.subplot(4,6,23)
        h4 = generate_hummocky4() 
        plt.imshow(h4.T,cmap="gray",aspect="auto")
        plt.xticks([]),plt.yticks([])
        plt.title("Slump hummocky",fontsize=15)
        
        
        
        plt.tight_layout()
    
    
    
    
########################################################################################
####################################### PARALLEL #######################################
########################################################################################
# Generate randomly parallel or subparallel peak data
def generate_parallel():
    """
    Generate randomly parallel or subparallel peak data
    """
    section = np.zeros((128, 128))
    num_lines = random.randint(11,15)
    d_k = 1/(num_lines-1)
    k = random.uniform(-d_k,d_k)
    for i in range(num_lines):
        # define line's k and b and noise
        ki = random.uniform(k-(d_k/10),k+(d_k/10))
        bi = random.uniform(-1,1)
        noisei = random.uniform(-8,8)

        # build lines with noise
        xi = np.arange(0,section.shape[0],1)
        yi = ki*xi+bi+np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,5)

        # build lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            yi[mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    yi[mxii[mi]-wdi],yi[mxii[mi]+wdi] = -1000,-1000

        # build section with lines
        depi = (i+random.uniform(0.3,0.7))*section.shape[0]/(num_lines-1)    
        for ip in range(section.shape[0]):
            if int(yi[ip]+depi) < section.shape[0] and int(yi[ip]+depi) >= 0:
                section[xi[ip],int(yi[ip]+depi)]=1
    return section

####################################################################################################
####################################### PROGRADING CLINOFORM #######################################
####################################################################################################
# Generate randomly Sigmoid symmetric clinoform peak data
def generate_clinoform0():
    """
    Generate randomly Sigmoid symmetric clinoform peak data
    """
    section = np.zeros((128, 128))
    lines = []

    # Parameters settings
    sig1 = random.randint(15,30)
    if sig1 <=20:
        line_num = random.randint(7,8)
        h1 = random.randint(40,90)
    elif sig1 <=25:
        line_num = random.randint(8,9)
        h1 = random.randint(50,90)
    elif sig1 <=30:
        line_num = random.randint(9,10)
        h1 = random.randint(60,90)

    d_h1,d_h2 = random.uniform(3,4),random.uniform(3,4)
    d_value, d_para = random.uniform(6,8), random.uniform(6,8)
    interval1,interval2 = random.randint(16,24),random.randint(16,24)
    Change1,Change2 = False,False

    x = np.arange(0,128,1)
    for i in range(line_num):
        if i==0:
            noisei = random.uniform(-5,5)
            noise0 = np.random.uniform(-noisei, noisei, size=x.shape)
            y0 = h1/(1+np.exp((64 - x)/sig1)) + noise0
            y0 = gaussian_filter1d(y0,4)
            lines.append(np.vstack([x,y0]))
        else:
            noisei = random.uniform(-5,5)
            noise1 = np.random.uniform(-noisei, noisei, size=x.shape)
            if Change1 is False:
                y1 = h1/(1+np.exp((64 - x - interval1*i)/sig1)) + d_h1*i + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x,y1]))
            else:
                y1 = y1 + d_h1 + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x,y1]))

            noisei = random.uniform(-5,5)
            noise2 = np.random.uniform(-noisei, noisei, size=x.shape)
            if Change2 is False:
                y2 = h1/(1+np.exp((64 - x + interval2*i)/sig1)) - d_h2*i + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x,y2]))
            else:
                y2 = y2 - d_h2 + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x,y2]))

            # When the slope becomes small, increasing the thickness of the stratum
            if (np.abs(y1[0]-y1[-1])<d_value) and (Change1 is False) : 
                d_h1 = d_para
                Change1 = True
            if (np.abs(y2[0]-y2[-1])<d_value) and (Change2 is False) : 
                d_h2 = d_para
                Change2 = True

    dmin = np.array(lines)[:,1,:].min() + random.uniform(-5,5)
    dmax = np.array(lines)[:,1,:].max() + random.uniform(-5,5) - dmin
    for li in range(len(lines)):
        line = lines[li]
        line[1,:] = 128*(line[1,:]-dmin)/dmax

        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,4))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1

    return section

# Generate randomly Sigmoid divergent clinoform peak data
def generate_clinoform1():
    """
    Generate randomly Sigmoid divergent clinoform peak data
    """
    section = np.zeros((128, 128))
    lines = []

    # Parameters settings
    sig1 = random.randint(15,30)
    if sig1 <=22:
        line_num = random.randint(9,10)
        h1 = random.randint(40,90)
    elif sig1 <=30:
        line_num = random.randint(10,11)
        h1 = random.randint(50,90)

    d_h1,d_h2 = random.uniform(2,4),random.uniform(2,4)
    d_value,d_para = random.uniform(6,10),random.uniform(6,8)
    interval1,interval2 = random.randint(8,12),random.randint(8,12)
    shift1,shift2 = random.randint(10,15),random.randint(10,15)
    Change1,Change2 = False,False

    x0 = np.arange(-200,300,1)
    for i in range(line_num):
        if i==0:
            noisei = random.uniform(-4,4)
            noise0 = np.random.uniform(-noisei, noisei, size=x0.shape)
            y0 = h1/(1+np.exp((64 - x0)/sig1)) + noise0
            y0 = gaussian_filter1d(y0,4)
            lines.append(np.vstack([x0,y0]))
        else:
            noisei = random.uniform(-4,4)
            noise1 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change1 is False:
                y1 = h1/(1+np.exp((64 - i*interval1 - x0)/sig1)) + d_h1*i + noise1
                x1 = x0 - i*shift1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x1,y1]))
            else:
                y1 = y1 + d_h1 + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x1,y1]))

            noisei = random.uniform(-4,4)
            noise2 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change2 is False:
                y2 = h1/(1+np.exp((64 + i*interval2 - x0)/sig1)) - d_h2*i + noise2
                x2 = x0 + i*shift2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x2,y2]))
            else:
                y2 = y2 - d_h2 + noise2
                y2 = gaussian_filter1d(y2,4) 
                lines.append(np.vstack([x2,y2]))

            # When the slope becomes small, increasing the thickness of the stratum
            if (np.abs(y1[np.where(x1==0)[0][0]]-y1[np.where(x1==127)[0][0]])<d_value) and (Change1 is False) : 
                d_h1 = d_para
                Change1 = True
            if (np.abs(y2[np.where(x2==0)[0][0]]-y2[np.where(x2==127)[0][0]])<d_value) and (Change2 is False) :
                d_h2 = d_para
                Change2 = True

    dmin = np.array(lines)[:,1,:].min() + random.uniform(-5,5)
    dmax = np.array(lines)[:,1,:].max() + random.uniform(-5,5) - dmin

    for li in range(len(lines)):
        line = lines[li]
        line[1,:] = 128*(line[1,:]-dmin)/dmax

        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,4))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1       
    return section
    
    
# Generate randomly Asymmetric top-heavy clinoform peak data
def generate_clinoform2():
    """
    Generate randomly Asymmetric top-heavy clinoform peak data
    """
    section = np.zeros((128, 128))
    lines = []

    x0 = [0,127]
    y0 = [random.randint(10,30),random.randint(100,120)]
    xi0 = np.arange(0,128,1)
    scale0 = random.randint(15,30)
    center = random.uniform(1, 1.6)
    mid = (x0[1]-x0[0])/center
    hh = np.exp((x0[-1]-mid)/scale0)/(1+np.exp((x0[-1]-mid)/scale0))-np.exp((x0[0]-mid)/scale0)/(1+np.exp((x0[0]-mid)/scale0))
    hi = (y0[1]-y0[0])/hh
    h_shift = y0[0] - hi*np.exp((x0[0]-mid)/scale0)/(1+np.exp((x0[0]-mid)/scale0))
    noisei = random.uniform(-4,4)
    yi0 = h_shift+hi*np.exp((xi0-mid)/scale0)/(1+np.exp((xi0-mid)/scale0))+np.random.uniform(-noisei, noisei, size=xi0.shape)
    yi0 =  gaussian_filter1d(yi0,4)
    lines.append(np.vstack([xi0,yi0]))

    yi1 = yi0.copy()
    yi2 = yi0.copy()

    delta = (y0[1]-y0[0])/10
    for i in range(10):
        x1 = [0,127]
        y1 = [yi1[x1[0]]+5+delta, yi1[x1[1]]+5]
        if y1[0]>y1[1]:
            y1[1] = y1[0]
        xi1 = np.arange(0,128,1)
        scale = scale0 + 2*(i+1)
        mid = (x1[1]-x1[0])/center
        hh = np.exp((x1[-1]-mid)/scale)/(1+np.exp((x1[-1]-mid)/scale))-np.exp((x1[0]-mid)/scale)/(1+np.exp((x1[0]-mid)/scale))
        hi = (y1[1]-y1[0])/hh
        h_shift = y1[0] - hi*np.exp((x1[0]-mid)/scale)/(1+np.exp((x1[0]-mid)/scale))
        noisei = random.uniform(-4,4)
        yi1 = h_shift+hi*np.exp((xi1-mid)/scale)/(1+np.exp((xi1-mid)/scale))+np.random.uniform(-noisei, noisei, size=xi1.shape)
        yi1 =  gaussian_filter1d(yi1,4)
        lines.append(np.vstack([xi1,yi1]))

        x2 = [0,127]
        y2 = [yi2[x2[0]]-6, yi2[x2[1]]-10]
        if y2[1]<y2[0]:
            y2[0] = y2[1]
        xi2 = np.arange(0,128,1)
        scale = scale0 
        mid = (x2[1]-x2[0])/center
        hh = np.exp((x2[-1]-mid)/scale)/(1+np.exp((x2[-1]-mid)/scale))-np.exp((x2[0]-mid)/scale)/(1+np.exp((x2[0]-mid)/scale))
        hi = (y2[1]-y2[0])/hh
        h_shift = y2[0] - hi*np.exp((x2[0]-mid)/scale)/(1+np.exp((x2[0]-mid)/scale))
        noisei = random.uniform(-4,4)
        yi2 = h_shift+hi*np.exp((xi2-mid)/scale)/(1+np.exp((xi2-mid)/scale))+np.random.uniform(-noisei, noisei, size=xi2.shape)
        yi2 =  gaussian_filter1d(yi2,4)
        lines.append(np.vstack([xi2,yi2]))

    for li in range(len(lines)):
        line = lines[li]
        # build lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,5))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,4)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1
    return section
                
# Generate randomly Asymmetric bottom-heavy clinoform peak data
def generate_clinoform3():
    """
    Generate randomly Asymmetric bottom-heavy clinoform peak data
    """
    section = np.zeros((128, 128))
    lines = []

    x0 = [0,127]
    y0 = [random.randint(40,64),random.randint(110,128)]
    xi0 = np.arange(0,128,1)
    scale0 = random.randint(15,30)
    center = random.uniform(2, 3)
    mid = (x0[1]-x0[0])/center
    hh = np.exp((x0[-1]-mid)/scale0)/(1+np.exp((x0[-1]-mid)/scale0))-np.exp((x0[0]-mid)/scale0)/(1+np.exp((x0[0]-mid)/scale0))
    hi = (y0[1]-y0[0])/hh
    h_shift = y0[0] - hi*np.exp((x0[0]-mid)/scale0)/(1+np.exp((x0[0]-mid)/scale0))
    noisei = random.uniform(-4,4)
    yi0 = h_shift+hi*np.exp((xi0-mid)/scale0)/(1+np.exp((xi0-mid)/scale0))+np.random.uniform(-noisei, noisei, size=xi0.shape)
    yi0 =  gaussian_filter1d(yi0,4)
    lines.append(np.vstack([xi0,yi0]))

    yi1 = yi0.copy()
    yi2 = yi0.copy()

    delta = (y0[1]-y0[0])/10
    for i in range(12):
        x1 = [0,127]
        y1 = [yi1[x1[0]]+10, yi1[x1[1]]+6]
        if y1[0]>y1[1]:
            y1[1] = y1[0]
        xi1 = np.arange(0,128,1)
        scale = scale0 + 2*(i+1)
        mid = (x1[1]-x1[0])/center
        hh = np.exp((x1[-1]-mid)/scale)/(1+np.exp((x1[-1]-mid)/scale))-np.exp((x1[0]-mid)/scale)/(1+np.exp((x1[0]-mid)/scale))
        hi = (y1[1]-y1[0])/hh
        h_shift = y1[0] - hi*np.exp((x1[0]-mid)/scale)/(1+np.exp((x1[0]-mid)/scale))
        noisei = random.uniform(-4,4)
        yi1 = h_shift+hi*np.exp((xi1-mid)/scale)/(1+np.exp((xi1-mid)/scale))+np.random.uniform(-noisei, noisei, size=xi1.shape)
        yi1 =  gaussian_filter1d(yi1,4)
        lines.append(np.vstack([xi1,yi1]))

        x2 = [0,127]
        y2 = [yi2[x2[0]]-5, yi2[x2[1]]-5-delta]
        if y2[1]<y2[0]:
            y2 = [yi2[x2[0]]-6, yi2[x2[0]]-6]
        xi2 = np.arange(0,128,1)
        scale = scale0 
        mid = (x2[1]-x2[0])/center
        hh = np.exp((x2[-1]-mid)/scale)/(1+np.exp((x2[-1]-mid)/scale))-np.exp((x2[0]-mid)/scale)/(1+np.exp((x2[0]-mid)/scale))
        hi = (y2[1]-y2[0])/hh
        h_shift = y2[0] - hi*np.exp((x2[0]-mid)/scale)/(1+np.exp((x2[0]-mid)/scale))
        noisei = random.uniform(-4,4)
        yi2 = h_shift+hi*np.exp((xi2-mid)/scale)/(1+np.exp((xi2-mid)/scale))+np.random.uniform(-noisei, noisei, size=xi2.shape)
        yi2 =  gaussian_filter1d(yi2,4)
        lines.append(np.vstack([xi2,yi2]))

    for li in range(len(lines)):
        line = lines[li]
        line[1,:] = line[1,:]-10
        # build lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1

    return section

# Generate randomly Parallel oblique clinoform peak data
def generate_clinoform4():
    """
    Generate randomly Parallel oblique clinoform peak data
    """    
    section = np.zeros((128, 128))
    lines = []

    k0 = random.uniform(0.2,0.6)
    b0 = random.uniform(20,60)
    k_s1 = random.uniform(0.03,0.05)
    k_s2 = random.uniform(0.03,0.05)
    b_s1 = random.uniform(6,9)
    b_s2 = random.uniform(9,12)
    d_value,d_para = random.uniform(6,10),random.uniform(6,8) 
    Change1,Change2 = False,False

    x0 = np.arange(0,128,1)
    for i in range(15):
        if i==0:
            noisei = random.uniform(-6,6)
            noise0 = np.random.uniform(-noisei, noisei, size=x0.shape)
            y0 = k0*x0 + b0  + noise0
            y0 = gaussian_filter1d(y0,4)
            lines.append(np.vstack([x0,y0]))

        else:
            noisei = random.uniform(-6,6)
            noise1 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change1 is False:
                y1 = (k0-k_s1*i)*x0 + b0 - b_s1*i + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x0,y1]))
            else:
                y1 = y1 - d_para + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x0,y1]))

            noisei = random.uniform(-6,6)
            noise2 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change2 is False:
                y2 = (k0-k_s2*i)*x0 + b0 + b_s2*i + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x0,y2]))
            else:
                y2 = y2 + d_para + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x0,y2]))

            # When the slope becomes small, increasing the thickness of the stratum
            if (np.abs(y1[0]-y1[-1])<d_value) and (Change1 is False) : 
                d_h1 = d_para
                Change1 = True
            if (np.abs(y2[0]-y2[-1])<d_value) and (Change2 is False) :
                d_h2 = d_para
                Change2 = True

    for li in range(len(lines)):
        line = lines[li]
        line[1,:] = line[1,:]-10
        # build lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(2,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,4)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1
    return section

# Generate randomly Oblique clinoform peak data
def generate_clinoform5():
    """
    Generate randomly Oblique clinoform peak data
    """    
    section = np.zeros((128, 128))
    lines = []

    x0 = np.arange(0,128,1)
    # add para layer
    kp = random.uniform(0.1,0.2)
    hp = random.uniform(10,35)
    for i in range(10):
        noisei = random.uniform(-5,5)
        noisep = np.random.uniform(-noisei, noisei, size=x0.shape)
        if i==0:
            yp = kp*x0  + hp + noisep
            yp = gaussian_filter1d(yp,4)
            lines.append(np.vstack([x0,yp]))
        else:
            hp_shift = random.uniform(6,10)
            yp = yp - hp_shift + noisep
            yp = gaussian_filter1d(yp,4)
            lines.append(np.vstack([x0,yp]))

    k0 = random.uniform(0.3,0.8)
    b0 = random.uniform(20,60)
    k_s1 = random.uniform(0.03,0.05)
    k_s2 = random.uniform(0.03,0.05)
    b_s1 = random.uniform(8,10)
    b_s2 = random.uniform(10,12)
    d_value,d_para = random.uniform(6,10),random.uniform(6,8) 
    Change1,Change2 = False,False

    for i in range(15):
        if i==0:
            noisei = random.uniform(-6,6)
            noise0 = np.random.uniform(-noisei, noisei, size=x0.shape)
            y0 = k0*x0 + b0  + noise0
            y0 = gaussian_filter1d(y0,4)
            lines.append(np.vstack([x0,y0]))

        else:
            noisei = random.uniform(-6,6)
            noise1 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change1 is False:
                y1 = (k0-k_s1*i)*x0 + b0 - b_s1*i + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x0,y1]))
            else:
                y1 = y1 - d_para + noise1
                y1 = gaussian_filter1d(y1,4)
                lines.append(np.vstack([x0,y1]))

            noisei = random.uniform(-6,6)
            noise2 = np.random.uniform(-noisei, noisei, size=x0.shape)
            if Change2 is False:
                y2 = (k0-k_s2*i)*x0 + b0 + b_s2*i + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x0,y2]))
            else:
                y2 = y2 + d_para + noise2
                y2 = gaussian_filter1d(y2,4)
                lines.append(np.vstack([x0,y2]))

            # When the slope becomes small, increasing the thickness of the stratum
            if (np.abs(y1[0]-y1[-1])<d_value) and (Change1 is False) : 
                d_h1 = d_para
                Change1 = True
            if (np.abs(y2[0]-y2[-1])<d_value) and (Change2 is False) :
                d_h2 = d_para
                Change2 = True

    for li in range(len(lines)):
        line = lines[li]
    #     line[1,:] = line[1,:]-10
        if li==0:
            lp_boundary = line.copy()

        if li>=10:
            for bi in range(len(lp_boundary[1,:])):
                if bi%10 ==0:
                    shift = random.uniform(3,6)
                if line[1,bi] <=  lp_boundary[1,bi]+shift:
                    line[1,bi] = -1000

        # build lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,5))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,4)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1
    return section


####################################################################################
####################################### FILL #######################################
####################################################################################
# Generate randomly divergent fill peak data
def generate_fill0():
    """
    Generate randomly divergent fill peak data
    """
    section = np.zeros((128, 128))
    lines = []
    line_num = 20
    locat = 5
    depmax= random.uniform(20,60)
    dep_int = depmax/int(line_num/2)
    yi1 = [4,8]
    yi2 = [6,10]

    for i in range(line_num):    
        if i <= int(line_num/2):
            if i==0:
                depi = dep_int
                y1 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
                y2 = [locat+depi+random.uniform(-5,5),locat+depi+random.uniform(-5,5)]
                y3 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
            else:
                depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
                depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
                y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),
                      y1[1] + random.uniform(yi1[0],yi1[1])]
                y2 = [y2[0] + depi1 + random.uniform(yi1[0],yi1[1]),
                      y2[1] + depi2 + random.uniform(yi1[0],yi1[1])]
                y3 = [y3[0] + random.uniform(yi1[0],yi1[1]),
                      y3[1] + random.uniform(yi1[0],yi1[1])]
        else:
            depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
            depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
            y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),
                  y1[1] + random.uniform(yi2[0],yi2[1])]
            y2 = [y2[0] - depi1 + random.uniform(yi2[0],yi2[1]),
                  y2[1] - depi2 + random.uniform(yi2[0],yi2[1])]
            y3 = [y3[0] + random.uniform(yi2[0],yi2[1]),
                  y3[1] + random.uniform(yi2[0],yi2[1])]
        if i == 0:
            x = [0,random.uniform(5,20),random.uniform(40,60),random.uniform(70,90),
                 random.uniform(110,123),128]
        else:
            x = [x[0],x[1]+random.uniform(-2,2),x[2]+random.uniform(-5,5),x[3]+random.uniform(-5,5),
                 x[4]+random.uniform(-2,2),128]
        y = [y1[0],y1[1],y2[0],y2[1],y3[0],y3[1]]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi)
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,3)
        lines.append(np.vstack([xi,yi]))

    for li in range(len(lines)):

        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1  
    return section
       
# Generate randomly onlap fill peak data
def generate_fill1():
    """
    Generate randomly onlap fill peak data
    """    
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(5,20),random.uniform(40,60),
          random.uniform(70,90),random.uniform(110,123),128]
    y0 = [0,random.uniform(5,20),random.uniform(50,100),random.uniform(50,100),
          random.uniform(5,20),0]
    func = interpolate.interp1d(x0,y0,kind="cubic")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi = yi + (128-yi.max()+yi.min())/1.6
    lines.append(np.vstack([xi,yi]))


    # add upper parallel lines
    for i in range(5):
        x = [0,32,64,96,128]
        loc = yi.min()-8*i-5
        shift = random.uniform(2,6)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5)+ np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))

    yi2 = [10,15]
    dep_int=random.uniform(5,10)
    # add bottom fill layer
    for i in range(10):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.5,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.5,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/1.6
        else:
            y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),y1[1]+ random.uniform(yi2[0],yi2[1]),
                  y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
                  y1[4] + random.uniform(yi2[0],yi2[1]),y1[5]+ random.uniform(yi2[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="cubic")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5)+ np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))

    # fill the hole
    fline_num = int((yi.max()-yi.min())/8)
    for i in range(fline_num):
        if i==0:
            depi = yi.min() + random.uniform(6,10)
        else:
            depi = depi + random.uniform(6,10)

        find_x1 = int(np.argsort(np.abs(yi-depi))[0])
        find_x2 = int(np.argsort(np.abs(yi-depi))[9])
        for kk in range(1,10):
            if np.abs(np.argsort(np.abs(yi-depi))[kk] - find_x1) >= 10:
                find_x2 = int(np.argsort(np.abs(yi-depi))[kk])  
        if find_x1 > find_x2:
            xx = find_x1
            find_x1 = find_x2
            find_x2 = xx
        if find_x1-10>=0: find_x1 = find_x1-10
        else: find_x1 = 0
        if find_x2+10<128: find_x2 = find_x2+10
        else: find_x2 = 128

        interval = float(find_x2 - find_x1)/3
        x2 = [find_x1,find_x1+interval,find_x2-interval,find_x2]
        y2 = [depi + random.uniform(-2,2),depi + random.uniform(-2,2),
              depi + random.uniform(-2,2),depi + random.uniform(-2,2)]
        func = interpolate.interp1d(x2,y2,kind="cubic") 
        xii = np.arange(find_x1,find_x2,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5)+ np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)

        for mi in range(len(xii)):
            if yii[mi] >= yi[xii[mi]]:
                yii[mi] = -1000
        lines.append(np.vstack([xii,yii]))

    for li in range(len(lines)):
        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1 
    return section

# Generate randomly mounded onlap fill peak data
def generate_fill2():
    """
    Generate randomly mounded onlap fill peak data
    """  
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(5,20),random.uniform(40,60),
          random.uniform(70,90),random.uniform(110,123),128]
    y0 = [0,random.uniform(5,20),random.uniform(50,100),random.uniform(50,100),
          random.uniform(5,20),0]
    func = interpolate.interp1d(x0,y0,kind="cubic")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi = yi + (128-yi.max()+yi.min())/1.6
    lines.append(np.vstack([xi,yi]))


    # add upper parallel lines
    for i in range(5):
        x = [0,32,64,96,128]
        loc = yi.min()-8*i-5
        shift = random.uniform(2,4)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)

        if i==0:
            yyi = yii.copy()
        lines.append(np.vstack([xii,yii]))


    yi2 = [10,15]
    dep_int=random.uniform(5,10)
    # add bottom fill layer
    for i in range(10):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.5,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.5,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/1.5
        y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),y1[1]+ random.uniform(yi2[0],yi2[1]),
              y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
              y1[4] + random.uniform(yi2[0],yi2[1]),y1[5]+ random.uniform(yi2[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="cubic")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))


    # fill the hole
    fline_num = int((yi.max()-yi.min())/8)
    for i in range(fline_num):
        if i==0:
            depi = yi.min() + random.uniform(6,10)
        else:
            depi = depi + random.uniform(6,10)

        find_x1 = int(np.argsort(np.abs(yi-depi))[0])
        find_x2 = int(np.argsort(np.abs(yi-depi))[9])
        for kk in range(1,10):
            if np.abs(np.argsort(np.abs(yi-depi))[kk] - find_x1) >= 5:
                find_x2 = int(np.argsort(np.abs(yi-depi))[kk])  
                
        if find_x1 > find_x2:
            xx = find_x1
            find_x1 = find_x2
            find_x2 = xx
        if find_x1-10>=0: find_x1 = find_x1-10
        else: find_x1 = 0
        if find_x2+10<128: find_x2 = find_x2+10
        else: find_x2 = 128

        interval = float(find_x2 - find_x1)/3
        x2 = [find_x1,find_x1+interval,find_x2-interval,find_x2]
        y2 = [depi + random.uniform(-2,2),depi + random.uniform(-2,2)-random.uniform(8,12),
              depi + random.uniform(-2,2)-random.uniform(8,12),depi + random.uniform(-2,2)]
        func = interpolate.interp1d(x2,y2,kind="cubic") 
        xii = np.arange(find_x1,find_x2,1)
        yii = func(xii)

        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)


        for mi in range(len(xii)):
            if yii[mi] >= yi[xii[mi]] or yii[mi] <= yyi[xii[mi]]:
                yii[mi] = -1000
        lines.append(np.vstack([xii,yii]))

    for li in range(len(lines)):
        line = lines[li]

        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1 
    return section

# Generate randomly prograded fill peak data
def generate_fill3():
    """
    Generate randomly prograded fill peak data
    """ 
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(5,20),random.uniform(40,60),
          random.uniform(70,90),random.uniform(110,123),128]
    y0 = [0,random.uniform(5,20),random.uniform(50,90),random.uniform(50,90),
          random.uniform(5,20),0]
    func = interpolate.interp1d(x0,y0,kind="cubic")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi = yi + (128-yi.max()+yi.min())/1.6
    lines.append(np.vstack([xi,yi]))


    # add upper parallel lines
    for i in range(5):
        x = [0,32,64,96,128]
        loc = yi.min()-10*i-5
        shift = random.uniform(2,4)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        if i==0:
            yyi = yii.copy()
        lines.append(np.vstack([xii,yii]))

    yi2 = [10,15]
    dep_int=random.uniform(5,10)
    # add bottom fill layer
    for i in range(10):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.6,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.6,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/1.5
        y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),y1[1]+ random.uniform(yi2[0],yi2[1]),
              y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
              y1[4] + random.uniform(yi2[0],yi2[1]),y1[5]+ random.uniform(yi2[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="cubic")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))

    # add fill prograded lines
    pline_num = random.randint(6,10)
    x_top = np.linspace(0,128,pline_num+2)[1:-1]
    b_point = np.argmax(yi)
    x_bot = np.linspace(b_point,128,pline_num+2)[1:-1]

    for i in range(pline_num):
        x = [0,x_bot[i]-x_top[i]]
        y = [yyi[int(x_top[i])],yi[int(x_bot[i])]]
        xii = np.arange(x[0],x[1],1)
        scale = random.uniform(10,20)
        mid = (x[1]-x[0])/2
        hh = np.exp((xii[-1]-mid)/scale)/(1+np.exp((xii[-1]-mid)/scale))-np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        hi = (y[1]-y[0])/hh
        h_shift = y[0] - hi*np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        yii = h_shift+hi*np.exp((xii-mid)/scale)/(1+np.exp((xii-mid)/scale))
        xii = xii + x_top[i]
        for mi in range(len(xii)):
            if yii[mi] >= yi[int(xii[mi])] or yii[mi] <= yyi[int(xii[mi])]:
                yii[mi] = -1000
        lines.append(np.vstack([xii,yii]))

    for li in range(len(lines)):
        line = lines[li]

        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1   
    return section
    
# Generate randomly chaotic fill peak data
def generate_fill4():
    """
    Generate randomly chaotic fill peak data
    """    
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(5,20),random.uniform(40,60),
          random.uniform(70,90),random.uniform(110,123),128]
    y0 = [0,random.uniform(5,20),random.uniform(50,90),random.uniform(50,90),
          random.uniform(5,20),0]
    func = interpolate.interp1d(x0,y0,kind="cubic")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi = yi + (128-yi.max()+yi.min())/1.6
    lines.append(np.vstack([xi,yi]))

    # add upper parallel lines
    for i in range(5):
        x = [0,32,64,96,128]
        loc = yi.min()-10*i-5
        shift = random.uniform(2,4)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        if i==0:
            yyi = yii.copy()
        lines.append(np.vstack([xii,yii]))

    yi2 = [10,15]
    dep_int=random.uniform(5,10)
    # add bottom fill layer
    for i in range(10):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.6,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.6,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/1.5
        y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),y1[1]+ random.uniform(yi2[0],yi2[1]),
              y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
              y1[4] + random.uniform(yi2[0],yi2[1]),y1[5]+ random.uniform(yi2[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="cubic")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))

    for li in range(len(lines)):
        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1    

    # fill the chaotic 
    def find_peak(data):
        data = data.T
        col_peaks = np.zeros_like(data).astype(int)
        for i in range(data.shape[1]):
            col_data = data[:,i]
            for j in range(1,data.shape[0]-1):
                col,col1,col2 = col_data[j],col_data[j-1],col_data[j+1]
                if col > col1 and col > col2:
                    col_peaks[j, i] = 1
        return col_peaks.T

    mean = random.randint(-10,10)
    std = random.randint(10,20)
    x = np.random.normal(mean,std,size=(128,128))
    x = ndimage.gaussian_filter(x,2)
    chaotic = find_peak(x)

    for i in range(section.shape[0]):
        yi1,yi2 = int(yyi[i]),int(yi[i])
        section[i,yi1:yi2] = chaotic[i,yi1:yi2]
    return section
    
# Generate randomly complex fill peak data
def generate_fill5():
    """
    Generate randomly complex fill peak data
    """    
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(5,20),random.uniform(40,60),
          random.uniform(70,90),random.uniform(110,123),128]
    y0 = [0,random.uniform(5,20),random.uniform(50,90),random.uniform(50,90),
          random.uniform(5,20),0]
    func = interpolate.interp1d(x0,y0,kind="cubic")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi = yi + (128-yi.max()+yi.min())/1.6
    lines.append(np.vstack([xi,yi]))

    # add upper parallel lines
    for i in range(5):
        x = [0,32,64,96,128]
        loc = yi.min()-10*i-5
        shift = random.uniform(2,4)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,5)
        if i==0:
            yyi = yii.copy()
        lines.append(np.vstack([xii,yii]))

    yi2 = [10,15]
    dep_int=random.uniform(5,10)
    # add bottom fill layer
    for i in range(10):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.6,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.6,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/1.5
        y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),y1[1]+ random.uniform(yi2[0],yi2[1]),
              y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
              y1[4] + random.uniform(yi2[0],yi2[1]),y1[5]+ random.uniform(yi2[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="cubic")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,5)
        lines.append(np.vstack([xii,yii]))

    # add fill prograded lines
    cline_num = random.randint(8,15)
    x_top = np.linspace(random.randint(5,15),128-random.randint(5,10),cline_num)
    b_point = np.argmax(yi)
    x_bot = np.linspace(b_point,128-random.randint(5,10),cline_num)

    pline_num = int(cline_num/2)+random.randint(-2,0)
    for i in range(pline_num):
        x = [0,x_bot[i]-x_top[i]]
        y = [yyi[int(x_top[i])],yi[int(x_bot[i])]]
        xii = np.arange(x[0],x[1],1)
        scale = random.uniform(5,10)
        mid = (x[1]-x[0])/2
        hh = np.exp((xii[-1]-mid)/scale)/(1+np.exp((xii[-1]-mid)/scale))-np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        hi = (y[1]-y[0])/hh
        h_shift = y[0] - hi*np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        yii = h_shift+hi*np.exp((xii-mid)/scale)/(1+np.exp((xii-mid)/scale))
        xii = xii + x_top[i]
        for mi in range(len(xii)):
            if yii[mi] >= yi[int(xii[mi])] or yii[mi] <= yyi[int(xii[mi])]:
                yii[mi] = np.nan
        lines.append(np.vstack([xii,yii]))
        if i== pline_num-1:
            yyyi = yii.copy()
            xxxi = xii.copy()

    # 2pline
    p2line_num = cline_num - pline_num
    yyyi = np.where(yyyi!=yyyi,np.nanmax(yyyi),yyyi)
    p1xmin, p1xmax = xxxi[0],xxxi[-1]
    x_top = np.linspace(p1xmax - random.randint(3,7),p1xmin+random.randint(3,7),p2line_num)
    x_bot = np.linspace(p1xmax + random.randint(5,10),128 + random.randint(20,40) ,p2line_num+2)  

    for i in range(p2line_num):
        x = [0,x_bot[i]-x_top[i]]
        if x_bot[i]<=127:
            y = [yyyi[int(x_top[i]-xxxi[0])],yi[int(x_bot[i])]]
        else:
            y = [yyyi[int(x_top[i]-xxxi[0])],y[1]-random.randint(10,20)]
        xii = np.arange(x[0],x[1],1)
        scale = random.uniform(5,10)
        mid = (x[1]-x[0])/2
        hh = np.exp((xii[-1]-mid)/scale)/(1+np.exp((xii[-1]-mid)/scale))-np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        hi = (y[1]-y[0])/hh
        h_shift = y[0] - hi*np.exp((-mid)/scale)/(1+np.exp((-mid)/scale))
        yii = h_shift+hi*np.exp((xii-mid)/scale)/(1+np.exp((xii-mid)/scale))
        xii = xii + x_top[i]
        for mi in range(len(xii)):
            if int(xii[mi])<128:
                if int(xii[mi]) <= p1xmax:
                    if yii[mi] >= yyyi[int(xii[mi]-xxxi[0])] or yii[mi] <= yyi[int(xii[mi])]:
                        yii[mi] = np.nan
                else: 
                    if yii[mi] >= yi[int(xii[mi]-xxxi[0])] or yii[mi] <= yyi[int(xii[mi])]:
                        yii[mi] = np.nan
        lines.append(np.vstack([xii,yii])) 


    for li in range(len(lines)):
        line = lines[li]

        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = np.nan
                    line[1][int(mxii[mi]-line[0][0])] = np.nan
                    line[1][int(mxii[mi]+wdi-line[0][0])] = np.nan

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1    
    return section    

########################################################################################
####################################### HUMMOCKY #######################################
########################################################################################

# Generate randomly Pinnacle with velocity pull-up peak data
def generate_hummocky0():
    """
    Generate randomly Pinnacle with velocity pull-up peak data
    """  
    section = np.zeros((128, 128))
    lines = []
    line_num = 20
    locat = 5
    depmax= random.uniform(20,50)
    dep_int = depmax/int(line_num/2)
    yi1 = [4,8]
    yi2 = [6,10]

    for i in range(line_num):    
        if i <= int(line_num/2):
            if i==0:
                depi = dep_int
                y1 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
                y2 = [locat+depi+random.uniform(-5,5),locat+depi+random.uniform(-5,5)]
                y3 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
            else:
                depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
                depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
                y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),
                      y1[1] + random.uniform(yi1[0],yi1[1])]
                y2 = [y2[0] + depi1 + random.uniform(yi1[0],yi1[1]),
                      y2[1] + depi2 + random.uniform(yi1[0],yi1[1])]
                y3 = [y3[0] + random.uniform(yi1[0],yi1[1]),
                      y3[1] + random.uniform(yi1[0],yi1[1])]
        else:
            depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
            depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
            y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),
                  y1[1] + random.uniform(yi2[0],yi2[1])]
            y2 = [y2[0] - depi1 + random.uniform(yi2[0],yi2[1]),
                  y2[1] - depi2 + random.uniform(yi2[0],yi2[1])]
            y3 = [y3[0] + random.uniform(yi2[0],yi2[1]),
                  y3[1] + random.uniform(yi2[0],yi2[1])]
        if i == 0:
            x = [0,random.uniform(5,20),random.uniform(40,60),random.uniform(70,90),
                 random.uniform(110,123),128]
        else:
            x = [x[0],x[1]+random.uniform(-2,2),x[2]+random.uniform(-5,5),x[3]+random.uniform(-5,5),
                 x[4]+random.uniform(-2,2),128]
        y = [y1[0],y1[1],y2[0],y2[1],y3[0],y3[1]]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi)
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = 127-gaussian_filter1d(yi,3)

        lines.append(np.vstack([xi,yi]))

    for li in range(len(lines)):

        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1 
    return section

# Generate randomly Fan complex simple peak data
def generate_hummocky1():
    """
    Generate randomly Fan complex simple peak data
    """  
    section = np.zeros((128, 128))
    lines = []
    line_num = random.randint(10,15)
    locat = random.randint(5,15)
    depmax= random.uniform(10,30)
    dep_int = depmax/int(line_num/2)
    yi1 = [4,8]
    yi2 = [6,10]

    for i in range(line_num):    
        if i==0:
            depi = dep_int
            y1 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
            y2 = [locat+depi+random.uniform(-5,5),locat+depi+random.uniform(-5,5)]
            y3 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
        else:
            depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
            depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
            y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),
                  y1[1] + random.uniform(yi1[0],yi1[1])]
            y2 = [y2[0] + depi1 + random.uniform(yi1[0],yi1[1]),
                  y2[1] + depi2 + random.uniform(yi1[0],yi1[1])]
            y3 = [y3[0] + random.uniform(yi1[0],yi1[1]),
                  y3[1] + random.uniform(yi1[0],yi1[1])]
        if i == 0:
            x = [0,random.uniform(5,20),random.uniform(40,60),random.uniform(70,90),
                 random.uniform(110,123),128]
        else:
            x = [x[0],x[1]+random.uniform(-2,2),x[2]+random.uniform(-5,5),x[3]+random.uniform(-5,5),
                 x[4]+random.uniform(-2,2),128]
        y = [y1[0],y1[1],y2[0],y2[1],y3[0],y3[1]]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi) - 20
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,3)
        if i == line_num-1:
            yi1 = (127-yi).copy()
        lines.append(np.vstack([xi,127-yi]))

    for i in range(20):
        if i==0:
            y = [yi1[0]-random.uniform(6,10), ((yi1[0]+yi1[-1])/2)-random.uniform(6,10) ,yi1[-1]-random.uniform(6,10)]
        else:
            y = [yi[0]-random.uniform(6,10), ((yi[0]+yi[-1])/2)-random.uniform(6,10) ,yi[-1]-random.uniform(6,10)]
        x = [0,random.uniform(50,80),128]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi)
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,3)
        for ip in range(xi.shape[0]):
            if yi[ip] >= yi1[int(xi[ip])]:
                yi[ip] = np.nan
        lines.append(np.vstack([xi,yi]))

    for li in range(len(lines)):

        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1 
    return section
    
# Generate randomly Bank edge with velocity sag peak data
def generate_hummocky2():
    """
    Generate randomly Bank edge with velocity sag peak data
    """  
    section = np.zeros((128, 128))
    lines = []
    line_num = random.randint(8,12)
    locat = 5
    depmax= random.uniform(5,20)
    dep_int = depmax/int(line_num/2)
    yi1 = [6,10]
    yi2 = [6,10]
    yi3 = [2,4]

    for i in range(line_num):    
        if i==0:
            depi = dep_int
            y1 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
            y2 = [locat+depi+random.uniform(-5,5),locat+depi+random.uniform(-5,5)]
            y3 = [locat-random.uniform(8,12)+random.uniform(-5,5),locat-random.uniform(15,25)+random.uniform(-5,5)]
        else:
            depi1 = random.uniform(dep_int*0.7,dep_int*1.1)
            depi2 = random.uniform(dep_int*0.7,dep_int*1.1)
            y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),
                  y1[1] + random.uniform(yi1[0],yi1[1])]
            y2 = [y2[0] + depi1 + random.uniform(yi1[0],yi1[1]),
                  y2[1] + depi2 + random.uniform(yi1[0],yi1[1])]
            y3 = [y3[0] + random.uniform(yi3[0],yi3[1]),
                  y3[1] + random.uniform(yi3[0],yi3[1])]
        if i == 0:
            x = [0,random.uniform(5,20),random.uniform(30,50),random.uniform(60,75),
                 random.uniform(110,123),128]
        else:
            x = [x[0],x[1]+random.uniform(-2,2),x[2]+random.uniform(-5,5),x[3]+random.uniform(-5,5),
                 x[4]+random.uniform(-2,2),128]
        y = [y1[0],y1[1],y2[0],y2[1],y3[0],y3[1]]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi) - 20
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,3)
        if i == line_num-1:
            yi1 = (127-yi).copy()
        lines.append(np.vstack([xi,127-yi]))

    yy1 = [6,10]
    for i in range(20):
        if i==0:
            y = [yi1[0]-random.uniform(yy1[0],yy1[1]), ((yi1[0]+yi1[-1])/2)-random.uniform(yy1[0],yy1[1]) ,yi1[-1]-random.uniform(yy1[0],yy1[1])]
        else:
            y = [yi[0]-random.uniform(yy1[0],yy1[1]), ((yi[0]+yi[-1])/2)-random.uniform(yy1[0],yy1[1]) ,yi[-1]-random.uniform(yy1[0],yy1[1])]
        x = [0,random.uniform(50,80),128]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi)
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = gaussian_filter1d(yi,3)
        for ip in range(xi.shape[0]):
            if yi[ip] >= yi1[int(xi[ip])]:
                yi[ip] = np.nan
        lines.append(np.vstack([xi,yi]))

    if random.randint(0,1)==0: flip = True
    else: flip = False

    for li in range(len(lines)):
        line = lines[li]
        if flip is True:
            line[0] = np.flip(line[0],0)
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = -1000
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = -1000,-1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1   
    return section
    
# Generate randomly homogeneous with drape peak data
def generate_hummocky3():
    """
    Generate randomly homogeneous with drape peak data
    """  
    section = np.zeros((128, 128))
    lines = []
    line_num = 20
    locat = 5
    depmax= random.uniform(20,50)
    dep_int = depmax/int(line_num/2)
    yi1 = [4,8]
    yi2 = [6,10]

    for i in range(line_num):    
        if i <= int(line_num/2):
            if i==0:
                depi = dep_int
                y1 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
                y2 = [locat+depi+random.uniform(-5,5),locat+depi+random.uniform(-5,5)]
                y3 = [locat+random.uniform(-5,5),locat+random.uniform(-5,5)]
            else:
                depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
                depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
                y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),
                      y1[1] + random.uniform(yi1[0],yi1[1])]
                y2 = [y2[0] + depi1 + random.uniform(yi1[0],yi1[1]),
                      y2[1] + depi2 + random.uniform(yi1[0],yi1[1])]
                y3 = [y3[0] + random.uniform(yi1[0],yi1[1]),
                      y3[1] + random.uniform(yi1[0],yi1[1])]
        else:
            depi1 = random.uniform(dep_int*0.8,dep_int*1.2)
            depi2 = random.uniform(dep_int*0.8,dep_int*1.2)
            y1 = [y1[0] + random.uniform(yi2[0],yi2[1]),
                  y1[1] + random.uniform(yi2[0],yi2[1])]
            y2 = [y2[0] - depi1 + random.uniform(yi2[0],yi2[1]),
                  y2[1] - depi2 + random.uniform(yi2[0],yi2[1])]
            y3 = [y3[0] + random.uniform(yi2[0],yi2[1]),
                  y3[1] + random.uniform(yi2[0],yi2[1])]
        if i == 0:
            x = [0,random.uniform(5,20),random.uniform(40,60),random.uniform(70,90),
                 random.uniform(110,123),128]
        else:
            x = [x[0],x[1]+random.uniform(-2,2),x[2]+random.uniform(-5,5),x[3]+random.uniform(-5,5),
                 x[4]+random.uniform(-2,2),128]
        y = [y1[0],y1[1],y2[0],y2[1],y3[0],y3[1]]
        func = interpolate.interp1d(x,y,kind="linear")
        xi = np.arange(0,128,1)
        noisei = random.uniform(-5,5)
        yi = func(xi)
        yi = gaussian_filter1d(yi,5)+ np.random.uniform(-noisei, noisei, size=xi.shape)
        yi = 127-gaussian_filter1d(yi,3)
        lines.append(np.vstack([xi,yi]))
        if i== int(line_num/3):
            yyy = yi.copy()

    hx = random.randint(60,70)
    hom_xrange = [hx - random.randint(20,40),hx + random.randint(20,40)]
    hom_lrange = [int(len(lines)/3)-random.randint(3,5), int(len(lines)/3)+random.randint(3,5)]

    for li in range(len(lines)):
        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            line[1][mxii[mi]] = np.nan
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < 128 :
                    line[1][mxii[mi]-wdi],line[1][mxii[mi]+wdi] = np.nan,np.nan

        # add homogenous with drape
        if li >= hom_lrange[0] and li<hom_lrange[1]:
            for ii in range(0,line[1].shape[0],5):
                if ii >= hom_xrange[0]  and ii < hom_xrange[1]:
                    if random.randint(0,2)<2:
                        line[1][ii] = np.nan
                        line[1][ii-1] = np.nan
                        line[1][ii+1] = np.nan
                        line[1][ii-2] = np.nan
                        line[1][ii+2] = np.nan
        hom_xrange = [hom_xrange[0]+random.randint(-4,4),hom_xrange[1]+random.randint(-4,4)]

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),int(line[1][ii])]=1   
    return section   
    
# Generate randomly slump peak data
def generate_hummocky4():
    """
    Generate randomly slump peak data
    """  
    section = np.zeros((128, 128))
    lines = []

    # build a fill 
    x0 = [0,random.uniform(10,30),random.uniform(50,65),random.uniform(75,90),
                 random.uniform(100,110),128]
    y0 = [0,random.uniform(0,5),random.uniform(10,15),random.uniform(15,40),
          -random.uniform(0,5),-random.uniform(20,50)]
    func = interpolate.interp1d(x0,y0,kind="linear")
    xi = np.arange(0,128,1)
    yi = func(xi)
    noisei = random.uniform(-5,5)
    yi = gaussian_filter1d(yi,5) + np.random.uniform(-noisei, noisei, size=xi.shape)
    yi = gaussian_filter1d(yi,3)
    yi =  yi + (128-yi.max()+yi.min())/2.5 - y0[-1]
    lines.append(np.vstack([xi,yi]))

    # add upper parallel lines
    for i in range(10):
        x = [0,32,64,96,128]
        loc = yi.min()-8*i-5
        shift = random.uniform(2,4)
        y = [loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift),
             loc+random.uniform(-shift,shift),loc+random.uniform(-shift,shift)]
        func = interpolate.interp1d(x,y,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        if i==0:
            yyi = yii.copy()
        lines.append(np.vstack([xii,yii]))
        if i ==0:
            yyy = yii.copy()

    yi1 = [8,12]   
    yi2 = [10,12]
    yi3 = [10,12]
    dep_int=random.uniform(2,6)
    # add bottom fill layer
    for i in range(20):
        x1 = np.array(x0)
        depi1 = random.uniform(dep_int*0.6,dep_int*0.8)
        depi2 = random.uniform(dep_int*0.6,dep_int*0.8)
        if i ==0:
            y1 = np.array(y0) + (128-yi.max()+yi.min())/2.5 - y0[-1]
        y1 = [y1[0] + random.uniform(yi1[0],yi1[1]),y1[1] + random.uniform(yi1[0],yi1[1]),
              y1[2] - depi1 + random.uniform(yi2[0],yi2[1]),y1[3] - depi2 + random.uniform(yi2[0],yi2[1]),
              y1[4] + random.uniform(yi3[0],yi2[1]),y1[5]+ random.uniform(yi3[0],yi2[1])]
        func = interpolate.interp1d(x1,y1,kind="linear")
        xii = np.arange(0,128,1)
        yii = func(xii)
        noisei = random.uniform(-5,5)
        yii = gaussian_filter1d(yii,5) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))

    ## add short lines
    ys1 = [yi[0], yyy[0]]
    sline_num = int((ys1[0]-int(ys1[1]))/random.randint(8,10))
    y2num = np.linspace(ys1[1]+5,ys1[0]-5,sline_num)
    y1num = np.linspace(yyy[0]+5,yi[0]-5,sline_num)
    xxmax = 0
    for i in range(sline_num):
        xx = [0,x0[1] + random.randint(10-i,20-i*2)]
        yy = [y1num[i],y2num[i]]
        sfunc = interpolate.interp1d(xx,yy,kind="linear")
        xii = np.arange(xx[0],xx[1],1)
        noisei = random.uniform(-5,5)
        yii = sfunc(xii) + np.random.uniform(-noisei, noisei, size=xii.shape)
        yii = gaussian_filter1d(yii,3)
        lines.append(np.vstack([xii,yii]))
        if xxmax < xii[-1]:
            xxmax = xii[-1]

    for li in range(len(lines)):
        line = lines[li]
        # add lines mask
        mxii = random.sample(range(0,section.shape[0]),random.randint(0,6))
        mxi = mxii.copy()
        for mi in range(len(mxii)):
            for wdi in range(random.randint(0,3)):
                if (mxii[mi]+ wdi) < line[0][-1] and (mxii[mi]- wdi) > line[0][0] :
                    line[1][int(mxii[mi]-wdi-line[0][0])] = -1000
                    line[1][int(mxii[mi]-line[0][0])] = -1000
                    line[1][int(mxii[mi]+wdi-line[0][0])] = -1000

        for ii in range(len(line[0])):
            if line[0][ii]>=0 and line[0][ii]<128 and line[1][ii]>=0 and line[1][ii]<128:
                section[int(line[0][ii]),127-int(line[1][ii])]=1    

    # fill the chaotic 
    def find_peak(data):
        data = data.T
        col_peaks = np.zeros_like(data).astype(int)
        for i in range(data.shape[1]):
            col_data = data[:,i]
            for j in range(1,data.shape[0]-1):
                col,col1,col2 = col_data[j],col_data[j-1],col_data[j+1]
                if col > col1 and col > col2:
                    col_peaks[j, i] = 1
        return col_peaks.T

    mean = random.randint(-10,10)
    std = random.randint(10,20)
    x = np.random.normal(mean,std,size=(128,128))
    x = ndimage.gaussian_filter(x,2)
    chaotic = find_peak(x)

    for i in range(int(xxmax),section.shape[0]):
        yi1,yi2 = 128-int(yyi[i]),128-int(yi[i])
        section[i,yi2:yi1] = chaotic[i,yi2:yi1]
    return section  

####################################################################################
####################################### CHAOTIC ####################################
####################################################################################
def find_peak(data):
    data = data.T
    col_peaks = np.zeros_like(data).astype(int)
    for i in range(data.shape[1]):
        col_data = data[:,i]
        for j in range(1,data.shape[0]-1):
            col = col_data[j]
            col1 = col_data[j-1]
            col2 = col_data[j+1]
            if col > col1 and col > col2:
                col_peaks[j, i] = 1
    return col_peaks.T    
    
# Generate randomly chaotic peak data
def generate_chaotic():
    """
    Generate randomly chaotic peak data
    """   
    mean = random.randint(-10,10)
    std = random.randint(10,20)
    x = np.random.normal(mean,std,size=(128,128))
    x = ndimage.gaussian_filter(x,2)
    section = find_peak(x)
    return section
    
    