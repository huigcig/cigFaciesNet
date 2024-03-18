import os
import numpy as np
from scipy import fftpack
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import gaussian_filter1d

# 均值方差归一化
def mean_std_norm(data):
    data = (data-np.mean(data))/np.std(data)
    return data

# 振幅均衡函数
def amplitude_equalization(gx):
    vmin_s, vmax_s = -3, 3 #直方图均衡化后数据体的最大最小值(自己定)
    # 线性拉伸到0-255范围内
    vmin, vmax = np.min(gx), np.max(gx)
    gx = (gx - vmin) * 255.0 / (vmax - vmin)
    # 计算直方图
    hist, bins = np.histogram(gx, bins=256, range=(0, 255))
    # 计算直方图均衡化
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    gx_eq = np.interp(gx.flatten(), bins[:-1], cdf).reshape(gx.shape)
    # 将灰度图像值域映射回数据值域范围
    gx_eq = gx_eq / 255.0 * (vmax_s - vmin_s) + vmin_s
    gx_eq.astype(np.float32)
    return gx_eq

# 带通滤波函数
def bandpass_filter(seis, freq_low, freq_high, dt=0.002, order=5, auto_pad = [16,16]):
    fs = 1/(2*dt)
    nyq = 0.5 * fs
    lims = [freq_low / nyq, freq_high / nyq]
    seis_pad = np.zeros((seis.shape[0], seis.shape[1] + auto_pad[0] + auto_pad[1]))
    seis_pad[:,auto_pad[0]:seis.shape[1]+auto_pad[0]]=seis.copy()
    
    sos = butter(order, lims, btype='band', output='sos')
    filter_seis = sosfiltfilt(sos, seis_pad)
    filter_seis = filter_seis[:,auto_pad[0]:filter_seis.shape[1]-auto_pad[1]]
    return filter_seis

# 沿着构造方向引导的平滑
def sos(fx,nither=1,kappa=50,gamma=0.1,step=(1.,1.),option=2):
    # 沿着构造方向引导的平滑
    # initialize output array
    gx = fx.copy()
    # initialize some internal variables
    deltaS,deltaE = np.zeros_like(gx),np.zeros_like(gx)
    NS,EW = np.zeros_like(gx),np.zeros_like(gx)
    gS,gE = np.ones_like(gx),np.ones_like(gx)
    for ii in range(nither):
        # calculate the diffs
        deltaS[:-1,:]=np.diff(gx,axis=0)
        deltaE[:,:-1]=np.diff(gx,axis=-1)
        # conduction gradients (only need to compute one per dim!)
        if option==1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option==2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
        # updata matrices
        E = gE*deltaE
        S = gS*deltaS
        # subtract a copy that has been shifted 'North/lest' by one pixel.
        NS[:]=S
        EW[:]=E
        NS[1:,:]-= S[:-1,:]
        EW[:,1:]-= E[:,:-1]
        # update the image
        gx += gamma*(NS+EW)
    return gx

# 重采样函数
def dataupsample_2d(x,axis,num_up,method = 'nearest'):
  #x = np.transpose(x)
    if axis == 0:
        x1 = np.linspace(0,x.shape[0]-1,x.shape[0])
        x_new = np.linspace(0,x.shape[0]-1,num_up)
        gxu = np.zeros((num_up,x.shape[1]),dtype=np.single)
        for i in range(x.shape[1]):
            f = interpolate.interp1d(x1,x[:,i],kind = method)
            gxu[:,i] = f(x_new)
    elif axis == 1:
        x1 = np.linspace(0,x.shape[1]-1,x.shape[1])
        x_new = np.linspace(0,x.shape[1]-1,num_up)
        gxu = np.zeros((x.shape[0],num_up),dtype=np.single)
        for i in range(x.shape[0]):
            f = interpolate.interp1d(x1,x[i,:],kind = method)
            gxu[i,:] = f(x_new)
    return gxu

# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed  

# 单道处理之后的横向平滑函数
def equalization_smoothing(fx, alpha=0.8):
    gx = np.zeros(fx.shape)
    for j in range(fx.shape[1]):
#         gx[:,j] = gaussian_filter1d(fx[:,j],sigma=sigma)
        gx[:,j] = exponential_filter(fx[:,j],alpha=alpha)
    return gx

# 频谱均衡函数 (在频段内给一个factor乘积)
def spectrum_equalization2(fx, freq_low, freq_high, dt, auto_pad = [16,16], sigma=2):
    gx = np.zeros((fx.shape[0], fx.shape[1] + auto_pad[0]+ auto_pad[1]))
    gx[:,auto_pad[0]:fx.shape[1]+auto_pad[0]] = fx
    
    # build equalization factor
    ms0 = np.zeros(gx.shape)
    for i in range(gx.shape[0]): 
        ms0[i,:] = np.abs(np.fft.fft(gx[i,:]))
    freq0 = np.fft.fftfreq(gx.shape[1],2*dt)
    ms0 = np.mean(ms0,axis=0)
    f1,f2 = freq_low, freq_high
    r1,r2 = np.abs(freq0-f1).argmin(),np.abs(freq0-f2).argmin()
    smooth_factor = np.ones(ms0.shape)
    ms_max = np.max(ms0[r1:r2+1])
    smooth_factor[r1:r2+1] = ms_max/ms0[r1:r2+1]
    smooth_factor[-r2:-r1+1] = ms_max/ms0[-r2:-r1+1]
    smooth_factor = gaussian_filter1d(smooth_factor,sigma=2)
    
#     plt.figure(figsize=(8,4))
#     plt.subplot(1,2,1)
#     plt.plot(freq0[0:smooth_factor.shape[0]//2],ms0[0:smooth_factor.shape[0]//2],linewidth=3)
#     plt.plot(freq0[0:smooth_factor.shape[0]//2],smooth_factor[0:smooth_factor.shape[0]//2],linewidth=3)
#     plt.grid(alpha=0.6),plt.xlabel('Frequency (Hz)'),plt.ylabel('Amplitude',labelpad=-10),plt.xlim(0,128)
#     plt.tight_layout()
    
#     plt.subplot(1,2,2)
#     plt.plot(freq0[0:smooth_factor.shape[0]//2],ms0[0:smooth_factor.shape[0]//2],linewidth=3)
#     plt.plot(freq0[0:smooth_factor.shape[0]//2],smooth_factor[0:smooth_factor.shape[0]//2],linewidth=3)
#     plt.grid(alpha=0.6),plt.xlabel('Frequency (Hz)'),plt.ylabel('Amplitude',labelpad=-10),plt.xlim(0,128)
#     plt.tight_layout()
        
    # equalizate spectrum and rebuild gx
    new_gx = np.zeros(gx.shape)
    for i in range(gx.shape[0]):
        fft_gx = np.fft.fft(gx[i,:]) 
        magnitude_spectrum = np.abs(fft_gx) * smooth_factor
        phase = np.angle(fft_gx)
        new_fft_gx = magnitude_spectrum * np.exp(1j * phase)
        ifft_gx = np.fft.ifft(new_fft_gx)
        new_gx[i,:] = np.real(ifft_gx)
    new_gx = new_gx[:,auto_pad[0]:fx.shape[1]+auto_pad[0]]
    return new_gx   


# 频谱均衡函数 (在频段内幅最大值)
def spectrum_equalization(fx, freq_low, freq_high, dt,auto_pad = [8,8]):
    gx = np.zeros((fx.shape[0], fx.shape[1] + auto_pad[0]+ auto_pad[1]))
    gx[:,auto_pad[0]:fx.shape[1]+auto_pad[0]] = fx
    # calculate magnitude spectrum (vis fft)
    magnitude_spectrum = np.zeros(gx.shape)
    phase = np.zeros(gx.shape)
    for i in range(gx.shape[0]): 
        fft_gx = np.fft.fft(gx[i,:])
        magnitude_spectrum[i, :] = np.abs(fft_gx)
        phase[i, :] = np.angle(fft_gx)
        
    # calculate spectral distribution
    freq = np.fft.fftfreq(gx.shape[1],2*dt)
    f1,f2 = freq_low, freq_high
    r1,r2 = np.abs(freq-f1).argmin(),np.abs(freq-f2).argmin()
#     value = np.max(magnitude_spectrum)
    value = (np.max(magnitude_spectrum[:,r1:r2]) + np.max(magnitude_spectrum[i, -r2:-r1]))/2
    
    # equalizate spectrum and rebuild gx
    new_gx = np.zeros(gx.shape)
    for i in range(gx.shape[0]):
        magnitude_spectrum[i, r1:r2] = value
        magnitude_spectrum[i, -r2:-r1] = value
        new_fft_gx = magnitude_spectrum[i,:] * np.exp(1j * phase[i,:])
        ifft_gx = np.fft.ifft(new_fft_gx)
        new_gx[i,:] = np.real(ifft_gx)
    new_gx = new_gx[:,auto_pad[0]:new_gx.shape[1]-auto_pad[1]]
    return new_gx   

# 频谱-振幅函数
def calculate_spectrum(gx, dt, mean = False):
    # calculate magnitude spectrum (vis fft)
    magnitude_spectrum = np.zeros(gx.shape)
    for i in range(gx.shape[0]): 
        magnitude_spectrum[i,:] = np.abs(np.fft.fft(gx[i,:]))
    freq = np.fft.fftfreq(gx.shape[1],2*dt)
    if mean is True:
        magnitude_spectrum = np.mean(magnitude_spectrum,axis=0)
    else:
        magnitude_spectrum = magnitude_spectrum.T
    return freq, magnitude_spectrum

# 一维递归指数滤波函数
def exponential_filter(data, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    filtered_data = []
    last_filtered_value = data[0]  # 初始值为第一个数据点

    for point in data:
        filtered_value = alpha * point + (1 - alpha) * last_filtered_value
        filtered_data.append(filtered_value)
        last_filtered_value = filtered_value

    return filtered_data

# 提取波峰(or 波谷)函数
def find_2D_peaks(data, threshold=None, find_through = False):
    data = data.T
    if threshold is None:
        threshold = np.max(data) / 2  # 默认阈值为最大值的一半
        threshold = -1000
    # 找每一列的波峰点
    col_peaks = np.zeros_like(data).astype(int)
    for i in range(data.shape[1]):
        col_data = exponential_filter(data[:,i], alpha=1) # 一维递归指数滤波
        for j in range(1,data.shape[0]-1):
            col = col_data[j]
            col1 = col_data[j-1]
            col2 = col_data[j+1]
            if col >= threshold:
                if col > col1 and col > col2:
                    col_peaks[j, i] = 1
                if find_through is True:
                    if col < col1 and col < col2:
                        col_peaks[j, i] = 1
    return col_peaks.T