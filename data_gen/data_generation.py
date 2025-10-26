import numpy as np
from .noise import noise_torch
import torch
# from torch.utils.data import DataLoader,data_utils
import torch.utils.data as data
from .fr import freq2fr
import matplotlib.pyplot as plt
"""
数据生成代码
author:wangziwen
data: 2024.3.20
"""

def amplitude_generation(dim, amplitude = 'uniform', floor_amplitude=0.2):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (1 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return 5*np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)
    elif amplitude == 'suiji':
        return   (np.random.rand(*dim)-0.5)
    
def range_generator(f, nf, min_sep, dist_distribution = 'normal'):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)

def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new



def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = (np.random.uniform() - 0.5)*0.9
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def gen_signal(num_samples, signal_dim, num_freq, min_sep,snr,snr_max,batch_size,upsample,gs_size = 0.003, distance='normal', amplitude='uniform',
               floor_amplitude=0.2, variable_num_freq=True):
    #生成数据的空矩阵
    s = np.zeros((num_samples, 2, signal_dim))
    s_gap = np.zeros((num_samples, 2, signal_dim))
    s_fft = np.zeros((num_samples, 2, signal_dim))
    fgrid = np.linspace(14e9,18e9,signal_dim)[:, None]
    #生成估计频率的矩阵
    R = np.ones((num_samples, num_freq)) * np.inf
    R_r = np.ones((num_samples, num_freq)) * np.inf
    #生成幅度
    afa = amplitude_generation((num_samples, num_freq),amplitude, floor_amplitude)
    #生成随机幅度
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    #频率最小间隔
    d_sep = min_sep / signal_dim
    #生成随机频率个数
    if variable_num_freq:
        nfreq =  np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        if n%100==0:
            print(n)
        #生成随机距离目标
        range_generator(R[n], nfreq[n], d_sep, distance)
        R_r[n] = (R[n]+0.5)*3e8/2/(fgrid[2]-fgrid[1])
        for i in range(nfreq[n]):
            sin = afa[n, i] * (np.exp(1j*theta[n,i] + 4j* np.pi/3e8 * R_r[n,i]* fgrid)).T
            s[n,0] = s[n,0] + sin.real
            s[n,1] = s[n,1] + sin.imag
        s[n] = s[n] / np.max(abs(s[n,0]+1j*s[n,1]))
    
    s_fft_sin = np.fft.fft(s[:,0,:]+1j*s[:,1,:])
    s_fft_sin = (s_fft_sin-np.min(abs(s_fft_sin))) / (np.max(abs(s_fft_sin))-np.min(abs(s_fft_sin)))
    s_fft[:,0,:] = s_fft_sin.real
    s_fft[:,1,:] = s_fft_sin.imag
    xgrid = np.linspace(-0.5, 0.5, signal_dim*upsample, endpoint=False)
    frequency_representation,fr_ground = freq2fr(R, xgrid, 'gaussian', gs_size, afa)
    frequency_representation = torch.from_numpy(frequency_representation).float()
    fr_ground = torch.from_numpy(fr_ground).float()
    s_fft = s_fft.astype('float32')
    nfreq = nfreq.astype('float32')
    s = noise_torch(s, snr,snr_max).astype('float32')
    for i in range(num_samples):
        s[i,:,:] = s[i,:,:] /np.max(abs(s[i,0,:]+1j*s[i,1,:]))
    
    clean_signals = torch.from_numpy(s).float()
    signal_fft = torch.from_numpy(s_fft).float()
    n_target = torch.from_numpy(nfreq).float()
    train_dataset = data.TensorDataset(clean_signals,frequency_representation,signal_fft,n_target,fr_ground)
    training_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return training_data_loader