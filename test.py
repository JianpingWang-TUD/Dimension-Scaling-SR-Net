from __future__ import print_function
import os
import time
import socket
import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from net_model.DSSR import DSSR_net
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from data_gen.data_generation import gen_signal
import matplotlib.pyplot as plt
from data_gen.data_generation import amplitude_generation
from data_gen.noise import noise_torch
import scipy
import random
from scipy.io import savemat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
仿真验证代码
相变实验

Simulation verification code
Phase transition experiment
"""

# ---------------------------
# Training / experiment settings
# ---------------------------
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')

parser.add_argument('--Ispretrained', type=bool, default=True, help='If load checkpoint model')
parser.add_argument('--pretrained_sr', default='net_epoch_upsample_8_layers_3_max_n_R_5_gs_size_0.003125_75_final_gass.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', default='./checkpoint/model', help='Location to load checkpoint models')
parser.add_argument('--save_folder', default='./checkpoint/', help='Location to save checkpoint models')
parser.add_argument('--pretrained_nomodel', default='./checkpoint/nomodel/net_epoch_upsample_8_layers_1_max_n_R_5_gs_size_0.003_50_final.pth', help='sr pretrained base model')
parser.add_argument('--model_type', type=str, default='Deam', help='the name of model')

# Global settings
parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='the dataset dir')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def guiyihua(data):
    """
    归一化函数 / Normalization function
    """
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    return data


def findpeak(prediction,distance,d_grid,p_gt):
    """
    Find peaks in the spectrum and check whether estimated distance matches ground-truth
    prediction: model output
    distance: ground-truth distance
    d_grid: grid resolution
    p_gt: ground-truth peak index
    """
    k = 0
    for i in range(len(prediction)):
        find_peaks_out,_ = scipy.signal.find_peaks(prediction[i],height=0.1)
        if len(find_peaks_out)==2:
            estimation_R = (find_peaks_out[1]-find_peaks_out[0])*d_grid
            if (estimation_R>= distance- dr/4) & (estimation_R <= distance+ dr/4) & (abs(find_peaks_out[0] - p_gt[i])<=int(dr/4/d_grid)):
                k += 1
    return k/len(prediction)


def signal_generation(R,fgrid,snr,num_samples,R_m,weak,signal_dim=64,num_freq=2):
    """
    Generate synthetic signals for testing
    R: true target ranges
    fgrid: frequency grid
    snr: signal-to-noise ratio
    num_samples: number of generated signals
    R_m: maximum range
    weak: amplitude of weak target
    """
    signal_dim = 64
    s = np.zeros((num_samples,2, signal_dim))
    s_no = np.zeros((num_samples,2, signal_dim))

    # Generate random phases
    theta = np.random.rand(num_samples,signal_dim) * 2 * np.pi
    num_freq = np.size(R,1)
    afa = amplitude_generation((num_samples, num_freq),'uniform', 1)
    
    xgrid = np.arange(signal_dim)[:, None]

    for n in range(num_samples):
        for i in range(num_freq):
            # Assign amplitudes: one strong target, one weak target
            afa[n,0] = 1
            afa[n,1] = weak

            # Generate sinusoidal signals with phase and frequency modulation
            sin = afa[n,i] * (np.exp(1j*theta[n,i] + 4j* np.pi/3e8 * R[n,i]* fgrid)).T
            sin_deep = afa[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * (R[n,i]-R_m/2)/R_m * xgrid.T ) 

            # Real & imaginary components
            s[n,0] = s[n,0] + sin.real
            s[n,1] = s[n,1] + sin.imag
            s_no[n,0] = s_no[n,0] + sin_deep.real
            s_no[n,1] = s_no[n,1] + sin_deep.imag

        # Normalize
        s[n] = s[n] / np.max(abs(s[n,0]+1j*s[n,1]))
        s_no[n] = s_no[n] /  np.sqrt(np.mean(np.power(s_no[n], 2)))

    # Add noise
    s = noise_torch(s, snr,snr).astype('float32')
    s_no = noise_torch(s_no, snr,snr, kind='gaussian').astype('float32')

    for i in range(num_samples):
        s[i,:,:] = s[i,:,:] /np.max(abs(s[i,0,:]+1j*s[i,1,:]))

    s = torch.from_numpy(s).float()
    s_no = torch.from_numpy(s_no).float()
    return s,s_no


def batch_PSNR(img, imclean, data_range):
    """
    Compute average PSNR for a batch
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :], Img[i, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def print_network(net):
    """
    Print network architecture and total number of parameters
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':

    print('===> Loading datasets')
    signal_dim = 64
    layer_num = 3
    upsample = 8
    double_upsample = upsample*2
    dim = 64

    # create database
    n_layers = 16
    kernel_size = 13

    print('===> Building model ', opt.model_type)
    model = DSSR_net(dim,layer_num,signal_dim,upsample)
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.MSELoss()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    # Load pretrained model if specified
    if opt.Ispretrained:
        model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print(model_name + ' model is loaded.')
        
    # ---------------------------
    # Parameter setting
    # ---------------------------
    c = 3e8
    fgrid = np.linspace(14e9,18e9,signal_dim)[:, None]
    dr = c/2/(fgrid[-1]-fgrid[0])    # range resolution
    R_m = dr*64                      # maximum unambiguous range
    d_grid = R_m/1024                # distance grid
    num_samples = 100
    R_p = np.linspace(1/1024,1,1024)*R_m

    # Sweep parameter tables
    distance_table = np.linspace(0.01875,3*0.01875,21)
    snr_table = np.linspace(0,30,31)
    weak_table = np.linspace(0.05,0.5,10)

    # For quick test, select specific values
    distance_table = np.array([0.0375])
    snr_table = np.array([15])
    weak_table = np.array([1])

    acc_table = np.zeros((len(distance_table),len(snr_table),8))
    acc_table = np.zeros((len(weak_table),len(snr_table),8))

    dis_k = 0
    for weak in weak_table:
        snr_k = 0
        for snr in snr_table:
            distance = 0.03
            R = np.ones((num_samples, 2))
            p_gt = np.ones((num_samples))

            # Generate ground-truth target positions
            for i in range(num_samples):
                R[i,:] = (np.array([R_m/2,R_m/2+distance]) + random.randint(-0, 0) * d_grid)[:,0]
                p_gt[i] = np.int32((R[i,0] + dr/2)/R_m*1024)

            R_mean = np.mean(R[0,:])
            mean_point =  np.int32(R_mean/R_m*1024)
            point_start = (mean_point-50)[0]
            point_end = (mean_point+50)[0]

            # Generate test signals
            testing_data,testing_data_no = signal_generation(R,fgrid,snr,num_samples,R_m,weak)
            testing_data = testing_data.cuda()
            testing_data_c = testing_data[:,0,:]+1j*testing_data[:,1,:]

            # FFT-based baseline
            time_fft = time.time()
            s_fft_sin = torch.fft.fft(testing_data[:,0,:]+1j*testing_data[:,1,:],1024,dim=-1).cpu().detach().numpy()
            s_fft_sin = s_fft_sin/np.max(abs(s_fft_sin),1)[0]
            time_fft = time.time() - time_fft
            
            # Proposed DSSR method
            model.eval()
            time_proposed = time.time()
            prediction = guiyihua(model(testing_data).cpu().detach().numpy())
            time_proposed = time.time() - time_proposed

            # Evaluate success rate
            success_fft = findpeak(abs(s_fft_sin),distance,d_grid,p_gt)
            success_proposed = findpeak(abs(prediction),distance,d_grid,p_gt)
            
            snr_k+=_
